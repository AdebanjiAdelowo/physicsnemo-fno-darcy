# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The training loop of ``train_run`` (model, MSE loss, Adam, LambdaLR decay per
# block of pseudo-epochs, StaticCapture forward passes, checkpoints) is adapted
# from examples/cfd/darcy_fno/train_fno_darcy.py of NVIDIA PhysicsNeMo, commit
# b45a5c810c741e6b41f8515be24c51121f8fc21f. Modified by Adebanji Adelowo (2026):
# seeding, an explicit device for Darcy2D, an optional finite cached training set,
# fixed validation and test sets, a relative L2 metric in physical units,
# synchronised timing, memory statistics, CSV/JSON records, and the loop over
# Fourier modes and seeds. The LaunchLogger/MLFlow logging, the validation figure
# and checkpoint resumption of the original are not used.

import csv
import statistics
import time
from math import ceil
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from physicsnemo.utils import (
    StaticCaptureEvaluateNoGrad,
    StaticCaptureTraining,
    load_checkpoint,
    save_checkpoint,
)
from physicsnemo.utils.logging import PythonLogger
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler

from . import data as D
from .metrics import relative_l2, summarise
from .model import build_fno, count_parameters, max_fno_modes
from .provenance import collect_metadata, write_json

HISTORY_FIELDS = ["pseudo_epoch", "learning_rate", "train_loss", "validation_loss", "validation_rel_l2", "elapsed_seconds"]


def resolve_device(name: str) -> torch.device:
    """``auto`` picks CUDA, then Apple MPS, then CPU. An explicit name must be available."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda was requested but CUDA is not available on this machine.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device=mps was requested but MPS is not available on this machine.")
    return device


def synchronise(device: torch.device) -> None:
    """Wait for queued accelerator work so that a wall-clock reading is meaningful."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def run_name(fno_modes: int, seed: int) -> str:
    return f"modes{fno_modes:02d}_seed{seed}"


def validate_study(cfg: DictConfig) -> None:
    """Reject settings the architecture or the data generator cannot represent."""
    limit = max_fno_modes(cfg.training.resolution, cfg.arch.fno.padding)
    for modes in cfg.study.fno_modes:
        if not 1 <= modes <= limit:
            raise ValueError(
                f"fno_modes={modes} is outside [1, {limit}] for resolution "
                f"{cfg.training.resolution} with padding {cfg.arch.fno.padding}."
            )
    if cfg.data.source not in ("stream", "fixed"):
        raise ValueError(f"data.source must be 'stream' or 'fixed', got {cfg.data.source!r}.")


def load_datasets(cfg: DictConfig, device: torch.device, root: Path) -> dict:
    """Validation and test sets (always finite), and the training set if ``data.source=fixed``."""
    sizes = {"validation": cfg.validation.sample_size, "test": cfg.data.test_samples}
    if cfg.data.source == "fixed":
        sizes["train"] = cfg.training.pseudo_epoch_sample_size
    return {split: D.load_or_generate(cfg, split, n, device, root) for split, n in sizes.items()}


@torch.no_grad()
def predict(forward_eval, k: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    """Model output for a CPU tensor of normalised inputs, returned on CPU."""
    return torch.cat([forward_eval(k[i : i + batch_size].to(device)).cpu() for i in range(0, len(k), batch_size)])


def error_metrics(pred: torch.Tensor, target: torch.Tensor, normaliser: dict) -> dict:
    """MSE in normalised units (the training loss) and relative L2 in physical units."""
    rel = relative_l2(D.denormalise(pred, normaliser["darcy"]), D.denormalise(target, normaliser["darcy"]))
    return {"mse_normalised": torch.mean((pred - target) ** 2).item(), "rel_l2": rel}


def train_run(cfg: DictConfig, fno_modes: int, seed: int, run_dir: Path, device: torch.device, sets: dict) -> dict:
    """Train one FNO and write ``history.csv``, ``train_metrics.json`` and a checkpoint."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = PythonLogger(name="darcy_fno")
    normaliser = D.make_normaliser(cfg)

    # The model is initialised on CPU from the seed and then moved, so the initial
    # weights do not depend on the device.
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_fno(cfg, fno_modes).to(device)
    loss_fun = MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=cfg.scheduler.initial_lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cfg.scheduler.decay_rate**step)

    batch_size = cfg.training.batch_size
    steps_per_pseudo_epoch = ceil(cfg.training.pseudo_epoch_sample_size / batch_size)
    if cfg.data.source == "fixed":
        train_k, train_u = D.to_normalised_tensors(sets["train"], normaliser)
        shuffle = torch.Generator().manual_seed(seed)
    else:
        pipe = D.make_datapipe(cfg, batch_size, device, normaliser)
    val_k, val_u = D.to_normalised_tensors(sets["validation"], normaliser)

    @StaticCaptureTraining(model=model, optim=optimizer, logger=log, use_amp=False, use_graphs=False)
    def forward_train(invars, target):
        pred = model(invars)
        loss = loss_fun(pred, target)
        return loss

    @StaticCaptureEvaluateNoGrad(model=model, logger=log, use_amp=False, use_graphs=False)
    def forward_eval(invars):
        return model(invars)

    history = []
    data_seconds = optim_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    synchronise(device)
    start = time.perf_counter()

    for pseudo_epoch in range(1, cfg.training.max_pseudo_epochs + 1):
        if cfg.data.source == "fixed":
            batches = D.fixed_batches(train_k, train_u, batch_size, shuffle, device)
        else:
            batches = D.stream_batches(pipe, steps_per_pseudo_epoch, device)
        losses = []
        tick = time.perf_counter()
        for invars, target in batches:
            synchronise(device)
            tock = time.perf_counter()
            data_seconds += tock - tick
            losses.append(forward_train(invars, target).detach())
            synchronise(device)
            tick = time.perf_counter()
            optim_seconds += tick - tock
        row = {
            "pseudo_epoch": pseudo_epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": torch.stack(losses).mean().item(),
            "validation_loss": None,
            "validation_rel_l2": None,
        }

        if pseudo_epoch % cfg.training.rec_results_freq == 0:
            save_checkpoint(run_dir / "checkpoints", models=model, optimizer=optimizer, scheduler=scheduler, epoch=pseudo_epoch)

        if pseudo_epoch % cfg.validation.validation_pseudo_epochs == 0:
            val = error_metrics(predict(forward_eval, val_k, batch_size, device), val_u, normaliser)
            row["validation_loss"] = val["mse_normalised"]
            row["validation_rel_l2"] = val["rel_l2"].mean().item()

        if pseudo_epoch % cfg.scheduler.decay_pseudo_epochs == 0:
            scheduler.step()

        synchronise(device)
        row["elapsed_seconds"] = time.perf_counter() - start
        history.append(row)
        if not torch.isfinite(torch.tensor(row["train_loss"])):
            raise RuntimeError(f"non-finite training loss at pseudo-epoch {pseudo_epoch}")

    synchronise(device)
    train_seconds = time.perf_counter() - start
    save_checkpoint(
        run_dir / "checkpoints", models=model, optimizer=optimizer, scheduler=scheduler, epoch=cfg.training.max_pseudo_epochs
    )

    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)

    metrics = {
        "run": run_dir.name,
        "fno_modes": int(fno_modes),
        "seed": int(seed),
        "parameters": count_parameters(model),
        "optimiser_steps": len(history) * steps_per_pseudo_epoch,
        "training_samples_seen": len(history) * steps_per_pseudo_epoch * batch_size,
        "final_train_loss": history[-1]["train_loss"],
        "train_seconds": train_seconds,
        "optimisation_seconds": optim_seconds,
        "data_seconds": data_seconds,
        "peak_train_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
    }
    write_json(run_dir / "train_metrics.json", metrics)
    return metrics


def evaluate_run(cfg: DictConfig, fno_modes: int, run_dir: Path, device: torch.device, sets: dict) -> tuple[dict, torch.Tensor]:
    """Evaluate the checkpoint of ``run_dir`` on the fixed validation and test sets.

    Returns the metrics and the test predictions in physical units for the plotting samples.
    """
    run_dir = Path(run_dir)
    normaliser = D.make_normaliser(cfg)
    model = build_fno(cfg, fno_modes).to(device)
    epoch = load_checkpoint(run_dir / "checkpoints", models=model, device=device)
    if epoch != cfg.training.max_pseudo_epochs:
        raise RuntimeError(f"{run_dir}: checkpoint is at pseudo-epoch {epoch}, expected {cfg.training.max_pseudo_epochs}.")
    model.eval()
    log = PythonLogger(name="darcy_fno")

    @StaticCaptureEvaluateNoGrad(model=model, logger=log, use_amp=False, use_graphs=False)
    def forward_eval(invars):
        return model(invars)

    batch_size = cfg.training.batch_size
    out = {"checkpoint_pseudo_epoch": int(epoch)}
    for split in ("validation", "test"):
        k, u = D.to_normalised_tensors(sets[split], normaliser)
        pred = predict(forward_eval, k, batch_size, device)
        err = error_metrics(pred, u, normaliser)
        out[f"{split}_mse_normalised"] = err["mse_normalised"]
        out[f"{split}_rel_l2"] = summarise(err["rel_l2"])
        if split == "test":
            out["test_rel_l2_per_sample"] = err["rel_l2"].tolist()
            fields = D.denormalise(pred[: cfg.evaluation.plot_samples, 0], normaliser["darcy"])

    # Inference time of one batch, forward pass only, synchronised, after warm-up.
    batch = k[:batch_size].to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(cfg.evaluation.timing_warmup + cfg.evaluation.timing_repeats):
        synchronise(device)
        tick = time.perf_counter()
        forward_eval(batch)
        synchronise(device)
        if i >= cfg.evaluation.timing_warmup:
            times.append(time.perf_counter() - tick)
    out["inference_batch_size"] = len(batch)
    out["inference_ms_per_batch"] = 1e3 * statistics.median(times)
    out["inference_ms_per_sample"] = out["inference_ms_per_batch"] / len(batch)
    out["peak_inference_memory_mb"] = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    write_json(run_dir / "eval_metrics.json", out)
    return out, fields


def study_runs(cfg: DictConfig) -> list[tuple[int, int]]:
    return [(int(m), int(s)) for m in cfg.study.fno_modes for s in cfg.study.seeds]


def prepare_study(cfg: DictConfig, root: Path, write_metadata: bool) -> tuple[Path, torch.device, dict]:
    """Validate the config, resolve the device, load the data, optionally write ``study_metadata.json``."""
    validate_study(cfg)
    device = resolve_device(cfg.device)
    out_dir = Path(root) / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = collect_metadata(device)  # before the data is generated: the code state at start
    sets = load_datasets(cfg, device, root)
    metadata["config"] = OmegaConf.to_container(cfg, resolve=True)
    metadata["datasets"] = {split: s["info"] for split, s in sets.items()}
    if cfg.data.source == "stream":
        metadata["datasets"]["train"] = {
            "source": "stream",
            "note": "new Darcy2D samples at every step; NumPy is seeded with the run seed",
        }
    if write_metadata:
        write_json(out_dir / "study_metadata.json", metadata)
    return out_dir, device, sets
