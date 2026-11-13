"""Run the single full-resolution CUDA training feasibility probe."""

import argparse
import hashlib
import math
import resource
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import scipy
import torch
import _bootstrap
from fno_darcy import data
from fno_darcy.config import load_config
from fno_darcy.engine import train_run
from fno_darcy.provenance import collect_metadata, write_json
from verify_cuda_batch64 import memory_monitor, write_rows
from verify_generator import plt

MIB = 2**20
OVERRIDES = ["study.fno_modes=[16]", "study.seeds=[0]", "training.max_pseudo_epochs=1",
             "validation.validation_pseudo_epochs=1"]


def memory():
    return {
        "allocated_mib": torch.cuda.memory_allocated() / MIB,
        "reserved_mib": torch.cuda.memory_reserved() / MIB,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
        "free_mib": torch.cuda.mem_get_info()[0] / MIB,
    }


def projection(epoch, validation, checkpoint, epochs, runs, validation_every, checkpoint_every):
    base = epoch * epochs
    val = validation * (epochs // validation_every)
    checkpoints = checkpoint * (epochs // checkpoint_every + 1)
    hours = (base + val + checkpoints) / 3600
    return dict(optimizer_hours_per_run=base / 3600, validation_hours_per_run=val / 3600,
                checkpoint_hours_per_run=checkpoints / 3600, hours_per_run=hours,
                hours_for_study=hours * runs, planning_hours_with_25_percent_margin=1.25 * hours * runs)


class Probe:
    def __init__(self, out):
        self.out = out
        self.phase = "initialization"
        self.rows = []
        self.train_peaks = []

    def attach(self, model, optimizer):
        self.model = model
        optimizer.register_step_pre_hook(self.optimizer_start)
        optimizer.register_step_post_hook(self.optimizer_end)

    def mark(self, phase):
        self.phase = phase
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()

    def snapshot(self, phase):
        torch.cuda.synchronize()
        stats = memory()
        self.row[f"{phase}_seconds"] = time.perf_counter() - self.started
        for name, value in stats.items():
            self.row[f"{phase}_{name}"] = value
        self.train_peaks.append(stats)

    def begin(self, transfer_seconds):
        self.row = dict(update=len(self.rows) + 1, data_seconds=transfer_seconds)
        self.update_start = time.perf_counter()
        self.phase = "zero_grad"

    def backward_start(self, grad):
        self.snapshot("forward")
        self.mark("backward")

    def optimizer_start(self, optimizer, args, kwargs):
        self.snapshot("backward")
        self.phase = "gradient_check"
        norms = [torch.linalg.vector_norm(p.grad.detach()) for p in self.model.parameters()
                 if p.grad is not None]
        norm = torch.linalg.vector_norm(torch.stack(norms)).item()
        self.row.update(gradient_norm=norm, gradients_finite=math.isfinite(norm))
        if not self.row["gradients_finite"]:
            raise RuntimeError("Non-finite gradient norm")
        self.mark("optimizer")

    def optimizer_end(self, optimizer, args, kwargs):
        self.snapshot("optimizer")

    def finish(self, loss):
        self.row.update(loss=loss.item(), update_seconds=time.perf_counter() - self.update_start)
        self.rows.append(self.row)
        write_rows(self.out / "training_metrics.csv", self.rows)
        if not math.isfinite(self.row["loss"]):
            raise RuntimeError("Non-finite training loss")

    def validation(self, metrics, prediction, target):
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(prediction).all() and torch.isfinite(target).all())
        self.validation_result = dict(seconds=time.perf_counter() - self.started, finite=finite,
                                      loss=metrics["mse_normalised"],
                                      mean_relative_l2=metrics["rel_l2"].mean().item(), **memory())
        write_json(self.out / "validation.json", self.validation_result)
        if not finite:
            raise RuntimeError("Non-finite validation prediction or target")

    def checkpoint(self):
        torch.cuda.synchronize()
        self.checkpoint_seconds = time.perf_counter() - self.started


def prepare(cfg, root, device, probe, report):
    sets, train_hashes = {}, set()
    for split, count in (("train", 4096), ("validation", 256)):
        probe.mark(f"dataset_{split}")
        batch = data.load_or_generate(cfg, split, count, device, root)
        torch.cuda.synchronize()
        duration = time.perf_counter() - probe.started
        for name in ("permeability", "darcy"):
            grid = batch[name]
            if grid.shape != (count, 256, 256) or grid.dtype != np.float32:
                raise RuntimeError(f"Unexpected {split} {name} layout")
            if not all(np.isfinite(field).all() for field in grid):
                raise RuntimeError(f"Non-finite {split} {name}")
        hashes = {hashlib.sha256(field.tobytes()).digest() for field in batch["permeability"]}
        if hashes & train_hashes:
            raise RuntimeError("Training and validation permeability overlap")
        if split == "train":
            train_hashes = hashes
        report["datasets"][split] = {
            **batch["info"], "preparation_seconds": duration, "distinct_fields": len(hashes),
            "file_bytes": (root / cfg.data.dir / batch["info"]["file"]).stat().st_size,
            "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "gpu": memory(),
        }
        sets[split] = batch
        write_json(probe.out / "report.json", report)
        print(split, report["datasets"][split], flush=True)
    return sets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        write_json(out / "report.json", {"status": "BLOCKED", "reason": "CUDA Tesla T4 unavailable"})
        return
    started = time.perf_counter()
    env = collect_metadata(torch.device("cuda"))
    env.update(scipy_version=scipy.__version__, total_vram_mib=torch.cuda.get_device_properties(0).total_memory / MIB)
    if env["git_dirty"] or env["warp_version"] != "1.18.0" or env["physicsnemo"]["commit"] != "b45a5c810c741e6b41f8515be24c51121f8fc21f":
        raise RuntimeError("Probe requires the clean checkout and pinned generator environment")
    write_json(out / "environment.json", env)
    full = load_config("full")
    cfg = load_config("full", OVERRIDES)
    assert (cfg.training.resolution, cfg.training.batch_size, cfg.training.pseudo_epoch_sample_size,
            cfg.validation.sample_size, cfg.data.convergence_threshold, cfg.data.independent_samples) == (256, 64, 4096, 256, 1e-7, True)
    probe = Probe(out)
    report = dict(status="running", datasets={}, expected_updates=64, optimizer_updates=0,
                  settings=dict(resolution=256, batch_size=64, modes=16, seed=0, epochs=1, tolerance=1e-7))
    stop, readings, errors = threading.Event(), [], []
    monitor = threading.Thread(target=memory_monitor, args=(stop, readings, errors))
    monitor.start()
    try:
        with tempfile.TemporaryDirectory(prefix="fno-probe-") as scratch:
            root = Path(scratch)
            sets = prepare(cfg, root, torch.device("cuda"), probe, report)
            probe.phase = "training_setup"
            result = train_run(cfg, 16, 0, root / "run", torch.device("cuda"), sets, probe=probe)
        if len(probe.rows) != report["expected_updates"]:
            raise RuntimeError("Unexpected number of updates")
        report.update(status="complete", training=result, checkpoint_seconds=probe.checkpoint_seconds,
                      mean_update_seconds=np.mean([row["update_seconds"] for row in probe.rows]).item(),
                      steady_update_seconds=np.mean([row["update_seconds"] for row in probe.rows[1:]]).item())
        epoch = max(0, result["train_seconds"] - probe.validation_result["seconds"])
        estimate = projection(epoch, probe.validation_result["seconds"], probe.checkpoint_seconds,
                              full.training.max_pseudo_epochs, len(full.study.seeds) * len(full.study.fno_modes),
                              full.validation.validation_pseudo_epochs, full.training.rec_results_freq)
        write_json(out / "runtime_projection.json", estimate)
        fig, ax = plt.subplots()
        ax.plot([row["update"] for row in probe.rows], [row["loss"] for row in probe.rows])
        ax.set(xlabel="Optimizer update", ylabel="Normalized MSE")
        fig.savefig(out / "training_loss.png")
        plt.close(fig)
    except torch.cuda.OutOfMemoryError as exc:
        report.update(status="OOM", failed_phase=probe.phase, error=str(exc), failure_memory=memory())
        print(f"CUDA OOM during {probe.phase}: {exc}", flush=True)
    except Exception as exc:
        report.update(status="error", failed_phase=probe.phase, error=repr(exc))
        raise
    finally:
        stop.set()
        monitor.join()
        if readings:
            write_rows(out / "gpu_memory.csv", [dict(seconds=s, device_memory_mib=m) for s, m in readings])
        report.update(optimizer_updates=len(probe.rows), elapsed_seconds=time.perf_counter() - started,
                      whole_device_sampled_peak_mib=max((m for _, m in readings), default=None),
                      memory_monitor_errors=errors)
        if probe.train_peaks:
            report["train_peak_allocated_mib"] = max(p["peak_allocated_mib"] for p in probe.train_peaks)
            report["train_peak_reserved_mib"] = max(p["peak_reserved_mib"] for p in probe.train_peaks)
        write_json(out / "report.json", report)
        print(report, flush=True)


if __name__ == "__main__":
    main()
