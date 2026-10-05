"""The loop over Fourier modes and seeds, and the summary tables."""

import csv
import json
import statistics
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from . import engine
from .provenance import write_json

SUMMARY_FIELDS = [
    "run",
    "fno_modes",
    "seed",
    "parameters",
    "final_train_loss",
    "validation_mse_normalised",
    "test_mse_normalised",
    "test_rel_l2_mean",
    "test_rel_l2_std",
    "test_rel_l2_median",
    "test_rel_l2_max",
    "train_seconds",
    "optimisation_seconds",
    "data_seconds",
    "inference_ms_per_batch",
    "inference_ms_per_sample",
    "peak_train_memory_mb",
    "peak_inference_memory_mb",
]
AGGREGATED = ["test_rel_l2_mean", "final_train_loss", "train_seconds", "optimisation_seconds", "inference_ms_per_sample", "peak_train_memory_mb"]


def train_study(cfg: DictConfig, root: Path, resume: bool = False) -> Path:
    """Train every (modes, seed) run of the study. ``resume`` skips runs that already finished."""
    out_dir, device, sets = engine.prepare_study(cfg, root, write_metadata=not resume)
    for modes, seed in engine.study_runs(cfg):
        run_dir = out_dir / engine.run_name(modes, seed)
        if resume and (run_dir / "train_metrics.json").exists():
            print(f"[skip] {run_dir.name} already trained", flush=True)
            continue
        metrics = engine.train_run(cfg, modes, seed, run_dir, device, sets)
        print(
            f"[train] {run_dir.name}: {metrics['parameters']} parameters, "
            f"final train loss {metrics['final_train_loss']:.3e}, {metrics['train_seconds']:.1f} s on {device}",
            flush=True,
        )
    return out_dir


def evaluate_study(cfg: DictConfig, root: Path) -> dict:
    """Evaluate every run from its checkpoint and write the summary files and plotting fields."""
    out_dir, device, sets = engine.prepare_study(cfg, root, write_metadata=False)
    n_plot = cfg.evaluation.plot_samples
    fields = {
        "permeability": sets["test"]["permeability"][:n_plot],
        "truth": sets["test"]["darcy"][:n_plot],
    }
    rows = []
    for modes, seed in engine.study_runs(cfg):
        run_dir = out_dir / engine.run_name(modes, seed)
        evaluation, prediction = engine.evaluate_run(cfg, modes, run_dir, device, sets)
        train = json.loads((run_dir / "train_metrics.json").read_text())
        row = {**train, **{k: v for k, v in evaluation.items() if not isinstance(v, (dict, list))}}
        for stat, value in evaluation["test_rel_l2"].items():
            row[f"test_rel_l2_{stat}"] = value
        rows.append({k: row[k] for k in SUMMARY_FIELDS})
        if seed == cfg.study.seeds[0]:
            fields[f"prediction_modes{modes:02d}"] = prediction.numpy()
        print(f"[eval] {run_dir.name}: test relative L2 {row['test_rel_l2_mean']:.4e}", flush=True)

    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "name": cfg.name,
        "device": str(device),
        "evaluation_device_note": "inference times and memory are those of the evaluating device",
        "by_fno_modes": aggregate(rows),
        "runs": rows,
    }
    write_json(out_dir / "summary.json", summary)
    np.savez_compressed(out_dir / "fields.npz", plot_seed=cfg.study.seeds[0], **fields)
    return summary


def aggregate(rows: list[dict]) -> list[dict]:
    """Mean and sample standard deviation over seeds for every number of modes."""
    out = []
    for modes in sorted({r["fno_modes"] for r in rows}):
        group = [r for r in rows if r["fno_modes"] == modes]
        entry = {"fno_modes": modes, "parameters": group[0]["parameters"], "n_seeds": len(group)}
        for key in AGGREGATED:
            values = [r[key] for r in group if r[key] is not None]
            entry[f"{key}_mean"] = statistics.fmean(values) if values else None
            entry[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else None
        out.append(entry)
    return out
