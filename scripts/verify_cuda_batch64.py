"""Measure the full-study Darcy generator against independent CPU sparse solves."""

import argparse
import csv
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import scipy
import torch
from _bootstrap import ROOT
from fno_darcy.config import load_config
from fno_darcy.data import make_datapipe
from fno_darcy.provenance import collect_metadata, write_json
from fno_darcy.reference import direct_solve
from verify_generator import plt, rel_l2

SEEDS = (1000, 1001, 1002, 1003)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    p95, p99 = np.percentile(values, [95, 99])
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                std=float(values.std()), p95=float(p95), p99=float(p99),
                max=float(values.max()))


def final_updates(current, previous, threshold):
    updates = np.max(np.abs(current - previous), axis=(1, 2))
    return updates, np.isfinite(updates) & (updates < threshold)


def compare_sample(k, u, dx, edge):
    k, u = k.astype(np.float64), u.astype(np.float64)
    start = time.perf_counter()
    kernel = direct_solve(k, dx, "kernel")
    flux = direct_solve(k, dx, "flux")
    return {
        "reference_seconds": time.perf_counter() - start,
        "stencil_rel_l2": rel_l2(u, kernel),
        "conservative_rel_l2": rel_l2(u, flux),
        "boundary_stencil_rel_l2": rel_l2(u[edge], kernel[edge]),
    }


def memory_monitor(stop, readings, errors):
    # Whole-device samples cover both Torch and Warp allocations.
    started = time.perf_counter()
    while not stop.is_set():
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "-i", "0", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"], text=True, timeout=5,
            )
            readings.append((time.perf_counter() - started, int(output.strip())))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            errors.append(str(exc))
            return
        stop.wait(0.1)


def generate(cfg, seed, size, threshold):
    np.random.seed(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    pipe = make_datapipe(cfg, size, "cuda", normaliser=None)
    torch.cuda.synchronize()
    init_seconds = time.perf_counter() - start

    start = time.perf_counter()
    batch = next(iter(pipe))
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start

    visible = {key: value.cpu().numpy().copy() for key, value in batch.items()}
    k, u = pipe.permeability.numpy(), pipe.darcy0.numpy()
    updates, met = final_updates(u, pipe.darcy1.numpy(), threshold)
    upstream = float(pipe.inf_residual.numpy()[0])
    if np.isfinite(updates).all() and float(updates.max()) != upstream:
        raise RuntimeError("Per-sample update maximum disagrees with the upstream reduction")
    values, counts = np.unique(k, return_counts=True)
    hashes = [hashlib.sha256(field.tobytes()).hexdigest() for field in visible["permeability"]]
    expected_shape = (size, 1, cfg.training.resolution, cfg.training.resolution)
    record = {
        "seed": seed, "init_seconds": init_seconds, "generation_seconds": seconds,
        "samples_per_second": size / seconds, "distinct_fields": len(set(hashes)),
        "public_fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                          for key, value in visible.items()},
        "valid_tensor_layout": all(v.shape == expected_shape and v.dtype == np.float32
                                   for v in visible.values()),
        "permeability_values": values.tolist(), "permeability_counts": counts.tolist(),
        "nan_count": int(np.isnan(k).sum() + np.isnan(u).sum()),
        "infinity_count": int(np.isinf(k).sum() + np.isinf(u).sum()),
        "upstream_max_update": upstream, "batch_met_tolerance": upstream < threshold,
        "samples_missing_tolerance": int((~met).sum()),
    }
    return record, hashes, k, u, updates, met


def analyze(out, report):
    rows = []
    dx = report["config"]["dx"]
    for record in report["batches"]:
        seed = record["seed"]
        with np.load(out / f"fields_{seed}.npz") as arrays:
            k, u = arrays["k"], arrays["u"]
            updates, met = arrays["updates"], arrays["met"]
        edge = np.ones(k.shape[1:], dtype=bool)
        edge[1:-1, 1:-1] = False
        for i, (field, pressure) in enumerate(zip(k, u, strict=True)):
            row = dict(seed=seed, sample=i, final_max_update=float(updates[i]),
                       sample_met_tolerance=bool(met[i]))
            if np.isfinite(field).all() and np.isfinite(pressure).all():
                row.update(compare_sample(field, pressure, dx, edge))
            rows.append(row)
        write_rows(out / "per_sample.csv", rows)
        print(f"CPU references saved through seed {seed}: {len(rows)} samples", flush=True)
    metrics = ("stencil_rel_l2", "conservative_rel_l2", "boundary_stencil_rel_l2",
               "final_max_update")
    report["statistics"] = {
        metric: summarize(values) for metric in metrics
        if (values := [r[metric] for r in rows if metric in r and np.isfinite(r[metric])])
    }
    report["reference_samples"] = sum("stencil_rel_l2" in row for row in rows)
    report["cpu_reference_seconds"] = sum(row.get("reference_seconds", 0) for row in rows)
    report["samples_missing_tolerance"] = sum(not r["sample_met_tolerance"] for r in rows)
    report["samples_missing_tolerance_percent"] = 100 * report["samples_missing_tolerance"] / len(rows)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), layout="constrained")
    for ax, metric, label in zip(axes, metrics[:2], ("NVIDIA stencil", "Conservative Darcy"), strict=True):
        errors = sorted(100 * r[metric] for r in rows if metric in r)
        if errors:
            ax.step(errors, np.arange(1, len(errors) + 1) / len(errors), where="post")
        ax.set(xlabel=f"{label} relative L2 error (%)", ylabel="Fraction of samples")
        ax.grid(alpha=0.25)
    fig.savefig(out / "error_distributions.png", dpi=160)
    plt.close(fig)


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--analyze-saved", action="store_true")
    args = parser.parse_args()
    out = args.out
    started = time.perf_counter()
    if args.analyze_saved:
        report = json.loads((out / "report.json").read_text())
    else:
        if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
            raise SystemExit("An NVIDIA Tesla T4 is required")
        out.mkdir(parents=True, exist_ok=False)
        cfg = load_config("full")
        size, threshold = cfg.data.generation_batch_size, cfg.data.convergence_threshold
        resolution = cfg.training.resolution
        if (resolution, size, threshold, cfg.data.independent_samples) != (256, 64, 1e-7, True):
            raise RuntimeError("Full-study generator configuration changed")
        report = {
            "environment": {**collect_metadata(torch.device("cuda")), "scipy_version": scipy.__version__},
            "config": dict(generator="IndependentDarcy2D", resolution=resolution, batch_size=size,
                           convergence_threshold=threshold, max_iterations=cfg.data.max_iterations,
                           dx=1 / (resolution + 1), seeds=SEEDS),
            "batches": [],
            "status": "generation_in_progress",
        }
        baseline = int(subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True))
        stop, readings, errors = threading.Event(), [], []
        monitor = threading.Thread(target=memory_monitor, args=(stop, readings, errors))
        monitor.start()
        hashes = set()
        try:
            for seed in SEEDS:
                record, batch_hashes, k, u, updates, met = generate(cfg, seed, size, threshold)
                np.savez(out / f"fields_{seed}.npz", k=k, u=u, updates=updates, met=met)
                hashes.update(batch_hashes)
                report["batches"].append(record)
                report["distinct_fields"] = len(hashes)
                write_json(out / "report.json", report)
                print(f"CUDA seed {seed}: {record}", flush=True)
            replay, _, k, u, _, _ = generate(cfg, SEEDS[0], size, threshold)
            with np.load(out / f"fields_{SEEDS[0]}.npz") as first:
                report["replay"] = {
                    "seed": SEEDS[0], "pressure_exact": bool(np.array_equal(u, first["u"])),
                    "permeability_exact": bool(np.array_equal(k, first["k"])), "timing": replay,
                }
        finally:
            stop.set()
            monitor.join()
            write_rows(out / "gpu_memory.csv", [
                dict(seconds=seconds, device_memory_mib=mib) for seconds, mib in readings
            ] or [dict(seconds="", device_memory_mib="")])
            peak = max((mib for _, mib in readings), default=None)
            report["memory"] = dict(baseline_mib=baseline, sampled_peak_mib=peak,
                                    sampled_increase_mib=None if peak is None else peak - baseline,
                                    samples=len(readings), errors=errors)
            report["generation_stage_seconds"] = time.perf_counter() - started
            write_json(out / "report.json", report)
    start_analysis = time.perf_counter()
    analyze(out, report)
    report["analysis_wall_seconds"] = time.perf_counter() - start_analysis
    report["verification_seconds"] = report["generation_stage_seconds"] + report["analysis_wall_seconds"]
    report["status"] = "complete"
    write_json(out / "report.json", report)
    print(f"Completed {report['reference_samples']} sparse comparisons in {report['verification_seconds']:.2f} s", flush=True)


if __name__ == "__main__":
    main()
