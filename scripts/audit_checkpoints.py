"""Recompute the test errors of a trained study from its checkpoints, independently of ``metrics.py``.

    python scripts/audit_checkpoints.py
    python scripts/audit_checkpoints.py --study runs/local --device cpu

For every run: load the final checkpoint, predict the cached test set, invert the
normalisation with the constants of the configuration, and compute the per-sample relative
L2 error in NumPy (float64). The result is compared with ``eval_metrics.json``.

Also checks, without data:

- the parameter count against the analytic count of the spectral weights,
  ``layers * 2 * C_in * C_out * m^2`` complex numbers, the rest being independent of ``m``;
- that the retained Fourier blocks ``[:m]`` and ``[-m:]`` of the first FFT dimension do not
  overlap for every number of modes the study validator accepts.

Writes ``results/audit/checkpoint_recomputation.json``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ROOT
from fno_darcy import engine
from fno_darcy.config import load_config
from fno_darcy.model import build_fno, count_parameters, max_fno_modes
from physicsnemo.utils import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="local")
    parser.add_argument("--study", type=Path, default=ROOT / "runs" / "local", help="run directory with checkpoints")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "audit" / "checkpoint_recomputation.json")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    device = torch.device(args.device)
    sets = engine.load_datasets(cfg, device, ROOT, with_train=False)
    test = sets["test"]
    k = torch.from_numpy((test["permeability"] - cfg.normaliser.permeability.mean) / cfg.normaliser.permeability.std_dev)[:, None]
    u_true = test["darcy"].astype(np.float64)

    width, layers = cfg.arch.fno.latent_channels, cfg.arch.fno.fno_layers
    study = args.study.resolve()
    report = {"study": str(study.relative_to(ROOT)) if study.is_relative_to(ROOT) else str(study), "device": str(device), "test_file": test["info"]["file"], "runs": [], "parameters": []}
    non_spectral = set()
    for m in cfg.study.fno_modes:
        n = count_parameters(build_fno(cfg, m))
        spectral = layers * 2 * width * width * m * m * 2  # two complex blocks per layer, stored as real pairs
        non_spectral.add(n - spectral)
        report["parameters"].append({"fno_modes": m, "parameters": n, "spectral_weights": spectral, "spectral_fraction": spectral / n})
    report["non_spectral_parameters_independent_of_modes"] = len(non_spectral) == 1

    # overlap of the two retained blocks along the full (non-rfft) FFT dimension
    padded = cfg.training.resolution + cfg.arch.fno.padding
    accepted = max_fno_modes(cfg.training.resolution, cfg.arch.fno.padding)
    report["mode_limit"] = {
        "padded_grid": padded,
        "validator_accepts_up_to": accepted,
        "largest_non_overlapping": padded // 2,
        "overlap_for_largest_accepted": 2 * accepted > padded,
    }

    worst = 0.0
    for modes, seed in engine.study_runs(cfg):
        run_dir = args.study / engine.run_name(modes, seed)
        model = build_fno(cfg, modes).to(device)
        epoch = load_checkpoint(str(run_dir / "checkpoints"), models=model, device=device)
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(k[i : i + 32].to(device)).cpu() for i in range(0, len(k), 32)])
        u_pred = pred[:, 0].double().numpy() * cfg.normaliser.darcy.std_dev + cfg.normaliser.darcy.mean
        err = np.linalg.norm((u_pred - u_true).reshape(len(u_true), -1), axis=1) / np.linalg.norm(u_true.reshape(len(u_true), -1), axis=1)
        stored = json.loads((run_dir / "eval_metrics.json").read_text())
        stored_per_sample = np.asarray(stored["test_rel_l2_per_sample"])
        rel = abs(err.mean() - stored["test_rel_l2"]["mean"]) / stored["test_rel_l2"]["mean"]
        worst = max(worst, rel)
        report["runs"].append(
            {
                "run": run_dir.name,
                "checkpoint_pseudo_epoch": int(epoch),
                "recomputed_mean": float(err.mean()),
                "stored_mean": stored["test_rel_l2"]["mean"],
                "relative_difference_of_mean": float(rel),
                "max_abs_per_sample_difference": float(np.abs(err - stored_per_sample).max()),
            }
        )
        print(f"{run_dir.name}: recomputed {err.mean():.6e} stored {stored['test_rel_l2']['mean']:.6e} rel diff {rel:.1e}")
    report["largest_relative_difference"] = worst
    print(json.dumps({k: report[k] for k in ("parameters", "non_spectral_parameters_independent_of_modes", "mode_limit")}, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
