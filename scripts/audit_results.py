"""Recompute the reduced-study table and its uncertainty from the per-run records.

    python scripts/audit_results.py
    python scripts/audit_results.py --study results/local --study runs/audit_seeds345

Reads only ``<study>/modes*_seed*/{train,eval}_metrics.json`` and ``history.csv``; it does
not use ``summary.csv`` or ``summary.json``, so it is an independent recomputation of them.
Several ``--study`` directories with the same protocol are pooled by seed.

Reports, per number of Fourier modes:

- mean, sample standard deviation and a t-based 95% interval over seeds of the per-run
  mean test relative L2 error (the seed is the unit of replication);
- the test-set sampling error of each run (standard error of its mean over test samples);
- the final-checkpoint jitter: standard deviation of the validation error over the last
  five pseudo-epochs of each run;
- Welch t-tests and a seed-level permutation test between settings;
- paired per-sample comparisons of the seed-averaged errors on the shared test set.

Writes ``results/audit/reduced_study_statistics.json``.
"""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

from _bootstrap import ROOT

README_CLAIMS = {4: 0.01458, 8: 0.00920, 12: 0.00979, 16: 0.00970}


def load_runs(study: Path) -> list[dict]:
    runs = []
    for run_dir in sorted(study.glob("modes*_seed*")):
        train = json.loads((run_dir / "train_metrics.json").read_text())
        evaluation = json.loads((run_dir / "eval_metrics.json").read_text())
        with open(run_dir / "history.csv") as f:
            history = list(csv.DictReader(f))
        per_sample = np.asarray(evaluation["test_rel_l2_per_sample"], dtype=np.float64)
        val = [float(r["validation_rel_l2"]) for r in history if r["validation_rel_l2"]]
        runs.append(
            {
                "study": _rel(study),
                "run": run_dir.name,
                "fno_modes": train["fno_modes"],
                "seed": train["seed"],
                "parameters": train["parameters"],
                "optimiser_steps": train["optimiser_steps"],
                "test_rel_l2_per_sample": per_sample,
                "test_rel_l2_mean": float(per_sample.mean()),
                "stored_test_rel_l2_mean": evaluation["test_rel_l2"]["mean"],
                "test_sampling_se": float(per_sample.std(ddof=1) / np.sqrt(len(per_sample))),
                "last5_validation_rel_l2_std": float(np.std(val[-5:], ddof=1)) if len(val) >= 5 else None,
                "last5_validation_rel_l2_mean": float(np.mean(val[-5:])) if len(val) >= 5 else None,
                "final_validation_rel_l2": val[-1],
                "validation_decreasing_at_end": bool(np.polyfit(np.arange(5), val[-5:], 1)[0] < 0) if len(val) >= 5 else None,
            }
        )
    return runs


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def t_interval(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    half = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() - half), float(x.mean() + half)


def permutation_p(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided permutation test of the difference of means over all relabellings."""
    pooled = np.concatenate([a, b])
    observed = abs(a.mean() - b.mean())
    count = total = 0
    for idx in itertools.combinations(range(len(pooled)), len(a)):
        mask = np.zeros(len(pooled), bool)
        mask[list(idx)] = True
        count += abs(pooled[mask].mean() - pooled[~mask].mean()) >= observed - 1e-15
        total += 1
    return count / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", type=Path, action="append", help="study directory (repeatable)")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "audit" / "reduced_study_statistics.json")
    args = parser.parse_args()
    studies = args.study or [ROOT / "results" / "local"]
    runs = [r for s in studies for r in load_runs(s)]
    steps = {r["optimiser_steps"] for r in runs}
    if len(steps) != 1:
        raise SystemExit(f"studies with different training budgets cannot be pooled: {sorted(steps)}")
    modes = sorted({r["fno_modes"] for r in runs})

    report = {"studies": [_rel(s) for s in studies], "optimiser_steps": steps.pop(), "by_fno_modes": [], "comparisons": []}
    by_mode = {}
    for m in modes:
        group = sorted((r for r in runs if r["fno_modes"] == m), key=lambda r: r["seed"])
        x = np.array([r["test_rel_l2_mean"] for r in group])
        by_mode[m] = (x, np.stack([r["test_rel_l2_per_sample"] for r in group]))
        lo, hi = t_interval(x) if len(x) > 1 else (None, None)
        entry = {
            "fno_modes": m,
            "seeds": [r["seed"] for r in group],
            "parameters": group[0]["parameters"],
            "per_seed_test_rel_l2": x.tolist(),
            "mean": float(x.mean()),
            "std_over_seeds": float(x.std(ddof=1)) if len(x) > 1 else None,
            "t95_interval_of_mean": [lo, hi],
            "readme_claim": README_CLAIMS.get(m),
            "max_abs_diff_recomputed_vs_stored_run_mean": float(max(abs(r["test_rel_l2_mean"] - r["stored_test_rel_l2_mean"]) for r in group)),
            "mean_test_sampling_se_per_run": float(np.mean([r["test_sampling_se"] for r in group])),
            "mean_last5_validation_jitter_std": float(np.mean([r["last5_validation_rel_l2_std"] for r in group])),
            "runs_with_validation_still_decreasing": int(sum(bool(r["validation_decreasing_at_end"]) for r in group)),
        }
        report["by_fno_modes"].append(entry)
        print(
            f"modes {m:>2}: n={len(x)} mean {100 * x.mean():.3f}% sd {100 * (entry['std_over_seeds'] or 0):.3f} pp "
            f"95% CI [{100 * (lo or 0):.3f}, {100 * (hi or 0):.3f}]%  claim {README_CLAIMS.get(m)}  "
            f"test-SE {100 * entry['mean_test_sampling_se_per_run']:.3f} pp  "
            f"val jitter {100 * entry['mean_last5_validation_jitter_std']:.3f} pp"
        )

    for a, b in itertools.combinations(modes, 2):
        xa, pa = by_mode[a]
        xb, pb = by_mode[b]
        welch = stats.ttest_ind(xa, xb, equal_var=False)
        # paired over the shared test samples, seed-averaged errors of each setting
        diff = pa.mean(0) - pb.mean(0)
        wilcoxon = stats.wilcoxon(pa.mean(0), pb.mean(0))
        comp = {
            "a": a,
            "b": b,
            "mean_difference_a_minus_b": float(xa.mean() - xb.mean()),
            "welch_t_p": float(welch.pvalue),
            "seed_permutation_p": permutation_p(xa, xb),
            "ranges_overlap": bool(min(xa.max(), xb.max()) >= max(xa.min(), xb.min())),
            "paired_per_sample_mean_difference": float(diff.mean()),
            "paired_per_sample_fraction_a_better": float((diff < 0).mean()),
            "paired_wilcoxon_p": float(wilcoxon.pvalue),
        }
        report["comparisons"].append(comp)
        print(
            f"{a:>2} vs {b:>2}: diff {100 * comp['mean_difference_a_minus_b']:+.3f} pp, Welch p={comp['welch_t_p']:.3f}, "
            f"permutation p={comp['seed_permutation_p']:.3f}, ranges overlap={comp['ranges_overlap']}, "
            f"paired: {100 * comp['paired_per_sample_fraction_a_better']:.0f}% of samples favour {a} (Wilcoxon p={comp['paired_wilcoxon_p']:.2g})"
        )
    report["note"] = (
        "Paired per-sample tests treat the test samples as the unit and ignore seed variation; "
        "they show where on the test set the seed-averaged models differ, not whether the difference "
        "would survive new training seeds. The seed-level tests are the relevant ones for that."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"written {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")


if __name__ == "__main__":
    main()
