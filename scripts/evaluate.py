"""Evaluate the checkpoints of a trained study on the fixed validation and test sets.

    python scripts/evaluate.py --config smoke

Writes ``summary.csv``, ``summary.json`` and ``fields.npz`` to the study directory. Pass the
same overrides that were used for training.

    python scripts/evaluate.py --config full --verify device=mps data.generation_device=cuda

``--verify`` recomputes the test errors from the checkpoints, compares them with the stored
values and writes nothing. It is the check for results produced on another machine.
"""

import argparse

from _bootstrap import ROOT
from fno_darcy.config import load_config
from fno_darcy.study import evaluate_study, verify_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--verify", action="store_true", help="compare with the stored errors; write nothing")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="relative tolerance of --verify")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.verify:
        worst = max(r["relative_difference"] for r in verify_study(cfg, ROOT))
        print(f"largest relative difference of the test error: {worst:.2e} (tolerance {args.tolerance:g})")
        raise SystemExit(0 if worst <= args.tolerance else 1)
    summary = evaluate_study(cfg, ROOT)
    for entry in summary["by_fno_modes"]:
        print(
            f"modes {entry['fno_modes']:>2}: {entry['parameters']:>9} parameters, "
            f"test relative L2 {entry['test_rel_l2_mean_mean']:.4e} over {entry['n_seeds']} seed(s)"
        )


if __name__ == "__main__":
    main()
