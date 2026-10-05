"""Evaluate the checkpoints of a trained study on the fixed validation and test sets.

    python scripts/evaluate.py --config smoke

Writes ``summary.csv``, ``summary.json`` and ``fields.npz`` to the study directory. Pass the
same overrides that were used for training.
"""

import argparse

from _bootstrap import ROOT
from fno_darcy.config import load_config
from fno_darcy.study import evaluate_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    summary = evaluate_study(cfg, ROOT)
    for entry in summary["by_fno_modes"]:
        print(
            f"modes {entry['fno_modes']:>2}: {entry['parameters']:>9} parameters, "
            f"test relative L2 {entry['test_rel_l2_mean_mean']:.4e} over {entry['n_seeds']} seed(s)"
        )


if __name__ == "__main__":
    main()
