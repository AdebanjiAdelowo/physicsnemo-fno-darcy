"""Plot a study directory.

    python scripts/plot_results.py --study runs/smoke --out runs/smoke/figures
    python scripts/plot_results.py --study results/local --out figures

Figures are named ``<study name>_<figure>.png``.
"""

import argparse
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from fno_darcy.plotting import plot_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True, type=Path, help="directory with summary.json and fields.npz")
    parser.add_argument("--out", required=True, type=Path, help="directory for the figures")
    args = parser.parse_args()
    for path in plot_all(args.study, args.out, prefix=args.study.resolve().name):
        print(path)


if __name__ == "__main__":
    main()
