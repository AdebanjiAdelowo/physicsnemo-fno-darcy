"""Train the runs of a study and evaluate them.

    python scripts/train.py --config smoke
    python scripts/train.py --config local training.max_pseudo_epochs=10

Arguments after the options are Hydra overrides of ``configs/<config>.yaml``.
"""

import argparse

from _bootstrap import ROOT
from fno_darcy.config import load_config
from fno_darcy.study import evaluate_study, train_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="config name: official, smoke, local or full")
    parser.add_argument("--resume", action="store_true", help="skip runs that already finished")
    parser.add_argument("--no-evaluate", action="store_true", help="train only")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. device=cpu")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    out_dir = train_study(cfg, ROOT, resume=args.resume)
    if not args.no_evaluate:
        evaluate_study(cfg, ROOT)
    print(f"Study written to {out_dir}")


if __name__ == "__main__":
    main()
