"""Move study results between a run directory, an archive and the tracked ``results/``.

    python scripts/results.py package runs/full              # on the GPU machine: runs/full.zip
    python scripts/results.py bundle runs/full               # on the GPU machine: runs/full_eval_bundle.zip
    python scripts/results.py publish runs/full.zip          # locally: results/full/
    python scripts/results.py publish runs/local             # from a local run directory

``package`` holds the small records that are tracked in ``results/``; checkpoints and datasets
are never included. ``bundle`` holds what a re-evaluation on another machine needs and is not
tracked: the final checkpoint of every run and the validation and test sets, with paths
relative to the repository root, so that ``unzip <bundle> -d .`` restores them. ``publish``
refuses to overwrite an existing ``results/<name>/`` unless ``--force`` is given.
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from _bootstrap import ROOT

KEEP_SUFFIXES = {".json", ".csv", ".npz"}


def tracked_files(study_dir: Path) -> list[Path]:
    """Summaries, metadata, histories and plotting fields: everything except checkpoints and figures."""
    files = [p for p in sorted(study_dir.rglob("*")) if p.is_file() and p.suffix in KEEP_SUFFIXES and "checkpoints" not in p.parts]
    if not (study_dir / "summary.json").exists():
        raise SystemExit(f"{study_dir} has no summary.json; run scripts/evaluate.py first.")
    return files


def package(study_dir: Path) -> Path:
    archive = study_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in tracked_files(study_dir):
            z.write(path, Path(study_dir.name) / path.relative_to(study_dir))
    return archive


def bundle(study_dir: Path) -> Path:
    meta = json.loads((study_dir / "study_metadata.json").read_text())
    epoch = meta["config"]["training"]["max_pseudo_epochs"]
    data_dir = study_dir.parents[1] / meta["config"]["data"]["dir"]
    files = sorted(p for p in study_dir.glob(f"*/checkpoints/*.{epoch}.*") if p.is_file())
    files += [data_dir / meta["datasets"][split]["file"] for split in ("validation", "test")]
    missing = [str(p) for p in files if not p.exists()]
    if missing or not any(p.suffix == ".mdlus" for p in files):
        raise SystemExit(f"cannot build the bundle, missing: {missing or 'final checkpoints'}")
    archive = study_dir.parent / f"{study_dir.name}_eval_bundle.zip"
    root = study_dir.parents[1]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        for path in files:
            z.write(path, path.relative_to(root))
    return archive


def publish(source: Path, force: bool) -> Path:
    name = source.stem if source.suffix == ".zip" else source.name
    target = ROOT / "results" / name
    if target.exists():
        if not force:
            raise SystemExit(f"{target} exists; pass --force to replace it.")
        shutil.rmtree(target)
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as z:
            members = [m for m in z.namelist() if m.startswith(f"{name}/") and Path(m).suffix in KEEP_SUFFIXES]
            z.extractall(ROOT / "results", members)
    else:
        for path in tracked_files(source):
            dest = target / path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("package", help="zip the small result files of a study directory")
    p.add_argument("study", type=Path)
    p = sub.add_parser("bundle", help="zip the final checkpoints and the validation and test sets")
    p.add_argument("study", type=Path)
    p = sub.add_parser("publish", help="copy a study directory or archive into results/")
    p.add_argument("source", type=Path)
    p.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "package":
        print(package(args.study.resolve()))
    elif args.command == "bundle":
        print(bundle(args.study.resolve()))
    else:
        print(publish(args.source.resolve(), args.force))


if __name__ == "__main__":
    main()
