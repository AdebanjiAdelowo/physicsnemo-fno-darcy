"""End-to-end run of a tiny study on CPU: train, evaluate, plot, package, publish."""

import csv
import json
import zipfile

import numpy as np
import pytest
import torch
from fno_darcy import engine
from fno_darcy.config import load_config
from fno_darcy.plotting import plot_all
from fno_darcy.study import evaluate_study, train_study

OVERRIDES = ["training.pseudo_epoch_sample_size=16", "validation.sample_size=8", "data.test_samples=8", "data.generation_batch_size=8"]


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("repo")
    cfg = load_config("smoke", OVERRIDES)
    out_dir = train_study(cfg, root)
    summary = evaluate_study(cfg, root)
    return cfg, root, out_dir, summary


def test_outputs_exist(study):
    cfg, _, out_dir, _ = study
    for modes, seed in engine.study_runs(cfg):
        run = out_dir / engine.run_name(modes, seed)
        for name in ("history.csv", "train_metrics.json", "eval_metrics.json"):
            assert (run / name).exists()
        assert list((run / "checkpoints").glob("*.mdlus"))
    for name in ("study_metadata.json", "summary.csv", "summary.json", "fields.npz"):
        assert (out_dir / name).exists()


def test_metadata_records_the_environment(study):
    _, _, out_dir, _ = study
    meta = json.loads((out_dir / "study_metadata.json").read_text())
    for key in ("timestamp_utc", "git_commit", "device", "python_version", "torch_version", "physicsnemo", "config", "datasets"):
        assert key in meta
    assert meta["physicsnemo"]["version"]
    assert meta["config"]["study"]["seeds"] == [0]
    assert set(meta["datasets"]) == {"train", "validation", "test"}
    assert meta["datasets"]["test"]["distinct_permeability_fields"] == 8


def test_history_and_summary_are_consistent(study):
    cfg, _, out_dir, summary = study
    with open(out_dir / "modes04_seed0" / "history.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == cfg.training.max_pseudo_epochs
    assert float(rows[1]["learning_rate"]) == pytest.approx(cfg.scheduler.initial_lr * cfg.scheduler.decay_rate)
    assert len(summary["runs"]) == 2
    by_modes = {e["fno_modes"]: e for e in summary["by_fno_modes"]}
    assert by_modes[8]["parameters"] > by_modes[4]["parameters"]
    for run in summary["runs"]:
        assert np.isfinite(run["test_rel_l2_mean"]) and run["test_rel_l2_std"] > 0
        assert run["train_seconds"] >= run["optimisation_seconds"] > 0
        assert run["peak_train_memory_mb"] is None  # CPU run


def test_evaluation_reproduces_from_checkpoint(study):
    """The stored test error equals a recomputation from the checkpoint and the cached test set."""
    cfg, root, out_dir, summary = study
    sets = engine.load_datasets(cfg, torch.device("cpu"), root)
    metrics, fields = engine.evaluate_run(cfg, 4, out_dir / "modes04_seed0", torch.device("cpu"), sets)
    assert metrics["test_rel_l2"]["mean"] == pytest.approx(summary["runs"][0]["test_rel_l2_mean"], rel=1e-6)
    stored = np.load(out_dir / "fields.npz")
    assert np.allclose(stored["prediction_modes04"], fields.numpy())
    assert np.array_equal(stored["truth"], sets["test"]["darcy"][: cfg.evaluation.plot_samples])


def test_training_is_reproducible(study, tmp_path):
    cfg, _, out_dir, _ = study
    sets = engine.load_datasets(cfg, torch.device("cpu"), tmp_path)
    again = engine.train_run(cfg, 4, 0, tmp_path / "again", torch.device("cpu"), sets)
    first = json.loads((out_dir / "modes04_seed0" / "train_metrics.json").read_text())
    assert again["final_train_loss"] == pytest.approx(first["final_train_loss"], rel=1e-5)


def test_figures_are_written(study, tmp_path):
    _, _, out_dir, _ = study
    paths = plot_all(out_dir, tmp_path, prefix="t")
    assert len(paths) == 4 and all(p.stat().st_size > 10_000 for p in paths)


def test_package_excludes_checkpoints(study):
    import sys

    from fno_darcy.config import REPO_ROOT

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import results as results_script

    _, _, out_dir, _ = study
    archive = results_script.package(out_dir)
    names = zipfile.ZipFile(archive).namelist()
    assert "smoke/summary.json" in names and "smoke/modes04_seed0/history.csv" in names
    assert not any("checkpoints" in n or n.endswith((".mdlus", ".pt")) for n in names)


def test_unavailable_device_is_an_error():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            engine.resolve_device("cuda")
    assert engine.resolve_device("cpu").type == "cpu"
    assert engine.resolve_device("auto").type in ("cuda", "mps", "cpu")
