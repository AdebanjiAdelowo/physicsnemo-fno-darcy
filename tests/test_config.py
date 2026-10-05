import pytest
from fno_darcy.config import CONFIG_DIR, load_config
from fno_darcy.engine import validate_study
from fno_darcy.model import max_fno_modes
from omegaconf import OmegaConf

UPSTREAM_KEYS = ("arch", "normaliser", "scheduler", "training", "validation")
# examples/cfd/darcy_fno/config.yaml at the pinned PhysicsNeMo commit
UPSTREAM = {
    "arch": {
        "decoder": {"out_features": 1, "layers": 1, "layer_size": 32},
        "fno": {"in_channels": 1, "dimension": 2, "latent_channels": 32, "fno_layers": 4, "fno_modes": 12, "padding": 9},
    },
    "normaliser": {"permeability": {"mean": 1.25, "std_dev": 0.75}, "darcy": {"mean": 4.52e-2, "std_dev": 2.79e-2}},
    "scheduler": {"initial_lr": 1e-3, "decay_rate": 0.85, "decay_pseudo_epochs": 8},
    "training": {"resolution": 256, "batch_size": 64, "rec_results_freq": 8, "max_pseudo_epochs": 256, "pseudo_epoch_sample_size": 2048},
    "validation": {"sample_size": 256, "validation_pseudo_epochs": 4},
}


@pytest.mark.parametrize("name", ["official", "smoke", "local", "full"])
def test_configs_load_and_are_valid(name):
    cfg = load_config(name)
    assert cfg.name == name
    assert cfg.output_dir == f"runs/{name}"
    validate_study(cfg)


def test_official_config_keeps_upstream_values():
    cfg = OmegaConf.to_container(load_config("official"))
    assert {k: cfg[k] for k in UPSTREAM_KEYS} == UPSTREAM
    assert cfg["data"]["source"] == "stream"
    assert cfg["data"]["independent_samples"] is False
    assert cfg["study"]["fno_modes"] == [12]


@pytest.mark.parametrize("name", ["local", "full"])
def test_study_configs_change_only_documented_architecture_keys(name):
    """The studies vary the number of modes and keep the upstream architecture."""
    cfg = OmegaConf.to_container(load_config(name))
    assert cfg["arch"] == UPSTREAM["arch"]
    assert cfg["normaliser"] == UPSTREAM["normaliser"]
    assert cfg["study"]["fno_modes"] == [4, 8, 12, 16]
    assert cfg["data"]["independent_samples"] is True


def test_overrides_are_applied():
    cfg = load_config("smoke", ["training.max_pseudo_epochs=5", "device=cpu"])
    assert cfg.training.max_pseudo_epochs == 5


def test_too_many_modes_is_rejected():
    cfg = load_config("smoke")
    limit = max_fno_modes(cfg.training.resolution, cfg.arch.fno.padding)
    validate_study(load_config("smoke", [f"study.fno_modes=[{limit}]"]))
    with pytest.raises(ValueError, match="fno_modes"):
        validate_study(load_config("smoke", [f"study.fno_modes=[{limit + 1}]"]))


def test_config_dir_exists():
    assert (CONFIG_DIR / "official.yaml").exists()
