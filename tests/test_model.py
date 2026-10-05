import pytest
import torch
from fno_darcy.config import load_config
from fno_darcy.model import build_fno, count_parameters, max_fno_modes


@pytest.mark.parametrize("modes", [4, 8, 12, 16])
def test_forward_shape(modes):
    cfg = load_config("official")
    model = build_fno(cfg, modes)
    x = torch.randn(2, 1, 32, 32)
    assert model(x).shape == (2, 1, 32, 32)


def test_parameter_count_grows_with_modes_only_in_spectral_weights():
    """Each of the 4 spectral layers holds 2 complex weight blocks of latent² × modes² entries."""
    cfg = load_config("official")
    latent, layers = cfg.arch.fno.latent_channels, cfg.arch.fno.fno_layers
    counts = {m: count_parameters(build_fno(cfg, m)) for m in (4, 8, 12, 16)}
    assert counts[12] == count_parameters(build_fno(cfg))  # default is the upstream value
    for a, b in ((4, 8), (8, 12), (12, 16)):
        assert counts[b] - counts[a] == layers * 2 * 2 * latent * latent * (b * b - a * a)


def test_same_seed_gives_same_initial_weights():
    cfg = load_config("smoke")
    torch.manual_seed(3)
    a = build_fno(cfg, 4).state_dict()
    torch.manual_seed(3)
    b = build_fno(cfg, 4).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_max_fno_modes_is_the_architectural_limit():
    cfg = load_config("smoke")
    res, pad = 16, cfg.arch.fno.padding
    limit = max_fno_modes(res, pad)
    build_fno(cfg, limit)(torch.randn(1, 1, res, res))
    with pytest.raises(RuntimeError):
        build_fno(cfg, limit + 1)(torch.randn(1, 1, res, res))
