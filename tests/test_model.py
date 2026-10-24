import pytest
import torch
from fno_darcy.config import load_config
from fno_darcy.model import build_fno, count_parameters, max_fno_modes
from physicsnemo.nn.module.spectral_layers import SpectralConv2d


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
        build_fno(cfg, limit + 2)(torch.randn(1, 1, res, res))


def _spectral_gain(n: int, modes: int) -> torch.Tensor:
    """Gain per retained Fourier coefficient of a spectral layer whose two weight blocks are the identity."""
    layer = SpectralConv2d(1, 1, modes, modes)
    with torch.no_grad():
        for w in (layer.weights1, layer.weights2):
            w.zero_()
            w[..., 0] = 1.0
    x = torch.randn(1, 1, n, n, generator=torch.Generator().manual_seed(0))
    # columns >= 1 only: column 0 is folded by the Hermitian symmetry of irfft2
    return (torch.fft.rfft2(layer(x)) / torch.fft.rfft2(x))[0, 0, :, 1:modes].real


@pytest.mark.parametrize("res", [16, 64, 256])
def test_max_fno_modes_counts_every_mode_once(res):
    """Regression: the former limit ``n // 2 + 1`` passed the middle row through both weight blocks."""
    n = res + 9
    limit = max_fno_modes(res, 9)
    assert torch.allclose(_spectral_gain(n, limit).abs().max(), torch.tensor(1.0), atol=1e-4)
    assert _spectral_gain(n, limit + 1).max() > 1.9
