import numpy as np
import pytest
import torch
from fno_darcy import data as D
from fno_darcy.config import load_config
from fno_darcy.reference import direct_solve


@pytest.fixture(scope="module")
def cfg():
    return load_config("smoke")


@pytest.fixture(scope="module")
def split(cfg):
    return D.generate_split(cfg, 6, seed=11, device="cpu")


def test_split_shapes_and_values(cfg, split):
    res = cfg.training.resolution
    assert split["permeability"].shape == split["darcy"].shape == (6, res, res)
    assert split["permeability"].dtype == np.float32
    assert set(np.unique(split["permeability"])) <= {0.5, 2.0}
    assert np.isfinite(split["darcy"]).all() and (split["darcy"] > 0).all()


def test_samples_are_distinct(split):
    assert split["info"]["distinct_permeability_fields"] == 6
    assert len({u.tobytes() for u in split["darcy"]}) == 6


def test_generation_is_reproducible_and_seed_dependent(cfg, split):
    again = D.generate_split(cfg, 6, seed=11, device="cpu")
    other = D.generate_split(cfg, 6, seed=12, device="cpu")
    assert np.array_equal(again["permeability"], split["permeability"])
    assert np.array_equal(again["darcy"], split["darcy"])
    assert not np.array_equal(other["permeability"], split["permeability"])


def test_generated_fields_satisfy_the_solver_stencil(split):
    """The kernel residual is small against the forcing f = 1; the flux-form residual is not."""
    assert split["info"]["kernel_residual_rms"] < 5e-3
    assert split["info"]["flux_form_residual_rms"] > 0.5


def test_cache_round_trip(cfg, tmp_path):
    first = D.load_or_generate(cfg, "test", 5, "cpu", tmp_path)
    second = D.load_or_generate(cfg, "test", 5, "cpu", tmp_path)
    assert first["info"]["loaded_from_cache"] is False and second["info"]["loaded_from_cache"] is True
    assert np.array_equal(first["darcy"], second["darcy"])
    assert len(list((tmp_path / cfg.data.dir).glob("*.npz"))) == 1
    # a different solver tolerance is a different dataset
    tighter = load_config("smoke", ["data.convergence_threshold=1e-7"])
    D.load_or_generate(tighter, "test", 5, "cpu", tmp_path)
    assert len(list((tmp_path / cfg.data.dir).glob("*.npz"))) == 2


def test_cuda_cache_is_not_regenerated_on_another_device(cfg, tmp_path):
    """A study generated on CUDA is evaluated elsewhere from its own files, never from regenerated ones."""
    if torch.cuda.is_available():
        pytest.skip("needs a machine without CUDA")
    cuda_cfg = load_config("smoke", ["data.generation_device=cuda"])
    with pytest.raises(FileNotFoundError, match="generation_device=cuda"):
        D.load_or_generate(cuda_cfg, "test", 5, "cpu", tmp_path)


def test_splits_use_different_seeds(cfg):
    seeds = cfg.data.seeds
    assert len({seeds.train, seeds.validation, seeds.test}) == 3


def test_normalisation_round_trip(cfg, split):
    norm = D.make_normaliser(cfg)
    k, u = D.to_normalised_tensors(split, norm)
    assert k.shape == u.shape == (6, 1, cfg.training.resolution, cfg.training.resolution)
    assert torch.allclose(D.denormalise(u, norm["darcy"])[:, 0], torch.from_numpy(split["darcy"]), atol=1e-7)


def test_fixed_batches_cover_every_sample_once():
    k = torch.arange(10.0).reshape(10, 1, 1, 1)
    gen = torch.Generator().manual_seed(0)
    batches = list(D.fixed_batches(k, k, 4, gen, "cpu"))
    assert [len(b[0]) for b in batches] == [4, 4, 2]
    assert sorted(torch.cat([b[0] for b in batches]).flatten().tolist()) == list(range(10))
    assert all(torch.equal(a, b) for a, b in batches)


def test_stream_batches_are_normalised_and_shaped(cfg):
    np.random.seed(0)
    pipe = D.make_datapipe(cfg, 4, "cpu", D.make_normaliser(cfg))
    batches = [(k.clone(), u.clone()) for k, u in D.stream_batches(pipe, 2, "cpu")]
    assert len(batches) == 2
    res = cfg.training.resolution
    assert batches[0][0].shape == batches[0][1].shape == (4, 1, res, res)
    assert not torch.equal(batches[0][0], batches[1][0])
    assert abs(batches[0][1].mean().item()) < 3.0


# --- residual functions, checked against manufactured solutions and direct solves ---


def _grid(n):
    dx = 1.0 / (n + 1)
    x = dx * np.arange(1, n + 1)
    return dx, *np.meshgrid(x, x, indexing="ij")


@pytest.mark.parametrize("form", ["kernel", "flux"])
def test_residual_is_second_order_for_constant_permeability(form):
    """u = sin(pi x) sin(pi y), k = 1.5: k * laplace(u) = -3 pi^2 u, both forms agree."""
    errors = []
    for n in (31, 63):
        dx, x, y = _grid(n)
        u = np.sin(np.pi * x) * np.sin(np.pi * y)
        k = np.full_like(u, 1.5)
        res = D.darcy_residual(u[None], k[None], dx, source=0.0, form=form)[0]
        errors.append(np.abs(res + 3.0 * np.pi**2 * u).max())
    assert 1.9 < np.log2(errors[0] / errors[1]) < 2.1


def test_flux_residual_is_second_order_for_smooth_variable_permeability():
    """k = 1 + x/2, u = sin(pi x) sin(pi y): compare with the analytic div(k grad u)."""
    errors = []
    for n in (31, 63):
        dx, x, y = _grid(n)
        u = np.sin(np.pi * x) * np.sin(np.pi * y)
        k = 1.0 + 0.5 * x
        exact = 0.5 * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) - 2.0 * np.pi**2 * k * u
        res = D.darcy_residual(u[None], k[None], dx, source=0.0, form="flux")[0]
        errors.append(np.abs(res - exact)[1:-1, 1:-1].max())  # the edge-extended k is not smooth at the border
    assert 1.8 < np.log2(errors[0] / errors[1]) < 2.2


@pytest.mark.parametrize("form", ["kernel", "flux"])
def test_direct_solve_has_zero_residual_in_its_own_form(form):
    rng = np.random.default_rng(0)
    n = 24
    k = np.where(rng.random((n, n)) > 0.5, 2.0, 0.5)
    dx = 1.0 / (n + 1)
    u = direct_solve(k, dx, form)
    assert np.abs(D.darcy_residual(u[None], k[None], dx, form=form)).max() < 1e-9


def test_forms_coincide_for_constant_permeability():
    n = 24
    k = np.full((n, n), 2.0)
    sols = [direct_solve(k, 1.0 / (n + 1), form) for form in ("kernel", "flux", "laplace")]
    assert np.allclose(sols[0], sols[1], atol=1e-12) and np.allclose(sols[0], sols[2], atol=1e-12)


def test_generator_matches_its_stencil_and_not_the_flux_form(cfg):
    """Regression record of the reference data: see README, 'Dataset'."""
    np.random.seed(5)
    pipe = D.IndependentDarcy2D(resolution=32, batch_size=2, device="cpu", convergence_threshold=1e-8)
    next(iter(pipe))
    u, k = pipe.darcy0.numpy().astype(np.float64), pipe.permeability.numpy().astype(np.float64)
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
    assert rel(u[0], direct_solve(k[0], pipe.dx, "kernel")) < 1e-3
    assert rel(u[0], direct_solve(k[0], pipe.dx, "flux")) > 0.05
