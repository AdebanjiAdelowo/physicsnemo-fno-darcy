"""Audit of the datasets and of the generator, independent of any trained model.

    python scripts/audit_data.py

Checks (all on CPU, deterministic):

1. Regeneration: the cached test split of ``configs/local.yaml`` is regenerated from its seed
   and compared bitwise with the cached file.
2. Leakage: Hamming distances between the binary permeability fields of the training,
   validation and test splits (exact and near duplicates), and the data statistics against
   the fixed normalisation constants of the upstream configuration.
3. The sample family of ``Darcy2D`` as shipped: how many numbers determine a field, and how
   close fresh validation draws come to the training draws of the upstream protocol,
   compared with ``IndependentDarcy2D``.
4. Discretisation: an assembly of the sparse operators independent of
   ``fno_darcy.reference`` (Kronecker products), applied to a smooth permeability under grid
   refinement, shows which continuous equation the generator stencil is consistent with.

Writes ``results/audit/data_audit.json``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import warp as wp

from _bootstrap import ROOT
from fno_darcy import data as D
from fno_darcy.config import load_config
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.datapipes.benchmarks.kernels.initialization import init_uniform_random_4d
from physicsnemo.datapipes.benchmarks.kernels.utils import fourier_to_array_batched_2d


def regeneration(cfg) -> dict:
    cached = D.load_or_generate(cfg, "test", cfg.data.test_samples, torch.device("cpu"), ROOT)
    fresh = D.generate_split(cfg, cfg.data.test_samples, int(cfg.data.seeds.test), torch.device("cpu"))
    return {
        "file": cached["info"]["file"],
        "permeability_bitwise_equal": bool(np.array_equal(cached["permeability"], fresh["permeability"])),
        "pressure_bitwise_equal": bool(np.array_equal(cached["darcy"], fresh["darcy"])),
        "pressure_max_abs_difference": float(np.abs(cached["darcy"] - fresh["darcy"]).max()),
    }


def signs(k: np.ndarray) -> np.ndarray:
    """Permeability field as +-1 vectors (2 -> +1, 0.5 -> -1)."""
    return np.where(k.reshape(len(k), -1) > 1.0, 1.0, -1.0)


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fraction of differing pixels between every row of ``a`` and every row of ``b``."""
    return (a.shape[1] - a @ b.T) / (2.0 * a.shape[1])


def leakage(cfg) -> dict:
    splits = {
        name: D.load_or_generate(cfg, name, n, torch.device("cpu"), ROOT)
        for name, n in (("train", cfg.training.pseudo_epoch_sample_size), ("validation", cfg.validation.sample_size), ("test", cfg.data.test_samples))
    }
    s = {name: signs(v["permeability"]) for name, v in splits.items()}
    out = {"pairs": {}}
    for a, b in (("test", "train"), ("validation", "train"), ("test", "validation")):
        h = hamming(s[a], s[b])
        nearest = h.min(1)
        out["pairs"][f"{a}_to_{b}"] = {
            "exact_duplicates": int((h == 0).sum()),
            "min_nearest_hamming": float(nearest.min()),
            "median_nearest_hamming": float(np.median(nearest)),
            "pressure_rel_l2_of_closest_pair": float(_closest_pressure(splits[a]["darcy"], splits[b]["darcy"], h)),
        }
    h = hamming(s["train"], s["train"])
    np.fill_diagonal(h, 1.0)
    out["train_within"] = {"exact_duplicates": int((h == 0).sum() // 2), "min_nearest_hamming": float(h.min())}
    norm = cfg.normaliser
    out["normalisation"] = {
        name: {
            "permeability_mean": float(v["permeability"].mean()),
            "permeability_std": float(v["permeability"].std()),
            "pressure_mean": float(v["darcy"].mean()),
            "pressure_std": float(v["darcy"].std()),
        }
        for name, v in splits.items()
    }
    out["normalisation"]["config_constants"] = {
        "permeability_mean": norm.permeability.mean,
        "permeability_std": norm.permeability.std_dev,
        "pressure_mean": norm.darcy.mean,
        "pressure_std": norm.darcy.std_dev,
    }
    return out


def _closest_pressure(ua: np.ndarray, ub: np.ndarray, h: np.ndarray) -> float:
    i, j = np.unravel_index(np.argmin(h), h.shape)
    return np.linalg.norm(ua[i] - ub[j]) / np.linalg.norm(ua[i])


def shipped_coefficients(seed: int) -> np.ndarray:
    """The Fourier coefficients ``init_uniform_random_4d`` writes for one batch seed."""
    arr = wp.zeros((4, 2, 5, 5), dtype=float, device="cpu")
    wp.launch(init_uniform_random_4d, dim=arr.shape, inputs=[arr, -1.0, 1.0, np.uint64(seed)], device="cpu")
    return arr.numpy()


def synthesise(coefficients: np.ndarray, resolution: int) -> np.ndarray:
    """NumPy copy of ``fourier_to_array_batched_2d`` and ``threshold_3d``: ``[4, b, f, f] -> [b, n, n]``."""
    n = resolution + 1
    r = 6.28318 / resolution * np.arange(n)
    f = coefficients.shape[-1]
    s = np.sin(np.outer(np.arange(f), r))  # [freq, n]
    c = np.cos(np.outer(np.arange(f), r))
    field = (
        np.einsum("bij,ix,jy->bxy", coefficients[0], s, s)
        + np.einsum("bij,ix,jy->bxy", coefficients[1], c, s)
        + np.einsum("bij,ix,jy->bxy", coefficients[2], s, c)
        + np.einsum("bij,ix,jy->bxy", coefficients[3], c, c)
    ) / f**2
    return np.where(field < 0.0, 0.5, 2.0)


def shipped_family(resolution: int = 64, n_train_batches: int = 8192, n_val: int = 256) -> dict:
    """Sample structure of ``Darcy2D`` as shipped, and nearest-neighbour distances of the upstream protocol."""
    rng = np.random.default_rng(0)
    out = {}
    # the NumPy synthesis reproduces the Warp kernels
    np.random.seed(0)
    pipe = Darcy2D(resolution=resolution, batch_size=4, device="cpu")
    pipe.initialize_batch()
    warp_k = pipe.permeability.numpy()
    out["numpy_synthesis_matches_warp_fraction"] = float((synthesise(pipe.rand_fourier.numpy(), resolution) == warp_k).mean())

    coeffs = shipped_coefficients(12345)
    per_component_distinct = [len(np.unique(coeffs[c])) for c in range(4)]
    out["shipped_distinct_values_per_component"] = per_component_distinct
    out["shipped_value_depends_only_on_leading_index"] = bool(all(n == 1 for n in per_component_distinct))
    # threshold at zero: positive rescaling of the four amplitudes leaves the field unchanged
    a = coeffs[:, :1]
    out["shipped_field_invariant_to_positive_scaling"] = bool(np.array_equal(synthesise(a, resolution), synthesise(0.37 * a, resolution)))
    out["shipped_effective_parameters"] = "a direction in R^4 (3 parameters): k = threshold(sum_c a_c S_c) with fixed S_c"

    # upstream protocol: 256 pseudo-epochs x 32 batches of 64 copies -> 8192 distinct training fields
    def shipped(n):
        seeds = rng.integers(0, 2**63, size=n, dtype=np.int64)
        return synthesise(np.concatenate([shipped_coefficients(int(sd))[:, :1] for sd in seeds], axis=1), resolution)[:, :resolution, :resolution]

    def independent(n):
        return synthesise(rng.uniform(-1, 1, size=(4, n, 5, 5)).astype(np.float32), resolution)[:, :resolution, :resolution]

    for name, draw in (("Darcy2D_shipped", shipped), ("IndependentDarcy2D", independent)):
        train, val = signs(draw(n_train_batches)), signs(draw(n_val))
        nearest = np.concatenate([hamming(val[i : i + 64], train).min(1) for i in range(0, n_val, 64)])
        out[name] = {
            "training_fields": n_train_batches,
            "validation_fields": n_val,
            "median_nearest_hamming": float(np.median(nearest)),
            "fraction_within_1_percent": float((nearest < 0.01).mean()),
            "fraction_within_5_percent": float((nearest < 0.05).mean()),
        }
    out["note"] = (
        "Darcy2D_shipped: one distinct field per batch, so the upstream run of 256 pseudo-epochs sees "
        f"{n_train_batches} distinct fields (each 64 times). IndependentDarcy2D drawn with the same count for comparison."
    )
    return out


def operators(n: int, k: np.ndarray, h: float) -> dict:
    """Kernel, conservative and k*Laplacian operators by Kronecker products (independent assembly).

    Index ``[x, y]`` maps to ``x * n + y``. Zero Dirichlet nodes outside the array, permeability
    extended by its edge value, as in ``Darcy2D``.
    """
    eye = sp.identity(n, format="csr")
    shift = sp.diags([np.ones(n - 1)], [1], shape=(n, n), format="csr")  # (S u)[i] = u[i + 1]
    E, W = sp.kron(shift, eye), sp.kron(shift.T, eye)
    N, S = sp.kron(eye, shift), sp.kron(eye, shift.T)
    kp = np.pad(k, 1, mode="edge")
    kv = k.ravel()
    kE, kW = kp[2:, 1:-1].ravel(), kp[:-2, 1:-1].ravel()
    kN, kS = kp[1:-1, 2:].ravel(), kp[1:-1, :-2].ravel()
    lap = (E + W + N + S - 4 * sp.identity(n * n)) / h**2
    K = sp.diags(kv)
    # Darcy2D kernel: k * lap(u) + (dk_x du_x + dk_y du_y) / (2h), with d the two-neighbour difference
    cross = (sp.diags(kE - kW) @ (E - W) + sp.diags(kN - kS) @ (N - S)) / (2 * h)
    # consistent central difference of grad k . grad u would divide by 4 h^2
    cross_consistent = cross * (2 * h) / (4 * h**2)
    hm = lambda a, b: 2 * a * b / (a + b)  # noqa: E731
    flux = (
        sp.diags(hm(kv, kE)) @ (E - sp.identity(n * n))
        + sp.diags(hm(kv, kW)) @ (W - sp.identity(n * n))
        + sp.diags(hm(kv, kN)) @ (N - sp.identity(n * n))
        + sp.diags(hm(kv, kS)) @ (S - sp.identity(n * n))
    ) / h**2
    return {"kernel": K @ lap + cross, "nonconservative_consistent": K @ lap + cross_consistent, "flux": flux, "laplace": K @ lap}


def discretisation_refinement(resolutions=(32, 64, 128, 256)) -> list[dict]:
    """Smooth k = exp(sin(2 pi x) cos(2 pi y)); differences between the discrete solutions under refinement."""
    rows = []
    for res in resolutions:
        n, h = res + 1, 1.0 / (res + 1)
        x = (np.arange(n) + 1) * h
        X, Y = np.meshgrid(x, x, indexing="ij")
        k = np.exp(np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y))
        sol = {name: spla.spsolve(op.tocsc(), -np.ones(n * n)) for name, op in operators(n, k, h).items()}
        rel = lambda a, b: float(np.linalg.norm(sol[a] - sol[b]) / np.linalg.norm(sol[b]))  # noqa: E731
        rows.append(
            {
                "resolution": res,
                "kernel_vs_laplace": rel("kernel", "laplace"),
                "kernel_vs_flux": rel("kernel", "flux"),
                "nonconservative_consistent_vs_flux": rel("nonconservative_consistent", "flux"),
                "laplace_vs_flux": rel("laplace", "flux"),
            }
        )
        print(rows[-1])
    return rows


def generator_vs_independent_assembly(resolution: int = 64, n: int = 4) -> dict:
    """``IndependentDarcy2D`` output against a direct solve of the independently assembled kernel stencil."""
    np.random.seed(0)
    pipe = D.IndependentDarcy2D(resolution=resolution, batch_size=n, device="cpu", convergence_threshold=1e-7)
    next(iter(pipe))
    u, k = pipe.darcy0.numpy().astype(np.float64), pipe.permeability.numpy().astype(np.float64)
    m = resolution + 1
    out = {}
    for form in ("kernel", "flux", "laplace"):
        errs = []
        for b in range(n):
            ref = spla.spsolve(operators(m, k[b], pipe.dx)[form].tocsc(), -np.ones(m * m)).reshape(m, m)
            errs.append(np.linalg.norm(u[b] - ref) / np.linalg.norm(ref))
        out[f"rel_l2_vs_{form}"] = float(np.mean(errs))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "audit" / "data_audit.json")
    args = parser.parse_args()
    cfg = load_config("local")
    report = {}
    report["regeneration"] = regeneration(cfg)
    print("regeneration", report["regeneration"])
    report["leakage"] = leakage(cfg)
    print("leakage", json.dumps(report["leakage"], indent=1))
    report["shipped_family"] = shipped_family()
    print("shipped family", json.dumps(report["shipped_family"], indent=1))
    report["generator_vs_independent_assembly_64"] = generator_vs_independent_assembly()
    print("generator vs independent assembly", report["generator_vs_independent_assembly_64"])
    report["smooth_k_refinement"] = discretisation_refinement()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
