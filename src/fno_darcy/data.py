# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ``IndependentDarcy2D.initialize_batch`` is adapted from ``Darcy2D.initialize_batch``
# in physicsnemo/datapipes/benchmarks/darcy.py of NVIDIA PhysicsNeMo, commit
# b45a5c810c741e6b41f8515be24c51121f8fc21f. Modified by Adebanji Adelowo (2026):
# the Fourier coefficients are drawn with NumPy instead of the Warp kernel
# ``init_uniform_random_4d``. Everything else in this file is new.

"""Darcy-flow data from the PhysicsNeMo ``Darcy2D`` generator.

``Darcy2D`` draws a piecewise-constant permeability field and solves for the pressure
with a multigrid Jacobi method written in NVIDIA Warp. This module wraps it: it seeds it,
collects finite splits, caches them, and measures the residual of the generated fields.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import physicsnemo
import torch
from omegaconf import DictConfig
import warp as wp
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.datapipes.benchmarks.kernels.utils import fourier_to_array_batched_2d, threshold_3d


def warp_device(device: torch.device | str) -> str:
    """Warp device for a torch device. Warp has no MPS backend, so MPS generates on CPU."""
    return "cuda" if torch.device(device).type == "cuda" else "cpu"


def make_normaliser(cfg: DictConfig) -> dict:
    """``{"permeability": (mean, std), "darcy": (mean, std)}`` as used upstream."""
    norm = cfg.normaliser
    return {
        "permeability": (norm.permeability.mean, norm.permeability.std_dev),
        "darcy": (norm.darcy.mean, norm.darcy.std_dev),
    }


def normalise(x: torch.Tensor, stats: tuple) -> torch.Tensor:
    return (x - stats[0]) / stats[1]


def denormalise(x: torch.Tensor, stats: tuple) -> torch.Tensor:
    return x * stats[1] + stats[0]


def darcy_residual(u: np.ndarray, k: np.ndarray, dx: float, source: float = 1.0, form: str = "kernel") -> np.ndarray:
    r"""Finite-difference residual of a generated pressure field, in units of the forcing.

    Both forms use :math:`u = 0` on the nodes outside the array and :math:`k` extended by
    its edge value, as the ``Darcy2D`` kernel does.

    ``form="kernel"`` is the equation the ``Darcy2D`` Jacobi kernel iterates,

    .. math:: k\,\Delta_h u + \frac{\delta_x k\,\delta_x u + \delta_y k\,\delta_y u}{2\,\Delta x} + f,

    with :math:`\delta` the difference of the two neighbours. It measures how far the
    iteration is from its own fixed point. The cross term of a central-difference
    discretisation of :math:`\nabla k\cdot\nabla u` would be divided by
    :math:`4\,\Delta x^2`; the kernel divides by :math:`2\,\Delta x`.

    ``form="flux"`` is the conservative discretisation of
    :math:`\nabla\cdot(k\nabla u) + f` with harmonic face permeabilities. It measures
    consistency with the Darcy equation in divergence form.

    Parameters
    ----------
    u, k : np.ndarray
        Pressure and permeability, shape ``[batch, nx, ny]``, on the full solver grid.
    dx : float
        Grid spacing.
    source : float
        Constant forcing :math:`f`.
    form : {"kernel", "flux"}
        Discretisation, see above.
    """
    u = np.asarray(u, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    up = np.pad(u, ((0, 0), (1, 1), (1, 1)))
    kp = np.pad(k, ((0, 0), (1, 1), (1, 1)), mode="edge")
    u_w, u_e = up[:, :-2, 1:-1], up[:, 2:, 1:-1]
    u_s, u_n = up[:, 1:-1, :-2], up[:, 1:-1, 2:]
    k_w, k_e = kp[:, :-2, 1:-1], kp[:, 2:, 1:-1]
    k_s, k_n = kp[:, 1:-1, :-2], kp[:, 1:-1, 2:]
    if form == "kernel":
        laplacian = (u_w + u_e + u_s + u_n - 4.0 * u) / dx**2
        cross = ((k_e - k_w) * (u_e - u_w) + (k_n - k_s) * (u_n - u_s)) / (2.0 * dx)
        return k * laplacian + cross + source
    if form == "flux":
        flux = sum(_harmonic(k, kn) * (un - u) for kn, un in ((k_w, u_w), (k_e, u_e), (k_s, u_s), (k_n, u_n)))
        return flux / dx**2 + source
    raise ValueError(f"unknown form {form!r}")


def _harmonic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 2.0 * a * b / (a + b)


class IndependentDarcy2D(Darcy2D):
    """``Darcy2D`` with independently drawn Fourier coefficients.

    ``Darcy2D`` fills its coefficient array ``[4, batch, freq, freq]`` with the kernel
    ``init_uniform_random_4d``, which seeds the generator of every entry with
    ``wp.rand_init(seed, wp.tid())``. In that four-dimensional launch only the first
    index reaches the seed, so the array holds four distinct numbers: all samples of a
    batch are the same field, and all frequencies of a sample share one amplitude
    (``scripts/verify_generator.py`` measures this).

    This subclass draws the coefficients from NumPy, uniformly on ``[-1, 1]`` as the
    kernel is called to do, and leaves the synthesis, the thresholding and the solver to
    the upstream kernels.
    """

    def initialize_batch(self) -> None:
        self.permeability.zero_()
        coefficients = np.random.uniform(-1.0, 1.0, size=self.fourier_dim).astype(np.float32)
        wp.copy(self.rand_fourier, wp.array(coefficients, dtype=float, device=self.device))
        wp.launch(
            kernel=fourier_to_array_batched_2d,
            dim=self.dim,
            inputs=[
                self.permeability,
                self.rand_fourier,
                self.nr_permeability_freq,
                self.resolution,
                self.resolution,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=threshold_3d,
            dim=self.dim,
            inputs=[
                self.permeability,
                0.0,
                self.min_permeability,
                self.max_permeability,
            ],
            device=self.device,
        )
        self.darcy0.zero_()
        self.darcy1.zero_()


def make_datapipe(cfg: DictConfig, batch_size: int, device, normaliser: dict | None) -> Darcy2D:
    """``Darcy2D`` (or ``IndependentDarcy2D``) with the solver settings of ``cfg.data``."""
    cls = IndependentDarcy2D if cfg.data.independent_samples else Darcy2D
    return cls(
        resolution=cfg.training.resolution,
        batch_size=batch_size,
        convergence_threshold=cfg.data.convergence_threshold,
        max_iterations=cfg.data.max_iterations,
        normaliser=normaliser,
        device=warp_device(device),
    )


def generate_split(cfg: DictConfig, n_samples: int, seed: int, device) -> dict:
    """Generate ``n_samples`` (permeability, pressure) pairs in physical units.

    Both samplers draw from NumPy's global generator (``Darcy2D`` draws its Warp seed
    from it), so seeding NumPy here makes the split reproducible.
    """
    batch_size = min(cfg.data.generation_batch_size, n_samples)
    pipe = make_datapipe(cfg, batch_size, device, normaliser=None)
    np.random.seed(seed)
    ks, us, count = [], [], 0
    sq_sum = {"kernel": 0.0, "flux": 0.0}
    start = time.perf_counter()
    iterator = iter(pipe)
    while sum(len(k) for k in ks) < n_samples:
        batch = next(iterator)
        ks.append(batch["permeability"][:, 0].cpu().numpy().copy())
        us.append(batch["darcy"][:, 0].cpu().numpy().copy())
        # residual on the full solver grid, before Darcy2D crops the last row and column
        full_u, full_k = pipe.darcy0.numpy(), pipe.permeability.numpy()
        for form in sq_sum:
            sq_sum[form] += float((darcy_residual(full_u, full_k, pipe.dx, form=form) ** 2).sum())
        count += full_u.size
    seconds = time.perf_counter() - start
    k = np.concatenate(ks)[:n_samples].astype(np.float32)
    u = np.concatenate(us)[:n_samples].astype(np.float32)
    info = {
        "n_samples": int(n_samples),
        "resolution": int(cfg.training.resolution),
        "seed": int(seed),
        "independent_samples": bool(cfg.data.independent_samples),
        "distinct_permeability_fields": len({f.tobytes() for f in k}),
        "convergence_threshold": float(cfg.data.convergence_threshold),
        "max_iterations": int(cfg.data.max_iterations),
        "generation_batch_size": int(batch_size),
        "generation_device": warp_device(device),
        "physicsnemo_version": physicsnemo.__version__,
        "generation_seconds": seconds,
        "generation_seconds_per_sample": seconds / (len(ks) * batch_size),
        "kernel_residual_rms": float(np.sqrt(sq_sum["kernel"] / count)),
        "flux_form_residual_rms": float(np.sqrt(sq_sum["flux"] / count)),
        "permeability_mean": float(k.mean()),
        "permeability_std": float(k.std()),
        "pressure_mean": float(u.mean()),
        "pressure_std": float(u.std()),
    }
    if not (np.isfinite(k).all() and np.isfinite(u).all()):
        raise RuntimeError(
            "Darcy2D returned non-finite fields. PhysicsNeMo 2.2.2 does this for resolutions "
            "above 32; install the commit pinned in requirements.txt."
        )
    return {"permeability": k, "darcy": u, "info": info}


_KEY_FIELDS = (
    "n_samples",
    "resolution",
    "seed",
    "independent_samples",
    "convergence_threshold",
    "max_iterations",
    "generation_batch_size",
    "generation_device",
    "physicsnemo_version",
)


def load_or_generate(cfg: DictConfig, split: str, n_samples: int, device, root: Path) -> dict:
    """Load a cached split, or generate and cache it.

    The cache key covers everything that changes the generated fields, including the
    device, because CPU and CUDA solves differ in floating-point round-off.
    """
    seed = int(cfg.data.seeds[split])
    batch_size = min(cfg.data.generation_batch_size, n_samples)
    key = {
        "n_samples": int(n_samples),
        "resolution": int(cfg.training.resolution),
        "seed": seed,
        "independent_samples": bool(cfg.data.independent_samples),
        "convergence_threshold": float(cfg.data.convergence_threshold),
        "max_iterations": int(cfg.data.max_iterations),
        "generation_batch_size": int(batch_size),
        "generation_device": warp_device(device),
        "physicsnemo_version": physicsnemo.__version__,
    }
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    path = Path(root) / cfg.data.dir / f"{split}_r{key['resolution']}_n{n_samples}_{digest}.npz"
    if path.exists():
        with np.load(path) as f:
            out = {"permeability": f["permeability"], "darcy": f["darcy"], "info": json.loads(str(f["info"]))}
        out["info"]["loaded_from_cache"] = True
    else:
        out = generate_split(cfg, n_samples, seed, device)
        assert all(out["info"][f] == key[f] for f in _KEY_FIELDS)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, permeability=out["permeability"], darcy=out["darcy"], info=json.dumps(out["info"]))
        out["info"]["loaded_from_cache"] = False
    out["info"]["split"] = split
    out["info"]["file"] = path.name
    return out


def to_normalised_tensors(split: dict, normaliser: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU tensors ``[n, 1, res, res]`` in the normalised units the model is trained in."""
    k = torch.from_numpy(split["permeability"]).unsqueeze(1)
    u = torch.from_numpy(split["darcy"]).unsqueeze(1)
    return normalise(k, normaliser["permeability"]), normalise(u, normaliser["darcy"])


def fixed_batches(k: torch.Tensor, u: torch.Tensor, batch_size: int, generator: torch.Generator, device):
    """One shuffled pass over a finite set. The last incomplete batch is kept."""
    order = torch.randperm(len(k), generator=generator)
    for start in range(0, len(k), batch_size):
        idx = order[start : start + batch_size]
        yield k[idx].to(device), u[idx].to(device)


def stream_batches(pipe: Darcy2D, n_batches: int, device):
    """``n_batches`` new samples from ``Darcy2D`` (the upstream training protocol)."""
    for _, batch in zip(range(n_batches), pipe):
        yield batch["permeability"].to(device), batch["darcy"].to(device)
