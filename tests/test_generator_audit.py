"""Findings of docs/GENERATOR_VALIDATION.md, kept as tests so that a change upstream is noticed."""

import numpy as np
import pytest
import warp as wp
from fno_darcy.reference import direct_solve
from physicsnemo.datapipes.benchmarks.kernels.initialization import init_uniform_random_4d


def test_shipped_coefficients_depend_only_on_the_leading_launch_index():
    """``wp.rand_init(seed, wp.tid())`` in a 4D launch seeds with the leading index only."""
    arr = wp.zeros((4, 3, 5, 5), dtype=float, device="cpu")
    wp.launch(init_uniform_random_4d, dim=arr.shape, inputs=[arr, -1.0, 1.0, np.uint64(12345)], device="cpu")
    a = arr.numpy()
    assert all(len(np.unique(a[c])) == 1 for c in range(4))
    assert len(np.unique(a[:, 0, 0, 0])) == 4


def _rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.fixture(scope="module")
def smooth_solves():
    out = {}
    for res in (32, 64, 128):
        n, h = res + 1, 1.0 / (res + 1)
        x = (np.arange(n) + 1) * h
        X, Y = np.meshgrid(x, x, indexing="ij")
        k = np.exp(np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y))
        out[res] = {form: direct_solve(k, h, form) for form in ("kernel", "flux", "laplace")}
    return out


def test_generator_stencil_converges_to_k_laplace_at_first_order(smooth_solves):
    d = [_rel(smooth_solves[r]["kernel"], smooth_solves[r]["laplace"]) for r in (32, 64, 128)]
    assert 1.7 < d[0] / d[1] < 2.3 and 1.7 < d[1] / d[2] < 2.3


def test_generator_stencil_does_not_converge_to_the_conservative_form(smooth_solves):
    d = [_rel(smooth_solves[r]["kernel"], smooth_solves[r]["flux"]) for r in (32, 64, 128)]
    assert min(d) > 0.1 and d[2] >= d[0]
