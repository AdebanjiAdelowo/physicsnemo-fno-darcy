"""CPU regression checks for the full-resolution CUDA analysis."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fno_darcy.reference import direct_solve
from verify_cuda_batch64 import compare_sample, final_updates, summarize


def test_stopping_criterion_uses_each_samples_max_and_strict_threshold():
    current = np.zeros((3, 2, 2), dtype=np.float32)
    current[:, 1, 1] = [0.0625, 0.125, 0.25]
    updates, met = final_updates(current, np.zeros_like(current), 0.125)
    assert updates.tolist() == [0.0625, 0.125, 0.25]
    assert met.tolist() == [True, False, False]


def test_summary_percentiles_and_population_standard_deviation():
    result = summarize([0, 1, 2, 3, 4])
    assert result == pytest.approx(dict(mean=2, median=2, std=np.sqrt(2), p95=3.8, p99=3.96, max=4))


def test_boundary_error_and_independent_reference_forms():
    k = np.ones((9, 9))
    k[4:] = 2.0
    dx = 0.1
    u = direct_solve(k, dx, "kernel")
    edge = np.ones_like(k, dtype=bool)
    edge[1:-1, 1:-1] = False
    result = compare_sample(k, u, dx, edge)
    assert result["stencil_rel_l2"] == result["boundary_stencil_rel_l2"] == 0.0
    assert result["conservative_rel_l2"] > 0.1
    perturbed = u.copy()
    perturbed[edge] *= 1.1
    assert compare_sample(k, perturbed, dx, edge)["boundary_stencil_rel_l2"] == pytest.approx(0.1)
