import pytest
import torch
from fno_darcy.metrics import relative_l2, summarise


def test_relative_l2_known_values():
    target = torch.ones(3, 1, 4, 4)
    pred = torch.stack([torch.ones(1, 4, 4), 1.1 * torch.ones(1, 4, 4), torch.zeros(1, 4, 4)])
    assert relative_l2(pred, target).tolist() == pytest.approx([0.0, 0.1, 1.0], abs=1e-6)


def test_relative_l2_is_per_sample_and_scale_invariant():
    torch.manual_seed(0)
    target, pred = torch.randn(5, 1, 8, 8), torch.randn(5, 1, 8, 8)
    err = relative_l2(pred, target)
    assert err.shape == (5,)
    assert torch.allclose(err, relative_l2(3.0 * pred, 3.0 * target), atol=1e-6)
    assert torch.allclose(err[2], relative_l2(pred[2:3], target[2:3])[0])


def test_summarise():
    s = summarise(torch.tensor([1.0, 2.0, 3.0, 6.0]))
    assert s["mean"] == pytest.approx(3.0)
    assert s["max"] == pytest.approx(6.0)
    assert s["std"] == pytest.approx(torch.tensor([1.0, 2.0, 3.0, 6.0]).std().item())
    assert summarise(torch.tensor([4.0]))["std"] == 0.0
