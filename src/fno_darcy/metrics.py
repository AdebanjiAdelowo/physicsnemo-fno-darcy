"""Error metrics, evaluated in physical units."""

import torch


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample relative L2 error ``||prediction - target||_2 / ||target||_2``.

    Parameters
    ----------
    prediction, target : torch.Tensor
        Tensors of shape ``[batch, ...]``.

    Returns
    -------
    torch.Tensor
        Shape ``[batch]``.
    """
    diff = (prediction - target).flatten(1)
    return torch.linalg.vector_norm(diff, dim=1) / torch.linalg.vector_norm(target.flatten(1), dim=1)


def summarise(values: torch.Tensor) -> dict:
    """Mean, sample standard deviation, median and maximum of a 1D tensor."""
    values = values.double()
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=True).item() if values.numel() > 1 else 0.0,
        "median": values.median().item(),
        "max": values.max().item(),
    }
