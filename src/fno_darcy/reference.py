"""Independent sparse direct solves used to check what ``Darcy2D`` computes.

Not used for training. The three discrete problems share the grid of ``Darcy2D``:
``u = 0`` on the nodes outside the array, permeability extended by its edge value.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

FORMS = ("kernel", "flux", "laplace")


def direct_solve(k: np.ndarray, dx: float, form: str, source: float = 1.0) -> np.ndarray:
    r"""Solve one of three five-point discretisations for a single permeability field.

    Parameters
    ----------
    k : np.ndarray
        Permeability on the full solver grid, shape ``[n, n]``.
    dx : float
        Grid spacing.
    form : {"kernel", "flux", "laplace"}
        ``kernel``: the stencil iterated by the ``Darcy2D`` Jacobi kernel,
        :math:`k\Delta_h u + (\delta_x k\,\delta_x u + \delta_y k\,\delta_y u)/(2\Delta x) + f = 0`.
        ``flux``: conservative :math:`-\nabla\cdot(k\nabla u) = f` with harmonic face permeability.
        ``laplace``: :math:`k\,\Delta_h u + f = 0`.
    source : float
        Constant forcing :math:`f`.
    """
    k = np.asarray(k, dtype=np.float64)
    n = k.shape[0]
    idx = np.arange(n * n).reshape(n, n)
    kp = np.pad(k, 1, mode="edge")
    nb = {"W": kp[:-2, 1:-1], "E": kp[2:, 1:-1], "S": kp[1:-1, :-2], "N": kp[1:-1, 2:]}
    if form == "flux":
        w = {d: 2.0 * k * nb[d] / (k + nb[d]) for d in nb}
        diag = sum(w.values())
    elif form == "kernel":
        gx = (nb["E"] - nb["W"]) * dx / 2.0
        gy = (nb["N"] - nb["S"]) * dx / 2.0
        w = {"W": k - gx, "E": k + gx, "S": k - gy, "N": k + gy}
        diag = 4.0 * k
    elif form == "laplace":
        w = {d: k for d in nb}
        diag = 4.0 * k
    else:
        raise ValueError(f"unknown form {form!r}")
    rows, cols, vals = [idx.ravel()], [idx.ravel()], [diag.ravel()]
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    for d, (di, dj) in {"W": (-1, 0), "E": (1, 0), "S": (0, -1), "N": (0, 1)}.items():
        ii, jj = i + di, j + dj
        inside = (ii >= 0) & (ii < n) & (jj >= 0) & (jj < n)
        rows.append(idx[i[inside], j[inside]])
        cols.append(idx[ii[inside], jj[inside]])
        vals.append(-w[d][inside])
    matrix = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n))
    return spla.spsolve(matrix, np.full(n * n, source * dx**2)).reshape(n, n)
