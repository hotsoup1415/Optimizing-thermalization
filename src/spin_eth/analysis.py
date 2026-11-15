"""Local analysis of minima: finite-difference Hessians, mode summaries, plane scans."""

from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm

from .constants import NEG_EIG_TOL


def fd_hessian(f, x0, eps, progress=False, desc="Hessian"):
    """Central-difference Hessian of ``f`` at ``x0`` with step ``eps``.

    Diagonal:     (f(x+e_i) - 2 f(x) + f(x-e_i)) / eps^2
    Off-diagonal: (f(x+e_i+e_j) - f(x+e_i-e_j) - f(x-e_i+e_j) + f(x-e_i-e_j)) / (4 eps^2)

    Costs 1 + 2n + 2n(n-1) evaluations. Returns ``(H, f(x0))``.
    """
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    H = np.zeros((n, n))
    f0 = f(x0)
    steps = [(i,) for i in range(n)] + [(i, j) for i in range(n) for j in range(i + 1, n)]
    for step in tqdm(steps, desc=desc, disable=not progress, leave=False):
        if len(step) == 1:
            (i,) = step
            ei = np.zeros(n); ei[i] = eps
            H[i, i] = (f(x0 + ei) - 2 * f0 + f(x0 - ei)) / eps**2
        else:
            i, j = step
            ei = np.zeros(n); ei[i] = eps
            ej = np.zeros(n); ej[j] = eps
            v = (f(x0 + ei + ej) - f(x0 + ei - ej) - f(x0 - ei + ej) + f(x0 - ei - ej)) / (4 * eps**2)
            H[i, j] = v
            H[j, i] = v
    return H, f0


def n_negative(evals, tol=NEG_EIG_TOL):
    """Number of eigenvalues below -tol."""
    return int(np.sum(np.asarray(evals) < -tol))


def curvature_label(ev, tol=NEG_EIG_TOL):
    return "pos" if ev > tol else ("NEG" if ev < -tol else "zero")


def describe_modes(evals, evecs, names, n_dominant=2, tol=NEG_EIG_TOL, indent="    "):
    """One line per Hessian mode: eigenvalue, curvature sign, largest components."""
    lines = []
    for k, ev in enumerate(evals):
        vec = evecs[:, k]
        order = np.argsort(np.abs(vec))[::-1][:n_dominant]
        dom = ", ".join(f"{names[i]}({vec[i]:+.2f})" for i in order)
        lines.append(f"{indent}mode {k + 1}: {ev:>+10.4e}  [{curvature_label(ev, tol)}]  {dom}")
    return "\n".join(lines)


def scan_plane(f, center, v1, v2, alphas, betas=None, progress=True, desc="2D scan"):
    """f(center + a v1 + b v2) on the grid a in alphas (rows) x b in betas (columns)."""
    betas = alphas if betas is None else betas
    grid = np.zeros((len(alphas), len(betas)))
    for ia, a in enumerate(tqdm(alphas, desc=desc, disable=not progress, leave=False)):
        for ib, b in enumerate(betas):
            grid[ia, ib] = f(center + a * v1 + b * v2)
    return grid


def grid_argmin(grid, alphas, betas=None):
    """``(min value, alpha, beta)`` of a plane scan."""
    betas = alphas if betas is None else betas
    ia, ib = np.unravel_index(np.argmin(grid), grid.shape)
    return grid[ia, ib], alphas[ia], betas[ib]
