"""Optimizers and minimum checks used for the two-spin coupling search."""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .analysis import describe_modes, fd_hessian, grid_argmin, n_negative, scan_plane
from .constants import PARAMS
from .models import to_standard_frame

#: L-BFGS-B settings used for every local search (finite-difference gradient).
LBFGS_OPTIONS = {"ftol": 1e-15, "gtol": 1e-7, "maxiter": 800, "eps": 0.03}

#: Differential-evolution settings shared by all DE searches.
DE_OPTIONS = {
    "tol": 1e-9,
    "atol": 0,
    "mutation": (0.5, 1.0),
    "recombination": 0.7,
    "polish": True,
    "workers": 1,
}


def lbfgs(f, x0, maxiter=800):
    """L-BFGS-B with a 2-point finite-difference gradient (step 0.03)."""
    options = {**LBFGS_OPTIONS, "maxiter": maxiter}
    return minimize(f, x0, method="L-BFGS-B", jac="2-point", options=options)


class RecordingObjective:
    """Wrap an objective to record every value and the best point seen."""

    def __init__(self, f):
        self.f = f
        self.history = []
        self.best = np.inf
        self.best_x = None

    def __call__(self, x):
        val = self.f(x)
        self.history.append(val)
        if val < self.best:
            self.best, self.best_x = val, np.array(x, copy=True)
        return val

    @property
    def n_evals(self):
        return len(self.history)

    def running_best(self):
        return np.minimum.accumulate(self.history)


def minimize_with_saddle_escape(f, x0, max_passes=8, hessian_eps=0.06, kick=0.3, log=print):
    """Alternate L-BFGS-B with a Hessian check until the point is a local minimum.

    After each local optimization the full finite-difference Hessian is
    diagonalized. If it has a negative eigenvalue, the point is displaced by
    ``kick`` along the softest mode (towards lower f) and the search restarts.

    Returns a dict with keys ``x``, ``fun``, ``evals`` and ``evecs`` (of the final
    Hessian) and ``passes``.
    """
    x = np.array(x0, dtype=float)
    for p in range(max_passes):
        log(f"Pass {p + 1}: L-BFGS-B  (start Phi = {f(x):.6e})")
        res = lbfgs(f, x)
        x, fun = res.x, res.fun
        log(f"  converged: Phi = {fun:.6e}  (success={res.success})")
        H, _ = fd_hessian(f, x, eps=hessian_eps)
        evals, evecs = np.linalg.eigh(H)
        log("  Hessian eigenvalues: " + "  ".join(f"{ev:+.3e}" for ev in evals))
        neg = n_negative(evals)
        if neg == 0:
            log("  all eigenvalues positive: local minimum")
            break
        soft = np.argmin(evals)
        xp, xm = x + kick * evecs[:, soft], x - kick * evecs[:, soft]
        x = xp if f(xp) < f(xm) else xm
        log(f"  {neg} negative eigenvalue(s): kicking along softest mode (lambda={evals[soft]:+.3e})")
    return {"x": x, "fun": fun, "evals": evals, "evecs": evecs, "passes": p + 1}


def warm_start_population(x_seed, n_tight=12, n_loose=12, sd_tight=0.2, sd_loose=1.0, bounds=(-5.0, 5.0), seed=42):
    """DE initial population: the seed, n_tight tight and n_loose loose Gaussian perturbations."""
    rng = np.random.default_rng(seed)
    n = len(x_seed)
    pop = np.zeros((1 + n_tight + n_loose, n))
    pop[0] = x_seed
    for i in range(1, 1 + n_tight):
        pop[i] = np.clip(x_seed + rng.normal(0, sd_tight, n), *bounds)
    for i in range(1 + n_tight, 1 + n_tight + n_loose):
        pop[i] = np.clip(x_seed + rng.normal(0, sd_loose, n), *bounds)
    return pop


def run_de(f, bounds, **kwargs):
    """``scipy.optimize.differential_evolution`` with :data:`DE_OPTIONS` defaults."""
    return differential_evolution(f, bounds, **{**DE_OPTIONS, **kwargs})


def reduced_standard_hessian(model, x_canonical, eps=0.05, progress=False, desc="5x5 Hessian"):
    """Hessian in the standard frame with YZ+ZY pinned at 0, over [J, XX, YY, XY+YX, XZ+ZX].

    Returns a dict with the standard-frame point ``x_std`` (length 6), the
    rotation angle ``rotation``, ``phi`` at the point, and ``evals``/``evecs``.
    """
    x_std, rotation = to_standard_frame(np.asarray(x_canonical, dtype=float))
    H, f0 = fd_hessian(model.phi_fixed_last(0.0), x_std[:5], eps=eps, progress=progress, desc=desc)
    evals, evecs = np.linalg.eigh(H)
    return {"x_std": x_std, "rotation": rotation, "phi": f0, "hessian": H, "evals": evals, "evecs": evecs}


def analyze_minimum(model, x_canonical, label, phi_ref, scan_range=2.5, scan_points=9, log=print):
    """Reduced standard-frame Hessian plus a plane scan along its two softest modes."""
    names = PARAMS[:5]
    log("=" * 70)
    log(f"  {label}")
    log("=" * 70)
    t = time.time()
    res = reduced_standard_hessian(model, x_canonical, progress=True, desc=f"{label}: Hessian")
    log(f"  Standard-frame point (rotation {np.degrees(res['rotation']):.4f} deg):")
    for name, v in zip(names, res["x_std"][:5]):
        log(f"    {name:>8} = {v:.6g}")
    log(f"    YZ+ZY    = {res['x_std'][5]:.3e}  (pinned to exactly 0 for the Hessian)")
    log(f"  Phi = {res['phi']:.8e}  ({phi_ref / res['phi']:.4f}x vs PRX)   [{time.time() - t:.1f}s]")
    log("  Reduced 5x5 Hessian (YZ+ZY = 0):")
    log(describe_modes(res["evals"], res["evecs"], names))
    res["n_neg"] = n_negative(res["evals"])
    log(f"  {res['n_neg']} negative eigenvalue(s)")

    t = time.time()
    alphas = np.linspace(-scan_range, scan_range, scan_points)
    v1, v2 = res["evecs"][:, 0], res["evecs"][:, 1]
    grid = scan_plane(model.phi_fixed_last(0.0), res["x_std"][:5], v1, v2, alphas, desc=f"{label}: 2D scan")
    grid_min, a_min, b_min = grid_argmin(grid, alphas)
    log(f"  {scan_points}x{scan_points} scan along modes 1, 2 (range +/-{scan_range})  [{time.time() - t:.1f}s]")
    log(f"    grid minimum: Phi = {grid_min:.6e} at (alpha={a_min:+.3f}, beta={b_min:+.3f})")
    if grid_min < res["phi"] * (1 - 5e-4):
        log("    *** lower point found off-centre: not confirmed as the minimum in this plane ***")
    else:
        log("    confirmed: the centre is the lowest point of the scan")
    res.update(alphas=alphas, grid=grid)
    return res
