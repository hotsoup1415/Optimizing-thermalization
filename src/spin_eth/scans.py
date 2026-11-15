"""(g, h) landscape scans of Phi for the mixed-field Ising model.

Each grid point needs one dense diagonalization. :func:`scan_mfim_grid`
diagonalizes once per point and evaluates every requested metric on the same
eigenvectors, so different block sizes, block positions and spectral windows
can share the cost. Points are distributed over worker processes, since LAPACK
gains little from multithreading at these sizes.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from tqdm.auto import tqdm

from .constants import SPECTRAL_FRACTION
from .metrics import phi0_from_eevs, phi_block, single_site_tables, spectral_window
from .operators import mfim_components


@dataclass(frozen=True)
class PhiSpec:
    """One metric to evaluate at every grid point.

    ``l`` is the block extension; ``positions`` restricts the block left edges
    (``None`` = all); ``frac`` is the spectral window. For l = 0 over all
    positions, Phi(0) is computed from single-site expectation values.
    """

    l: int
    positions: tuple | None = None
    frac: float = SPECTRAL_FRACTION

    @property
    def key(self):
        pos = "all" if self.positions is None else "-".join(map(str, self.positions))
        return f"l{self.l}_pos{pos}_frac{self.frac:g}"


def mfim_phi_at(L, g, h, specs, components=None, tables=None):
    """Evaluate each PhiSpec for H(g, h) on L sites from one diagonalization.

    The eigenvectors are computed for the widest requested window; narrower
    windows use the corresponding central columns.
    """
    H_zz, H_x, H_z, H_edge = components if components is not None else mfim_components(L)
    dim = 2**L
    lo, hi = spectral_window(dim, max(s.frac for s in specs))
    H = H_zz + g * H_x + h * H_z + H_edge
    _, V_wide = eigh(H, subset_by_index=[lo, hi])

    out = {}
    for spec in specs:
        lo_s, hi_s = spectral_window(dim, spec.frac)
        V = V_wide[:, lo_s - lo : hi_s - lo + 1]
        if spec.l == 0 and spec.positions is None:
            sz_diag, sx_perm = tables if tables is not None else single_site_tables(L)
            out[spec.key] = phi0_from_eevs(V, sz_diag, sx_perm)
        else:
            out[spec.key] = phi_block(V, L, spec.l, spec.positions)
    return out


# Per-process state for the worker pool (built once per worker).
_WORKER = {}
_BLAS_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _init_worker(L):
    _WORKER["L"] = L
    _WORKER["components"] = mfim_components(L)
    _WORKER["tables"] = single_site_tables(L)


def _worker_point(args):
    ih, ig, g, h, specs = args
    vals = mfim_phi_at(_WORKER["L"], g, h, specs, _WORKER["components"], _WORKER["tables"])
    return ih, ig, vals


def scan_mfim_grid(L, g_values, h_values, specs, n_workers=1, progress=True):
    """Phi landscapes over the (g, h) grid for every spec.

    Returns a dict mapping ``spec.key`` to an array of shape
    ``(len(h_values), len(g_values))`` (rows = h, columns = g).
    """
    specs = tuple(specs)
    maps = {s.key: np.zeros((len(h_values), len(g_values))) for s in specs}
    tasks = [(ih, ig, g, h, specs) for ih, h in enumerate(h_values) for ig, g in enumerate(g_values)]
    bar = tqdm(total=len(tasks), desc=f"L={L} grid", disable=not progress)

    def store(result):
        ih, ig, vals = result
        for key, v in vals.items():
            maps[key][ih, ig] = v
        bar.update(1)

    if n_workers <= 1:
        _init_worker(L)
        for t in tasks:
            store(_worker_point(t))
    else:
        # Workers are spawned fresh and read these at import: one BLAS thread each.
        saved = {k: os.environ.get(k) for k in _BLAS_THREAD_VARS}
        os.environ.update({k: "1" for k in _BLAS_THREAD_VARS})
        try:
            with ProcessPoolExecutor(n_workers, initializer=_init_worker, initargs=(L,)) as pool:
                for result in pool.map(_worker_point, tasks, chunksize=1):
                    store(result)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    bar.close()
    return maps


def grid_minimum(data, g_values, h_values):
    """``(min value, g, h)`` of a landscape with rows = h and columns = g."""
    ih, ig = np.unravel_index(np.nanargmin(data), data.shape)
    return data[ih, ig], g_values[ig], h_values[ih]
