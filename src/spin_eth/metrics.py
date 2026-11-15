"""Thermalization diagnostics: block reduced density matrices, Phi(l), level statistics.

The central quantity is the block ETH metric

    Phi(l) = mean_{j, n} Tr[(rho(j, l; n+1) - rho(j, l; n))^2],

where rho(j, l; n) is the reduced density matrix of the l+1 contiguous spins
j, ..., j+l in the n-th eigenstate. The mean runs over block positions j and
over consecutive pairs (n, n+1) of eigenstates in the middle of the spectrum.
For Hermitian Delta, Tr[Delta^2] is the squared Frobenius norm, so Phi >= 0.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.linalg import eigh, eigvalsh

from .constants import SPECTRAL_FRACTION


def spectral_window(dim, frac=SPECTRAL_FRACTION):
    """Inclusive index range ``(lo, hi)`` of the middle ``frac`` of a spectrum of size ``dim``."""
    lo = int(dim * (0.5 - frac / 2))
    hi = int(dim * (0.5 + frac / 2)) - 1
    return lo, hi


def mid_spectrum_eigenvectors(H, frac=SPECTRAL_FRACTION):
    """Eigenvectors (columns, ascending energy) of the middle ``frac`` of the spectrum of ``H``."""
    lo, hi = spectral_window(H.shape[0], frac)
    _, V = eigh(H, subset_by_index=[lo, hi])
    return V


# ---------------------------------------------------------------------------
# Explicit reference implementation (one eigenstate at a time)
# ---------------------------------------------------------------------------
def reduced_dm_block(psi, j, l, L):
    """Reduced density matrix of spins j, ..., j+l for the pure state ``psi``.

    Returns a ``2**(l+1)`` square matrix in the computational basis of the
    block, with site j as the most significant bit.
    """
    if j + l >= L:
        raise ValueError(f"j+l={j + l} must be < L={L}")
    block = list(range(j, j + l + 1))
    env = [k for k in range(L) if k not in block]
    psi_mat = np.transpose(psi.reshape([2] * L), block + env).reshape(2 ** (l + 1), -1)
    return psi_mat @ psi_mat.conj().T


def reduced_dm_pair(psi, j, L):
    """4x4 reduced density matrix of the adjacent sites (j, j+1)."""
    return reduced_dm_block(psi, j, 1, L)


def compute_block_metric(V, L, l):
    """Phi(l) built explicitly from every rho(j, l; n).

    Returns ``(rho_block, tr_delta_sq, Phi)`` where ``rho_block[j]`` has shape
    ``(n_eig, 2**(l+1), 2**(l+1))`` and ``tr_delta_sq[j, n]`` is
    Tr[(rho(j, l; n+1) - rho(j, l; n))^2]. Used to validate :func:`phi_block`.
    """
    n_eig = V.shape[1]
    n_valid = L - l
    rho_block = {
        j: np.stack([reduced_dm_block(V[:, n], j, l, L) for n in range(n_eig)], axis=0)
        for j in range(n_valid)
    }
    delta_rho = {j: np.diff(rho_block[j], axis=0) for j in range(n_valid)}
    tr_delta_sq = np.array(
        [np.einsum("nij,nji->n", delta_rho[j], delta_rho[j]).real for j in range(n_valid)]
    )
    return rho_block, tr_delta_sq, float(np.mean(tr_delta_sq))


# ---------------------------------------------------------------------------
# Batched implementation (all eigenstates at once) used for scans and searches
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _block_axis_orders(L, l, positions):
    """Axis orders bringing block (j..j+l) to the front of a ([2]*L + [n_eig]) tensor."""
    orders = []
    for j in positions:
        block = list(range(j, j + l + 1))
        env = [k for k in range(L) if k not in block]
        orders.append(tuple(block + env + [L]))  # eigenstate axis stays last
    return tuple(orders)


def phi_block(V, L, l, positions=None):
    """Block ETH metric Phi(l) for the eigenvectors in the columns of ``V``.

    Parameters
    ----------
    V : (2**L, n_eig) array
        Eigenvectors ordered by energy (real or complex).
    L : int
        Chain length.
    l : int
        Block extension; the block has l+1 spins.
    positions : iterable of int, optional
        Left edges j of the blocks to average over (default: all L-l).
    """
    positions = tuple(range(L - l)) if positions is None else tuple(positions)
    n_eig = V.shape[1]
    dim_block = 2 ** (l + 1)
    psi = V.reshape([2] * L + [n_eig])
    total = 0.0
    for order in _block_axis_orders(L, l, positions):
        psi_mat = np.transpose(psi, order).reshape(dim_block, -1, n_eig)
        rho = np.einsum("ien,jen->nij", psi_mat.conj(), psi_mat)
        delta = np.diff(rho, axis=0)
        total += np.einsum("nij,nji->n", delta, delta).real.sum()
    return total / (len(positions) * (n_eig - 1))


# ---------------------------------------------------------------------------
# l = 0 via single-site expectation values (real eigenvectors only)
# ---------------------------------------------------------------------------
def single_site_tables(L):
    """Index tables for fast single-site expectation values.

    Returns ``(sz_diag, sx_perm)``: ``sz_diag[:, s]`` is the diagonal of Z_s and
    ``sx_perm[:, s]`` is the basis permutation implemented by X_s.
    """
    idx = np.arange(2**L, dtype=np.int64)
    sz_diag = np.column_stack(
        [(1 - 2 * ((idx >> (L - 1 - s)) & 1)).astype(np.float64) for s in range(L)]
    )
    sx_perm = np.column_stack([(idx ^ (1 << (L - 1 - s))).astype(np.int64) for s in range(L)])
    return sz_diag, sx_perm


def single_site_eev_metric(V, sz_diag, sx_perm):
    """mean_{n, s} [(d<Z_s>)^2 + (d<X_s>)^2] over consecutive eigenstates (real V).

    This is the all-site squared EEV metric of the Huse-Khemani project. For
    a real Hamiltonian the single-site reduced density matrix is fixed by
    <Z_s> and <X_s>, so Phi(0) equals exactly half of this quantity.
    """
    L = sz_diag.shape[1]
    z_vals = (V**2).T @ sz_diag
    x_vals = np.column_stack([(V * V[sx_perm[:, s]]).sum(axis=0) for s in range(L)])
    dz = np.diff(z_vals, axis=0)
    dx = np.diff(x_vals, axis=0)
    return float(np.mean(dz**2 + dx**2))


def phi0_from_eevs(V, sz_diag, sx_perm):
    """Phi(0) for real eigenvectors, via :func:`single_site_eev_metric`."""
    return single_site_eev_metric(V, sz_diag, sx_perm) / 2


# ---------------------------------------------------------------------------
# Level statistics
# ---------------------------------------------------------------------------
def level_spacing_ratio(H, frac=SPECTRAL_FRACTION):
    """Mean, std and count of r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}).

    s_n are consecutive level spacings inside the middle ``frac`` of the
    spectrum. Ratios need no unfolding.
    """
    evals = eigvalsh(H)
    lo, hi = spectral_window(len(evals), frac)
    gaps = np.diff(evals[lo : hi + 1])
    r_n = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])
    return r_n.mean(), r_n.std(), len(r_n)
