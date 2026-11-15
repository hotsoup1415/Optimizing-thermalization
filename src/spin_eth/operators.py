"""Dense spin-1/2 chain operators.

Site 0 is the leftmost tensor factor (the most significant bit of a
computational-basis index), so ``kron_n([A_0, ..., A_{L-1}])`` acts with
``A_i`` on site ``i``.
"""

from __future__ import annotations

import numpy as np

from .constants import EDGE_FIELDS

SX = np.array([[0, 1], [1, 0]], dtype=np.float64)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.float64)
ID2 = np.eye(2, dtype=np.float64)
PAULI = {"I": ID2, "X": SX, "Y": SY, "Z": SZ}

#: Each two-site coupling is a sum over nearest-neighbour bonds (i, i+1) of
#: the listed Pauli products, e.g. "XY+YX" = sum_i (X_i Y_{i+1} + Y_i X_{i+1}).
TWO_SITE_TERMS = {
    "XX": (("X", "X"),),
    "YY": (("Y", "Y"),),
    "ZZ": (("Z", "Z"),),
    "XY+YX": (("X", "Y"), ("Y", "X")),
    "XZ+ZX": (("X", "Z"), ("Z", "X")),
    "YZ+ZY": (("Y", "Z"), ("Z", "Y")),
}


def kron_n(ops):
    """Kronecker product of a sequence of matrices, left to right."""
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def site_operator(op, i, L):
    """``op`` acting on site ``i`` of an L-site chain."""
    ops = [ID2] * L
    ops[i] = op
    return kron_n(ops)


def bond_operator(a, b, i, L):
    """``a`` on site ``i`` times ``b`` on site ``i + 1``."""
    ops = [ID2] * L
    ops[i], ops[i + 1] = a, b
    return kron_n(ops)


def mfim_components(L, edge_fields=EDGE_FIELDS):
    """Terms of the mixed-field Ising chain on L sites (open boundaries).

    Returns ``(H_zz, H_x, H_z, H_edge)`` such that

        H(g, h) = H_zz + g * H_x + h * H_z + H_edge,

    with ``H_zz = sum_i Z_i Z_{i+1}``, ``H_x = sum_i X_i``, ``H_z = sum_i Z_i``
    and ``H_edge = left * Z_0 + right * Z_{L-1}``. Pass ``edge_fields=None``
    to make ``H_edge`` zero. All matrices are real (float64).
    """
    dim = 2**L
    H_zz = np.zeros((dim, dim), dtype=np.float64)
    H_x = np.zeros((dim, dim), dtype=np.float64)
    H_z = np.zeros((dim, dim), dtype=np.float64)
    H_edge = np.zeros((dim, dim), dtype=np.float64)

    for i in range(L - 1):
        H_zz += bond_operator(SZ, SZ, i, L)
    for i in range(L):
        H_x += site_operator(SX, i, L)
        H_z += site_operator(SZ, i, L)
    if edge_fields is not None:
        left, right = edge_fields
        H_edge += left * site_operator(SZ, 0, L)
        H_edge += right * site_operator(SZ, L - 1, L)
    return H_zz, H_x, H_z, H_edge


def two_site_coupling(name, L):
    """Nearest-neighbour coupling ``name`` (a key of TWO_SITE_TERMS) as a complex matrix."""
    try:
        terms = TWO_SITE_TERMS[name]
    except KeyError:
        raise ValueError(f"unknown coupling {name!r}; expected one of {list(TWO_SITE_TERMS)}") from None
    dim = 2**L
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(L - 1):
        for a, b in terms:
            H += bond_operator(PAULI[a], PAULI[b], i, L)
    return H
