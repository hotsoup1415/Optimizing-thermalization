"""Hamiltonian families and the gauge choices used to parametrize them.

Two families appear in this project:

* The mixed-field Ising model (MFIM), H(g, h) = sum ZZ + g sum X + h sum Z + edge,
  which is real-symmetric.
* The canonical two-spin model,

      H(x) = J sum ZZ + sum Z + edge + sum_c x_c O_c,
      x = [J, c_XX, c_YY, c_XY+YX, c_XZ+ZX, c_YZ+ZY],

  obtained by fixing two gauge freedoms of Phi: a uniform rotation about Y
  (which removes the X field) and the overall energy scale (Z field = 1).

Phi is also invariant under a uniform rotation about Z. That rotation mixes
(XX, YY, XY+YX) and (XZ+ZX, YZ+ZY) but leaves ZZ, Z and the edge fields
unchanged. The *standard frame* uses it to set YZ+ZY = 0 with XZ+ZX > 0.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from .constants import COUPLINGS, PRX_G, PRX_H, SPECTRAL_FRACTION
from .metrics import level_spacing_ratio, phi_block, spectral_window
from .operators import mfim_components, two_site_coupling


class CanonicalModel:
    """Dense operators and Phi(l) evaluation for the canonical two-spin model.

    Parameters
    ----------
    L : int
        Chain length.
    l : int
        Block extension for Phi(l) (block of l+1 spins).
    frac : float
        Middle fraction of the spectrum used for Phi.
    couplings : sequence of str
        Two-site couplings matching x[1:], in order.
    """

    def __init__(self, L, l, frac=SPECTRAL_FRACTION, couplings=COUPLINGS):
        self.L, self.l, self.frac = L, l, frac
        self.couplings = tuple(couplings)
        H_zz, H_x, H_z, H_edge = mfim_components(L)
        self.H_zz = H_zz.astype(np.complex128)
        self.H_x = H_x.astype(np.complex128)
        self.H_z = H_z.astype(np.complex128)
        self.H_edge = H_edge.astype(np.complex128)
        self.coupling_ops = {name: two_site_coupling(name, L) for name in self.couplings}
        self.window = spectral_window(2**L, frac)

    # -- Hamiltonians ------------------------------------------------------
    def hamiltonian(self, x):
        """H(x) = x[0] ZZ + Z + edge + sum_k x[k] O_k."""
        H = x[0] * self.H_zz + self.H_z + self.H_edge
        for c, name in zip(x[1:], self.couplings):
            H += c * self.coupling_ops[name]
        return H

    def single_coupling_hamiltonian(self, J, name, c):
        """J ZZ + Z + edge + c * O_name."""
        return J * self.H_zz + self.H_z + self.H_edge + c * self.coupling_ops[name]

    def mfim_hamiltonian(self, g=PRX_G, h=PRX_H):
        """The MFIM point H(g, h) (no two-spin couplings, unit ZZ)."""
        return self.H_zz + g * self.H_x + h * self.H_z + self.H_edge

    # -- Diagnostics -------------------------------------------------------
    def eigenvectors(self, H):
        lo, hi = self.window
        _, V = eigh(H, subset_by_index=[lo, hi])
        return V

    def phi_of(self, H):
        """Phi(l) of an arbitrary Hamiltonian on this chain."""
        return phi_block(self.eigenvectors(H), self.L, self.l)

    def phi(self, x):
        """Phi(l) at canonical parameters x."""
        return self.phi_of(self.hamiltonian(x))

    def phi_mfim(self, g=PRX_G, h=PRX_H):
        return self.phi_of(self.mfim_hamiltonian(g, h))

    def level_spacing_ratio(self, H):
        return level_spacing_ratio(H, self.frac)

    # -- Restricted objectives --------------------------------------------
    def phi_fixed_J(self, J):
        """Objective c -> Phi([J, *c]) over the couplings with J held fixed."""
        return lambda c: self.phi(np.concatenate([[J], c]))

    def phi_fixed_last(self, value):
        """Objective x -> Phi([*x, value]) with the last coupling (YZ+ZY) held fixed."""
        return lambda x: self.phi(np.append(x, value))


# ---------------------------------------------------------------------------
# Gauge transformations
# ---------------------------------------------------------------------------
def prx_canonical_point(g=PRX_G, h=PRX_H):
    """Canonical-model coordinates equivalent to the MFIM point (g, h).

    Rotating about Y by theta = arctan(g/h) points the net field along Z, and
    rescaling by h' = sqrt(g^2 + h^2) sets its strength to 1. The ZZ term then
    becomes cos^2(theta) ZZ + sin^2(theta) XX - cos(theta) sin(theta) (XZ+ZX),
    all divided by h'. The edge fields are *not* rotated, so this point differs
    slightly from the MFIM at open boundaries (a 1.4% change in Phi at L=9).
    """
    h_prime = np.sqrt(g**2 + h**2)
    theta = np.arctan2(g, h)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([ct**2 / h_prime, st**2 / h_prime, 0.0, 0.0, -ct * st / h_prime, 0.0])


#: J of the PRX point in the canonical frame (independent of L).
J_PRX = prx_canonical_point()[0]


def z_rotation_matrix(phi):
    """Action of a uniform Z-rotation by ``phi`` on c = [XX, YY, XY+YX, XZ+ZX, YZ+ZY].

    Block diagonal: a 3x3 block on (XX, YY, XY+YX) and a 2x2 rotation on
    (XZ+ZX, YZ+ZY). The 3x3 block is not orthogonal in these coordinates
    (it would be if XY+YX were scaled by 1/sqrt 2), so ``R @ v`` does not map
    Hessian eigenvectors between frames exactly.
    """
    cp, sp = np.cos(phi), np.sin(phi)
    s2, c2 = 2 * cp * sp, cp**2 - sp**2
    R = np.zeros((5, 5))
    R[0, 0] = cp**2;  R[0, 1] = sp**2;  R[0, 2] = s2
    R[1, 0] = sp**2;  R[1, 1] = cp**2;  R[1, 2] = -s2
    R[2, 0] = -s2 / 2; R[2, 1] = s2 / 2; R[2, 2] = c2
    R[3, 3] = cp;     R[3, 4] = sp
    R[4, 3] = -sp;    R[4, 4] = cp
    return R


def to_standard_frame(x):
    """Rotate canonical parameters x (length 6) to the standard frame.

    Returns ``(x_std, phi)`` with x_std[5] (YZ+ZY) = 0 and x_std[4] (XZ+ZX) > 0.
    J is unchanged. Phi(x_std) = Phi(x).
    """
    J, c5 = x[0], x[1:]
    phi = np.arctan2(c5[4], c5[3])
    c5_std = z_rotation_matrix(phi) @ c5
    if c5_std[3] < 0:
        phi += np.pi
        c5_std = z_rotation_matrix(phi) @ c5
    return np.concatenate([[J], c5_std]), phi
