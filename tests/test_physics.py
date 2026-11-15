"""Physical and mathematical consistency checks."""

import numpy as np
import pytest
from scipy.linalg import eigh

from spin_eth import (
    CanonicalModel,
    compute_block_metric,
    mfim_components,
    phi_block,
    reduced_dm_block,
    single_site_eev_metric,
    single_site_tables,
    spectral_window,
    to_standard_frame,
    two_site_coupling,
)
from spin_eth.operators import SZ, site_operator

L = 8


@pytest.fixture(scope="module")
def mfim_eigenvectors():
    H_zz, H_x, H_z, H_edge = mfim_components(L)
    lo, hi = spectral_window(2**L)
    _, V = eigh(H_zz + 1.08 * H_x + 0.30 * H_z + H_edge, subset_by_index=[lo, hi])
    return V


@pytest.mark.parametrize("l", [0, 1, 3])
def test_reduced_density_matrices_are_states(mfim_eigenvectors, l):
    psi = mfim_eigenvectors[:, 10]
    rho = reduced_dm_block(psi, 2, l, L)
    assert rho.shape == (2 ** (l + 1),) * 2
    assert np.isclose(np.trace(rho).real, 1.0)
    assert np.allclose(rho, rho.conj().T)
    assert np.linalg.eigvalsh(rho).min() > -1e-12


@pytest.mark.parametrize("l", [0, 1, 2])
def test_batched_phi_matches_explicit(mfim_eigenvectors, l):
    explicit = compute_block_metric(mfim_eigenvectors, L, l)[2]
    assert np.isclose(phi_block(mfim_eigenvectors, L, l), explicit, rtol=1e-12)


def test_phi0_is_half_the_single_site_eev_metric(mfim_eigenvectors):
    sz_diag, sx_perm = single_site_tables(L)
    explicit = compute_block_metric(mfim_eigenvectors, L, 0)[2]
    assert np.isclose(explicit, single_site_eev_metric(mfim_eigenvectors, sz_diag, sx_perm) / 2, rtol=1e-12)


def test_phi_invariant_under_z_rotation():
    model = CanonicalModel(7, 2)
    x = np.array([0.08, 0.54, 0.76, -0.44, 0.36, 0.25])
    x_std, rotation = to_standard_frame(x)
    assert abs(x_std[5]) < 1e-12 and x_std[4] > 0 and rotation != 0
    assert np.isclose(model.phi(x_std), model.phi(x), rtol=1e-9)


def test_flipping_imaginary_couplings_gives_complex_conjugate():
    """XY+YX and YZ+ZY are the only couplings containing one Y, i.e. purely imaginary."""
    n = 4
    ops = {name: two_site_coupling(name, n) for name in ("ZZ", "XX", "YY", "XY+YX", "XZ+ZX", "YZ+ZY")}
    field = sum(site_operator(SZ, i, n) for i in range(n)) + 0.25 * site_operator(SZ, 0, n) - 0.25 * site_operator(SZ, n - 1, n)

    def H(cXYYX, cYZZY):
        return (0.37 * ops["ZZ"] + 0.81 * ops["XX"] - 0.52 * ops["YY"] + cXYYX * ops["XY+YX"]
                - 0.29 * ops["XZ+ZX"] + cYZZY * ops["YZ+ZY"] + field)

    assert np.allclose(H(-0.64, 0.0), H(0.64, 0.0).conj())
    assert not np.allclose(H(-0.64, 0.45), H(0.64, 0.45).conj())
    assert np.allclose(H(-0.64, -0.45), H(0.64, 0.45).conj())


def test_level_spacing_ratio_bounds():
    model = CanonicalModel(7, 2)
    r, std, n = model.level_spacing_ratio(model.mfim_hamiltonian())
    assert 0 < r < 1 and n == spectral_window(2**7)[1] - spectral_window(2**7)[0] - 1
