"""The package reproduces the original notebook implementations.

Most comparisons are exact: the same floating-point operations are performed
in the same order, so the results are bit-for-bit identical.
"""

import numpy as np
import pytest

import reference_impl as ref
from spin_eth import CanonicalModel, mfim_components, prx_canonical_point, to_standard_frame, two_site_coupling
from spin_eth.analysis import fd_hessian
from spin_eth.models import z_rotation_matrix
from spin_eth.operators import TWO_SITE_TERMS
from spin_eth.scans import PhiSpec, scan_mfim_grid
from spin_eth.search import warm_start_population

L_TEST = 7
RNG = np.random.default_rng(1234)
X_RANDOM = [RNG.normal(0, 0.8, 6) for _ in range(3)]


def test_mfim_components_identical():
    new = mfim_components(L_TEST)
    old = ref.build_hamiltonian_components(L_TEST)[:4]
    for a, b in zip(new, old):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("name", list(TWO_SITE_TERMS))
def test_two_site_couplings_identical(name):
    assert np.array_equal(two_site_coupling(name, L_TEST), ref.make_build_coupling(L_TEST)(name))


@pytest.mark.parametrize("l", [2, 3])
def test_canonical_phi_identical(l):
    model = CanonicalModel(L_TEST, l)
    phi_old = ref.make_phi_nc(L_TEST, l)
    for x in X_RANDOM:
        assert model.phi(x) == phi_old(x)


def test_phi_fixed_J_and_pinned_objectives():
    model = CanonicalModel(L_TEST, 2)
    x = X_RANDOM[0]
    assert model.phi_fixed_J(x[0])(x[1:]) == model.phi(x)
    assert model.phi_fixed_last(x[5])(x[:5]) == model.phi(x)


def test_mfim_grid_scans_match_original():
    g = np.array([1.0, 1.3, 1.6])
    h = np.array([0.2, 0.5])
    specs = [PhiSpec(0), PhiSpec(1), PhiSpec(2)]
    maps = scan_mfim_grid(L_TEST, g, h, specs, progress=False)
    np.testing.assert_array_equal(maps[specs[0].key], ref.scan_phi0_grid(L_TEST, g, h))
    for spec in specs[1:]:
        np.testing.assert_allclose(maps[spec.key], ref.scan_phi_l_grid(spec.l, L_TEST, g, h), rtol=1e-13)


def test_mfim_grid_parallel_matches_serial():
    g = np.array([1.0, 1.3])
    h = np.array([0.2, 0.5])
    specs = [PhiSpec(1)]
    serial = scan_mfim_grid(6, g, h, specs, progress=False)
    parallel = scan_mfim_grid(6, g, h, specs, n_workers=2, progress=False)
    np.testing.assert_allclose(parallel[specs[0].key], serial[specs[0].key], rtol=1e-12)


def test_gauge_helpers_identical():
    J, cXX, cXZ = ref.prx_canonical()
    x = prx_canonical_point()
    assert (x[0], x[1], x[4]) == (J, cXX, cXZ)
    for phi in (0.3, -1.2, 2.9):
        assert np.array_equal(z_rotation_matrix(phi), ref._R5(phi))
    for x6 in X_RANDOM:
        new, rot_new = to_standard_frame(x6)
        old, rot_old = ref._to_standard(x6)
        assert np.array_equal(new, old) and rot_new == rot_old


def test_fd_hessian_identical():
    f = ref.make_phi_nc(6, 2)
    x0 = X_RANDOM[1]
    H_new, f_new = fd_hessian(f, x0, eps=0.06)
    H_old, f_old = ref._hessian6(f, x0, eps=0.06)
    assert np.array_equal(H_new, H_old) and f_new == f_old


def test_warm_start_population_identical():
    seed = np.array([1.3102469777746573, 0.1563727956639216, -0.009625111783115314,
                     0.2659923597166658, -0.7049065840130392])
    assert np.array_equal(warm_start_population(seed), ref.warm_start_population(seed))
