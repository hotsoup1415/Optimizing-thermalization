"""Original notebook implementations, kept verbatim (modulo global state).

These are the functions from ``Adjacent_Spin_Thermalization.ipynb`` and
``Spin_Phi_Optimization.ipynb`` that the package replaces. The originals read
module-level globals (``_L``, ``_perms``, ...); here those globals are closed
over by ``make_*`` factories. They exist only so the tests can check that the
refactored code reproduces the original numbers.
"""

import numpy as np
from scipy.linalg import eigh

sx = np.array([[0, 1], [1, 0]], dtype=np.float64)
sz = np.array([[1, 0], [0, -1]], dtype=np.float64)
id2 = np.eye(2, dtype=np.float64)
_sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)


def kron_n(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def build_local_operators(L):
    sx_list, sz_list = [], []
    for i in range(L):
        ops_x = [id2] * L; ops_x[i] = sx; sx_list.append(kron_n(ops_x))
        ops_z = [id2] * L; ops_z[i] = sz; sz_list.append(kron_n(ops_z))
    return sx_list, sz_list


def build_hamiltonian_components(L, use_edge_fields=True, left_edge=0.25, right_edge=-0.25):
    sx_list, sz_list = build_local_operators(L)
    dim = 2 ** L
    H_zz = np.zeros((dim, dim), dtype=np.float64)
    H_x = np.zeros((dim, dim), dtype=np.float64)
    H_z = np.zeros((dim, dim), dtype=np.float64)
    H_edge = np.zeros((dim, dim), dtype=np.float64)
    for i in range(L - 1):
        ops = [id2] * L; ops[i] = sz; ops[i + 1] = sz
        H_zz += kron_n(ops)
    for i in range(L):
        H_x += sx_list[i]
        H_z += sz_list[i]
    if use_edge_fields:
        H_edge += left_edge * sz_list[0]
        H_edge += right_edge * sz_list[-1]
    return H_zz, H_x, H_z, H_edge, sx_list, sz_list


def make_build_coupling(L):
    dim = 2 ** L

    def _build_coupling(name):
        sx_c = sx.astype(np.complex128)
        sz_c = sz.astype(np.complex128)
        id_c = id2.astype(np.complex128)
        H = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(L - 1):
            def _op(a, b, _i=i):
                ops = [id_c] * L; ops[_i] = a; ops[_i + 1] = b
                return kron_n(ops)
            if name == 'XX': H += _op(sx_c, sx_c)
            elif name == 'YY': H += _op(_sy, _sy)
            elif name == 'ZZ': H += _op(sz_c, sz_c)
            elif name == 'XY+YX': H += _op(sx_c, _sy) + _op(_sy, sx_c)
            elif name == 'XZ+ZX': H += _op(sx_c, sz_c) + _op(sz_c, sx_c)
            elif name == 'YZ+ZY': H += _op(_sy, sz_c) + _op(sz_c, _sy)
        return H

    return _build_coupling


def make_phi_nc(L, l_block):
    """The original ``_phi_nc`` / ``_phi_l3`` pair (child notebook, canonical model)."""
    dim = 2 ** L
    _lo, _hi = int(dim * 0.25), int(dim * 0.75) - 1
    _dim_block = 2 ** (l_block + 1)
    _n_valid = L - l_block
    _perms = []
    for _j in range(_n_valid):
        _keep = list(range(_j, _j + l_block + 1))
        _env = [k for k in range(L) if k not in _keep]
        _perms.append(_keep + _env + [L])

    def _phi_l3(V_loc):
        n_eig = V_loc.shape[1]
        psi_t = V_loc.reshape([2] * L + [n_eig])
        total = 0.0
        for _j in range(_n_valid):
            psi_p = np.transpose(psi_t, _perms[_j])
            psi_mat = psi_p.reshape(_dim_block, -1, n_eig)
            rho_all = np.einsum('ien,jen->nij', psi_mat.conj(), psi_mat)
            delta = np.diff(rho_all, axis=0)
            total += np.einsum('nij,nji->n', delta, delta).real.sum()
        return total / (_n_valid * (n_eig - 1))

    H_zz, H_x, H_z, H_edge, _, _ = build_hamiltonian_components(L, use_edge_fields=True)
    _Hzz_nc, _Hz_nc, _Hedge_nc = (A.astype(np.complex128) for A in (H_zz, H_z, H_edge))
    _couplings_nc = ['XX', 'YY', 'XY+YX', 'XZ+ZX', 'YZ+ZY']
    bc = make_build_coupling(L)
    _H_coup_dict = {cp: bc(cp) for cp in _couplings_nc}

    def _phi_nc(x):
        H = x[0] * _Hzz_nc + _Hz_nc + _Hedge_nc
        for ci, cp in zip(x[1:], _couplings_nc):
            H += ci * _H_coup_dict[cp]
        _, V = eigh(H, subset_by_index=[_lo, _hi])
        return _phi_l3(V)

    return _phi_nc


def scan_phi0_grid(L, g_values, h_values, use_edge_fields=True, frac=0.5):
    H_zz, H_x, H_z, H_edge, _, _ = build_hamiltonian_components(L, use_edge_fields=use_edge_fields)
    dim = 2 ** L
    idx = np.arange(dim, dtype=np.int64)
    sz_diag_all = np.column_stack([(1 - 2 * ((idx >> (L - 1 - s)) & 1)).astype(np.float64) for s in range(L)])
    sx_perm_all = np.column_stack([(idx ^ (1 << (L - 1 - s))).astype(np.int64) for s in range(L)])
    lo = int(dim * (0.5 - frac / 2))
    hi = int(dim * (0.5 + frac / 2)) - 1
    phi_map = np.zeros((len(h_values), len(g_values)))
    for ih, h in enumerate(h_values):
        for ig, g in enumerate(g_values):
            H = H_zz + g * H_x + h * H_z + H_edge
            _, V_mid_loc = eigh(H, subset_by_index=[lo, hi])
            z_vals = (V_mid_loc ** 2).T @ sz_diag_all
            x_vals = np.column_stack([(V_mid_loc * V_mid_loc[sx_perm_all[:, s]]).sum(axis=0) for s in range(L)])
            dz = np.diff(z_vals, axis=0)
            dx = np.diff(x_vals, axis=0)
            phi_map[ih, ig] = float(np.mean(dz ** 2 + dx ** 2)) / 2
    return phi_map


def scan_phi_l_grid(l, L, g_values, h_values, use_edge_fields=True, frac=0.5):
    """General batched path of the original ``scan_phi_l_grid`` (also equals ``scan_phi1_grid`` for l=1)."""
    H_zz, H_x, H_z, H_edge, _, _ = build_hamiltonian_components(L, use_edge_fields=use_edge_fields)
    dim = 2 ** L
    dim_block = 2 ** (l + 1)
    n_valid = L - l
    lo = int(dim * (0.5 - frac / 2))
    hi = int(dim * (0.5 + frac / 2)) - 1
    perms = []
    for j in range(n_valid):
        keep = list(range(j, j + l + 1))
        env_axes = [k for k in range(L) if k not in keep]
        perms.append(keep + env_axes + [L])
    phi_map = np.zeros((len(h_values), len(g_values)))
    for ih, h in enumerate(h_values):
        for ig, g in enumerate(g_values):
            H = H_zz + g * H_x + h * H_z + H_edge
            _, V_mid_loc = eigh(H, subset_by_index=[lo, hi])
            n_eig = V_mid_loc.shape[1]
            psi_tensor = V_mid_loc.reshape([2] * L + [n_eig])
            total = 0.0
            for j in range(n_valid):
                psi_p = np.transpose(psi_tensor, perms[j])
                psi_mat = psi_p.reshape(dim_block, -1, n_eig)
                rho_all_j = np.einsum('ien,jen->nij', psi_mat, psi_mat)
                delta = np.diff(rho_all_j, axis=0)
                tr_dsq = np.einsum('nij,nji->n', delta, delta).real
                total += tr_dsq.sum()
            phi_map[ih, ig] = total / (n_valid * (n_eig - 1))
    return phi_map


def _R5(phi):
    cp, sp = np.cos(phi), np.sin(phi)
    s2, c2 = 2*cp*sp, cp**2 - sp**2
    R = np.zeros((5, 5))
    R[0, 0] = cp**2;  R[0, 1] = sp**2;  R[0, 2] =  s2
    R[1, 0] = sp**2;  R[1, 1] = cp**2;  R[1, 2] = -s2
    R[2, 0] = -s2/2;  R[2, 1] =  s2/2;  R[2, 2] =  c2
    R[3, 3] =  cp;    R[3, 4] = sp
    R[4, 3] = -sp;    R[4, 4] = cp
    return R


def _to_standard(x6):
    J = x6[0]
    c5 = x6[1:]
    phi_rot = np.arctan2(c5[4], c5[3])
    c5_std = _R5(phi_rot) @ c5
    if c5_std[3] < 0:
        phi_rot += np.pi
        c5_std = _R5(phi_rot) @ c5
    return np.concatenate([[J], c5_std]), phi_rot


def _hessian6(fn, x0, eps=0.06):
    n = len(x0)
    H = np.zeros((n, n))
    f0 = fn(x0)
    steps = [(i,) for i in range(n)] + [(i, j) for i in range(n) for j in range(i + 1, n)]
    for step in steps:
        if len(step) == 1:
            i = step[0]
            ei = np.zeros(n); ei[i] = eps
            H[i, i] = (fn(x0 + ei) - 2 * f0 + fn(x0 - ei)) / eps**2
        else:
            i, j = step
            ei = np.zeros(n); ei[i] = eps
            ej = np.zeros(n); ej[j] = eps
            v = (fn(x0 + ei + ej) - fn(x0 + ei - ej)
                 - fn(x0 - ei + ej) + fn(x0 - ei - ej)) / (4 * eps**2)
            H[i, j] = v; H[j, i] = v
    return H, f0


def prx_canonical(g=1.08, h=0.30):
    _h_prime = np.sqrt(g**2 + h**2)
    _theta = np.arctan2(g, h)
    _ct, _st = np.cos(_theta), np.sin(_theta)
    return _ct**2 / _h_prime, _st**2 / _h_prime, -_ct * _st / _h_prime


def warm_start_population(_x_de_l9):
    _rng = np.random.default_rng(42)
    _pop = np.zeros((25, 5))
    _pop[0] = _x_de_l9
    for i in range(1, 13):
        _pop[i] = np.clip(_x_de_l9 + _rng.normal(0, 0.2, 5), -5.0, 5.0)
    for i in range(13, 25):
        _pop[i] = np.clip(_x_de_l9 + _rng.normal(0, 1.0, 5), -5.0, 5.0)
    return _pop
