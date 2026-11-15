"""Block ETH diagnostics and two-spin coupling optimization for spin-1/2 chains."""

from .constants import COUPLINGS, EDGE_FIELDS, KH_G, KH_H, PARAMS, PRX_G, PRX_H, SPECTRAL_FRACTION
from .metrics import (
    compute_block_metric,
    level_spacing_ratio,
    phi0_from_eevs,
    phi_block,
    reduced_dm_block,
    reduced_dm_pair,
    single_site_eev_metric,
    single_site_tables,
    spectral_window,
)
from .models import J_PRX, CanonicalModel, prx_canonical_point, to_standard_frame, z_rotation_matrix
from .operators import mfim_components, two_site_coupling

__version__ = "1.0.0"

__all__ = [
    "COUPLINGS", "EDGE_FIELDS", "KH_G", "KH_H", "PARAMS", "PRX_G", "PRX_H", "SPECTRAL_FRACTION",
    "J_PRX", "CanonicalModel", "compute_block_metric", "level_spacing_ratio", "mfim_components",
    "phi0_from_eevs", "phi_block", "prx_canonical_point", "reduced_dm_block", "reduced_dm_pair",
    "single_site_eev_metric", "single_site_tables", "spectral_window", "to_standard_frame",
    "two_site_coupling", "z_rotation_matrix",
]
