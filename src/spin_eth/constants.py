"""Physical reference points and numerical conventions shared across the project."""

# Tilted-field Ising reference point used as the baseline throughout ("PRX").
PRX_G, PRX_H = 1.08, 0.30

# Kim & Huse, PRL 111, 127205 (2013): g = (5 + sqrt 5)/8, h = (1 + sqrt 5)/4.
KH_G, KH_H = 0.9045, 0.8090

# (left, right) boundary Z fields. They break the reflection symmetry of the open chain.
EDGE_FIELDS = (0.25, -0.25)

# Fraction of the spectrum (centred on the middle) whose eigenstates enter Phi.
SPECTRAL_FRACTION = 0.5

# Two-spin couplings of the canonical model. ZZ is not listed: it is degenerate with J.
COUPLINGS = ("XX", "YY", "XY+YX", "XZ+ZX", "YZ+ZY")

# Parameter order of the canonical model, x = [J, c_XX, c_YY, c_XY+YX, c_XZ+ZX, c_YZ+ZY].
PARAMS = ("J",) + COUPLINGS

# Hessian eigenvalues below -NEG_EIG_TOL are counted as negative curvature.
NEG_EIG_TOL = 1e-4

# Mean level-spacing ratio <r> for Poisson, GOE and GUE statistics.
R_POISSON, R_GOE, R_GUE = 0.3863, 0.5307, 0.5996
