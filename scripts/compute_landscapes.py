"""Phi(l) landscapes of the mixed-field Ising model over the (g, h) grid.

For each chain length, every grid point is diagonalized once and all metrics
below are evaluated on the same eigenvectors:

* Phi(0), Phi(1), Phi(2), Phi(5)    all block positions, middle 50% of the spectrum
* L=12 only: Phi(5) for the two half-chain blocks (j=0 and j=6), and Phi(5) in
  the middle 20%, 10% and 5% of the spectrum

Output: results/landscapes/mfim_L{L}.npz with arrays ``g``, ``h`` and one
landscape (rows = h, columns = g) per metric key, e.g. ``l2_posall_frac0.5``.
L=12 needs about 1.5 GB of memory per worker.

    python scripts/compute_landscapes.py --L 9 10 11 12 --workers 5
"""

import argparse
import time
from pathlib import Path

import numpy as np

from spin_eth.io import save_npz
from spin_eth.scans import PhiSpec, scan_mfim_grid

ROOT = Path(__file__).resolve().parents[1]

G_VALUES = np.linspace(1.0, 2.0, 21)
H_VALUES = np.linspace(0.2, 0.8, 13)


def specs_for(L):
    specs = [PhiSpec(0), PhiSpec(1), PhiSpec(2), PhiSpec(5)]
    if L == 12:
        specs += [PhiSpec(5, positions=(0,)), PhiSpec(5, positions=(6,))]
        specs += [PhiSpec(5, frac=f) for f in (0.20, 0.10, 0.05)]
    return specs


def main(L_values, workers, outdir):
    for L in L_values:
        t = time.time()
        maps = scan_mfim_grid(L, G_VALUES, H_VALUES, specs_for(L), n_workers=workers)
        save_npz(outdir / f"mfim_L{L}.npz", g=G_VALUES, h=H_VALUES, **maps)
        print(f"L={L}: {len(maps)} landscapes saved in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--L", type=int, nargs="+", default=[9, 10, 11, 12])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--outdir", type=Path, default=ROOT / "results" / "landscapes")
    args = parser.parse_args()
    main(args.L, args.workers, args.outdir)
