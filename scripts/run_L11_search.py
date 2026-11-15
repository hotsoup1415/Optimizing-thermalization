"""Two-spin coupling search at L=11 (block of 5 spins, l=4), warm-started from L=9.

* Min 1 (Newton): L-BFGS-B over all six canonical parameters, seeded at the L=9 Min 1.
* Min 2 (DE): differential evolution over the five couplings with J = J_PRX,
  started from a population around the L=9 Min 2 and limited to 20 generations.

Each optimum is then checked with the reduced standard-frame Hessian and a 9x9
plane scan along its two softest modes. Results go to results/L11_search.json.
Runtime is roughly 1.5-3 hours on a laptop.

    python scripts/run_L11_search.py
"""

import argparse
import time
from pathlib import Path

import numpy as np

from spin_eth import COUPLINGS, J_PRX, CanonicalModel
from spin_eth.io import load_json, save_json
from spin_eth.search import RecordingObjective, analyze_minimum, lbfgs, run_de, warm_start_population

ROOT = Path(__file__).resolve().parents[1]


def log(*args):
    print(*args, flush=True)


def main(out):
    t0 = time.time()
    seeds = load_json(ROOT / "results" / "minima.json")["L9"]

    log("Building L=11 operators...")
    model = CanonicalModel(L=11, l=4)
    phi_prx = model.phi_mfim()
    log(f"Phi(PRX, L=11, l=4) = {phi_prx:.6e}")
    log(f"J_PRX = {J_PRX:.10f}")

    # -- Min 1: Newton (L-BFGS-B), warm start from L=9 ------------------------
    x0 = np.array(seeds["min1_canonical"])
    f1 = RecordingObjective(model.phi)
    log(f"\nMin 1 seed (L=9 optimum): Phi = {model.phi(x0):.6e}")
    t = time.time()
    res1 = lbfgs(f1, x0)
    log(f"L-BFGS-B: Phi = {res1.fun:.6e} (success={res1.success}, nfev={res1.nfev})  [{time.time() - t:.0f}s]")
    min1 = analyze_minimum(model, res1.x, "Min 1 (Newton)", phi_prx, log=log)

    # -- Min 2: DE with J = J_PRX, warm-started population ---------------------
    c0 = np.array(seeds["min2_canonical"][1:])
    f2 = RecordingObjective(model.phi_fixed_J(J_PRX))
    log(f"\nMin 2 seed (L=9 optimum): Phi = {f2.f(c0):.6e}")
    t = time.time()
    de = run_de(f2, [(-5.0, 5.0)] * len(COUPLINGS), init=warm_start_population(c0), seed=42, maxiter=20)
    log(f"DE: Phi = {de.fun:.6e} (success={de.success}, nfev={de.nfev})  [{time.time() - t:.0f}s]")
    min2 = analyze_minimum(model, np.concatenate([[J_PRX], de.x]), "Min 2 (DE)", phi_prx, log=log)

    def summary(res, opt):
        keep = ("x_std", "rotation", "phi", "evals", "evecs", "n_neg", "alphas", "grid")
        return {"x_canonical": opt.x if len(opt.x) == 6 else np.concatenate([[J_PRX], opt.x]),
                "optimizer_fun": opt.fun, "optimizer_success": opt.success, "optimizer_nfev": opt.nfev,
                **{k: res[k] for k in keep}}

    save_json(out, {"L": 11, "l": 4, "phi_prx": phi_prx, "min1": summary(min1, res1),
                    "min2": summary(min2, de), "wall_time_s": time.time() - t0})
    log(f"\nSaved {out}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "L11_search.json")
    main(parser.parse_args().out)
