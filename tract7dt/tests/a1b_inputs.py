"""Frozen inputs of the M2 a.1b registration (development record prereg_a1b.md, not distributed).

The P5 scene and training population come from the committed truth module prereg_a3_r5_truth.py (plan repository
65e9dd9); the detection / selection / extraction functions and the iteration-0 noise (registered input 1) are those of
m2-S1 c15c72b, read from git objects into the package (read-only), so they do not depend on this branch's code.
Not a test module (no test functions); the a.1b tests import it.  Both are development records (dev_records.py): where
they are not available, the functions below skip the calling test.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

import dev_records

C15 = "c15c72b"
C15_PFS_SHA256 = "cfbfcb1b7664fd8d5f5dc25eec72fa5fb5a19f8ff23a57a1d43a4ad97348f473"
TRUTH_SHA256 = "aa37fc285c1f7aef7131d5a435dd62d77dd41fcc622f9d7e57884b832ea2ecda"
EVAL_SEEDS = (8901, 8902, 8903, 8904, 8905)
_CACHE: dict = {}


def truth():
    """prereg_a3_r5_truth (checked against the registered sha256)."""
    if "truth" not in _CACHE:
        path = Path(dev_records.plan_file("prereg_a3_r5_truth.py"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == TRUTH_SHA256, "truth module changed"
        plan_card = str(path.parent)
        if plan_card not in sys.path:
            sys.path.insert(0, plan_card)
        import prereg_a3_r5_truth as tr
        _CACHE["truth"] = tr
    return _CACHE["truth"]


def pfs_c15():
    """psf_from_stars of m2-S1 c15c72b from git objects, imported inside the package."""
    if "pfs" not in _CACHE:
        src = dev_records.git_show(C15, "tract7dt/fit/grid_gpu/psf_from_stars.py")
        assert hashlib.sha256(src).hexdigest() == C15_PFS_SHA256, "c15c72b psf_from_stars changed"
        with tempfile.TemporaryDirectory(prefix="a1b_c15_") as tmp:   # removed once the module is imported
            fn = Path(tmp) / f"psf_from_stars_{C15}.py"
            fn.write_bytes(src)
            name = f"tract7dt.fit.grid_gpu.psf_from_stars_{C15}"
            spec = importlib.util.spec_from_file_location(name, fn)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        _CACHE["pfs"] = mod
    return _CACHE["pfs"]


def seed_inputs(seed: int) -> dict:
    """Scene, population and input 1 (c15c72b iteration-0 a and info) of one seed; cached per process."""
    key = ("seed", int(seed))
    if key not in _CACHE:
        tr, pfs = truth(), pfs_c15()
        sc = tr.p5_scene(int(seed))
        pop = tr.population(sc, pfs)
        a1, info = pfs.psf0_training_noise(sc["data"], pop["objs"], pop["star_det"], pop["fx"], pop["fy"], pop["F"],
                                           pop["stamps"], np.ones_like(sc["data"]), 50.0, pop["bad"], pop["fm"],
                                           half=27)
        _CACHE[key] = dict(sc=sc, pop=pop, a1=np.asarray(a1, dtype=np.float64), info=info,
                           population_hash=tr.population_hash(pop))
    return _CACHE[key]


def input2(d: dict) -> np.ndarray:
    """Registered input 2 (true a), reported only."""
    tr = truth()
    nb = tr.true_sigma2_nb(d["sc"], d["pop"])
    return tr.a_formula(d["pop"]["F"], d["info"]["sigma2_sky"], nb, float(d["info"]["kappa"]),
                        float(d["info"]["phi_norm2"]))


def tau2_old(r2, a) -> float:
    """The rule a.1b deletes (c15c72b / d35dfed psf_model.tau2_floor), re-written here: tau^2 >= 0 solving
    median_s r2_s / (a_s + tau^2) = 1 (0 when the median is <= 1 at 0); bisection, 200 halvings."""
    r2 = np.asarray(r2, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    f = lambda t2: float(np.median(r2 / (a + t2))) - 1.0
    if f(0.0) <= 0.0:
        return 0.0
    lo, hi = 0.0, max(float(np.max(r2)), 1e-300)
    while f(hi) > 0.0:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def grid_xy(shape):
    h, w = shape
    gx, gy = np.meshgrid(np.linspace(300.0, w - 300.0, 4), np.linspace(300.0, h - 300.0, 3))
    return gx.ravel(), gy.ravel()
