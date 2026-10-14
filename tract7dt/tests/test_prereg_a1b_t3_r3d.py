"""Registered T3 re-test of M2 unit a.1b (card 024): prereg_a1b_t3_r3d.md with its addendum 2026-10-12.

What is fixed (registration 1 and 5; nothing here may change it):
- the two fixed files, asserted by sha256 before use: prereg_a3_r5d_truth.py and prereg_a3_r5d_population.json in the
  plan repository (card folder);
- estimator (1.1): `psf_model.tgm_floor` of m2-S1b 48a79e4 (TGM + the lambda identification check), moved onto the
  a.3 r5d merge M = f8ae6b7 by a merge; the function bodies and constants of the estimator are asserted equal to
  48a79e4's, and psf_from_stars.py (input 1) byte-equal to M's;
- scene and population (1.2): `p5_scene(seed)` and the M engine's `population(sc, pfs)`; the hash of every evaluation
  seed must equal `t3_r3d_populations[seed].sha256` of the JSON, otherwise the run stops before any metric;
- input 1 (the judged input): the M engine's iteration-0 a by THE SAME CALL as the a.3 r5d P-A3 (r5d) arm, i.e.
  `test_prereg_a3_r5d._engine(sc, pop)` (its body asserted equal to M's);
- r2: `star_r2` of `build_psf_model(stamps, fx, fy, a1, shape)` (no tau2: two passes); theta = tau2 / lambda, and
  `tgm_floor(r2, a1)` must equal the model record exactly (wiring assertion, before any draw);
- contamination, statistic D and draws (1.3, 1.4): `t3_alternative_draws(tgm_floor, r2_s, a_s)` for s = 9401..9405 in
  order (twenty draws per seed, rng default_rng(9421 + j));
- the verdict (1.4): `v = t3_verdict_1p4b(per_seed)`; T3 passes iff `v["pass"]` -- the ONLY judged assertion here.
Reported only (no threshold; 1.4 "report-only fields"): what `t3_verdict_1p4b` returns besides `pass`; per case and draw the
theta ratio and lambda_identified (from `t3_contamination_d`, whose D must equal the judged draws); nominal theta,
lambda-hat, a_min / theta; input 2 (true a) and the unlisted-list population (hash = the JSON's
`t3_r3d_populations[seed].unlisted.sha256`) through the same computation.

Evaluation seeds (the 9401-9405 metrics and the 9421-9440 draws) are computed only in the deliberate, recorded run:
every test that touches them SKIPs (not passes) unless $UNIPHOT_PREREG_A1B_T3_RUN holds the engine repository's HEAD
commit (>= 7 hex digits), `git status --porcelain -- tract7dt` of that repository is empty, and $UNIPHOT_PREREG_OUT
names an existing directory holding no t3_r3d_*.json yet (one run per output directory).  Every record JSON carries
the HEAD and the time.  A plain `pytest tract7dt/tests` runs only the fixed-file and code-identity checks.

Official run statement: development record impl_a1b.md (not distributed), section "T3 r3d" (HEAD, clean tree, output
directory, command).
"""
from __future__ import annotations

import functools
import glob
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import time

import numpy as np
import pytest

import dev_records
import test_prereg_a3_r5d as A3
from tract7dt.fit.grid_gpu import psf_from_stars as pfs
from tract7dt.fit.grid_gpu import psf_model as pm

# Every test here reads the development records (dev_records.py): skipped, with the reason, where they are absent.
pytestmark = pytest.mark.dev_record

TRUTH_NAME = "prereg_a3_r5d_truth.py"
JSON_NAME = "prereg_a3_r5d_population.json"
TRUTH_SHA256 = "19bef25e018f8dfc2af3b1f9f97bded1a0dc3c6103789ccf900e36e675725c3a"
JSON_SHA256 = "e8d7ca39afaea7aec2a03a96508f04bdc82b2d25db84fedfb798a5a83551649e"
RUN_ENV = "UNIPHOT_PREREG_A1B_T3_RUN"
OUT_ENV = "UNIPHOT_PREREG_OUT"
M_COMMIT = "f8ae6b7249a9aaf55a0afeecbed8fdbdd0ccda51"          # main after the a.3 r5d merge (0.3-1)
A1B_COMMIT = "48a79e4"                                          # the a.1b code of registration 1.1
PFS_PATH = "tract7dt/fit/grid_gpu/psf_from_stars.py"
PSF_MODEL_PATH = "tract7dt/fit/grid_gpu/psf_model.py"
A3_TEST_PATH = "tract7dt/tests/test_prereg_a3_r5d.py"
# sha256 of psf_from_stars.py at M (input 1 is M's engine; the a.1b merge does not touch the file)
PFS_SHA256_AT_M = "8f57f64aa1425052bbbc8a21f734cc71ee16cdacace28b10a326e65bd7dd1a74"
# sha256 of the body of the a.3 r5d P-A3 engine call `_engine` at M (test_body_sha256 of the truth module)
A3_ENGINE_BODY_SHA256 = "21cad47d5a2e08673ad9a3dd17ab42fd83213a873bd40d1bc26b9d64fc53a4a9"
# the estimator of registration 1.1: these functions of psf_model.py at 48a79e4 (bodies by test_body_sha256)
TGM_BODIES = ("tgm_floor", "_tgm_groups", "_mean_shift_mode", "_ms_global_gap", "_wmi", "_n50")
TGM_CONSTANTS = dict(TGM_GROUP_MIN=15, TGM_BANDWIDTH=0.5, TGM_MS_TOL=1e-14, TGM_MS_MAX=1000, TGM_FLOOR=1e-3,
                     TGM_GAP_GRID=2000, TGM_MIN_STARS=30)
_STATE: dict = {}


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@functools.lru_cache(maxsize=1)
def TR():
    truth_path = dev_records.plan_file(TRUTH_NAME)
    assert _sha(truth_path) == TRUTH_SHA256, "prereg_a3_r5d_truth.py is not the registered file"
    spec = importlib.util.spec_from_file_location("prereg_a3_r5d_truth", truth_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=1)
def J():
    TR()
    json_path = dev_records.plan_file(JSON_NAME)
    assert _sha(json_path) == JSON_SHA256, "prereg_a3_r5d_population.json is not the registered file"
    return json.load(open(json_path))


def _repo():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(pfs.__file__)))))


def _git(*args, text=True):
    return subprocess.run(["git", "-C", _repo(), *args], capture_output=True, text=text, check=True).stdout


def _official_run() -> dict:
    """Compute evaluation seeds only in the deliberate run of a committed engine, recorded to a fresh OUT."""
    want = os.environ.get(RUN_ENV, "")
    out = os.environ.get(OUT_ENV)
    try:
        head = _git("rev-parse", "HEAD").strip()
        dirty = _git("status", "--porcelain", "--", "tract7dt").strip()
    except Exception as exc:  # noqa: BLE001 -- not a git work tree (e.g. a copy): not the official run
        pytest.skip(f"prereg_a1b_t3_r3d evaluation seeds: engine repository not readable by git ({exc})")
    if not (len(want) >= 7 and head.startswith(want)):
        pytest.skip(f"prereg_a1b_t3_r3d evaluation seeds run once, deliberately: set {RUN_ENV} to the engine HEAD")
    if dirty:
        pytest.skip("prereg_a1b_t3_r3d evaluation seeds: the engine tree has uncommitted changes under tract7dt/")
    if not (out and os.path.isdir(out)):
        pytest.skip(f"prereg_a1b_t3_r3d evaluation seeds: {OUT_ENV} must name an existing directory")
    if "out" not in _STATE and glob.glob(os.path.join(out, "t3_r3d_*.json")):
        pytest.skip(f"prereg_a1b_t3_r3d evaluation seeds: {out} already holds a run's records (one run per directory)")
    _STATE.update(engine_head=head, out=out,
                  head_time=_git("log", "-1", "--format=%ci", "HEAD").strip())
    return dict(engine_head=head)


def _record(name, obj):
    out = _STATE.get("out")
    if out and os.path.isdir(out):
        meta = dict(engine_head=_STATE.get("engine_head"), engine_head_time=_STATE.get("head_time"),
                    written=time.strftime("%Y-%m-%dT%H:%M:%S%z"), test_file_sha256=_sha(__file__))
        with open(os.path.join(out, f"t3_r3d_{name}.json"), "w") as fh:
            json.dump(dict(meta, **obj), fh, indent=1,
                      default=lambda v: v.tolist() if hasattr(v, "tolist") else float(v))


# ---------------------------------------------------------------------------------------------------------------
# Checks that run in every plain pytest (no evaluation seed)
# ---------------------------------------------------------------------------------------------------------------
def test_fixed_files_are_the_registered_ones():
    T = TR()
    j = J()
    assert tuple(j["seeds"]["T3"]) == T.T3_SEEDS == (9401, 9402, 9403, 9404, 9405)
    assert j["seeds"]["T3_contam"] == T.T3_CONTAM_SEED == 9421
    assert tuple(j["seeds"]["T3_appendixA"]) == T.T3_ALT_CONTAM_SEEDS == tuple(range(9421, 9441))
    assert T.T3_GA_MAX_EXCEED == 3 and T.T3_LIMITS == {"all20x10": 0.2, "Q40x10": 0.35}
    assert sorted(j["t3_r3d_populations"]) == [str(s) for s in T.T3_SEEDS]


def _code_identity() -> dict:
    """Registration 1.1 / 1.2 / 0.3-2: the estimator is 48a79e4's, input 1 is M's engine by the a.3 P-A3 call."""
    T = TR()
    src = dev_records.git_show(A1B_COMMIT, PSF_MODEL_PATH)
    assert hashlib.sha256(src).hexdigest() == J()["a1b"]["psf_model_sha256"], "48a79e4 psf_model.py is not JSON a1b"
    with tempfile.NamedTemporaryFile("wb", prefix="a1b_t3_pm_", suffix=".py", delete=False) as fh:
        fh.write(src)
    tmp = fh.name
    try:
        bodies = {f: T.test_body_sha256(pm.__file__, f) == T.test_body_sha256(tmp, f) for f in TGM_BODIES}
    finally:
        os.unlink(tmp)
    res = dict(tgm_bodies=bodies,
               tgm_constants={k: getattr(pm, k, None) == v for k, v in TGM_CONSTANTS.items()},
               pfs_file_is_M=_sha(pfs.__file__) == PFS_SHA256_AT_M,
               pfs_blob_at_M=hashlib.sha256(dev_records.git_show(M_COMMIT, PFS_PATH)).hexdigest()
               == PFS_SHA256_AT_M,
               a3_engine_call_is_M=T.test_body_sha256(A3.__file__, "_engine") == A3_ENGINE_BODY_SHA256,
               M_is_ancestor=subprocess.run(["git", "-C", _repo(), "merge-base", "--is-ancestor", M_COMMIT, "HEAD"])
               .returncode == 0,
               a1b_is_ancestor=subprocess.run(["git", "-C", _repo(), "merge-base", "--is-ancestor", A1B_COMMIT,
                                               "HEAD"]).returncode == 0)
    res["pass"] = bool(all(res["tgm_bodies"].values()) and all(res["tgm_constants"].values())
                       and all(v for k, v in res.items() if isinstance(v, bool)))
    return res


def test_code_identity_of_estimator_and_input():
    res = _code_identity()
    assert res["pass"], res


# ---------------------------------------------------------------------------------------------------------------
# The computation of one seed (also used by the pre-run dry check on SEEN seeds; no draw is made here)
# ---------------------------------------------------------------------------------------------------------------
def seed_inputs(sc: dict, pop: dict) -> dict:
    """Input 1 (the a.3 r5d P-A3 engine call), the a.1b model on it (two passes), r2 = star_r2, theta, and the
    wiring check tgm_floor(r2, a1) == the model record (exact)."""
    h, w = pop["shape"]
    a1, info1 = A3._engine(sc, pop)
    a1 = np.asarray(a1, dtype=np.float64)
    m = pm.build_psf_model(pop["stamps"], pop["fx"], pop["fy"], a1, (h, w))
    r2 = np.asarray(m.star_r2, dtype=np.float64)
    lam, t2, inf = pm.tgm_floor(r2, a1)
    rec = m.record
    return dict(a=a1, info=info1, model=m, r2=r2, lam=float(rec["lam"]), tau2=float(rec["tau2"]),
                theta=float(rec["tau2"] / rec["lam"]),
                a_min_over_theta=float(np.min(a1) / (rec["tau2"] / rec["lam"])) if rec["tau2"] > 0 else None,
                lambda_identified=rec["lambda_identified"], n=int(a1.size),
                wiring_exact=bool(lam == rec["lam"] and t2 == rec["tau2"]),
                wiring_identified_same=bool(inf.get("lambda_identified") == rec["lambda_identified"]))


def draw_details(T, r2, a, contam_seeds) -> dict:
    """Report columns per case and draw: D, theta ratio, lambda_identified (t3_contamination_d, frozen)."""
    out = {name: dict(D=[], theta_ratio=[], lambda_identified=[]) for name, _, _, _ in T.T3_CASES}
    base = None
    for cs in contam_seeds:
        o = T.t3_contamination_d(pm.tgm_floor, r2, a, cs)
        base = dict(theta=o["theta"], lam=o["lam"], lambda_identified=o["lambda_identified"])
        for name in out:
            for k in ("D", "theta_ratio", "lambda_identified"):
                out[name][k].append(o[name][k])
    return dict(out, nominal=base)


# ---------------------------------------------------------------------------------------------------------------
# Evaluation seeds 9401-9405 (official run only)
# ---------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def t3():
    _official_run()
    T = TR()
    ident = _code_identity()
    _record("code_identity", ident)
    if not ident["pass"]:
        pytest.fail("prereg_a1b_t3_r3d 0.3-2 / 1.1: estimator or input code is not the registered one -- stopped "
                    "before any evaluation seed")
    jp = J()["t3_r3d_populations"]
    seeds = {}
    for seed in T.T3_SEEDS:
        sc = T.p5_scene(seed)
        pop = T.population(sc, pfs)
        pu = T.population(T.unlisted_scene(sc), pfs)
        seeds[seed] = dict(sc=sc, pop=pop, pu=pu, hash=T.population_hash(pop), hash_u=T.population_hash(pu))
    hashes = {str(k): dict(hash=s["hash"], hash_u=s["hash_u"], n_train=int(len(s["pop"]["fx"])),
                           n_train_u=int(len(s["pu"]["fx"]))) for k, s in seeds.items()}
    hashes_ok = all(s["hash"] == jp[str(k)]["sha256"] and s["hash_u"] == jp[str(k)]["unlisted"]["sha256"]
                    for k, s in seeds.items())
    _record("population_hashes", dict(per_seed=hashes, pass_=hashes_ok))
    if not hashes_ok:
        return dict(stage="hashes", hashes_ok=False, hashes=hashes)
    # input 1, model, theta, wiring: all five seeds before any draw
    for seed, s in seeds.items():
        s["in1"] = seed_inputs(s["sc"], s["pop"])
    nominal = {str(k): {q: s["in1"][q] for q in ("n", "lam", "tau2", "theta", "a_min_over_theta",
                                                 "lambda_identified", "wiring_exact", "wiring_identified_same")}
               for k, s in seeds.items()}
    wiring_ok = all(s["in1"]["wiring_exact"] and s["in1"]["wiring_identified_same"] for s in seeds.values())
    _record("nominal", dict(per_seed=nominal, wiring_pass=wiring_ok))
    if not wiring_ok:
        return dict(stage="wiring", hashes_ok=True, wiring_ok=False, nominal=nominal)
    # the registered draws and verdict (1.4)
    per_seed = [T.t3_alternative_draws(pm.tgm_floor, seeds[s]["in1"]["r2"], seeds[s]["in1"]["a"]) for s in T.T3_SEEDS]
    v = T.t3_verdict_1p4b(per_seed)
    _record("verdict", dict(verdict=v, draws={str(s): {k: d[k] for k in d} for s, d in zip(T.T3_SEEDS, per_seed)}))
    return dict(stage="done", hashes_ok=True, wiring_ok=True, seeds=seeds, per_seed=per_seed, verdict=v,
                nominal=nominal)


def _require(t3, stage):
    if not t3["hashes_ok"]:
        pytest.fail(f"prereg_a1b_t3_r3d 1.2: a population hash differs -- stopped before any metric: {t3['hashes']}")
    if stage == "verdict" and not t3.get("wiring_ok"):
        pytest.fail(f"prereg_a1b_t3_r3d 1.2: tgm_floor(r2, a) != the model record -- stopped before any draw: "
                    f"{t3['nominal']}")


def test_t3_population_hashes(t3):
    _require(t3, "hashes")


def test_t3_theta_wiring(t3):
    _require(t3, "verdict")


def test_t3_verdict_1p4b(t3):
    """THE registered T3 verdict (prereg_a1b_t3_r3d.md 1.4): pass = t3_verdict_1p4b(per_seed)["pass"]."""
    _require(t3, "verdict")
    v = t3["verdict"]
    assert v["pass"], {k: v[k] for k in ("n_ga", "n_ga_per_seed", "ga_max", "ga_pass", "na_pass", "na_medians",
                                         "na_frac_over")}


def test_t3_reports(t3):
    """Reported only (no threshold): theta ratios and identification per case and draw, input 2 (true a), the
    unlisted-list population.  The only assertion: the report's D equal the judged draws (same frozen code)."""
    _require(t3, "verdict")
    T = TR()
    rep = {}
    same = True
    per_seed_true, per_seed_unl = [], []
    for seed, d_judged in zip(T.T3_SEEDS, t3["per_seed"]):
        s = t3["seeds"][seed]
        sc, pop, pu, in1 = s["sc"], s["pop"], s["pu"], s["in1"]
        det = draw_details(T, in1["r2"], in1["a"], T.T3_ALT_CONTAM_SEEDS)
        same = same and all(np.array_equal(np.asarray(det[k]["D"]), np.asarray(d_judged[k])) for k in d_judged)
        # input 2: the true a (1.2), same computation
        info = in1["info"]
        a2 = T.a_formula(pop["F"], info["sigma2_sky"], T.true_sigma2_nb(sc, pop), float(info["kappa"]),
                         float(info["phi_norm2"]))
        m2 = pm.build_psf_model(pop["stamps"], pop["fx"], pop["fy"], a2, pop["shape"])
        r2_2 = np.asarray(m2.star_r2, dtype=np.float64)
        d2 = T.t3_alternative_draws(pm.tgm_floor, r2_2, a2)
        det2 = draw_details(T, r2_2, a2, T.T3_ALT_CONTAM_SEEDS)
        per_seed_true.append(d2)
        # unlisted list (planted rows not in the list): the same engine call on that population
        inu = seed_inputs(sc, pu)
        du = T.t3_alternative_draws(pm.tgm_floor, inu["r2"], inu["a"])
        per_seed_unl.append(du)
        rep[str(seed)] = dict(
            input1=dict(details=det, a_ratio_true_over_input1=[float(np.min(a2 / in1["a"])),
                                                               float(np.median(a2 / in1["a"])),
                                                               float(np.max(a2 / in1["a"]))]),
            input2=dict(lam=float(m2.record["lam"]), tau2=float(m2.record["tau2"]),
                        theta=float(m2.record["tau2"] / m2.record["lam"]),
                        lambda_identified=m2.record["lambda_identified"], details=det2),
            unlisted=dict(n=inu["n"], lam=inu["lam"], tau2=inu["tau2"], theta=inu["theta"],
                          lambda_identified=inu["lambda_identified"], wiring_exact=inu["wiring_exact"],
                          D={k: du[k] for k in du}))
    rep["input2_verdict_form"] = T.t3_verdict_1p4b(per_seed_true)
    rep["unlisted_verdict_form"] = T.t3_verdict_1p4b(per_seed_unl)
    rep["report_D_equal_judged_D"] = bool(same)
    _record("reports", rep)
    assert same
