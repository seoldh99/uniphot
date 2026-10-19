"""Registered tests of M2 unit a.3, round 5d (card 024): prereg_a.3_r5d.md with its addendum A.0-A.11.

Fixed before any result (do not change the statistics, thresholds, seeds or rules; prereg_a.3_r5d.md 4):
- the two fixed files are asserted by sha256 before use (4, end): prereg_a3_r5d_truth.py and
  prereg_a3_r5d_population.json in the plan repository (card folder);
- order (3.1-2): P-A2 (seed 9101) -> P-A1 -> the 1.5 conformance check on the SEEN seed 5 (full list, unlisted list,
  near-row list 1.5-6) -> the population hashes of the evaluation seeds 9141-9145 -> P-A3 -> P-A4 -> P-A5;
- 3.1-3: a failed 1.5 check or hash assertion stops the run BEFORE any metric is computed on an evaluation seed (the
  evaluation fixtures call `_gate()`, which fails without computing anything when 1.5 or a hash did not pass);
- P-A3: five statistics on the five evaluation seeds pooled, all must hold; positive control r4_control pooled
  (1 >= 10 and 2 >= 0.06); P-A4 (ii): mean over the five seeds of [peak_err(r5d) - peak_err(R3)] <= 0 (judged and
  recorded; 3.2: a failure does not block the merge and is recorded as an open failure); P-A4 (i) and (iii) and P-A5 are
  recorded only.
Evaluation seeds are computed only in the deliberate, recorded run: the
P-A2 test and the evaluation fixture SKIP (not pass) unless $UNIPHOT_PREREG_R5D_RUN holds the engine repository's HEAD
commit (>= 7 hex digits), `git status --porcelain -- tract7dt` of that repository is empty, and $UNIPHOT_PREREG_OUT is an
existing directory; every record JSON then carries the HEAD and the time.  Before any evaluation seed the gate also
needs: the default path (detection_fits False) gives the same a, sigma2_nb,
kappa, phi_norm2 as the judged path on seed 5; the engine's diagnostics are complete; the engine a equals the
reference a (relative 1e-6) on the three seed-5 lists; the projected / non-converged box counts equal the reference's;
star_noise_a, star_noise_t, neighbour_amplitude and psf_model.py are those of c15c72b (1.4 "unchanged").
The engine is called with sky sigma 1 and gain 50 (the P5 scene), the saturation mask data >= SATURATE / 1.3 and the
list rows of the population -- the same inputs as the reference `r5d_arm`.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

import dev_records
from tract7dt.fit.grid_gpu import psf_from_stars as pfs
from tract7dt.fit.grid_gpu import psf_model as pm

# Every test here reads the development records (dev_records.py): skipped, with the reason, where they are absent.
pytestmark = pytest.mark.dev_record

TRUTH_NAME = "prereg_a3_r5d_truth.py"
JSON_NAME = "prereg_a3_r5d_population.json"
TRUTH_SHA256 = "19bef25e018f8dfc2af3b1f9f97bded1a0dc3c6103789ccf900e36e675725c3a"
JSON_SHA256 = "e8d7ca39afaea7aec2a03a96508f04bdc82b2d25db84fedfb798a5a83551649e"
HELPERS = ("sep_auto_flux", "isolated_in_box", "empirical_star_stamp", "box_sum_kappa")
PA1_TEST = ("test_psf_star_weights.py", "test_radial_profile_weight_has_no_jump_as_a_neighbour_moves")
OUT = os.environ.get("UNIPHOT_PREREG_OUT")
RUN_ENV = "UNIPHOT_PREREG_R5D_RUN"
C15 = "c15c72b"
UNCHANGED_BODIES = ("star_noise_a", "star_noise_t", "neighbour_amplitude")
PSF_MODEL_PATH = "tract7dt/fit/grid_gpu/psf_model.py"
PFS_PATH = "tract7dt/fit/grid_gpu/psf_from_stars.py"


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


def _git(*args):
    return subprocess.run(["git", "-C", _repo(), *args], capture_output=True, text=True, check=True).stdout


def _official_run() -> dict:
    """Compute evaluation seeds only in the deliberate run of a committed engine, recorded to OUT."""
    want = os.environ.get(RUN_ENV, "")
    try:
        head = _git("rev-parse", "HEAD").strip()
        dirty = _git("status", "--porcelain", "--", "tract7dt").strip()
    except Exception as exc:  # noqa: BLE001 -- not a git work tree (e.g. a copy): not the official run
        pytest.skip(f"prereg_a.3_r5d evaluation seeds: engine repository not readable by git ({exc})")
    if not (len(want) >= 7 and head.startswith(want)):
        pytest.skip(f"prereg_a.3_r5d evaluation seeds run once, deliberately: set {RUN_ENV} to the engine HEAD")
    if dirty:
        pytest.skip("prereg_a.3_r5d evaluation seeds: the engine tree has uncommitted changes under tract7dt/")
    if not (OUT and os.path.isdir(OUT)):
        pytest.skip("prereg_a.3_r5d evaluation seeds: UNIPHOT_PREREG_OUT must name an existing directory")
    _STATE["engine_head"] = head
    return dict(engine_head=head)


def _record(name, obj):
    if OUT and os.path.isdir(OUT):
        meta = dict(engine_head=_STATE.get("engine_head"), written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        with open(os.path.join(OUT, f"{name}.json"), "w") as fh:
            json.dump(dict(meta, **obj), fh, indent=1,
                      default=lambda v: v.tolist() if hasattr(v, "tolist") else float(v))


_STATE: dict = {}


def _gate():
    """3.1-3: no evaluation-seed metric unless the 1.5 check and the hash assertions passed in this run."""
    if not _STATE.get("conformance_pass"):
        pytest.fail("prereg_a.3_r5d.md 3.1-3: the 1.5 conformance check has not passed in this run -- stopped "
                    "before any metric on an evaluation seed")


def _engine(sc, pop, detection_fits=True):
    data = sc["data"]
    return pfs.psf0_training_noise(
        data, pop["objs"], pop["star_det"], pop["fx"], pop["fy"], pop["F"], pop["stamps"], np.ones_like(data), 50.0,
        pop["bad"], pop["fm"], rows_x=np.asarray(pop["rows"].x, dtype=np.float64),
        rows_y=np.asarray(pop["rows"].y, dtype=np.float64),
        star_row=np.asarray(pop["paired"], dtype=np.int64)[pop["keep"]],
        sat_mask=sc["data"] >= sc["saturate"] / 1.3, half=27, detection_fits=detection_fits)


def _git_copy_dirs() -> set:
    """Folders of the module copies the truth module made from git objects (prefixes r5d_ / r3mod_) in this process."""
    out = set()
    for mod in list(sys.modules.values()):
        d = os.path.dirname(getattr(mod, "__file__", None) or "")
        if os.path.basename(d).startswith(("r5d_", "r3mod_")):
            out.add(d)
    return out


def _cleanup_git_copies(before):
    for d in _git_copy_dirs() - before:
        shutil.rmtree(d, ignore_errors=True)


def test_fixed_files_are_the_registered_ones():
    TR()
    J()
    assert J()["seeds"]["PA2"] == TR().PA2_SEED == 9101
    assert tuple(J()["seeds"]["EVAL"]) == TR().EVAL_SEEDS == (9141, 9142, 9143, 9144, 9145)


# ---------------------------------------------------------------------------------------------------------------
# P-A2 (seed 9101): accuracy of the phase table (prereg_a.3_r5.md P-A2, seed of r5b 2.2)
# ---------------------------------------------------------------------------------------------------------------
def test_pa2_phase_table_accuracy():
    from synth import psf as SP
    TR()
    _official_run()
    prof = pfs.MoffatProfile(4.5, 3.0)
    p, h = 16, 1.0 / 64.0
    size = 2 * prof.m + 1
    peak = float(np.max(prof.tab[0, 0]))
    tol = 1e-12 * peak

    def direct(phx, phy):
        return SP.pixel_integrate(prof.func, phx, phy, size=size, order=3)

    # (1) the 289 nodes = the direct GL3 integral of the same function (wiring check)
    node_err = max(float(np.max(np.abs(prof.tab[k, i] - direct(i / p, k / p)))) for k in range(p + 1) for i in range(p + 1))
    phases = np.random.default_rng(TR().PA2_SEED).uniform(0.0, 1.0, (200, 2))
    n_viol2 = n_neg = n_rise = 0
    worst2 = 0.0
    cache: dict = {}

    def d2(qx, qy):
        """Pixelwise |second differences| along phase x and phase y at (qx, qy) (step h)."""
        key = (round(qx * 4096), round(qy * 4096))
        if key not in cache:
            f0 = direct(qx, qy)
            cache[key] = (np.abs(direct(qx + h, qy) - 2 * f0 + direct(qx - h, qy)) / h ** 2,
                          np.abs(direct(qx, qy + h) - 2 * f0 + direct(qx, qy - h)) / h ** 2)
        return cache[key]

    for phx, phy in phases:
        st, nx0, ny0 = prof.placed(100.0 + phx, 100.0 + phy)
        assert (nx0, ny0) == (100, 100)
        # (2) between the nodes: |blend - direct| <= (D^2 / 8)(max f_xx + max f_yy) + 1e-12 peak, pixel by pixel; the max
        # over the cell's four corners and its centre
        i0, k0 = min(int(np.floor(phx * p)), p - 1), min(int(np.floor(phy * p)), p - 1)
        pts = [(i0 / p, k0 / p), ((i0 + 1) / p, k0 / p), (i0 / p, (k0 + 1) / p), ((i0 + 1) / p, (k0 + 1) / p),
               ((i0 + 0.5) / p, (k0 + 0.5) / p)]
        dxx = np.max([d2(*q)[0] for q in pts], axis=0)
        dyy = np.max([d2(*q)[1] for q in pts], axis=0)
        bound = (1.0 / p) ** 2 / 8.0 * (dxx + dyy) + tol
        err = np.abs(st - direct(phx, phy))
        n_viol2 += int(np.sum(err > bound))
        worst2 = max(worst2, float(np.max(err / bound)))
        # (3) no negative pixel; along the source's pixel row, beyond 2 x 4.5 px, the values never rise moving away
        m = prof.m
        n_neg += int(np.any(st < -1e-15 * peak))
        row = st[m + (1 if phy >= 0.5 else 0)]
        c = np.arange(-m, m + 1) - phx
        far = np.abs(c) > 2 * 4.5
        right = row[(c > 0) & far]
        left = row[(c < 0) & far][::-1]
        n_rise += int(np.any(np.diff(right) > 1e-15 * peak) or np.any(np.diff(left) > 1e-15 * peak))
    res = dict(node_err_over_peak=node_err / peak, n_violations_2=n_viol2, worst_ratio_2=worst2, n_negative=n_neg,
               n_rise=n_rise, peak=peak, m=prof.m)
    _record("pa2", res)
    assert node_err <= tol, res
    assert n_viol2 == 0, res
    assert n_neg == 0 and n_rise == 0, res


# ---------------------------------------------------------------------------------------------------------------
# P-A1: the frozen wobble test (only its decorator line changed), run here in the registered order
# ---------------------------------------------------------------------------------------------------------------
def test_pa1_frozen_body_and_result():
    path = os.path.join(os.path.dirname(__file__), PA1_TEST[0])
    assert TR().test_body_sha256(path, PA1_TEST[1]) == J()["pa1_test_body_sha256"]
    spec = importlib.util.spec_from_file_location("_pa1_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    getattr(mod, PA1_TEST[1])()


# ---------------------------------------------------------------------------------------------------------------
# 1.5: conformance with the independent reference on the SEEN seed 5 (before any evaluation-seed metric)
# ---------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def conformance():
    T = TR()
    out: dict = {}
    sc = T.p5_scene(5)
    pop = T.population(sc, pfs)
    out["hash_seed5"] = T.population_hash(pop)
    out["helpers"] = {f: T.test_body_sha256(pfs.__file__, f) for f in HELPERS}
    m_ref = T.r5d_reference_moffat(sc, pop, pfs)
    lists = {"full": pop, "unlisted": T.population(T.unlisted_scene(sc), pfs)}
    lists["nearrow"] = T.r5d_nearrow_pop(pop)
    extra_ok = True
    for name, p_ in lists.items():
        a_e, info_e = _engine(sc, p_)
        m_e = (float(info_e["moffat_fwhm"]), float(info_e["moffat_beta"]))
        a_r, info_r = T.r5d_arm(sc, p_, pfs, profile=m_e)
        chk = T.r5d_conformance(info_e, info_r, m_e, m_ref)
        # the engine a itself equals the reference a (relative 1e-6)
        a_rel = float(np.max(np.abs(a_e - a_r) / np.abs(a_r)))
        # the same definition gives the same projected / non-converged box counts
        counts_same = all(int(info_e[k]) == int(info_r[k]) for k in
                          ("n_star_projected", "n_star_not_converged", "n_det_projected", "n_det_not_converged"))
        # every diagnostic the evaluation tests read is there
        try:
            _diag(info_e, a_e)
            diag_ok = True
        except KeyError as exc:
            diag_ok = repr(exc)
        extra_ok = extra_ok and a_rel <= 1e-6 and counts_same and diag_ok is True
        diag = {k: info_e[k] for k in ("n_degenerate", "n_degenerate_det", "n_rows_merged", "n_unlisted", "n_known",
                                       "n_F_inf")}
        out[name] = dict(check=chk, a_max_rel_vs_reference=a_rel, counts_same=counts_same, diag_complete=diag_ok,
                         moffat_engine=m_e, moffat_ref=m_ref, diag=diag,
                         ref_counts=dict(n_degenerate=int(info_r["n_degenerate"]),
                                         n_degenerate_det=int(info_r["n_degenerate_det"]),
                                         n_star_projected=int(info_r["n_star_projected"]),
                                         n_star_not_converged=int(info_r["n_star_not_converged"]),
                                         n_det_projected=int(info_r["n_det_projected"]),
                                         n_det_not_converged=int(info_r["n_det_not_converged"])),
                         engine_counts={k: info_e[k] for k in ("n_star_projected", "n_star_not_converged",
                                                               "n_det_projected", "n_det_not_converged")},
                         sums=dict(sigma2_nb=float(np.sum(info_e["sigma2_nb"])),
                                   neighbour_flux=float(np.sum(info_e["neighbour_flux"])), a=float(np.sum(a_e)),
                                   ref_sigma2_nb=float(np.sum(info_r["sigma2_nb"])),
                                   ref_neighbour_flux=float(np.sum(info_r["neighbour_flux"])), ref_a=float(np.sum(a_r))))
    # registration 4 item 2: the default path (no detection fits) is the same code for a and sigma2_nb
    a_d, info_d = _engine(sc, pop, detection_fits=False)
    a_t, info_t = _engine(sc, pop, detection_fits=True)
    out["default_path_same"] = bool(np.array_equal(a_d, a_t) and np.array_equal(info_d["sigma2_nb"], info_t["sigma2_nb"])
                                    and info_d["kappa"] == info_t["kappa"] and info_d["phi_norm2"] == info_t["phi_norm2"])
    # the parts registration 1.4 keeps are those of c15c72b
    out["unchanged_vs_c15"] = _unchanged_vs_c15()
    _record("conformance_seed5", out)
    ok = (out["hash_seed5"] == J()["populations"]["5"]["sha256"]
          and out["helpers"] == J()["c15_helper_body_sha256"]
          and all(out[k]["check"]["pass"] for k in ("full", "unlisted", "nearrow"))
          and extra_ok and out["default_path_same"] and all(out["unchanged_vs_c15"].values()))
    _STATE["conformance_pass"] = bool(ok)
    return out


def _unchanged_vs_c15() -> dict:
    import tempfile
    T = TR()
    src = dev_records.git_show(C15, PFS_PATH)
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as fh:
        fh.write(src)
    try:
        res = {f: T.test_body_sha256(pfs.__file__, f) == T.test_body_sha256(fh.name, f) for f in UNCHANGED_BODIES}
    finally:
        os.unlink(fh.name)
    old_pm = dev_records.git_show(C15, PSF_MODEL_PATH)
    res["psf_model.py"] = hashlib.sha256(old_pm).hexdigest() == _sha(pm.__file__)
    return res


def test_1p5_1_population_hash_seed5(conformance):
    assert conformance["hash_seed5"] == J()["populations"]["5"]["sha256"]


def test_1p5_2_c15_helper_bodies_unchanged(conformance):
    assert conformance["helpers"] == J()["c15_helper_body_sha256"]


@pytest.mark.parametrize("which", ["full", "unlisted", "nearrow"])
def test_1p5_3_to_6_conformance(conformance, which):
    chk = conformance[which]["check"]
    assert max(chk["moffat_rel_diff"]) <= 1e-4, chk
    assert chk["same_shape"] and chk["same_degenerate_counts"], chk
    assert chk["pass"], chk


@pytest.mark.parametrize("which", ["full", "unlisted", "nearrow"])
def test_engine_a_counts_and_diagnostics_match_the_reference(conformance, which):
    """Wiring, seed 5: a equals the reference a, same projected / non-converged counts,
    all diagnostics present."""
    c = conformance[which]
    assert c["a_max_rel_vs_reference"] <= 1e-6, c["a_max_rel_vs_reference"]
    assert c["counts_same"], (c["engine_counts"], c["ref_counts"])
    assert c["diag_complete"] is True, c["diag_complete"]


def test_default_path_is_the_same_code_as_the_detection_path(conformance):
    """Registration 4 item 2: detection fits only on request; a, sigma2_nb, kappa and
    phi_norm2 of the default path are bit for bit those of the judged path (seed 5, full list)."""
    assert conformance["default_path_same"]


def test_parts_kept_by_registration_1p4_are_c15c72b(conformance):
    """star_noise_a, star_noise_t, neighbour_amplitude (body hashes in this tree) = c15c72b, and
    psf_model.py of the a.3 official-run commit 98c87d5 = c15c72b's (fixed commits, not this tree: unit a.1b changes
    psf_model.py after a.3 by design -- card 024 impl_a1b.md "T3 r3d", review_a1b_t3_prerun.md (3)).  This tree's
    psf_model.py comparison stays in the evaluation gate (`conformance_pass`) and is shown here as information."""
    assert (all(v for k, v in conformance["unchanged_vs_c15"].items() if k != "psf_model.py")
            and _git("rev-parse", f"{C15}:{PSF_MODEL_PATH}") == _git("rev-parse", f"98c87d5:{PSF_MODEL_PATH}")), \
        conformance["unchanged_vs_c15"]


# ---------------------------------------------------------------------------------------------------------------
# Evaluation seeds 9141-9145: population hashes, then P-A3, P-A4, P-A5
# ---------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def evaluation(conformance):
    _official_run()
    _gate()
    T = TR()
    before = _git_copy_dirs()
    seeds = {}
    try:
        for seed in T.EVAL_SEEDS:
            sc = T.p5_scene(seed)
            pop = T.population(sc, pfs)
            su = T.unlisted_scene(sc)
            pu = T.population(su, pfs)
            seeds[seed] = dict(sc=sc, pop=pop, pu=pu, hash=T.population_hash(pop), hash_u=T.population_hash(pu))
        hashes_ok = all(s["hash"] == J()["populations"][str(k)]["sha256"] and
                        s["hash_u"] == J()["populations"][str(k)]["unlisted"]["sha256"] for k, s in seeds.items())
        _STATE["hashes_pass"] = bool(hashes_ok)
        if not hashes_ok:
            return dict(seeds={k: dict(hash=s["hash"], hash_u=s["hash_u"]) for k, s in seeds.items()}, hashes_ok=False)
        pfs3, pm3 = T.load_r3_modules()
        for seed, s in seeds.items():
            sc, pop, pu = s["sc"], s["pop"], s["pu"]
            h, w = pop["shape"]
            tn = T.true_sigma2_nb(sc, pop)
            a, info = _engine(sc, pop)
            s["a"], s["info"] = a, info
            s["arr"] = T.pa3_arrays(a, info, pop, sc, tn)
            a4, info4 = T.r4_control(sc, pop, pfs)
            s["arr_ctrl"] = T.pa3_arrays(a4, info4, pop, sc, tn)
            s["pa4"] = dict(
                r5d=T.peak_stats(pm.build_psf_model(pop["stamps"], pop["fx"], pop["fy"], a, (h, w)), sc, pop),
                R3=T.r3_arm(sc, pop, pfs3, pm3),
                true=T.peak_stats(pm.build_psf_model(
                    pop["stamps"], pop["fx"], pop["fy"],
                    T.a_formula(pop["F"], info["sigma2_sky"], tn, float(info["kappa"]), float(info["phi_norm2"])),
                    (h, w)), sc, pop),
                tau2_sens=T.tau2_sensitivity(pm, pop, a))
            tnu = T.true_sigma2_nb(sc, pu)
            au, info_u = _engine(sc, pu)
            s["arr_u"] = T.pa3_arrays(au, info_u, pu, sc, tnu)
            s["pa5"] = dict(n_train=int(len(pu["fx"])), n_unlisted_known=int(info_u["n_unlisted"]),
                            hidden_merged_companions=int(T.hidden_merged_companions(sc, pu)))
    finally:
        _cleanup_git_copies(before)
    return dict(seeds=seeds, hashes_ok=True)


def test_evaluation_population_hashes(evaluation):
    assert evaluation["hashes_ok"], {k: (v["hash"], v["hash_u"]) for k, v in evaluation["seeds"].items()}


def _require_hashes(evaluation):
    if not evaluation["hashes_ok"]:
        pytest.fail("prereg_a.3_r5d.md 3.1-3: a population hash differs -- stopped before any metric")


def test_pa3_iteration0_weight_accuracy(evaluation):
    _require_hashes(evaluation)
    T = TR()
    seeds = evaluation["seeds"]
    pooled = T.pa3_stats(T.pool([s["arr"] for s in seeds.values()]))
    ctrl = T.pa3_stats(T.pool([s["arr_ctrl"] for s in seeds.values()]))
    per_seed = {k: T.pa3_stats(s["arr"]) for k, s in seeds.items()}
    res = dict(pooled=pooled, passes=T.passes(pooled, T.PA3_LIMITS), control=ctrl,
               control_passes=T.passes(ctrl, T.PA3_CONTROL_LIMITS), per_seed=per_seed)
    _record("pa3", res)
    _record("pa3_diagnostics", {str(k): _diag(s["info"], s["a"]) for k, s in seeds.items()})
    assert all(res["control_passes"].values()), res["control"]
    assert all(res["passes"].values()), res["pooled"]


def _diag(info, a):
    keys = ("n_known", "n_unlisted", "n_rows_merged", "n_F_inf", "n_degenerate", "n_degenerate_det",
            "n_star_projected", "n_star_projected_at_edge", "n_star_projected_returned", "n_star_not_converged",
            "n_star_own_nonpositive", "n_star_iterations_exhausted", "n_det_projected", "n_det_projected_at_edge",
            "n_det_projected_returned", "n_det_not_converged", "n_det_own_nonpositive", "n_det_iterations_exhausted",
            "neighbour_bins", "n_stars_other_row_near_paired", "moffat_fwhm", "moffat_beta")
    mv = np.hypot(info["own_move_px"][:, 0], info["own_move_px"][:, 1])
    return dict({k: info[k] for k in keys}, a_sum=float(np.sum(a)),
                move_px=dict(median=float(np.median(mv)), p99=float(np.percentile(mv, 99)), max=float(np.max(mv))),
                kept_median=float(np.median(info["n_sources"])), kept_max=int(np.max(info["n_sources"])),
                cond_max=float(np.max(info["cond"])))


def test_pa4_psf0_peak_not_worse_than_round3(evaluation):
    """(ii) judged and recorded (3.2: a failure is an open failure and does not block the merge); (i), (iii) recorded."""
    _require_hashes(evaluation)
    seeds = evaluation["seeds"]
    diff = {k: s["pa4"]["r5d"]["peak_err"] - s["pa4"]["R3"]["peak_err"] for k, s in seeds.items()}
    mean_diff = float(np.mean(list(diff.values())))
    sens = {k: s["pa4"]["tau2_sens"] for k, s in seeds.items()}
    knife = {k: bool(v["1.04"] <= 0.1 * v["0.96"]) for k, v in sens.items()}
    res = dict(per_seed={k: {arm: {q: s["pa4"][arm][q] for q in ("peak_err", "mean", "spread", "tau2", "n_eff")}
                             for arm in ("r5d", "R3", "true")} for k, s in seeds.items()},
               i_p5_tolerance_0p2pct={k: bool(s["pa4"]["r5d"]["peak_err"] <= 2e-3) for k, s in seeds.items()},
               ii_diff_r5d_minus_R3=diff, ii_mean_diff=mean_diff, ii_pass=bool(mean_diff <= 0.0),
               iii_tau2_sensitivity=sens, iii_knife_count=int(sum(knife.values())))
    _record("pa4", res)
    assert mean_diff <= 0.0, res


def test_pa5_unlisted_companions_report(evaluation):
    """Report only (no threshold): the r5d arm with the planted rows left out of the list."""
    _require_hashes(evaluation)
    T = TR()
    seeds = evaluation["seeds"]
    res = dict(pooled=T.pa3_stats(T.pool([s["arr_u"] for s in seeds.values()])),
               per_seed={k: dict(T.pa3_stats(s["arr_u"]), **s["pa5"]) for k, s in seeds.items()})
    _record("pa5", res)
