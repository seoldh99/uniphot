"""M2 unit a.1b: robust tau^2 and a-scale estimation TGM (two groups, Gaussian mean-shift mode, interpolated
weighted median; two fixed steps).

Design: development design note unit_a1b_tau2_spec_v3.md, not distributed (with v2 where v3 changes nothing).
Registration: development record prereg_a1b.md (not distributed; committed before this file).
The registered definition is written here independently (`_tgm`); the evaluation tests use the frozen inputs of
a1b_inputs.py (P5 scene / population of prereg_a3_r5_truth 65e9dd9, input 1 = the m2-S1 c15c72b iteration-0 a).
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pytest

import a1b_inputs as I
from tract7dt.fit.grid_gpu import psf_model as pm

RECORD_KEYS = ("lam", "tau2", "theta", "ms_iters", "n_B", "n_F", "N50", "n_eff_stars", "a_min_over_theta",
               "ms_global_gap", "tau2_zero")


# ---------------------------------------------------------------------------
# The registered estimator (prereg_a1b.md 1.1), written here
# ---------------------------------------------------------------------------

def _groups(a):
    o = np.argsort(a, kind="mergesort")
    n = a.size
    return o[:max(15, n // 2)], o[n - max(15, n - n // 2):], o[:max(15, n // 4)]


def _ms(v):
    v = np.asarray(v, dtype=np.float64)
    m = float(np.median(v))
    h = 0.5 * 1.4826 * float(np.median(np.abs(v - m)))
    if not h > 0:
        return m, 0
    for j in range(1000):
        k = np.exp(-0.5 * ((v - m) / h) ** 2)
        mn = float(np.sum(k * v) / np.sum(k))
        if abs(mn - m) <= 1e-14 * abs(mn):
            return mn, j + 1
        m = mn
    return m, 1000


def _wmi(v, w):
    o = np.argsort(v, kind="mergesort")
    v, w = np.asarray(v, dtype=np.float64)[o], np.asarray(w, dtype=np.float64)[o]
    s = np.cumsum(w) - 0.5 * w
    return float(np.interp(0.5 * w.sum(), s, v))


def _tgm(r2, a):
    r2 = np.asarray(r2, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    if a.size < 30:
        f = max(float(np.median(r2 - a)), 1e-3 * float(np.min(a)))
        return 1.0, max(0.0, _wmi(r2 - a, 1.0 / (a + f) ** 2))
    b, fg, q = _groups(a)
    lam, _ = _ms(r2[fg] / a[fg])
    f = max(float(np.median(r2[q] - lam * a[q])), 1e-3 * float(np.min(lam * a[b])))
    t2 = max(0.0, _wmi(r2[b] - lam * a[b], 1.0 / (lam * a[b] + f) ** 2))
    lam, _ = _ms((r2[fg] - t2) / a[fg])
    f = max(t2, 1e-3 * float(np.min(lam * a[b])))
    t2 = max(0.0, _wmi(r2[b] - lam * a[b], 1.0 / (lam * a[b] + f) ** 2))
    return lam, t2


def _synthetic(n=370, seed=1, lam=1.03, tau2=1e-5, n_out=20):
    rng = np.random.default_rng(seed)
    a = 10.0 ** rng.uniform(-5.0, -3.0, n)
    r2 = (lam * a + tau2) * rng.chisquare(3025, n) / 3025.0
    r2[rng.choice(n, n_out, replace=False)] *= 6.0                  # some contaminated stars
    return r2, a


# ---------------------------------------------------------------------------
# The estimator itself (synthetic, not the evaluation seeds)
# ---------------------------------------------------------------------------

def test_tgm_follows_the_registered_definition():
    for seed in (1, 2, 3):
        r2, a = _synthetic(seed=seed)
        lam, t2, info = pm.tgm_floor(r2, a)
        want_lam, want_t2 = _tgm(r2, a)
        assert lam == pytest.approx(want_lam, rel=1e-14) and t2 == pytest.approx(want_t2, rel=1e-14, abs=0.0)
        assert info.get("n_B") == max(15, len(a) // 2) and info.get("n_F") == max(15, len(a) - len(a) // 2)
        assert len(info.get("ms_iters", ())) == 2 and all(0 < k < 1000 for k in info["ms_iters"])


def test_tgm_small_sample_falls_back_to_one_step():
    r2, a = _synthetic(n=20, seed=4, n_out=0)
    lam, t2, info = pm.tgm_floor(r2, a)
    want_lam, want_t2 = _tgm(r2, a)
    assert lam == 1.0 and want_lam == 1.0
    assert t2 == pytest.approx(want_t2, rel=1e-14, abs=0.0)
    assert info.get("fallback") is True


def test_tgm_is_scale_equivariant():
    r2, a = _synthetic(seed=5)
    lam0, t20, _ = pm.tgm_floor(r2, a)
    w0 = 1.0 / (lam0 * a + t20)
    for c in (0.5, 0.9, 1.1, 2.0):
        lam, t2, _ = pm.tgm_floor(r2, c * a)
        assert lam == pytest.approx(lam0 / c, rel=1e-12) and t2 == pytest.approx(t20, rel=1e-12)
        w = 1.0 / (lam * c * a + t2)
        q = (w / w.sum()) / (w0 / w0.sum())
        assert q.max() / q.min() - 1.0 <= 1e-12


def test_tau2_zero_warns(caplog):
    r2, a = _synthetic(seed=6, n_out=0)
    o = np.argsort(a, kind="mergesort")
    r2[o[:200]] = 0.5 * a[o[:200]]                                 # bright stars below lambda a
    with caplog.at_level(logging.WARNING, logger="tract7dt.fit.grid_gpu.psf_model"):
        lam, t2, info = pm.tgm_floor(r2, a)
    assert t2 == 0.0 and info.get("tau2_zero") is True
    assert any("tau^2 = 0" in r.getMessage() for r in caplog.records)
    caplog.clear()
    r2, a = _synthetic(seed=6)
    with caplog.at_level(logging.WARNING, logger="tract7dt.fit.grid_gpu.psf_model"):
        lam, t2, info = pm.tgm_floor(r2, a)
    assert t2 > 0.0 and info.get("tau2_zero") is False
    assert not any("tau^2 = 0" in r.getMessage() for r in caplog.records)


def _gap(v, m):
    """|MS limit - argmax of the same-kernel density on a 2,000-point grid over [min v, max v]| / h."""
    v = np.asarray(v, dtype=np.float64)
    h = 0.5 * 1.4826 * float(np.median(np.abs(v - np.median(v))))
    if not h > 0:
        return 0.0
    g = np.linspace(float(v.min()), float(v.max()), 2000)
    dens = np.exp(-0.5 * ((g[:, None] - v[None, :]) / h) ** 2).sum(axis=1)
    return abs(m - float(g[int(np.argmax(dens))])) / h


def test_ms_global_gap_follows_its_definition():
    for seed in (1, 9):
        r2, a = _synthetic(seed=seed)
        _, _, info = pm.tgm_floor(r2, a)
        b, fg, q = _groups(a)
        v1 = r2[fg] / a[fg]
        lam1, _ = _ms(v1)
        f = max(float(np.median(r2[q] - lam1 * a[q])), 1e-3 * float(np.min(lam1 * a[b])))
        t21 = max(0.0, _wmi(r2[b] - lam1 * a[b], 1.0 / (lam1 * a[b] + f) ** 2))
        v2 = (r2[fg] - t21) / a[fg]
        lam2, _ = _ms(v2)
        want = (_gap(v1, lam1), _gap(v2, lam2))
        got = info.get("ms_global_gap")
        assert got is not None and len(got) == 2
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# The model: weights 1 / (lambda a + tau^2), records, star_r2, (lambda, tau^2) reuse
# ---------------------------------------------------------------------------

def _small_build(tau2=None):
    import synth
    import test_psf_model as T
    rng = np.random.default_rng(19)
    st, x, y, a, _ = T.training_set(rng, 300, synth.PSFField(T.SHAPE))
    return pm.build_psf_model(st, x, y, a, T.SHAPE, render_refine=1, tau2=tau2), (st, x, y, a, T.SHAPE)


def test_model_records_the_estimate_and_reuses_it(tmp_path):
    m2, (st, x, y, a, shape) = _small_build()
    rec = m2.record
    assert all(k in rec for k in RECORD_KEYS), [k for k in RECORD_KEYS if k not in rec]
    r2 = getattr(m2, "star_r2", None)
    assert r2 is not None and len(r2) == len(a)
    lam, t2 = _tgm(r2, a)
    assert rec["lam"] == pytest.approx(lam, rel=1e-14) and rec["tau2"] == pytest.approx(t2, rel=1e-14, abs=0.0)
    np.testing.assert_allclose(m2.star_weight, 1.0 / (lam * a + t2), rtol=1e-14)
    # reuse: (lambda, tau^2) given -> one pass, that build's second pass
    m1 = pm.build_psf_model(st, x, y, a, shape, render_refine=1, tau2=(rec["lam"], rec["tau2"]))
    assert m1.record["passes"] == 1 and m2.record["passes"] == 2
    assert np.max(np.abs(m1.stamps - m2.stamps)) <= 1e-12
    # star_r2 in the side car
    d = pm.save_psf_model(tmp_path / "psf_model", m2)
    back = pm.load_psf_model(tmp_path / "psf_model", m2.image_id)
    np.testing.assert_array_equal(back.star_r2, m2.star_r2)
    assert back.record["lam"] == rec["lam"]


# ---------------------------------------------------------------------------
# Evaluation (prereg_a1b.md 2): frozen inputs, seeds 8901-8905
# ---------------------------------------------------------------------------

_MODELS: dict = {}


def _model(seed: int, c: float = 1.0):
    """The a.1b model of a seed's input 1 times c (cached)."""
    key = (int(seed), float(c))
    if key not in _MODELS:
        d = I.seed_inputs(seed)
        pop = d["pop"]
        _MODELS[key] = pm.build_psf_model(pop["stamps"], pop["fx"], pop["fy"], d["a1"] * c, pop["shape"])
    return _MODELS[key]


def _nominal(seed: int):
    """(model, r2, a, theta) of a seed; asserts the registered records first."""
    m = _model(seed)
    assert all(k in m.record for k in RECORD_KEYS), [k for k in RECORD_KEYS if k not in m.record]
    r2 = np.asarray(m.star_r2, dtype=np.float64)
    a = I.seed_inputs(seed)["a1"]
    return m, r2, a, m.record["tau2"] / m.record["lam"]


def _theta(r2, a):
    lam, t2, _ = pm.tgm_floor(r2, a)
    return t2 / lam


def _d(theta0, theta1, a1):
    return float(np.max(np.abs(theta1 - theta0) / (a1 + theta1)))


def _neff(w):
    w = np.asarray(w, dtype=np.float64)
    return float(w.sum() ** 2 / np.sum(w * w))


@pytest.mark.dev_record
def test_control_reproduces_the_golden_build():
    """Wiring check of the control (prereg 1.5): on seed 5 the old rule re-written here, applied to the a.1b build's
    star_r2, and the model built with tau2 = (1, tau2_old) equal the pre-change build stored as golden values."""
    import pathlib
    g = np.load(pathlib.Path(__file__).parent / "data" / "a1b_golden_seed5.npz")
    d = I.seed_inputs(5)
    assert str(g["population_hash"]) == d["population_hash"]
    m, r2, a, _ = _nominal(5)
    t2_old = I.tau2_old(r2, a)
    assert t2_old == pytest.approx(float(g["tau2_old"]), rel=1e-12)
    pop = d["pop"]
    ctl = pm.build_psf_model(pop["stamps"], pop["fx"], pop["fy"], a, pop["shape"], tau2=(1.0, t2_old))
    np.testing.assert_allclose(ctl.star_weight, g["star_weight"], rtol=1e-12)
    gx, gy = I.grid_xy(pop["shape"])
    st = ctl.render(gx, gy)
    assert np.max(np.abs(st - g["grid_stamps"])) <= 1e-12 * np.max(np.abs(g["grid_stamps"]))


@pytest.mark.dev_record
def test_t1_scale_invariance():
    m1, r2, a, th = _nominal(8901)
    pop = I.seed_inputs(8901)["pop"]
    gx, gy = I.grid_xy(pop["shape"])
    px, py = np.concatenate([gx, pop["fx"]]), np.concatenate([gy, pop["fy"]])
    s1 = m1.render(px, py)
    peak = float(np.max(np.abs(s1)))
    w1 = np.asarray(m1.star_weight, dtype=np.float64)
    for c in (0.9, 0.95, 1.05, 1.1):
        mc = _model(8901, c)
        wc = np.asarray(mc.star_weight, dtype=np.float64)
        q = (wc / wc.sum()) / (w1 / w1.sum())
        assert q.max() / q.min() - 1.0 <= 1e-12, (c, q.max() / q.min() - 1.0)
        assert np.max(np.abs(mc.render(px, py) - s1)) <= 1e-12 * peak, c
    # positive control: the old rule's D(a x 0.9) >= 0.5 (same r2, estimator only)
    t_old, t_old9 = I.tau2_old(r2, a), I.tau2_old(r2, 0.9 * a)
    assert _d(t_old, t_old9, 0.9 * a) >= 0.5


@pytest.mark.dev_record
@pytest.mark.parametrize("seed", I.EVAL_SEEDS)
def test_t2a_nonuniform_error(seed):
    m, r2, a, th = _nominal(seed)
    rng = np.random.default_rng(8910 + (seed - 8900))
    ds, first = [], None
    for k in range(100):
        ap = a * np.exp(rng.normal(0.0, 0.1, a.size))
        if first is None:
            first = ap
        ds.append(_d(th, _theta(r2, ap), ap))
    assert max(ds) <= 0.25, max(ds)
    pop = I.seed_inputs(seed)["pop"]
    mp = pm.build_psf_model(pop["stamps"], pop["fx"], pop["fy"], first, pop["shape"])
    ratio = _neff(mp.star_weight) / _neff(m.star_weight)
    assert 0.9 <= ratio <= 1.1, ratio


def _sky_poisson(seed):
    d = I.seed_inputs(seed)
    info, s = d["info"], d["pop"]["F"]
    sky = 3025.0 * np.asarray(info["sigma2_sky"], float) * (1.0 + float(info["kappa"]) * float(info["phi_norm2"])) / s ** 2
    return sky, 1.0 / (50.0 * s)


@pytest.mark.dev_record
@pytest.mark.parametrize("seed", I.EVAL_SEEDS)
def test_t2b_sky_term(seed):
    m, r2, a, th = _nominal(seed)
    sky, _ = _sky_poisson(seed)
    ap = a + 0.1 * sky
    th1 = _theta(r2, ap)
    assert 0.67 <= th1 / th <= 1.5, th1 / th
    assert _d(th, th1, ap) <= 0.1


@pytest.mark.dev_record
@pytest.mark.parametrize("seed", I.EVAL_SEEDS)
def test_t2c_warning_if_and_only_if_the_floor_collapses(seed, caplog):
    m, r2, a, th = _nominal(seed)
    b, _, _ = _groups(a)
    o = np.argsort(a, kind="mergesort")
    _, poi = _sky_poisson(seed)
    cases = {}
    a_i = a.copy(); a_i[b] *= 1.1; cases["i"] = (r2, a_i)
    a_ii = a.copy(); a_ii[b] *= 1.3; cases["ii"] = (r2, a_ii)
    r2_iii = r2.copy(); r2_iii[o[:15]] = 0.9 * m.record["lam"] * a[o[:15]]; cases["iii"] = (r2_iii, a)
    cases["iv"] = (r2, a + 9.0 * poi)
    for name, (rr, aa) in cases.items():
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="tract7dt.fit.grid_gpu.psf_model"):
            lam, t2, info = pm.tgm_floor(rr, aa)
        warned = any("tau^2 = 0" in r.getMessage() for r in caplog.records)
        assert (t2 == 0.0) == bool(info.get("tau2_zero")) == warned, (name, t2, info.get("tau2_zero"), warned)


# Round 1 FAILED T3 (a) on 8902; kept as the record (prereg_a1b_r2.md 5): strict, so it breaks if it ever passes.
_T3_PARAMS = [pytest.param(s, marks=pytest.mark.xfail(
    reason="development record, round 1: FAILED registered T3(a) on 8902 (D 0.268 > 0.2) with the frozen "
           "development input; not a re-test -- the re-test is test_prereg_a1b_t3_r3d (seeds 9401-9405)",
    strict=True, raises=AssertionError))
              if s == 8902 else s for s in I.EVAL_SEEDS]


@pytest.mark.dev_record
@pytest.mark.parametrize("seed", _T3_PARAMS)
def test_t3_contamination(seed):
    m, r2, a, th = _nominal(seed)
    _, _, q = _groups(a)
    allp = np.arange(a.size)
    out = {}
    for name, pool, frac, mult in (("all20x10", allp, 0.2, 10.0), ("Q40x10", q, 0.4, 10.0), ("all40x2", allp, 0.4, 2.0)):
        r2c = r2.copy()
        pick = np.random.default_rng(8921).choice(pool, int(round(frac * len(pool))), replace=False)
        r2c[pick] *= mult
        out[name] = _d(th, _theta(r2c, a), a)
    assert out["all20x10"] <= 0.2, out
    assert out["Q40x10"] <= 0.35, out


_T5: dict = {}


def _t5_seed(seed):
    if seed not in _T5:
        tr = I.truth()
        d = I.seed_inputs(seed)
        pop, sc = d["pop"], d["sc"]
        res = {}
        for c in (0.9, 1.0, 1.1):
            mc = _model(seed, c)
            assert all(k in mc.record for k in RECORD_KEYS)
            t_old = I.tau2_old(mc.star_r2, d["a1"] * c)
            ctl = pm.build_psf_model(pop["stamps"], pop["fx"], pop["fy"], d["a1"] * c, pop["shape"], tau2=(1.0, t_old))
            res[c] = (tr.peak_stats(mc, sc, pop)["peak_err"], tr.peak_stats(ctl, sc, pop)["peak_err"])
        _T5[seed] = res
    return _T5[seed]


@pytest.mark.dev_record
@pytest.mark.parametrize("seed", I.EVAL_SEEDS)
def test_t5a_stability(seed):
    res = _t5_seed(seed)
    worst_new = max(v[0] for v in res.values())
    worst_ctl = max(v[1] for v in res.values())
    assert worst_new <= worst_ctl, (worst_new, worst_ctl)


@pytest.mark.dev_record
def test_t5b_nominal_noninferiority():
    diffs = [(_t5_seed(s)[1.0][0] - _t5_seed(s)[1.0][1]) for s in I.EVAL_SEEDS]
    assert float(np.mean(diffs)) <= 0.002, diffs
