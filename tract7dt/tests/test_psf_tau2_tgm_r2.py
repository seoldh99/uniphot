"""M2 unit a.1b round 2: the lambda identification check of TGM and the round-2 evaluation.

Registration: development record prereg_a1b_r2.md (not distributed; committed before this file).
Design: a1b_fix_answer.md 2 (identification check), lead decision impl_plan.md 10-10 19:21 (judgment input = the
engine a).  Round 1 (prereg_a1b.md, seeds 8901-8905) stays in test_psf_tau2_tgm.py; its helpers are reused here.
The identification check is written here independently (`_tgm_r2`).  T3 is deferred until a.3 merges (prereg 2.2).
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

import a1b_inputs as I
import test_psf_tau2_tgm as R1
from tract7dt.fit.grid_gpu import psf_model as pm

EVAL_SEEDS_R2 = (9001, 9002, 9003, 9004, 9005)
R2_KEYS = ("lambda_identified", "lam_hat", "F_median_r2_over_a")
IDENT_TEXT = "not identified"            # key phrase of the identification warning (prereg_a1b_r2 1-5)
LOGGER = "tract7dt.fit.grid_gpu.psf_model"


# ---------------------------------------------------------------------------
# The registered addition (prereg_a1b_r2.md 1), written here
# ---------------------------------------------------------------------------

def _tgm_r2(r2, a) -> dict:
    """Round-1 TGM, then the identification check lambda_2 > 0 and lambda_2 median_F(a) >= tau2_2; otherwise
    lambda = 1 and one step on B / Q.  n < 30: the round-1 one-step path, no check."""
    r2 = np.asarray(r2, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    if a.size < 30:
        lam, t2 = R1._tgm(r2, a)
        return dict(lam=lam, t2=t2, identified=None, lam_hat=None, ratio=None, t2_step2=None)
    lam2, t22 = R1._tgm(r2, a)
    b, fg, q = R1._groups(a)
    ratio = float(np.median(r2[fg] / a[fg]))
    if lam2 > 0.0 and lam2 * float(np.median(a[fg])) >= t22:
        return dict(lam=lam2, t2=t22, identified=True, lam_hat=lam2, ratio=ratio, t2_step2=t22)
    f = max(float(np.median(r2[q] - a[q])), 1e-3 * float(np.min(a[b])))
    t2 = max(0.0, R1._wmi(r2[b] - a[b], 1.0 / (a[b] + f) ** 2))
    return dict(lam=1.0, t2=t2, identified=False, lam_hat=lam2, ratio=ratio, t2_step2=t22)


def _require_r2():
    """Capability check BEFORE any evaluation-seed work: the round-2 records exist."""
    r2, a = R1._synthetic(seed=1)
    info = pm.tgm_floor(r2, a)[2]
    assert all(k in info for k in R2_KEYS), [k for k in R2_KEYS if k not in info]


def _synth_negative_lambda():
    """Residuals independent of a and slightly lower for the faint stars: lambda_2 < 0."""
    rng = np.random.default_rng(31)
    n = 300
    a = 10.0 ** rng.uniform(-3.0, 0.0, n)
    rank = np.argsort(np.argsort(a))
    r2 = 1e-6 * (1.0 - 0.2 * rank / n) * (1.0 + 0.01 * rng.normal(size=n))
    return r2, a


def _synth_floor_dominated():
    """A floor (0.1) far above lambda a for every star: lambda_2 > 0 but lambda_2 median_F(a) < tau2_2."""
    rng = np.random.default_rng(32)
    n = 300
    a = 10.0 ** rng.uniform(-5.0, -3.0, n)
    r2 = (a + 1e-1) * rng.chisquare(3025, n) / 3025.0
    return r2, a


def _ident_messages(caplog):
    return [r.getMessage() for r in caplog.records if IDENT_TEXT in r.getMessage()]


# ---------------------------------------------------------------------------
# The definition (synthetic, not the evaluation seeds)
# ---------------------------------------------------------------------------

def test_r2_identified_output_equals_round1():
    for seed in (1, 2, 3):
        r2, a = R1._synthetic(seed=seed)
        want = _tgm_r2(r2, a)
        assert want["identified"] is True                         # the input is identified (test has power below)
        lam, t2, info = pm.tgm_floor(r2, a)
        assert info.get("lambda_identified") is True
        lam1, t21 = R1._tgm(r2, a)
        assert lam == pytest.approx(lam1, rel=1e-14) and t2 == pytest.approx(t21, rel=1e-14, abs=0.0)
        assert info.get("lam_hat") == pytest.approx(lam1, rel=1e-14)
        assert info.get("F_median_r2_over_a") == pytest.approx(want["ratio"], rel=1e-14)


@pytest.mark.parametrize("case", ("negative_lambda", "floor_dominated"))
def test_r2_unidentified_falls_back_to_lambda_one(case, caplog):
    r2, a = _synth_negative_lambda() if case == "negative_lambda" else _synth_floor_dominated()
    want = _tgm_r2(r2, a)
    _, fg, _ = R1._groups(a)
    if case == "negative_lambda":
        assert want["lam_hat"] <= 0.0                              # the case is what it says
    else:
        assert want["lam_hat"] > 0.0 and want["lam_hat"] * float(np.median(a[fg])) < want["t2_step2"]
    assert want["identified"] is False
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        lam, t2, info = pm.tgm_floor(r2, a)
    assert lam == 1.0, lam
    assert t2 == pytest.approx(want["t2"], rel=1e-14, abs=0.0)
    assert info.get("lambda_identified") is False
    assert info.get("lam_hat") == pytest.approx(want["lam_hat"], rel=1e-14)
    assert info.get("F_median_r2_over_a") == pytest.approx(want["ratio"], rel=1e-14)
    msgs = _ident_messages(caplog)
    assert len(msgs) == 1 and "tau^2 = 0" not in msgs[0], msgs
    assert (t2 == 0.0) == bool(info.get("tau2_zero"))


def test_r2_small_sample_records_none(caplog):
    r2, a = R1._synthetic(n=20, seed=4, n_out=0)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        lam, t2, info = pm.tgm_floor(r2, a)
    assert all(k in info for k in R2_KEYS), info
    assert info["lambda_identified"] is None and info["lam_hat"] is None and info["F_median_r2_over_a"] is None
    assert lam == 1.0 and not _ident_messages(caplog)


def test_r2_model_records_follow_the_definition():
    m, (st, x, y, a, shape) = R1._small_build()
    rec = m.record
    assert all(k in rec for k in R2_KEYS), [k for k in R2_KEYS if k not in rec]
    want = _tgm_r2(np.asarray(m.star_r2), a)
    assert rec["lambda_identified"] is want["identified"]
    assert rec["lam"] == pytest.approx(want["lam"], rel=1e-14)
    assert rec["tau2"] == pytest.approx(want["t2"], rel=1e-14, abs=0.0)
    assert rec["lam_hat"] == pytest.approx(want["lam_hat"], rel=1e-14)
    np.testing.assert_allclose(m.star_weight, 1.0 / (want["lam"] * a + want["t2"]), rtol=1e-14)


# ---------------------------------------------------------------------------
# Evaluation (prereg_a1b_r2.md 4): seeds 9001-9005, judgment input 1 (the engine a)
# ---------------------------------------------------------------------------

def test_r2_seeds_are_not_round_one_seeds():
    assert not set(EVAL_SEEDS_R2) & set(I.EVAL_SEEDS)


@pytest.mark.dev_record
def test_r2_t1_scale_invariance():
    _require_r2()
    seed = 9001
    m1, r2, a, th = R1._nominal(seed)
    models = {c: R1._model(seed, c) for c in (0.9, 0.95, 1.05, 1.1)}
    assert m1.record["lambda_identified"] is True
    for c, mc in models.items():
        assert mc.record["lambda_identified"] is True, c
    pop = I.seed_inputs(seed)["pop"]
    gx, gy = I.grid_xy(pop["shape"])
    px, py = np.concatenate([gx, pop["fx"]]), np.concatenate([gy, pop["fy"]])
    s1 = m1.render(px, py)
    peak = float(np.max(np.abs(s1)))
    w1 = np.asarray(m1.star_weight, dtype=np.float64)
    for c, mc in models.items():
        wc = np.asarray(mc.star_weight, dtype=np.float64)
        q = (wc / wc.sum()) / (w1 / w1.sum())
        assert q.max() / q.min() - 1.0 <= 1e-12, (c, q.max() / q.min() - 1.0)
        assert np.max(np.abs(mc.render(px, py) - s1)) <= 1e-12 * peak, c
    t_old, t_old9 = I.tau2_old(r2, a), I.tau2_old(r2, 0.9 * a)
    assert R1._d(t_old, t_old9, 0.9 * a) >= 0.5


_T2A: dict = {}


def _t2a_seed(seed):
    """T2a of one seed (prereg_a1b 1.3 with rng default_rng(9010 + i)); cached."""
    if seed not in _T2A:
        m, r2, a, th = R1._nominal(seed)
        rng = np.random.default_rng(9010 + (seed - 9000))
        ds, first = [], None
        for _ in range(100):
            ap = a * np.exp(rng.normal(0.0, 0.1, a.size))
            if first is None:
                first = ap
            ds.append(R1._d(th, R1._theta(r2, ap), ap))
        pop = I.seed_inputs(seed)["pop"]
        mp = pm.build_psf_model(pop["stamps"], pop["fx"], pop["fy"], first, pop["shape"])
        _T2A[seed] = dict(D=ds, rebuild=mp, neff_ratio=R1._neff(mp.star_weight) / R1._neff(m.star_weight))
    return _T2A[seed]


@pytest.mark.dev_record
@pytest.mark.parametrize("seed", EVAL_SEEDS_R2)
def test_r2_t2a_nonuniform_error(seed):
    _require_r2()
    r = _t2a_seed(seed)
    assert max(r["D"]) <= 0.25, max(r["D"])
    assert 0.9 <= r["neff_ratio"] <= 1.1, r["neff_ratio"]


@pytest.mark.dev_record
@pytest.mark.parametrize("seed", EVAL_SEEDS_R2)
def test_r2_t2b_sky_term(seed):
    _require_r2()
    m, r2, a, th = R1._nominal(seed)
    sky, _ = R1._sky_poisson(seed)
    ap = a + 0.1 * sky
    th1 = R1._theta(r2, ap)
    assert 0.67 <= th1 / th <= 1.5, th1 / th
    assert R1._d(th, th1, ap) <= 0.1


@pytest.mark.dev_record
@pytest.mark.parametrize("seed", EVAL_SEEDS_R2)
def test_r2_t2c_warning_if_and_only_if_the_floor_collapses(seed, caplog):
    _require_r2()
    m, r2, a, th = R1._nominal(seed)
    b, _, _ = R1._groups(a)
    o = np.argsort(a, kind="mergesort")
    _, poi = R1._sky_poisson(seed)
    cases = {}
    a_i = a.copy(); a_i[b] *= 1.1; cases["i"] = (r2, a_i)
    a_ii = a.copy(); a_ii[b] *= 1.3; cases["ii"] = (r2, a_ii)
    r2_iii = r2.copy(); r2_iii[o[:15]] = 0.9 * m.record["lam"] * a[o[:15]]; cases["iii"] = (r2_iii, a)
    cases["iv"] = (r2, a + 9.0 * poi)
    for name, (rr, aa) in cases.items():
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            lam, t2, info = pm.tgm_floor(rr, aa)
        warned = any("tau^2 = 0" in r.getMessage() for r in caplog.records)
        assert (t2 == 0.0) == bool(info.get("tau2_zero")) == warned, (name, t2, info.get("tau2_zero"), warned)


@pytest.mark.dev_record
@pytest.mark.parametrize("seed", EVAL_SEEDS_R2)
def test_r2_t5a_stability(seed):
    _require_r2()
    res = R1._t5_seed(seed)
    worst_new = max(v[0] for v in res.values())
    worst_ctl = max(v[1] for v in res.values())
    assert worst_new <= worst_ctl, (worst_new, worst_ctl)


@pytest.mark.dev_record
def test_r2_t5b_nominal_noninferiority():
    _require_r2()
    diffs = [(R1._t5_seed(s)[1.0][0] - R1._t5_seed(s)[1.0][1]) for s in EVAL_SEEDS_R2]
    assert float(np.mean(diffs)) <= 0.002, diffs


# ---------------------------------------------------------------------------
# T7: identification wiring
# ---------------------------------------------------------------------------

def _noiseless_span_scene():
    """The quadratic field of test_psf_model.test_fields_in_the_span_are_reproduced_exactly, unchanged."""
    rng = np.random.default_rng(5)
    h, w, s, n = 660, 1000, 9, 300
    shape = (h, w)
    x, y = rng.uniform(0, w - 1, n), rng.uniform(0, h - 1, n)
    a = rng.uniform(0.5, 1.5, (s, s))
    a /= a.sum()
    zs = [z - z.mean() for z in rng.normal(size=(3, s, s)) * 0.01]
    a_noise = 1.0 / rng.uniform(1, 100, n)

    def quad(px, py):
        return (a[None] + zs[0][None] * (px / w)[:, None, None]
                + zs[1][None] * ((py / h) ** 2)[:, None, None]
                + zs[2][None] * (px * py / (w * h))[:, None, None])

    return quad(x, y), x, y, a_noise, shape


_T7A: dict = {}


class _ListHandler(logging.Handler):
    def __init__(self, out: list):
        super().__init__(level=logging.WARNING)
        self.out = out

    def emit(self, record):
        self.out.append(record.getMessage())


def _t7a_fits():
    """The six fits of T7 (a): {(k, p): (mixture via the estimator, the identification warnings of that call,
    mixture with the weights fixed at 1/a)}; cached."""
    if _T7A:
        return _T7A
    st, x, y, a, shape = _noiseless_span_scene()
    d = pm.image_diagonal(shape)
    for k in (2.0, 4.0, 8.0):
        for p in (1, 2):
            cand = [pm.Candidate(d / k, p)]
            logger = logging.getLogger(LOGGER)
            recs: list = []
            handler = _ListHandler(recs)
            logger.addHandler(handler)
            try:
                mix, _ = pm.fit_mixture(st, x, y, a, shape, candidates=cand)
            finally:
                logger.removeHandler(handler)
            fixed, _ = pm.fit_mixture(st, x, y, a, shape, candidates=cand, tau2=(1.0, 0.0))
            _T7A[(k, p)] = (mix, [m for m in recs if IDENT_TEXT in m], fixed)
    return _T7A


def test_r2_t7a_noiseless_scene_falls_back():
    _require_r2()
    st, x, y, a, shape = _noiseless_span_scene()
    xt, yt = np.linspace(10, shape[1] - 10, 50), np.linspace(10, shape[0] - 10, 50)
    for key, (mix, msgs, fixed) in _t7a_fits().items():
        rec = mix.record
        assert rec["lambda_identified"] is False, key
        assert len(msgs) == 1, (key, msgs)
        assert rec["tau2"] == 0.0 and rec["lam"] == 1.0, (key, rec["lam"], rec["tau2"])
        w = 1.0 / (rec["lam"] * a + rec["tau2"])
        np.testing.assert_allclose(w, 1.0 / a, rtol=1e-12)
        ev, ef = mix.evaluate(xt, yt), fixed.evaluate(xt, yt)        # the weights actually used are 1 / a
        assert np.max(np.abs(ev - ef)) <= 1e-12 * np.max(np.abs(ef)), key


@pytest.mark.dev_record
@pytest.mark.parametrize("seed", EVAL_SEEDS_R2)
def test_r2_t7b_evaluation_seeds_are_identified(seed):
    _require_r2()
    assert R1._model(seed).record["lambda_identified"] is True


@pytest.mark.dev_record
def test_r2_t7c_every_weight_is_positive():
    _require_r2()
    st, x, y, a, shape = _noiseless_span_scene()
    for key, (mix, _, _) in _t7a_fits().items():
        rec = mix.record
        assert np.min(1.0 / (rec["lam"] * a + rec["tau2"])) > 0.0, key
    models = [R1._model(9001, c) for c in (0.9, 0.95, 1.0, 1.05, 1.1)]
    models += [_t2a_seed(s)["rebuild"] for s in EVAL_SEEDS_R2]
    models += [R1._model(s, c) for s in EVAL_SEEDS_R2 for c in (0.9, 1.0, 1.1)]
    for m in models:
        assert np.min(m.star_weight) > 0.0 and np.all(np.isfinite(m.star_weight))
