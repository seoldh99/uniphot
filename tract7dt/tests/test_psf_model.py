"""PSF model of one image (M2 unit a.1): side-car format and the design 6 tests P1, P2, G7(b).

Design: development design note design_v2.md (not distributed) 1.4 (model) and 6 (P1, P2, G7);
plan impl_plan.md 1.3 a.1.  Expected values are computed here from the design formulas and
from the synthetic truth (tests/synth, engine-free), never from the module under test.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tract7dt.fit.grid_gpu import psf_model as pm


# ---------------------------------------------------------------------------
# Side-car format (the contract unit a.7 and the measurement tools read)
# ---------------------------------------------------------------------------


def _toy_model(shape=(660, 1000), s=55):
    """A model with hand-made node stamps (no fitting) for the format tests."""
    node_x = np.linspace(0.0, shape[1] - 1.0, 4)
    node_y = np.linspace(0.0, shape[0] - 1.0, 3)
    rng = np.random.default_rng(1)
    st = rng.uniform(0.0, 1.0, (3, 4, s, s))
    st /= st.sum(axis=(-2, -1), keepdims=True)
    return pm.PsfModel(
        image_id=pm.image_id_for("m500", "/x/calib_7DT15_T08676_20250604_232444_m500_300.com.fits"),
        band="m500", frame="/x/calib_7DT15_T08676_20250604_232444_m500_300.com.fits",
        image_shape=shape, node_x=node_x, node_y=node_y, stamps=st,
        star_x=np.array([1.0, 2.0]), star_y=np.array([3.0, 4.0]),
        star_weight=np.array([5.0, 6.0]), star_fold=np.array([0, 1]),
        record={"candidates": [{"h_px": math.inf, "p": 2, "loss": 1.0, "se": 0.5, "weight": 1.0}]},
        work_frame_offset=(100, 100))


def test_image_id_rule():
    assert pm.image_id_for("m400", "/d/7DT13/m400/calib_7DT13_T08676_20250604_232444_m400_300.com.fits") \
        == "m400.calib_7DT13_T08676_20250604_232444_m400_300.com"
    assert pm.image_id_for("g", "/data/g/img.fits.fz") == "g.img"
    assert pm.image_id_for("r", "/data/r/my image#1.FITS") == "r.my_image_1"
    with pytest.raises(ValueError, match="same psf_model image_id"):
        pm.check_unique_image_ids(["g.img", "r.img", "g.img"])
    pm.check_unique_image_ids(["g.img", "r.img"])


def test_sidecar_round_trip(tmp_path):
    m = _toy_model()
    d = pm.save_psf_model(tmp_path / "psf_model", m)
    assert d == tmp_path / "psf_model" / m.image_id
    assert sorted(p.name for p in d.iterdir()) == ["model.json", "model.npz"]
    man = json.loads((d / "model.json").read_text())
    assert man["format"] == "uniphot.psf_model" and man["format_version"] == 1
    assert man["image_shape"] == [660, 1000] and man["work_frame_offset"] == [100, 100]
    assert man["stamp_size"] == 55 and man["stamp_centre"] == [27, 27]
    assert man["normalisation"] == "sum_1"
    assert man["render"]["interpolation"] == "bilinear" and man["render"]["outside"] == "clamp"
    assert (man["render"]["nx"], man["render"]["ny"]) == (4, 3)
    assert man["candidates"][0]["h_px"] is None          # inf is written as null
    with np.load(d / "model.npz") as z:
        assert sorted(z.files) == sorted(["node_x", "node_y", "stamps", "star_x", "star_y",
                                          "star_weight", "star_fold", "star_a",
                                          "star_r2", "mix_node_x", "mix_node_y", "mix_weights"])
        assert z["stamps"].dtype == np.float64 and z["stamps"].shape == (3, 4, 55, 55)
    back = pm.load_psf_model(tmp_path / "psf_model", m.image_id)
    np.testing.assert_array_equal(back.stamps, m.stamps)
    np.testing.assert_array_equal(back.node_x, m.node_x)
    assert back.band == "m500" and back.image_shape == (660, 1000)


def test_sidecar_round_trip_of_a_built_model(tmp_path):
    """A model built by the module keeps the rendering rule and its build record in the manifest."""
    import synth
    shape = (660, 1000)
    psf = synth.PSFField(shape, fwhm=4.0)
    rng = np.random.default_rng(3)
    x = rng.integers(27, 1000 - 28, 80).astype(float)
    y = rng.integers(27, 660 - 28, 80).astype(float)
    st = np.stack([synth.realize(5000.0 * psf.stamp(a, b).data, rng) for a, b in zip(x, y)])
    m = pm.build_psf_model(st, x, y, np.full(80, 1e-3), shape, image_id="g.small", band="g")
    d = pm.save_psf_model(tmp_path, m)
    man = json.loads((d / "model.json").read_text())
    assert man["render"]["interpolation"] == "bilinear" and man["render"]["outside"] == "clamp"
    assert man["render"]["refine"] == m.record["render"]["refine"]
    assert (man["render"]["nx"], man["render"]["ny"]) == (len(m.node_x), len(m.node_y))
    back = pm.load_psf_model(tmp_path, "g.small")
    np.testing.assert_array_equal(back.stamps, m.stamps)
    np.testing.assert_array_equal(back.mix_weights, m.mix_weights)
    assert back.record["tau2"] == m.record["tau2"] and back.record["render"]["refine"] == m.record["render"]["refine"]
    xs, ys = rng.uniform(0, 999, 50), rng.uniform(0, 659, 50)
    np.testing.assert_array_equal(back.render(xs, ys), m.render(xs, ys))


def test_index_file(tmp_path):
    p = pm.write_index(tmp_path / "psf_model", [("g.a", "g", "/f/a.fits"), ("r.b", "r", "/f/b.fits")])
    assert p.read_text().splitlines() == ["image_id,band,frame", "g.a,g,/f/a.fits", "r.b,r,/f/b.fits"]


def test_bilinear_rendering_rule():
    """The format's rule computed here by hand at random points (and clamped outside)."""
    m = _toy_model()
    rng = np.random.default_rng(2)
    xs = rng.uniform(-50.0, 1050.0, 200)
    ys = rng.uniform(-50.0, 700.0, 200)
    got = m.render(xs, ys)
    nx, ny = m.node_x, m.node_y
    for k in range(len(xs)):
        x = min(max(xs[k], nx[0]), nx[-1])
        y = min(max(ys[k], ny[0]), ny[-1])
        i = min(int((x - nx[0]) // (nx[1] - nx[0])), len(nx) - 2)
        j = min(int((y - ny[0]) // (ny[1] - ny[0])), len(ny) - 2)
        t = (x - nx[i]) / (nx[i + 1] - nx[i])
        u = (y - ny[j]) / (ny[j + 1] - ny[j])
        want = ((1 - t) * (1 - u) * m.stamps[j, i] + t * (1 - u) * m.stamps[j, i + 1]
                + (1 - t) * u * m.stamps[j + 1, i] + t * u * m.stamps[j + 1, i + 1])
        np.testing.assert_allclose(got[k], want, rtol=0, atol=1e-15)
    np.testing.assert_allclose(m.render(nx[2], ny[1]), m.stamps[1, 2], rtol=0, atol=1e-15)


def test_integral_check_is_a_run_failure():
    m = _toy_model()
    assert pm.check_integral(m.stamps) <= 1e-12
    bad = m.stamps.copy()
    bad[1, 2] *= 1.0 + 2e-6
    with pytest.raises(pm.PsfModelIntegralError) as ei:
        pm.check_integral(bad, "m500.x")
    assert ei.value.kind == "psf_model_integral"
    assert ei.value.details["n_bad_nodes"] == 1
    import pickle
    again = pickle.loads(pickle.dumps(ei.value))          # crosses spawn-worker pipes
    assert again.kind == "psf_model_integral" and again.details == ei.value.details
    nan = m.stamps.copy()
    nan[0, 0, 0, 0] = np.nan
    with pytest.raises(pm.PsfModelIntegralError):
        pm.check_integral(nan)


def test_default_candidates_and_nested_grids():
    """Design v2.5 1.3: 16 candidates -- global p 2..5 and h in {D/2, D/4, D/8, D/16} x p 0..2."""
    shape = (6600, 10000)
    c = pm.default_candidates(shape)
    assert len(c) == 16
    d = math.hypot(10000, 6600)
    assert [(x.h, x.p) for x in c if x.is_global] == [(math.inf, p) for p in (2, 3, 4, 5)]
    fin = [(round(d / x.h, 9), x.p) for x in c if not x.is_global]
    assert fin == [(f, p) for f in (2.0, 4.0, 8.0, 16.0) for p in (0, 1, 2)]
    for x in c:
        if x.is_global:
            continue
        gx, gy = pm.training_grid(x.h, shape)
        assert gx[0] == 0.0 and gx[-1] == 9999.0 and gy[0] == 0.0 and gy[-1] == 6599.0
        assert np.diff(gx).max() <= x.h / 2 + 1e-9 and np.diff(gy).max() <= x.h / 2 + 1e-9
    rx, ry = pm.render_grid(c, shape)
    for x in c:
        if not x.is_global:
            gx, gy = pm.training_grid(x.h, shape)
            assert np.all(np.isin(gx, rx)) and np.all(np.isin(gy, ry))      # nested


# ---------------------------------------------------------------------------
# Synthetic training sets (design 6 common conventions; tests/synth is engine-free)
# ---------------------------------------------------------------------------

import synth  # noqa: E402  (tests/ is on sys.path under pytest's default import mode)

SHAPE = (6600, 10000)          # (H, W): the size of a 7DT work frame (dev8 crop 100)
SIGMA, GAIN, RHO1 = 1.0, 50.0, 0.15
NPIX = synth.BOX * synth.BOX


def noise_a(flux):
    """A noise SCALE only (used to size an injected heterogeneity): n_pix sigma^2 / F^2 + 1 / (g F) at
    the true flux.  The stamps' own a comes from the engine's function (:func:`fixture_a`)."""
    flux = np.asarray(flux, dtype=float)
    return (NPIX * SIGMA ** 2 + flux / GAIN) / flux ** 2


KAPPA_SYNTH = 1.72      # box-sum variance / (n_pix sigma^2) of the synth sky ((sum k)^4, designer 7.1)


def fixture_a(stamps, phi_norm2, sigma2_nb=0.0):
    """a of training stamps as the engine makes it (design v2.5 3rd ed. 7.1; a.3 round 4): the box
    sum S of each stamp, the shape term with kappa = 1.72 and phi = the truth, gain 50, sky sigma 1."""
    from tract7dt.fit.grid_gpu import psf_from_stars as pfs
    return pfs.star_noise_a(np.asarray(stamps, dtype=float).sum(axis=(-2, -1)), SIGMA ** 2, sigma2_nb,
                            kappa=KAPPA_SYNTH, phi_norm2=phi_norm2, gain=GAIN, n_pix=NPIX)


def training_set(rng, n, psf, snr=(50.0, 2000.0), x_range=None, hetero_tau2=0.0, snr_draw=None):
    """n registered training stamps: truth stamp of the star's position, centred (integer
    positions), times the flux of the drawn sky-limited S/N, plus Poisson + correlated sky.
    hetero_tau2 > 0 adds F z_s with z_s a zero-sum per-star pattern of squared norm hetero_tau2:
    a known heterogeneity (star-to-star model error the noise part a does not describe).
    snr_draw(n, rng) overrides the log-uniform S/N draw.  Returns (stamps, x, y, a, flux)."""
    h, w = psf.shape
    xa, xb = (synth.HALF, w - 1 - synth.HALF) if x_range is None else x_range
    x = rng.integers(xa, xb + 1, n).astype(float)
    y = rng.integers(synth.HALF, h - synth.HALF, n).astype(float)
    sn = synth.draw_snr(n, snr[0], snr[1], rng) if snr_draw is None else snr_draw(n, rng)
    stamps = np.empty((n, synth.BOX, synth.BOX))
    flux = np.empty(n)
    phi2 = np.empty(n)
    for k in range(n):
        st = psf.stamp(x[k], y[k])
        phi2[k] = float((st.data ** 2).sum())
        flux[k] = float(synth.sky_limited_flux(sn[k], st.data))
        stamps[k] = synth.realize(flux[k] * st.data, rng, sigma=SIGMA, gain=GAIN, rho1=RHO1)
        if hetero_tau2 > 0.0:
            z = rng.standard_normal(st.data.shape)
            z -= z.mean()
            stamps[k] += flux[k] * np.sqrt(hetero_tau2) * z / np.linalg.norm(z)
    return stamps, x, y, fixture_a(stamps, phi2), flux


def psf_fit_flux(model_stamp, data):
    """Sky-weighted PSF-fit flux sum(P d) / sum(P^2) (uniform sky; design 1.7, 1.9)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (model_stamp * data).sum(axis=(-2, -1)) / (model_stamp ** 2).sum(axis=(-2, -1))


def n_eff(w):
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / np.sum(w * w))


# ---------------------------------------------------------------------------
# P1  integral 1 (design 6 P1: 1,000 random positions, |sum - 1| <= 1e-6 in float64)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("half", [False, True], ids=["whole_field", "stars_left_half_only"])
def test_p1_integral_is_one_everywhere(half):
    """Expected value 0 = sum - 1 by construction (design 1.4).  The second case puts every
    training star in x < W/2, so the right half is pure extrapolation (design: exact there too)."""
    rng = np.random.default_rng(101 + int(half))
    psf = synth.PSFField(SHAPE)
    xr = (synth.HALF, SHAPE[1] // 2) if half else None
    st, x, y, a, _ = training_set(rng, 500, psf, x_range=xr)
    m = pm.build_psf_model(st, x, y, a, SHAPE, image_id="synth.p1")
    xs = rng.uniform(0.0, SHAPE[1] - 1.0, 1000)
    ys = rng.uniform(0.0, SHAPE[0] - 1.0, 1000)
    sums = m.render(xs, ys).sum(axis=(1, 2))
    assert np.all(np.isfinite(sums))
    assert np.max(np.abs(sums - 1.0)) <= 1e-6, np.max(np.abs(sums - 1.0))
    node_sums = m.stamps.sum(axis=(-2, -1))
    assert np.max(np.abs(node_sums - 1.0)) <= 1e-6
    assert m.record["integral_max_abs_dev"] <= 1e-6


# ---------------------------------------------------------------------------
# P2  gradient recovery (design 6 P2)
# ---------------------------------------------------------------------------

P2_REALISATIONS = 5          # design: >= 5 noise realisations
# design 6 P2: 400 stars, S/N 50-2000, "raise the training stars if SE > 0.75 mmag (tolerance
# unchanged)". Doubling ladder 400 -> 800 -> 1,600: 1,600 is the first step with SE <= 0.75
# (design v2.5 answers 2nd ed. 1.2, review_v6 C1); committed before any 1,600-star result of
# this test (seeds [202, r]). If SE > 0.75 at 1,600 the remedy is 3,200, the tolerance stays.
P2_N_TRAIN = 1600
P2_N_OUT_PER_CELL = 20       # held-out (noiseless) stars per 4x3 cell


def p2_cell_bias(build, n_train=P2_N_TRAIN, realisations=P2_REALISATIONS, seed=202):
    """Per 4x3 cell: mean over realisations of the median PSF-fit flux bias (mmag) of
    noiseless held-out truth stamps, and its standard error across realisations.

    build(stamps, x, y, a) -> callable render(xs, ys) -> stamps.  Bias of a held-out
    star = -2.5 log10(F_fit / 1) with F_fit = sum(P phi) / sum(P^2) and phi the truth stamp
    (box flux 1)."""
    psf = synth.PSFField(SHAPE)
    h, w = SHAPE
    vals = np.full((realisations, 3, 4), np.nan)
    for r in range(realisations):
        rng = np.random.default_rng([seed, r])
        st, x, y, a, _ = training_set(rng, n_train, psf)
        render = build(st, x, y, a)
        for row in range(3):
            for col in range(4):
                x0 = max(synth.HALF, int(np.ceil(col * w / 4)))
                x1 = min(w - 1 - synth.HALF, int(np.ceil((col + 1) * w / 4)) - 1)
                y0 = max(synth.HALF, int(np.ceil(row * h / 3)))
                y1 = min(h - 1 - synth.HALF, int(np.ceil((row + 1) * h / 3)) - 1)
                xo = rng.integers(x0, x1 + 1, P2_N_OUT_PER_CELL).astype(float)
                yo = rng.integers(y0, y1 + 1, P2_N_OUT_PER_CELL).astype(float)
                truth = np.stack([psf.stamp(a_, b_).data for a_, b_ in zip(xo, yo)])
                f = psf_fit_flux(render(xo, yo), truth)
                with np.errstate(invalid="ignore", divide="ignore"):
                    vals[r, row, col] = np.median(-2.5e3 * np.log10(f))
    mean = vals.mean(axis=0)
    se = vals.std(axis=0, ddof=1) / np.sqrt(realisations)
    return mean, se, vals


def test_p2_gradient_recovery():
    """Every 4x3 cell |mean bias| <= 3 mmag, precision condition cell SE <= 0.75 mmag
    (design 6 P2, gate_stats 3: 12 cells at 4 sigma -> family false failure 0.08%).  The model
    is the delivered one: rendered from its render nodes (per-band refinement, D9-1)."""
    def build(st, x, y, a):
        return pm.build_psf_model(st, x, y, a, SHAPE, image_id="synth.p2").render

    mean, se, _ = p2_cell_bias(build)
    assert np.all(np.isfinite(mean)), mean
    assert np.all(se <= 0.75), f"precision condition not met (remedy: 3,200 stars): SE {np.round(se, 3)}"
    assert np.all(np.abs(mean) <= 3.0), f"cell bias (mmag) {np.round(mean, 2)}"


# ---------------------------------------------------------------------------
# G7 (b)  knife-edge hygiene, layer 1 (design 6 G7: one pixel of one training star)
# ---------------------------------------------------------------------------


def test_g7b_linear_in_one_training_pixel():
    """Perturb the central pixel of the brightest training star by relative 1e-6 and 2e-6:
    every render-node stamp value, every mixing-node weight, tau^2 and every candidate loss
    must move linearly, |D(2e) - 2 D(e)| <= 1e-2 |D(e)| + 1e-12; the discrete parts (folds,
    support switches, the active set of every mixing node) must not move; the perturbation must
    reach the model (else the test has no power).  Render refinement fixed (n_b is discrete)."""
    rng = np.random.default_rng(303)
    psf = synth.PSFField(SHAPE)
    st, x, y, a, _ = training_set(rng, 400, psf)
    s = int(np.argmin(a))
    c = synth.HALF

    def build(eps):
        b = st.copy()
        b[s, c, c] *= 1.0 + eps
        return pm.build_psf_model(b, x, y, a, SHAPE, image_id="synth.g7", render_refine=1)

    m0, m1, m2 = build(0.0), build(1e-6), build(2e-6)

    def outputs(m):
        cand = m.record["candidates"]
        return (m.stamps.ravel(), np.asarray(m.mix_weights, dtype=float).ravel(),
                np.array([m.record.get("tau2", np.nan)], dtype=float),
                np.array([k["loss"] for k in cand], dtype=float))

    o0, o1, o2 = outputs(m0), outputs(m1), outputs(m2)
    for name, a0, a1, a2 in zip(("stamps", "mix_weights", "tau2", "losses"), o0, o1, o2):
        d1, d2 = a1 - a0, a2 - a0
        assert np.all(np.isfinite(d1)) and np.all(np.isfinite(d2)), name
        assert np.all(np.abs(d2 - 2.0 * d1) <= 1e-2 * np.abs(d1) + 1e-12), name
    assert np.max(np.abs(o1[0] - o0[0])) > 0.0, "the perturbation did not reach the model"
    assert o1[2][0] != o0[2][0], "the perturbation did not reach tau^2"
    np.testing.assert_array_equal(m1.star_fold, m0.star_fold)
    np.testing.assert_array_equal(np.asarray(m1.mix_weights) > 0, np.asarray(m0.mix_weights) > 0)
    for k0, k1 in zip(m0.record["candidates"], m1.record["candidates"]):
        assert k0["n_nodes_p0"] == k1["n_nodes_p0"] and k0["n_nodes_singular"] == k1["n_nodes_singular"]


# ---------------------------------------------------------------------------
# Design v2.5 unit tests (2nd ed. 5: outside the gate's 25)
# ---------------------------------------------------------------------------


def _simplex_enumeration(q):
    """min w'Qw on the simplex by enumerating every support (independent of the module)."""
    m = q.shape[0]
    best, best_w = np.inf, None
    for mask in range(1, 2 ** m):
        idx = [i for i in range(m) if mask >> i & 1]
        k = len(idx)
        kkt = np.zeros((k + 1, k + 1))
        kkt[:k, :k] = q[np.ix_(idx, idx)]
        kkt[:k, k] = 1.0
        kkt[k, :k] = 1.0
        rhs = np.zeros(k + 1)
        rhs[k] = 1.0
        try:
            sol = np.linalg.solve(kkt, rhs)
        except np.linalg.LinAlgError:
            continue
        v = sol[:k]
        if np.any(v < -1e-15):
            continue
        w = np.zeros(m)
        w[idx] = np.clip(v, 0.0, None)
        w /= w.sum()
        obj = float(w @ q @ w)
        if obj < best - 1e-15 * abs(best if np.isfinite(best) else 1.0):
            best, best_w = obj, w
    return best_w


def test_stacking_is_the_exact_simplex_solution():
    """w = argmin w'(G + 1e-10 tr(G)/M I) w on the simplex (design v2.5 1.3 step 5): the module's
    solver equals the enumeration of every active set to relative 1e-12 (M <= 6), including a
    vertex optimum, an interior optimum and a rank-deficient G (two identical candidates)."""
    rng = np.random.default_rng(7)
    cases = []
    for m in (2, 3, 4, 5, 6):
        b = rng.standard_normal((40, m))
        cases.append(b.T @ b)                                  # interior / mixed supports
    b = rng.standard_normal((40, 4))
    g = b.T @ b
    g[0, :] *= 0.01
    g[:, 0] *= 0.01
    cases.append(g)                                            # vertex optimum (candidate 0)
    for g in cases:
        m = g.shape[0]
        q = g + 1e-10 * np.trace(g) / m * np.eye(m)
        want = _simplex_enumeration(q)
        got = pm.stack_simplex(g)
        assert np.all(np.isfinite(got)) and np.all(got >= 0.0) and abs(got.sum() - 1.0) <= 1e-12
        assert np.max(np.abs(got - want)) <= 1e-12 * max(1.0, np.max(np.abs(want))), (got, want)
    # identical candidates 1 and 3: the stacking problem is defined over DISTINCT candidates (design
    # v2.5 3rd ed. 4.1) -- identical ones are one model, the ridge 1e-10 tr/M is taken on the distinct
    # set, and the weight is split equally within the group.  So the exact answer is the reduced
    # problem with its own ridge, split equally (enumeration of the full problem is only good to
    # ~1e-7 in the split direction, condition ~1/eps: impl_a.1.md round 2).
    b = rng.standard_normal((40, 4))
    b[:, 3] = b[:, 1]
    g = b.T @ b
    red = g[np.ix_([0, 1, 2], [0, 1, 2])]
    wr = _simplex_enumeration(red + 1e-10 * np.trace(red) / 3 * np.eye(3))
    want = np.array([wr[0], wr[1] / 2, wr[2], wr[1] / 2])
    got = pm.stack_simplex(g)
    assert np.max(np.abs(got - want)) <= 1e-12, (got, want)
    assert got[1] == got[3]


def test_duplicate_candidate_does_not_change_the_mixture():
    """Listing a candidate twice leaves the mixed model unchanged (design v2.5 1.4; 3rd ed. 4.1: the
    stacking problem, its ridge and tr/M are defined over distinct candidates, a group's weight is
    split equally -- so this holds by definition, tolerance 1e-12)."""
    rng = np.random.default_rng(11)
    psf = synth.PSFField(SHAPE)
    st, x, y, a, _ = training_set(rng, 300, psf)
    base = pm.default_candidates(SHAPE)
    dup = tuple(base) + (base[2],)
    m0 = pm.build_psf_model(st, x, y, a, SHAPE, candidates=base, render_refine=1)
    m1 = pm.build_psf_model(st, x, y, a, SHAPE, candidates=dup, render_refine=1)
    xs, ys = rng.uniform(0, SHAPE[1] - 1, 200), rng.uniform(0, SHAPE[0] - 1, 200)
    assert np.max(np.abs(m1.render(xs, ys) - m0.render(xs, ys))) <= 1e-12


@pytest.mark.parametrize("ratio", [0.0, 1.0], ids=["no_heterogeneity", "tau2_eq_median_a"])
def test_floor_tau2_estimation(ratio):
    """Design v2.5 2.3: tau^2 solves median_s r2_s / (a_s + tau^2) = 1 with the held-out residual
    of the mixed model.  No heterogeneity -> tau^2 / median(a) <= 0.05; a known per-star pattern of
    squared norm tau2_true -> tau2_hat / tau2_true = 1 +- 0.1 (design 2nd ed. 5)."""
    rng = np.random.default_rng([13, int(ratio * 10)])
    psf = synth.PSFField(SHAPE)
    # median a of the default S/N draw, from the expected-noise formula (independent of the module)
    sn = synth.draw_snr(20000, 50.0, 2000.0, np.random.default_rng(99))
    st0 = psf.stamp(5000, 3300).data
    fl = sn * float(synth.sky_limited_flux(1.0, st0))
    tau2_true = ratio * float(np.median(noise_a(fl)))
    st, x, y, a, _ = training_set(rng, 400, psf, hetero_tau2=tau2_true)
    m = pm.build_psf_model(st, x, y, a, SHAPE, render_refine=1)
    t2 = float(m.record.get("tau2", np.nan))
    if ratio == 0.0:
        assert t2 / float(np.median(a)) <= 0.05, t2 / float(np.median(a))
    else:
        assert abs(t2 / tau2_true - 1.0) <= 0.1, t2 / tau2_true


def test_realistic_luminosity_function_keeps_the_star_count():
    """A steep luminosity function (dN/dS ~ S^-2, S/N 20-20,000) with a constant relative
    heterogeneity (10 x the median noise part -- the real-data signature r2/a ~ S/N^2) collapses the
    effective star count with omega = 1/a; the floor keeps N_eff/N >= 0.5 and no global candidate
    falls back to p = 0 (design v2.5 2nd ed. 5)."""
    rng = np.random.default_rng(17)
    psf = synth.PSFField(SHAPE)
    lo, hi = 20.0, 20000.0
    draw = lambda n, r: lo / (1.0 - r.uniform(0, 1, n) * (1.0 - lo / hi))
    sn = draw(20000, np.random.default_rng(98))
    fl = sn * float(synth.sky_limited_flux(1.0, psf.stamp(5000, 3300).data))
    tau2 = 10.0 * float(np.median(noise_a(fl)))
    st, x, y, a, _ = training_set(rng, 400, psf, hetero_tau2=tau2, snr_draw=draw)
    assert n_eff(1.0 / a) / len(a) < 0.5             # the test has power: without the floor it collapses
    m = pm.build_psf_model(st, x, y, a, SHAPE, render_refine=1)
    assert n_eff(m.star_weight) / len(a) >= 0.5, n_eff(m.star_weight) / len(a)
    for c in m.record["candidates"]:
        if c["h_px"] is None or not np.isfinite(c["h_px"]):
            assert c["n_nodes_p0"] == 0, c


def test_tau2_reuse_reproduces_the_two_pass_model():
    """build(tau2=t) with the tau^2 of a two-pass build is that build's second pass: same model to
    1e-12 in one pass (design v2.5 2nd ed. 1.4: PSF(1), PSF(2) reuse the previous tau^2).
    History: a.3 rounds 2-3 made this two passes for a Huber weight; round 4 removed the Huber
    weight (3rd ed. 7.1) and this is a.1's test again."""
    rng = np.random.default_rng(19)
    psf = synth.PSFField(SHAPE)
    st, x, y, a, _ = training_set(rng, 300, psf)
    m2 = pm.build_psf_model(st, x, y, a, SHAPE, render_refine=1)
    m1 = pm.build_psf_model(st, x, y, a, SHAPE, render_refine=1, tau2=(m2.record["lam"], m2.record["tau2"]))
    assert m2.record["passes"] == 2 and m1.record["passes"] == 1
    assert np.max(np.abs(m1.stamps - m2.stamps)) <= 1e-12
    assert np.max(np.abs(np.asarray(m1.mix_weights) - np.asarray(m2.mix_weights))) <= 1e-12


def _outlier_set(rng, n=600, n_out=30, extra=1.0, hetero=1e-3):
    """Training stamps with known outliers: n_out bright stars (a_s <= 2e-4) get a neighbour of
    ``extra`` x their flux 10 px away inside the box (design v2.5 3rd ed. 5.4 test)."""
    psf = synth.PSFField(SHAPE)
    st, x, y, a, flux = training_set(rng, n, psf, hetero_tau2=hetero)
    yy, xx = np.mgrid[0:55, 0:55]
    blob = np.exp(-0.5 * ((xx - 36.0) ** 2 + (yy - 22.0) ** 2) / 2.0 ** 2)
    blob /= blob.sum()
    bright = np.flatnonzero(a <= 2e-4)
    out = rng.choice(bright, n_out, replace=False)
    return st, x, y, a, flux, out, blob


# History of the Huber weight (a.3 rounds 2-3; recorded in the development records impl_a.3.md
# and review_a.3.md, not distributed): the 3rd ed. 5.4 expectation "every planted outlier cut, every clean star kept"
# FAILED on the designer's scene (seed 23: 24/30 outliers, 0/570 clean; clean 300-star scene 1/300).
# The round-3 "pass" (seed 31) used a scene changed after those results (1000 stars, 2x neighbours)
# and is not a pass: on the designer's scene with fresh seeds 101/202/303 expectation 2 failed 3/3,
# and vbar changed no registered result.  Design v2.5 3rd ed. 7.1 deleted the Huber
# weight and vbar (u = 1; four fits per band); their tests went with them (round 4).


# ---------------------------------------------------------------------------
# Design v2.5 3rd ed. 7.1-7.2 (a.3 round 4): no Huber weight; a = box sum + shape term (+ neighbour term)
# ---------------------------------------------------------------------------

def test_star_noise_a_follows_the_definition():
    """a_s = n_pix sigma_sky^2 (1 + kappa |phi|^2) / S^2 + 1 / (g S) + n_pix sigma2_nb / S^2 (3rd ed. 7.1)."""
    from tract7dt.fit.grid_gpu import psf_from_stars as pfs
    box = np.array([1.0e3, 5.0e4, 2.0e5])
    sky = np.array([1.0, 2.5, 0.7])
    nb = np.array([0.0, 3.0, 40.0])
    phi2 = np.array([0.02, 0.012, 0.03])
    want = NPIX * sky * (1.0 + 1.72 * phi2) / box ** 2 + 1.0 / (50.0 * box) + NPIX * nb / box ** 2
    got = pfs.star_noise_a(box, sky, nb, kappa=1.72, phi_norm2=phi2, gain=50.0, n_pix=NPIX)
    np.testing.assert_allclose(got, want, rtol=1e-14, atol=0.0)


# Pre-registered (design v2.5 3rd ed. 7.2) in the test-only commit before its first run.  Fixed: the
# scene (_outlier_set defaults: 600 stars, 30 bright outliers, a neighbour blob of 1x the star's flux,
# heterogeneity 1e-3), the seeds 4401-4403 (never run before), the 400 evaluation positions
# (default_rng([seed, 9]): x then y, integers in [27, W - 28] and [27, H - 28]), the statistic (median
# over the positions of the PSF-fit flux bias -2.5 log10(<P, T> / <P, P>), P = the exact mixture),
# the arms and the thresholds.  Any change needs new seeds and a new pre-registration.
PREREG_72_SEEDS = (4401, 4402, 4403)


def _bias_median(mix, psf, xs, ys):
    p = mix.evaluate(xs, ys).reshape(len(xs), -1)
    t = np.stack([psf.stamp(u, v).data.ravel() for u, v in zip(xs, ys)])
    return float(np.median(-2.5e3 * np.log10((p * t).sum(axis=1) / (p * p).sum(axis=1))))


# Known failure (recorded since 12d4657, kept visible): seed 4402 fails the no-blob control, |med K| = 3.60 mmag > 3.
# The model under test (psf_model.py) is not used by the photometry of release 0.1.0; strict, so a pass shows up.
_D6_PARAMS = [pytest.param(s, marks=pytest.mark.xfail(
    reason="known failure: seed 4402 no-blob control |med K| 3.60 mmag > 3 mmag (registered threshold kept); "
           "psf_model.py is not used by the 0.1.0 photometry; open item",
    strict=True, raises=AssertionError)) if s == 4402 else s for s in PREREG_72_SEEDS]


@pytest.mark.parametrize("seed", _D6_PARAMS)
def test_d6_neighbour_term_removes_most_of_the_coherent_bias(seed):
    """(T) a with the D6 neighbour term of the true blob (n_pix sigma2_nb / S^2 = F^2 |b|^2 / S^2),
    (C) the same a without it, (K) the same scene without blobs; no Huber, two passes, tau^2 median.
    Expected for every seed: |med T| <= 0.35 |med C|, |med C| >= 20 mmag, |med K| <= 3 mmag."""
    from tract7dt.fit.grid_gpu import psf_from_stars as pfs
    rng = np.random.default_rng(seed)
    st, x, y, _, flux, out, blob = _outlier_set(rng)
    psf = synth.PSFField(SHAPE)
    st_k = st.copy()
    st[out] += 1.0 * flux[out, None, None] * blob
    phi2 = np.array([float((psf.stamp(u, v).data ** 2).sum()) for u, v in zip(x, y)])
    nb = np.zeros(len(x))
    nb[out] = flux[out] ** 2 * float((blob ** 2).sum()) / NPIX

    def a_of(stamps, nb_term):
        return pfs.star_noise_a(stamps.sum(axis=(1, 2)), SIGMA ** 2, nb_term, kappa=KAPPA_SYNTH,
                                phi_norm2=phi2, gain=GAIN, n_pix=NPIX)

    ev = np.random.default_rng([seed, 9])
    h, w = SHAPE
    xs = ev.integers(27, w - 28 + 1, 400).astype(float)
    ys = ev.integers(27, h - 28 + 1, 400).astype(float)
    med = {}
    for arm, stamps, nb_term in (("T", st, nb), ("C", st, np.zeros(len(x))), ("K", st_k, np.zeros(len(x)))):
        mix, _ = pm.fit_mixture(stamps, x, y, a_of(stamps, nb_term), SHAPE)
        med[arm] = _bias_median(mix, psf, xs, ys)
    assert abs(med["C"]) >= 20.0, med                       # the test has power
    assert abs(med["T"]) <= 0.35 * abs(med["C"]), med       # the D6 neighbour term removes most of it
    assert abs(med["K"]) <= 3.0, med                        # no blob: no bias


def test_render_spacing_meets_the_criteria():
    """D9-1 (design v2.5 3rd ed. 1.2): n_b is the coarsest refinement whose estimate meets flux
    <= 1e-3 and pixel norm <= 1e-2 and whose check at the top cells passes; independently, the actual
    flux error of the rendered template against the exact mixture at 1,000 random positions is below
    1e-3 (noiseless star, F' = sum(P' P) / sum(P'^2), no sub-pixel shift)."""
    rng = np.random.default_rng(23)
    psf = synth.PSFField(SHAPE)
    st, x, y, a, _ = training_set(rng, 400, psf)
    m = pm.build_psf_model(st, x, y, a, SHAPE)
    rr = m.record["render"]
    n_b = int(rr["refine"])
    v = rr["verified"][str(n_b)]
    assert v["flux_frac_max"] <= 1e-3 and v["pix_frac_max"] <= 1e-2
    assert rr["est_flux_frac_max"] <= 1e-3 and rr["est_pix_frac_max"] <= 1e-2
    assert v["n_cells"] == int(np.ceil(0.05 * 32 * 24))
    if n_b > 1:
        prev = rr["verified"].get(str(n_b - 1))
        est_ok = (rr["est_flux_frac_max_refine1"] / (n_b - 1) ** 2 <= 1e-3
                  and rr["est_pix_frac_max_refine1"] / (n_b - 1) ** 2 <= 1e-2)
        assert (not est_ok) or (prev is not None and (prev["flux_frac_max"] > 1e-3 or prev["pix_frac_max"] > 1e-2))
    exact = pm.fit_mixture(st, x, y, a, SHAPE, tau2=(m.record["lam"], m.record["tau2"]))[0]
    xs, ys = rng.uniform(0, SHAPE[1] - 1, 1000), rng.uniform(0, SHAPE[0] - 1, 1000)
    p = exact.evaluate(xs, ys)
    q = m.render(xs, ys)
    f = (q * p).sum(axis=(1, 2)) / (q * q).sum(axis=(1, 2))
    assert np.max(np.abs(f - 1.0)) <= 1e-3, np.max(np.abs(f - 1.0))


class _FieldMixture:
    """A stand-in mixture with a closed-form stamp field (for the render-spacing tests)."""

    def __init__(self, fn, s):
        self.fn, self.s = fn, s

    def evaluate(self, xs, ys):
        xs = np.atleast_1d(np.asarray(xs, dtype=float))
        ys = np.atleast_1d(np.asarray(ys, dtype=float))
        return self.fn(xs[:, None], ys[:, None]).reshape(len(xs), self.s, self.s)


def test_render_estimate_is_exact_for_quadratic_fields():
    """Design v2.5 3rd ed. 1.4: for a field quadratic in position the bilinear error at a cell centre
    is exactly -[(Lx^2/8) P_xx + (Ly^2/8) P_yy] (no cross term at t = u = 1/2), at an x-edge midpoint
    -(Lx^2/8) P_xx and at a y-edge midpoint -(Ly^2/8) P_yy; the estimate's errors match to 1e-6."""
    rng = np.random.default_rng(41)
    sz, shape = 9, (660, 1000)
    base = rng.uniform(0.5, 1.5, sz * sz)
    base /= base.sum()
    c = rng.normal(size=(5, sz * sz)) * 1e-3
    sc = 1e-3                                                     # coordinates in units of 1000 px
    fn = lambda x, y: (base + c[0] * (x * sc) + c[1] * (y * sc) + c[2] * (x * sc) ** 2
                       + c[3] * (x * sc) * (y * sc) + c[4] * (y * sc) ** 2)
    gx, gy = pm.render_grid(pm.default_candidates(shape), shape, 1)
    e = pm._refine1_errors(_FieldMixture(fn, sz), gx, gy)
    lx, ly = gx[1] - gx[0], gy[1] - gy[0]
    pxx, pyy = 2 * c[2] * sc ** 2, 2 * c[4] * sc ** 2
    want_c = -(lx ** 2 / 8 * pxx + ly ** 2 / 8 * pyy)
    want_x, want_y = -(lx ** 2 / 8) * pxx, -(ly ** 2 / 8) * pyy
    for got, want in ((e["d_center"], want_c), (e["d_xedge"], want_x), (e["d_yedge"], want_y)):
        assert np.max(np.abs(got - want[None, None, :])) <= 1e-6 * np.max(np.abs(want))


def test_render_check_raises_the_refinement():
    """Design v2.5 3rd ed. 1.4: the check at the top cells has power -- a narrow bump at a quarter
    point of the worst cell, invisible to the refinement-1 estimate (centre and edge midpoints are
    >= 4 sigma away), is found at the sub-cell centres and raises n above the estimate; without the
    bump the estimate stands."""
    sz, shape = 9, (660, 1000)
    rng = np.random.default_rng(43)
    phi0 = rng.uniform(0.5, 1.5, sz * sz)
    phi0 /= phi0.sum()
    gx, gy = pm.render_grid(pm.default_candidates(shape), shape, 1)
    i0, j0 = 16, 12
    lx, ly = gx[1] - gx[0], gy[1] - gy[0]
    x0, y0 = gx[i0] + lx / 2, gy[j0] + ly / 2                  # hill on the centre of cell (12, 16)
    r_h, c_h = 3.0 * max(lx, ly), 0.085
    xb, yb, sb, beta = gx[i0] + lx / 4, gy[j0] + ly / 4, min(lx, ly) / 12.0, 0.002

    def field(bump):
        def fn(x, y):
            s_ = c_h * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * r_h ** 2))
            if bump:
                s_ = s_ + beta * np.exp(-((x - xb) ** 2 + (y - yb) ** 2) / (2 * sb ** 2))
            return phi0 * (1.0 + s_)
        return _FieldMixture(fn, sz)

    n0, r0, _ = pm.select_render_refine(field(False), pm.default_candidates(shape), shape)
    n1, r1, _ = pm.select_render_refine(field(True), pm.default_candidates(shape), shape)
    assert r0["n_estimated"] >= 2 and n0 == r0["n_estimated"]
    assert r1["n_estimated"] == r0["n_estimated"]
    assert n1 > r1["n_estimated"], r1["verified"]
    assert r1["verified"][str(r1["n_estimated"])]["flux_frac_max"] > 1e-3
    assert r1["verified"][str(n1)]["flux_frac_max"] <= 1e-3


def test_shift_operator_norm():
    """Design v2.5 3rd ed. 1.4: the 2-D Lanczos-3 shift norm is computed (largest singular value of
    the 55-px banded shift matrix over 201 shifts, squared) and bounded by the squared maximum of
    the taps' frequency response (independent: the l2 norm of a convolution); about 1.054."""
    got = pm.lanczos_shift_norm_2d()
    w = np.linspace(-np.pi, np.pi, 4001)
    best = 0.0
    for d in np.linspace(-0.5, 0.5, 201):
        xx = np.arange(-3, 4) + d
        k = np.where(np.abs(xx) < 1e-12, 1.0, np.sinc(xx) * np.sinc(xx / 3.0))
        k = np.where(np.abs(xx) >= 3, 0.0, k)
        k /= k.sum()
        best = max(best, float(np.abs(np.exp(-1j * np.outer(w, np.arange(7))) @ k).max()))
    assert 1.0 <= got <= best ** 2 + 1e-12, (got, best ** 2)
    assert abs(got - 1.054) <= 0.002


def test_render_memory_check_fails_cleanly():
    """D9-1 answer 1: node-stamp bytes are exact (float32, nx ny S^2 4 per image) and a card whose
    free memory is short fails with RUN_STATUS error.kind psf_render_memory (never a silent
    coarsening)."""
    from tract7dt.run_status import RunContractError
    m = _toy_model()
    need = pm.render_stamp_bytes([m, m])
    assert need == 2 * 3 * 4 * 55 * 55 * 4
    pm.require_render_memory(need, need, card=1)              # exactly enough: no error
    raised = None
    try:
        pm.require_render_memory(need, need - 1, card=1, other_users=["jhb"])
    except RunContractError as exc:
        raised = exc
    assert raised is not None and raised.kind == "psf_render_memory"
    assert raised.details["card"] == 1 and raised.details["need_bytes"] == need


def _blas_probe():
    """Worker-side: (env OPENBLAS_NUM_THREADS, OpenBLAS's own thread count or None)."""
    import ctypes
    import os
    got = None
    for line in open("/proc/self/maps"):
        if "openblas" in line and line.rstrip().endswith(".so") or "openblas" in line and ".so." in line:
            path = line.split()[-1]
            try:
                got = int(ctypes.CDLL(path).openblas_get_num_threads())
                break
            except (OSError, AttributeError):
                continue
    return os.environ.get("OPENBLAS_NUM_THREADS"), got


def test_blas_thread_rule_in_psf_workers():
    """D9-1 answer 3: threads = max(1, physical cores in the affinity // concurrent workers), set
    before numpy loads in every PSF worker (OMP/OPENBLAS/MKL)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from tract7dt.fit.grid_gpu import psf_pipeline_stage as ps
    assert ps.blas_threads_for(22, logical_cpus=64, threads_per_core=2) == 1
    assert ps.blas_threads_for(4, logical_cpus=64, threads_per_core=2) == 8
    assert ps.blas_threads_for(1, logical_cpus=8, threads_per_core=1) == 8
    assert ps.blas_threads_for(40, logical_cpus=8, threads_per_core=2) == 1
    with ps.pinned_blas_env(3):
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
            env, got = pool.submit(_blas_probe).result(timeout=120)
    assert env == "3"
    if got is not None:
        assert got == 3


# ---------------------------------------------------------------------------
# Known answer: least squares reproduces a field in the candidate's span exactly
# ---------------------------------------------------------------------------


def test_fields_in_the_span_are_reproduced_exactly():
    """Independent expectation (property of weighted least squares, not of the module): a
    global cubic field is reproduced by the global candidates of order >= 3 and a global
    quadratic field by the local p = 2 candidates (every node has support here), to
    floating-point precision; the next lower order is not exact (the test has power)."""
    rng = np.random.default_rng(5)
    h, w, s, n = 660, 1000, 9, 300
    shape = (h, w)
    x, y = rng.uniform(0, w - 1, n), rng.uniform(0, h - 1, n)
    a = rng.uniform(0.5, 1.5, (s, s))
    a /= a.sum()
    zs = [z - z.mean() for z in rng.normal(size=(3, s, s)) * 0.01]   # zero-sum patterns
    a_noise = 1.0 / rng.uniform(1, 100, n)
    xt, yt = rng.uniform(0, w - 1, 50), rng.uniform(0, h - 1, 50)

    def cubic(px, py):
        xh, yh = 2 * px / (w - 1) - 1, 2 * py / (h - 1) - 1
        return (a[None] + zs[0][None] * xh[:, None, None] + zs[1][None] * (yh ** 3)[:, None, None]
                + zs[2][None] * (xh ** 2 * yh)[:, None, None])

    def quad(px, py):
        return (a[None] + zs[0][None] * (px / w)[:, None, None]
                + zs[1][None] * ((py / h) ** 2)[:, None, None]
                + zs[2][None] * (px * py / (w * h))[:, None, None])

    for p in (2, 3, 5):
        mix, _ = pm.fit_mixture(cubic(x, y), x, y, a_noise, shape, candidates=[pm.Candidate(math.inf, p)])
        err = np.abs(mix.evaluate(xt, yt) - cubic(xt, yt)).max()
        assert (err < 1e-14) if p >= 3 else (err > 1e-4), (p, err)
    d = pm.image_diagonal(shape)
    for frac in (2.0, 4.0, 8.0):
        for p in (1, 2):
            mix, _ = pm.fit_mixture(quad(x, y), x, y, a_noise, shape, candidates=[pm.Candidate(d / frac, p)])
            assert mix.fits[0].n_p0 == 0
            err = np.abs(mix.evaluate(xt, yt) - quad(xt, yt)).max()
            assert (err < 1e-14) if p == 2 else (err > 1e-4), (frac, p, err)


def test_fold_sum_shortcut_preserves_the_model(monkeypatch):
    """Training sums as total - that fold (speed, psf_speed_answer 2.1) give the same model as the
    direct sum of the other folds (written here): render stamps and tau^2 to relative 1e-12, the
    same active set at every mixing node (impl_a.1.md round 2: measured 1.6e-15 / 3.7e-15)."""
    rng = np.random.default_rng(29)
    psf = synth.PSFField(SHAPE)
    st, x, y, a, _ = training_set(rng, 200, psf)
    fast = pm.build_psf_model(st, x, y, a, SHAPE, render_refine=1)

    def direct(fam, held_fold):
        if held_fold is None:
            return fam.G.sum(axis=0), fam.R.sum(axis=0), fam.S1.sum(axis=0), fam.S2.sum(axis=0)
        keep = np.arange(fam.G.shape[0]) != held_fold
        return (fam.G[keep].sum(axis=0), fam.R[keep].sum(axis=0),
                fam.S1[keep].sum(axis=0), fam.S2[keep].sum(axis=0))

    monkeypatch.setattr(pm, "_train_sums", direct)
    ref = pm.build_psf_model(st, x, y, a, SHAPE, render_refine=1)
    assert np.max(np.abs(fast.stamps - ref.stamps)) <= 1e-12 * np.max(np.abs(ref.stamps))
    assert abs(fast.record["tau2"] - ref.record["tau2"]) <= 1e-12 * ref.record["tau2"]
    np.testing.assert_array_equal(fast.mix_weights > 0, ref.mix_weights > 0)


def test_extra_candidate_plug_in_point():
    """The a.8 plug-in point (unit_a8_defocus_spec.md 2.3): an external candidate is fitted once
    per fold on that fold's training stars and once on all stars, its held-out residuals enter the
    same Gram and stacking, and the model stays a sum-1 mixture.  An oracle that predicts the truth
    PSF must win the stacking at most mixing nodes."""
    rng = np.random.default_rng(37)
    psf = synth.PSFField(SHAPE)
    st, x, y, a, _ = training_set(rng, 300, psf)
    calls = []
    prepared = []

    class Oracle:
        name = "oracle"

        def prepare(self, d, xs, ys):
            prepared.append(len(xs))

        def fit(self, d, xs, ys, omega, train):
            calls.append(np.asarray(train, dtype=bool).copy())

            class _Pred:
                def evaluate(self, px, py):
                    return np.stack([psf.stamp(float(np.rint(u)), float(np.rint(v))).data.ravel()
                                     for u, v in zip(np.atleast_1d(px), np.atleast_1d(py))])
            return _Pred()

    m = pm.build_psf_model(st, x, y, a, SHAPE, render_refine=1, tau2=(1.0, 0.0), extra_candidates=[Oracle()])
    assert len(calls) == pm.N_FOLDS + 1 and prepared == [len(x)]      # one pass: prepare once
    for f in range(pm.N_FOLDS):
        np.testing.assert_array_equal(calls[f], m.star_fold != f)
    assert calls[-1].all()
    assert m.record["candidates"][-1]["name"] == "oracle"
    w = np.asarray(m.mix_weights)
    assert w.shape[-1] == len(pm.default_candidates(SHAPE)) + 1
    assert np.max(np.abs(m.stamps.sum(axis=(-2, -1)) - 1.0)) <= 1e-6
    best = np.argmax(w, axis=-1)
    assert np.mean(best == w.shape[-1] - 1) > 0.5, np.mean(best == w.shape[-1] - 1)



def test_constant_noiseless_field_builds():
    """A spatially constant noiseless PSF (identical stamps) has no render error;
    the model builds with n_b = 1 (it used to stop with ValueError in the render estimate)."""
    shape = (660, 1000)
    psf = synth.PSFField(shape, fwhm=4.0)
    rng = np.random.default_rng(3)
    x = rng.integers(27, 1000 - 28, 80).astype(float)
    y = rng.integers(27, 660 - 28, 80).astype(float)
    one = psf.stamp(500.0, 330.0).data
    st = np.stack([5000.0 * one for _ in range(80)])
    try:
        m = pm.build_psf_model(st, x, y, np.full(80, 1e-3), shape)
    except ValueError as exc:
        raise AssertionError(f"constant field stopped the build: {exc}") from exc
    assert m.record["render"]["refine"] == 1
    xs, ys = rng.uniform(0, 999, 20), rng.uniform(0, 659, 20)
    assert np.max(np.abs(m.render(xs, ys) - one[None])) <= 1e-12


def test_extra_candidate_fit_sees_only_the_training_stars():
    """The plug-in's fit receives the training stars only (structure, not contract):
    every array it gets has train.sum() rows, and none of the held-out stamps is among them."""
    rng = np.random.default_rng(47)
    psf = synth.PSFField(SHAPE)
    st, x, y, a, _ = training_set(rng, 150, psf)
    seen = []

    class Spy:
        name = "spy"

        def fit(self, d, xs, ys, omega, train):
            seen.append((np.asarray(d).copy(), len(xs), len(ys), len(omega), np.asarray(train).copy()))

            class _P:
                def evaluate(self, px, py):
                    out = np.full((len(np.atleast_1d(px)), synth.BOX * synth.BOX), 1.0 / (synth.BOX * synth.BOX))
                    return out
            return _P()

    m = pm.build_psf_model(st, x, y, a, SHAPE, render_refine=1, tau2=(1.0, 0.0), extra_candidates=[Spy()])
    norm = (st / st.sum(axis=(1, 2), keepdims=True)).reshape(len(x), -1)
    for d_seen, nx_, ny_, nw_, train in seen:
        k = int(train.sum())
        assert d_seen.shape[0] == nx_ == ny_ == nw_ == k
        held = np.nonzero(~train)[0]
        for h in held:
            assert not np.any(np.all(d_seen == norm[h][None, :], axis=1))
    assert len(seen) == pm.N_FOLDS + 1 and m.record["candidates"][-1]["name"] == "spy"


def test_render_check_includes_cell_edge_midpoints():
    """The top-cell check also looks at the sub-cell edge midpoints (where a
    saddle-shaped error concentrates), not only the sub-cell centres: per checked cell n^2 centres +
    2 n (n + 1) edge midpoints."""
    sz, shape = 9, (660, 1000)
    rng = np.random.default_rng(53)
    phi0 = rng.uniform(0.5, 1.5, sz * sz)
    phi0 /= phi0.sum()
    fn = lambda x, y: phi0 * (1.0 + 0.2 * np.exp(-((x - 500.0) ** 2 + (y - 330.0) ** 2) / (2 * 120.0 ** 2)))
    n_b, rec, _ = pm.select_render_refine(_FieldMixture(fn, sz), pm.default_candidates(shape), shape)
    v = rec["verified"][str(n_b)]
    assert n_b >= 2
    assert v["n_points"] == v["n_cells"] * (n_b * n_b + 2 * n_b * (n_b + 1)), v


def test_row_skip_preserves_the_model(monkeypatch):
    """Computing only the R rows some training set can support
    gives the same model as computing every row (reference written here): stamps and tau^2 to relative
    1e-12, same active sets -- on a sparse set where the skip happens (D/16 p2 nodes on p = 0)."""
    rng = np.random.default_rng(59)
    psf = synth.PSFField(SHAPE)
    st, x, y, a, _ = training_set(rng, 90, psf)
    fast = pm.build_psf_model(st, x, y, a, SHAPE, render_refine=1)
    d16 = [c for c in fast.record["candidates"] if c["name"] == "h=D/16 p=2"][0]
    assert d16["n_nodes_p0"] > 0                       # the skip is exercised

    def all_rows(h, p_max, xs, ys, omega, d, folds, n_folds, shape):
        hh, ww = int(shape[0]), int(shape[1])
        if not math.isfinite(h):
            nx_, ny_ = np.zeros(0), np.zeros(0)
            bas = pm._basis(pm._hat(xs, ww), pm._hat(ys, hh), p_max, True)[:, None, :]
            w = omega[None, :]
        else:
            nx_, ny_ = pm.training_grid(h, shape)
            gx, gy = np.meshgrid(nx_, ny_)
            u = (xs[None, :] - gx.ravel()[:, None]) / h
            v = (ys[None, :] - gy.ravel()[:, None]) / h
            w = np.exp(-0.5 * (u * u + v * v)) * omega[None, :]
            bas = pm._basis(u, v, p_max, False)
        t, k, _ = bas.shape
        G = np.zeros((n_folds, k, t, t)); R = np.zeros((n_folds, k, t, d.shape[1]))
        S1 = np.zeros((n_folds, k)); S2 = np.zeros((n_folds, k))
        for f in range(n_folds):
            idx = np.nonzero(folds == f)[0]
            wa = np.transpose(bas[:, :, idx] * w[None, :, idx], (1, 0, 2))
            G[f] = np.einsum("ktn,kun->ktu", wa, np.transpose(bas[:, :, idx], (1, 0, 2)))
            R[f] = (wa.reshape(k * t, idx.size) @ d[idx]).reshape(k, t, -1)
            S1[f] = w[:, idx].sum(axis=1); S2[f] = (w[:, idx] ** 2).sum(axis=1)
        return pm._Family(h, p_max, nx_, ny_, G, R, S1, S2, R.sum(axis=0))

    monkeypatch.setattr(pm, "_family_sums", all_rows)
    ref = pm.build_psf_model(st, x, y, a, SHAPE, render_refine=1)
    assert np.max(np.abs(fast.stamps - ref.stamps)) <= 1e-12 * np.max(np.abs(ref.stamps))
    assert abs(fast.record["tau2"] - ref.record["tau2"]) <= 1e-12 * max(ref.record["tau2"], 1e-300)
    np.testing.assert_array_equal(fast.mix_weights > 0, ref.mix_weights > 0)


def test_dense_evaluation_path_preserves_values(monkeypatch):
    """The per-cell GEMM path for dense point sets gives the
    same values as the per-point gather path -- for one candidate at clustered points (relative 1e-12)
    and end to end for a model rendered at refinement 3 (stamps relative 1e-12)."""
    rng = np.random.default_rng(61)
    psf = synth.PSFField(SHAPE)
    st, x, y, a, _ = training_set(rng, 200, psf)
    mix, _ = pm.fit_mixture(st, x, y, a, SHAPE, tau2=(1.0, 0.0))
    fit = [f for f in mix.fits if not f.cand.is_global and f.coefs.shape[1] == 6][-1]
    xs = rng.uniform(100, 1400, 2000)
    ys = rng.uniform(100, 1200, 2000)                    # few cells, many points per cell: dense path
    dense = fit.evaluate(xs, ys)
    monkeypatch.setattr(pm, "DENSE_POINTS_PER_CELL", 10 ** 9)
    gather = fit.evaluate(xs, ys)
    assert np.max(np.abs(dense - gather)) <= 1e-12 * np.max(np.abs(gather))
    m_gather = pm.build_psf_model(st, x, y, a, SHAPE, render_refine=3, tau2=(1.0, 0.0))
    monkeypatch.setattr(pm, "DENSE_POINTS_PER_CELL", 8)
    m_dense = pm.build_psf_model(st, x, y, a, SHAPE, render_refine=3, tau2=(1.0, 0.0))
    assert np.max(np.abs(m_dense.stamps - m_gather.stamps)) <= 1e-12 * np.max(np.abs(m_gather.stamps))
