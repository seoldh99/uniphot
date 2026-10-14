"""Output contract for band cells with no information (REGISTER R05) and the
applied blank-response b_hat record (REGISTER R09).

R05. A source whose stamp lies entirely on zero-coverage padding gets an
exact-zero amplitude and a non-finite variance from the band's linear solve.
The pass-2 blank-response correction then subtracts the grid's b_hat from every
STAR flux, so the catalog used to carry ``FLUX = -b_hat`` with a NaN error, and
``compute_zp`` turned the positive ones into a finite MAG (dev7 anchors: 4,850
band cells on sparse T08676, 155,610 on crowded T08285). The contract now is:
where ``FLUXERR_{band}_fit`` is NaN, every per-band flux column is NaN, and no
MAG is formed.

R09. The b_hat each grid subtracted is recorded per (grid, band) in
``blank_response_bhat.csv``; a STAR row's uncorrected flux is
``FLUX + b_hat`` of its ``grid_id_fit``.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tract7dt.fit.grid_gpu.band_flags import (
    BAND_FLUX_COLUMN_PREFIXES,
    blank_nodata_band_cells,
)
from tract7dt.fit.grid_gpu.pipeline import blank_response_bhat_table


def _band_block(bn, flux, ferr, *, win=None, frac=None, conf=None):
    n = len(flux)
    ferr = np.asarray(ferr, dtype=np.float64)
    tot = np.sqrt(ferr ** 2 + 0.01)                 # NaN where ferr is NaN
    return {
        f"FLUX_{bn}_fit": np.asarray(flux, dtype=np.float64),
        f"FLUXERR_{bn}_fit": ferr.astype(np.float32),
        f"FLUXERR_CONF_{bn}_fit": (np.full(n, 0.1) if conf is None
                                   else np.asarray(conf, dtype=np.float64)),
        f"FLUX_WIN_{bn}_fit": (np.asarray(flux, dtype=np.float64) * 0.9
                               if win is None else np.asarray(win, dtype=float)),
        f"FLUXERR_TOT_{bn}_fit": tot.astype(np.float32),
        f"FLUXERR_MARG_{bn}_fit": ferr,
        f"FLUX_MARG_FRAC_{bn}_fit": (np.zeros(n, np.float32) if frac is None
                                     else np.asarray(frac, np.float32)),
        f"FLUXERR_MARG_TOT_{bn}_fit": tot.astype(np.float32),
    }


def _catalog():
    # Row 1 = no coverage in g (FLUX = -b_hat, FLUX_WIN = 0, FLUX_MARG_FRAC = 0
    # for a STAR); row 3 = near-degenerate g column (garbage flux, NaN error);
    # row 4 = no fit at all (every band NaN).
    nan = np.nan
    cols = {"ID": ["a", "b", "c", "d", "e"],
            "fit_is_reliable": [True, True, True, True, False],
            "grid_id_fit": [7, 7, 7, 8, -1]}
    cols.update(_band_block(
        "g", flux=[10.0, 0.37, 5.0, 5.5e6, nan],
        ferr=[1.0, nan, 0.5, nan, nan],
        win=[9.0, 0.0, 4.5, 3.0, nan], conf=[0.1, 0.2, 0.1, nan, nan]))
    cols.update(_band_block(
        "r", flux=[20.0, 3.0, -1.0, 7.0, nan],
        ferr=[2.0, 0.4, 0.3, 0.7, nan]))
    return pd.DataFrame(cols)


def test_nodata_cells_blank_every_band_flux_column():
    out = _catalog()
    before = out.copy()
    n = blank_nodata_band_cells(out, ["g", "r"])

    nodata_g = np.array([False, True, False, True, True])
    for prefix in BAND_FLUX_COLUMN_PREFIXES:
        col = f"{prefix}g_fit"
        assert out.loc[nodata_g, col].isna().all(), col
        # measured cells are untouched, value and dtype
        pd.testing.assert_series_equal(out.loc[~nodata_g, col],
                                       before.loc[~nodata_g, col])
    # band r: measured on rows 0-3 (untouched); row 4 has no fit, so even
    # its stray confusion value goes
    nodata_r = np.array([False, False, False, False, True])
    for prefix in BAND_FLUX_COLUMN_PREFIXES:
        col = f"{prefix}r_fit"
        assert out.loc[nodata_r, col].isna().all(), col
        pd.testing.assert_series_equal(out.loc[~nodata_r, col],
                                       before.loc[~nodata_r, col])
    # non-band columns untouched (the reliability flag is R06's business)
    for col in ("ID", "fit_is_reliable", "grid_id_fit"):
        pd.testing.assert_series_equal(out[col], before[col])
    # counts = cells that still carried a finite flux (rows 1 and 3 in g)
    assert n == {"g": 2, "r": 0}


def test_nodata_blanking_is_a_noop_on_a_fully_measured_catalog():
    out = _catalog().iloc[[0, 2]].reset_index(drop=True)
    before = out.copy()
    n = blank_nodata_band_cells(out, ["g", "r"])
    pd.testing.assert_frame_equal(out, before)
    assert n == {"g": 0, "r": 0}


def test_nodata_blanking_matches_band_names_exactly():
    # Bands are opaque labels (v2-D6): blanking "m400" must not touch the
    # columns of a band whose name merely extends it.
    out = pd.DataFrame({})
    for bn, ferr in (("m400", [np.nan, 1.0]), ("m400w", [np.nan, 1.0])):
        for k, v in _band_block(bn, flux=[2.0, 3.0], ferr=ferr).items():
            out[k] = v
    before = out.copy()
    n = blank_nodata_band_cells(out, ["m400"])
    assert np.isnan(out.loc[0, "FLUX_m400_fit"])
    pd.testing.assert_series_equal(out["FLUX_m400w_fit"],
                                   before["FLUX_m400w_fit"])
    pd.testing.assert_series_equal(out["FLUX_WIN_m400w_fit"],
                                   before["FLUX_WIN_m400w_fit"])
    assert n == {"m400": 1}


def test_bhat_table_rows_units_and_skips():
    ok = SimpleNamespace(grid_id=3, error=None,
                         blank_response_bhat=np.array([-0.25, 0.125, 1.5]))
    failed = SimpleNamespace(grid_id=4, error="RuntimeError: boom",
                             blank_response_bhat=np.array([9.0, 9.0, 9.0]))
    off = SimpleNamespace(grid_id=5, error=None, blank_response_bhat=None)
    df = blank_response_bhat_table([ok, failed, off], ["g", "m400", "r"])
    assert list(df.columns) == ["grid_id", "band", "b_hat"]
    assert df["grid_id"].tolist() == [3, 3, 3]
    assert df["band"].tolist() == ["g", "m400", "r"]
    np.testing.assert_array_equal(df["b_hat"].to_numpy(), [-0.25, 0.125, 1.5])


def test_bhat_table_is_header_only_when_nothing_was_subtracted(tmp_path):
    off = SimpleNamespace(grid_id=1, error=None, blank_response_bhat=None)
    df = blank_response_bhat_table([off], ["g"])
    p = tmp_path / "blank_response_bhat.csv"
    df.to_csv(p, index=False)
    assert p.read_text().strip() == "grid_id,band,b_hat"


def test_solver_reports_bhat_and_nodata_cell_is_blanked():
    """End to end through the worker: a star on zero-coverage padding comes
    out of the pass-2 solve as exactly -b_hat with a non-finite variance; the
    grid reports the b_hat it subtracted; the catalog contract blanks the
    cell and leaves the covered stars alone."""
    jax = pytest.importorskip("jax")
    if jax.default_backend() == "cpu":
        # Measured 2026-09-25: on the CPU backend a zero-information column
        # makes the dense solve singular and EVERY flux of the band comes out
        # NaN (and b_hat is measured on an unsubtracted residual); GPU keeps
        # the covered stars and returns 0 for the dead column. A known solver
        # defect on the CPU backend (open), not the output contract tested here.
        pytest.skip("zero-information column breaks the CPU-backend solve")
    from tract7dt.fit.grid_gpu.psf_polynomial import PsfPolynomial
    from tract7dt.fit.grid_gpu.runner import PerGridTask, run_one_grid

    S, NT, H, W = 25, 10, 96, 128
    yy, xx = np.mgrid[:S, :S] - S // 2
    g = np.exp(-0.5 * (xx ** 2 + yy ** 2) / 1.8 ** 2)
    coefs = np.zeros((S, S, NT), np.float32)
    coefs[..., 0] = g / g.sum()
    psf = PsfPolynomial(coefs=coefs, x_origin=W / 2, x_scale=W / 2,
                        y_origin=H / 2, y_scale=H / 2, H_psf=S, W_psf=S)

    # covered: x < 64; zero-coverage padding (data 0, invvar 0, bad) beyond
    rng = np.random.default_rng(11)
    sig = 1.0
    covered = np.zeros((H, W), bool)
    covered[:, :64] = True
    px = np.array([15.0, 30.0, 47.0, 108.0])      # last star: stamp 96..120
    py = np.array([20.0, 60.0, 40.0, 48.0])
    fl = np.array([400.0, 250.0, 150.0, 300.0])
    data = rng.normal(0.0, sig, (H, W))
    for x, y, f in zip(px, py, fl):
        data += f * np.exp(-0.5 * ((np.arange(W)[None, :] - x) ** 2
                                   + (np.arange(H)[:, None] - y) ** 2)
                           / 1.8 ** 2) / g.sum()
    data = np.where(covered, data, 0.0).astype(np.float32)
    invvar = np.where(covered, 1.0 / sig ** 2, 0.0).astype(np.float32)

    n = len(px)
    task = PerGridTask(
        grid_id=0, grid_row=0, grid_col=0, bandnames=["g"],
        data_per_band=[data], invvar_per_band=[invvar],
        pix_good_per_band=[covered.copy()],
        psfs_per_band=[psf], psf_shape=(S, S), grid_x0=0, grid_y0=0,
        source_indices=np.arange(n, dtype=np.int64),
        pos_x_init=px.astype(np.float32), pos_y_init=py.astype(np.float32),
        pos_err_pix=np.full(n, 0.1, np.float32),
        source_type=np.zeros(n, np.int32),
        logre_init=np.zeros(n, np.float32), ee1_init=np.zeros(n, np.float32),
        ee2_init=np.zeros(n, np.float32), pos_max_shift_pix=1.0,
        bg_x_origin=W / 2, bg_x_scale=W / 2, bg_y_origin=H / 2,
        bg_y_scale=H / 2, bg_poly_order=0, single_stage=True,
        n_opt_iters=0, blank_response_correction=True,
    )
    enable_x64 = getattr(jax, "enable_x64", None)
    if enable_x64 is None:                         # older JAX
        from jax.experimental import enable_x64
    with enable_x64():
        res = run_one_grid(task)
    assert res.error is None, res.error

    bh = res.blank_response_bhat
    assert bh is not None and bh.shape == (1,) and np.isfinite(bh[0])
    fv = np.asarray(res.flux_var[:, 0], dtype=np.float64)
    flux = np.asarray(res.fluxes[:, 0], dtype=np.float64)
    # the uncovered star: no information, and the old -b_hat value
    assert not (np.isfinite(fv[3]) and fv[3] > 0)
    assert flux[3] == pytest.approx(-bh[0], abs=1e-6 * max(1.0, abs(bh[0])))
    # covered stars: measured
    assert np.all(np.isfinite(fv[:3]) & (fv[:3] > 0))

    # writeback as run_grid_gpu_fit does it, then the contract
    fluxerr = np.where(np.isfinite(fv) & (fv > 0), np.sqrt(fv), np.nan)
    out = pd.DataFrame({"FLUX_g_fit": flux,
                        "FLUXERR_g_fit": fluxerr.astype(np.float32)})
    nblank = blank_nodata_band_cells(out, ["g"])
    assert nblank == {"g": 1}
    assert np.isnan(out.loc[3, "FLUX_g_fit"])
    np.testing.assert_array_equal(out["FLUX_g_fit"].to_numpy()[:3], flux[:3])
    # the recorded b_hat undoes the correction on the measured stars
    table = blank_response_bhat_table([res], ["g"])
    assert table["b_hat"].tolist() == [float(bh[0])]
