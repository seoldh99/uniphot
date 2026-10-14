"""The error-column assembly moved out of the grid pipeline unchanged (M2 unit 0.6, card 024 impl_plan 1.2).

M2 unit 0.6 moves the reporting error columns (FLUXERR_TOT, the geometric *_err_corr / *_err_tot,
FLUXERR_MARG, FLUX_MARG_FRAC, FLUXERR_MARG_TOT) and the source-Poisson gain derivation from
``fit/grid_gpu/pipeline.py`` into ``fit/grid_gpu/error_columns.py``, so that one stream owns the error
columns and another the solve pipeline. Values must not change.

Expected values are independent of the moved code: ``data/error_columns_c6dc1c1.npz`` was written by the
code BEFORE the move -- commit c6dc1c1, the block of ``run_grid_gpu_fit`` (pipeline.py 3112-3353) and
``_source_poisson_gains`` (1351-1440) executed as they stood -- on the synthetic inputs built below
(generator: the development script ``make_error_columns_golden.py``, not distributed). Every output column must
match it byte for byte, in the same order and dtype, and the gain derivation must give the same gains and
the same log lines. The scenarios switch every branch of the block (confusion, correlated-noise F_corr,
chi2_nu, island / non-island error bases, the shape-marginalised columns with star / galaxy / off-island
rows, the rchisq scale, the source-Poisson term with every header convention) and include NaN fluxes,
negative fluxes, zero / NaN flux errors, a band without a PSF and rows with no S/N weight in any band
(the median F_corr fallback).

When a later M2 unit changes these columns on purpose (c.1 gain contract, c.3 rchisq scale, c.4 PSF term,
b.5 giant rows), that unit replaces the affected expectation with its own independent one.
"""
from __future__ import annotations

import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GOLDEN = Path(__file__).resolve().parent / "data" / "error_columns_c6dc1c1.npz"

BANDS = ["g", "i", "m400", "m500", "m525", "r", "u", "z"]
# one header convention per band, covering every branch of the gain derivation (impl_plan 1.2, c.1 later
# deletes the guessing branches): override, UDS-style N*EGAIN with a MEDIAN combine, EGAIN effective,
# GAIN effective, legacy gpPy with a MEDIAN combine of 3, no gain keywords, no ZP_AUTO; "u" has no image.
HEADERS = {
    "g": {"ZP_AUTO": 26.1, "EGAIN": 1.2},
    "i": {"ZP_AUTO": 25.4},
    "m400": {"ZP_AUTO": 24.2, "EGAIN": 0.5, "EXPTIME": 300.0, "COMBINET": "MEDIAN"},
    "m500": {"ZP_AUTO": 24.9, "GAIN": 2750.0, "EGAIN": 1.83, "COMBINET": "MEDIAN"},
    "m525": {"ZP_AUTO": 25.3, "EGAIN": 57.0},
    "r": {"ZP_AUTO": 26.6, "GAIN": 40.0},
    "z": {"GAIN": 40.0},
}
GAIN_OVERRIDE = {"g": 57.0}

SCENARIOS = {
    # name: (island, param_errors, emit_flux_marg, confusion, correlated, fcorr, chi2, rchisq_scale, sp)
    "all_on": (True, True, True, True, True, True, True, True, True),
    "no_island": (False, True, False, True, True, True, True, False, True),
    "all_off": (False, False, False, False, False, False, False, False, False),
    "island_partial": (True, False, True, False, False, False, True, False, True),
    "fcorr_empty": (True, True, True, True, True, False, False, True, False),
    # every switch on, plus rows without a finite flux in any PSF band: no S/N weight anywhere, so the
    # per-source F_corr falls back to the median per-band F_corr (review_0.6.md C1)
    "fmed_rows": (True, True, True, True, True, True, True, True, True),
}
NO_SNR_ROWS = {"fmed_rows"}


def write_images(tmp_dir: Path) -> Path:
    """Header-only FITS files + image list; returns the image list path."""
    from astropy.io import fits

    lines = []
    for bn, hdr in HEADERS.items():
        fp = tmp_dir / f"calib_7DT01_T00001_{bn}_com.fits"
        fits.PrimaryHDU(np.zeros((2, 2), np.float32), header=fits.Header(hdr)).writeto(fp, overwrite=True)
        lines.append(str(fp))
    lst = tmp_dir / "image_list.txt"
    lst.write_text("# synthetic\n" + "\n".join(lines) + "\n")
    return lst


def make_scenario(name: str, tmp_dir: Path, n: int = 160) -> dict:
    """Synthetic inputs of the error-column block for one scenario (deterministic)."""
    island, perr, marg, conf, corr, fcorr_on, chi2, rq_scale, sp = SCENARIOS[name]
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    bands_psf = [b for b in BANDS if b != "u"]           # "u": no PSF, still a catalog band
    out = pd.DataFrame({"ID": [f"s{i:04d}" for i in range(n)]})
    kind = rng.choice(np.array(["star", "gal", "off"]), size=n, p=[0.5, 0.35, 0.15])
    for bn in BANDS:
        flux = rng.lognormal(6.0, 1.5, n)
        flux[rng.random(n) < 0.05] *= -0.3                  # negative fluxes (clipped in the Poisson term)
        flux[rng.random(n) < 0.05] = np.nan
        ferr = np.sqrt(np.abs(flux) / 50.0 + 25.0)
        ferr[rng.random(n) < 0.03] = 0.0
        ferr[rng.random(n) < 0.03] = np.nan
        # pipeline columns are float64 columns holding float32 values (NaN init + .loc assign)
        out[f"FLUX_{bn}_fit"] = flux.astype(np.float32).astype(np.float64)
        out[f"FLUXERR_{bn}_fit"] = ferr.astype(np.float32).astype(np.float64)
        if conf:
            fc = rng.gamma(2.0, 2.0, n)
            fc[rng.random(n) < 0.1] = np.nan
            out[f"FLUXERR_CONF_{bn}_fit"] = fc.astype(np.float32)
    for base in ("x_pix_white_err", "y_pix_white_err", "Re_err", "ELL_err", "THETA_err"):
        if island and perr:
            v = rng.gamma(2.0, 0.05, n)
            v[kind == "off"] = np.nan
            out[base] = v.astype(np.float32)
    sn = rng.gamma(2.0, 0.1, n)
    sn[(kind != "gal") | (rng.random(n) < 0.3)] = np.nan
    out["sersic_n_err"] = sn.astype(np.float32)
    if chi2 and island:
        c = rng.lognormal(0.0, 0.6, n)
        c[kind == "off"] = np.nan
        out["chi2_nu_nl"] = c.astype(np.float32)
    if island:
        rq = rng.lognormal(0.1, 0.7, n)
        rq[kind == "off"] = np.nan
        out["rchisq"] = rq.astype(np.float32)
    fcorr = ({bn: float(1.0 + rng.gamma(2.0, 0.2)) for bn in bands_psf[:-1]} if fcorr_on else {})
    if fcorr_on:
        fcorr[bands_psf[0]] = 0.8                          # F_corr < 1 is floored at 1
    island_meta = None
    if island:
        fwb = bands_psf[1:]                                 # flux_win_bands: not every band
        fmv = rng.gamma(1.5, 30.0, (n, len(fwb)))
        fmv[kind == "star"] = 0.0                           # no free shape -> exactly 0
        fmv[kind == "off"] = np.nan                         # giants / off-island
        island_meta = {"flux_marg_var": fmv.astype(np.float32), "flux_win_bands": list(fwb)}
    if name in NO_SNR_ROWS:     # after every draw, so the other scenarios' inputs are unchanged
        rows = np.arange(n) % 8 == 3
        for bn in bands_psf:
            out.loc[rows, f"FLUX_{bn}_fit"] = np.nan
    lst = write_images(tmp_dir)
    cfg = {"inputs": {"image_list_file": str(lst)}, "image_scaling": {"zp_ref": 25.0}}
    error_model_cfg = {"rchisq_scale_marg_tot": bool(rq_scale),
                       "source_poisson": {"enabled": bool(sp), "gain_eff": dict(GAIN_OVERRIDE)}}
    return {"out": out, "cfg": cfg, "bandnames": list(BANDS), "bandnames_with_psf": bands_psf,
            "error_model_cfg": error_model_cfg, "confusion_enabled": bool(conf),
            "correlated_noise_enabled": bool(corr), "fcorr_by_band": fcorr, "chi2_nu_enabled": bool(chi2),
            "island_meta": island_meta, "param_errors_enabled": bool(perr), "emit_flux_marg": bool(marg)}


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(f"{record.levelname}|{record.getMessage()}")


def capture_logger(name: str):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    h = _ListHandler()
    log.handlers[:] = [h]
    return log, h


def _golden():
    return np.load(GOLDEN, allow_pickle=False)


# "all_off": no column is added when every switch is off (before and after the move).
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_error_columns_byte_identical_to_pre_move(scenario, tmp_path):
    from tract7dt.fit.grid_gpu.error_columns import add_error_columns

    kw = make_scenario(scenario, tmp_path)
    out = kw.pop("out")
    log, h = capture_logger(f"t_ec_{scenario}")
    add_error_columns(out, log=log, **kw)
    g = _golden()
    cols = [str(c) for c in g[f"{scenario}/__columns__"]]
    assert list(out.columns) == cols
    for c in cols:
        want = g[f"{scenario}/{c}"]
        got = out[c].to_numpy()
        if want.dtype.kind == "U":          # object (string) columns are stored as text
            assert got.dtype == object and np.array_equal(got.astype(str), want), c
            continue
        assert got.dtype == want.dtype, c
        assert got.tobytes() == want.tobytes(), c
    assert h.lines == [str(s) for s in g[f"{scenario}/__log__"]]


def test_source_poisson_gains_identical_to_pre_move(tmp_path):
    from tract7dt.fit.grid_gpu.error_columns import _source_poisson_gains

    lst = write_images(tmp_path)
    cfg = {"inputs": {"image_list_file": str(lst)}, "image_scaling": {"zp_ref": 25.0}}
    log, h = capture_logger("t_ec_gains")
    gains = _source_poisson_gains(cfg, list(BANDS), {"gain_eff": dict(GAIN_OVERRIDE)}, log)
    g = _golden()
    want = {str(b): float(v) for b, v in zip(g["gains/__bands__"], g["gains/__values__"])}
    assert sorted(gains) == sorted(want)
    for b, v in want.items():
        assert np.float64(gains[b]).tobytes() == np.float64(v).tobytes(), b
    assert h.lines == [str(s) for s in g["gains/__log__"]]
