"""Reporting error columns of the grid GPU fit.

``add_error_columns`` adds, to the merged catalog of ``run_grid_gpu_fit``, the reporting error columns
built from the fit's own outputs: ``FLUXERR_TOT_{b}_fit`` (statistical + confusion + source Poisson), the
geometric ``*_err_corr`` (correlated coadd noise, LEG C) and ``*_err_tot`` (LEG C x LEG D chi2_nu), and the
shape-marginalised ``FLUXERR_MARG_{b}_fit``, ``FLUX_MARG_FRAC_{b}_fit`` and ``FLUXERR_MARG_TOT_{b}_fit``.
``_source_poisson_gains`` derives the per-band effective gain of the source-Poisson term.

M2 unit 0.6 (card 024, impl_plan 1.2) moved this code here unchanged from ``fit/grid_gpu/pipeline.py``
(lines 3112-3353 and 1351-1439 at c6dc1c1), so that the error columns and the solve pipeline can change in
separate streams. ``tests/test_error_columns_move.py`` holds the pre-move output as its expectation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _source_poisson_gains(cfg, bandnames, sp_cfg, log):
    """Per-band EFFECTIVE gain (detected-electron equivalents per INTERNAL
    zp_ref-scaled count) for the source-Poisson error term.

    UNIVERSAL HEADER CONTRACT (uniphot): the tool needs g_eff = electrons
    per pixel unit OF THE DELIVERED IMAGE (stacking/rescaling included).
    Resolution order:
      1) config override map error_model.source_poisson.gain_eff
         ({band: e-/native-ADU}) — always wins (any instrument);
      2) a well-formed product: GAIN or EGAIN > 5 taken as the delivered
         image's effective e-/ADU (standard FITS meaning; new py7DT);
      3) 7DT product shims (documented conventions, auto-detected):
         (a) UDS-style deep stack: GAIN = N x EGAIN (EGAIN = raw single
             gain) -> g = GAIN;
         (b) legacy gpPy .com.fits: EGAIN = (2/3) N g1, FLXSCALE~0.01
             dropped -> g = 1.5 x EGAIN / 0.01;
      4) otherwise: warn + skip the band (never guess silently).
    A MEDIAN combine pays the exact information tax (var_median/var_mean:
    1.348 at N=3, asymptotically pi/2). Internal units: divide by the
    zp_ref load scale 10**((zp_ref - ZP_hdr)/2.5).
    """
    import numpy as _np
    from astropy.io import fits as _fits
    over = sp_cfg.get("gain_eff") or {}
    # The images were loaded at 10**((zp_ref - ZP_hdr)/2.5); until R14 the
    # 25.0 default was hard-coded here, which mis-scaled the gain (and the
    # source-Poisson term) of any run with another image_scaling.zp_ref.
    zp_ref = float(cfg.get("image_scaling", {}).get("zp_ref", 25.0))
    paths = {}
    try:
        bset = set(bandnames)
        for line in open(cfg["inputs"]["image_list_file"]):
            p = line.strip()
            if p and not p.startswith("#"):
                # band token position varies by product naming; match any
                # underscore/dot token against the run's own band list.
                toks = p.rsplit("/", 1)[-1].replace(".", "_").split("_")
                for tk in toks:
                    if tk in bset:
                        paths[tk] = p
                        break
    except Exception as e:  # noqa: BLE001
        log.warning("source_poisson: image list unreadable (%s)", e)
    gains = {}
    for bn in bandnames:
        try:
            h = _fits.getheader(paths[bn]) if bn in paths else {}
            zp = float(h.get("ZP_AUTO", _np.nan))
            scale = 10.0 ** ((zp_ref - zp) / 2.5) if _np.isfinite(zp) else None
            eg = float(h.get("EGAIN", _np.nan))
            gn = float(h.get("GAIN", _np.nan))
            comb = str(h.get("COMBINET", "")).upper()
            if bn in over and over[bn]:
                g_nat, how, ncmb = float(over[bn]), "override", None
            elif (_np.isfinite(gn) and gn > 5.0 and _np.isfinite(eg)
                    and eg < 5.0 and gn / eg > 2.0):
                # GAIN>5 alongside a sub-5 EGAIN = the combine-scaled-GAIN
                # pair convention (UDS-style); the ratio is the EFFECTIVE
                # frame count and need NOT be integer (mixed exposures).
                g_nat, how, ncmb = gn, "uds-style N*EGAIN", round(gn / eg)
            elif _np.isfinite(eg) and eg > 5.0:
                g_nat, how, ncmb = eg, "EGAIN effective", None
            elif _np.isfinite(gn) and 5.0 < gn < 500.0 and not _np.isfinite(eg):
                g_nat, how, ncmb = gn, "GAIN effective", None
            elif _np.isfinite(eg) and 0.05 < eg < 5.0:
                g_nat = 1.5 * eg / 0.01
                how = "legacy gpPy (1.5*EGAIN/0.01)"
                ncmb = round(float(h.get("EXPTIME", 0.0)) / 100.0) or None
            else:
                log.warning("source_poisson[%s]: no usable gain keywords "
                            "(GAIN=%s EGAIN=%s) and no override -> band "
                            "SKIPPED", bn, gn, eg)
                continue
            tax = 1.0
            if comb == "MEDIAN" and how != "override":
                tax = {1: 1.0, 2: 1.0, 3: 1.348, 4: 1.193,
                       5: 1.435}.get(ncmb, float(_np.pi / 2.0))
            if scale is None:
                log.warning("source_poisson[%s]: no ZP_AUTO -> cannot "
                            "convert to internal units, band SKIPPED", bn)
                continue
            gains[bn] = (g_nat / tax) / scale
            log.info("source_poisson[%s]: g_eff %.1f e-/native-ADU (%s, "
                     "median-tax %.3f, N=%s) -> %.2f per internal count",
                     bn, g_nat, how, tax, ncmb, gains[bn])
        except Exception as e:  # noqa: BLE001
            log.warning("source_poisson[%s]: derivation failed (%s) -> "
                        "band skipped", bn, e)
    return gains


def add_error_columns(out: pd.DataFrame, *, cfg: dict[str, Any], bandnames, bandnames_with_psf,
                      error_model_cfg: dict[str, Any], confusion_enabled: bool, correlated_noise_enabled: bool,
                      fcorr_by_band: dict[str, float], chi2_nu_enabled: bool, island_meta, param_errors_enabled: bool,
                      emit_flux_marg: bool, log) -> None:
    """Add the reporting error columns to ``out`` in place (column formulas in the comments below).

    The arguments are the values ``run_grid_gpu_fit`` holds at that point: the resolved config, the catalog
    bands and the bands with a PSF, the ``compute.gpu.error_model`` block and the switches read from it,
    the per-band F_corr, the island-stage meta (None outside island mode) and the logger.
    """
    # ---- Error-model output columns (task #27) -----------------------
    # FLUXERR_TOT_* = sqrt(FLUXERR^2 + FLUXERR_CONF^2). Where the
    # confusion measurement is unavailable (NaN: masked annulus, grid
    # edge, or a failed grid) the total falls back to FLUXERR unchanged.
    # FLUXERR_* itself is untouched (backward compat; it remains the
    # marginal statistical error of the linear solve).
    sp_cfg = (error_model_cfg.get("source_poisson", {}) or {})
    sp_gains = (_source_poisson_gains(cfg, bandnames, sp_cfg, log)
                if bool(sp_cfg.get("enabled", False)) else {})

    def _sp_var(bn):
        # source-Poisson variance term (internal units); 0 when disabled/
        # underivable. REPORTING side only -- never the fit invvar.
        if bn not in sp_gains:
            return 0.0
        fl = pd.to_numeric(out[f"FLUX_{bn}_fit"],
                           errors="coerce").to_numpy(dtype=np.float64)
        return np.where(np.isfinite(fl), np.clip(fl, 0.0, None)
                        / sp_gains[bn], 0.0)

    if confusion_enabled:
        for bn in bandnames:
            fe = pd.to_numeric(
                out[f"FLUXERR_{bn}_fit"], errors="coerce").to_numpy(dtype=np.float64)
            fc = pd.to_numeric(
                out[f"FLUXERR_CONF_{bn}_fit"], errors="coerce").to_numpy(dtype=np.float64)
            fc2 = np.where(np.isfinite(fc), fc * fc, 0.0)
            out[f"FLUXERR_TOT_{bn}_fit"] = np.sqrt(
                fe * fe + fc2 + _sp_var(bn)).astype(np.float32)

    # ---- LEG C: correlated-coadd-noise-inflated GEOMETRIC error columns -----
    # A per-band SCALAR variance inflation F_corr (measured from the coadd's
    # own sky/residual above, with a confusion-rejecting ACF-SHAPE gate)
    # applied ONLY to the reported GEOMETRIC errors, as SEPARATE
    # sqrt(F_corr)-scaled columns. FLUX / FLUXERR / *_err and the fit weights
    # are all UNCHANGED. Re-scope 2026-07-08: F_corr is NOT applied to the
    # flux error (the flux budget is #27-only); it scales only the nonlinear
    # (pos/shape/n) covariances, which #27's sigma_conf does not touch.
    if correlated_noise_enabled and fcorr_by_band:
        # Nonlinear (island pos/shape/n) errors: the same white-weighted
        # sandwich gives sqrt(F_corr) on each covariance. The island fit
        # is JOINT-multiband, so use a per-source EFFECTIVE F_corr =
        # SNR^2-weighted mean of the per-band F_corr over the bands that
        # constrain the source. For a point source band b contributes to
        # the position/shape Hessian in proportion to flux_b^2 * (flux
        # Fisher info) = (FLUX_b/FLUXERR_b)^2 = SNR_b^2, so SNR^2 is the
        # correct per-band weight for the joint-chi2 curvature (documented
        # choice). sqrt(F_corr_eff) then scales every nonlinear 1-sigma.
        n_rows = len(out)
        wsum = np.zeros(n_rows, dtype=np.float64)   # sum_b SNR_b^2
        fnum = np.zeros(n_rows, dtype=np.float64)   # sum_b SNR_b^2 * F_b
        for bn in bandnames_with_psf:
            fc_b = float(fcorr_by_band.get(bn, 1.0))
            fl = pd.to_numeric(
                out[f"FLUX_{bn}_fit"], errors="coerce").to_numpy(dtype=np.float64)
            fe = pd.to_numeric(
                out[f"FLUXERR_{bn}_fit"], errors="coerce").to_numpy(dtype=np.float64)
            snr2 = np.where(np.isfinite(fl) & np.isfinite(fe) & (fe > 0),
                            (fl / fe) ** 2, 0.0)
            wsum += snr2
            fnum += snr2 * fc_b
        # F_corr_eff: SNR^2-weighted mean where any band constrains the
        # source; else fall back to the median per-band F_corr.
        fmed = float(np.median(list(fcorr_by_band.values())))
        f_eff = np.where(wsum > 0, fnum / np.maximum(wsum, 1e-30), fmed)
        sqrt_feff = np.sqrt(np.maximum(f_eff, 1.0))
        # Which nonlinear *_err columns exist to scale: island mode adds the
        # full pos/shape set (task #28a); sersic_n_err exists in all modes.
        if island_meta is not None and param_errors_enabled:
            err_bases = ("x_pix_white_err", "y_pix_white_err",
                         "Re_err", "ELL_err", "THETA_err", "sersic_n_err")
        else:
            err_bases = ("sersic_n_err",)
        for base in err_bases:
            raw = pd.to_numeric(
                out[base], errors="coerce").to_numpy(dtype=np.float64)
            out[f"{base}_corr"] = (sqrt_feff * raw).astype(np.float32)

    # ---- LEG D + combined C x D total GEOMETRIC error columns (task #20) ----
    # sigma_tot = sigma * sqrt(F_corr_eff) * sqrt(max(1, chi2_nu_nl)) for the
    # nonlinear (pos/shape/n) errors ONLY. LEG C (sqrt(F_corr_eff)) folds in
    # the correlated coadd noise; LEG D (sqrt(max(1, chi2_nu_nl))) folds in
    # confusion + model mismatch from each island's OWN joint residual against
    # the MARGINAL (sky-only) variance. The two are INDEPENDENT (chi2_nu ~ 1
    # under pure correlation, so no double-count) and MULTIPLY on sigma.
    # SEPARATE *_err_tot columns; the raw *_err (task #28a) is UNCHANGED.
    # Re-scope 2026-07-08: there is NO flux *_TOT_NEW column -- the flux error
    # budget is #27-only (FLUXERR / FLUXERR_CONF / FLUXERR_TOT). C defaults to
    # 1 when correlated_noise is off, D to 1 where chi2_nu_nl is unavailable.
    if chi2_nu_enabled:
        n_rows = len(out)

        def _dfac(colname):
            if colname in out.columns:
                cnu = pd.to_numeric(
                    out[colname], errors="coerce").to_numpy(dtype=np.float64)
            else:
                cnu = np.full(n_rows, np.nan)
            return np.sqrt(np.where(np.isfinite(cnu), np.maximum(cnu, 1.0), 1.0))

        # NONLINEAR (island pos/shape/n; sersic_n in all modes):
        #     F_corr_eff (SNR^2-weighted per-band F_corr, identical to LEG C's
        #     *_err_corr weighting) x the per-island chi2_nu_nl.
        wsum = np.zeros(n_rows, dtype=np.float64)
        fnum = np.zeros(n_rows, dtype=np.float64)
        for bn in bandnames_with_psf:
            fc_b = (float(fcorr_by_band.get(bn, 1.0))
                    if correlated_noise_enabled else 1.0)
            fl = pd.to_numeric(
                out[f"FLUX_{bn}_fit"], errors="coerce").to_numpy(dtype=np.float64)
            fe = pd.to_numeric(
                out[f"FLUXERR_{bn}_fit"], errors="coerce").to_numpy(dtype=np.float64)
            snr2 = np.where(np.isfinite(fl) & np.isfinite(fe) & (fe > 0),
                            (fl / fe) ** 2, 0.0)
            wsum += snr2
            fnum += snr2 * fc_b
        fmed = (float(np.median(list(fcorr_by_band.values())))
                if (correlated_noise_enabled and fcorr_by_band) else 1.0)
        f_eff = np.where(wsum > 0, fnum / np.maximum(wsum, 1e-30), fmed)
        c_fac_nl = np.sqrt(np.maximum(f_eff, 1.0))
        d_fac_nl = _dfac("chi2_nu_nl")
        if island_meta is not None and param_errors_enabled:
            err_bases = ("x_pix_white_err", "y_pix_white_err",
                         "Re_err", "ELL_err", "THETA_err", "sersic_n_err")
        else:
            err_bases = ("sersic_n_err",)
        for base in err_bases:
            raw = pd.to_numeric(
                out[base], errors="coerce").to_numpy(dtype=np.float64)
            out[f"{base}_tot"] = (c_fac_nl * d_fac_nl * raw).astype(np.float32)

    # ---- D2 (root-fix D): SHAPE-marginalized flux error columns ------------
    # FLUXERR is CONDITIONAL on the freely-fitted shape/position; for low-S/N
    # extended sources the flux-shape covariance is the dominant missing term
    # (D-diagnosis: UDS S/N<=30 pull_tot 2.46 -> 1.07 after decorrelating vs
    # dlogRe/dell/dx/dy). The island stage propagates that covariance into the
    # flux via the exact Schur increment (island_meta['flux_marg_var'], the
    # VarPro amplitude sensitivity squared through Cov(shape)); the additive
    # columns are
    #   FLUXERR_MARG_{b}     = sqrt(FLUXERR^2 + dvar)                 [statistical]
    #   FLUXERR_MARG_TOT_{b} = sqrt(FLUXERR^2 + dvar*G + FLUXERR_CONF^2 + SP)
    #                          * sqrt(max(1, rchisq))  [rchisq_scale_marg_tot, default on]
    # with G = max(1,F_corr_eff)*max(1,chi2_nu_nl) the SAME per-source geometric
    # variance inflation the *_err_tot columns use (the increment IS a geometric-
    # error quantity propagated into flux), FLUXERR_CONF the #27 confusion
    # term, SP the source-Poisson variance (source_poisson, default on) and
    # rchisq ONE per-source value applied to every band. STAR rows carry
    # dvar==0 exactly (no free shape block) -> FLUXERR_MARG == FLUXERR
    # bit-for-bit, and FLUXERR_MARG_TOT == FLUXERR_TOT * sqrt(max(1, rchisq)),
    # i.e. equal to FLUXERR_TOT only where rchisq <= 1. Existing
    # FLUXERR / FLUXERR_TOT are UNTOUCHED (additive-only). Giants / off-island:
    # dvar is NaN -> MARG columns NaN (disclosed by island_giant).
    if emit_flux_marg:
        n_rows = len(out)
        fmv = island_meta.get("flux_marg_var")          # (n_total, n_bands)
        fmv_bands = {bn: bi for bi, bn in enumerate(
            island_meta.get("flux_win_bands", []))}
        # Per-source geometric inflation G (variance) = F_corr_eff * chi2_nu_nl,
        # both floored at 1 -- identical SNR^2-weighting to the *_err_tot block.
        wsum = np.zeros(n_rows, dtype=np.float64)
        fnum = np.zeros(n_rows, dtype=np.float64)
        for bn in bandnames_with_psf:
            fc_b = (float(fcorr_by_band.get(bn, 1.0))
                    if correlated_noise_enabled else 1.0)
            fl = pd.to_numeric(
                out[f"FLUX_{bn}_fit"], errors="coerce").to_numpy(dtype=np.float64)
            fe = pd.to_numeric(
                out[f"FLUXERR_{bn}_fit"], errors="coerce").to_numpy(dtype=np.float64)
            snr2 = np.where(np.isfinite(fl) & np.isfinite(fe) & (fe > 0),
                            (fl / fe) ** 2, 0.0)
            wsum += snr2
            fnum += snr2 * fc_b
        fmed = (float(np.median(list(fcorr_by_band.values())))
                if (correlated_noise_enabled and fcorr_by_band) else 1.0)
        f_eff = np.where(wsum > 0, fnum / np.maximum(wsum, 1e-30), fmed)
        g_corr = np.maximum(f_eff, 1.0)
        if chi2_nu_enabled and "chi2_nu_nl" in out.columns:
            cnu = pd.to_numeric(
                out["chi2_nu_nl"], errors="coerce").to_numpy(dtype=np.float64)
            g_corr = g_corr * np.where(
                np.isfinite(cnu), np.maximum(cnu, 1.0), 1.0)
        # #18 rchisq TOT scaling (default on since 2026-07-22; off => the
        # pre-#18 bar): sqrt(max(1, per-source rchisq)), one factor for all bands.
        rchisq_tot_scale = None
        if (bool(error_model_cfg.get("rchisq_scale_marg_tot", False))
                and "rchisq" in out.columns):
            _rq = pd.to_numeric(
                out["rchisq"], errors="coerce").to_numpy(dtype=np.float64)
            rchisq_tot_scale = np.sqrt(
                np.where(np.isfinite(_rq), np.maximum(_rq, 1.0), 1.0))
        for bn in bandnames:
            fe = pd.to_numeric(
                out[f"FLUXERR_{bn}_fit"], errors="coerce").to_numpy(dtype=np.float64)
            dv = (fmv[:, fmv_bands[bn]].astype(np.float64)
                  if (fmv is not None and bn in fmv_bands)
                  else np.full(n_rows, np.nan))
            # Statistical MARG: dvar==0 (star / no free shape) -> FLUXERR
            # exactly (avoid the sqrt(fe^2) round-trip); dvar NaN -> NaN.
            marg = np.where(dv == 0.0, fe, np.sqrt(fe * fe + dv))
            # Store float32-PRECISION values in a float64 column so the CSV
            # string repr matches FLUXERR (a float64 column holding float32
            # values via its NaN-init + .loc-assign) -> STAR rows
            # (marg == fe) are byte-identical, not merely equal-valued.
            out[f"FLUXERR_MARG_{bn}_fit"] = (
                marg.astype(np.float32).astype(np.float64))
            # Degeneracy DISCLOSURE (#7 free-n): fractional flux-variance
            # inflation from marginalizing over the shape + Sersic-n block,
            # FLUX_MARG_FRAC = dv / FLUXERR^2 = (FLUXERR_MARG/FLUXERR)^2 - 1.
            # A parameter-free per-source indicator of how flux-degenerate the
            # fit is with its (Re, ellipticity, n) -- large on railed large-Re
            # / free-n rows where the reported flux is extrapolation-coupled to
            # the shape (the FLUXERR_MARG already WIDENS the error for this;
            # this column just surfaces the magnitude as a ready cut). Pure
            # disclosure (propagation, not comparison): 0 for STAR / no-free-
            # shape (dv==0), NaN off-island. Apply your own cut (flags-are-not-
            # fixes).
            with np.errstate(divide="ignore", invalid="ignore"):
                fmf = np.where(dv == 0.0, 0.0,
                               np.where(fe > 0.0, dv / (fe * fe), np.nan))
            out[f"FLUX_MARG_FRAC_{bn}_fit"] = fmf.astype(np.float32)
            # TOTAL MARG: add the F_corr*chi2_nu-inflated increment + the #27
            # confusion term (FLUXERR_CONF). dvar==0 -> FLUXERR_TOT exactly,
            # before the #18 rchisq scale below.
            fc_col = f"FLUXERR_CONF_{bn}_fit"
            if confusion_enabled and fc_col in out.columns:
                fc = pd.to_numeric(
                    out[fc_col], errors="coerce").to_numpy(dtype=np.float64)
                fc2 = np.where(np.isfinite(fc), fc * fc, 0.0)
            else:
                fc2 = np.zeros(n_rows, dtype=np.float64)
            spv = _sp_var(bn)
            tot = np.where(dv == 0.0, np.sqrt(fe * fe + fc2 + spv),
                           np.sqrt(fe * fe + dv * g_corr + fc2 + spv))
            if rchisq_tot_scale is not None:
                # #18 (2026-07-22): per-source rchisq error-scaling of the
                # TOT bar -- standard chi2-based error rescale, no tuned
                # constant. Referee-measured on the full-field crossband
                # truth gate: stamped silent-wrong (|pull_TOT|>3) 29 -> 7
                # at median collateral x1.00-1.10 (the 22 fixed rows carry
                # rchisq 3-98 that the stamp/TOT did not reflect).
                tot = tot * rchisq_tot_scale
            out[f"FLUXERR_MARG_TOT_{bn}_fit"] = tot.astype(np.float32)
