"""
Reliability flag (``fit_is_reliable``, ``fit_quality_issues``) shared by the GPU fit.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger("tract7dt.merge")


def compute_reliability_flag(
    merged: pd.DataFrame, *, island_stalled_mode: str = "fail",
    island_giant_mode: str = "fail",
    high_rchisq_mode: str = "warn",
    blend_unresolved_mode: str = "fail",
    island_plateau_mode: str = "stalled",
) -> None:
    """Compute ``fit_is_reliable`` and ``fit_quality_issues`` columns in-place.

    This is the SHARED reliability logic used by both the CPU per-patch
    merge path AND the GPU grid pipeline.  Both paths should produce the
    same ``fit_is_reliable`` semantics so downstream tools (ZP, plots,
    science cuts) behave identically regardless of compute backend.

    Hard-fail conditions (``fit_is_reliable = False``):
      excluded         – source was excluded from fitting
      no_fit_result    – non-excluded source has no fitted position
      hit_max_iters    – optimizer exhausted iteration budget
      bound_stalled    – bounded parameter pinned at constraint limit
      bright_mask      – source inside a bright-source mask
      island_stalled   – island LM converged only via the reject-plateau prong
                         (``island_stalled_mode="fail"``, the default).  Such a
                         row is reported ``opt_converged=True`` by the island
                         solver but is NOT at a stationary point and carries a
                         measured median flux inflation of tens of percent on
                         crowded fields, so it must not pass as reliable.
      giant_frozen_geometry – the source is in a giant island (column
                         ``island_giant``: larger than the biggest island class,
                         more members than ``joint_nsrc_cap``, or too big for
                         GPU memory), which gets no island fit
                         (``island_giant_mode="fail"``, the default).  Its
                         RA/DEC/Re/ELL/THETA are the raw catalog seed, never
                         measured; pass 2 measures the flux at that frozen seed
                         geometry.  The row is reported ``opt_converged=True``
                         with ``position_stranded=False``, so it must not
                         certify as a clean reliable fit.
      no_usable_band   – GPU band-level contract (R06): the fitted source has
                         no band whose value is a measurement free of any
                         reason flag (``n_bands_usable == 0``).

    Soft warnings (``fit_quality_issues`` only):
      not_converged    – optimizer did not formally converge (but did not hit max iters)
      all_psf_fallback – all bands used analytic fallback PSF
      shape_frozen     – DEV/SERSIC row whose shape (Re/ELL/THETA) was frozen at
                         the catalog seed by a shape-freezing solver
                         (per_source/joint_lsqr); the flux is a real measurement
                         conditioned on the assumed shape (informational only)
      island_stalled   – same signal, surfaced as a warning only when
                         ``island_stalled_mode="warn"`` (keeps the row's
                         ``fit_is_reliable`` untouched)
      giant_frozen_geometry – same signal, surfaced as a warning only when
                         ``island_giant_mode="warn"``
      island_diverged  – #35 disclosure (island mode only): the island's LM
                         trajectory contained STALL_REJECT_K consecutive
                         rejected steps whose clip/bound-limited proposal was
                         IDENTICAL each time (clip-saturated runaway class).
                         ALWAYS soft — deliberately no fail mode: under the
                         default ``stall_count_mode: proposals`` every
                         diverged row is already ``island_stalled`` (which
                         hard-fails above); under ``"changed"`` the tag marks
                         rows that continued past the frozen-proposal episode
                         without any reliability change.
      blend_suspect    – item-4 disclosure (island mode, only when
                         ``rchisq_flag_thresh`` is set): fit-vs-seed
                         displacement above the run's median PSF sigma AND
                         high rchisq — the signature of an uncataloged point
                         source blended inside the bright-mask match radius
                         being absorbed by this source's fit. ALWAYS soft.

    ``island_stalled_mode`` / ``island_giant_mode`` select how the
    ``island_stalled`` / ``island_giant`` columns (present only in
    ``position_solver: island`` runs) enter the contract: ``"fail"`` (default,
    honest) hard-fails; ``"warn"`` tags it softly; ``"ignore"`` restores
    pre-wiring behaviour (no signal).

    Missing columns are gracefully handled (treated as "no signal") — so CPU /
    legacy catalogs, which never carry ``island_stalled`` / ``island_giant``,
    are unaffected regardless of the mode arguments.
    """
    n = len(merged)
    if n == 0:
        merged["fit_is_reliable"] = pd.Series(dtype=bool)
        merged["fit_quality_issues"] = pd.Series(dtype=str)
        return

    hard: list[tuple[str, np.ndarray]] = []
    soft: list[tuple[str, np.ndarray]] = []

    if "excluded_any" in merged.columns:
        excluded = merged["excluded_any"].fillna(False).to_numpy(dtype=bool)
    else:
        excluded = np.zeros(n, dtype=bool)
    hard.append(("excluded", excluded))

    if "x_pix_white_fit" in merged.columns:
        no_fit = merged["x_pix_white_fit"].isna().to_numpy(dtype=bool)
    else:
        sb_x = [c for c in merged.columns
                if re.match(r"^x_pix_white_.+_fit$", c) and c != "x_pix_white_fit"]
        no_fit = (
            merged[sb_x].isna().all(axis=1).to_numpy(dtype=bool)
            if sb_x else np.ones(n, dtype=bool)
        )
    no_fit = no_fit & ~excluded
    hard.append(("no_fit_result", no_fit))

    if "opt_hit_max_iters" in merged.columns:
        hmi = merged["opt_hit_max_iters"].fillna(False).to_numpy(dtype=bool)
    else:
        hmi_cols = [c for c in merged.columns if re.match(r"^opt_hit_max_iters_.+$", c)]
        hmi = (
            merged[hmi_cols].fillna(False).astype(bool).any(axis=1).to_numpy(dtype=bool)
            if hmi_cols else np.zeros(n, dtype=bool)
        )
    hard.append(("hit_max_iters", hmi))

    if "flag_bound_stalled" in merged.columns:
        stalled = merged["flag_bound_stalled"].fillna(False).to_numpy(dtype=bool)
    else:
        sc = [c for c in merged.columns
              if c.startswith("flag_bound_stalled_") and "which_parameter" not in c]
        stalled = (
            merged[sc].fillna(False).astype(bool).any(axis=1).to_numpy(dtype=bool)
            if sc else np.zeros(n, dtype=bool)
        )
    hard.append(("bound_stalled", stalled))

    if "affected_by_bright_mask" in merged.columns:
        hard.append((
            "bright_mask",
            merged["affected_by_bright_mask"].fillna(False).to_numpy(dtype=bool),
        ))

    # island_stalled: island-mode-only signal. A stalled island reports
    # opt_converged=True (stall-plateau prong) but is not at a minimum and is
    # measurably flux-inflated on crowded fields, so by default it hard-fails
    # reliability rather than passing silently. Absent on CPU/legacy catalogs.
    if island_stalled_mode not in ("fail", "warn", "ignore"):
        raise ValueError(
            f"island_stalled_mode must be 'fail'|'warn'|'ignore', got {island_stalled_mode!r}"
        )
    if island_plateau_mode not in ("stalled", "converged"):
        raise ValueError(
            f"island_plateau_mode must be 'stalled'|'converged', "
            f"got {island_plateau_mode!r}")
    if island_stalled_mode != "ignore" and "island_stalled" in merged.columns:
        isl_stalled = merged["island_stalled"].fillna(False).to_numpy(dtype=bool)
        # Stall-yield (2026-07-28, user decision): plateau-certified stalls
        # (island_plateau: the probability-convergence certificate -- no
        # damped step on the full lambda ladder improves chi2 by more than
        # the dlnp quantum) are CONVERGED, not stuck. "converged" (opt-in;
        # RECOMMENDED with confusion_whitening crossband, where 84% of
        # stalls are certified and the certified pool is truth-clean at
        # snr>=100 + on the star oracle) removes them from the stall signal
        # and tags them softly instead ("island_plateau" in
        # fit_quality_issues). Measured caveat (disclosed): faint certified
        # GALAXIES (snr<=30) include displaced-valley plateaus (n=27 gate
        # sample, |dmag| 851 vs converged 426 mmag). Default "stalled"
        # keeps the pre-2026-07-28 behaviour bit-for-bit.
        if (island_plateau_mode == "converged"
                and "island_plateau" in merged.columns):
            plat = merged["island_plateau"].fillna(False).to_numpy(dtype=bool)
            if np.any(plat):
                soft.append(("island_plateau", plat & isl_stalled))
                isl_stalled = isl_stalled & ~plat
        if np.any(isl_stalled):
            if island_stalled_mode == "warn":
                soft.append(("island_stalled", isl_stalled))
            else:
                hard.append(("island_stalled", isl_stalled))

    # island_diverged (#35): island-mode-only DISCLOSURE signal, always soft.
    # The island's stall (or, under stall_count_mode="changed", a survived
    # episode) fired on a frozen clip/bound-limited proposal -- the
    # clip-saturated-runaway class whose Re/flux ride a displaced chi2 valley.
    # No fail mode by design: under the default counting diverged is a subset
    # of island_stalled (hard-fails above), and the flag must never change
    # fit_is_reliable on its own. Absent on CPU/legacy catalogs.
    if "island_diverged" in merged.columns:
        isl_div = merged["island_diverged"].fillna(False).to_numpy(dtype=bool)
        if np.any(isl_div):
            soft.append(("island_diverged", isl_div))

    # blend_suspect (item 4): GPU-island silent-channel DISCLOSURE, always
    # soft. The fitted position moved from the catalog seed by more than
    # the run's own median PSF core sigma (half-max isophotal area, wing-
    # robust) AND the own-profile rchisq is above the configured threshold -- the measured signature of an uncataloged
    # point source blended inside the bright-mask match radius being
    # absorbed by this source's fit (host displaced toward the interloper,
    # flux stolen, every other quality metric quiet). A suspect flux is
    # still a real measurement, so the tag must never flip
    # fit_is_reliable on its own. Column absent unless the GPU pipeline
    # emitted it (requires rchisq_flag_thresh) -> no-op on CPU/legacy.
    if "flag_blend_suspect" in merged.columns:
        bs = merged["flag_blend_suspect"].fillna(False).to_numpy(dtype=bool)
        if np.any(bs):
            soft.append(("blend_suspect", bs))

    # blend_resolved (item 4 Layer 2): informational soft tag -- this
    # host's island was REFIT with an absorbed uncataloged star subtracted
    # (blend_resolve mode "resolve", chi2 not-worse accepted; the star is
    # a solve-only pass-2 column recorded in blend_resolved.csv). The
    # host's values are the repaired fit; the tag discloses that a
    # contamination was found and corrected. Never flips fit_is_reliable.
    if "blend_resolved" in merged.columns:
        brv = merged["blend_resolved"].fillna(False).to_numpy(dtype=bool)
        if np.any(brv):
            soft.append(("blend_resolved", brv))

    # blend_unresolved (item 4 Layer 2): a candidate star passed EVERY
    # detection gate (>= nsigma S1-residual peak, off-core within the stamp
    # radius) but the repair refit was REJECTED by the rchisq guard --
    # positive evidence of a specific contamination that could not be
    # corrected (the near-degenerate blend class; oracle-measured
    # 0.5-2.5 mag wrong, worse than the island_stalled hard-fail bar).
    # Default HARD-fails fit_is_reliable (user decision 2026-07-14,
    # following the "values measured systematically wrong => hard"
    # policy line); merge.blend_unresolved_reliability selects the mode.
    if blend_unresolved_mode not in ("fail", "warn", "ignore"):
        raise ValueError(
            "blend_unresolved_mode must be 'fail'|'warn'|'ignore', "
            f"got {blend_unresolved_mode!r}")
    if (blend_unresolved_mode != "ignore"
            and "blend_unresolved" in merged.columns):
        buv = merged["blend_unresolved"].fillna(False).to_numpy(dtype=bool)
        if np.any(buv):
            if blend_unresolved_mode == "warn":
                soft.append(("blend_unresolved", buv))
            else:
                hard.append(("blend_unresolved", buv))

    # giant_frozen_geometry (column island_giant): island-mode-only signal. A
    # giant island gets no island fit: position and shape stay at the catalog
    # seed and pass 2 measures the flux at that geometry, yet the row reports
    # opt_converged=True / position_stranded=False. Its geometry was never
    # measured, so by default it hard-fails reliability rather than
    # certifying as a clean fit (plan v2-D3; lifting that is Q-H). Absent on
    # CPU/legacy catalogs.
    if island_giant_mode not in ("fail", "warn", "ignore"):
        raise ValueError(
            f"island_giant_mode must be 'fail'|'warn'|'ignore', got {island_giant_mode!r}"
        )
    if island_giant_mode != "ignore" and "island_giant" in merged.columns:
        isl_giant = merged["island_giant"].fillna(False).to_numpy(dtype=bool)
        if np.any(isl_giant):
            if island_giant_mode == "warn":
                soft.append(("giant_frozen_geometry", isl_giant))
            else:
                hard.append(("giant_frozen_geometry", isl_giant))

    # no_usable_band (R06): GPU band-level contract. ``n_bands_usable`` counts
    # the bands whose value is a measurement with no reason flag
    # (FLAGS_{band}_fit == 0). A fitted source with none has nothing a
    # consumer may use, whatever the fit's health. Absent on CPU/legacy
    # catalogs -> no-op.
    if "n_bands_usable" in merged.columns:
        nbu = pd.to_numeric(merged["n_bands_usable"], errors="coerce") \
            .fillna(0).to_numpy(dtype=float)
        hard.append(("no_usable_band", (nbu <= 0) & ~excluded & ~no_fit))

    # high_rchisq (#17): island-mode-only soft signal. The pipeline sets the
    # boolean ``flag_high_rchisq`` column (per-source own-profile-weighted
    # reduced chi2 above a config threshold) ONLY when the user enables a
    # threshold (compute.gpu.error_model.rchisq_flag_thresh); the DEFAULT emits
    # the rchisq/fracflux/fracin/fracmasked columns with NO flag (legacypipe
    # ships these as fit-quality columns, not a hard cut -- users apply their
    # own thresholds). A high-rchisq flux is still a real measurement (poorer-
    # fitting, not un-measured), so the default mode is "warn" (surfaced in
    # fit_quality_issues, fit_is_reliable untouched); "fail" hard-fails it,
    # "ignore" restores no-signal. Fair-benchmark caveat: a single rchisq cut
    # is a MODEST discriminator (AUC ~0.62 for silent-bad); it is disclosure,
    # not a clean gate. Absent on CPU/legacy catalogs -> no-op.
    if high_rchisq_mode not in ("fail", "warn", "ignore"):
        raise ValueError(
            f"high_rchisq_mode must be 'fail'|'warn'|'ignore', got {high_rchisq_mode!r}"
        )
    if high_rchisq_mode != "ignore" and "flag_high_rchisq" in merged.columns:
        hi_rc = merged["flag_high_rchisq"].fillna(False).to_numpy(dtype=bool)
        if np.any(hi_rc):
            if high_rchisq_mode == "fail":
                hard.append(("high_rchisq", hi_rc))
            else:
                soft.append(("high_rchisq", hi_rc))

    if "opt_converged" in merged.columns:
        not_conv = ~merged["opt_converged"].fillna(True).to_numpy(dtype=bool)
    else:
        cc = [c for c in merged.columns if re.match(r"^opt_converged_.+$", c)]
        not_conv = (
            ~merged[cc].fillna(True).astype(bool).all(axis=1).to_numpy(dtype=bool)
            if cc else np.zeros(n, dtype=bool)
        )
    soft_nc = not_conv & ~hmi & ~excluded & ~no_fit
    if np.any(soft_nc):
        soft.append(("not_converged", soft_nc))

    if "psf_fallback_band_count" in merged.columns and "psf_used_epsf_band_count" in merged.columns:
        fb = pd.to_numeric(merged["psf_fallback_band_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
        used = pd.to_numeric(merged["psf_used_epsf_band_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
        all_fb = (fb > 0) & (used == 0)
        if np.any(all_fb):
            soft.append(("all_psf_fallback", all_fb))

    # shape_frozen (GPU grid mode): informational SOFT tag only. A DEV/SERSIC
    # row fit under a shape-freezing solver (per_source/joint_lsqr) reports
    # its Re/ELL/THETA verbatim from the catalog seed -- the shape was NOT
    # fit. The flux IS a real measurement (conditioned on the assumed shape,
    # like a frozen-n flux), so this does NOT hard-fail fit_is_reliable; it
    # surfaces in fit_quality_issues so the seed-echoed shape is self-evident.
    # Absent on CPU/legacy catalogs and all-False on star-only runs -> no-op.
    if "shape_frozen" in merged.columns:
        shp = merged["shape_frozen"].fillna(False).to_numpy(dtype=bool)
        if np.any(shp):
            soft.append(("shape_frozen", shp))

    all_items = hard + soft
    n_hard = len(hard)

    if all_items:
        tags = [t for t, _ in all_items]
        arr = np.column_stack([m for _, m in all_items])
        merged["fit_is_reliable"] = ~arr[:, :n_hard].any(axis=1)
        merged["fit_quality_issues"] = [
            ",".join(tags[j] for j in range(len(tags)) if arr[i, j])
            for i in range(n)
        ]
    else:
        merged["fit_is_reliable"] = True
        merged["fit_quality_issues"] = ""
