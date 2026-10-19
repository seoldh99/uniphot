from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "inputs": {
        "tile_name": "your_tile_name",
        "input_catalog": "path/to/input_catalog.csv",
        "image_list_file": "path/to/image_list.txt",
        "gaiaxp_synphot_csv": "path/to/gaiaxp_synphot.csv",
    },
    "outputs": {
        "work_dir": "path/to/work_dir",
        "final_catalog": "output_catalog.csv",
        "cropped_images_dir": "cropped_images",
        "overlay_dir": "overlay",
    },
    "image_scaling": {
        "zp_ref": 25.0,
        # No-coverage masking (2026-08-01). Coadders mark "no exposure here"
        # either with NaN (caught by the finiteness test) or with an exact 0.0
        # fill; the latter is invisible to that test, so those pixels enter the
        # fit as ordinary sky and sources standing on them are measured against
        # nothing -- multiple magnitudes too faint with fracmasked 0.0000 and
        # fit_is_reliable True (measured on 7DT coadds: 11-24% of a tile's
        # sources carry >=1 such band; paired median vs DECaPS2 external truth
        # 1.3-2.1 mag too faint on arm A, worst sub-population 3.8 mag). true
        # (default): a pixel is no-coverage iff it is BOTH a border-connected
        # exact zero AND outside the data's own envelope (orthogonally convex
        # hull of the non-zero pixels -- never the bounding box, which a rotated
        # footprint collapses to the whole frame). The intersection IS the rule:
        # no threshold and no tuned constant anywhere (2026-08-01c), and
        # percolation on zero-common products is handled by construction.
        # Interior zeros are left alone, so a product whose zeros are real data
        # is untouched. false => pre-fix behaviour, byte-identical.
        "mask_zero_coverage": True,
    },
    "crop": {
        "enabled": True,
        "margin": 100,
        # Pre-EPSF diagnostic PNGs default ON: the matplotlib savefig
        # is dispatched to the background plot pool (see
        # performance.background_plotting), so they don't block the
        # next pipeline stage.
        "plot_pre_crop": True,
        "plot_post_crop": True,
        "display_downsample": 4,
        "post_crop_display_downsample": 4,
    },
    "overlay": {
        # Full-frame white+catalog overlay PNG; saved in the background
        # so it doesn't block the rest of load_inputs / augment_gaia.
        "enabled": True,
        "downsample_full": 4,
        "zoom_enabled": True,
        "zoom_box": [4000, 5000, 4000, 5000],
    },
    "checks": {
        "require_wcs_alignment": True,
        "wcs_tolerance": {
            "crval": 1e-6,
            "crpix": 1e-6,
            "cd": 1e-9,
            "cdelt": 1e-9,
        },
        "require_same_shape": True,
    },
    "source_saturation_cut": {
        # Per band (R11, 2026-09-25): a source within radius_pix of a
        # saturated pixel of a band has that band excluded (NaN + FLAGS bit
        # 8); it leaves the fit only when this holds in EVERY band.
        "enabled": True,
        "radius_pix": 20.0,
        "saturation_divisor": 1.3,
    },
    "bright_mask": {
        "enabled": True,
        "thresh_sigma": 5.0,
        "minarea": 20,
        "deblend_nthresh": 32,
        "deblend_cont": 0.005,
        "clean": True,
        "clean_param": 1.0,
        "ellipse_scale": 3.0,
        "min_radius_pix": 6.0,
        "dilate_pix": 1,
        # Mask geometry: "ellipse" (legacy: scaled SEP ellipse floored at
        # min_radius_pix + dilate_pix -- three tuned geometric constants) or
        # "physical" (mask out to where the source's isophotal flux, spread
        # over the run's OWN from_stars PSF, exceeds physical_nsigma * the
        # local per-pixel noise -- a propagated wing-vs-noise radius, no
        # dataset-tuned constants; ellipse_scale/min_radius/dilate ignored).
        # "physical" falls back to "ellipse" when no from_stars PSF is
        # available (cpu/legacy modes). DEFAULT "physical" since 2026-07-12
        # (full-field T08285 validated: mask 29.6%->13.8%, +10,794 reliable
        # sources, control unchanged, contamination 0.004 dex, 3 tuned
        # constants removed). Set "ellipse" to reproduce the pre-2026-07-12
        # geometric mask.
        "mask_radius_mode": "physical",
        # Surface-brightness significance for mask_radius_mode="physical"
        # (principled floor, the masking analogue of thresh_sigma; not tuned).
        "physical_nsigma": 1.0,
        # Route unmatched bright detections into the island fit as compact
        # n-frozen DET (soaked, unemitted) instead of masking their wings, so
        # previously-masked neighbours stay fittable ("model the wings").
        # Saturated cores remain masked by the per-band saturation mask.
        # Mutually exclusive with compute.gpu.solver.island.detect_unattributed
        # (shared _append_det path). Off => byte-identical.
        "model_unmatched": False,
        # Exp C "subtract, don't mask": an unmatched bright detection whose
        # above-noise wing extent (the SAME physical radius the mask would
        # use: flux x PSF(r) > physical_nsigma * sigma_local) fits INSIDE
        # the from_stars PSF stamp's modelable radius is NOT masked --
        # instead it is handed to the island stage's subtract-only pass
        # (PSF forced flux, subtracted field-wide, NEVER enters the fit;
        # footprint protection applies). Detections whose wings extend
        # BEYOND the stamp (r_mask claims the profile edge) keep the wing
        # mask -- a 55px stamp cannot subtract light it cannot model --
        # and saturated cores stay masked by the per-band saturation mask
        # regardless. Derived split, no new constants (reuses the physical
        # mask radius + the stamp's own radius). Motivation (oracle star-
        # injection test 2026-07-13): masking an uncataloged star ANNIHILATES
        # nearby galaxy fits (29% of contaminated targets lost; 47% within
        # 4 FWHM at snr>=30) -- the mask collateral, not wing bias, is the
        # dominant incomplete-catalog harm. Requires
        # position_solver=island + scene_subtract (else falls back to
        # masking, with a warning). Mutually exclusive with model_unmatched.
        # DEFAULT ON since 2026-07-14 (user decision, "subtract don't
        # mask" PROMOTED in its v2/"s1" joint-solve form): oracle- and
        # full-field-validated -- wing-masking an uncataloged star
        # annihilates nearby galaxy fits (29% of contaminated targets;
        # full field: +10,524 fitted / +9,446 reliable / +310 measurable
        # injected galaxies when routed instead), the paired position
        # "regression" was shown to be a statistical (Rician) artifact,
        # and phantom columns cancel ~75% of the mask-removal
        # perturbation. Costs (disclosed): ZP mad +1.3%, wall ~2x (the
        # recovered hard regions are genuine work), net stall +28.
        # DEEP EXTRAGALACTIC FIELDS: uncataloged detections are mostly
        # galaxies -> point-source phantoms churn photometry (UDS: 40% of
        # sources move >10 mmag) -- set subtract_unmatched: false there
        # (and/or declare unmatched_population). false => byte-identical
        # to the pre-routing behavior.
        "subtract_unmatched": True,
        # HOW routed detections are handled (only with subtract_unmatched):
        # "sequential" (v1): per-detection PSF forced flux on the coadd,
        #   subtracted one by one at the island stage. ORACLE-MEASURED
        #   NET-NEGATIVE (2026-07-13): under-subtraction residuals inflate
        #   the L2 background -> field-wide additive faint bias (paired
        #   truth stars +16->+58 mmag; ZP mad x1.9 worse in 20/20 bands).
        #   Kept for A/B only.
        # "s1" (v2, THE DEFAULT since 2026-07-14): routed detections
        #   become flux-only STAR columns of the pass-1/pass-2 S1 JOINT
        #   solves (fluxes + background solved self-consistently -- the
        #   root-cause form; G1 guard: never fixed-flux pre-subtraction),
        #   forced-singleton islands (never joint-fit with real sources
        #   -- the measured-negative DET free-fit channel stays closed),
        #   scene-subtracted like any non-member STAR, and DROPPED from
        #   the output catalog (ID prefix sub_; user-catalog purity).
        #   Footprint protection demotes phantoms attributable to a
        #   cataloged galaxy's own light (post-S1 test; demoted = zero
        #   subtraction + excluded from the final flux solve).
        "subtract_unmatched_mode": "s1",
        # USER DECLARATION of what the catalog-unmatched detections in this
        # field ARE. Star/galaxy separation at detection-limit significance
        # is classically unreliable, so this is deliberately NOT an
        # automatic classifier: the user knows their field. The on/off
        # gate is subtract_unmatched alone -- this knob NEVER silently
        # disables the feature. "stars" (Galactic-bulge / low-latitude
        # fields): detections modeled/subtracted as point sources (the
        # oracle-validated case). "extended"/"mixed" (deep extragalactic
        # fields): a dedicated extended treatment is PLANNED but not yet
        # implemented -- currently FALLS BACK to point-source modeling
        # with a loud warning. Measured caveat for that fallback (UDS
        # deep field): heavy photometry-churn tail (40% of sources move
        # >10 mmag, extremes multi-magnitude) -- for deep extragalactic
        # fields the recommendation is to keep subtract_unmatched OFF.
        "unmatched_population": "stars",
        # Save the white_bright_mask_overlay.png diagnostic plot.
        # Dispatched to the background plot pool, so the EPSF stage
        # starts immediately. Default on.
        "save_overlay": True,
    },
    "logging": {
        "ignore_warnings": True,
        "level": "INFO",
        "file": "auto",
        "format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    },
    "plotting": {
        # DPI for all diagnostic PNGs. Default 150 is a good balance of
        # readability and render speed - matplotlib rasterization scales
        # roughly with DPI^2, so 300 takes ~3-4x longer to render and
        # produces ~4x larger files for no science benefit. Bump to 300
        # only when you specifically need very high-res output.
        "dpi": 150,
    },
    "performance": {
        # Worker counts used by the load_inputs stage. Both are honored
        # up to the host's cpu_count, with a per-stage natural cap on
        # top of that:
        #   - frame_prep_workers: capped at the number of input FITS
        #     frames (one task per frame).
        #   - white_stack_workers: capped at the number of row chunks
        #     the white stack is split into; the chunking adapts to the
        #     requested worker count so 32 workers actually get 32
        #     chunks.
        # "auto" still works for backward compat (resolves to cpu_count).
        "frame_prep_workers": 32,
        "white_stack_workers": 32,
        # Background plotting: pre-EPSF stages dispatch their full-frame
        # diagnostic PNGs to a small thread pool so the next stage can
        # start while matplotlib serializes. The plots are write-only
        # (no downstream stage reads them), the pool is drained at
        # pipeline end, so the user-visible output is identical.
        # Disabling this falls back to inline savefig everywhere.
        "background_plotting": True,
        "background_plot_workers": 4,
    },
    "compute": {
        "gpu": {
            # ---- JAX runtime --------------------------------------------
            # "auto" -> all CUDA devices visible to JAX
            # (`jax.devices('cuda')`); or an explicit list of device
            # indices, e.g. [0, 1, 2, 3]. Workers are striped across
            # the devices round-robin (one grid worker per device).
            "devices": "auto",
            # Per-grid-worker XLA memory fraction. With 4 grids = 4
            # procs on 4 GPUs (1 proc per GPU), each can use most of
            # the 24 GB. Default 0.5 (~12 GB on a 24 GB card) is
            # conservative and leaves headroom for transient JIT
            # compile bursts. Note: when
            # XLA_PYTHON_CLIENT_ALLOCATOR=platform (set automatically
            # in runner._worker_init), this fraction is largely
            # ignored because cudaMalloc is called per-allocation.
            "mem_fraction": 0.5,
            # JAX persistent JIT-cache directory. "auto" ->
            # ~/.cache/tract7dt/jax_jit_cache_grid; null disables.
            "jit_cache_dir": "auto",

            # ---- Adaptive worker packing --------------------------------
            # How many grid-worker PROCESSES to run PER GPU. "auto"
            # (default) probes free GPU memory at runtime and packs the
            # most workers that fit given the estimated per-grid peak --
            # safe on ANY GPU size and ANY image/catalog (large grids on a
            # 24 GB card -> 1/GPU; small grids or big GPUs -> pack). An int
            # forces that count. Packed workers get a hard per-worker BFC
            # memory cap (~0.9/N of the card) so they cannot collectively
            # OOM; any grid that still exceeds its cap is retried 1/GPU
            # (results stay bit-identical, never corrupted).
            "workers_per_gpu": "auto",
            # Use NVIDIA MPS while packing (>1 worker/GPU) so co-located
            # workers run CONCURRENT GPU kernels instead of time-slicing.
            # Private per-user daemon, auto-started only while packing and
            # stopped after; graceful no-op if MPS is unavailable or at
            # 1 worker/GPU. Set false to disable.
            "use_mps": True,

            # ---- Image grid layout --------------------------------------
            "grid": {
                # Image is split into n_rows x n_cols quadrants;
                # each quadrant is fit on its own GPU worker.
                "n_rows": 2,
                "n_cols": 2,
                # Per-grid halo overlap (pixels). Sources whose center
                # falls in this halo (i.e., are owned by a NEIGHBOR
                # grid) are included in this grid's FORWARD MODEL so
                # PSF-tail flux at boundaries is captured correctly.
                # Default 30 px = ~half the 55-px PSF stamp + a few.
                "halo_pix": 30,
                # Automatically refine n_rows/n_cols when a large crop
                # would put too many owned sources in one grid. The linear
                # solve scales steeply with sources per grid; keeping this
                # bounded makes full-tile catalogs tractable. Set to null
                # to force the explicit n_rows x n_cols layout.
                "max_sources_per_grid": 1500,
            },

            # ---- PSF model ----------------------------------------------
            "psf": {
                # PSF stamp size (px).
                "stamp_size": 55,
                # ---- mode="from_stars" parameters -------------------
                # PSFEx-style direct-from-stars polynomial PSF (active
                # only when mode="from_stars"). Algorithm follows PSFEx
                # 3.24.2 (Bertin 2011) with a few EPSFBuilder-style
                # refinements (iterative sub-pixel recentering,
                # sigma-clipping inside the per-pixel polynomial fit).
                "from_stars": {
                    # Polynomial spatial order. Quartic (4 = 15
                    # spatial terms) was the T01829 sweep winner:
                    # tighter ZP scatter than cubic (3) by ~5%, no
                    # benefit from quintic (5). Supported: 0..5.
                    "polynomial_order": 4,
                    # SEP detection. T01829 sweep winner is
                    # thresh_sigma=5 (best balance of PSF candidate
                    # count vs detection purity -> tightest
                    # zp_err_std at the bright end). Lower to 3 in
                    # low-density fields; raise to 10 if SEP
                    # detection time is a bottleneck.
                    "thresh_sigma": 5.0,
                    "minarea": 25,
                    "deblend_nthresh": 32,
                    "deblend_cont": 0.005,
                    #   sample_variability (PSFEx SAMPLE_VARIABILITY,
                    #     default 0.2): auto-mode FWHM finder uses
                    #     fwhm_min/fwhm_max =
                    #     mode / sqrt(1+var) and mode * sqrt(1+var).
                    "sample_variability": 0.2,
                    #   min_snr: PSFEx SAMPLE_MINSN default = 20. Defines the
                    #     PSF training-star pool (M2 unit a.3, design 1.5):
                    #     the star weight grows as S/N^2, so a star at the
                    #     boundary holds ~400/sum(S/N^2) of the weight.
                    "min_snr": 20.0,
                    #   max_stars_per_band: hard cap to bound
                    #     polynomial-fit time. PSFEx has no cap; we
                    #     cap at 5000 to keep memory reasonable.
                    "max_stars_per_band": 5000,
                    # Star cutout extraction (matches build_epsf).
                    "cutout_size": 61,
                    "psf_size": 55,
                    "edge_pad": 10,
                    # Sub-pixel centering algorithm.
                    #   "gw_centroid" (default, ~100x faster):
                    #     vectorized iterative Gaussian-weighted
                    #     centroid (a la SExtractor WIN moments).
                    #     PSF model differs from "epsfbuilder" by
                    #     <0.1% peak amplitude on real T01829 data.
                    #     Pure numpy/scipy; does NOT import JAX in
                    #     the per-band PSF workers (which previously
                    #     cost ~80 s of stage wall at 18 workers).
                    #   "epsfbuilder" (legacy, slow):
                    #     photutils' EPSFBuilder iterative recentering.
                    #     30x more compute than needed because the
                    #     averaged ePSF EPSFBuilder builds is then
                    #     DISCARDED -- only the per-star fit centers
                    #     are used. Use only if you specifically need
                    #     to reproduce pre-Apr-2026 PSF outputs.
                    "centering_method": "gw_centroid",
                    # Legacy EPSFBuilder-only knobs (ignored when
                    # centering_method="gw_centroid").
                    "recenter": True,
                    "recenter_boxsize": 9,
                    "recenter_maxiters": 30,
                    "sigma_clip": 5.0,
                    # Local background annulus subtraction per star.
                    "star_local_bkg_subtract": True,
                    "star_local_bkg_annulus_rin_pix": 20.0,
                    "star_local_bkg_annulus_rout_pix": 30.0,
                    "star_local_bkg_sigma": 3.0,
                    "star_local_bkg_minpix": 30,
                    # PSFEx-style outer kappa-sigma rejection iters
                    # on the polynomial fit.
                    "max_outer_iters": 4,
                    "kappa": 3.0,
                    #   wing_from_median: replace the model wings beyond
                    #     ~wing_blend_start_fwhm x FWHM with the pixelwise
                    #     MEDIAN of the training stamps (blended, spatial
                    #     terms damped, integral preserved). The per-pixel
                    #     LSQ stack inherits the MEAN of sub-threshold
                    #     neighbour contamination, which in crowded fields
                    #     bakes a false wing pedestal ~10x the true wing
                    #     into the model and one-sidedly depresses faint-
                    #     source fluxes (measured on T08285). DEFAULT ON;
                    #     set false for exact pre-2026-06 model output.
                    "wing_from_median": True,
                    "wing_blend_start_fwhm": 2.5,
                    # Per-band parallelism (process pool). Per-proc
                    # cost ~600 MB (image.copy() + photutils buffers).
                    # "all_bands" (default) -> N bands at once
                    # (~14 GB extra for 18 bands). Drop to a small
                    # int (e.g. 4) on RAM-constrained machines, or
                    # use "auto" (= n_cpus // 8) for a moderate cap.
                    "max_workers": "all_bands",
                    # Diagnostic outputs per band: summary.json,
                    # stars.csv, stars_centered_stamps.npz, and a PSF
                    # residual PNG montage.
                    "save_diagnostics": True,
                },
            },

            # ---- Background model ---------------------------------------
            "background": {
                # 2-D polynomial background spatial order.
                # Linear (1, x, y) = 3 terms. T01829 sweep: bg=1 ties
                # for best zp_err_std AND is fastest. Higher orders
                # (2, 3) don't improve ZP quality on T01829-class
                # images and add wall cost. Increase only if your
                # background has obvious large-scale structure beyond
                # a plane.
                "poly_order": 1,
            },

            # ---- LM solver ----------------------------------------------
            "solver": {
                # If False (default), run the classic 3-stage fit:
                #   S1: positions frozen, linear flux + bg refit only.
                #   S2: one Newton step on positions (flux + bg fixed).
                #   S3: alternating joint loop until convergence.
                # This mirrors the CPU-mode staged fit and is more
                # robust when input positions are uncertain. Set to
                # True to skip S1 + S2 and run only the joint loop
                # (faster but less robust).
                "single_stage": False,
                # Max outer iterations of the joint LM loop (S3, or
                # the only stage when single_stage=true). Sweep showed
                # 20 is enough for T01829-class images.
                "n_opt_iters": 20,
                # Convergence: stop when max position update across
                # all sources drops below this (pixels). T01829 sweep:
                # 0.1 gives identical ZP to 1e-3 but ~30% faster.
                "dlnp_stop_pix": 0.1,
                # S3 quantile stop: declare position convergence when at
                # most floor(frac * n_sources) sources still step above
                # dlnp_stop_pix. 0 = exact legacy (every source). On
                # T08285, 1-4 'walker' sources/grid (catalog positions
                # off 1-2px) drag 55 of 145 S3 iterations while >=99.8%
                # of sources converge by iteration 2; 0.002 stops those
                # iterations and flags the stragglers per source in the
                # 'position_stranded' catalog column. NEVER emulate this
                # by capping n_opt_iters -- that marks whole grids
                # hit_max_iters and poisons fit_is_reliable.
                "pos_stop_unconverged_frac": 0.002,
                # dlnp-style per-source ftol (extended-source Newton path
                # only): a source whose best chi^2 improves by less than
                # this on two consecutive S3 iterations counts as converged
                # even if its parameter steps still oscillate. 0 disables
                # (exact step-only legacy rule). Star-only grids are
                # unaffected either way (bit-compat).
                "chi2_ftol": 0.1,
                # Max per-iteration position update (clipped to bounds).
                # 0.5 is the conservative default after observing that
                # 1.0 caused faint sources near bright neighbours to
                # overshoot into noise-induced local minima during the
                # damped-Newton step (see SXDS-iC-057332 isolated test).
                # CPU-mode patches via Tractor's bounded LM naturally
                # take adaptive smaller steps; the explicit cap here is
                # what diverged. Smaller values (e.g. 0.25) match CPU
                # better but increase iteration count.
                "pos_step_max_pix": 0.5,
                # ---- Position+shape solver choice -----------------------
                # DEFAULT (2026-07-11): "island" (see below). Flipped from
                # "auto" after the re-audit showed island >= auto on every
                # solver-isolable axis (tighter galaxy scatter, fewer >30%
                # flux outliers; star bulk unaffected) AND is the only
                # solver that emits the reliability/quality columns
                # (rchisq/fracflux/fracin/fracmasked, param errors, the
                # island_* flags). "auto"/"per_source" remain opt-in for
                # star-only-optimized or legacy-bit-compat runs.
                # "auto" : resolved PER GRID at task build (the pre-2026-07-11
                #   default): a grid whose fit set contains any extended
                #   source (EXP/DEV/SERSIC) uses "joint_varpro" (joint
                #   flux-position-shape coupling + free Sersic-n for
                #   gated SERSIC rows); star-only grids keep
                #   "per_source" (fast, and bit-gated against the
                #   pre-extended-source baseline).
                # "per_source" : per-source Newton
                #   step with Gauss-Seidel updates and pre-subtraction
                #   of neighbour models.  Fast and simple but doesn't
                #   include off-diagonal Hessian coupling between
                #   overlapping sources -- can settle to negative-flux
                #   blend solutions.
                # "joint_lsqr" : preconditioned CG on the normal
                #   equations of the JOINT 5*n_sources Jacobian for the
                #   whole grid (positions+shapes only; flux+bg are still
                #   block-Gauss-Seidel'd against this).  Captures the
                #   inter-source POSITION coupling that CPU sees but
                #   does NOT capture the flux-position cross coupling.
                # "joint_varpro" : variable projection / Schur complement.
                #   Closed-form linear flux+bg solve is computed at every
                #   chi^2 evaluation INSIDE the residual function, so
                #   autograd's chain rule produces the joint (F, X, B)
                #   Jacobian-vector products without ever assembling the
                #   full big system.  Mathematically equivalent to a full
                #   joint LSQR over (F, X, B) -- this is what closes the
                #   silent policy gap with CPU's Stage 3 ``thawAll``.
                #   Recommended for crowded fields where blend cases
                #   matter.
                # "island" : island-architecture joint fit (2026-07,
                #   the DEFAULT since 2026-07-11). S1
                #   runs per grid as usual (positions/shapes frozen at
                #   the catalog seeds) to get seed fluxes + the frozen
                #   per-grid background planes; the FIELD's sources are
                #   then decomposed into flux-aware coupling islands
                #   (connected components where a source's light
                #   competes with the sky noise; islands.py) and every
                #   island gets an independent joint VarPro-LM fit of
                #   positions + shapes [+ Sersic-n above the S/N gate],
                #   vmapped per (bbox, nsrc) bucket on ONE GPU
                #   (island_solver.py). A final per-grid S1 rerun at
                #   the converged nonlinear parameters produces the
                #   output fluxes/errors (exact cross-island flux
                #   consistency). Replaces the S2/S3 stages; adds
                #   island_id / island_nsrc / island_giant /
                #   island_stalled catalog columns (+ the #28a
                #   parameter-error columns when
                #   error_model.param_errors_enabled); existing fit
                #   columns keep their names. Knobs below in "island".
                "position_solver": "island",
                # ---- Island solver knobs (position_solver: island) ----
                "island": {
                    # Coupling threshold (units of sigma_sky) for the
                    # flux-aware island radii: source i couples out to
                    # the radius where flux_i * profile_i(r) drops
                    # below kappa * sigma_sky (floored at the PSF FWHM,
                    # capped at the geometric stamp-overlap radius).
                    # Locked at 0.5 by the P1.5 truncation audit
                    # (worst dropped coupling 0.21 sigma field-wide on
                    # the T08285 bulge).
                    "kappa": 0.5,
                    # Max LM steps per island; islands still moving
                    # after this many steps are reported unconverged
                    # (opt_hit_max_iters / position_stranded).
                    "max_steps": 30,
                    # Difficulty re-queue (#25, survey note S5 "batch
                    # cost = slowest member"): every chunk runs at most
                    # this many LM steps in the wide main pass; islands
                    # that neither converged nor stalled by then are
                    # re-queued into compact NARROW chunks (same
                    # compiled kernel family) for the remaining
                    # max_steps budget, carrying their exact LM state.
                    # Most islands converge in 2-7 steps, so they stop
                    # paying for 30-step stragglers. 0/null (or >=
                    # max_steps) disables the re-queue (legacy
                    # single-pass behavior).
                    "max_steps_main": 8,
                    # Islands with more members than this skip the
                    # joint nonlinear fit and take the giant path
                    # (positions/shapes stay at the seeds, pass 2 measures
                    # the fluxes, island_giant=True). ~16-25 is the practical
                    # joint-nonlinear limit (design survey note S5).
                    "joint_nsrc_cap": 32,
                    # PERF #14: star/galaxy render split in a MIXED island
                    # (stars + galaxies together). true (default): STAR
                    # members render via the cheap PSF-eval + Lanczos path
                    # and only the galaxies pay the table-MoG FFT -- kills
                    # the wasted 512-pad FFT every star lane used to render
                    # then discard (2.2-2.6x faster on the dominant mixed
                    # buckets). The placed stamp is byte-identical per
                    # island, but the narrower (width-G) galaxy FFT vmap
                    # shifts XLA's batched reduction order at the ulp level
                    # -- the SAME chunk-width fp-noise class this solver
                    # already carries (< 1 sigma, no systematic bias; see
                    # island_solver.py). false: the pre-#14 single-N-wide
                    # render.
                    # CORRECTED 2026-08-02 (this comment used to promise
                    # "byte-reproducible; pair with TRACT7DT_POS_STEP_CHUNK=0
                    # for a strictly bit-exact island run" -- MEASURED FALSE).
                    # On T08676 (island + phantom pass) four independent
                    # same-configuration pairs ALL differed: plain defaults;
                    # POS_STEP_CHUNK=0 + GPU_WORKERS_PER_GPU=1; and that pair
                    # PLUS star_render_split=false. There is no known knob
                    # combination that makes gpu+island bit-reproducible.
                    # What IS reproducible run to run: white_bright_mask.fits,
                    # psf_polynomial/** and phantom_columns.csv (byte-identical),
                    # and the host-side raw-sky F_corr columns; the GPU-derived
                    # F_corr_residual moves ~3e-10, i.e. the jitter originates
                    # in the GPU fit. Observable envelope, and the right gate
                    # for any A/B: MAG_*_fit |delta| median 0.0000 / p99 ~0.007
                    # / max ~0.25 mmag with ZERO fit_is_reliable /
                    # island_stalled / island_plateau flips (T08676; a crowded
                    # packed tile is wider -- measure its own pairs). Estimate
                    # such an envelope from ALL available pairs: 4 hand-picked
                    # pairs gave max 0.15 mmag where the full 6-pair set
                    # reaches 0.247. See docs/configuration.md, which already
                    # says "island GPU mode is not bit-reproducible by default".
                    "star_render_split": True,
                    # #35: how rejected LM steps count toward the 4-reject
                    # stall detector (island_stalled).
                    # "proposals" (default, legacy): every rejected step
                    # counts. Measured caveat (rootfix_a2 trajectory
                    # forensics, T08285 bulge): after a long accept run
                    # lambda has bottomed at ~1e-9 and the raw Newton step
                    # is far beyond the per-step clips, so the x10 lambda
                    # escalation cannot change the clipped proposal -- the
                    # 4 rejects then test ONE frozen step (the
                    # island_diverged column discloses exactly this class).
                    # "changed" (opt-in semantic fix): a reject counts only
                    # if the clip/bound-limited proposal DIFFERS from the
                    # previously rejected one; identical-proposal rejects
                    # keep escalating lambda without counting, so the
                    # solver eventually tests genuinely smaller steps.
                    # CAUTION (measured): in crowded fields the extended-
                    # source chi2 minimum can be genuinely DISPLACED by
                    # uncataloged ambient light -- "changed" helps
                    # correct-valley islands but rides displaced-valley
                    # islands FURTHER into the wrong minimum; use only
                    # together with island_diverged + FLUX_WIN (see
                    # docs/configuration.md + the outputs.md information-
                    # boundary note).
                    "stall_count_mode": "proposals",
                    # Stall-yield (2026-07-27): tolerance of the POST-HOC
                    # probability-convergence certificate on stalled
                    # islands (Tractor-style dlnp semantics; a stalled fit
                    # where NO damped step on the full lambda ladder
                    # improves chi2 by more than 2*this quantum stopped
                    # because it is CONVERGED ON A PLATEAU, not stuck).
                    # 0.5 <=> dchi2 = 1 = the 1-sigma single-parameter
                    # statistical quantum -- a convention, not a fitted
                    # constant. Applied host-side to the always-emitted
                    # island_stall_dchi2 column to derive the SOFT
                    # island_plateau disclosure; changing it never
                    # recompiles kernels and fit_is_reliable is untouched.
                    # (Under convergence: "dlnp" the same constant ALSO
                    # becomes the in-loop statistical quantum 2*tol and
                    # is baked into the compiled step program.)
                    "plateau_dlnp_tol": 0.5,
                    # dlnp campaign (2026-07-30): in-loop convergence
                    # criteria of the island LM step program (static,
                    # kernel-cache-keyed). "legacy" (default): the
                    # byte-identical pre-dlnp criteria -- relative ftol
                    # 1e-4*chi2 on 2 consecutive accepts, fixed
                    # 0.1px/0.01 step tolerances, 4-consecutive-reject
                    # stall, float32 chi2 carry. "dlnp" (opt-in,
                    # science-affecting -- own truth campaign): (1) the
                    # chi2 accept test runs on a FLOAT64 carry (residual
                    # math stays f32) -- E0-measured: the f32 accept
                    # test near an optimum is a coin flip on big islands
                    # (crossband wrong-rejects = 41% of rejects = the
                    # phantom-stall mechanism); (2) the ftol prong
                    # becomes ABSOLUTE -- accepted dchi2 <
                    # 2*plateau_dlnp_tol (the island-size-independent
                    # 1-sigma quantum; the relative prong stopped
                    # chi2>1e5 islands still harvesting ~20 quanta per
                    # step); (3) the step-tolerance prong fires only
                    # when the accepted step is also sub-quantum
                    # (bright-star exits were the largest leaked-descent
                    # channel); (4) rejects count toward the stall only
                    # when the proposal actually CHANGED (identical
                    # clip-pinned proposals keep escalating lambda), so
                    # a stall certifies K distinct dampings found no
                    # improvement; (5) reject-exits with no further
                    # information reachable terminate immediately (a
                    # reject tested AT the lambda cap, or a NO-MOVE
                    # proposal whose damped update rounds to zero
                    # against the f32 params -- every future proposal
                    # is provably bit-identical). EXIT-STATE MAPPING
                    # (E1a): every reject-exit gets the same post-hoc
                    # island_stall_dchi2 ladder certificate as legacy
                    # mode; certificate-PASSING exits are
                    # CONVERGED-ON-PLATEAU (island_plateau=True, NOT
                    # island_stalled -- under exact f64 accepts a
                    # finished island's natural terminal state is a
                    # reject run; measured: 95% of dlnp reject-exits
                    # certify at dchi2 ~ 0 with legacy-identical
                    # endpoints); certificate-FAILING exits stay
                    # island_stalled (honest stall, hard-fail default
                    # unchanged). Requires step_mode: "clip" and
                    # stall_count_mode: "proposals" (dlnp subsumes
                    # "changed"); config load errors on either combo.
                    "convergence": "legacy",
                    # LM inner-step geometry (static, baked into the kernel
                    # cache key). "clip" (default): the byte-identical legacy
                    # damped-Newton step with a fixed per-param clip +
                    # lambda x10/x0.1 escalation. "trust_region": a
                    # parameter-free scaled-dogleg trust region (radius from
                    # the chi2 gain ratio, Nocedal-Wright Alg 4.1 constants --
                    # algorithm-standard, not dataset-tuned) that removes the
                    # fixed-clip artifact + converts the ping-pong-reject
                    # island_stalled population into clean convergences and is
                    # deterministic. ROBUSTNESS/DETERMINISM/STALL-CONVERSION
                    # ONLY -- NOT an accuracy fix: on a displaced-min valley
                    # (crowded uncataloged ambient light, #35) it reaches the
                    # wrong minimum FASTER, so bound-pinned rows are disclosed
                    # via island_railed (never gates fit_is_reliable) and the
                    # accuracy limit is unchanged. STAR-only classes stay
                    # "clip" (bit-identical). Off => byte-identical.
                    "step_mode": "clip",
                    # SCENE SUBTRACTION (#35 root accuracy fix, opt-in).
                    # Crowded free-n railing is fit-set INCOMPLETENESS: bright
                    # CATALOGED neighbours placed in ADJACENT islands leak
                    # unmodeled wing light into a target's island cutout, which
                    # displaces its chi2 minimum (Re/n rail). Fix = render every
                    # NON-member cataloged source at its S1 (seed) flux and
                    # subtract it from each island cutout so the joint fit sees
                    # only its own members + noise. Implemented as the exact
                    # equivalent "subtract-all-at-S1 field-wide, add-back own
                    # members in-kernel" (each source is a member of exactly one
                    # island; members keep their TRUE data light, non-members are
                    # removed). No tuned constants (S1 flux + catalog only;
                    # sensitivity-validated forgiving to +/-30% flux error), and
                    # NON-circular (never uses the untrusted free-n fit). The
                    # field-wide subtract is a single field pass (grid-free ->
                    # P7-de-gridding aligned). DEFAULT true since 2026-07-13
                    # (STAR-only, validated): full-field T08285 gate = net WIN
                    # (free-n n-rail 41->9, island_stalled -37%, fit_is_reliable
                    # +970, injected flux TOWARD truth 720/1140), sparse UDS =
                    # do-no-harm (91% MAG bit-identical, stall/reliable flat).
                    # false => the whole path is skipped, output byte-identical
                    # to the pre-scene-subtract kernel (same off-discipline as
                    # step_mode / has_priors / detect_unattributed).
                    "scene_subtract": True,
                    # Blank-response (empty-aperture) median flux-bias
                    # correction (DEFAULT ON since 2026-07-14): the
                    # classical deep-photometry bias correction adapted
                    # to the joint solve.
                    # Unmodeled ambient point-light is positively skewed:
                    # the L2 background sits at its local MEAN while a
                    # TYPICAL source aperture holds less ambient than
                    # that mean, so every fitted flux carries an additive
                    # bias (measured: ~+2 ADU faint-star deficit). This
                    # measures EXACTLY that bias from the run itself --
                    # the matched-filter response a phantom source would
                    # see at blank positions of the FINAL pass-2 residual
                    # (response map over the whole band, median over
                    # good pixels) -- and subtracts it from every flux.
                    # Self-consistent with whatever weighting ran (plain
                    # or confusion_weight): it removes the ACTUAL
                    # estimator's residual bias, deficit or over-
                    # correction alike. Derived, no tuned constants.
                    # STAR rows only (the PSF-template response is exactly
                    # a point source's bias; the star-template correction
                    # moves extended sources' different net systematics
                    # the wrong way -- measured). Pass-2 reported fluxes
                    # only. false => byte-identical (pre-2026-07-14).
                    # PROMOTION EVIDENCE: crowded T08285 full-field gate
                    # (correction = +2.24 native ADU at m600 vs the
                    # independently-measured +2.06+-1.39 deficit; ZP mad
                    # -9.1% better 20/20 bands; galaxies fp-dust; flags
                    # noise-only) AND a UDS deep-field star-injection
                    # oracle (the deep deficit is REAL: snr10 stars
                    # +114..+189 mmag too faint; corrected to +4/+13 mmag
                    # at m600/m700 with snr1000 untouched; net better in
                    # 15/20 snr x band cells). CAVEAT: the correction is
                    # a per-grid MEDIAN -- local ambient variation leaves
                    # +-0.5-0.9 ADU per-region residuals (worst measured
                    # cell: -25 mmag at snr30/m600 whose local bias was
                    # ~0), and the faintest blue-band tier closes only
                    # partially (under-correction, safe direction).
                    "blank_response_correction": True,
                    # Item-4 Layer-2 "blend resolve" (matched-blend silent-
                    # channel countermeasure). An uncataloged point source
                    # within ~1 PSF FWHM of a cataloged source escapes the
                    # bright mask + phantom routing (its blend detection
                    # MATCHES the host's own catalog entry) and the joint
                    # fit absorbs it -- host displaced toward the star,
                    # flux stolen, all other flags quiet. The star IS
                    # visible in the S1-STATE residual (positions frozen at
                    # seeds; oracle: 46-150 sigma matched-filter). This
                    # block finds those stars: for BLEND-SUSPECT sources
                    # (per-source rchisq above the flag convention AND
                    # fit-vs-seed displacement above the run's median PSF
                    # core sigma -- the flag_blend_suspect operating point,
                    # oracle 3/3 @ 0.20% base rate), scan the S1-state
                    # residual around the host (dmb minus extended-source
                    # seed-shape S1 models; stars already scene-subtracted)
                    # with the run's own noise map. Candidate gates (all
                    # derived): peak >= nsigma x local noise; separation
                    # from host > PSF core sigma (on-core star columns are
                    # flux-degenerate with the host -- measured); within
                    # the PSF stamp radius. NOTE (measured, offline
                    # validation): BIC / delta-chi2 model comparison FAILS
                    # as the junk gate (crowded fields always have ~3sigma
                    # unmodeled light nearby) -- the DETECTION significance
                    # is the gate. mode "detect" = disclosure only: write
                    # {work_dir}/blend_candidates.csv, change NOTHING in
                    # the fit (off/detect are science-identical; enabled
                    # false skips the whole path -> byte-identical).
                    # Requires scene_subtract. Future modes: "resolve"
                    # (refit suspect islands with the star scene-
                    # subtracted + pass-2 star column).
                    "blend_resolve": {
                        "enabled": False,      # OFF by default
                        "mode": "detect",      # "detect" (dry-run CSV only)
                        # candidate peak threshold in units of the local
                        # white-coadd noise (the conventional single-source
                        # significance floor; 8 misses the weakest measured
                        # silent case -- 17 sigma raw peak, 5 catches 3/3).
                        "nsigma": 5.0,
                        # SEP minarea for the residual-cutout detection
                        # (small: the target is a PSF core).
                        "minarea": 5,
                    },
                },
                # joint_lsqr: max CG iterations per LM step.  Higher
                # values give cleaner per-step descent at the cost of
                # walltime.  20 is plenty for typical grids with
                # block-Jacobi preconditioning.
                "joint_cg_max_iters": 20,
                # joint_lsqr: CG residual tolerance.  Tightening below
                # FP32 epsilon doesn't help.
                "joint_cg_tol": 1.0e-6,
                # joint_lsqr: enable diag(J^T J) block-Jacobi
                # preconditioner.  Adds one diagonal-scan pass per LM
                # step but typically cuts CG iterations 2-4x on dense
                # grids.  Default True.
                "joint_use_preconditioner": True,
            },

            # ---- Position priors ----------------------------------------
            "positions": {
                # DEPRECATED (2026-07-08) -- NO EFFECT. Kept only so
                # existing configs still load. An ABSENT POS_ERR column
                # now means NO position prior (data-driven fit), matching
                # the CPU path: a catalog position is a SEED (starting
                # point), not a belief. Supply a per-source POS_ERR sigma
                # to OPT IN to a Gaussian position prior. Previously this
                # applied a weak 1px auto-pull toward the seed.
                "default_pos_err_pix": 1.0,
                # Hard position bound +/- this many pixels from the
                # input catalog seed.
                "pos_max_shift_pix": 5.0,
            },

            # ---- Extended-source (galaxy) model ------------------------
            "galaxy": {
                # Renderer for extended sources (EXP/DEV/SERSIC):
                #   "auto" (default): pure point-source catalogs keep the
                #     historical direct-convolution path (bit-identical);
                #     any catalog with extended sources uses "fft".
                #   "fft": analytic-total-flux renderer -- closed-form
                #     Gaussian-mixture Fourier transform x rfft2 of the
                #     pixelized PSF stamp (exact pixel integration; fixes
                #     the large-Re flux-truncation and cuspy-core biases
                #     of the direct path). Required for DEV/SERSIC.
                #   "direct": the pre-2026-07 path (STAR + EXP only),
                #     kept for regression / A-B comparisons.
                "render_backend": "auto",
                # Stage-2 free-n fitting for TYPE=SERSIC sources (DEV/EXP
                # keep their profile-defined n). Fit n jointly with position
                # + shape via the varpro (joint-LM) step, using a smooth
                # logit bijection onto (0.29, 6.3). AstroPhot cross-fits on
                # real 7DT data showed free-n is unconstrained for faint
                # sources (half the sample pinned at the bound), so n is
                # AUTO-FROZEN at the catalog seed for sources below
                # n_fit_min_snr (per-band-median S/N estimated from the S1
                # solve); frozen sources get sersic_n_frozen=True.
                "fit_sersic_n": True,
                "n_fit_min_snr": 30.0,
                # Wing peeling for large galaxies: keep the solver at the
                # native stamp but subtract per-band WING images (flux x
                # [tier render - native render]) of the few "big" sources
                # (nre(n)*Re > half the native stamp) from the data before
                # every linear solve and position/shape step -- recovers
                # the -10..-14% Re>=16px flux bias and most of the cuspy
                # free-n bias without the OOM-prone whole-grid stamp-tier
                # promotion (stamp_tiers stays off). Refreshed from the
                # current fluxes/shapes/n each S3 outer iteration.
                # DEFAULT ON since 2026-07-04 (validated on inject_v2 +
                # UDS real-field regression: non-big sources unchanged
                # within run-to-run noise, no walltime cost).
                "wing_peeling": True,
                # Render-box tier for the wing image (px, odd).
                "wing_peel_tier": 221,
                # Per-grid cap on peeled sources (largest by nre*Re win).
                "wing_peel_max_sources": 64,
                # EXPERIMENTAL model-stamp size ladder for large galaxies
                # (e.g. [55, 111, 221]). A grid containing sources whose
                # render box needs ~4-8 effective radii is promoted to the
                # smallest adequate tier (linear flux solves only; the
                # position/shape steps keep the native stamp). Default []
                # = disabled: measured on UDS injections 2026-07-04, big
                # tiers OOM'd one grid (neighbour gathers ~n_src*k_max*
                # tier^2) and mixed-window solving gave irregular gains --
                # needs a memory-aware bucketing pass before promotion to
                # default. Large-Re flux bias with the native 55px stamp:
                # ~-10..-14% at Re=16-40 px (documented).
                "stamp_tiers": [],
            },


            # ---- Post-solve error model (task #27, 2026-07) -------------
            "error_model": {
                # Measure per-source CONFUSION noise from the run's own
                # residual map (matched-filter excess variance in a local
                # annulus; fit/grid_gpu/error_model.py -- no tuned
                # constants) and write FLUXERR_CONF_{band}_fit +
                # FLUXERR_TOT_{band}_fit next to the existing columns.
                # FLUXERR_* itself is unchanged (marginal statistical
                # error of the linear solve); in sparse fields the term
                # is dormant (FLUXERR_TOT ~= FLUXERR). Validated on the
                # T08285 bulge + UDS light injections. false: skip the
                # measurement and omit both column families.
                "confusion_enabled": True,
                # Correlated-coadd-noise variance inflation (LEG C, task
                # #20) -- applied ONLY to the GEOMETRIC (pos/shape/n) errors
                # (re-scope 2026-07-08; the flux side is #27-only). The coadds
                # are SWarp-resampled MEDIAN combines, so neighbouring pixels
                # are correlated and the per-pixel SKYSIG under-estimates the
                # noise integrated over a PSF stamp; the white sky-only Fisher
                # covariances are therefore too small by a per-band factor
                # F_corr. F_corr is MEASURED from each coadd's OWN source-free
                # sky / model-subtracted residual (PSF-autocorrelation-weighted
                # 2-D autocorrelation, accepted only when its SHAPE is
                # resampling- not confusion-consistent; fit/grid_gpu/
                # error_model.py -- zero tuned constants) and applied as
                # SEPARATE sqrt(F_corr_eff)-scaled columns
                # {x,y}_pix_white_err_corr / Re_err_corr / ELL_err_corr /
                # THETA_err_corr / sersic_n_err_corr (island shapes). The FIT
                # weights, fitted VALUES, raw *_err, and ALL flux columns are
                # UNCHANGED. On a field with no confusion-free tile (dense
                # bulge) the shape gate falls the confused bands back to
                # F_corr=1.0 (honest; see correlated_noise_factors.csv).
                # false: skip the measurement and omit every *_err_corr column.
                # CORRECTED 2026-08-02 -- this used to claim "all existing
                # columns byte-identical either way", which is FALSE. F_corr
                # also feeds (i) the *_err_tot geometric totals and (ii) the
                # per-source geometric inflation G = max(1,F_corr_eff) *
                # max(1,chi2_nu_nl) inside FLUXERR_MARG_TOT / MAGERR_MARG_TOT
                # (see the f_eff blocks in fit/grid_gpu/pipeline.py). Turning
                # this off therefore CHANGES the recommended error column
                # (the one a downstream SED fit uses). Flux/MAG and the raw *_err are
                # untouched either way. Cost note: the raw-sky estimator has
                # never once been SELECTED in 148 recorded runs (2,311 band
                # entries) -- residual wins or both fall back to 1.0 -- and the
                # pass costs ~4.7 s/band; it is now band-parallel
                # (_fcorr_raw_sky_pass), see docs/performance.md.
                "correlated_noise_enabled": True,
                # Per-island reduced-chi2 error rescaling of the GEOMETRIC
                # errors (LEG D, task #20; re-scope 2026-07-08 -- no flux
                # chi2_nu). After the fit, each island gets a per-island joint
                # reduced chi2 (chi2_nu_nl) from its OWN residual computed with
                # the MARGINAL (sky-only) per-pixel variance
                # (fit/grid_gpu/island_plan.py). chi2_nu reads ~1 under pure
                # correlated noise (so it does NOT double-count LEG C) and
                # inflates on confusion (unmodeled neighbours) + model
                # mismatch. The nonlinear error is rescaled by
                # sqrt(max(1, chi2_nu_nl)) and combined MULTIPLICATIVELY with
                # LEG C's sqrt(F_corr_eff) into the SEPARATE columns
                # {x,y}_pix_white_err_tot / Re_err_tot / ELL_err_tot /
                # THETA_err_tot / sersic_n_err_tot; chi2_nu_nl is written as a
                # diagnostic. Raw *_err and ALL flux columns (incl. the #27
                # FLUXERR_CONF/TOT) are UNCHANGED. NOTE (validated on the
                # T08285 bulge injections): the flexible fit ABSORBS crowded
                # geometric confusion, so chi2_nu_nl reads ~1 there -- *_err_tot
                # then does NOT recover the crowded geometric under-coverage; a
                # geometric-annulus confusion term (analogue of #27's flux
                # sigma_conf) is future work. false: skip the measurement and
                # omit every *_err_tot / chi2_nu_nl column (all existing
                # columns byte-identical either way).
                "chi2_nu_enabled": True,
                # Island nonlinear-parameter errors (task #28a,
                # position_solver: island only): one read-only
                # evaluation of the flux-marginalized Gauss-Newton
                # Hessian at each island's converged state; the
                # diagonal of its inverse gives MARGINAL 1-sigma
                # STATISTICAL errors written as x_pix_white_err /
                # y_pix_white_err / Re_err / ELL_err / THETA_err, and
                # fills sersic_n_err for free-n rows (exact algebra,
                # no tuned constants; background frozen per the island
                # design). NaN where undefined (stars' shapes, frozen
                # n, giant islands). false: skip the evaluation and
                # omit the five new columns (sersic_n_err stays NaN);
                # every existing column is identical either way.
                "param_errors_enabled": True,
                # #17 per-source model-quality metrics (island mode):
                # rchisq (own-profile-weighted per-SOURCE reduced chi2 --
                # unlike the per-island chi2_nu it sees model misfit at the
                # individual source), fracflux (neighbour-flux contamination),
                # fracin (analytic-total flux contained in the fitted stamp),
                # fracmasked (model flux on masked pixels) -- the legacypipe
                # DR9 fit-quality/blend convention. One extra read-only band
                # scan per island chunk at convergence; NaN for giants/off-
                # island. false: skip + omit the four columns (every existing
                # column identical either way). Fair-benchmark note: on a b=-5
                # bulge injection field a survey-standard reliability set
                # {fit-S/N, fracflux, fracmasked, per-source rchisq} reaches
                # leave-island CV AUC ~0.83 for silent-bad detection; the
                # per-SOURCE rchisq is the metric this pipeline was missing
                # (it previously emitted only the blind per-island chi2_nu).
                "source_metrics_enabled": True,
                # D2 (root-fix D): SHAPE-marginalized flux error columns
                # (island mode only). The reported FLUXERR is the VarPro
                # amplitude 1-sigma -- marginal over the flux block but
                # CONDITIONAL on the freely-fitted extended-source shape/
                # position. For low-S/N extended sources the flux-shape
                # covariance is the dominant missing error term. When true
                # (default) the island stage propagates that covariance into
                # the flux via the exact Schur increment (VarPro amplitude
                # sensitivity through the SAME flux-marginalized nonlinear
                # covariance #28a inverts; zero tuned constants) and emits
                # ADDITIVE columns FLUXERR_MARG_{band}_fit /
                # FLUXERR_MARG_TOT_{band}_fit (+ MAGERR_MARG* from compute_zp).
                # STAR rows have no free shape block, so their increment is
                # exactly 0 and FLUXERR_MARG == FLUXERR bit-for-bit. Existing
                # FLUXERR / FLUXERR_TOT are UNCHANGED. false: skip the extra
                # per-island evaluation and omit the columns (every existing
                # column identical either way). Giants / off-island -> NaN.
                "marginal_flux_err_enabled": True,
                # #18 (2026-07-22; default on, see below): scale FLUXERR_MARG_TOT by
                # sqrt(max(1, per-source rchisq)) -- one value for every band;
                # standard chi2 error
                # rescaling, no tuned constant. Fixes the measured stamp
                # artifact where rows with rchisq 3-98 kept un-inflated TOT
                # bars (full-field crossband gate: stamped silent-wrong
                # |pull_TOT|>3 count 29 -> 7 at median collateral x1.00-1.10).
                # Needs source_metrics_enabled (no rchisq column -> no-op).
                # DEFAULT TRUE since 2026-07-22 (user promotion; gates: FF
                # crossband 29->7 silent, star/color oracle pulls toward 1,
                # only cost C4 0.98->0.89 mild over-cover). false => the
                # pre-#18 TOT, byte-identical.
                "rchisq_scale_marg_tot": True,
                # Source-Poisson error term (2026-07-22, DEFAULT ON): adds
                # FLUX/g_eff to the TOT/MARG_TOT variance (REPORTING only,
                # never the fit invvar -- the sigma_total bias trap).
                # g_eff per band: "gain_eff" override map (e-/native-ADU,
                # any instrument) else header auto-derivation -- see the
                # UNIVERSAL HEADER CONTRACT in pipeline._source_poisson_
                # gains (GAIN/EGAIN>5 = delivered-image effective gain;
                # 7DT legacy/UDS shims; MEDIAN combine pays the exact
                # information tax; unknown => warn + skip, never guess).
                # DEFAULT TRUE 2026-07-22 (user principle: an exact
                # forward-propagated physical term is never image-policy;
                # + the half-normal anchor correction: median |pull| target
                # is 0.674, so the measured 0.98->0.80 / 2.65->2.52 moves
                # are IMPROVEMENTS, not over-coverage).
                "source_poisson": {
                    "enabled": True,
                    "gain_eff": None,
                },
                # #17 optional reliability flag: flag_high_rchisq =
                # (rchisq > this) for fitted island rows. null (default) emits
                # the rchisq/fracflux/... COLUMNS with NO flag (legacypipe-style
                # -- users apply their own cut; avoids baking a survey-tuned
                # constant into fit_is_reliable). Set a value (e.g. 3.0) + the
                # merge.high_rchisq_reliability mode to turn it into a
                # warn/fail contribution. Calibrate per field/depth.
                "rchisq_flag_thresh": None,
                # Column-only soft quality flag (island solver only, where
                # opt_niters is a per-source LM step count): flag_slow_converge =
                # a source's island did NOT converge within the solver's own fast
                # MAIN-PASS budget and had to be re-queued into the hard pass --
                #   opt_niters > compute.gpu.solver.island.max_steps_main.
                # This is the solver's OWN "fast converger vs straggler" boundary
                # (the #25 difficulty re-queue line): CONFIG-derived from the
                # solver's existing convergence budget, robust to the field's
                # convergence distribution (a spiky "most converge in 1 step"
                # bulge field breaks a median+MAD outlier test but not this), and
                # NOT calibrated to injection truth -- it replaces the old truth-
                # tuned `slow_converge_niters: 12`. When the re-queue is disabled
                # (max_steps_main null/0/>= max_steps) there is no fast/slow split,
                # so it falls back to opt_niters >= island.max_steps (the hard
                # iteration cap = genuine non-convergence). Does NOT change
                # fit_is_reliable. true (default): emit the column; false/null:
                # omit it.
                "slow_converge_flag": True,
            },
        },
    },
    "patch_run": {
        "gal_model": "exp",
        "re_fallback_pix": 3.0,
    },
    "merge": {
        # How the island-mode `island_stalled` column enters the shared
        # fit_is_reliable contract (compute_reliability_flag). A stalled island
        # reports opt_converged=True but is not at a minimum and is measurably
        # flux-inflated on crowded fields.
        #   "fail"   – hard-fail fit_is_reliable (default, honest)
        #   "warn"   – tag in fit_quality_issues only, leave fit_is_reliable
        #   "ignore" – no signal (pre-2026-07 behaviour)
        # No effect on CPU/legacy runs (they carry no island_stalled column).
        "island_stalled_reliability": "fail",
        # Stall-yield (2026-07-28, user decision A): how PLATEAU-CERTIFIED
        # stalls (island_plateau = the Tractor-style probability-convergence
        # certificate) enter fit_is_reliable. "stalled" (default) = certified
        # rows still follow island_stalled_reliability (pre-2026-07-28
        # behaviour bit-for-bit). "converged" (opt-in; RECOMMENDED whenever
        # island.confusion_whitening crossband is on): certified stalls are
        # treated as converged -- removed from the stall hard-fail and tagged
        # softly as "island_plateau". Full-field crossband gate: 84% of
        # stalls certify, recovering ~4,131 reliable rows (chain reliable
        # 83,409 -> ~87,540, ABOVE the no-crossband arms); certified pool
        # truth-clean at snr>=100 (79 vs converged 113 mmag) and on the star
        # oracle (4-60 mmag). Disclosed caveat: faint (snr<=30) certified
        # GALAXIES include displaced-valley plateaus (gate n=27, 851 vs 426
        # mmag). Legacy (non-crossband) certified pools are small (10%) and
        # not truth-gated -- hence opt-in, not default.
        "island_plateau_reliability": "stalled",
        # How the island-mode `island_giant` column enters the shared
        # fit_is_reliable contract (reason tag giant_frozen_geometry). A giant
        # island gets no island fit: position AND shape stay at the catalog
        # seed and pass 2 measures the flux there, yet the row reports
        # opt_converged=True; its geometry was never measured (raw seed).
        #   "fail"   – hard-fail fit_is_reliable (default, honest)
        #   "warn"   – tag in fit_quality_issues only, leave fit_is_reliable
        #   "ignore" – no signal
        # No effect on CPU/legacy runs (they carry no island_giant column).
        "island_giant_reliability": "fail",
        # How the #17 `flag_high_rchisq` column (per-source reduced chi2 above
        # compute.gpu.error_model.rchisq_flag_thresh) enters fit_is_reliable.
        #   "warn"   – tag in fit_quality_issues only (default, honest: a high-
        #              rchisq flux is still a real measurement, just poorer-
        #              fitting; a single rchisq cut is a MODEST discriminator)
        #   "fail"   – hard-fail fit_is_reliable
        #   "ignore" – no signal
        # No-op unless rchisq_flag_thresh is set (else the column is absent).
        "high_rchisq_reliability": "warn",
        # How a blend_resolve mode="resolve" UNRESOLVED host (a candidate
        # star passed every detection gate -- >= nsigma S1-residual peak,
        # off-core, off-modeled-sources -- but the repair refit was
        # REJECTED by the rchisq guard) enters fit_is_reliable. Unlike
        # blend_suspect (suspicion, soft) this is POSITIVE evidence of a
        # specific contamination that could NOT be corrected: the oracle
        # measures this class 0.5-2.5 mag wrong (worse than the
        # island_stalled hard-fail bar). Follows the workspace policy
        # line: "still a real measurement" => soft; "values measured
        # systematically wrong" => hard.
        #   "fail"   – hard-fail fit_is_reliable (default, honest)
        #   "warn"   – tag in fit_quality_issues only
        #   "ignore" – no signal
        # No-op unless blend_resolve ran in mode "resolve" (column absent).
        "blend_unresolved_reliability": "fail",
    },
    "zp": {
        "enabled": True,
        "inject_gaia_sources": True,
        # ZP source selection: every GaiaXP source inside the (cropped)
        # image with (4, 28) Gaia mag -- from very bright unsaturated stars
        # down to the survey limit. The calibrator set belongs to the image,
        # not to the input catalog (R28: it used to be a box around the
        # catalog, so a few-target run put its outermost targets on the
        # calibrators' hull edge and the ZP depended on which targets were
        # run together).
        "gaia_mag_min": 4.0,
        "gaia_mag_max": 28.0,
        "match_radius_arcsec": 3.0,
        "clip_sigma": 3.0,
        "clip_max_iters": 10,
        # POS_ERR [pix] policy for the pipeline-INJECTED Gaia ZP-calibrator
        # rows (user catalog rows are never auto-filled).
        #   gaia_pos_err_mode:
        #     "propagate" (DEFAULT): per-source width epoch-propagated from
        #        the Gaia DR3 astrometric errors — sigma_axis(t_obs) =
        #        sqrt(sigma_pos_2016^2 + (t_obs-2016)^2 * sigma_pm^2), max
        #        axis, mas -> pix via the image WCS plate scale (see
        #        gaia_astrometry.propagate_gaia_pos_err; t_obs from the
        #        image MJD-OBS/DATE-OBS headers). Requires ra_error/
        #        dec_error/pmra_error/pmdec_error in the GaiaXP synphot CSV;
        #        when absent (standard synphot CSVs) it FALLS BACK to the
        #        fixed width below (logged once). Sources without a PM
        #        solution (2-parameter, astrometric_params_solved==3) get
        #        NaN = no prior — a constant there would fake knowledge.
        #     "fixed": legacy constant width on every injected row.
        "gaia_pos_err_mode": "propagate",
        # Fixed width for "fixed" mode / fallback width for "propagate"
        # mode. null = no position prior for injected Gaia sources.
        "gaia_pos_err_pix": 0.1,
        # Optional systematic floor [mas] added in quadrature to PROPAGATED
        # widths only (explicit user belief, e.g. residual astrometric-tie
        # error; it never substitutes for a missing PM solution).
        "gaia_pos_err_floor_mas": 0.0,
        # Image<->Gaia FRAME-TIE field-level DIAGNOSTIC (root-fix C2, REVERTED
        # 2026-07-11 -- was `gaia_pos_err_frame_tie`). Freely-fit bright stars
        # sit a coherent ~50-70 mas from their Gaia seeds on crowded fields. That
        # offset is the systematic image<->Gaia FRAME-TIE (a WCS / absolute-
        # astrometry property of the whole field), NOT a per-source position
        # error: position is data-defined and absolute, so it must not be
        # back-measured against the Gaia "answer" and fed into a per-source
        # error bar -- and the offset is undefined for the reference-less science
        # sources such a floor would then modify. The honest per-source position
        # error is the epoch-propagated Gaia width alone (gaia_pos_err_mode:
        # "propagate"); the frame-tie belongs to WCS calibration / downstream
        # absolute-astrometry correction. This knob therefore NEVER modifies
        # POS_ERR, any prior width, or any fit input.
        #   "off" (DEFAULT): measure nothing; POS_ERR is pure-propagated.
        #   "measure": compute the coherent field offset from the ZP calibrators'
        #      fit-vs-Gaia residuals, LOG it (one line) and write the sidecar
        #      "<stem>.frame_tie.json" FOR THE RECORD only. The sidecar is never
        #      read back into POS_ERR.
        # Deprecated alias `gaia_pos_err_frame_tie` (any value -> "off") is
        # accepted with a one-time warning so old configs -- including ones that
        # said "auto" -- no longer silently modify POS_ERR.
        "frame_tie_diagnostic": "off",
        "plot_mag_lo": 10.0,
        "plot_mag_hi": 20.0,
        "plot_zp_offset": 0.25,
        "zp_err_method": "bright_std",  # honest MAGERR floor from high-S/N calibrators (adopted 2026-06-12; "all_mad" = legacy, adds the all-kept MAD in quadrature)
        "zp_err_snr_min": 100.0,
        # S/N floor for ZP DERIVATION (which calibrators define the per-band
        # ZP). 0 = off (exact legacy: no cut). Faint calibrators carry photon
        # noise plus a one-sided Gaia-XP crowding bias; with the weighted
        # median they barely move the ZP value (<=0.0013 mag measured on
        # T08285) but inflate the reported kept-MAD ~25%. Bookkeeping only --
        # a kept-MAD drop from this cut is NOT a photometry improvement.
        "min_calibrator_snr": 20.0,
        # Spatially-varying ZP (ubercal-style flat/illumination refinement).
        # A single scalar ZP per band leaves a position-dependent achromatic
        # ZP residual (flat/illumination/coadd), most significant in crowded /
        # low-galactic-latitude fields (measured ~0.015 mag RMS, ~16% of bright
        # zp_err, on the T08285 bulge field). The model is fit to the kept Gaia
        # stars and applied to every source's MAG.
        #   "poly"     (DEFAULT, 2026-06-12): robust 2-D polynomial in
        #                normalized image coords (smooth; no grid
        #                discontinuities). Validated on T08285 (crowded,
        #                kept-MAD -15%, bright-std -6% median; held-out
        #                split-half confirmed; pattern matches the independent
        #                60-tile star-flat maps r=0.84-0.95 on the strong
        #                telescopes) and T08676 (sparse, kept-MAD -4%, worst
        #                telescopes 7DT10/11 -20..-40%). The scalar zp_median
        #                anchor is unchanged (the model is a pure delta around
        #                it). Bands with < 10x poly terms calibrators fall
        #                back to scalar automatically.
        #   "scalar"   : exact-legacy single ZP per band.
        #   "per_grid" : per-grid (grid_id_fit) median offset (captures the
        #                full per-grid structure; cells with < min_per_cell
        #                Gaia stars fall back to the scalar ZP).
        "spatial_zp_model": "poly",
        "spatial_zp_poly_order": 4,
        "spatial_zp_min_per_cell": 30,
        # Gaia-XP reference quality filter (DEFAULT ON). In crowded/low-b fields,
        # Gaia's own BP/RP spectra are blended -> contaminated XP synphot -> a
        # per-source gray ZP error (verified on T08285: flagged sources have ~2x
        # the scatter; removing them improves zp_err ~14% with the ZP median
        # unbiased). Excludes Gaia calibrators with poor astrometry
        # (RUWE > gaia_max_ruwe) or anomalous corrected BP-RP excess factor
        # (|C*| > gaia_cstar_nsigma * sigma_C*(G), Riello+2021) from the ZP
        # DERIVATION only -- it does NOT change injection (inject_gaia_sources)
        # or which sources are fit/output. Needs `ruwe` and
        # `phot_bp_rp_excess_factor` per calibrator (plus bp_rp,
        # phot_g_mean_mag in the GaiaXP CSV): taken from the CSV if it has
        # them (`tract7dt augment-gaia`), else from <work_dir>/ZP/
        # gaia_quality_cache.csv, else fetched from the Gaia DR3 archive in
        # the background and awaited (bounded) before the GPU fit. If they
        # cannot be obtained the run fails before the fit (R29): a ZP is
        # never calibrated on the unfiltered set behind the user's back.
        # Self-targeting: removes ~0% in sparse high-|b| fields, ~30% in the
        # bulge. Set false to disable (recorded in run_provenance.json).
        "gaia_quality_filter": True,
        "gaia_max_ruwe": 1.4,
        "gaia_cstar_nsigma": 3.0,
        # If the filter would leave fewer than this many ZP calibrators, fall
        # back to the UNFILTERED set rather than calibrating on a thin sample.
        "gaia_min_calibrators": 10,
        # Save the gaia_augmentation_overlay.png diagnostic plot.
        # Dispatched to the background plot pool, so it doesn't block
        # the bright_mask stage. Default on.
        "save_gaia_overlay": True,
        # Performance only (NO science effect). Width of the parallel,
        # byte-identical CSV writer used for output_catalog.csv and the
        # per-object ZP CSV. pandas to_csv is single-threaded float->string
        # formatting; contiguous row-blocks format identically and concatenate
        # exactly, so the file is byte-for-byte identical to a 1-worker write.
        # 8 keeps the writer + plot pool under a ~32-core budget; 1 = serial.
        "catalog_write_workers": 8,
        # Worker processes for the ZP diagnostic plots (co-critical with the
        # catalog write). 0 = auto (= min(n_bands, 20)); an int forces the count.
        # Identical PNGs -- only the dispatch parallelism changes.
        "plot_workers": 0,
    },
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_path(value: Any, base_dir: Path) -> Path | None:
    if value is None:
        return None
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def _join_path(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def _validate_user_config(user_cfg: dict[str, Any], defaults: dict[str, Any], prefix: str = "") -> None:
    """Validate user_cfg keys against defaults: KeyError on unknown keys
    (retired keys were already handled by ``_drop_retired_keys``), recursing
    into nested dicts."""
    if not isinstance(user_cfg, dict):
        raise TypeError("Config root must be a mapping (YAML dict).")
    for k, v in user_cfg.items():
        full = _join_path(prefix, str(k))
        if k not in defaults:
            raise KeyError(f"Unknown config key: {full}")
        d = defaults[k]
        if isinstance(v, dict) and isinstance(d, dict):
            _validate_user_config(v, d, full)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_config_types(cfg: dict[str, Any], defaults: dict[str, Any], prefix: str = "") -> None:
    type_overrides: dict[str, tuple[type, ...]] = {
        "performance.frame_prep_workers": (int, str),
        "performance.white_stack_workers": (int, str),
        "logging.file": (str, Path, type(None)),
        "zp.gaia_pos_err_pix": (float, int, type(None)),
        # frame-tie field-level diagnostic: "off" (default) / "measure".
        "zp.frame_tie_diagnostic": (str,),
        # GPU compute config (new grid-based layout):
        #   devices is "auto" or list[int]; jit_cache_dir is "auto" or
        #   path-like or null.
        "compute.gpu.devices": (str, list),
        "compute.gpu.jit_cache_dir": (str, Path, type(None)),
        # PSFEx-style PSF builder: max_workers accepts int, "all_bands",
        # or "auto". The runtime parser in psf_pipeline_stage.py handles
        # the string forms (all_bands -> n_bands, auto -> n_cpus // 8).
        "compute.gpu.psf.from_stars.max_workers": (int, str),
        # Post-solve error model: slow-converge column enable flag (bool or
        # null; threshold is the solver's own convergence budget, not truth).
        "compute.gpu.error_model.slow_converge_flag": (bool, type(None)),
        "compute.gpu.error_model.rchisq_flag_thresh": (int, float, type(None)),
    }
    for k, d in defaults.items():
        path = _join_path(prefix, str(k))
        if k not in cfg:
            continue
        v = cfg[k]
        if path in type_overrides:
            if not isinstance(v, type_overrides[path]):
                allowed = ", ".join(t.__name__ for t in type_overrides[path])
                raise TypeError(f"Config key '{path}' must be one of: {allowed}")
            continue
        if isinstance(d, dict):
            if not isinstance(v, dict):
                raise TypeError(f"Config key '{path}' must be a mapping (YAML dict).")
            _validate_config_types(v, d, path)
            continue
        if isinstance(d, list):
            if not isinstance(v, list):
                raise TypeError(f"Config key '{path}' must be a list.")
            continue
        if d is None:
            continue
        if isinstance(d, bool):
            if not isinstance(v, bool):
                raise TypeError(f"Config key '{path}' must be a bool.")
            continue
        if isinstance(d, int):
            if not _is_int(v):
                raise TypeError(f"Config key '{path}' must be an int.")
            continue
        if isinstance(d, float):
            if not (_is_int(v) or isinstance(v, float)):
                raise TypeError(f"Config key '{path}' must be a float.")
            continue
        if isinstance(d, str):
            if not isinstance(v, str):
                raise TypeError(f"Config key '{path}' must be a string.")
            continue


def _validate_solver_semantics(cfg: dict[str, Any]) -> None:
    """Reject degenerate grid-GPU solver combinations.

    ``compute.gpu.solver.single_stage: true`` with ``n_opt_iters <= 0``
    outside island mode runs ZERO nonlinear iterations yet returns
    ``opt_converged=True`` with every position/shape still at the catalog
    seed -- i.e. a full catalog of un-fit seed geometry certified as a
    converged, reliable fit. That combination is only ever valid
    INTERNALLY for ``position_solver: island`` (both per-grid passes are
    S1-only by design, and the island stage overrides the convergence
    flags with the true per-source island fit), so it is gated out here
    for every other solver. Only meaningful for the grid-GPU path.
    """
    compute = cfg.get("compute", {}) or {}
    solver = ((compute.get("gpu", {}) or {}).get("solver", {}) or {})
    # The "per_source" fallback here is dead: this validator runs AFTER
    # _deep_update onto DEFAULT_CONFIG (single source of truth for the shipped
    # default, currently "island"), so the key is always present. Kept only as
    # defense-in-depth for a hand-built cfg dict.
    position_solver = str(
        solver.get("position_solver", "per_source")).strip().lower()
    single_stage = bool(solver.get("single_stage", False))
    try:
        n_opt_iters = int(solver.get("n_opt_iters", 20))
    except (TypeError, ValueError):
        return  # type validator already flagged a non-int
    if single_stage and n_opt_iters <= 0 and position_solver != "island":
        raise ValueError(
            "compute.gpu.solver: single_stage=true with n_opt_iters<=0 fits "
            "no positions/shapes, yet would report opt_converged=True with "
            "the *_fit columns equal to the catalog seeds. This "
            "zero-iteration frozen-at-seed combination is only valid "
            "internally for position_solver='island'. Set n_opt_iters>=1 or "
            "use position_solver='island'."
        )
    # #35: island stall-counter mode is a closed two-value enum; fail at
    # config load rather than deep inside the island stage.
    island_block = solver.get("island", {}) or {}
    scm = str(island_block.get(
        "stall_count_mode", "proposals")).strip().lower()
    if scm not in ("proposals", "changed"):
        raise ValueError(
            "compute.gpu.solver.island.stall_count_mode must be "
            f"'proposals' or 'changed', got {scm!r}"
        )
    # dlnp campaign: convergence-mode enum + its interaction locks (the
    # island stage re-validates; failing at config load is friendlier).
    conv = str(island_block.get(
        "convergence", "legacy")).strip().lower()
    if conv not in ("legacy", "dlnp"):
        raise ValueError(
            "compute.gpu.solver.island.convergence must be 'legacy' or "
            f"'dlnp', got {conv!r}"
        )
    if conv == "dlnp":
        smode = str(island_block.get("step_mode", "clip")).strip().lower()
        if smode != "clip":
            raise ValueError(
                "compute.gpu.solver.island.convergence: 'dlnp' requires "
                "step_mode: 'clip' (trust-region support pending)")
        if scm == "changed":
            raise ValueError(
                "compute.gpu.solver.island.convergence: 'dlnp' subsumes "
                "stall_count_mode: 'changed' (distinct-proposal counting "
                "is the dlnp behavior); leave stall_count_mode at "
                "'proposals'")


# Keys removed from the engine. ``load_config``/``resolve_config`` handle a user
# config that still pins them (campaign YAMLs pin every key) by kind:
#   "ignore"  - any value: dropped (the key no longer affects any result);
#   "only"    - the single value that remains: dropped; any other value -> ValueError;
#   "default" - the dev8 default (a removed opt-in at its off/default value, or a
#               value the remaining code now hard-codes): dropped; anything else ->
#               ValueError, because silently ignoring it would drop a feature the
#               user turned on (it is still available in tract7dt@dev8);
#   "reject"  - any value -> ValueError: the key used to select code that is gone
#               and no value means the remaining behaviour.
# A path may name a whole block ("epsf").
# ``compute.mode`` is also refused next to a compute.gpu block in the pre-2026-04
# per-patch GPU layout (``_dev8_legacy_gpu_layout``).
_RETIRED_KEYS: dict[str, tuple[str, Any, str]] = {
    'source_saturation_cut.require_all_bands': ("ignore", None,
        'saturation is now handled per band (a band near a saturated pixel is excluded in that band only; the source leaves the fit only when every band is affected), so the any-band/all-band switch is gone'),
    'zp.min_box_size_pix': ("ignore", None,
        'the Gaia ZP calibrators are now every GaiaXP source in the image, independent of the input catalog, so there is no selection box'),
    'zp.gaia_quality_autofetch': ("ignore", None,
        "with zp.gaia_quality_filter on, the Gaia quality columns are now always obtained (GaiaXP CSV, the run's cache, or the Gaia archive) before the GPU fit, and the run fails if they cannot be. Offline: pre-fetch with `tract7dt augment-gaia <csv>`, or set zp.gaia_quality_filter: false"),
    'zp.gaia_pos_err_frame_tie': ("ignore", None,
        'reverted 2026-07-11: POS_ERR stays pure-propagated; set zp.frame_tie_diagnostic: measure to log the field offset'),
    'compute.gpu.error_model.slow_converge_niters': ("ignore", None,
        'replaced 2026-07-11 by slow_converge_flag (bool)'),
    'compute.gpu.framework': ("reject", None,
        'a pre-2026-04 per-patch GPU key, unread since; tract7dt@dev8 read a compute.gpu '
        'block holding it without grid/psf/background/solver/positions as the old '
        'per-patch GPU layout, which could select the removed legacy per-patch GPU path. '
        'Delete the key (the grid GPU path is the only path)'),
    # One pass, no opt-in left: the two-pass residual-phantom form and the keys that only
    # served it.
    'bright_mask.residual_phantom_pass.enabled': ('default', False, 'the two-pass residual-phantom form is removed (uniphot 0.1.0: one pass); it existed in the unreleased tract7dt dev8'),
    'bright_mask.residual_phantom_pass.nsigma': ('ignore', None, 'knob of the removed two-pass form (uniphot 0.1.0)'),
    'bright_mask.residual_phantom_pass.minarea': ('ignore', None, 'knob of the removed two-pass form (uniphot 0.1.0)'),
    'bright_mask.residual_phantom_pass.match_radius_pix': ('ignore', None, 'knob of the removed two-pass form (uniphot 0.1.0)'),
    'bright_mask.extra_unmatched_csv': ('default', None, "the external unmatched-detection list (the two-pass form's speck channel) is removed (uniphot 0.1.0); it existed in the unreleased tract7dt dev8"),
    'compute.gpu.background.dump_diagnostics': ('default', False, "the bg_diag dump (the two-pass form's residual input) is removed (uniphot 0.1.0); it existed in the unreleased tract7dt dev8"),
    'zp.calibrate': ('default', True, "skipping only the ZP solve was the two-pass form's pass-A setting, removed with it (uniphot 0.1.0); it existed in the unreleased tract7dt dev8 (zp.enabled: false skips the whole ZP stage, Gaia source augmentation included)"),
    # uniphot 0.1.1: the bright-object mask's detection-catalogue association radius is measured in each run.
    'bright_mask.match_radius_pix': ('default', 3.0, "the fixed association radius is replaced (uniphot 0.1.1): a white-stack detection is the light of a listed source when its light peak lies within max(the run's PSF FWHM, 3.72 x the list-image scatter measured in the run) of the row, at most compute.gpu.positions.pos_max_shift_pix; 3.0 was the uniphot 0.1.0 default"),
    'epsf': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patches': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_inputs': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'runner': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'moffat_psf': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'compute.legacy_gpu': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'outputs.epsf_dir': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'outputs.patches_dir': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    # PSF training-star cuts that a continuous star weight replaces (new PSF model design): the
    # value equal to the remaining behaviour loads with the note, any other value fails at load.
    'compute.gpu.psf.from_stars.max_ellipticity': ('default', 10.0, 'the ellipticity cut on PSF training stars is removed (uniphot 0.1.0): cleanliness is a continuous star weight; 10.0, the dev8 default, cut nothing (a SEP ellipticity is below 1); it existed in the unreleased tract7dt dev8'),
    'compute.gpu.psf.from_stars.min_sep': ('only', 0.0, 'the greedy minimum-separation rule between PSF training stars is removed (uniphot 0.1.0): a neighbour lowers the star weight continuously and a brighter neighbour peak inside the 55 px box excludes the star; 0 (no rule) is what remains, so the dev8 default 30 is refused too -- delete the key; it existed in the unreleased tract7dt dev8'),
    'outputs.patch_inputs_dir': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'outputs.tractor_out_dir': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'merge.pattern': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'compute.gpu.psf.from_cells': ('ignore', None, 'psf.mode from_cells removed (uniphot 0.1.0)'),
    'patch_run.enable_multi_band_simultaneous_fitting': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.python_exe': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.resume': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.max_workers': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.threads_per_process': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.n_opt_iters': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.dlnp_stop': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.r_ap': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.eps_flux': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.sersic_n_init': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.psf_model': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.psf_fallback_model': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.min_epsf_nstars_for_use': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.ncircular_gaussian_n': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.hybrid_psf_n': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.gaussian_mixture_n': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.fallback_default_fwhm_pix': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.fallback_stamp_radius_pix': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.cutout_size': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.cutout_max_sources': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.cutout_start': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.cutout_data_p_lo': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.cutout_data_p_hi': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.cutout_model_p_lo': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.cutout_model_p_hi': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.cutout_resid_p_abs': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.no_cutouts': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.no_patch_overview': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.flag_maxiter_margin': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.pos_max_shift_pix': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'patch_run.enable_staged_fit': ('ignore', None, 'CPU/Tractor per-patch path removed (uniphot 0.1.0)'),
    'compute.mode': ('only', 'gpu', 'only the GPU path remains (uniphot 0.1.0)'),
    'compute.gpu.psf.mode': ('only', 'from_stars', 'only the from_stars PSF remains (uniphot 0.1.0)'),
    'compute.gpu.psf.from_stars.psf_normalization': ('only', 'sum', 'only the sum-normalised PSF-star path remains (uniphot 0.1.0)'),
    'compute.gpu.psf.from_stars.apply_fwhm_cut': ('default', False, 'opt-in removed (uniphot 0.1.0); it existed in the unreleased tract7dt dev8'),
    'compute.gpu.psf.from_stars.use_gaia_match': ('default', False, 'opt-in removed (uniphot 0.1.0); it existed in the unreleased tract7dt dev8'),
    'compute.gpu.psf.from_stars.use_segmap': ('default', False, 'opt-in removed (uniphot 0.1.0); it existed in the unreleased tract7dt dev8'),
    'compute.gpu.psf.from_stars.per_pixel_weights': ('default', False, 'opt-in removed (uniphot 0.1.0); it existed in the unreleased tract7dt dev8'),
    'compute.gpu.psf.from_stars.q_range': ('ignore', None, 'knob of a removed opt-in PSF path; inert at the enforced defaults (uniphot 0.1.0)'),
    'compute.gpu.psf.from_stars.gaia_match_r_pix': ('ignore', None, 'knob of a removed opt-in PSF path; inert at the enforced defaults (uniphot 0.1.0)'),
    'compute.gpu.psf.from_stars.gaia_mag_min': ('ignore', None, 'knob of a removed opt-in PSF path; inert at the enforced defaults (uniphot 0.1.0)'),
    'compute.gpu.psf.from_stars.gaia_mag_max': ('ignore', None, 'knob of a removed opt-in PSF path; inert at the enforced defaults (uniphot 0.1.0)'),
    'compute.gpu.psf.from_stars.crowd_norm_threshold': ('ignore', None, 'knob of a removed opt-in PSF path; inert at the enforced defaults (uniphot 0.1.0)'),
    'compute.gpu.psf.from_stars.norm_core_fwhm_mult': ('ignore', None, 'knob of a removed opt-in PSF path; inert at the enforced defaults (uniphot 0.1.0)'),
    'compute.gpu.psf.from_stars.neighbor_mask_mode': ('ignore', None, 'knob of a removed opt-in PSF path; inert at the enforced defaults (uniphot 0.1.0)'),
    'compute.gpu.psf.from_stars.neighbor_inner_keep_fwhm': ('ignore', None, 'knob of a removed opt-in PSF path; inert at the enforced defaults (uniphot 0.1.0)'),
    'compute.gpu.psf.from_stars.star_local_bkg_method': ('ignore', None, 'knob of a removed opt-in PSF path; inert at the enforced defaults (uniphot 0.1.0)'),
    'compute.gpu.psf.from_stars.per_pixel_irls': ('ignore', None, 'knob of a removed opt-in PSF path; inert at the enforced defaults (uniphot 0.1.0)'),
    'compute.gpu.psf.from_stars.max_irls_iters': ('ignore', None, 'knob of a removed opt-in PSF path; inert at the enforced defaults (uniphot 0.1.0)'),
    'compute.gpu.psf.from_stars.irls_c': ('ignore', None, 'knob of a removed opt-in PSF path; inert at the enforced defaults (uniphot 0.1.0)'),
    'compute.gpu.psf.from_stars.psf_accuracy_alpha': ('ignore', None, 'knob of a removed opt-in PSF path; inert at the enforced defaults (uniphot 0.1.0)'),
    'compute.gpu.psf.from_stars.fit_reg_eps_abs': ('ignore', None, 'knob of a removed opt-in PSF path; inert at the enforced defaults (uniphot 0.1.0)'),
    'compute.gpu.psf.from_stars.min_neff_per_pixel': ('ignore', None, 'knob of a removed opt-in PSF path; inert at the enforced defaults (uniphot 0.1.0)'),
    'compute.gpu.solver.use_batched_band_solver': ('default', False, 'opt-in removed (uniphot 0.1.0); it existed in the unreleased tract7dt dev8'),
    'compute.gpu.solver.island.subtract_detected': ('default', False, 'opt-in removed (uniphot 0.1.0); it existed in the unreleased tract7dt dev8'),
    'compute.gpu.solver.island.subtract_detected_protect': ('default', True, 'opt-in removed (uniphot 0.1.0); it existed in the unreleased tract7dt dev8'),
    'compute.gpu.solver.island.confusion_weight': ('default', False, 'opt-in removed (uniphot 0.1.0); it existed in the unreleased tract7dt dev8'),
    'compute.gpu.solver.island.blank_response_extended': ('default', False, 'opt-in removed (uniphot 0.1.0); it existed in the unreleased tract7dt dev8'),
    'compute.gpu.solver.island.detect_unattributed.enabled': ('default', False, 'opt-in removed (uniphot 0.1.0); it existed in the unreleased tract7dt dev8'),
    'compute.gpu.solver.island.detect_unattributed.nsigma': ('default', 5.0, 'dev8 value still used by bright_mask DET routing (uniphot 0.1.0)'),
    'compute.gpu.solver.island.detect_unattributed.max_iter': ('ignore', None, 'knob of removed detect_unattributed (uniphot 0.1.0)'),
    'compute.gpu.solver.island.detect_unattributed.min_separation_fwhm': ('ignore', None, 'knob of removed detect_unattributed (uniphot 0.1.0)'),
    'compute.gpu.solver.island.detect_unattributed.min_separation_re': ('ignore', None, 'knob of removed detect_unattributed (uniphot 0.1.0)'),
    'compute.gpu.solver.island.detect_unattributed.min_area': ('ignore', None, 'knob of removed detect_unattributed (uniphot 0.1.0)'),
    'compute.gpu.solver.island.detect_unattributed.source_type': ('default', 'sersic', 'dev8 value still used by bright_mask DET routing (uniphot 0.1.0)'),
    'compute.gpu.solver.island.detect_unattributed.max_new_per_island': ('ignore', None, 'knob of removed detect_unattributed (uniphot 0.1.0)'),
    'compute.gpu.solver.island.detect_unattributed.max_new_total': ('ignore', None, 'knob of removed detect_unattributed (uniphot 0.1.0)'),
    'compute.gpu.solver.island.detect_unattributed.psf_matched_filter': ('ignore', None, 'knob of removed detect_unattributed (uniphot 0.1.0)'),
    'compute.gpu.solver.island.detect_unattributed.filter_fwhm_scale': ('ignore', None, 'knob of removed detect_unattributed (uniphot 0.1.0)'),
    'compute.gpu.solver.island.detect_unattributed.deblend_nthresh': ('ignore', None, 'knob of removed detect_unattributed (uniphot 0.1.0)'),
    'compute.gpu.solver.island.detect_unattributed.deblend_cont': ('ignore', None, 'knob of removed detect_unattributed (uniphot 0.1.0)'),
    'compute.gpu.solver.island.detect_unattributed.clean': ('ignore', None, 'knob of removed detect_unattributed (uniphot 0.1.0)'),
    'compute.gpu.solver.island.detect_unattributed.clean_param': ('ignore', None, 'knob of removed detect_unattributed (uniphot 0.1.0)'),
    'compute.gpu.solver.island.model_uncertainty.enabled': ('default', False, 'opt-in removed (uniphot 0.1.0); it existed in the unreleased tract7dt dev8'),
    'compute.gpu.solver.island.model_uncertainty.use_bg': ('ignore', None, 'knob of removed model_uncertainty (uniphot 0.1.0)'),
    'compute.gpu.solver.island.model_uncertainty.use_sub': ('ignore', None, 'knob of removed model_uncertainty (uniphot 0.1.0)'),
    'compute.gpu.solver.island.model_uncertainty.use_psf': ('ignore', None, 'knob of removed model_uncertainty (uniphot 0.1.0)'),
    'compute.gpu.solver.island.model_uncertainty.use_conf': ('ignore', None, 'knob of removed model_uncertainty (uniphot 0.1.0)'),
    'compute.gpu.solver.island.confusion_whitening.enabled': ('default', False, 'opt-in removed (uniphot 0.1.0); it existed in the unreleased tract7dt dev8'),
    'compute.gpu.solver.island.confusion_whitening.s_file': ('ignore', None, 'knob of removed confusion_whitening (uniphot 0.1.0)'),
    'compute.gpu.solver.island.confusion_whitening.mode': ('ignore', None, 'knob of removed confusion_whitening (uniphot 0.1.0)'),
    'compute.gpu.solver.island.confusion_whitening.local': ('ignore', None, 'knob of removed confusion_whitening (uniphot 0.1.0)'),
    'compute.gpu.galaxy_diagnostics.enabled': ('default', False, 'opt-in removed (uniphot 0.1.0); it existed in the unreleased tract7dt dev8'),
    'compute.gpu.galaxy_diagnostics.max_sources': ('ignore', None, 'knob of removed galaxy_diagnostics (uniphot 0.1.0)'),
    'compute.gpu.galaxy_diagnostics.cutout_size': ('ignore', None, 'knob of removed galaxy_diagnostics (uniphot 0.1.0)'),
    'compute.gpu.error_model.re_profile_enabled': ('default', False, 'opt-in removed (uniphot 0.1.0); it existed in the unreleased tract7dt dev8'),
}

def _flatten_paths(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        p = _join_path(prefix, str(k))
        if isinstance(v, dict) and v:
            out.update(_flatten_paths(v, p))
        else:
            out[p] = v
    return out


def _same_value(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


# tract7dt@dev8's _migrate_legacy_compute_keys read ``compute.mode: gpu`` as the
# removed legacy per-patch GPU path when ``compute.gpu`` held one of these per-patch
# keys and none of the grid-layout groups, and there was no ``compute.gpu_grid`` block.
_DEV8_PER_PATCH_GPU_KEYS = frozenset({
    "framework", "devices", "device_assignment", "dtype", "fallback_to_cpu_on_unsupported",
    "log_jit_events", "jacobian_mode", "padded_layout", "bucket_sources"})
_GPU_LAYOUT_GROUPS = frozenset({"grid", "psf", "background", "solver", "positions"})


def _dev8_legacy_gpu_layout(user_cfg: dict[str, Any]) -> str | None:
    """The error line for compute.mode next to a compute.gpu block in the pre-2026-04
    per-patch layout (dev8's own test on the user's keys), else None.

    dev8 moved such a block to compute.legacy_gpu whatever compute.mode said, and
    switched the run to the legacy per-patch GPU path only for the exact string 'gpu'.
    Here the mode test is the 'only' rule's ('GPU', ' gpu' count as gpu), so those are
    refused too; the message claims the switch only where dev8 made it."""
    comp = user_cfg.get("compute")
    if not isinstance(comp, dict) or "mode" not in comp or not _same_value(comp["mode"], "gpu"):
        return None
    gpu = comp.get("gpu")
    if not isinstance(gpu, dict) or isinstance(comp.get("gpu_grid"), dict):
        return None
    keys = set(gpu)
    if not keys & _DEV8_PER_PATCH_GPU_KEYS or keys & _GPU_LAYOUT_GROUPS:
        return None
    switched = (" and switched compute.mode 'gpu' to the removed legacy per-patch GPU path"
                if comp["mode"] == "gpu" else "")
    return (f"compute.mode={comp['mode']!r} with compute.gpu keys {sorted(keys)} (retired; "
            "tract7dt@dev8 read a compute.gpu block with these keys and none of "
            f"grid/psf/background/solver/positions as the old per-patch GPU layout{switched}. "
            "Delete compute.mode to run the grid GPU path with these settings)")


def _drop_retired_keys(user_cfg: dict[str, Any]) -> list[str]:
    """Apply ``_RETIRED_KEYS`` to a user config in place.

    Drops retired keys (see the table for the four kinds), prunes blocks
    left empty, writes ONE stderr note listing what was dropped, and raises
    ValueError listing every retired key whose value would have changed the
    result (including a dev8 legacy per-patch GPU layout). Returns the
    dropped dotted paths.
    """
    if not isinstance(user_cfg, dict):
        return []
    flat = _flatten_paths(user_cfg)
    dropped: list[str] = []
    errors: list[str] = []
    for path, val in flat.items():
        hit = None
        parts = path.split(".")
        for n in range(len(parts), 0, -1):
            cand = ".".join(parts[:n])
            if cand in _RETIRED_KEYS:
                hit = cand
                break
        if hit is None:
            continue
        kind, keep, why = _RETIRED_KEYS[hit]
        if kind == "reject":
            errors.append(f"{path}={val!r} (retired; no value is accepted: {why})")
            continue
        if kind in ("only", "default") and not (hit == path and _same_value(val, keep)):
            errors.append(f"{path}={val!r} (retired; only {keep!r} is accepted: {why})")
            continue
        dropped.append((path, hit))
    legacy = _dev8_legacy_gpu_layout(user_cfg)
    if legacy is not None:
        errors.append(legacy)
    if errors:
        raise ValueError("retired config key(s) with a value that is no longer "
                         "supported:\n  " + "\n  ".join(errors))

    def _pop(d: dict[str, Any], parts: list[str]) -> None:
        if len(parts) == 1:
            d.pop(parts[0], None)
            return
        sub = d.get(parts[0])
        if isinstance(sub, dict):
            _pop(sub, parts[1:])
            if not sub:
                d.pop(parts[0], None)

    for path, _ in dropped:
        _pop(user_cfg, path.split("."))
    if dropped:
        import sys as _sys
        entries = sorted({hit for _, hit in dropped})
        _sys.stderr.write(
            f"[tract7dt] NOTE: {len(dropped)} retired config key(s) ignored "
            f"(removed from the engine; value at the dev8 default or inert) "
            f"under: " + ", ".join(entries) + "\n")
    return [path for path, _ in dropped]


def resolve_config(user_cfg: dict[str, Any] | None, base_dir: str | Path) -> dict[str, Any]:
    """Turn a user config mapping into the engine's resolved config.

    The one code path behind ``load_config`` (YAML) and ``run_pipeline`` (a
    hand-built dict): retired keys (``_RETIRED_KEYS``) -> unknown-key check ->
    deep merge onto ``DEFAULT_CONFIG`` -> type and semantic checks -> paths
    (inputs relative to ``base_dir``, outputs relative to ``work_dir``). The
    caller's mapping is not modified. The result carries ``config_dir``
    (= ``base_dir``) and ``config_path`` (None here; ``load_config`` sets it),
    which is how ``run_pipeline`` recognises an already-resolved config.
    """
    import copy as _copy
    user_cfg = _copy.deepcopy(user_cfg) if user_cfg else {}
    base_dir = Path(base_dir).expanduser().resolve()

    _drop_retired_keys(user_cfg)

    _validate_user_config(user_cfg, DEFAULT_CONFIG)
    # Deep copy: _deep_update shares every sub-dict the user did not
    # override, so the path resolution below would otherwise write into
    # DEFAULT_CONFIG itself (e.g. inputs.* for a dict without `inputs`).
    cfg = _deep_update(_copy.deepcopy(DEFAULT_CONFIG), user_cfg)
    _validate_config_types(cfg, DEFAULT_CONFIG)
    _validate_solver_semantics(cfg)

    cfg["config_path"] = None
    cfg["config_dir"] = base_dir

    inputs = cfg["inputs"]
    outputs = cfg["outputs"]

    inputs["input_catalog"] = _resolve_path(inputs["input_catalog"], base_dir)
    inputs["image_list_file"] = _resolve_path(inputs["image_list_file"], base_dir)
    inputs["gaiaxp_synphot_csv"] = _resolve_path(inputs["gaiaxp_synphot_csv"], base_dir)

    work_dir = _resolve_path(outputs["work_dir"], base_dir)
    outputs["work_dir"] = work_dir
    outputs["final_catalog"] = _resolve_path(outputs["final_catalog"], work_dir)
    outputs["cropped_images_dir"] = _resolve_path(outputs["cropped_images_dir"], work_dir)
    outputs["overlay_dir"] = _resolve_path(outputs["overlay_dir"], work_dir)

    logging_cfg = cfg["logging"]
    log_file = logging_cfg.get("file")
    if log_file is not None and str(log_file).strip().lower() != "auto":
        logging_cfg["file"] = _resolve_path(log_file, work_dir)

    return cfg


def is_resolved_config(cfg: dict[str, Any]) -> bool:
    """True for the output of ``resolve_config`` / ``load_config`` (read only)."""
    return isinstance(cfg, dict) and "config_dir" in cfg and "config_path" in cfg


def load_config(config_path: str | Path) -> dict[str, Any]:
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = resolve_config(user_cfg, cfg_path.parent)
    cfg["config_path"] = cfg_path
    return cfg


def write_sample_config(out_path: str | Path, overwrite: bool = False) -> Path:
    out = Path(out_path).expanduser()
    out_abs = out.resolve()
    sample_path = Path(__file__).resolve().parent / "data" / "sample_config.yaml"
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample config not found: {sample_path}")
    if out_abs.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {out_abs}")
    out_abs.parent.mkdir(parents=True, exist_ok=True)
    out_abs.write_text(sample_path.read_text(encoding="utf-8"), encoding="utf-8")
    return out_abs
