"""Post-fit per-source flux-error decomposition for the grid-GPU pipeline.

Motivation (task #27, 2026-07): in crowded fields the reported
``FLUXERR_*_fit`` under-covers the TRUE flux error of extended sources by
an order of magnitude (T08285 bulge light-injection: pull MAD ~15-30 vs
the expected 1). The reported error is the exact MARGINAL statistical
error of the linear flux solve -- ``diag(inv(A^T W A))`` over sources +
background, so modeled-neighbour degeneracy is already included (audited:
the k_max-truncated marginal matches an exact local normal-matrix
inversion to 0.2% median / 5% at p90). What is MISSING is the error from
everything the model does NOT contain: unmodeled faint sources, neighbour
model mismatch, PSF/shape errors -- i.e. CONFUSION noise.

This module MEASURES that confusion term from the run's own residual map,
with no tuned constants:

1. residual  r = data - model(all fitted sources incl. big-source wing
   tiers) - fitted background;
2. matched-filter map  MF(x) = corr(r*g, phi) / corr(g, phi^2)  where
   phi is the band's central PSF (unit sum) and g the good-pixel mask.
   MF(x) is exactly the flux perturbation a forced PSF fit centred at x
   would absorb from the residual structure;
3. per-pixel statistical expectation
   V_stat(x) = corr(sigma^2*g, phi^2) / corr(g, phi^2)^2
   (== sigma_sky^2/sum(phi^2) for uniform weights, the isolated-source
   statistical variance);
4. local confusion variance = max(0, Var_annulus[MF] - <V_stat>_annulus)
   in an annulus r in [2, 6] x FWHM_eff around each source. The inner
   radius excludes the source's own fit-suppressed core (the normal
   equations zero phi^T W r AT the source), the outer radius keeps the
   estimate local to the source's crowding environment; the local mean
   <MF> is subtracted so smooth background residue does not count;
5. extended-source kernel factor: a fit template wider than the PSF
   integrates more confusion. Under a locally-white field of PSF-shaped
   residual bumps the exact Gaussian-equivalent ratio is
       R = 2*s_g^4 / ((s_g^2 + s_p^2) * s_p^2),   R(star) = 1,
   with s_p^2 = 1/(4*pi*sum(phi^2)) the PSF's Gaussian-equivalent
   variance and s_g^2 = s_p^2 + (Re_fit/1.1774)^2 the convolved
   template's. This is analytic (no fitted coefficient); the Gaussian
   equivalence and bump whiteness are the documented approximations.

   sigma_conf(source, band) = sqrt(R * confusion variance at source)

Validated offline on the T08285 bulge injections (crowded; pulls
med/MAD +5.7/17.7 -> +0.6/1.9 on m600) and UDS injections (sparse;
med/MAD -2.3/7.3 -> -1.0/3.1). A free bonus of measuring the excess
from the real map: COADD CORRELATED PIXEL NOISE (resampling kernels;
measured rho(1px) ~ 0.15 on the UDS m600 coadd -> true statistical
flux variance is ~3.2x the white-noise FLUXERR^2) is automatically
absorbed, because the local map variance is real while only the WHITE
expectation is subtracted -- the UDS isolated-star inflation
FLUXERR_TOT/FLUXERR ~ 1.5-1.7 matches the measured sqrt(3.2) = 1.8.
The internal design notes (not distributed) give the full
error-budget tables and residual caveats (systematic flux BIASES --
crowding absorption, flux-definition offsets, the few-percent bright
extended-source floor -- are NOT absorbed here; error bars must not
hide a bias; they are reported separately).

Geometry choices (documented, not calibration): annulus [2, 6] x
FWHM_eff, kernel-weighted good-pixel occupancy >= 0.6 for a valid MF
sample, >= 50 valid annulus pixels for a valid variance. Sources whose
annulus is invalid (heavy masking / grid edge) get sigma_conf = NaN and
``FLUXERR_TOT_* = FLUXERR_*`` downstream.

Cost: one extra render + ~4 grid-sized FFTs per band per grid, after the
solver has converged; existing outputs are untouched (read-only on the
solve state). Disable via ``compute.gpu.error_model.confusion_enabled``.
"""
from __future__ import annotations

import ctypes
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

__all__ = ["compute_grid_confusion", "measure_correlated_noise_factor",
           "measure_correlated_noise_factor_residual"]

# Geometric / robustness choices (see module docstring; these define the
# measurement window, they are not calibration factors).
_OCC_MIN = 0.6          # min kernel-weighted good-pixel occupancy for MF
_R_IN_FWHM = 2.0        # annulus inner radius, units of FWHM_eff
_R_OUT_FWHM = 6.0       # annulus outer radius, units of FWHM_eff
_MIN_ANNULUS_PX = 50    # min valid pixels for a variance estimate
_GAUSS_HALF_LIGHT = 1.1774  # half-light radius of a unit Gaussian, in sigma

# ---- LEG C: correlated-coadd-noise variance inflation (task #20) ----------
# Measurement-window / robustness choices for measure_correlated_noise_factor.
# These select WHICH pixels/lags enter the estimate; NONE scales the value
# (the magnitude is measured from the coadd's own sky -- zero tuned constants).
_FCORR_CLIP_NSIGMA = 3.0     # robust sky sigma-clip (reject source pixels)
_FCORR_LOW_NSIGMA = 5.0      # asymmetric low cut (deep negatives = artifacts)
_FCORR_DILATE = 2            # px dilation of the source mask (catch PSF wings)
_FCORR_MAX_CROP = 4096       # bound the FFT to a central crop (mem / time)
_FCORR_MIN_SKY_FRAC = 0.20   # need this fraction of clean sky, else -> 1.0
# Whether a measured F_corr reflects the (recoverable) coadd-RESAMPLING
# correlation or (spurious) crowded-field CONFUSION is decided from the ACF
# SHAPE, not its magnitude (see _acf_shape_is_resampling: the lag-2 decay
# ratio q2 vs the PSF-autocorrelation prediction). A raw magnitude threshold
# (the retired >15 cut) let the T08285 bulge's confusion-shaped ACF (F_corr
# 9-14) through; the shape test (empirical q2: sparse UDS 0.18-0.47 kept,
# bulge 0.61-0.95 -> 1.0) is parameter-free (only the 0.5 midpoint) and
# correctly rejects the bulge -> F_corr=1.0.

# ---- LEG C (crowded-robust): residual-based F_corr (task #20 part A) -------
# On a CROWDED field the raw-sky ACF above nulls to 1.0 (its confusion SHAPE
# is rejected by the ACF-shape gate because unresolved-source structure
# dominates the "source-free" sky). But the resampling correlation is a
# property of the COADD and is FIELD-UNIFORM, so it is recoverable from the
# model-subtracted RESIDUAL map in the LEAST-crowded regions (where the
# residual is consistent with noise). These select WHICH tiles/pixels enter
# the estimate; none scales the value.
_FCORR_RES_TILE = 256        # px side of the cleanliness tiles
_FCORR_RES_KEEP_Q = 0.25     # keep the cleanest quartile of tiles (lowest
                             #   residual structure) -- confusion-robust
_FCORR_RES_MIN_TILES = 8     # need at least this many kept tiles
_FCORR_RES_MIN_CLEAN = 20000  # and this many clean residual pixels


def _boxsum(a, r: int):
    """Sum over the (2r+1)^2 box around each pixel (edge-truncated)."""
    import jax.numpy as jnp
    H, W = a.shape
    c = jnp.cumsum(jnp.cumsum(a, axis=0), axis=1)
    c = jnp.pad(c, ((1, 0), (1, 0)))
    y = jnp.arange(H)
    x = jnp.arange(W)
    y0 = jnp.clip(y - r, 0, H)
    y1 = jnp.clip(y + r + 1, 0, H)
    x0 = jnp.clip(x - r, 0, W)
    x1 = jnp.clip(x + r + 1, 0, W)
    return (c[y1][:, x1] - c[y1][:, x0] - c[y0][:, x1] + c[y0][:, x0])


def _annulus_mean(a, valid_f, r_in: int, r_out: int):
    """Mean of ``a`` over valid pixels in the square annulus [r_in, r_out]."""
    import jax.numpy as jnp
    av = a * valid_f
    n = _boxsum(valid_f, r_out) - _boxsum(valid_f, r_in)
    s = _boxsum(av, r_out) - _boxsum(av, r_in)
    return jnp.where(n >= _MIN_ANNULUS_PX, s / jnp.maximum(n, 1.0), jnp.nan), n


def _corr_same(a, k):
    """2-D cross-correlation, mode='same' (via fftconvolve with the
    flipped kernel)."""
    from jax.scipy.signal import fftconvolve
    return fftconvolve(a, k[::-1, ::-1], mode="same")


def _radial_profile(arr2d: np.ndarray, L: int) -> np.ndarray:
    """Azimuthal mean of a ``(2L+1, 2L+1)`` lag map (peak at ``[L, L]``,
    i.e. zero lag) as a function of integer lag radius ``k = 0 .. L``.
    Returns ``prof`` with ``prof[k]`` the mean over the ring
    ``round(sqrt(dy^2+dx^2)) == k`` (NaN for empty rings)."""
    a = np.asarray(arr2d, dtype=np.float64)
    yy, xx = np.mgrid[-L:L + 1, -L:L + 1]
    rr = np.rint(np.sqrt(yy * yy + xx * xx)).astype(int)
    prof = np.full(L + 1, np.nan)
    for k in range(L + 1):
        m = (rr == k)
        if m.any():
            vals = a[m]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                prof[k] = float(vals.mean())
    return prof


def _acf_shape_is_resampling(
    rho2d: np.ndarray, w2d: np.ndarray, L: int,
) -> "tuple[bool, dict]":
    """Decide -- from the ACF SHAPE alone (no magnitude threshold) -- whether
    a measured 2-D noise autocorrelation ``rho`` is consistent with the
    compact, sub-PSF coadd-RESAMPLING kernel (accept the measured F_corr) or
    with PSF-scale CONFUSION structure (reject -> F_corr = 1.0).

    Physics (zero tuned constants): residual/sky confusion is unmodelled
    PSF-convolved source light, so its autocorrelation has EXACTLY the
    PSF-AUTOCORRELATION shape ``w`` (a point source convolved with the PSF,
    summed over sources -> a field of PSF-shaped bumps whose ACF is the PSF
    autocorrelation) -- ``rho_r(k)/rho_r(1) = w_r(k)/w_r(1)`` at every lag.
    The SWarp coadd-resampling correlation acts on the output pixel grid at
    the nearest-neighbour scale, so it is COMPACT: essentially spent by lag 1
    and largely gone by lag 2, i.e. strictly NARROWER than ``w``.

    So we test the ACF decay at the FIRST lag beyond the nearest-neighbour
    core (lag 2), relative to the confusion (PSF-autocorr) prediction::

        q2 = [rho_r(2)/rho_r(1)] / [w_r(2)/w_r(1)]
        accept (resampling) iff q2 < 0.5

    ``q2`` -> 1 when the ACF tracks the PSF autocorrelation at lag 2
    (confusion) and -> 0 when the ACF has collapsed by lag 2 (resampling).
    Empirically (raw-sky ACFs): sparse UDS coadds give q2 = 0.18-0.47 (real
    resampling, F_corr 3.3-4.8 kept) and the T08285 b=-5 bulge gives q2 =
    0.61-0.95 (confusion, all bands -> F_corr=1.0) -- a clean gap around the
    0.5 midpoint. Lag 2 is used (not a wider window) because the large-lag
    ACF carries a flat non-PSF plateau -- large-scale sky/background residue,
    NOT confusion -- that contaminates any core-averaged or projected
    statistic; the lag-1->2 slope is where resampling and confusion diverge
    most sharply and is very well determined (millions of pixel pairs). The
    only numeric constant is the 0.5 MIDPOINT (the ACF has decayed to less
    than half the confusion prediction -- a definitional midpoint like the
    half-maximum, not a dataset-tuned magnitude); the probe lag and both decay
    ratios are measured from the coadd PSF and the data. This replaces the
    retired raw-magnitude ``F_corr > 15`` rejection, which passed the bulge's
    confusion-shaped ACF (values 9-14). Assumes a well-sampled PSF (7DT: FWHM
    2-4 px, so w_r(2)/w_r(1) ~ 0.8); the ratio self-normalises for the PSF
    sampling.

    Returns ``(is_resampling, diag)``; on any degenerate/untestable case it
    returns ``True`` (keep the measurement -- the value is then clamped to
    ``>= 1`` and, off a crowded field, is the honest resampling factor)."""
    diag: dict = {"q2": np.nan, "acf_shape": None}
    if L < 2:
        diag["acf_shape"] = "window-too-small"
        return True, diag
    rho_r = _radial_profile(rho2d, L)
    w_r = _radial_profile(w2d, L)
    r1, w1, r2, w2 = rho_r[1], w_r[1], rho_r[2], w_r[2]
    if not (np.isfinite(r1) and np.isfinite(w1) and w1 > 0
            and np.isfinite(r2) and np.isfinite(w2) and w2 > 0):
        diag["acf_shape"] = "degenerate"
        return True, diag
    if not (r1 > 0):
        # No positive short-lag correlation -> F_corr ~ 1 regardless; keep
        # (clamped to >= 1 downstream).
        diag["acf_shape"] = "no-short-lag-corr"
        return True, diag
    # Lag-2 decay ratio vs the PSF-autocorrelation (confusion) prediction.
    q2 = float((r2 / r1) / (w2 / w1))
    diag["q2"] = q2
    is_resampling = bool(np.isfinite(q2) and q2 < 0.5)
    diag["acf_shape"] = "resampling" if is_resampling else "confusion"
    return is_resampling, diag


# ---- PERF #13: jitted per-band confusion numerics -------------------------
# The #27 flux-confusion measurement (the matched-filter fftconvolves, the
# annulus box-sum moments, and the extended-template R factor) used to run
# EAGER, band-SERIAL, with a device->host sync EVERY band -- both
# ``float(jnp.sum(phi*phi))`` (for the annulus radii) and
# ``np.asarray(excess[yi, xi])`` (the sampled column) forced the GPU stream
# to drain before the next band was even dispatched, so the GPUs sat at
# 0-13% util for ~52 s/grid (memory gpu-mode-audit-2026-07-07). This kernel
# folds the whole per-band numerical body into ONE jitted program that
# returns the per-band ``sigma_conf`` as a TRACED ``(n_sources,)`` array --
# no host read inside. The driver loop below dispatches all bands async and
# does a SINGLE host transfer after the last band. Bands are INDEPENDENT
# (each column depends only on its own residual/PSF plus the SHARED source
# geometry -- no cross-band term), so the async loop pipelines on the GPU.
#
# vmap vs scan vs loop-over-jit: bands being independent makes all three
# semantically valid, but a jax.vmap over bands would materialize every
# band's grid-sized fftconvolve working set at once (up to ~46 bands x a
# ~5k x 5k grid quadrant) and OOM the 24 GB card; a lax.scan would need the
# non-pytree PsfPolynomialJax/PsfPerCellLookupJax stacked+reconstructed per
# band for the (PSF-type-polymorphic) render and buys nothing over a reused
# jit here. A Python loop over ONE cached @jit kernel gives scan's per-band
# peak memory + async pipelining with none of that friction.


def _confusion_band_numeric(data, model, pix_good, invvar, phi, xi, yi,
                            s_gal2, no_fit):
    """JITTED single-band #27 confusion numerics (PERF #13).

    ``phi`` is the band's already-unit-sum central PSF stamp; ``xi``/``yi``
    the rounded grid-local source pixels; ``s_gal2`` the per-source
    Gaussian-equivalent template variance; ``no_fit`` the no-position mask.
    Returns this band's ``sigma_conf``, shape ``(n_sources,)`` float32, NaN
    where the annulus is invalid or the source has no fitted position. Pure
    JAX -- NO device->host sync (that is the whole point). Same math /
    dtypes / geometry as the retired eager body: sumsq summed in float32
    then promoted to float64 (matching the old ``float(jnp.sum(phi*phi))``),
    annulus radii ``ceil({2,6} x FWHM_eff)``, MF/Vstat/annulus moments in
    float64 (cancellation guard), occupancy/pixel-count guards, and the
    analytic R = 2 s_g^4 / ((s_g^2 + s_p^2) s_p^2) kernel factor."""
    import jax.numpy as jnp

    # PSF Gaussian-equivalent scale. sumsq is the float32 reduction (matching
    # the eager ``float(jnp.sum(phi*phi))``) promoted to float64 for the scale
    # and integer annulus-radius algebra.
    sumsq = jnp.sum(phi * phi)
    sumsq64 = sumsq.astype(jnp.float64)
    sigma_p2 = 1.0 / (4.0 * np.pi * sumsq64)          # Gaussian-equiv var
    fwhm_eff = 2.355 * jnp.sqrt(sigma_p2)
    r_in = jnp.ceil(_R_IN_FWHM * fwhm_eff).astype(jnp.int32)
    r_out = jnp.ceil(_R_OUT_FWHM * fwhm_eff).astype(jnp.int32)

    gf = pix_good.astype(jnp.float32)
    resid = jnp.where(pix_good, data - model, 0.0)
    var_pix = jnp.where(
        pix_good & (invvar > 0), 1.0 / jnp.maximum(invvar, 1e-30), 0.0)

    # --- matched-filter maps (the 3 grid-sized fftconvolves) -----------
    num = _corr_same(resid, phi)
    den = _corr_same(gf, phi * phi)
    vnum = _corr_same(var_pix, phi * phi)
    valid = (den > _OCC_MIN * sumsq) & jnp.isfinite(num)
    den_safe = jnp.maximum(den, 1e-30)
    MF = jnp.where(valid, num / den_safe, 0.0).astype(jnp.float64)
    Vstat = jnp.where(valid, vnum / (den_safe * den_safe), 0.0).astype(jnp.float64)
    valid_f = valid.astype(jnp.float64)

    # --- local annulus moments (float64) -------------------------------
    m1, _ = _annulus_mean(MF, valid_f, r_in, r_out)
    m2, _ = _annulus_mean(MF * MF, valid_f, r_in, r_out)
    vsl, _ = _annulus_mean(Vstat, valid_f, r_in, r_out)
    excess = jnp.maximum(0.0, (m2 - m1 * m1) - vsl)   # NaN-propagating

    # --- sample at source positions + extended-template kernel factor --
    exc_src = excess[yi, xi]
    s_gi2 = sigma_p2 + s_gal2
    R = 2.0 * s_gi2 ** 2 / ((s_gi2 + sigma_p2) * sigma_p2)
    sig = jnp.sqrt(R * exc_src)
    sig = jnp.where(no_fit, jnp.nan, sig)
    return sig.astype(jnp.float32)


_CONFUSION_KERNEL = None


def _get_confusion_kernel():
    """Lazily build + cache the jitted per-band confusion kernel so it
    compiles once and is reused across bands and grids (shape-polymorphic
    recompiles only when a grid's band/PSF/source dims change). The lazy
    jax import keeps this module importable without JAX."""
    global _CONFUSION_KERNEL
    if _CONFUSION_KERNEL is None:
        import jax
        _CONFUSION_KERNEL = jax.jit(_confusion_band_numeric)
    return _CONFUSION_KERNEL


def confusion_call_bytes(
    band_h: int, band_w: int, psf_h: int, psf_w: int, n_sources: int,
) -> "int | None":
    """XLA's report (``temp + argument + output``) for ONE call of the jitted
    confusion kernel at these shapes -- the G1 grid-peak probe's post-solve
    term (``pipeline._measured_grid_peak_mib``).

    The abstract argument dtypes MIRROR the call in
    :func:`compute_grid_confusion` (data/model/invvar f32, pix_good bool,
    phi f32, xi/yi i32, s_gal2 **f32**, no_fit bool) and live HERE, next to
    both the kernel and its call site, so the probe cannot drift from the
    real call silently. ``s_gal2`` is float32 at the call site even under
    x64: ``logre`` is cast f32 and numpy-2 weak promotion keeps
    ``(re_pix / const) ** 2`` f32 (verified empirically 2026-08-19 -- an
    f64 probe arg compiles a DIFFERENT executable the worker never reuses;
    the kernel upcasts to f64 internally either way). Lowered under
    ``jax.enable_x64()`` to match the worker runtime (the kernel's MF/Vstat
    planes are float64 by design). Compiled with PRODUCTION options so the
    executable lands in the persistent JIT cache for the workers. None on
    any failure (the caller keeps the S1-only composition)."""
    try:
        import jax
        import jax.numpy as jnp

        fn = _get_confusion_kernel()
        with jax.enable_x64():
            S = jax.ShapeDtypeStruct
            f32 = jnp.float32
            i32, b1 = jnp.int32, jnp.bool_
            args = (
                S((band_h, band_w), f32),   # data
                S((band_h, band_w), f32),   # model
                S((band_h, band_w), b1),    # pix_good
                S((band_h, band_w), f32),   # invvar
                S((psf_h, psf_w), f32),     # phi (unit-sum central stamp)
                S((n_sources,), i32),       # xi
                S((n_sources,), i32),       # yi
                S((n_sources,), f32),       # s_gal2 (f32 at the call site)
                S((n_sources,), b1),        # no_fit
            )
            comp = fn.lower(*args).compile()
            m = comp.memory_analysis()
        if m is None:
            return None
        return int(m.temp_size_in_bytes + m.argument_size_in_bytes
                   + m.output_size_in_bytes)
    except Exception:  # noqa: BLE001
        return None


# ---- band-parallel residual F_corr (T2) ----------------------------------
# The residual F_corr leg of the band loop below is pure numpy/scipy on the
# host and takes ~2 s per band at production grid sizes -- while it runs, this
# worker's GPU is idle and its share of the cores is idle too. It is
# band-independent and deterministic (no RNG, no shared state), so it runs on a
# thread pool; results are stored BY BAND INDEX, never by completion order.
#
# THREADS, not processes: the hot calls (np.fft.rfft2/irfft2,
# scipy.signal.fftconvolve, scipy.ndimage.binary_dilation) release the GIL, and
# a process pool would have to pickle a 64 MiB residual per band. Measured on a
# real T08676 grid (22 bands, 3330x2530): serial 42.8 s, 8 threads 7.6 s
# (5.6x), 8 processes 10.0 s (4.3x); every mode bit-identical to serial.

_ALLOC_BASELINE_UNREADABLE = "allocator not readable"

_resid_bpp_lock = threading.Lock()
_RESID_BYTES_PER_PIXEL: "float | None" = None
_RESID_BPP_WHY: str = "not measured yet"
_RESID_POOL: "ThreadPoolExecutor | None" = None
_RESID_POOL_SIZE: int = 0
_RESID_PLAN_LOGGED: int = -1


def _c_alloc_in_use_bytes() -> int:
    """Bytes currently handed out by the C allocator (glibc ``mallinfo2``).

    Used to size memory-bounded worker pools by MEASURING what one unit of
    work actually holds, instead of hand-counting arrays in the source.
    Returns 0 when unreadable (non-glibc / restricted), which callers must
    treat as "cannot measure" -- NOT as "zero bytes".

    Deliberately not RSS: right after a fit the allocator reuses its arenas,
    so an RSS delta reads ~0 for a genuinely multi-hundred-MB working set.
    """
    try:
        class _MI2(ctypes.Structure):
            _fields_ = [(n, ctypes.c_size_t) for n in (
                "arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks",
                "fsmblks", "uordblks", "fordblks", "keepcost")]

        libc = ctypes.CDLL("libc.so.6")
        libc.mallinfo2.restype = _MI2
        mi = libc.mallinfo2()
        return int(mi.uordblks) + int(mi.hblkhd)
    except Exception:                                          # noqa: BLE001
        return 0


def _usable_physical_cores() -> "tuple[int, str]":
    """PHYSICAL cores this process may run on, not hyperthreads.

    The band pass is FFT-bound on grid-sized f64 arrays, so SMT siblings share
    the very resources it is limited by (execution units, L1/L2, the memory
    pipeline) and buy essentially nothing. MEASURED on a 2x16-core Xeon with
    SMT (64 logical): 4 concurrent grid workers running this pass took 10.54 s
    aggregate at 8 bands in flight each and 10.58 s at 15 -- identical time for
    nearly twice the resident memory (~0.75 GiB per extra in-flight band, per
    worker). 32 physical / 4 workers = 8 is exactly where it flattened.

    Counts distinct (physical id, core id) pairs among the CPUs in this
    process's affinity mask, so cpusets and containers are respected. Falls
    back to the logical count when the topology cannot be read -- that is the
    old behaviour, i.e. safe but possibly generous.
    """
    try:
        allowed = os.sched_getaffinity(0)
    except (AttributeError, OSError):                          # noqa: BLE001
        allowed = None
    try:
        cores, cpu, phys, core = set(), None, None, None
        with open("/proc/cpuinfo", "r") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                k, v = k.strip(), v.strip()
                if k == "processor":
                    cpu = int(v)
                elif k == "physical id":
                    phys = v
                elif k == "core id":
                    core = v
                elif not line.strip() and cpu is not None:
                    if allowed is None or cpu in allowed:
                        cores.add((phys, core))
                    cpu = phys = core = None
        if cpu is not None and (allowed is None or cpu in allowed):
            cores.add((phys, core))
        cores.discard((None, None))
        if cores:
            return len(cores), f"{len(cores)} physical core(s)"
    except Exception:                                          # noqa: BLE001
        pass
    n = len(allowed) if allowed else (os.cpu_count() or 1)
    return max(1, n), f"{n} logical cpu(s) (topology unreadable)"


def _host_mem_available_bytes() -> int:
    """MemAvailable from /proc/meminfo (0 if unreadable). Already excludes
    this process's own live arrays, so no headroom factor is needed."""
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:                                          # noqa: BLE001
        pass
    return 0


class _AllocSampler:
    """Peak C-allocator in-use bytes over a code region, sampled from a
    daemon thread. ``delta`` is the peak rise above the entry baseline."""

    def __init__(self, period_s: float = 0.002):
        self.period = period_s
        self.base = _c_alloc_in_use_bytes()
        self.peak = self.base
        self._stop = threading.Event()
        self._th: "threading.Thread | None" = None

    @property
    def usable(self) -> bool:
        return self.base > 0

    def __enter__(self):
        if self.usable:
            def _run():
                while not self._stop.is_set():
                    self.peak = max(self.peak, _c_alloc_in_use_bytes())
                    self._stop.wait(self.period)
            self._th = threading.Thread(target=_run, daemon=True)
            self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=1.0)
        return False

    @property
    def delta(self) -> int:
        return max(0, self.peak - self.base)


def _resid_band_workers(n_bands: int, n_pixels: int, n_slots: int
                        ) -> "tuple[int, str]":
    """In-flight bands for the residual F_corr pass. Measured, not tuned.

    Bounded by four things, every one of them read at the call site:

      (a) work    -- never more in flight than there are bands;
      (b) cores   -- this process's share of the PHYSICAL cores
          (:func:`_usable_physical_cores` ``// n_slots``), because ``n_slots``
          sibling grid workers are running the same code at the same time on
          the same machine, and because SMT siblings do not help FFT-bound work
          (measured: identical aggregate time at 8 and 15 in flight, for nearly
          twice the memory);
      (c) memory  -- this process's share of MemAvailable, divided by what ONE
          in-flight band actually holds: the MEASURED bytes per residual pixel
          (:func:`_resid_bytes_per_pixel`) times this grid's pixel count;
      (d) measurability -- if the allocator cannot be read we cannot bound (c),
          so we stay SERIAL (today's behaviour) rather than guess a number.

    Same probe-and-divide shape as ``pipeline._choose_workers_per_gpu`` and
    ``pipeline._fcorr_band_workers``: nothing here is fitted to 7DT data or to
    this machine, so it re-measures itself anywhere.
    """
    if n_bands <= 1 or n_pixels <= 0:
        return 1, f"{n_bands} band(s)"
    bpp, why = _resid_bytes_per_pixel()
    if bpp is None:
        return 1, f"serial ({why})"
    per_band = max(1, int(bpp * n_pixels))
    avail = _host_mem_available_bytes()
    if avail <= 0:
        return 1, "serial (MemAvailable unreadable)"
    slots = max(1, int(n_slots))
    by_mem = max(1, (avail // slots) // per_band)
    n_cores, core_why = _usable_physical_cores()
    by_cpu = max(1, n_cores // slots)
    n = int(max(1, min(n_bands, by_cpu, by_mem)))
    return n, (f"{n_bands} band(s), {core_why}/{slots} grid "
               f"slot(s) -> {by_cpu}, {avail / 2**30:.0f} GiB avail/{slots} / "
               f"{per_band / 2**30:.2f} GiB per band ({bpp:.0f} B/pixel, "
               f"{why}) -> {by_mem} -> {n} in flight")


def _resid_bytes_per_pixel() -> "tuple[float | None, str]":
    """Host bytes ONE in-flight residual band holds, per residual pixel.

    ``None`` means "not measured yet / unmeasurable"; the caller then stays
    serial. The value is MEASURED IN SITU by
    :func:`_measure_resid_bytes_per_pixel` on the first band this process ever
    processes -- the real estimator, on the real residual, at the real size --
    so there is no synthetic stand-in to drift from the production path and no
    hand-counted array tally to go stale when the estimator allocates one more
    plane. Cached for the life of the process (the working set is linear in
    pixels, so one measurement sizes every later grid).
    """
    with _resid_bpp_lock:
        return _RESID_BYTES_PER_PIXEL, _RESID_BPP_WHY


def _measure_resid_bytes_per_pixel(delta_bytes: int, n_pixels: int,
                                   usable: bool) -> None:
    """Record the in-situ measurement from the first band (see above)."""
    global _RESID_BYTES_PER_PIXEL, _RESID_BPP_WHY
    with _resid_bpp_lock:
        if _RESID_BYTES_PER_PIXEL is not None:
            return
        if not usable:
            _RESID_BPP_WHY = _ALLOC_BASELINE_UNREADABLE
            return
        if delta_bytes <= 0 or n_pixels <= 0:
            _RESID_BPP_WHY = "in-situ probe returned no allocation"
            return
        _RESID_BYTES_PER_PIXEL = float(delta_bytes) / float(n_pixels)
        _RESID_BPP_WHY = (
            f"measured in situ on band 1 of this process "
            f"({delta_bytes / 2**20:.0f} MiB / {n_pixels / 1e6:.2f} Mpix)")


def _fcorr_resid_band(bi: int, resid_np, good_np, phi_np,
                      sem: "threading.Semaphore | None"):
    """One band's residual F_corr, as stored in ``extra['fcorr_resid'][bi]``.

    Identical to the historical inline body (including "a failed band is
    ``None`` and never poisons its neighbours"); it is a function only so a
    pool thread can run it. ``sem``, when given, bounds how many bands are in
    flight and is released here whatever happens.
    """
    try:
        F_res, dgr = measure_correlated_noise_factor_residual(
            resid_np, good_np, phi_np, band=str(bi), log_fn=None)
        return (float(F_res), int(dgr.get("n_clean", 0)), dgr.get("fallback"))
    except Exception:                                          # noqa: BLE001
        return None
    finally:
        if sem is not None:
            sem.release()


def _note_resid_plan(n_flight: int) -> bool:
    """True the first time this process settles on a given in-flight count, so
    a 148-grid tile logs the band-pass plan once instead of 148 times."""
    global _RESID_PLAN_LOGGED
    with _resid_bpp_lock:
        if n_flight == _RESID_PLAN_LOGGED:
            return False
        _RESID_PLAN_LOGGED = n_flight
        return True


def _resid_pool(n_workers: int) -> "ThreadPoolExecutor | None":
    """Process-wide thread pool for the residual F_corr bands, created once and
    reused across grids (a grid worker fits many grids). It only ever GROWS:
    idle threads cost nothing, and the real in-flight bound is the per-grid
    semaphore in :func:`compute_grid_confusion`, which is re-derived per grid
    because grid sizes differ."""
    global _RESID_POOL, _RESID_POOL_SIZE
    if n_workers <= 1:
        return None
    with _resid_bpp_lock:
        if _RESID_POOL is None or n_workers > _RESID_POOL_SIZE:
            old = _RESID_POOL
            _RESID_POOL = ThreadPoolExecutor(
                max_workers=n_workers, thread_name_prefix="fcorr_resid")
            _RESID_POOL_SIZE = n_workers
            if old is not None:
                old.shutdown(wait=False)
        return _RESID_POOL


def compute_grid_confusion(
    multi_inputs: Any,
    result: Any,
    *,
    wing_peeling: bool = False,
    wing_peel_tier: int = 221,
    wing_peel_max_sources: int = 64,
    want_confusion: bool = True,
    want_fcorr_resid: bool = False,
    n_grid_slots: int = 1,
    log_fn=None,
) -> "tuple[np.ndarray | None, dict]":
    """Post-solve read-only error measurements for one grid, from the
    converged fit state and its residual map ``r = data - full model``.

    Parameters mirror the wing-peeling knobs of :func:`solve_grid_3stage`
    so the model image reproduces what the final linear solve saw
    (native stamps + tier wings for the "big" extended sources).

    Computes up to two things from the SAME per-band residual (one render):

    - ``want_confusion`` (task #27): per-source per-band ``sigma_conf``
      (annulus matched-filter excess variance). Returned as ``sigma_conf``.
      This is the FLUX confusion term and feeds FLUXERR_CONF/FLUXERR_TOT.
    - ``want_fcorr_resid`` (LEG C part A, task #20): per-band crowded-robust
      correlated-noise factor from the residual's least-crowded tiles
      (:func:`measure_correlated_noise_factor_residual`), used ONLY for the
      GEOMETRIC (pos/shape/n) error columns. Returned in
      ``extra['fcorr_resid']`` as a list over bands of ``(F_corr, n_clean)``
      (or ``None`` where unavailable).

    NOTE (task #20 re-scope, 2026-07-08): the per-source LOCAL flux
    ``chi2_nu`` (LEG D flux) was REMOVED. Validation on the T08285 bulge
    injections showed the flexible per-source fit ABSORBS crowded confusion
    (local chi2_nu median ~1.06), so it did not capture the flux
    under-coverage that task #27's annulus ``sigma_conf`` does; the flux
    error budget is therefore #27-only. LEG C (F_corr) + LEG D (chi2_nu) are
    retained only for the geometric errors (which #27 does not touch).

    ``n_grid_slots`` is how many sibling grid workers share this machine right
    now; it divides the core/memory budget of the band-parallel residual pass
    (T2) so the workers cannot collectively oversubscribe. 1 = this process has
    the machine to itself.

    Returns ``(sigma_conf, extra)``. ``sigma_conf`` is ``(n_sources,
    n_bands) float32`` (NaN where unavailable) or ``None`` when
    ``want_confusion`` is False. Read-only on ``multi_inputs`` / ``result``.
    """
    import jax.numpy as jnp
    from .forced_phot_sparse import render_source_stamps, stamp_top_left
    from .source_model import render_star_stamps as _render_star_stamps
    from .psf_polynomial import evaluate_psf_at
    from .source_model import SOURCE_STAR
    from .solver_grid import (
        _compute_wing_images,
        _pad_psf_jax,
        _scatter_stamps_to_band,
        _select_wing_sources,
    )

    t0 = time.perf_counter()
    n_bands = int(multi_inputs.n_bands)
    n_sources = int(np.asarray(result.pos_x).shape[0])
    sigma_conf = None   # built once from the stacked per-band kernel outputs
    fcorr_resid: list = [None] * n_bands if want_fcorr_resid else []
    extra: dict = {"fcorr_resid": fcorr_resid}

    pos_x = np.asarray(result.pos_x, dtype=np.float32)
    pos_y = np.asarray(result.pos_y, dtype=np.float32)
    logre = np.asarray(result.logre, dtype=np.float32)
    ee1 = np.asarray(result.ee1, dtype=np.float32)
    ee2 = np.asarray(result.ee2, dtype=np.float32)
    sersic_n = (
        np.asarray(result.sersic_n_fit, dtype=np.float32)
        if getattr(result, "sersic_n_fit", None) is not None
        else np.ones(n_sources, dtype=np.float32)
    )
    fluxes = np.asarray(result.fluxes, dtype=np.float32)      # (n, n_bands)
    fluxes = np.where(np.isfinite(fluxes), fluxes, 0.0).astype(np.float32)
    source_type = (
        np.asarray(multi_inputs.source_type, dtype=np.int32)
        if multi_inputs.source_type is not None
        else np.zeros(n_sources, dtype=np.int32)
    )

    pos_x_jax = jnp.asarray(pos_x)
    pos_y_jax = jnp.asarray(pos_y)
    st_jax = jnp.asarray(source_type)
    is_ext = source_type != SOURCE_STAR
    logre_jax = jnp.asarray(np.where(is_ext, logre, 0.0), dtype=jnp.float32)
    ee1_jax = jnp.asarray(np.where(is_ext, ee1, 1.0), dtype=jnp.float32)
    ee2_jax = jnp.asarray(np.where(is_ext, ee2, 0.0), dtype=jnp.float32)
    sn_jax = jnp.asarray(sersic_n)

    # Big-source wing images (tier - native), same selection as the solve.
    wings = None
    if wing_peeling:
        big_idx = _select_wing_sources(
            source_type, logre, sersic_n,
            int(multi_inputs.bands[0].psf_jax.W_psf),
            int(wing_peel_max_sources),
        )
        if big_idx.size > 0:
            psf_native_jaxes = [b.psf_jax for b in multi_inputs.bands]
            psf_wing_jaxes = [
                _pad_psf_jax(p, int(wing_peel_tier)) for p in psf_native_jaxes
            ]
            wings = _compute_wing_images(
                multi_inputs, psf_native_jaxes, psf_wing_jaxes, big_idx,
                pos_x, pos_y, logre, ee1, ee2, sersic_n, fluxes,
            )

    # Extended-template kernel factor R (per source; band-dependent via s_p).
    re_pix = np.where(
        is_ext & np.isfinite(logre), np.exp(np.clip(logre, -5.0, 5.0)), 0.0
    )
    s_gal2 = (re_pix / _GAUSS_HALF_LIGHT) ** 2                   # (n,)
    _all_star_grid = not bool(np.any(is_ext))                    # T6/G2

    # PERF #13: precompute the band-shared kernel inputs ONCE, build the jitted
    # kernel ONCE, and collect the per-band TRACED confusion columns without
    # any per-band host read -> the bands' GPU work pipelines instead of the
    # old sync-per-band stall.
    _kernel = _get_confusion_kernel() if want_confusion else None
    s_gal2_jax = jnp.asarray(s_gal2) if want_confusion else None
    no_fit = ~np.isfinite(pos_x) | ~np.isfinite(pos_y)   # band-independent
    no_fit_jax = jnp.asarray(no_fit) if want_confusion else None
    conf_cols: list = []

    # --- residual F_corr band pass (T2) -----------------------------------
    # Sized from a measurement, then run in flight while the loop keeps the
    # GPU fed with the next band's render. Serial when we cannot bound it.
    resid_pool: "ThreadPoolExecutor | None" = None
    resid_sem: "threading.Semaphore | None" = None
    resid_futs: list = []
    # The first band this PROCESS ever measures runs inline under an allocator
    # sampler; its footprint sizes every later band, here and in later grids.
    resid_measure_pending = want_fcorr_resid and (
        _resid_bytes_per_pixel()[0] is None)

    def _pull(inp_, model_, phi_):
        """Host copies of one band's residual inputs (the device->host sync).
        Kept on the calling thread so GPU op order is unchanged."""
        return (np.asarray(inp_.data - model_, dtype=np.float64),
                np.asarray(inp_.pix_good, dtype=bool),
                np.asarray(phi_, dtype=np.float64))

    def _start_pool(n_pixels: int):
        nonlocal resid_pool, resid_sem
        n_flight, why = _resid_band_workers(n_bands, n_pixels, n_grid_slots)
        resid_pool = _resid_pool(n_flight)
        resid_sem = (threading.Semaphore(n_flight)
                     if resid_pool is not None else None)
        if _note_resid_plan(n_flight) and log_fn is not None:
            log_fn(f"error model: residual F_corr band pass: {why}")

    if want_fcorr_resid and not resid_measure_pending:
        _start_pool(max(int(b.band_h) * int(b.band_w)
                        for b in multi_inputs.bands))

    for bi in range(n_bands):
        inp = multi_inputs.bands[bi]
        band_h, band_w = int(inp.band_h), int(inp.band_w)
        S = int(inp.psf_jax.H_psf)

        # --- model image at the converged state (async; no host read) ----
        if _all_star_grid:
            # T6/G2: all-STAR grid -> exact star-only render (bitwise the
            # STAR rows of the full chain; no dead galaxy work). Dispatch on
            # the host-known types, like forced_phot_sparse's S1 path.
            stamps = _render_star_stamps(
                inp.psf_jax, pos_x_jax, pos_y_jax, inp.grid_x0, inp.grid_y0)
        else:
            stamps = render_source_stamps(
                inp.psf_jax, pos_x_jax, pos_y_jax, inp.grid_x0, inp.grid_y0,
                st_jax, logre_jax, ee1_jax, ee2_jax,
                sersic_n=sn_jax, render_backend=multi_inputs.render_backend,
            )
        stamps = stamps * jnp.asarray(fluxes[:, bi])[:, None, None]
        ix_top, iy_top = stamp_top_left(
            pos_x_jax, pos_y_jax, inp.grid_x0, inp.grid_y0, S, S,
        )
        model = _scatter_stamps_to_band(stamps, ix_top, iy_top, band_h, band_w)
        if wings is not None:
            model = model + wings[bi]
        if inp.bg_basis_flat is not None:
            bg_map = (
                inp.bg_basis_flat @ jnp.asarray(result.bg_coefs[bi])
            ).reshape(band_h, band_w)
            model = model + bg_map

        # central unit-sum PSF stamp (traced; the normalisation is a reduction,
        # NOT a host read).
        phi = evaluate_psf_at(
            inp.psf_jax,
            jnp.float32(inp.grid_x0 + 0.5 * band_w),
            jnp.float32(inp.grid_y0 + 0.5 * band_h),
        )
        phi = phi / jnp.sum(phi)

        if want_confusion:
            # Rounded grid-local source pixel (the confusion annulus sample
            # point).
            xi = jnp.clip(
                jnp.round(pos_x_jax - inp.grid_x0).astype(jnp.int32),
                0, band_w - 1)
            yi = jnp.clip(
                jnp.round(pos_y_jax - inp.grid_y0).astype(jnp.int32),
                0, band_h - 1)
            # ONE jitted kernel -> a TRACED (n_sources,) column. sumsq / the
            # annulus radii / the excess sample all live inside, so there is
            # NO per-band device->host sync (PERF #13); the single transfer
            # happens after the loop.
            conf_cols.append(_kernel(
                inp.data, model, inp.pix_good, inp.invvar, phi, xi, yi,
                s_gal2_jax, no_fit_jax,
            ))

        if want_fcorr_resid:
            # LEG C part A (GEOMETRIC errors only): crowded-robust
            # correlated-noise factor from THIS band's residual (least-crowded
            # tiles). Pure numpy/scipy on the host, band-independent and
            # deterministic -> dispatched to the band pool (T2) so this band's
            # ~2 s of FFTs overlaps the NEXT band's GPU render instead of
            # stalling both. Results are stored by band INDEX. Per-band,
            # per-grid; the parent combines across grids (clean-pixel weighted).
            if resid_measure_pending:
                resid_measure_pending = False
                with _AllocSampler() as smp:
                    args = _pull(inp, model, phi)
                    fcorr_resid[bi] = _fcorr_resid_band(bi, *args, None)
                    n_pix = int(args[0].size)
                    del args
                _measure_resid_bytes_per_pixel(smp.delta, n_pix, smp.usable)
                _start_pool(n_pix)
            elif resid_pool is not None:
                # acquire BEFORE the pull: the bound covers the residual array
                # itself, not just the estimator's working set.
                resid_sem.acquire()
                resid_futs.append((bi, resid_pool.submit(
                    _fcorr_resid_band, bi, *_pull(inp, model, phi),
                    resid_sem)))
            else:
                fcorr_resid[bi] = _fcorr_resid_band(
                    bi, *_pull(inp, model, phi), None)

    for bi, fut in resid_futs:
        fcorr_resid[bi] = fut.result()

    if want_confusion:
        # SINGLE host transfer for ALL bands' confusion columns (n_sources,
        # n_bands) -- the one and only device->host sync of the confusion pass.
        with np.errstate(invalid="ignore"):
            sigma_conf = np.asarray(
                jnp.stack(conf_cols, axis=1), dtype=np.float32)

    if log_fn is not None:
        parts = []
        if want_confusion:
            parts.append(f"{int(np.isfinite(sigma_conf).sum())}/"
                         f"{sigma_conf.size} conf")
        if want_fcorr_resid:
            parts.append(
                f"{sum(1 for x in fcorr_resid if x is not None)}/{n_bands} "
                f"F_corr_resid")
        log_fn("error model: " + ", ".join(parts) +
               f" in {time.perf_counter() - t0:.2f}s")
    return sigma_conf, extra


def measure_correlated_noise_factor(
    img_scaled: np.ndarray,
    bad: "np.ndarray | None",
    psf_stamp: np.ndarray,
    *,
    band: str = "",
    log_fn=None,
) -> "tuple[float, dict]":
    """Per-band correlated-coadd-noise variance-inflation factor ``F_corr``
    for a white-weighted PSF-flux (matched-filter) estimator (LEG C, #20).

    WHY: 7DT coadds are SWarp-resampled MEDIAN combines, so neighbouring
    output pixels share input samples and their noise is positively
    correlated. The reported statistical flux error is
    ``sqrt(diag(inv(A^T W A)))`` with the WHITE weight ``W = diag(1/SKYSIG^2)``
    -- it assumes independent pixels, so it under-estimates the noise
    integrated over a PSF stamp by a per-band factor. For a white-weighted
    estimator the exact error is the sandwich
    ``(J^T W J)^-1 (J^T W C W J) (J^T W J)^-1``; with a stationary compact
    resampling correlation ``C = sigma^2 R`` and ``W = I / sigma^2`` this
    collapses EXACTLY to ``F_corr * (J^T W J)^-1`` -- a per-band SCALAR
    variance inflation (the ngmix / LSST coadd-correlation variance rescale).

    FORMULA (measured, zero magnitude-tuned constants)::

        F_corr = sum_d rho(d) * w(d) / w(0),   w = phi (corr) phi

    where ``phi`` is the band's unit-sum central PSF, ``w = phi (corr) phi``
    is the PSF autocorrelation (= [phi^T R phi]/[phi^T phi] expanded over
    lags -- the PSF autocorrelation IS the physical correlation support, so
    there is no arbitrary integration radius), and ``rho(d)`` is the
    normalized 2-D autocorrelation of the coadd's OWN source-free sky. rho
    is estimated by a mask-normalized FFT autocovariance on the
    sigma-clipped, source-masked sky::

        rho(d) = [sum_i x_i x_{i+d}] / [sum_i m_i m_{i+d}] / rho(0)

    with ``x`` the mean-subtracted sky (zero-filled off-mask) and ``m`` the
    clean-sky mask. This reproduces the design's 3 independent UDS
    measurements (m600 ~ 3.15 in variance; sigma ~ 1.77; rho(1px) ~ 0.15).

    Cross-check (returned in ``diag``): an INDEPENDENT empty-aperture
    box-sum variance vs the white expectation, at a box matched to the PSF
    effective area ``N_eff = 1/sum(phi^2)`` --
    ``F_box = Var[box_sum] / (N_box * per_pixel_var)``. On the clean sky
    the two agree to ~15%.

    Robustness (honest, not made-up): too little clean sky
    (< ``_FCORR_MIN_SKY_FRAC``), or an ACF whose SHAPE is confusion- rather
    than resampling-consistent (:func:`_acf_shape_is_resampling` -- the ACF
    is as broad as the PSF autocorrelation, so it is unresolved-source
    structure, not the sub-PSF resampling kernel) -> fall back to
    ``F_corr = 1.0`` (no inflation). The shape test (zero tuned constants)
    REPLACED the retired raw-magnitude ``F_corr > 15`` cut, which let the
    T08285 bulge's confusion-shaped ACF (9-14) through.

    Read-only. Pure numpy / scipy (no GPU, no JAX). Returns
    ``(F_corr, diag)`` where ``F_corr >= 1.0`` and ``diag`` holds the
    cross-check + diagnostics for logging.
    """
    from scipy.signal import fftconvolve
    from scipy.ndimage import binary_dilation

    diag: dict = {"band": band, "F_corr_acf": np.nan, "F_corr_box": np.nan,
                  "sky_frac": np.nan, "rho_1px": np.nan, "n_sky": 0,
                  "acf_shape": None, "acf_q2": np.nan, "fallback": None}

    img = np.asarray(img_scaled)
    H, W = img.shape
    ch = int(min(H, _FCORR_MAX_CROP))
    cw = int(min(W, _FCORR_MAX_CROP))
    y0 = (H - ch) // 2
    x0 = (W - cw) // 2
    sub = np.asarray(img[y0:y0 + ch, x0:x0 + cw], dtype=np.float64)
    if bad is not None:
        badc = np.asarray(bad[y0:y0 + ch, x0:x0 + cw], dtype=bool)
    else:
        badc = ~np.isfinite(sub)
    fin = np.isfinite(sub) & (~badc)

    def _fallback(reason):
        diag["fallback"] = reason
        if log_fn is not None:
            log_fn(f"correlated-noise F_corr[{band}]: {reason} -> F_corr=1.0")
        return 1.0, diag

    if fin.sum() < _FCORR_MIN_SKY_FRAC * fin.size:
        return _fallback("too few finite sky pixels")

    v = sub[fin]
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med)))
    if not np.isfinite(mad) or mad <= 0:
        mad = float(np.std(v))
    if not np.isfinite(mad) or mad <= 0:
        return _fallback("degenerate sky (zero spread)")

    # Source-free sky mask: reject bright (sources) + deep-negative (artifact)
    # outliers, dilate to catch PSF wings.
    src = (~fin) | (sub > med + _FCORR_CLIP_NSIGMA * mad) | \
          (sub < med - _FCORR_LOW_NSIGMA * mad)
    g = ~binary_dilation(src, iterations=_FCORR_DILATE)
    sky_frac = float(g.mean())
    diag["sky_frac"] = sky_frac
    diag["n_sky"] = int(g.sum())
    if g.sum() < _FCORR_MIN_SKY_FRAC * g.size:
        return _fallback(f"too few clean sky pixels (frac={sky_frac:.2f})")

    gf = g.astype(np.float64)
    x = np.where(g, sub - float(sub[g].mean()), 0.0)
    Fx = np.fft.rfft2(x)
    Fm = np.fft.rfft2(gf)
    A = np.fft.irfft2(Fx * np.conj(Fx), s=(ch, cw))     # sum_i x_i x_{i+d}
    K = np.fft.irfft2(Fm * np.conj(Fm), s=(ch, cw))     # valid-pair count(d)
    C = A / np.maximum(K, 1.0)                            # autocovariance(d)
    C0 = float(C[0, 0])
    if not (np.isfinite(C0) and C0 > 0):
        return _fallback("non-positive sky variance")

    phi = np.asarray(psf_stamp, dtype=np.float64)
    psum = float(phi.sum())
    if not (np.isfinite(psum) and psum > 0):
        return _fallback("invalid PSF stamp")
    phi = phi / psum
    L = int(min(phi.shape[0] - 1, ch // 2 - 1, cw // 2 - 1))
    # PSF autocorrelation weight w(d)/w(0) over the +-L window.
    wfull = fftconvolve(phi, phi[::-1, ::-1], mode="full")
    cyw, cxw = wfull.shape[0] // 2, wfull.shape[1] // 2
    w = wfull[cyw - L:cyw + L + 1, cxw - L:cxw + L + 1] / wfull[cyw, cxw]
    # rho(d) over the same window via wrap-around indexing of C.
    dy = np.arange(-L, L + 1)
    dx = np.arange(-L, L + 1)
    rho = C[np.mod(dy[:, None], ch), np.mod(dx[None, :], cw)] / C0
    F_acf = float((rho * w).sum())
    diag["F_corr_acf"] = F_acf
    diag["rho_1px"] = 0.5 * (float(rho[L, L + 1]) + float(rho[L + 1, L]))

    # Independent cross-check: empty-aperture box-sum variance vs white
    # expectation, box matched to the PSF effective area N_eff = 1/sum(phi^2).
    try:
        sumsq = float((phi * phi).sum())
        side = max(2, int(round(np.sqrt(1.0 / max(sumsq, 1e-30)))))
        cs = np.pad(np.cumsum(np.cumsum(x, 0), 1), ((1, 0), (1, 0)))
        ks = np.pad(np.cumsum(np.cumsum(gf, 0), 1), ((1, 0), (1, 0)))
        rng = np.random.default_rng(0)
        n_ap = 20000
        ys = rng.integers(0, ch - side, 4 * n_ap)
        xs = rng.integers(0, cw - side, 4 * n_ap)
        bs = (cs[ys + side, xs + side] - cs[ys, xs + side]
              - cs[ys + side, xs] + cs[ys, xs])
        bc = (ks[ys + side, xs + side] - ks[ys, xs + side]
              - ks[ys + side, xs] + ks[ys, xs])
        full = bc >= side * side - 0.5
        bs = bs[full][:n_ap]
        per_pix_var = float(x[g].var())
        if bs.size >= 50 and per_pix_var > 0:
            diag["F_corr_box"] = float(bs.var() / (side * side * per_pix_var))
        diag["box_side"] = side
    except Exception:
        pass

    if not np.isfinite(F_acf):
        return _fallback("non-finite F_corr")
    # ACF-SHAPE gate (zero tuned constants; replaces the retired F_corr>15
    # magnitude cut): accept only a resampling-consistent (sub-PSF, compact)
    # ACF; a confusion-shaped (PSF-scale-or-broader) ACF -> F_corr = 1.0.
    is_resamp, shp = _acf_shape_is_resampling(rho, w, L)
    diag["acf_shape"] = shp.get("acf_shape")
    diag["acf_q2"] = shp.get("q2")
    if not is_resamp:
        return _fallback(
            f"confusion-shaped ACF (q2={shp.get('q2'):.2f} >= 0.5; "
            f"tracks the PSF autocorrelation at lag 2, not resampling)")
    F_corr = float(max(F_acf, 1.0))   # correlation can only INFLATE variance
    if log_fn is not None:
        log_fn(
            f"correlated-noise F_corr[{band}]={F_corr:.2f} "
            f"(sigma x{np.sqrt(F_corr):.2f}; ACF={F_acf:.2f} "
            f"box={diag['F_corr_box']:.2f} rho1px={diag['rho_1px']:.3f} "
            f"skyfrac={sky_frac:.2f}; shape={diag['acf_shape']})")
    return F_corr, diag


def measure_correlated_noise_factor_residual(
    resid: np.ndarray,
    good: "np.ndarray | None",
    psf_stamp: np.ndarray,
    *,
    band: str = "",
    log_fn=None,
) -> "tuple[float, dict]":
    """CROWDED-ROBUST correlated-coadd-noise F_corr from a model-subtracted
    RESIDUAL map (LEG C part A, task #20).

    The raw-sky estimator :func:`measure_correlated_noise_factor` falls back
    to 1.0 on crowded fields: the "source-free" sky is contaminated by
    unresolved-source structure, so its ACF is confusion-SHAPED (as broad as
    the PSF autocorrelation) and the shape gate rejects it. The
    SWarp-resampling correlation is a property of the COADD and is
    FIELD-UNIFORM, so it CAN survive on the residual ``r = data - full model``:
    the catalogued sources are subtracted, and in the LEAST-crowded regions
    ``r`` IS the correlated coadd noise. This measures the SAME
    PSF-autocorrelation-weighted sky-ACF estimator (identical formula,
    ``F_corr = sum_d rho(d) w(d)/w(0)``, ``w = phi (corr) phi``) but restricts
    the pixels to the cleanest tiles of the residual (lowest residual
    structure), so the (spatially non-uniform) confusion cannot enter while
    the (uniform) resampling correlation does.

    HONEST LIMIT (task #20 re-scope, T08285 b=-5 bulge): on a field with NO
    confusion-free tile (a dense bulge), even the cleanest residual tiles
    carry confusion, so the residual ACF is ALSO confusion-shaped (rho(1px)
    ~0.5 vs the ~0.15 of a clean coadd) -- and the same ACF-SHAPE gate applied
    below (:func:`_acf_shape_is_resampling`) rejects it -> F_corr = 1.0. This
    replaces the retired raw-magnitude ``F_corr > 15`` cut, which let the
    bulge's confusion-shaped ACF (values 9-14) through. The consequence is a
    documented ~x1.85 (resampling) under-coverage of the GEOMETRIC error on
    such a field -- small next to the confusion itself, and honest.

    Tile selection (measurement window, not calibration): split the field
    into ``_FCORR_RES_TILE``-px tiles, score each by its clean-pixel robust
    variance (correlated noise inflates ALL tiles equally, so the ordering
    isolates confusion), and keep the cleanest ``_FCORR_RES_KEEP_Q`` quantile.
    A per-pixel sigma-clip on the residual (plus the ``good`` mask, dilated)
    removes bright residual bumps within a kept tile. rho is then the
    mask-normalized FFT autocovariance over exactly those pixels.

    Read-only, pure numpy/scipy. Returns ``(F_corr, diag)`` with
    ``F_corr >= 1.0``; ``diag['n_clean']`` is the number of clean residual
    pixels used (0 with ``fallback`` set when the measurement is unavailable).
    """
    from scipy.signal import fftconvolve
    from scipy.ndimage import binary_dilation

    diag: dict = {"band": band, "F_corr_acf": np.nan, "rho_1px": np.nan,
                  "n_clean": 0, "n_tiles_kept": 0, "acf_shape": None,
                  "acf_q2": np.nan, "fallback": None}

    r = np.asarray(resid, dtype=np.float64)
    H, W = r.shape
    if good is not None:
        g0 = np.asarray(good, dtype=bool) & np.isfinite(r)
    else:
        g0 = np.isfinite(r)

    def _fallback(reason):
        diag["fallback"] = reason
        if log_fn is not None:
            log_fn(f"correlated-noise F_corr_resid[{band}]: {reason} "
                   f"-> unavailable")
        return 1.0, diag

    if g0.sum() < _FCORR_RES_MIN_CLEAN:
        return _fallback("too few good residual pixels")

    # Robust residual sky level + spread from the good pixels.
    v = r[g0]
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med)))
    if not np.isfinite(mad) or mad <= 0:
        mad = float(np.std(v))
    if not np.isfinite(mad) or mad <= 0:
        return _fallback("degenerate residual (zero spread)")

    # Per-pixel structure mask: reject residual bumps (unmodeled sources,
    # model mismatch) symmetrically; dilate to catch their wings.
    struct = (~g0) | (np.abs(r - med) > _FCORR_CLIP_NSIGMA * mad)
    clean = ~binary_dilation(struct, iterations=_FCORR_DILATE)

    # Tile the field; score each tile by its clean-pixel variance and keep
    # the cleanest quantile (least confusion structure). Correlated noise
    # inflates every tile's variance by the SAME F_corr, so the ranking
    # isolates the (non-uniform) confusion, not the (uniform) resampling.
    ts = int(_FCORR_RES_TILE)
    ny, nx = (H + ts - 1) // ts, (W + ts - 1) // ts
    tile_var = np.full((ny, nx), np.inf)
    for ty in range(ny):
        for tx in range(nx):
            sy = slice(ty * ts, min((ty + 1) * ts, H))
            sx = slice(tx * ts, min((tx + 1) * ts, W))
            cm = clean[sy, sx]
            nc = int(cm.sum())
            if nc < 0.25 * cm.size:      # tile too masked -> not a clean tile
                continue
            tile_var[ty, tx] = float(r[sy, sx][cm].var())
    finite = np.isfinite(tile_var)
    if int(finite.sum()) < _FCORR_RES_MIN_TILES:
        return _fallback("too few clean tiles")
    thr = float(np.quantile(tile_var[finite], _FCORR_RES_KEEP_Q))
    keep_tile = finite & (tile_var <= thr)
    # Build the analysis mask: clean pixels inside a kept tile.
    tile_ok = np.zeros((H, W), dtype=bool)
    for ty in range(ny):
        for tx in range(nx):
            if keep_tile[ty, tx]:
                tile_ok[ty * ts:min((ty + 1) * ts, H),
                        tx * ts:min((tx + 1) * ts, W)] = True
    m = clean & tile_ok
    n_clean = int(m.sum())
    diag["n_clean"] = n_clean
    diag["n_tiles_kept"] = int(keep_tile.sum())
    if n_clean < _FCORR_RES_MIN_CLEAN:
        return _fallback(f"too few clean tile pixels ({n_clean})")

    # Per-tile mean detrending: subtract each kept tile's OWN clean-pixel
    # mean so inter-tile DC offsets (residual large-scale gradients / diffuse
    # light the bg poly did not fully absorb) cannot inject spurious
    # short-lag correlation. The resampling correlation is compact (<< the
    # tile), so removing the per-tile DC leaves it intact.
    r_dt = r.copy()
    for ty in range(ny):
        for tx in range(nx):
            if not keep_tile[ty, tx]:
                continue
            sy = slice(ty * ts, min((ty + 1) * ts, H))
            sx = slice(tx * ts, min((tx + 1) * ts, W))
            sub_m = m[sy, sx]
            if sub_m.any():
                r_dt[sy, sx] = r[sy, sx] - float(r[sy, sx][sub_m].mean())

    # Mask-normalized FFT autocovariance of the detrended residual over the
    # kept mask (identical estimator to the raw-sky path).
    gf = m.astype(np.float64)
    x = np.where(m, r_dt, 0.0)
    Fx = np.fft.rfft2(x)
    Fm = np.fft.rfft2(gf)
    A = np.fft.irfft2(Fx * np.conj(Fx), s=(H, W))
    K = np.fft.irfft2(Fm * np.conj(Fm), s=(H, W))
    C = A / np.maximum(K, 1.0)
    C0 = float(C[0, 0])
    if not (np.isfinite(C0) and C0 > 0):
        return _fallback("non-positive residual variance")

    phi = np.asarray(psf_stamp, dtype=np.float64)
    psum = float(phi.sum())
    if not (np.isfinite(psum) and psum > 0):
        return _fallback("invalid PSF stamp")
    phi = phi / psum
    L = int(min(phi.shape[0] - 1, H // 2 - 1, W // 2 - 1))
    wfull = fftconvolve(phi, phi[::-1, ::-1], mode="full")
    cyw, cxw = wfull.shape[0] // 2, wfull.shape[1] // 2
    w = wfull[cyw - L:cyw + L + 1, cxw - L:cxw + L + 1] / wfull[cyw, cxw]
    dy = np.arange(-L, L + 1)
    dx = np.arange(-L, L + 1)
    rho = C[np.mod(dy[:, None], H), np.mod(dx[None, :], W)] / C0
    F_acf = float((rho * w).sum())
    diag["F_corr_acf"] = F_acf
    diag["rho_1px"] = 0.5 * (float(rho[L, L + 1]) + float(rho[L + 1, L]))

    if not np.isfinite(F_acf):
        return _fallback("non-finite F_corr")
    # ACF-SHAPE gate (zero tuned constants; replaces the retired F_corr>15
    # magnitude cut): even on the cleanest residual tiles, if the ACF is as
    # broad as the PSF autocorrelation the field has no confusion-free support
    # -> reject to F_corr = 1.0 (the honest bulge outcome).
    is_resamp, shp = _acf_shape_is_resampling(rho, w, L)
    diag["acf_shape"] = shp.get("acf_shape")
    diag["acf_q2"] = shp.get("q2")
    if not is_resamp:
        return _fallback(
            f"confusion-shaped residual ACF (q2={shp.get('q2'):.2f} "
            f">= 0.5; tracks the PSF autocorrelation at lag 2, no clean "
            f"residual support)")
    F_corr = float(max(F_acf, 1.0))
    if log_fn is not None:
        log_fn(
            f"correlated-noise F_corr_resid[{band}]={F_corr:.2f} "
            f"(sigma x{np.sqrt(F_corr):.2f}; ACF={F_acf:.2f} "
            f"rho1px={diag['rho_1px']:.3f} n_clean={n_clean} "
            f"tiles={diag['n_tiles_kept']}; shape={diag['acf_shape']})")
    return F_corr, diag
