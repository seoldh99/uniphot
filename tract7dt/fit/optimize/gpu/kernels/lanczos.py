"""Lanczos-3 separable image shift in JAX.

Mirrors the Python fallback path of Tractor's
:func:`tractor.psf.lanczos_shift_image` (the C-extension
fast path is selected by upstream Tractor only when the
``mp_fourier`` C extension is available - the Tractor build that tract7dt
once bundled fell back to this Python path, so this is the path that
executed in that pipeline).

The Python reference is::

    Lx = lanczos_filter(L=3, np.arange(-3, 4) + dx)   # 7 taps
    Ly = lanczos_filter(L=3, np.arange(-3, 4) + dy)
    Lx /= Lx.sum()
    Ly /= Ly.sum()
    sx     = scipy.ndimage.correlate1d(img, Lx, axis=1, mode='constant')
    outimg = scipy.ndimage.correlate1d(sx,  Ly, axis=0, mode='constant')

with :func:`astrometry.util.miscutils.lanczos_filter` defined as::

    L_3(x) = 0                                      if |x| >= 3
           = 1                                      if x == 0
           = 3 * sin(pi*x) * sin(pi*x/3) / (pi*x)^2 otherwise

The kernel is computed in float64 on host-side then cast to float32 to
match the upstream behavior (``lanczos_filter`` returns float64; the
input image is float32 for PixelizedPSF stamps).

Provenance (card 018): this implementation is modelled on Tractor's
``lanczos_shift_image`` and is one of the four Tractor-derived pieces still in
the package (with the MoG constants in ``fit/grid_gpu/source_model.py``, the
Sersic mixture table ``data/sersic_mog_table.npz`` and its generator
``fit/grid_gpu/sersic_mog_gen.py``, and the alpha grid in
``fit/grid_gpu/solver_grid.py``). It is to be re-derived under a permissive
licence and moved to :mod:`tract7dt.fit.grid_gpu` after the licence decision.
The vendored Tractor itself is no longer part of this package.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# Module-level constant: tap positions [-3, -2, -1, 0, 1, 2, 3].
# Computed eagerly so it lives on host until the kernel is jit-traced.
_TAP_OFFSETS_F64 = jnp.arange(-3, 4, dtype=jnp.float64)


def lanczos3_kernel(d):
    """Compute the 7-tap normalized Lanczos-3 filter for sub-pixel shift ``d``.

    Parameters
    ----------
    d : float or jnp.ndarray scalar
        Sub-pixel shift in [-0.5, 0.5] (no enforcement; values outside
        that range produce a valid but less-useful filter, just as in
        the upstream code). Should be a JAX-traceable scalar so the
        kernel can be JIT'd; promoted to float64 internally to match
        upstream's float64 ``lanczos_filter`` precision, then cast to
        float32 on return.

    Returns
    -------
    weights : jnp.ndarray, shape (7,), dtype float32
        Lanczos-3 weights at positions ``[d-3, d-2, ..., d+3]``,
        normalized to sum to 1.
    """
    # Upstream uses dtype=float (i.e. float64) for the kernel computation,
    # then implicitly casts to float32 when convolving with a float32
    # image. Match that ordering for parity within FP32 tolerance.
    d_f64 = jnp.asarray(d, dtype=jnp.float64)
    x = _TAP_OFFSETS_F64 + d_f64

    # lanczos_filter(3, x):
    #   - 0     where |x| >= 3
    #   - 1     where x == 0
    #   - 3 * sin(pi*x) * sin(pi*x/3) / (pi*x)^2  otherwise
    # We compute the formula everywhere using a `safe_x` (replacing zero
    # with one to avoid 0/0), then mask afterwards. This is the standard
    # trick to keep grad-safe under autodiff.
    abs_x = jnp.abs(x)
    is_zero = abs_x < 1e-12
    is_outside = abs_x >= 3.0
    is_inside_nonzero = (~is_zero) & (~is_outside)

    safe_x = jnp.where(is_zero, 1.0, x)
    pi_safe_x = jnp.pi * safe_x
    val = 3.0 * jnp.sin(pi_safe_x) * jnp.sin(pi_safe_x / 3.0) / (pi_safe_x ** 2)

    weights = jnp.where(is_zero, 1.0, jnp.where(is_outside, 0.0, val))
    # Match upstream: only the `inside-nonzero` band is set to the formula;
    # the `outside` band is explicitly zeroed. ``is_inside_nonzero`` is
    # actually unused after the chained ``where`` above but kept around
    # to make the intent explicit if a future reader debugs this.
    _ = is_inside_nonzero

    weights = weights / jnp.sum(weights)
    return weights.astype(jnp.float32)


def _correlate1d_axis(img, kernel, axis):
    """Equivalent to ``scipy.ndimage.correlate1d(img, kernel, axis=axis, mode='constant')``.

    ``mode='constant'`` with default ``cval=0`` means pixels outside the
    image are treated as zero. The kernel of length K is centered at
    index K//2, so for K=7 the center is at index 3 and ``out[i, j] =
    sum_k kernel[k] * img[i, j + k - 3]`` (along the chosen axis).

    Implemented as an EXPLICIT weighted sum of K shifted slices of the
    zero-padded image, accumulated in kernel-index order (k = 0..K-1) --
    the same TERMS the scipy reference sums, in the same order, in float32.
    (Not bit-for-bit equal to scipy: `correlate1d` accumulates in the weight
    dtype, and the taps here are float32 by the time they are applied.
    Measured against that reference this form is nonetheless CLOSER than the
    convolution it replaces: 3.09e-7 vs 4.18e-7 max relative.)

    Inside ``jax.jit`` XLA fuses the taps into one elementwise kernel per axis;
    called EAGERLY each tap is its own dispatch, so eager callers of this
    function pay more launches than the convolution did (all of the hot callers
    -- the island kernels, the S1 render, the compiled scene-subtract render --
    are jitted).

    History (effarc2, 2026-08-26): this used to be one
    ``jax.lax.conv_general_dilated`` per axis. XLA lowers that to a cuDNN
    grouped convolution; inside the vmapped/scanned island kernels every
    stamp x band x tangent became its own cuDNN launch (plus the pad /
    transpose / FFT-candidate kernels around it), and nsys measured that
    conv path at 60-80% of the island LM step's GPU time. The explicit
    taps fuse into one elementwise XLA fusion per axis. Numerically this
    is the fixed-order f32 sum; the cuDNN path (algorithm chosen by the
    autotuner, FFT-based candidates included) was never bit-identical to
    the scipy reference either, so the change is of the fp-noise class.
    """
    K = int(kernel.shape[0])
    pad = K // 2
    if axis == 1:
        n = img.shape[1]
        padded = jnp.pad(img, ((0, 0), (pad, pad)), mode="constant")
        out = kernel[0] * padded[:, 0:n]
        for k in range(1, K):
            out = out + kernel[k] * padded[:, k:k + n]
    elif axis == 0:
        n = img.shape[0]
        padded = jnp.pad(img, ((pad, pad), (0, 0)), mode="constant")
        out = kernel[0] * padded[0:n, :]
        for k in range(1, K):
            out = out + kernel[k] * padded[k:k + n, :]
    else:
        raise ValueError(f"axis must be 0 or 1, got {axis!r}")
    return out


def lanczos_shift_image(img, dx, dy):
    """Sub-pixel-shift a 2-D image by (dx, dy) using a separable
    Lanczos-3 kernel; mirrors the Python fallback path of Tractor's
    :func:`tractor.psf.lanczos_shift_image`.

    Parameters
    ----------
    img : jnp.ndarray, shape (H, W)
        Input image. Must be 2-D. The upstream code uses float32 as the
        canonical PSF-stamp dtype; we accept any float dtype and return
        float32 to match.
    dx, dy : scalar (Python float or JAX scalar)
        Sub-pixel shift in the x and y directions, expected in
        [-0.5, 0.5]. Positive ``dx`` shifts content in the +x direction
        (i.e. the output sample at column j picks up content from
        roughly column j-dx of the input).
    Returns
    -------
    out : jnp.ndarray, shape (H, W), dtype float32
        Shifted image. Pixels outside the input boundary are treated
        as zero (``mode='constant'`` in ``scipy.ndimage.correlate1d``).
    """
    img = jnp.asarray(img, dtype=jnp.float32)
    Lx = lanczos3_kernel(dx)
    Ly = lanczos3_kernel(dy)
    sx = _correlate1d_axis(img, Lx, axis=1)
    out = _correlate1d_axis(sx, Ly, axis=0)
    return out


__all__ = [
    "lanczos3_kernel",
    "lanczos_shift_image",
]
