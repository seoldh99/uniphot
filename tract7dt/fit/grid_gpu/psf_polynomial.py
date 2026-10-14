"""PSFEx-style 2-D polynomial PSF model.

For each band, the PSF stamp is modeled as

.. code-block::

    PSF_{i,j}(x, y) = sum_{m+n <= ORDER} c_{i,j,m,n} * X(x)^m * Y(y)^n

where (i, j) is a stamp pixel index (the stamp is ``H_psf x W_psf``,
typically 55x55), (x, y) is the sky position in the image (in pixels),
and X(x), Y(y) are normalized coordinates in [-1, 1] for numerical
stability of the polynomial. The coefficients ``c_{i,j,m,n}`` are
fitted once per band from individual stars (``psf_from_stars``).

For ORDER=3 (cubic), there are 10 spatial terms per pixel:
``(1, x, y, x^2, xy, y^2, x^3, x^2 y, x y^2, y^3)``.

This file provides:

- :func:`evaluate_psf_at` - given the fitted coefs and a position (x, y),
  return the PSF stamp at that position.
- :func:`evaluate_psf_at_batch` - vmapped over a batch of positions
  (used in forced photometry to render N sources' PSF stamps).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# JAX is only needed for the `to_jax()` conversion + the `_spatial_basis_jax`
# / `evaluate_psf_at*` rendering helpers used by the GPU forward model.
# We intentionally do NOT import jax at module import time so that
# downstream callers that only need `PsfPolynomial` + `terms_for_order`
# (e.g. the PSF-from-stars builder workers) don't pay the ~5 s cold
# spawn-worker JAX import cost. Functions below import jax / jnp lazily.

logger = logging.getLogger("tract7dt.fit.grid_gpu.psf_polynomial")


# Cubic = (m, n) tuples with m + n <= 3.  Total: 10 spatial terms.
# Order matches PSFEx convention: low order first.
_CUBIC_TERMS: tuple[tuple[int, int], ...] = (
    (0, 0),                                # 1
    (1, 0), (0, 1),                        # x, y
    (2, 0), (1, 1), (0, 2),                # x^2, xy, y^2
    (3, 0), (2, 1), (1, 2), (0, 3),        # x^3, x^2 y, x y^2, y^3
)
N_TERMS_CUBIC = len(_CUBIC_TERMS)  # = 10


@dataclass
class PsfPolynomial:
    """Fitted PSFEx-style polynomial PSF for one band.

    Attributes
    ----------
    coefs : np.ndarray
        Polynomial coefficients of shape ``(H_psf, W_psf, N_TERMS)``;
        ``coefs[i, j, t]`` is the coefficient of the ``t``-th spatial
        polynomial term for stamp pixel (i, j).
    x_origin, x_scale : float
        Linear-normalize the input x coordinate via
        ``X = (x - x_origin) / x_scale`` so X lies roughly in [-1, 1]
        across the image.
    y_origin, y_scale : float
        Same for y.
    H_psf, W_psf : int
        Stamp dimensions.
    """

    coefs: np.ndarray
    x_origin: float
    x_scale: float
    y_origin: float
    y_scale: float
    H_psf: int
    W_psf: int

    def to_jax(self) -> "PsfPolynomialJax":
        # Lazy JAX import: callers that never invoke to_jax() (e.g. the
        # PSF-from-stars builder workers) avoid the cold JAX import.
        import jax.numpy as jnp
        return PsfPolynomialJax(
            coefs=jnp.asarray(self.coefs),
            x_origin=float(self.x_origin),
            x_scale=float(self.x_scale),
            y_origin=float(self.y_origin),
            y_scale=float(self.y_scale),
            H_psf=int(self.H_psf),
            W_psf=int(self.W_psf),
        )


@dataclass
class PsfPolynomialJax:
    """JAX-pytree-friendly view of :class:`PsfPolynomial`. Used inside
    JIT-compiled rendering / fitting code.

    NOTE: the ``coefs`` field is annotated as ``jax.Array`` only as a
    forward-reference string (``from __future__ import annotations``
    makes all annotations lazy), so importing this dataclass does NOT
    import jax. Construct via :meth:`PsfPolynomial.to_jax` which does
    the lazy import.
    """
    coefs: "jax.Array"  # noqa: F821 -- lazy forward ref so module import doesn't pull jax
    x_origin: float
    x_scale: float
    y_origin: float
    y_scale: float
    H_psf: int
    W_psf: int


def _normalize_xy(x: np.ndarray, y: np.ndarray, x_origin: float, x_scale: float,
                  y_origin: float, y_scale: float) -> tuple[np.ndarray, np.ndarray]:
    return (x - x_origin) / x_scale, (y - y_origin) / y_scale


def _build_design_row(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Build the design-matrix row(s) for the cubic spatial polynomial.

    For input arrays of shape ``(N,)``, returns ``(N, N_TERMS_CUBIC)``.
    """
    rows = []
    for (m, n) in _CUBIC_TERMS:
        rows.append((X ** m) * (Y ** n))
    return np.stack(rows, axis=-1)


def _spatial_basis_jax(X, Y, n_terms: int = 10):
    """JAX version of the spatial basis evaluation. ``X``, ``Y`` are
    scalars or arrays of the same shape; returns the basis stacked in
    the last dim with size ``n_terms``.

    Supported polynomial orders (mapped from n_terms):
        - n_terms=1: order=0 (constant)
        - n_terms=3: order=1 (linear plane)
        - n_terms=6: order=2 (quadratic)
        - n_terms=10: order=3 (cubic)
        - n_terms=15: order=4 (quartic)
        - n_terms=21: order=5 (quintic)

    Implementation: compute X**m * Y**n for each (m, n) tuple in
    :func:`bg_polynomial.terms_for_order`. This is fully generic --
    extending to higher orders only requires extending ``terms_for_order``.
    """
    import jax.numpy as jnp
    order_for_n_terms = {1: 0, 3: 1, 6: 2, 10: 3, 15: 4, 21: 5}
    if n_terms not in order_for_n_terms:
        raise ValueError(
            f"_spatial_basis_jax: n_terms must be one of "
            f"{sorted(order_for_n_terms.keys())}; got {n_terms}"
        )
    from .bg_polynomial import terms_for_order
    terms = terms_for_order(order_for_n_terms[n_terms])
    one = jnp.ones_like(X)
    cols = []
    for (m, n) in terms:
        if m == 0 and n == 0:
            cols.append(one)
        elif m == 0:
            cols.append(Y ** n)
        elif n == 0:
            cols.append(X ** m)
        else:
            cols.append((X ** m) * (Y ** n))
    return jnp.stack(cols, axis=-1)


def evaluate_psf_at(psf_jax, x, y):
    """Evaluate the PSF stamp at image position (x, y).

    ``psf_jax`` is a :class:`PsfPolynomialJax` (polynomial spatial fit
    eval, differentiable w.r.t. (x, y), good for position Newton steps).

    Returns a stamp of shape ``(H_psf, W_psf)``.
    """
    import jax
    import jax.numpy as jnp
    X = (x - psf_jax.x_origin) / psf_jax.x_scale
    Y = (y - psf_jax.y_origin) / psf_jax.y_scale
    n_terms = int(psf_jax.coefs.shape[-1])
    basis = _spatial_basis_jax(X, Y, n_terms=n_terms)
    # PRECISION pin (development ledger SS23-32 + SS24): this
    # contraction BUILDS the spatially varying PSF stamp, upstream of
    # every render and every flux, and forced photometry never
    # renormalises the stamp afterwards -- so a truncated mantissa
    # here is a multiplicative error on every measured flux. XLA's
    # default demotes it to TF32 (10-bit mantissa). n_terms is 15, so
    # the contraction is trivial and HIGHEST is free.
    stamp = jnp.tensordot(psf_jax.coefs, basis, axes=([-1], [-1]),
                          precision=jax.lax.Precision.HIGHEST)
    return stamp  # (H_psf, W_psf)


def evaluate_psf_at_batch(psf_jax, xs, ys):
    """Vmapped evaluation: returns ``(N, H_psf, W_psf)`` for ``(N,)``
    input position arrays. Polymorphic on PSF type (see
    :func:`evaluate_psf_at`)."""
    import jax
    return jax.vmap(lambda x, y: evaluate_psf_at(psf_jax, x, y))(xs, ys)


def evaluate_psf_at_np(psf: PsfPolynomial, x: float, y: float) -> np.ndarray:
    """Pure-numpy PSF stamp evaluation.

    Identical math to :func:`evaluate_psf_at` for ``PsfPolynomial`` (no
    per-cell lookup branch), but uses numpy throughout so callers that
    only need a few stamp evaluations on the CPU (e.g. PSF residual
    diagnostic plots) don't have to import / spin up JAX. Returns a
    ``(H_psf, W_psf)`` float32 stamp.
    """
    X = (float(x) - psf.x_origin) / psf.x_scale
    Y = (float(y) - psf.y_origin) / psf.y_scale
    from .bg_polynomial import terms_for_order
    n_terms = int(psf.coefs.shape[-1])
    order_for_n_terms = {1: 0, 3: 1, 6: 2, 10: 3, 15: 4, 21: 5}
    if n_terms not in order_for_n_terms:
        raise ValueError(
            f"evaluate_psf_at_np: unsupported n_terms={n_terms}; "
            f"expected one of {sorted(order_for_n_terms.keys())}"
        )
    terms = terms_for_order(order_for_n_terms[n_terms])
    basis = np.empty(n_terms, dtype=np.float64)
    for t, (m, n) in enumerate(terms):
        basis[t] = (X ** m) * (Y ** n)
    coefs = psf.coefs.astype(np.float32, copy=False)  # (H, W, T)
    stamp = np.tensordot(coefs, basis.astype(np.float32), axes=([-1], [-1]))
    return stamp.astype(np.float32, copy=False)


def evaluate_psf_at_batch_np(
    psf: PsfPolynomial, xs: np.ndarray, ys: np.ndarray,
) -> np.ndarray:
    """Vectorized numpy PSF stamp evaluation: ``(N, H_psf, W_psf)``.

    All N positions are evaluated in a single matrix multiply
    ``coefs (H*W, T) @ basis (T, N)``, then reshaped. Avoids the
    per-stamp JIT-trace + host->device->host round trip that
    :func:`evaluate_psf_at` pays via JAX. Used by diagnostic-plot code
    in the parent process so the PSF-from-stars stage can avoid any
    JAX import.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.shape != ys.shape:
        raise ValueError(f"xs/ys shape mismatch: {xs.shape} vs {ys.shape}")
    X = (xs - psf.x_origin) / psf.x_scale
    Y = (ys - psf.y_origin) / psf.y_scale
    from .bg_polynomial import terms_for_order
    n_terms = int(psf.coefs.shape[-1])
    order_for_n_terms = {1: 0, 3: 1, 6: 2, 10: 3, 15: 4, 21: 5}
    if n_terms not in order_for_n_terms:
        raise ValueError(
            f"evaluate_psf_at_batch_np: unsupported n_terms={n_terms}"
        )
    terms = terms_for_order(order_for_n_terms[n_terms])
    N = xs.size
    basis = np.empty((n_terms, N), dtype=np.float32)  # (T, N)
    for t, (m, n) in enumerate(terms):
        basis[t] = ((X ** m) * (Y ** n)).astype(np.float32)
    H_psf, W_psf, _ = psf.coefs.shape
    coefs_flat = psf.coefs.reshape(H_psf * W_psf, n_terms).astype(np.float32, copy=False)
    stamps_flat = coefs_flat @ basis  # (H*W, N)
    stamps = stamps_flat.T.reshape(N, H_psf, W_psf).astype(np.float32, copy=False)
    return stamps


__all__ = [
    "N_TERMS_CUBIC",
    "PsfPolynomial",
    "PsfPolynomialJax",
    "evaluate_psf_at",
    "evaluate_psf_at_batch",
]
