"""2-D polynomial background model.

Per band, the background is

.. code-block::

    bg(x, y) = sum_{m+n <= ORDER} a_{m,n} * X(x)^m * Y(y)^n

where (X, Y) are normalized coordinates in [-1, 1] (same convention as
:mod:`.psf_polynomial`). For ORDER=2 (quadratic), there are 6 terms:
``(1, X, Y, X^2, XY, Y^2)``.

The background coefficients are fitted JOINTLY with the per-source
fluxes in :mod:`.forced_phot_sparse` - this module just provides the basis
evaluation + design-matrix construction utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

# JAX is only used by `make_bg_basis_jax` and `evaluate_bg` (the GPU
# forward-model path). We import it lazily inside those functions so
# that callers that just need `terms_for_order` / `n_terms_for_order`
# (e.g. the PSF-from-stars builder workers) avoid the cold spawn-worker
# JAX import cost.

_CONST_TERMS: tuple[tuple[int, int], ...] = (
    (0, 0),
)
_LINEAR_TERMS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 0), (0, 1),
)
_QUADRATIC_TERMS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 0), (0, 1),
    (2, 0), (1, 1), (0, 2),
)
N_TERMS_CONST = len(_CONST_TERMS)            # = 1
N_TERMS_LINEAR = len(_LINEAR_TERMS)          # = 3
N_TERMS_QUADRATIC = len(_QUADRATIC_TERMS)    # = 6


_CUBIC_TERMS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 0), (0, 1),
    (2, 0), (1, 1), (0, 2),
    (3, 0), (2, 1), (1, 2), (0, 3),
)
N_TERMS_CUBIC = len(_CUBIC_TERMS)             # = 10

_QUARTIC_TERMS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 0), (0, 1),
    (2, 0), (1, 1), (0, 2),
    (3, 0), (2, 1), (1, 2), (0, 3),
    (4, 0), (3, 1), (2, 2), (1, 3), (0, 4),
)
N_TERMS_QUARTIC = len(_QUARTIC_TERMS)          # = 15

_QUINTIC_TERMS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 0), (0, 1),
    (2, 0), (1, 1), (0, 2),
    (3, 0), (2, 1), (1, 2), (0, 3),
    (4, 0), (3, 1), (2, 2), (1, 3), (0, 4),
    (5, 0), (4, 1), (3, 2), (2, 3), (1, 4), (0, 5),
)
N_TERMS_QUINTIC = len(_QUINTIC_TERMS)          # = 21


def terms_for_order(order: int) -> tuple[tuple[int, int], ...]:
    """Return the polynomial terms (m, n) tuples for a given spatial order.

    - order=0: constant (1 term: (0, 0))
    - order=1: linear PLANE (3 terms: 1, x, y)
    - order=2: quadratic (6 terms: 1, x, y, x^2, x*y, y^2)
    - order=3: cubic (10 terms: + x^3, x^2*y, x*y^2, y^3)
    - order=4: quartic (15 terms: + x^4, x^3*y, x^2*y^2, x*y^3, y^4)
    - order=5: quintic (21 terms: + x^5, x^4*y, ..., y^5)
    """
    if order == 0:
        return _CONST_TERMS
    if order == 1:
        return _LINEAR_TERMS
    if order == 2:
        return _QUADRATIC_TERMS
    if order == 3:
        return _CUBIC_TERMS
    if order == 4:
        return _QUARTIC_TERMS
    if order == 5:
        return _QUINTIC_TERMS
    raise ValueError(
        f"polynomial_order must be 0..5; got {order}"
    )


def n_terms_for_order(order: int) -> int:
    return len(terms_for_order(order))


@dataclass
class BgPolynomial:
    """Fitted 2-D polynomial background coefficients for one band.

    Attributes
    ----------
    coefs : np.ndarray, shape (N_TERMS_QUADRATIC,)
        ``coefs[t]`` is the coefficient of the ``t``-th polynomial term.
    x_origin, x_scale, y_origin, y_scale : float
        Same as :class:`.psf_polynomial.PsfPolynomial`.
    """
    coefs: np.ndarray
    x_origin: float
    x_scale: float
    y_origin: float
    y_scale: float


def make_bg_basis(
    band_h: int, band_w: int,
    x0: int, y0: int,
    x_origin: float, x_scale: float,
    y_origin: float, y_scale: float,
    order: int = 2,
) -> np.ndarray:
    """Build the per-pixel background basis array.

    Parameters
    ----------
    band_h, band_w : int
        Grid image height and width in pixels.
    x0, y0 : int
        Pixel offset of (this grid's pixel (0, 0)) in the FULL image's
        pixel coords. This is so the polynomial uses the SAME
        normalization across grids as a single per-band fit would.
    x_origin, x_scale, y_origin, y_scale : float
        Normalization params (typically the same as the per-band PSF
        polynomial uses).
    order : int
        Polynomial spatial order (0=const, 1=linear plane, 2=quadratic).

    Returns
    -------
    basis : np.ndarray, shape (band_h * band_w, n_terms_for_order(order)), float32
        Each row is the basis evaluated at one pixel of the grid image.
    """
    if order <= 1 and os.environ.get("TRACT7DT_BENCH_DISABLE_BG_FASTPATH") != "1":
        x = np.arange(band_w, dtype=np.float64) + x0
        y = np.arange(band_h, dtype=np.float64) + y0
        X = (x - x_origin) / x_scale
        Y = (y - y_origin) / y_scale
        if order == 0:
            return np.ones((band_h * band_w, 1), dtype=np.float32)
        return np.stack(
            [
                np.ones(band_h * band_w, dtype=np.float32),
                np.broadcast_to(X[None, :], (band_h, band_w)).reshape(-1).astype(np.float32),
                np.broadcast_to(Y[:, None], (band_h, band_w)).reshape(-1).astype(np.float32),
            ],
            axis=-1,
        )

    terms = terms_for_order(order)
    yy, xx = np.mgrid[:band_h, :band_w].astype(np.float64)
    xx = xx + x0
    yy = yy + y0
    X = (xx - x_origin) / x_scale
    Y = (yy - y_origin) / y_scale
    cols = []
    for (m, n) in terms:
        cols.append(((X ** m) * (Y ** n)).reshape(-1))
    return np.stack(cols, axis=-1).astype(np.float32)


def make_bg_basis_jax(
    band_h: int, band_w: int,
    x0: int, y0: int,
    x_origin: float, x_scale: float,
    y_origin: float, y_scale: float,
    order: int = 2,
):
    """JAX version (output device array). Same shape as
    :func:`make_bg_basis`."""
    import jax.numpy as jnp
    if order <= 1 and os.environ.get("TRACT7DT_BENCH_DISABLE_BG_FASTPATH") != "1":
        x = jnp.arange(band_w, dtype=jnp.float32)
        y = jnp.arange(band_h, dtype=jnp.float32)
        X = ((x + x0).astype(jnp.float64) - x_origin) / x_scale
        Y = ((y + y0).astype(jnp.float64) - y_origin) / y_scale
        if order == 0:
            return jnp.ones((band_h * band_w, 1), dtype=jnp.float32)
        return jnp.stack(
            [
                jnp.ones((band_h * band_w,), dtype=jnp.float32),
                jnp.broadcast_to(X[None, :], (band_h, band_w)).reshape(-1).astype(jnp.float32),
                jnp.broadcast_to(Y[:, None], (band_h, band_w)).reshape(-1).astype(jnp.float32),
            ],
            axis=-1,
        )

    terms = terms_for_order(order)
    yy, xx = jnp.meshgrid(jnp.arange(band_h, dtype=jnp.float32),
                          jnp.arange(band_w, dtype=jnp.float32),
                          indexing="ij")
    xx = xx + x0
    yy = yy + y0
    X = (xx.astype(jnp.float64) - x_origin) / x_scale
    Y = (yy.astype(jnp.float64) - y_origin) / y_scale
    cols = []
    for (m, n) in terms:
        cols.append(((X ** m) * (Y ** n)).reshape(-1))
    return jnp.stack(cols, axis=-1).astype(jnp.float32)


def evaluate_bg(
    coefs,
    band_h: int, band_w: int,
    x0: int, y0: int,
    x_origin: float, x_scale: float,
    y_origin: float, y_scale: float,
    order: int = 2,
):
    """Evaluate the background polynomial on the grid image. Returns
    a ``(band_h, band_w)`` array. ``order`` must match the order used
    to fit ``coefs``."""
    basis = make_bg_basis_jax(band_h, band_w, x0, y0, x_origin, x_scale,
                              y_origin, y_scale, order=order)
    return (basis @ coefs).reshape(band_h, band_w)


__all__ = [
    "N_TERMS_CONST", "N_TERMS_LINEAR", "N_TERMS_QUADRATIC", "N_TERMS_CUBIC",
    "N_TERMS_QUARTIC", "N_TERMS_QUINTIC",
    "BgPolynomial",
    "terms_for_order", "n_terms_for_order",
    "make_bg_basis",
    "make_bg_basis_jax",
    "evaluate_bg",
]
