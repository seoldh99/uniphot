"""Star fields: densities, positions, S/N-based fluxes, rendering, noisy images."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .noise import realize
from .psf import BOX, HALF, PSFField

# Design 6: crowded = 3.3e-3 stars / px^2 (T08285 input 215,886 rows / 6.6e7 px^2);
# sparse = 400 to 3,000 stars per image.
DENSITY_CROWDED = 3.3e-3
SPARSE_N_RANGE = (400, 3000)


def star_count(shape, kind: str, n: int | None = None, margin: float = HALF) -> int:
    """Number of stars for an image.

    "crowded": round(3.3e-3 x the area over which ``draw_positions(shape, n, rng, margin)`` draws
    centres), so the centre density equals the design value for any margin.  "sparse": n in
    [400, 3000] (the margin is not used).
    """
    h, w = shape
    if kind == "crowded":
        area = (w - 1 - 2 * margin) * (h - 1 - 2 * margin)
        if w - 1 - 2 * margin <= 0 or h - 1 - 2 * margin <= 0:
            raise ValueError(f"margin {margin} leaves no room in shape {shape}")
        return int(round(DENSITY_CROWDED * area))
    if kind == "sparse":
        if n is None or not SPARSE_N_RANGE[0] <= int(n) <= SPARSE_N_RANGE[1]:
            raise ValueError(f"sparse images have {SPARSE_N_RANGE[0]} to {SPARSE_N_RANGE[1]} stars, got {n!r}")
        return int(n)
    raise ValueError(f"kind must be 'sparse' or 'crowded', not {kind!r}")


def draw_positions(shape, n: int, rng: np.random.Generator, margin: float = HALF):
    """n uniform positions (x, y) with margin <= x <= W - 1 - margin (same for y)."""
    h, w = shape
    if w - 1 - margin < margin or h - 1 - margin < margin:
        raise ValueError(f"margin {margin} leaves no room in shape {shape}")
    x = rng.uniform(margin, w - 1 - margin, size=n)
    y = rng.uniform(margin, h - 1 - margin, size=n)
    return x, y


def crowded_positions(shape, rng: np.random.Generator, margin: float = HALF):
    """Centres of a crowded field at the design density over the drawing area.

    margin HALF (default): every box inside the image, no centres in the outer 27 px band.
    margin -HALF: centres also up to 27 px outside the image, so the image edge is as crowded as
    its interior (light of the outside sources enters the image; render() clips their boxes).
    """
    return draw_positions(shape, star_count(shape, "crowded", margin=margin), rng, margin)


def draw_snr(n: int, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    """n S/N values, uniform in log between lo and hi."""
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))


@dataclass
class SynthImage:
    """One synthetic image and its truth.

    data: noisy image.  model: noiseless image (expected value of data).
    x, y: positions where the light was rendered (this image's pixel frame).
    flux: truth box flux.  ix0, iy0: box centre pixels.  box_fraction: share of each
    source's unit-integral PSF inside its box.  sumsq, sumcube: sum(phi^2), sum(phi^3) of
    each source's box-normalised stamp.
    """
    data: np.ndarray
    model: np.ndarray
    x: np.ndarray
    y: np.ndarray
    flux: np.ndarray
    ix0: np.ndarray
    iy0: np.ndarray
    box_fraction: np.ndarray
    sumsq: np.ndarray
    sumcube: np.ndarray
    psf: PSFField
    sigma: float
    gain: float | None
    rho1: float
    meta: dict = field(default_factory=dict)


def render(shape, x, y, flux, psf: PSFField, fwhm_scale=None):
    """Noiseless image: sum of flux * box stamp.  Returns (model, info dict of per-source arrays)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    flux = np.atleast_1d(np.asarray(flux, dtype=float))
    n = len(x)
    scale = np.ones(n) if fwhm_scale is None else np.broadcast_to(np.asarray(fwhm_scale, float), (n,))
    h, w = shape
    model = np.zeros(shape)
    info = {"ix0": np.zeros(n, dtype=np.int64), "iy0": np.zeros(n, dtype=np.int64),
            "box_fraction": np.zeros(n), "sumsq": np.zeros(n), "sumcube": np.zeros(n)}
    for k in range(n):
        st = psf.stamp(x[k], y[k], fwhm_scale=scale[k])
        info["ix0"][k], info["iy0"][k] = st.ix0, st.iy0
        info["box_fraction"][k] = st.box_fraction
        info["sumsq"][k] = (st.data ** 2).sum()
        info["sumcube"][k] = (st.data ** 3).sum()
        x0, y0 = st.ix0 - HALF, st.iy0 - HALF
        xa, xb = max(0, x0), min(w, x0 + BOX)
        ya, yb = max(0, y0), min(h, y0 + BOX)
        if xa >= xb or ya >= yb:
            continue
        model[ya:yb, xa:xb] += flux[k] * st.data[ya - y0:yb - y0, xa - x0:xb - x0]
    return model, info


def make_image(shape, x, y, flux, psf: PSFField, rng: np.random.Generator,
               sigma: float = 1.0, gain: float | None = 50.0, rho1: float = 0.15,
               fwhm_scale=None) -> SynthImage:
    """Render the sources and add Poisson(g model)/g + correlated sky noise."""
    model, info = render(shape, x, y, flux, psf, fwhm_scale=fwhm_scale)
    data = realize(model, rng, sigma=sigma, gain=gain, rho1=rho1)
    return SynthImage(data, model, np.atleast_1d(np.asarray(x, dtype=float)),
                      np.atleast_1d(np.asarray(y, dtype=float)),
                      np.atleast_1d(np.asarray(flux, dtype=float)), info["ix0"], info["iy0"],
                      info["box_fraction"], info["sumsq"], info["sumcube"], psf, sigma, gain, rho1,
                      meta={"fwhm_scale": None if fwhm_scale is None else np.asarray(fwhm_scale, float)})
