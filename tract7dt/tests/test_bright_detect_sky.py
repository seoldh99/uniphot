"""White-stack bright-source detection measures significance above the local sky (uniphot 0.1.1).

Why: some coadds carry an unsubtracted sky pedestal (the public UDS example: SKYVAL = 0.73-0.84 SKYSIG per band).
The white stack then sits about +1.6 sigma_white above zero for 4 bands, +4.3 sigma after SEP's 3x3 matched
filter, so with a 5 sigma threshold about 30% of the frame is "detected" as large connected regions; on the
example the largest grows past SEP's 2,000,000-pixel active stack and the run stops ("internal pixel buffer
full"). Runs that did not overflow masked the pedestal regions as "bright unmatched sources".

Expected values are independent of the implementation: a synthetic white stack with unit noise, a flat
pedestal of 2.0 sigma plus a 0.5 sigma gradient across the frame, and 36 isolated Gaussian point sources
(peak 100 sigma, sigma 1.5 px) on a 150 px grid. Above the local sky the only significant structure is the 36
sources: each is found within 2 px, nothing else passes 5 sigma with 20 connected pixels in pure Gaussian
noise (at most 2 allowed), and no detection is larger than 2,000 px (a source covers about 60-100 px above
the threshold). Without the sky removed, the 2.0 sigma pedestal is 5.3 sigma after the filter and the frame
is one connected detection.

The noise is measured too: sigma_white comes from one header SKYSIG per frame, but where a coadd thins out
(its footprint rim) the scatter is several times larger (UDS example: the rim, 14-284 px from the working-frame
edge, passed the threshold as one ~1e6-pixel region after the sky was removed). Second scene: the same sources,
no pedestal, and a 150 px band along one edge whose true noise is 5x the stated sigma_white (the UDS example: measured scatter p99 = 4.97 x sigma_white). Expected: the same
36 sources, at most 2 other detections, none larger than 2,000 px.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from tract7dt.bright_mask import detect_bright_sources
from tract7dt.config import DEFAULT_CONFIG


def _scene(seed: int = 7, rim: bool = False):
    rng = np.random.default_rng(seed)
    n = 1200
    yy, xx = np.mgrid[0:n, 0:n]
    if rim:
        img = rng.normal(0.0, 1.0, (n, n)) * np.where(yy < 150, 5.0, 1.0)
    else:
        img = 2.0 + 0.5 * (xx / (n - 1)) + rng.normal(0.0, 1.0, (n, n))
    # sources are placed at y >= 150, outside the noisy band of the rim scene
    pos = [(x, y) for y in range(150, n - 100, 150) for x in range(150, n - 100, 150)][:36]
    pos = [(x + rng.uniform(-0.5, 0.5), y + rng.uniform(-0.5, 0.5)) for x, y in pos]
    for x0, y0 in pos:
        sl = (slice(int(y0) - 10, int(y0) + 11), slice(int(x0) - 10, int(x0) + 11))
        img[sl] += 100.0 * np.exp(-((xx[sl] - x0) ** 2 + (yy[sl] - y0) ** 2) / (2 * 1.5 ** 2))
    return img.astype(np.float32), np.ones((n, n), np.float32), np.array(pos)


@pytest.mark.parametrize("rim", [False, True], ids=["sky-pedestal", "noisy-rim"])
def test_bright_detection_is_relative_to_the_local_sky_and_noise(rim):
    img, err, pos = _scene(rim=rim)
    assert len(pos) == 36
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    objs = detect_bright_sources(cfg, {"white": img, "sigma_white": err})
    det = np.column_stack([objs["x"], objs["y"]])
    d_src = np.hypot(det[:, None, 0] - pos[None, :, 0], det[:, None, 1] - pos[None, :, 1])
    assert np.all(d_src.min(axis=0) < 2.0), "an injected source has no detection within 2 px"
    n_spurious = int(np.sum(d_src.min(axis=1) > 3.0))
    assert n_spurious <= 2, f"{n_spurious} detections are not injected sources"
    assert int(objs["npix"].max()) < 2000, f"largest detection {int(objs['npix'].max())} px"


def test_detection_noise_stays_positive_where_the_measured_scatter_interpolates_to_zero_or_below(monkeypatch):
    """A steep step in the noise: the same 36 sources (at y >= 300) and a 150 px band along one edge whose true
    noise is 30x the stated sigma_white (the public example's measured scatter reaches 36x at the coadd rim).
    SEP's rms map is a bicubic spline through the 64 px mesh values; next to such a step it undershoots to zero
    or below in a strip of normal-noise pixels (precondition, checked on SEP's own background of this scene).
    A noise that is not positive has no meaning as a significance unit (a negative one turns the 5 sigma
    threshold negative there). Expected: the noise handed to SEP is finite and positive everywhere, and the 36
    sources are found within 2 px.

    Known limitation, not asserted here: the 64 px mesh cannot follow a 30x step, so a few noise peaks at the
    band edge and next to the undershoot strip still pass the threshold (17 detections in this scene)."""
    import sep

    from tract7dt import bright_mask

    rng = np.random.default_rng(7)
    n = 1200
    yy, xx = np.mgrid[0:n, 0:n]
    img = rng.normal(0.0, 1.0, (n, n)) * np.where(yy < 150, 30.0, 1.0)
    pos = [(x, y) for y in range(300, n - 100, 150) for x in range(150, n - 100, 150)][:36]
    pos = np.array([(x + rng.uniform(-0.5, 0.5), y + rng.uniform(-0.5, 0.5)) for x, y in pos])
    for x0, y0 in pos:
        sl = (slice(int(y0) - 10, int(y0) + 11), slice(int(x0) - 10, int(x0) + 11))
        img[sl] += 100.0 * np.exp(-((xx[sl] - x0) ** 2 + (yy[sl] - y0) ** 2) / (2 * 1.5 ** 2))
    img = img.astype(np.float32)
    err = np.ones((n, n), np.float32)
    none = np.zeros((n, n), bool)
    sky_free = img - sep.Background(img, mask=none, bw=64, bh=64, fw=3, fh=3).back(dtype=np.float32)
    rms = sep.Background(sky_free, mask=none, bw=64, bh=64, fw=3, fh=3).rms(dtype=np.float32)
    assert (rms[150:] <= 0).any(), "scene precondition: no undershoot"

    seen = {}
    real_extract = bright_mask.sep.extract

    def spy(data, *args, **kwargs):
        seen["err"] = np.array(kwargs["err"], copy=True)
        return real_extract(data, *args, **kwargs)

    monkeypatch.setattr(bright_mask.sep, "extract", spy)
    objs = detect_bright_sources(copy.deepcopy(DEFAULT_CONFIG), {"white": img, "sigma_white": err})
    e = seen["err"]
    assert np.all(np.isfinite(e)) and np.all(e > 0), f"{int(np.sum(~(e > 0)))} pixels with a noise <= 0"
    det = np.column_stack([objs["x"], objs["y"]])
    d_src = np.hypot(det[:, None, 0] - pos[None, :, 0], det[:, None, 1] - pos[None, :, 1])
    assert np.all(d_src.min(axis=0) < 2.0), "an injected source has no detection within 2 px"


def test_white_stack_and_noise_are_not_modified():
    img, err, _ = _scene()
    img0, err0 = img.copy(), err.copy()
    detect_bright_sources(copy.deepcopy(DEFAULT_CONFIG), {"white": img, "sigma_white": err})
    assert np.array_equal(img, img0) and np.array_equal(err, err0)
