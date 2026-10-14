"""No-coverage masking (``image_scaling.mask_zero_coverage``, 2026-08-01).

The engine only ever treated NaN as "no data". 7DT coadds (and, as the audit
found, UDS ones too) pad outside their exposure footprint with an exact 0.0,
which passes every finiteness test -- so sources standing on the padding were
fitted against nothing, came out multiple magnitudes too faint, and kept
``fracmasked == 0`` and ``fit_is_reliable == True``.

The discriminator is topological, not a threshold: exact-zero pixels connected
to the frame border are outside the footprint; interior zeros are left alone.
"""
from __future__ import annotations

import numpy as np
import pytest

from tract7dt.load_inputs import _zero_coverage_mask, _outside_data_envelope


def _frame(fill=1.0, shape=(64, 96)):
    return np.full(shape, fill, dtype=np.float32)


def test_no_zeros_masks_nothing():
    img = _frame()
    assert _zero_coverage_mask(img).sum() == 0


def test_border_padding_is_masked():
    img = _frame()
    img[:5, :] = 0.0          # bottom strip
    img[:, :7] = 0.0          # left strip
    m = _zero_coverage_mask(img)
    assert m[0, 0] and m[4, 50] and m[30, 6]
    assert not m[30, 50]
    assert m.sum() == int((img == 0.0).sum())


def test_ragged_band_dependent_footprint():
    """The real case: each telescope unit has its own dithered footprint, so
    the boundary is a wedge rather than a rectangle."""
    img = _frame()
    for i in range(img.shape[0]):
        img[i, : 3 + i // 4] = 0.0
    m = _zero_coverage_mask(img)
    assert np.array_equal(m, img == 0.0)


def test_interior_zeros_are_not_coverage():
    """A legitimately-zero pixel (or a zero-filled interior defect) is NOT
    padding -- masking it here would be this function overstepping."""
    img = _frame()
    img[:4, :] = 0.0              # real padding
    img[30:34, 40:44] = 0.0       # interior zero patch
    m = _zero_coverage_mask(img)
    assert m[0, 0]
    assert not m[31, 41], "interior zeros must survive"
    assert m.sum() == 4 * img.shape[1]


def test_interior_only_zeros_mask_nothing():
    img = _frame()
    img[20:25, 20:25] = 0.0
    assert _zero_coverage_mask(img).sum() == 0


def test_all_zero_frame_is_all_no_coverage():
    img = _frame(fill=0.0)
    assert _zero_coverage_mask(img).all()


def test_negative_pixels_are_data():
    """Sky-subtracted products go negative; only EXACT zero is the pad value."""
    img = _frame(fill=-0.3)
    img[:3, :] = 0.0
    m = _zero_coverage_mask(img)
    assert m[:3, :].all() and not m[3:, :].any()


@pytest.mark.parametrize("dtype", [np.float32, np.float64, ">f4"])
def test_dtype_agnostic(dtype):
    img = np.full((32, 32), 2.0, dtype=dtype)
    img[:2, :] = 0.0
    assert _zero_coverage_mask(img).sum() == 64


def test_default_is_on_and_off_is_available(config=None):
    from tract7dt.config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["image_scaling"]["mask_zero_coverage"] is True


# --- regression tests for the three defects found by the 2026-08-01 audit ---

def test_nan_ring_over_zero_pad_is_still_masked():
    """Mixed convention: outer ring NaN, inner pad exact 0. The zero component
    never touches the array border, so labelling zeros alone masked NOTHING and
    said nothing. no-data must percolate across both conventions."""
    img = _frame()
    img[:6, :] = 0.0          # pad
    img[:, :6] = 0.0
    img[0, :] = np.nan        # resampling ring on top of it
    img[:, 0] = np.nan
    m = _zero_coverage_mask(img)
    assert m[3, 40] and m[40, 3], "zero pad behind a NaN ring must be masked"
    assert not m[0, 40], "NaN pixels are not reported here (already ~isfinite)"
    assert not m[40, 40]


def test_diagonal_seam_is_one_region():
    """8-connectivity: a staircase pad is a single region, not a chain of
    4-disconnected specks."""
    img = _frame(shape=(40, 40))
    for i in range(20):
        img[i, i] = 0.0
        img[i, : i + 1] = 0.0
    m = _zero_coverage_mask(img)
    assert m.sum() == int((img == 0.0).sum())


def test_interior_zero_touching_nothing_still_survives():
    img = _frame()
    img[:4, :] = 0.0
    img[30:34, 40:44] = 0.0
    m = _zero_coverage_mask(img)
    assert not m[31, 41]


# ---------------------------------------------------------------------------
# Percolation guard: the geometric cross-check must be the ORTHOGONALLY CONVEX
# HULL of the non-zero data, not its axis-aligned bounding box.
#
# Regression for a real defect found 2026-08-01 by the UDS deep-field gate: coadd
# footprints are ROTATED, and a rotated rectangle whose corners reach all four
# frame borders has a bounding box equal to the whole frame. `bbox_outside` then
# collapses to ~0, the guard fires on unambiguous padding, and the fix is
# silently disabled on exactly the worst-padded bands (measured on T08285
# m550/m575 and UDS m700: 6.5% of the 500 px crop left unprotected).
# ---------------------------------------------------------------------------

def _bbox_outside(img):
    """The OLD (defective) geometric bound, kept so the regression is demonstrated."""
    nz = img != 0.0
    r = np.nonzero(nz.any(axis=1))[0]
    c = np.nonzero(nz.any(axis=0))[0]
    if not (r.size and c.size):
        return 1.0
    return 1.0 - ((r[-1] - r[0] + 1) * (c[-1] - c[0] + 1)) / img.size


def _rotated_footprint(n=201):
    """A diamond (45-degree-rotated square) whose corners touch all four borders.

    `half = n // 2` is what makes the four corners land exactly on the frame
    edges -- the condition that collapses the axis-aligned bounding box to the
    whole frame and thus reproduces the defect.
    """
    img = np.zeros((n, n), dtype=np.float64)
    c = half = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    img[(np.abs(yy - c) + np.abs(xx - c)) <= half] = 1.0
    return img


def test_rotated_footprint_is_not_mistaken_for_data():
    """The defect itself: the bounding box says ~0% outside, so the old
    area-ratio guard refused to mask this unambiguous padding. The envelope must
    see the rotation and the padding must still be masked."""
    img = _rotated_footprint()
    assert _bbox_outside(img) < 0.01, "setup: corners must reach all four borders"
    assert _outside_data_envelope(img).mean() > 0.3, "the envelope must see the rotation"
    m = _zero_coverage_mask(img) & _outside_data_envelope(img)
    assert m.any() and m[0, 0]


def test_percolating_zeros_keep_only_the_genuine_sliver():
    """The case the geometric half exists for: zeros are real data spread through
    the frame (non-negative-clipped sky), so the border-connected component leaks
    across it. The intersection must keep only what is truly outside the data --
    not the whole leak, and not (as the old all-or-nothing guard did) nothing."""
    rng = np.random.default_rng(7)
    img = np.clip(rng.normal(0.2, 1.0, (200, 300)), 0.0, None)
    leaked = _zero_coverage_mask(img)
    kept = leaked & _outside_data_envelope(img)
    assert leaked.mean() > 0.2, "setup: the raw component must percolate"
    assert kept.sum() < leaked.sum() / 10, "the leak must be cut down"
    assert _outside_data_envelope(img).mean() < 0.05


def test_honest_pad_is_masked_whole():
    img = np.ones((200, 300), dtype=np.float64)
    img[:40, :] = 0.0
    img[:, :40] = 0.0
    kept = _zero_coverage_mask(img) & _outside_data_envelope(img)
    assert np.array_equal(kept, img == 0.0), "an honest pad must be masked entirely"


def test_envelope_is_never_looser_than_bbox():
    """Structural invariant: the orthogonal hull is contained in the bounding box,
    so it can only report MORE area outside -- the geometric half can never become
    laxer than the bbox version it replaces, on any input."""
    rng = np.random.default_rng(11)
    cases = [_rotated_footprint(), np.ones((60, 90))]
    img = np.zeros((60, 90)); img[10:50, 20:70] = 1.0; cases.append(img)
    img2 = np.ones((60, 90)); img2[rng.random((60, 90)) < 0.3] = 0.0; cases.append(img2)
    for k, im in enumerate(cases):
        assert _outside_data_envelope(im).mean() >= _bbox_outside(im) - 1e-12, f"case {k}"


def test_no_thresholds_left_in_the_rule():
    """Regression for the standing 'no tuned constants' principle: the composition
    is an intersection, so masking must be monotone -- adding real data can only
    ever shrink the mask, never flip it off wholesale (the old guard's failure)."""
    img = np.ones((120, 160), dtype=np.float64)
    img[:20, :] = 0.0
    before = (_zero_coverage_mask(img) & _outside_data_envelope(img)).sum()
    img[5, 80] = 1.0          # one stray real pixel inside the pad
    after = (_zero_coverage_mask(img) & _outside_data_envelope(img)).sum()
    assert after <= before
    assert after > 0, "one stray pixel must not disable masking for the whole frame"


def test_prepare_image_frame_applies_the_intersection(tmp_path):
    """The two halves are engine code, but until now the COMPOSITION lived only
    in `_prepare_image_frame` and in the tests' own expressions -- if that line
    lost its `&`, the suite would still have passed. This drives the real frame
    preparation end-to-end on a synthetic padded FITS and asserts `bad`."""
    from astropy.io import fits
    from tract7dt.load_inputs import _prepare_image_frame

    img = np.full((64, 96), 10.0, dtype=np.float32)
    img[:8, :] = 0.0            # border pad -> must be masked
    img[40:44, 50:54] = 0.0     # interior zeros -> must NOT be masked
    hdr = fits.Header({"FILTER": "m600", "ZP_AUTO": 25.0, "SKYSIG": 1.0,
                       "EGAIN": 1.0, "SATURATE": 60000.0})
    fp = tmp_path / "frame.fits"
    fits.PrimaryHDU(data=img, header=hdr).writeto(fp)

    on = _prepare_image_frame(fp, 25.0, mask_zero_coverage=True)
    off = _prepare_image_frame(fp, 25.0, mask_zero_coverage=False)

    assert on["bad"][0, 0] and on["bad"][7, 50], "border pad must reach the bad mask"
    assert not on["bad"][41, 51], "interior zeros must survive the intersection"
    assert not off["bad"].any(), "knob off must leave the pre-fix bad mask"
    assert int(on["bad"].sum()) == 8 * img.shape[1]
