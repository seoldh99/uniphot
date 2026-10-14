"""Island decomposition of a source catalog (island-architecture P1).

Connected components of the source *coupling* graph, plus bucketing of
the resulting islands into a small set of padded (bbox, n_src) classes
suitable for vmapped per-island GPU fitting.

Design reference: the internal design note ``DESIGN_island_arch.md`` (not distributed)
(stages IB + BK). This module is pure host-side NumPy/SciPy -- it is
deliberately import-light (no JAX, no other grid_gpu modules) so it can
be used from analysis tooling without pulling in the GPU stack.

Coupling model
--------------
Each source ``i`` gets a coupling radius

    star     : r_i = r_psf
    extended : r_i = nre(n_i) * Re_i + r_psf

with ``Re_i = clip(exp(logre_i), RE_MIN, RE_MAX)`` and the
profile-dependent wing multiplier

    nre(n) = 4  if n < 1.5      (EXP-like, compact wings)
             6  if n < 3.0
             8  otherwise       (DEV/high-n, extended wings)

Two sources are *coupled* (same island) iff

    dist(i, j) < r_i + r_j

i.e. their model-support disks overlap. Islands are the connected
components of this graph; by construction no pixel receives significant
flux from two different islands, so islands can be fit independently.

Candidate pairs are found with a :class:`scipy.spatial.cKDTree` (radius
query at the bin max radius, exact predicate filter afterwards -- same
machinery as :func:`.forced_phot_sparse.build_neighbor_list`), then
merged via connected components (vectorized union-find equivalent).
Labels are deterministic: islands are numbered in increasing order of
their minimum member index.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# Matches tract7dt.fit.grid_gpu.source_model.SOURCE_STAR; duplicated here
# (single int) to keep this module importable without JAX.
SOURCE_STAR = 0

# Re (pixels) clip range applied to exp(logre) before computing radii.
RE_MIN = 0.3
RE_MAX = 200.0

DEFAULT_BBOX_CLASSES = (64, 96, 128, 192, 256, 384)
DEFAULT_NSRC_CLASSES = (1, 2, 4, 8, 16, 32, 64, 128)


# ---------------------------------------------------------------------------
# Coupling radii
# ---------------------------------------------------------------------------


def nre_multiplier(sersic_n: np.ndarray) -> np.ndarray:
    """Wing-extent multiplier ``nre(n)``: 4 for n<1.5, 6 for n<3, else 8.

    Non-finite n is treated as 1.0 (EXP-like)."""
    n = np.asarray(sersic_n, dtype=np.float64)
    n = np.where(np.isfinite(n), n, 1.0)
    out = np.full(n.shape, 8.0)
    out[n < 3.0] = 6.0
    out[n < 1.5] = 4.0
    return out


def coupling_radii(
    source_type: np.ndarray,
    logre: np.ndarray,
    sersic_n: np.ndarray,
    r_psf: float,
) -> np.ndarray:
    """Per-source coupling radius (pixels).

    Stars: ``r_psf``. Extended: ``nre(n)*Re + r_psf`` with
    ``Re = clip(exp(logre), RE_MIN, RE_MAX)`` (non-finite logre falls
    back to ``RE_MIN``).
    """
    if r_psf <= 0:
        raise ValueError(f"r_psf must be positive, got {r_psf}")
    source_type = np.asarray(source_type)
    logre = np.asarray(logre, dtype=np.float64)
    sersic_n = np.asarray(sersic_n, dtype=np.float64)

    is_star = source_type == SOURCE_STAR
    re = np.exp(logre)
    re = np.where(np.isfinite(re), re, RE_MIN)
    re = np.clip(re, RE_MIN, RE_MAX)
    r_ext = nre_multiplier(sersic_n) * re + float(r_psf)
    return np.where(is_star, float(r_psf), r_ext)


# ---------------------------------------------------------------------------
# Flux-aware coupling radii (P1.5, design D2 revision)
# ---------------------------------------------------------------------------
#
# Geometric stamp-overlap coupling PERCOLATES crowded fields (P1 finding:
# T08285 puts 99.95% of sources in one frame-sized island). The flux-aware
# radius couples i and j only where their light actually competes with the
# sky noise:
#
#     r_i = radius where  flux_i * p_i(r) = kappa * sigma_sky
#
# with p_i the source's UNIT-FLUX radial surface-brightness profile
# (azimuthally-averaged PSF lookup for stars; analytic Sersic falloff for
# extended -- precision uncritical, the radii only shape the graph).
# Radii are floored at r_core (~PSF FWHM: nothing decouples inside the
# core) and capped at the old geometric radius.

# Exact b_1 for a Sersic n=1 (EXP) profile; for n >= 1.5 the crude
# b_n ~ 2n - 1/3 asymptote is plenty here.
SERSIC_B1 = 1.6783469900166605


def psf_radial_profile(
    stamp: np.ndarray, bin_width: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Monotone unit-flux radial profile from a (centered) PSF stamp.

    Returns ``(r, p)``: annulus-mean surface brightness ``p`` (per pixel,
    for a total flux of 1) at radii ``r`` (pixels), normalized so the
    stamp sums to 1, floored at 1e-12 and forced non-increasing (running
    min) so it can be inverted.
    """
    stamp = np.asarray(stamp, dtype=np.float64)
    total = stamp.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("psf_radial_profile: stamp must have positive sum")
    stamp = stamp / total
    H, W = stamp.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot(yy - cy, xx - cx).ravel()
    nbin = int(np.ceil(rr.max() / bin_width)) + 1
    idx = np.minimum((rr / bin_width).astype(np.int64), nbin - 1)
    sums = np.bincount(idx, weights=stamp.ravel(), minlength=nbin)
    cnts = np.bincount(idx, minlength=nbin)
    ok = cnts > 0
    r = (np.nonzero(ok)[0] + 0.5) * bin_width
    p = sums[ok] / cnts[ok]
    p = np.maximum(p, 1e-12)
    p = np.minimum.accumulate(p)
    # anchor r=0 at the central (max) value
    return np.concatenate([[0.0], r]), np.concatenate([[p[0]], p])


def _invert_monotone_profile(
    psf_r: np.ndarray, psf_p: np.ndarray, thresh: np.ndarray
) -> np.ndarray:
    """Radius where the monotone profile drops to ``thresh`` (vectorized).

    thresh >= p(0) -> 0; thresh <= p(r_max) (incl. non-positive) -> r_max.
    Interpolates linearly in log(p).
    """
    lp = np.log(psf_p[::-1])          # increasing
    rr = psf_r[::-1]                  # decreasing radius
    t = np.asarray(thresh, dtype=np.float64)
    out = np.full(t.shape, psf_r[-1], dtype=np.float64)
    pos = np.isfinite(t) & (t > 0)
    tc = np.clip(np.log(np.where(pos, t, 1.0)), lp[0], lp[-1])
    out[pos] = np.interp(tc, lp, rr)[pos]
    # thresh above the central value -> inside the core -> radius 0
    out[np.isfinite(t) & (t >= psf_p[0])] = 0.0
    # thresh = +inf (no flux) -> radius 0
    out[np.isinf(t) & (t > 0)] = 0.0
    return out


def _sersic_b(n_eff: np.ndarray) -> np.ndarray:
    return np.where(n_eff < 1.5, SERSIC_B1, 2.0 * n_eff - 1.0 / 3.0)


def _sersic_central(re: np.ndarray, n_eff: np.ndarray) -> np.ndarray:
    """Central surface brightness I0 of a unit-total-flux Sersic profile
    ``I(r) = I0 * exp(-b * (r/Re)^(1/n))`` (per design: EXP profile for
    n < 1.5, crude n-scaled falloff otherwise)."""
    from scipy.special import gamma

    b = _sersic_b(n_eff)
    return b ** (2.0 * n_eff) / (2.0 * np.pi * n_eff * gamma(2.0 * n_eff) * re ** 2)


def _ext_shape(logre: np.ndarray, sersic_n: np.ndarray):
    """Sanitized (Re, n_eff, b, I0) for the extended-source approximation."""
    re = np.clip(np.where(np.isfinite(logre), np.exp(logre), RE_MIN), RE_MIN, RE_MAX)
    n = np.where(np.isfinite(sersic_n), np.asarray(sersic_n, np.float64), 1.0)
    n_eff = np.where(n < 1.5, 1.0, n)
    b = _sersic_b(n_eff)
    i0 = _sersic_central(re, n_eff)
    return re, n_eff, b, i0


def unit_profile_value(
    source_type: np.ndarray,
    logre: np.ndarray,
    sersic_n: np.ndarray,
    r: np.ndarray,
    psf_r: np.ndarray,
    psf_p: np.ndarray,
) -> np.ndarray:
    """Unit-flux surface brightness of each source at radius ``r`` (per
    source). Stars: PSF lookup (0 beyond the stamp). Extended: analytic
    Sersic approximation (unconvolved -- slightly overestimates the peak,
    conservative for coupling/audit purposes)."""
    source_type = np.asarray(source_type)
    r = np.asarray(r, dtype=np.float64)
    is_star = source_type == SOURCE_STAR

    # log-space interpolation: the exact inverse of the polyline used by
    # _invert_monotone_profile (and more physical for ~exponential wings);
    # 0 beyond the stamp edge.
    p_star = np.exp(np.interp(r, psf_r, np.log(psf_p)))
    p_star = np.where(r > psf_r[-1], 0.0, p_star)

    re, n_eff, b, i0 = _ext_shape(np.asarray(logre, np.float64),
                                  np.asarray(sersic_n, np.float64))
    p_ext = i0 * np.exp(-b * (r / re) ** (1.0 / n_eff))
    return np.where(is_star, p_star, p_ext)


def flux_aware_radii(
    source_type: np.ndarray,
    logre: np.ndarray,
    sersic_n: np.ndarray,
    flux: np.ndarray,
    sigma_sky: float,
    kappa: float,
    psf_r: np.ndarray,
    psf_p: np.ndarray,
    r_psf: float,
    r_core: float | None = None,
) -> np.ndarray:
    """Flux-aware per-source coupling radius (pixels).

    Solves ``flux_i * p_i(r) = kappa * sigma_sky`` for each source
    (see module notes above), then floors at ``r_core`` (default:
    PSF FWHM from the lookup profile) and caps at the geometric radius
    (:func:`coupling_radii`): stars at ``r_psf``, extended at
    ``nre(n)*Re + r_psf``. Non-finite / non-positive fluxes get
    ``r_core`` (an undetected source's wings are sub-noise everywhere).

    Parameters beyond :func:`coupling_radii`:

    flux : (n,) float
        Per-source scalar flux in the same units as ``sigma_sky``
        (e.g. median over bands of the S1 fitted fluxes).
    sigma_sky : float
        Per-pixel sky noise in flux units.
    kappa : float
        Coupling threshold in units of sigma_sky (~0.5-1).
    psf_r, psf_p : arrays
        Monotone unit-flux PSF radial profile from
        :func:`psf_radial_profile`.
    r_core : float, optional
        Floor radius; default = PSF FWHM (2 x HWHM of the lookup).
    """
    if sigma_sky <= 0 or not np.isfinite(sigma_sky):
        raise ValueError(f"sigma_sky must be positive, got {sigma_sky}")
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    source_type = np.asarray(source_type)
    logre = np.asarray(logre, dtype=np.float64)
    sersic_n = np.asarray(sersic_n, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    if r_core is None:
        hwhm = float(_invert_monotone_profile(psf_r, psf_p,
                                              np.array([0.5 * psf_p[0]]))[0])
        r_core = 2.0 * hwhm
    if r_core <= 0:
        raise ValueError(f"r_core must be positive, got {r_core}")

    good = np.isfinite(flux) & (flux > 0)
    thresh = np.where(good, kappa * float(sigma_sky) / np.where(good, flux, 1.0),
                      np.inf)

    # stars: invert the PSF lookup
    r_star = _invert_monotone_profile(psf_r, psf_p, thresh)

    # extended: invert the analytic Sersic falloff
    re, n_eff, b, i0 = _ext_shape(logre, sersic_n)
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = np.log(i0 / thresh)  # thresh=inf -> arg=-inf -> radius 0
    r_ext = np.where(arg > 0, re * (np.maximum(arg, 0.0) / b) ** n_eff, 0.0)

    geo_cap = coupling_radii(source_type, logre, sersic_n, r_psf)
    is_star = source_type == SOURCE_STAR
    r = np.where(is_star, r_star, r_ext)
    return np.clip(r, r_core, geo_cap)


# ---------------------------------------------------------------------------
# Island build (connected components of the coupling graph)
# ---------------------------------------------------------------------------


@dataclass
class IslandSet:
    """Result of :func:`build_islands`.

    Attributes
    ----------
    n_sources : int
        Total number of catalog sources.
    n_islands : int
        Number of islands (== ``len(members)``).
    labels : (n_sources,) int32
        Island id of each source. Deterministic: island ``k`` has the
        ``k``-th smallest minimum member index.
    radius : (n_sources,) float64
        Coupling radius used for each source.
    members : list of int32 arrays
        ``members[k]`` = sorted (ascending) source indices of island k.
    n_src : (n_islands,) int32
        Member count per island.
    bbox : (n_islands, 4) int64
        Half-open integer bbox ``(x0, y0, x1, y1)`` per island:
        ``x0 = floor(min_i(x_i - r_i))``, ``x1 = ceil(max_i(x_i + r_i))``
        (likewise y), so every member's coupling disk fits inside
        ``[x0, x1) x [y0, y1)`` (up to the disk boundary itself).
    r_psf : float
        The PSF coupling radius the set was built with.
    """

    n_sources: int
    n_islands: int
    labels: np.ndarray
    radius: np.ndarray
    members: list = field(repr=False)
    n_src: np.ndarray = field(repr=False)
    bbox: np.ndarray = field(repr=False)
    r_psf: float = 0.0

    @property
    def bbox_w(self) -> np.ndarray:
        return self.bbox[:, 2] - self.bbox[:, 0]

    @property
    def bbox_h(self) -> np.ndarray:
        return self.bbox[:, 3] - self.bbox[:, 1]

    @property
    def bbox_size(self) -> np.ndarray:
        """max(width, height) per island -- the square-class size."""
        return np.maximum(self.bbox_w, self.bbox_h)


def _candidate_coupled_pairs(
    pos_x: np.ndarray, pos_y: np.ndarray, radius: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """All (i, j) i<j with ``dist(i,j) < r_i + r_j``.

    Sources are binned by radius (log2 bins, so bin_rmax/bin_rmin <= 2);
    candidate pairs come from cKDTree radius queries at the *bin max*
    radius sum, then the exact per-pair predicate is applied. This keeps
    the query radius tight for the (dominant) star population even when
    a few large galaxies inflate the global max radius.
    """
    n = pos_x.shape[0]
    if n < 2:
        return (np.empty(0, np.int64), np.empty(0, np.int64))

    r_min = float(radius.min())
    # log2 bin per source (radius/r_min in [2^b, 2^(b+1)))
    bin_id = np.floor(np.log2(radius / r_min)).astype(np.int64)
    bins = np.unique(bin_id)

    trees: dict[int, tuple[cKDTree, np.ndarray, float]] = {}
    for b in bins:
        sel = np.where(bin_id == b)[0]
        xy = np.column_stack([pos_x[sel], pos_y[sel]])
        trees[b] = (cKDTree(xy), sel, float(radius[sel].max()))

    pi_parts: list[np.ndarray] = []
    pj_parts: list[np.ndarray] = []

    def _exact_filter(gi: np.ndarray, gj: np.ndarray) -> None:
        if gi.size == 0:
            return
        d2 = (pos_x[gi] - pos_x[gj]) ** 2 + (pos_y[gi] - pos_y[gj]) ** 2
        rsum = radius[gi] + radius[gj]
        keep = d2 < rsum * rsum
        pi_parts.append(gi[keep])
        pj_parts.append(gj[keep])

    for ia, a in enumerate(bins):
        tree_a, idx_a, rmax_a = trees[a]
        # same-bin pairs
        pairs = tree_a.query_pairs(r=2.0 * rmax_a, output_type="ndarray")
        if pairs.size:
            _exact_filter(idx_a[pairs[:, 0]], idx_a[pairs[:, 1]])
        # cross-bin pairs
        for b in bins[ia + 1:]:
            tree_b, idx_b, rmax_b = trees[b]
            lists = tree_a.query_ball_tree(tree_b, r=rmax_a + rmax_b)
            la = np.fromiter(
                (len(l) for l in lists), count=len(lists), dtype=np.int64
            )
            if la.sum() == 0:
                continue
            gi = np.repeat(idx_a, la)
            gj = idx_b[np.concatenate([np.asarray(l, np.int64) for l in lists if l])]
            _exact_filter(gi, gj)

    if not pi_parts:
        return (np.empty(0, np.int64), np.empty(0, np.int64))
    return np.concatenate(pi_parts), np.concatenate(pj_parts)


def build_islands(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    source_type: np.ndarray | None,
    logre: np.ndarray | None,
    sersic_n: np.ndarray | None,
    r_psf: float,
    *,
    radius: np.ndarray | None = None,
    forced_singleton: np.ndarray | None = None,
) -> IslandSet:
    """Decompose a catalog into coupling islands.

    Parameters
    ----------
    pos_x, pos_y : (n,) float
        Source positions (pixels, any common frame). Must be finite.
    source_type : (n,) int or None
        Integer type codes (``source_model.SOURCE_*``). None = all stars.
    logre : (n,) float or None
        ln(Re/pixels) for extended sources (ignored for stars).
        None = all zeros.
    sersic_n : (n,) float or None
        Sersic index for extended sources. None = all ones.
    r_psf : float
        PSF coupling radius in pixels (e.g. 27.5 for a 55px stamp).
    radius : (n,) float, optional
        Precomputed per-source coupling radii (e.g. from
        :func:`flux_aware_radii`). When given, overrides the geometric
        :func:`coupling_radii` computation; must be finite and positive.
    forced_singleton : (n,) bool, optional
        Rows that must form their OWN single-member island: every
        coupling edge touching such a row is dropped before the
        connected-components labeling (used by the Exp C v2 phantom
        subtract-only sources -- they participate in the S1 solves and
        the scene-subtract field pass but may never joint-fit with, or
        merge into, a real source's island). None => unchanged.

    Returns
    -------
    IslandSet
    """
    pos_x = np.asarray(pos_x, dtype=np.float64)
    pos_y = np.asarray(pos_y, dtype=np.float64)
    n = pos_x.shape[0]
    if pos_y.shape[0] != n:
        raise ValueError("pos_x and pos_y must have the same length")
    if not (np.all(np.isfinite(pos_x)) and np.all(np.isfinite(pos_y))):
        raise ValueError(
            "build_islands requires finite positions; filter the catalog first"
        )

    if source_type is None:
        source_type = np.zeros(n, dtype=np.int32)
    if logre is None:
        logre = np.zeros(n, dtype=np.float64)
    if sersic_n is None:
        sersic_n = np.ones(n, dtype=np.float64)

    if radius is not None:
        radius = np.asarray(radius, dtype=np.float64)
        if radius.shape[0] != n:
            raise ValueError("radius must have the same length as pos_x")
        if n and not (np.all(np.isfinite(radius)) and np.all(radius > 0)):
            raise ValueError("radius override must be finite and positive")
    else:
        radius = coupling_radii(source_type, logre, sersic_n, r_psf)

    if n == 0:
        return IslandSet(
            n_sources=0, n_islands=0,
            labels=np.empty(0, np.int32), radius=radius,
            members=[], n_src=np.empty(0, np.int32),
            bbox=np.empty((0, 4), np.int64), r_psf=float(r_psf),
        )

    pi, pj = _candidate_coupled_pairs(pos_x, pos_y, radius)

    if forced_singleton is not None and pi.size:
        _fs = np.asarray(forced_singleton, dtype=bool)
        if _fs.shape[0] != n:
            raise ValueError(
                "forced_singleton must have the same length as pos_x")
        _keep = ~(_fs[pi] | _fs[pj])
        pi, pj = pi[_keep], pj[_keep]

    # Connected components (vectorized union-find).
    if pi.size:
        graph = sparse.coo_matrix(
            (np.ones(pi.size, dtype=np.int8), (pi, pj)), shape=(n, n)
        )
        _, raw_labels = connected_components(graph, directed=False)
    else:
        raw_labels = np.arange(n, dtype=np.int64)

    # Deterministic relabel: order islands by minimum member index.
    n_islands = int(raw_labels.max()) + 1
    min_member = np.full(n_islands, n, dtype=np.int64)
    np.minimum.at(min_member, raw_labels, np.arange(n, dtype=np.int64))
    remap = np.empty(n_islands, dtype=np.int64)
    remap[np.argsort(min_member, kind="stable")] = np.arange(n_islands)
    labels = remap[raw_labels].astype(np.int32)

    # Member lists (ascending within each island).
    order = np.argsort(labels, kind="stable")
    counts = np.bincount(labels, minlength=n_islands).astype(np.int32)
    starts = np.zeros(n_islands, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    members = [
        order[starts[k]: starts[k] + counts[k]].astype(np.int32)
        for k in range(n_islands)
    ]

    # Integer bboxes over member coupling disks (half-open).
    x_lo = (pos_x - radius)[order]
    x_hi = (pos_x + radius)[order]
    y_lo = (pos_y - radius)[order]
    y_hi = (pos_y + radius)[order]
    bbox = np.empty((n_islands, 4), dtype=np.int64)
    bbox[:, 0] = np.floor(np.minimum.reduceat(x_lo, starts))
    bbox[:, 1] = np.floor(np.minimum.reduceat(y_lo, starts))
    bbox[:, 2] = np.ceil(np.maximum.reduceat(x_hi, starts))
    bbox[:, 3] = np.ceil(np.maximum.reduceat(y_hi, starts))

    return IslandSet(
        n_sources=n, n_islands=n_islands, labels=labels, radius=radius,
        members=members, n_src=counts, bbox=bbox, r_psf=float(r_psf),
    )


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------


@dataclass
class IslandBucket:
    """All islands assigned to one (bbox_class, nsrc_class) cell.

    ``member_idx`` is the padded member matrix: shape
    ``(n_islands_in_bucket, nsrc_class)``, int32, ``-1`` in padding
    slots; ``member_mask`` is True where the slot is a real member.
    """

    bbox_class: int
    nsrc_class: int
    island_ids: np.ndarray            # (n_b,) int32, ascending
    member_idx: np.ndarray = field(repr=False)   # (n_b, nsrc_class) int32
    member_mask: np.ndarray = field(repr=False)  # (n_b, nsrc_class) bool
    bbox: np.ndarray = field(repr=False)         # (n_b, 4) int64

    @property
    def n_islands(self) -> int:
        return int(self.island_ids.shape[0])

    @property
    def n_members(self) -> int:
        return int(self.member_mask.sum())

    @property
    def n_padded_slots(self) -> int:
        return int(self.member_idx.size)


@dataclass
class BucketedIslands:
    """Result of :func:`bucket_islands`.

    ``buckets`` maps ``(bbox_class, nsrc_class)`` -> :class:`IslandBucket`
    (only occupied cells are present). ``giant_ids`` lists islands that
    exceed the largest class in *either* dimension -- these must be
    handled by the giant-island policy (design D2), not the vmapped
    bucket path.
    """

    bbox_classes: tuple
    nsrc_classes: tuple
    buckets: dict
    giant_ids: np.ndarray

    @property
    def n_giant(self) -> int:
        return int(self.giant_ids.shape[0])

    def padded_overhead(self) -> float:
        """sum(padded member slots) / sum(actual members), buckets only."""
        slots = sum(b.n_padded_slots for b in self.buckets.values())
        actual = sum(b.n_members for b in self.buckets.values())
        return slots / actual if actual else float("nan")


def bucket_islands(
    island_set: IslandSet,
    bbox_classes: tuple = DEFAULT_BBOX_CLASSES,
    nsrc_classes: tuple = DEFAULT_NSRC_CLASSES,
) -> BucketedIslands:
    """Assign each island to the smallest adequate (bbox, nsrc) class.

    Island size is ``max(bbox_w, bbox_h)`` (square classes). An island
    whose size exceeds ``max(bbox_classes)`` or whose member count
    exceeds ``max(nsrc_classes)`` goes to the giant list.
    """
    bbox_classes = tuple(sorted(int(c) for c in bbox_classes))
    nsrc_classes = tuple(sorted(int(c) for c in nsrc_classes))

    sizes = island_set.bbox_size
    n_src = island_set.n_src

    bc_idx = np.searchsorted(np.asarray(bbox_classes), sizes, side="left")
    nc_idx = np.searchsorted(np.asarray(nsrc_classes), n_src, side="left")
    is_giant = (bc_idx >= len(bbox_classes)) | (nc_idx >= len(nsrc_classes))

    giant_ids = np.where(is_giant)[0].astype(np.int32)

    buckets: dict = {}
    ok = ~is_giant
    if np.any(ok):
        keys = bc_idx[ok] * len(nsrc_classes) + nc_idx[ok]
        island_ids_ok = np.where(ok)[0]
        for key in np.unique(keys):
            sel = island_ids_ok[keys == key]  # ascending island ids
            bc = bbox_classes[int(key) // len(nsrc_classes)]
            nc = nsrc_classes[int(key) % len(nsrc_classes)]
            n_b = sel.shape[0]
            member_idx = np.full((n_b, nc), -1, dtype=np.int32)
            member_mask = np.zeros((n_b, nc), dtype=bool)
            for row, isl in enumerate(sel):
                m = island_set.members[isl]
                member_idx[row, : m.shape[0]] = m
                member_mask[row, : m.shape[0]] = True
            buckets[(bc, nc)] = IslandBucket(
                bbox_class=bc, nsrc_class=nc,
                island_ids=sel.astype(np.int32),
                member_idx=member_idx, member_mask=member_mask,
                bbox=island_set.bbox[sel],
            )

    return BucketedIslands(
        bbox_classes=bbox_classes, nsrc_classes=nsrc_classes,
        buckets=buckets, giant_ids=giant_ids,
    )


__all__ = [
    "SOURCE_STAR",
    "RE_MIN",
    "RE_MAX",
    "SERSIC_B1",
    "DEFAULT_BBOX_CLASSES",
    "DEFAULT_NSRC_CLASSES",
    "nre_multiplier",
    "coupling_radii",
    "psf_radial_profile",
    "unit_profile_value",
    "flux_aware_radii",
    "IslandSet",
    "build_islands",
    "IslandBucket",
    "BucketedIslands",
    "bucket_islands",
]
