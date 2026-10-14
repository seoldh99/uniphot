"""Locally adaptive spatial PSF model of one image (M2 unit a.1).

Design: the internal design notes of this model (not distributed): the model, no wing
surgery, local stacking, heterogeneity floor, D/16 candidates, render spacing, memory and
BLAS threads.

Model
-----
For one image (today the band coadd, from M3-2 a frame) the PSF at work-frame position
r = (x, y) is a 55 x 55 stamp P(r), the PSF of a source centred exactly on the stamp's
central pixel, normalised so that its pixels sum to 1 (box-flux convention).

- Candidates m = (h, p): Gaussian bandwidth h in px and polynomial order p.
  Default (16): h in {inf, D/2, D/4, D/8, D/16} (D = image diagonal, hypot(W, H)); for finite h,
  p in {0, 1, 2} (local polynomial in (r - r_k) / h); for h = inf, p in {2, 3, 4, 5}
  (global polynomial, Legendre products in xh = 2x/(W-1) - 1, yh = 2y/(H-1) - 1).
  The set is a ratio of the image size, not a constant of any telescope or tile.
- A finite-h candidate has a node grid r_k (spacing <= h/2, see ``training_grid``).
  At each node the stamp pixels are a polynomial Q_k(r) fitted by weighted least
  squares with ONE weight per star, w_ks = exp(-|r_s - r_k|^2 / (2 h^2)) * omega_s.
  Because the weight does not depend on the pixel, Q_k(r) is a linear combination
  sum_s lambda_s(r) d_s of the training stamps with sum_s lambda_s = 1, so every Q_k sums to
  exactly 1 wherever it is evaluated (also outside the stars' hull).
  P_m(r) = sum_k beta_k(r) Q_k(r), beta = bilinear partition of unity on the node grid.
  The h = inf candidate is the same with a single global polynomial.
- Star weight (v2.5 2.3; unit a.1b): omega_s = 1 / (lambda a_s + tau^2); a_s = expected squared
  norm of the noise of the normalised stamp (given by the caller), lambda a scale of a and tau^2
  the heterogeneity floor, both from the mixed model's held-out residual norms r2_s by TGM
  (:func:`tgm_floor`, with the lambda identification check).  Two fixed passes (weights 1 / a,
  then (lambda, tau^2)) unless the caller passes the pair (lambda, tau^2).
- Support (design 1.4): at a node whose effective star count
  N_eff = (sum_s w_ks)^2 / sum_s w_ks^2 is below 3 x (number of terms of order p), that
  candidate's node uses p = 0 (the local weighted mean; no extrapolation).
- Mixing = local stacking (v2.5 1.3): K = 5 folds stratified in space; every candidate is
  fitted without each fold and its held-out residuals e_ms give per-star Grams
  g_s[m, n] = <e_ms, e_ns>.  At each mixing node q_j (the D/8 training grid) the weights are
  w_j = argmin_{w >= 0, sum w = 1} w' (G_j + 1e-10 tr(G_j)/M I) w with
  G_j = sum_s exp(-|r_s - q_j|^2 / (2 g^2)) omega_s g_s, g = D/8 (exact active-set solution;
  exact duplicate candidates share one weight equally); w(r) is bilinear between mixing nodes and
  P(r) = sum_m w_m(r) P_m(r), every candidate refitted on all stars.  Each P_m sums to 1 and
  w(r) does, so P does too.
- Render: P is evaluated on a regular render node grid = the finest training grid refined by
  the coarsest integer n_b with max b <= 1e-3 (D9-1; b = the a.2 bilinear flux bound), and
  consumers interpolate bilinearly between render nodes; every rendered stamp sums to 1.  The
  integral is asserted when the model is built: max_k |sum(stamp_k) - 1| <= 1e-6 at every render
  node, else :class:`PsfModelIntegralError` (a run failure with ``error.kind`` "psf_model_integral",
  never a warning; plan D9-3).

Side-car format ``psf_model/<image_id>/`` (format "uniphot.psf_model", version 1)
---------------------------------------------------------------------------------
Written by :func:`save_psf_model`, read by :func:`load_psf_model`; the public numpy-only
reader for measurement tools is unit a.7 (``tract7dt/psf_access.py``), which must follow
this text.

``<work_dir>/psf_model/index.csv``
    One row per image of the run: ``image_id,band,frame``.  Consumers map a band or a
    frame to its image id here; they do not rebuild the id.
``<work_dir>/psf_model/<image_id>/model.json``  (manifest, UTF-8 JSON)
    format = "uniphot.psf_model", format_version = 1, image_id, band, frame (path of
    the source frame), image_shape [H, W] of the work frame (the cropped frame the
    engine fits; ``work_frame_offset`` [x0, y0] is its origin in the source frame, so
    source-frame x = work x + x0), stamp_size S (55), stamp_centre [S//2, S//2],
    normalisation "sum_1", render {interpolation "bilinear", outside "clamp",
    nx, ny, spacing_x, spacing_y}, integral_max_abs_dev (max |sum - 1| over render
    nodes, float64), n_folds, fold_sizes, n_stars, n_eff_stars, n_eff_stars_no_floor, tau2,
    tau2_over_median_a, median_a, lam, theta (= tau2 / lam), a_min_over_theta, ms_iters, n_B, n_F, N50,
    ms_global_gap, tau2_zero, tau2_fallback, B_excess, lambda_identified, lam_hat (the step-2 lambda,
    kept when it is rejected), F_median_r2_over_a, tau2_rule (unit a.1b: the TGM estimate; null
    where tau2 was given; lambda_identified/lam_hat/F_median_r2_over_a also null below 30 stars), passes, tau2_given,
    mix {rule, nodes_h_over_diag, kernel_g_over_diag,
    nx, ny, ridge, mean_active}, render {refine, b_max, b_max_by_refine, bytes_float32, rule},
    candidates (list, one entry per candidate: name, h_px (null = inf), h_over_diag, p, loss,
    loss_folds, n_nodes, n_nodes_p0 (nodes switched to p = 0 by the support rule),
    n_nodes_singular, n_nodes_unreached, max_mix_weight), timing_s, created.
``<work_dir>/psf_model/<image_id>/model.npz``
    node_x (nx,) and node_y (ny,) float64 work-frame pixel positions of the render nodes
    (regular: node_x = linspace(0, W - 1, nx), same for y); stamps (ny, nx, S, S) float64,
    stamps[j, i] = P(node_x[i], node_y[j]); star_x, star_y, star_weight (final omega), star_a,
    star_r2 (pass-a held-out residual energy; empty when tau2 was given) (N,) float64 and
    star_fold (N,) int64 of the training stars; mix_node_x (nx_m,),
    mix_node_y (ny_m,) and mix_weights (ny_m, nx_m, M) of the mixing grid (candidate order of the
    manifest's ``candidates``).  Round 2 (2026-10-10) added the star_a / star_u / mix_* arrays and
    the manifest's tau2 / passes / mix / render.refine fields, and removed the per-candidate
    se / weight entries (weights are local now).  Unit a.3 rounds 2-3 added a Huber weight
    (star_u, star_r2, star_vbar, manifest huber_*) and round 4 removed it again (design v2.5
    3rd ed. 7.1); star_u, which a.1 round 2 had added for a caller-given u, went with it.  Unit
    a.1b added star_r2 (again; now the TGM input) and the TGM fields.  Still format_version 1 (no
    reader existed; a missing star_r2 reads as an empty array).
Rendering rule (the contract of every consumer): clamp x to [node_x[0], node_x[-1]] and y
likewise, find the render cell (i, j) holding (x, y), t = (x - node_x[i]) / dx,
u = (y - node_y[j]) / dy, and P = (1-t)(1-u) S[j,i] + t(1-u) S[j,i+1] + (1-t)u S[j+1,i]
+ t u S[j+1,i+1].  The stamp is centred on its central pixel; placing a source at a
sub-pixel position (the engine's Lanczos-3 shift) is the consumer's job.
Names reserved in the same folder: ``psf_error_map.npz`` (unit a.5, PSF error term),
``stars_centered_stamps.npz``, ``stars.csv``, ``psf_residual.png`` (diagnostics).

Image id
--------
:func:`image_id_for` = ``<band>.<frame file name without .fits/.fit/.fts[.fz|.gz]>``
with characters outside [A-Za-z0-9._+-] replaced by "_".  Two images of one run that map
to the same id are an error (``check_unique_image_ids``); one image per band today.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from ...run_status import RunContractError

logger = logging.getLogger("tract7dt.fit.grid_gpu.psf_model")

FORMAT_NAME = "uniphot.psf_model"
FORMAT_VERSION = 1
INTEGRAL_TOL = 1e-6
INTEGRAL_ERROR_KIND = "psf_model_integral"
N_FOLDS = 5
INDEX_NAME = "index.csv"


class PsfModelIntegralError(RunContractError):
    """A built PSF model whose render-node stamps do not sum to 1 (a bug, D9-3)."""

    def __init__(self, message: str, **details):
        super().__init__(INTEGRAL_ERROR_KIND, message, **details)

    def __reduce__(self):  # picklable across spawn workers
        return (_rebuild_integral_error, (self.message, self.details))


def _rebuild_integral_error(message, details):
    return PsfModelIntegralError(message, **details)


# ---------------------------------------------------------------------------
# Image id
# ---------------------------------------------------------------------------

_FITS_SUFFIX = re.compile(r"\.(fits|fit|fts)(\.(fz|gz|bz2))?$", re.IGNORECASE)
_UNSAFE = re.compile(r"[^A-Za-z0-9._+-]")


def image_id_for(band: str, frame) -> str:
    """Image id of one image: ``<band>.<frame file name without the FITS suffix>``."""
    name = _FITS_SUFFIX.sub("", Path(str(frame)).name)
    return _UNSAFE.sub("_", f"{band}.{name}")


def check_unique_image_ids(ids: Sequence[str]) -> None:
    """Raise ValueError when two images of one run share an id."""
    seen: dict[str, int] = {}
    for i in ids:
        seen[i] = seen.get(i, 0) + 1
    dup = sorted(k for k, v in seen.items() if v > 1)
    if dup:
        raise ValueError(f"two images map to the same psf_model image_id: {dup} "
                         "(rename one of the frames)")


# ---------------------------------------------------------------------------
# Candidates and grids
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One spatial model: Gaussian bandwidth h (px, math.inf = global) and order p."""
    h: float
    p: int

    @property
    def is_global(self) -> bool:
        return not math.isfinite(self.h)


def image_diagonal(image_shape) -> float:
    h, w = int(image_shape[0]), int(image_shape[1])
    return float(math.hypot(w, h))


def default_candidates(image_shape) -> tuple[Candidate, ...]:
    """The 16 design v2.5 candidates (1.3 step 1): 4 global orders + 4 bandwidths x 3 orders."""
    d = image_diagonal(image_shape)
    out = [Candidate(math.inf, p) for p in (2, 3, 4, 5)]
    for frac in (2.0, 4.0, 8.0, 16.0):
        out += [Candidate(d / frac, p) for p in (0, 1, 2)]
    return tuple(out)


def _base_cells(image_shape) -> tuple[int, int]:
    """Cell counts (cx, cy) of the coarsest default training grid (spacing <= D/4)."""
    h, w = int(image_shape[0]), int(image_shape[1])
    d = image_diagonal(image_shape)
    return (max(1, math.ceil((w - 1) / (d / 4.0))), max(1, math.ceil((h - 1) / (d / 4.0))))


def refine_factor(h: float, image_shape) -> int:
    """Training-grid refinement f = ceil(D / (2 h)) of the base grid: spacing <= h / 2."""
    return max(1, math.ceil(image_diagonal(image_shape) / (2.0 * float(h))))


def training_grid(h: float, image_shape) -> tuple[np.ndarray, np.ndarray]:
    """Node positions (node_x, node_y) of a finite-h candidate.

    The base grid has (cx, cy) cells of size <= D/4 over [0, W-1] x [0, H-1]; candidate h
    refines it by f = ceil(D / (2h)), so its spacing is <= h/2 and the default bandwidths
    (D/2, D/4, D/8 -> f = 1, 2, 4) give nested grids.
    """
    hh, ww = int(image_shape[0]), int(image_shape[1])
    cx, cy = _base_cells(image_shape)
    f = refine_factor(h, image_shape)
    return (np.linspace(0.0, ww - 1.0, cx * f + 1), np.linspace(0.0, hh - 1.0, cy * f + 1))


def render_grid(candidates: Sequence[Candidate], image_shape,
                refine: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Render nodes: the finest training grid among the candidates refined by ``refine``.

    Every training grid must divide the finest one (nested), so the model is smooth inside
    each render cell (ValueError otherwise).
    """
    hh, ww = int(image_shape[0]), int(image_shape[1])
    cx, cy = _base_cells(image_shape)
    fs = [refine_factor(c.h, image_shape) for c in candidates if not c.is_global]
    fmax = max(fs) if fs else 1
    if any(fmax % f for f in fs):
        raise ValueError(f"training grids are not nested (refinement factors {sorted(set(fs))})")
    n = int(refine)
    if n < 1:
        raise ValueError(f"render refine must be >= 1, got {refine!r}")
    return (np.linspace(0.0, ww - 1.0, cx * fmax * n + 1),
            np.linspace(0.0, hh - 1.0, cy * fmax * n + 1))


# ---------------------------------------------------------------------------
# The model object, rendering, side-car I/O
# ---------------------------------------------------------------------------


@dataclass
class PsfModel:
    """A built PSF model of one image: render-node stamps + the build record."""
    image_id: str
    band: str
    frame: str
    image_shape: tuple
    node_x: np.ndarray
    node_y: np.ndarray
    stamps: np.ndarray            # (ny, nx, S, S) float64, each sums to 1
    star_x: np.ndarray
    star_y: np.ndarray
    star_weight: np.ndarray
    star_fold: np.ndarray
    record: dict = field(default_factory=dict)
    work_frame_offset: tuple = (0, 0)
    mix_node_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mix_node_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mix_weights: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    star_a: np.ndarray = field(default_factory=lambda: np.zeros(0))
    star_r2: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def stamp_size(self) -> int:
        return int(self.stamps.shape[-1])

    def render(self, x, y) -> np.ndarray:
        """Stamp(s) at work-frame position(s) by the format's bilinear rule.

        Scalars give (S, S); arrays of shape (n,) give (n, S, S).
        """
        return render_bilinear(self.node_x, self.node_y, self.stamps, x, y)


def _cell(nodes: np.ndarray, v: np.ndarray):
    """Cell index i and fraction t of positions v on a regular node axis (clamped)."""
    n = len(nodes)
    v = np.clip(np.asarray(v, dtype=np.float64), nodes[0], nodes[-1])
    if n == 1:
        return np.zeros(v.shape, dtype=np.int64), np.zeros(v.shape)
    step = (nodes[-1] - nodes[0]) / (n - 1)
    i = np.minimum(np.floor((v - nodes[0]) / step).astype(np.int64), n - 2)
    t = (v - nodes[i]) / (nodes[i + 1] - nodes[i])
    return i, np.clip(t, 0.0, 1.0)


def render_bilinear(node_x, node_y, stamps, x, y) -> np.ndarray:
    """Bilinear interpolation of node stamps (the format's rendering rule)."""
    scalar = np.ndim(x) == 0 and np.ndim(y) == 0
    xs = np.atleast_1d(np.asarray(x, dtype=np.float64))
    ys = np.atleast_1d(np.asarray(y, dtype=np.float64))
    i, t = _cell(np.asarray(node_x, dtype=np.float64), xs)
    j, u = _cell(np.asarray(node_y, dtype=np.float64), ys)
    nx = stamps.shape[1]
    if nx == 1:
        i1 = i
    else:
        i1 = i + 1
    ny = stamps.shape[0]
    j1 = j if ny == 1 else j + 1
    t = t[:, None, None]
    u = u[:, None, None]
    out = ((1.0 - t) * (1.0 - u) * stamps[j, i] + t * (1.0 - u) * stamps[j, i1]
           + (1.0 - t) * u * stamps[j1, i] + t * u * stamps[j1, i1])
    return out[0] if scalar else out


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    return v


def save_psf_model(psf_root, model: PsfModel) -> Path:
    """Write ``<psf_root>/<image_id>/model.json`` + ``model.npz``; returns the folder."""
    d = Path(psf_root) / model.image_id
    d.mkdir(parents=True, exist_ok=True)
    s = model.stamp_size
    rec = dict(model.record)
    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "image_id": model.image_id,
        "band": model.band,
        "frame": str(model.frame),
        "image_shape": [int(model.image_shape[0]), int(model.image_shape[1])],
        "work_frame_offset": [int(model.work_frame_offset[0]), int(model.work_frame_offset[1])],
        "stamp_size": s,
        "stamp_centre": [s // 2, s // 2],
        "normalisation": "sum_1",
        "render": {
            "interpolation": "bilinear", "outside": "clamp",
            "nx": int(len(model.node_x)), "ny": int(len(model.node_y)),
            "spacing_x": float(model.node_x[1] - model.node_x[0]) if len(model.node_x) > 1 else None,
            "spacing_y": float(model.node_y[1] - model.node_y[0]) if len(model.node_y) > 1 else None,
        },
        **{k: v for k, v in rec.items() if k not in ("format", "format_version", "render")},
    }
    manifest["render"].update(rec.get("render", {}))      # the rendering rule + the build's n_b, b
    (d / "model.json").write_text(json.dumps(_jsonable(manifest), indent=1, allow_nan=False))
    np.savez(d / "model.npz",
             node_x=np.asarray(model.node_x, dtype=np.float64),
             node_y=np.asarray(model.node_y, dtype=np.float64),
             stamps=np.asarray(model.stamps, dtype=np.float64),
             star_x=np.asarray(model.star_x, dtype=np.float64),
             star_y=np.asarray(model.star_y, dtype=np.float64),
             star_weight=np.asarray(model.star_weight, dtype=np.float64),
             star_fold=np.asarray(model.star_fold, dtype=np.int64),
             star_a=np.asarray(model.star_a, dtype=np.float64),
             star_r2=np.asarray(model.star_r2, dtype=np.float64),
             mix_node_x=np.asarray(model.mix_node_x, dtype=np.float64),
             mix_node_y=np.asarray(model.mix_node_y, dtype=np.float64),
             mix_weights=np.asarray(model.mix_weights, dtype=np.float64))
    return d


def load_psf_model(psf_root, image_id: str) -> PsfModel:
    """Read a model written by :func:`save_psf_model` (format version 1)."""
    d = Path(psf_root) / image_id
    man = json.loads((d / "model.json").read_text())
    if man.get("format") != FORMAT_NAME or int(man.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(f"{d}: not a {FORMAT_NAME} v{FORMAT_VERSION} model "
                         f"({man.get('format')!r} v{man.get('format_version')!r})")
    with np.load(d / "model.npz") as z:
        arrs = {k: z[k] for k in z.files}
    rec = {k: v for k, v in man.items() if k not in (
        "format", "format_version", "image_id", "band", "frame", "image_shape",
        "work_frame_offset", "stamp_size", "stamp_centre", "normalisation")}
    return PsfModel(
        image_id=man["image_id"], band=man["band"], frame=man["frame"],
        image_shape=tuple(man["image_shape"]),
        node_x=arrs["node_x"], node_y=arrs["node_y"], stamps=arrs["stamps"],
        star_x=arrs["star_x"], star_y=arrs["star_y"], star_weight=arrs["star_weight"],
        star_fold=arrs["star_fold"], record=rec,
        work_frame_offset=tuple(man.get("work_frame_offset", (0, 0))),
        mix_node_x=arrs["mix_node_x"], mix_node_y=arrs["mix_node_y"], mix_weights=arrs["mix_weights"],
        star_a=arrs["star_a"], star_r2=arrs.get("star_r2", np.zeros(0)))


def write_index(psf_root, rows: Sequence[tuple[str, str, str]]) -> Path:
    """Write ``<psf_root>/index.csv`` (image_id, band, frame), one row per image."""
    import csv
    p = Path(psf_root) / INDEX_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image_id", "band", "frame"])
        for r in rows:
            wr.writerow([str(r[0]), str(r[1]), str(r[2])])
    return p


def check_integral(stamps: np.ndarray, image_id: str = "") -> float:
    """max |sum - 1| over render-node stamps; raises PsfModelIntegralError above 1e-6."""
    sums = np.asarray(stamps, dtype=np.float64).sum(axis=(-2, -1))
    dev = np.abs(sums - 1.0)
    worst = float(np.max(dev)) if dev.size else float("nan")
    if not np.all(np.isfinite(sums)) or not worst <= INTEGRAL_TOL:
        bad = np.argwhere(~(dev <= INTEGRAL_TOL))
        raise PsfModelIntegralError(
            f"PSF model {image_id!r}: render-node stamp sums deviate from 1 by up to "
            f"{worst!r} (tolerance {INTEGRAL_TOL}); {len(bad)} of {dev.size} nodes",
            image_id=image_id, max_abs_dev=worst, n_bad_nodes=int(len(bad)),
            first_bad_node=[int(v) for v in bad[0]] if len(bad) else None)
    return worst


# ---------------------------------------------------------------------------
# Building (design v2.5: local stacking, heterogeneity floor, per-band render refinement)
# ---------------------------------------------------------------------------

SINGULAR_RCOND = 1e-12   # a node normal matrix with eig_min <= 1e-12 eig_max is degenerate
DENSE_POINTS_PER_CELL = 8    # evaluation path only (values equal to rounding): per-cell GEMM when dense
MIX_RIDGE = 1e-10        # design v2.5 1.3 step 5: ridge eps tr(G)/M of the stacking problem
FLUX_FRAC_LIMIT = 1e-3   # D9-1 (v2.5 3rd ed. 1.2): render error, flux fraction (~1.09 mmag)
PIX_FRAC_LIMIT = 1e-2    # D9-1: render error, pixel-norm fraction (neighbour residual pattern)
VERIFY_TOP_FRACTION = 0.05   # D9-1: the estimate is checked in the top 5% of cells
MAX_RENDER_REFINE = 16   # guard: refinement 16 of the D/16 grid is ~2.4 GB of float32 stamps per band
RENDER_SPACING_ERROR_KIND = "psf_render_spacing"
RENDER_MEMORY_ERROR_KIND = "psf_render_memory"


def _terms(p: int) -> list[tuple[int, int]]:
    """Exponent pairs (a, b), a + b <= p, by total degree (lower orders are prefixes)."""
    return [(a, d - a) for d in range(p + 1) for a in range(d, -1, -1)]


def _n_terms(p: int) -> int:
    return (p + 1) * (p + 2) // 2


def _legendre(v: np.ndarray, deg: int) -> np.ndarray:
    """Legendre P_0..P_deg at v: shape (deg + 1, *v.shape)."""
    out = [np.ones_like(v), v.copy()]
    for n in range(1, deg):
        out.append(((2 * n + 1) * v * out[n] - n * out[n - 1]) / (n + 1))
    return np.stack(out[:deg + 1])


def _basis(xu: np.ndarray, yv: np.ndarray, p: int, legendre: bool) -> np.ndarray:
    """Basis values (T, n): monomials u^a v^b (local) or Legendre L_a(u) L_b(v) (global)."""
    if legendre:
        lx, ly = _legendre(xu, p), _legendre(yv, p)
        return np.stack([lx[a] * ly[b] for a, b in _terms(p)])
    px = [np.ones_like(xu)]
    py = [np.ones_like(yv)]
    for _ in range(p):
        px.append(px[-1] * xu)
        py.append(py[-1] * yv)
    return np.stack([px[a] * py[b] for a, b in _terms(p)])


def _hat(v: np.ndarray, n_pix: int) -> np.ndarray:
    """Field coordinate in [-1, 1] of a pixel position (0 .. n_pix - 1)."""
    return 2.0 * v / max(n_pix - 1, 1) - 1.0


@dataclass
class _Family:
    """Normal-equation partial sums of one bandwidth (all orders <= p_max), per fold + total."""
    h: float
    p_max: int
    node_x: np.ndarray            # (nx,) training nodes (global: empty)
    node_y: np.ndarray
    G: np.ndarray                 # (K_folds, n_nodes, T, T)
    R: np.ndarray                 # (K_folds, n_nodes, T, P)
    S1: np.ndarray                # (K_folds, n_nodes)  sum of weights
    S2: np.ndarray                # (K_folds, n_nodes)  sum of squared weights
    R_tot: np.ndarray             # (n_nodes, T, P)     sum over folds


def _family_sums(h: float, p_max: int, x, y, omega, d, folds, n_folds, shape) -> _Family:
    hh, ww = int(shape[0]), int(shape[1])
    glob = not math.isfinite(h)
    if glob:
        nx_, ny_ = np.zeros(0), np.zeros(0)
        a = _basis(_hat(x, ww), _hat(y, hh), p_max, True)[:, None, :]      # (T, 1, N)
        w = omega[None, :]                                                    # (1, N)
    else:
        nx_, ny_ = training_grid(h, shape)
        gx, gy = np.meshgrid(nx_, ny_)                                        # (ny, nx)
        kx, ky = gx.ravel(), gy.ravel()                                       # node k = j*nx + i
        u = (x[None, :] - kx[:, None]) / h                                    # (K, N)
        v = (y[None, :] - ky[:, None]) / h
        w = np.exp(-0.5 * (u * u + v * v)) * omega[None, :]
        a = _basis(u, v, p_max, False)                                        # (T, K, N)
    t, k_nodes, _ = a.shape
    p_pix = d.shape[1]
    G = np.zeros((n_folds, k_nodes, t, t))
    R = np.zeros((n_folds, k_nodes, t, p_pix))
    S1 = np.zeros((n_folds, k_nodes))
    S2 = np.zeros((n_folds, k_nodes))
    fold_idx = [np.nonzero(folds == f)[0] for f in range(n_folds)]
    for f, idx in enumerate(fold_idx):
        if idx.size:
            S1[f] = w[:, idx].sum(axis=1)
            S2[f] = (w[:, idx] * w[:, idx]).sum(axis=1)
    # rows of R a node can ever use: up to the highest order that the support rule allows in some
    # training set (every fold's complement and the full set), from the same direct sums _solve uses;
    # the other rows stay zero and are never read (psf_speed_answer 2.4; value-preserving)
    t_need = np.ones(k_nodes, dtype=np.int64)
    sets = [np.arange(n_folds) != f for f in range(n_folds)] + [np.ones(n_folds, dtype=bool)]
    for keep in sets:
        s1, s2 = S1[keep].sum(axis=0), S2[keep].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            neff = np.where(s2 > 0, s1 * s1 / s2, 0.0)
        for p in range(1, p_max + 1):
            ok = (s1 > 0) & (neff >= 3 * _n_terms(p))
            t_need[ok] = np.maximum(t_need[ok], _n_terms(p))
    rows = (np.arange(t)[None, :] < t_need[:, None]).ravel()                  # (K*T,) row mask
    every = bool(rows.all())
    for f, idx in enumerate(fold_idx):
        if idx.size == 0:
            continue
        wf = w[:, idx]                                                        # (K, Nf)
        af = a[:, :, idx]                                                     # (T, K, Nf)
        wa = np.transpose(af * wf[None, :, :], (1, 0, 2))                     # (K, T, Nf)
        G[f] = np.einsum("ktn,kun->ktu", wa, np.transpose(af, (1, 0, 2)))
        wa2 = wa.reshape(k_nodes * t, idx.size)
        if every:
            R[f] = (wa2 @ d[idx]).reshape(k_nodes, t, p_pix)
        else:
            R[f].reshape(k_nodes * t, p_pix)[rows] = wa2[rows] @ d[idx]
    return _Family(h, p_max, nx_, ny_, G, R, S1, S2, R.sum(axis=0))


@dataclass
class _Fit:
    """One candidate fitted on one training set: coefficients at its nodes."""
    cand: Candidate
    node_x: np.ndarray
    node_y: np.ndarray
    coefs: np.ndarray             # (n_nodes, T_p, P)
    shape: tuple
    n_p0: int = 0
    n_singular: int = 0
    n_unreached: int = 0

    def evaluate(self, xs, ys, chunk: int = 256) -> np.ndarray:
        """Exact P_m at positions (clamped to the image): (n, P)."""
        hh, ww = int(self.shape[0]), int(self.shape[1])
        xs = np.clip(np.atleast_1d(np.asarray(xs, dtype=np.float64)), 0.0, ww - 1.0)
        ys = np.clip(np.atleast_1d(np.asarray(ys, dtype=np.float64)), 0.0, hh - 1.0)
        tp = self.coefs.shape[1]
        p = int(round((math.sqrt(8 * tp + 1) - 3) / 2))
        if self.cand.is_global:
            b = _basis(_hat(xs, ww), _hat(ys, hh), p, True)                   # (T, n)
            return b.T @ self.coefs[0]
        nx = len(self.node_x)
        i, t = _cell(self.node_x, xs)
        j, u = _cell(self.node_y, ys)
        out = np.empty((len(xs), self.coefs.shape[2]))
        h = self.cand.h
        corners = ((0, 0), (1, 0), (0, 1), (1, 1))
        cell = j * nx + i
        if len(xs) >= DENSE_POINTS_PER_CELL * len(np.unique(cell)):
            # many points per training cell (render grids, check sub-grids): one GEMM per cell and
            # corner, (n_c x T) @ (T x P), instead of gathering a (T, P) block per point
            order = np.argsort(cell, kind="stable")
            cs = cell[order]
            cut = np.flatnonzero(np.diff(cs)) + 1
            for grp in np.split(order, cut):
                ii, jj = int(i[grp[0]]), int(j[grp[0]])
                acc = np.zeros((grp.size, self.coefs.shape[2]))
                for di, dj in corners:
                    ic, jc = min(ii + di, nx - 1), min(jj + dj, len(self.node_y) - 1)
                    bw = (t[grp] if di else 1.0 - t[grp]) * (u[grp] if dj else 1.0 - u[grp])
                    b = _basis((xs[grp] - self.node_x[ic]) / h, (ys[grp] - self.node_y[jc]) / h, p, False)
                    acc += bw[:, None] * (b.T @ self.coefs[jc * nx + ic])
                out[grp] = acc
            return out
        for s0 in range(0, len(xs), chunk):
            sl = slice(s0, s0 + chunk)
            acc = np.zeros((len(xs[sl]), self.coefs.shape[2]))
            for di, dj in corners:
                ii = np.minimum(i[sl] + di, nx - 1)
                jj = np.minimum(j[sl] + dj, len(self.node_y) - 1)
                bw = (t[sl] if di else 1.0 - t[sl]) * (u[sl] if dj else 1.0 - u[sl])
                b = _basis((xs[sl] - self.node_x[ii]) / h, (ys[sl] - self.node_y[jj]) / h, p, False)
                q = np.einsum("tn,ntp->np", b, self.coefs[jj * nx + ii])
                acc += bw[:, None] * q
            out[sl] = acc
        return out


def _train_sums(fam: _Family, held_fold: int | None):
    """Normal sums of the stars outside fold ``held_fold`` (None: all stars): (G, R, S1, S2).

    The large right-hand side R is the total minus that fold (one subtraction instead of
    re-adding K - 1 folds; gpu-engineer psf_speed_answer 2.1).  The small arrays G, S1, S2 are
    summed directly over the other folds, so the support / reach / degeneracy decisions never see
    a cancellation residue."""
    if held_fold is None:
        return fam.G.sum(axis=0), fam.R_tot, fam.S1.sum(axis=0), fam.S2.sum(axis=0)
    keep = np.arange(fam.G.shape[0]) != held_fold
    return (fam.G[keep].sum(axis=0), fam.R_tot - fam.R[held_fold],
            fam.S1[keep].sum(axis=0), fam.S2[keep].sum(axis=0))


def _solve(fam: _Family, sums, p: int, cand: Candidate, shape) -> _Fit:
    """Fit order p of a family from its training sums (see :func:`_train_sums`)."""
    tp = _n_terms(p)
    G = sums[0][:, :tp, :tp]
    R = sums[1][:, :tp, :]
    S1, S2 = sums[2], sums[3]
    k_nodes, _, p_pix = R.shape
    reached = S1 > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        neff = np.where(S2 > 0, S1 * S1 / S2, 0.0)
    use_p = reached & (neff >= 3 * tp) if tp > 1 else np.zeros(k_nodes, dtype=bool)
    n_p0 = int((reached & ~use_p).sum()) if tp > 1 else 0     # support rule (design 1.4)
    n_singular = 0
    if use_p.any():
        ev = np.linalg.eigvalsh(G[use_p])
        ok = ev[:, 0] > SINGULAR_RCOND * ev[:, -1]
        idx = np.nonzero(use_p)[0]
        n_singular = int((~ok).sum())
        use_p[idx[~ok]] = False
    if use_p.any():
        # one batched solve over every node (no copy of R's rows); nodes without support get an
        # identity system whose answer is overwritten below
        coefs = np.linalg.solve(np.where(use_p[:, None, None], G, np.eye(tp)), R)
    else:
        coefs = np.zeros((k_nodes, tp, p_pix))
    mean = reached & ~use_p
    if mean.any():
        coefs[mean] = 0.0
        coefs[mean, 0, :] = R[mean, 0, :] / G[mean, 0, 0][:, None]
    unreached = ~reached
    n_unreached = int(unreached.sum())
    if n_unreached:
        if not reached.any():
            raise ValueError(f"PSF candidate h={cand.h}: no star reaches any node")
        gx, gy = np.meshgrid(fam.node_x, fam.node_y) if len(fam.node_x) else (np.zeros(1), np.zeros(1))
        kx, ky = gx.ravel(), gy.ravel()
        src = np.nonzero(reached)[0]
        for k in np.nonzero(unreached)[0]:
            near = src[np.argmin((kx[src] - kx[k]) ** 2 + (ky[src] - ky[k]) ** 2)]
            coefs[k] = 0.0
            coefs[k, 0, :] = R[near, 0, :] / G[near, 0, 0]
    return _Fit(cand, fam.node_x, fam.node_y, coefs, tuple(shape), n_p0=n_p0,
                n_singular=n_singular, n_unreached=n_unreached)


def assign_folds(x, y, image_shape, candidates: Sequence[Candidate], n_folds: int = N_FOLDS):
    """Stratified folds: stars in raster order of the finest training-grid cells (then y, x),
    dealt round-robin into the folds.  Depends on positions only (not on pixel values)."""
    gx, gy = render_grid(candidates, image_shape, 1)
    ci, _ = _cell(gx, x)
    cj, _ = _cell(gy, y)
    order = np.lexsort((np.arange(len(x)), x, y, ci, cj))
    folds = np.empty(len(x), dtype=np.int64)
    folds[order] = np.arange(len(x)) % int(n_folds)
    return folds


def stack_simplex(G) -> np.ndarray:
    """Exact minimiser of w'(G + 1e-10 tr(G)/M I) w over the simplex (w >= 0, sum w = 1).

    Primal active-set method (strictly convex quadratic thanks to the ridge, so the solution is
    unique and continuous in G).  Start at the best vertex; on the active set A solve the equality
    problem [Q_AA 1; 1' 0][v; mu] = [0; 1]; if v > 0, add the inactive index with the most negative
    reduced gradient (Qw)_i - w'Qw, else step back to the first weight that reaches zero and drop
    it.  Design v2.5 1.3 step 5.

    The problem is defined over DISTINCT candidates (design v2.5 3rd ed. 4.1): exact duplicates
    (bitwise identical rows of G: a candidate listed twice, or candidates the support rule turned into
    the same model) are one model -- the problem, its ridge and tr/M are taken on the distinct set,
    and the weight is split equally within the group.  So listing a candidate twice leaves the mixture
    unchanged by definition (the eps'/k_g alternative moved it by O(eps) and was withdrawn); solving
    duplicates jointly would leave their split to the ridge alone (condition ~1/eps = 1e10)."""
    G = np.asarray(G, dtype=np.float64)
    m = G.shape[0]
    if m == 1:
        return np.ones(1)
    G = 0.5 * (G + G.T)
    _, first, inverse = np.unique(G, axis=0, return_index=True, return_inverse=True)
    if len(first) < m:
        order = np.argsort(first)
        reps = first[order]
        pos = np.empty(len(first), dtype=np.int64)
        pos[order] = np.arange(len(first))
        grp = pos[np.asarray(inverse).ravel()]
        counts = np.bincount(grp, minlength=len(reps))
        vr = _stack_simplex_unique(G[np.ix_(reps, reps)])
        return vr[grp] / counts[grp]
    return _stack_simplex_unique(G)


def _stack_simplex_unique(G: np.ndarray) -> np.ndarray:
    """The active-set solve of :func:`stack_simplex` for a G of distinct candidates (ridge
    1e-10 tr(G)/M on this set)."""
    m = G.shape[0]
    if m == 1:
        return np.ones(1)
    G = 0.5 * (G + G.T)
    tr = float(np.trace(G))
    scale = tr / m if tr > 0 else 1.0
    q = G / scale + MIX_RIDGE * np.eye(m)
    act = [int(np.argmin(np.diag(q)))]
    w = np.zeros(m)
    w[act[0]] = 1.0
    for _ in range(20 * m + 20):
        k = len(act)
        kkt = np.zeros((k + 1, k + 1))
        kkt[:k, :k] = q[np.ix_(act, act)]
        kkt[:k, k] = 1.0
        kkt[k, :k] = 1.0
        rhs = np.zeros(k + 1)
        rhs[k] = 1.0
        v = np.linalg.solve(kkt, rhs)[:k]
        if np.all(v > 0.0):
            w = np.zeros(m)
            w[act] = v
            gr = q @ w
            lam = float(w @ gr)
            out = [i for i in range(m) if i not in act]
            if not out:
                break
            red = gr[out] - lam
            j = int(np.argmin(red))
            if red[j] >= -1e-14 * max(abs(lam), 1e-300):
                break
            act.append(out[j])
            continue
        wa = w[act]
        neg = v <= 0.0
        ratio = wa[neg] / (wa[neg] - v[neg])
        alpha = float(np.min(ratio))
        wa = wa + alpha * (v - wa)
        drop = np.nonzero(neg)[0][int(np.argmin(ratio))]
        keep = [ii for ii in range(k) if ii != drop and wa[ii] > 0.0]
        act = [act[ii] for ii in keep]
        w = np.zeros(m)
        w[act] = wa[keep]
        w /= w.sum()
    return w / w.sum()


def mixing_grid(image_shape) -> tuple[np.ndarray, np.ndarray, float]:
    """Mixing nodes and kernel width (design v2.5 1.3 step 4): the D/8 training grid, g = D/8."""
    g = image_diagonal(image_shape) / 8.0
    mx, my = training_grid(g, image_shape)
    return mx, my, g


def _weights_at(mix_x, mix_y, mix_w, xs, ys) -> np.ndarray:
    """Bilinear interpolation of mixing-node weights at positions: (n, M), rows sum to 1."""
    i, t = _cell(mix_x, xs)
    j, u = _cell(mix_y, ys)
    i1 = np.minimum(i + 1, len(mix_x) - 1)
    j1 = np.minimum(j + 1, len(mix_y) - 1)
    t, u = t[:, None], u[:, None]
    return ((1 - t) * (1 - u) * mix_w[j, i] + t * (1 - u) * mix_w[j, i1]
            + (1 - t) * u * mix_w[j1, i] + t * u * mix_w[j1, i1])


def local_stack_weights(gram, omega, x, y, mix_x, mix_y, g) -> np.ndarray:
    """Stacking weights at the mixing nodes (ny, nx, M): G_j = sum_s kappa_j(s) omega_s g_s with
    kappa_j = exp(-|r_s - q_j|^2 / (2 g^2)), w_j = stack_simplex(G_j) (``gram`` over distinct
    candidates; the caller splits a group's weight equally)."""
    gx, gy = np.meshgrid(mix_x, mix_y)
    qx, qy = gx.ravel(), gy.ravel()
    kap = np.exp(-0.5 * ((x[None, :] - qx[:, None]) ** 2 + (y[None, :] - qy[:, None]) ** 2) / (g * g))
    n, m, _ = gram.shape
    gj = ((kap * omega[None, :]) @ gram.reshape(n, m * m)).reshape(len(qx), m, m)
    w = np.stack([stack_simplex(gj[k]) for k in range(len(qx))])
    return w.reshape(len(mix_y), len(mix_x), m)


#: TGM constants (design unit_a1b_tau2_spec_v3.md 1.1; registration prereg_a1b.md 1.1).  The mean-shift bandwidth
#: is 0.5 x the MAD scale: the usual half-sigma choice, and the most stable of 0.5 / 0.75 / 1.0 on the designer's
#: seen seeds (review_a1b_spec C-7); the record's ``ms_global_gap`` shows when the median start missed the highest
#: density mode on real data.
TGM_GROUP_MIN = 15
TGM_BANDWIDTH = 0.5
TGM_MS_TOL = 1e-14
TGM_MS_MAX = 1000
TGM_FLOOR = 1e-3
TGM_GAP_GRID = 2000
TGM_MIN_STARS = 30


def _tgm_groups(a):
    o = np.argsort(a, kind="mergesort")
    n = a.size
    return o[:max(TGM_GROUP_MIN, n // 2)], o[n - max(TGM_GROUP_MIN, n - n // 2):], o[:max(TGM_GROUP_MIN, n // 4)]


def _mean_shift_mode(v):
    """Continuous mode by Gaussian mean shift from the median, bandwidth 0.5 x 1.4826 MAD (fixed per call).

    Returns (mode, iterations, bandwidth, capped)."""
    v = np.asarray(v, dtype=np.float64)
    m = float(np.median(v))
    h = TGM_BANDWIDTH * 1.4826 * float(np.median(np.abs(v - m)))
    if not h > 0:
        return m, 0, 0.0, False
    for j in range(TGM_MS_MAX):
        k = np.exp(-0.5 * ((v - m) / h) ** 2)
        mn = float(np.sum(k * v) / np.sum(k))
        if abs(mn - m) <= TGM_MS_TOL * abs(mn):
            return mn, j + 1, h, False
        m = mn
    return m, TGM_MS_MAX, h, True


def _ms_global_gap(v, m: float, h: float) -> float:
    """|mean-shift limit - the highest point of the same-kernel density on a 2,000-point grid over
    [min v, max v]| / h (review_a1b_spec C-7)."""
    if not h > 0:
        return 0.0
    v = np.asarray(v, dtype=np.float64)
    g = np.linspace(float(v.min()), float(v.max()), TGM_GAP_GRID)
    dens = np.exp(-0.5 * ((g[:, None] - v[None, :]) / h) ** 2).sum(axis=1)
    return abs(m - float(g[int(np.argmax(dens))])) / h


def _wmi(v, w) -> float:
    """Interpolated weighted median: stable sort, mid-point cumulative weights, linear interpolation at half the total."""
    o = np.argsort(v, kind="mergesort")
    vv, ww = np.asarray(v, dtype=np.float64)[o], np.asarray(w, dtype=np.float64)[o]
    cum = np.cumsum(ww) - 0.5 * ww
    return float(np.interp(0.5 * ww.sum(), cum, vv))


def _n50(w) -> int:
    ws = np.sort(np.asarray(w, dtype=np.float64))[::-1]
    return int(np.searchsorted(np.cumsum(ws), 0.5 * ws.sum()) + 1)


def tgm_floor(r2, a):
    """(lambda, tau^2, info) of the star weight omega = 1 / (lambda a + tau^2) by TGM (design v2.5 a.1b 3rd ed. 1.1).

    r2 = each star's held-out (pass a) residual energy under the mixed model, a = its expected noise energy.  Two
    groups by a (stable sort): B = the max(15, n // 2) smallest a, F = the max(15, n - n // 2) largest, Q = the
    max(15, n // 4) smallest.  Two fixed steps (no convergence test, so no cycle can occur):
      lambda_1 = MS(r2_F / a_F);  f_1 = max(median_Q(r2 - lambda_1 a), 1e-3 min_B lambda_1 a);
      tau2_1 = max(0, WMI_B(r2 - lambda_1 a; (lambda_1 a + f_1)^-2));
      lambda_2 = MS((r2 - tau2_1)_F / a_F);  f_2 = max(tau2_1, 1e-3 min_B lambda_2 a);
      tau2_2 = max(0, WMI_B(r2 - lambda_2 a; (lambda_2 a + f_2)^-2))  ->  (lambda_2, tau2_2).
    MS = Gaussian mean shift from the median (:func:`_mean_shift_mode`), WMI = interpolated weighted median.
    Identification check (a1b_fix_answer.md 2; registration prereg_a1b_r2.md 1): lambda is the noise scale seen in
    the faint group, identified only if lambda_2 > 0 and lambda_2 median_F(a) >= tau2_2 (the median faint star is
    noise dominated).  Otherwise the data do not tell the scale (noiseless data, or an a that does not describe the
    residual noise): lambda = 1 (the given a) and one step on B / Q,
      f = max(median_Q(r2 - a), 1e-3 min_B a);  tau^2 = max(0, WMI_B(r2 - a; (a + f)^-2)),
    with a warning.  Every weight is then positive.  Identified, the weights are invariant to a common scale of a;
    the check itself is too (the fallback is not).  Fewer than 30 stars: lambda = 1 and one step on all stars.
    tau^2 = 0 is logged as a warning (the bright stars' residuals do not exceed lambda a: the weight floor collapsed).
    A pure function apart from those log lines."""
    r2 = np.asarray(r2, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    n = a.size
    info: dict = {"n": int(n)}
    if n < TGM_MIN_STARS:
        f = max(float(np.median(r2 - a)), TGM_FLOOR * float(np.min(a)))
        wb = 1.0 / (a + f) ** 2
        excess = _wmi(r2 - a, wb)
        lam, t2 = 1.0, max(0.0, excess)
        logger.info("PSF tau^2: %d stars (< %d): lambda = 1 and one step on all stars", n, TGM_MIN_STARS)
        info.update(fallback=True, n_B=int(n), n_F=0, ms_iters=(0, 0), ms_global_gap=(0.0, 0.0),
                    ms_capped=False, N50=_n50(wb), B_excess=float(excess), lambda_identified=None,
                    lam_hat=None, F_median_r2_over_a=None)
    else:
        b, fg, q = _tgm_groups(a)
        v1 = r2[fg] / a[fg]
        lam1, it1, h1, cap1 = _mean_shift_mode(v1)
        f1 = max(float(np.median(r2[q] - lam1 * a[q])), TGM_FLOOR * float(np.min(lam1 * a[b])))
        t21 = max(0.0, _wmi(r2[b] - lam1 * a[b], 1.0 / (lam1 * a[b] + f1) ** 2))
        v2 = (r2[fg] - t21) / a[fg]
        lam, it2, h2, cap2 = _mean_shift_mode(v2)
        f2 = max(t21, TGM_FLOOR * float(np.min(lam * a[b])))
        wb = 1.0 / (lam * a[b] + f2) ** 2
        excess = _wmi(r2[b] - lam * a[b], wb)
        t2 = max(0.0, excess)
        if cap1 or cap2:
            logger.warning("PSF tau^2: the mean shift reached %d iterations; its last value is used", TGM_MS_MAX)
        gaps = (_ms_global_gap(v1, lam1, h1), _ms_global_gap(v2, lam, h2))
        lam_hat, t2_step2, med_af = lam, t2, float(np.median(a[fg]))
        ratio = float(np.median(v1))
        identified = bool(lam_hat > 0.0 and lam_hat * med_af >= t2_step2)
        if not identified:
            lam = 1.0
            f = max(float(np.median(r2[q] - a[q])), TGM_FLOOR * float(np.min(a[b])))
            wb = 1.0 / (a[b] + f) ** 2
            excess = _wmi(r2[b] - a[b], wb)
            t2 = max(0.0, excess)
            logger.warning(
                "PSF model: the noise scale lambda is not identified -- the faint stars' residuals do not follow a "
                "(median over F of r2 / a = %.4g; lambda-hat = %.4g, lambda-hat x median_F(a) = %.4g, step-2 "
                "floor = %.4g): noiseless data, or an a that does not describe the residual noise; falling back to "
                "lambda = 1 (the given a)", ratio, lam_hat, lam_hat * med_af, t2_step2)
        info.update(fallback=False, n_B=int(b.size), n_F=int(fg.size), ms_iters=(int(it1), int(it2)),
                    ms_global_gap=gaps, ms_capped=bool(cap1 or cap2), N50=_n50(wb), B_excess=float(excess),
                    lambda_identified=identified, lam_hat=float(lam_hat), F_median_r2_over_a=ratio)
    info["tau2_zero"] = bool(t2 == 0.0)
    if t2 == 0.0:
        logger.warning(
            "PSF model: tau^2 = 0 -- the bright stars' residuals do not exceed lambda a; the weight floor "
            "collapsed and the weights follow 1/a (n_B = %d, N50 = %d, lambda = %.6g, B weighted-median "
            "excess = %.4g)", info["n_B"], info["N50"], lam, info["B_excess"])
    return float(lam), float(t2), info


@dataclass
class Mixture:
    """The fitted mixture of one image: P(r) = sum_m w_m(r) P_m(r) (exact, before rendering);
    w(r) bilinear between the mixing nodes."""
    fits: list
    mix_node_x: np.ndarray
    mix_node_y: np.ndarray
    mix_weights: np.ndarray       # (ny_mix, nx_mix, M)
    stamp_size: int
    record: dict
    star_r2: np.ndarray = field(default_factory=lambda: np.zeros(0))   # pass-a residual energy (a.1b)

    def weights_at(self, xs, ys) -> np.ndarray:
        return _weights_at(self.mix_node_x, self.mix_node_y, self.mix_weights,
                           np.atleast_1d(np.asarray(xs, dtype=np.float64)),
                           np.atleast_1d(np.asarray(ys, dtype=np.float64)))

    def evaluate(self, xs, ys) -> np.ndarray:
        """Exact P at positions: (n, S, S); a candidate is evaluated only where it has weight."""
        xs = np.atleast_1d(np.asarray(xs, dtype=np.float64))
        ys = np.atleast_1d(np.asarray(ys, dtype=np.float64))
        wr = self.weights_at(xs, ys)
        out = np.zeros((len(xs), self.stamp_size * self.stamp_size))
        for m, f in enumerate(self.fits):
            sel = np.nonzero(wr[:, m] != 0.0)[0]
            if sel.size:
                out[sel] += wr[sel, m:m + 1] * f.evaluate(xs[sel], ys[sel])
        return out.reshape(len(xs), self.stamp_size, self.stamp_size)


def _cv_pass(d, x, y, omega, folds, k, cands, extra, shape, mix, want_fits: bool):
    """One pass: K-fold held-out residuals of every candidate -> per-star Gram -> local stacking.

    Returns (fits or None, mix_weights, gram, loss_folds)."""
    fam_p: dict = {}
    for c in cands:
        hk = c.h if math.isfinite(c.h) else math.inf
        fam_p[hk] = max(fam_p.get(hk, 0), int(c.p))
    fams = {hk: _family_sums(hk, p, x, y, omega, d, folds, k, shape) for hk, p in fam_p.items()}
    for xc in extra:                          # a.8 plug-in: fold-independent work once per pass
        if hasattr(xc, "prepare"):
            xc.prepare(d, x, y)
    n, p_pix = d.shape
    m = len(cands) + len(extra)
    gram = np.zeros((n, m, m))
    loss_folds = np.zeros((m, k))
    same = np.ones((m, m), dtype=bool)        # candidates whose held-out residuals are identical
    for f in range(k):
        held = np.nonzero(folds == f)[0]
        if held.size == 0:
            continue
        train = folds != f
        e = np.empty((held.size, m, p_pix))
        sums = {hk: _train_sums(fm, f) for hk, fm in fams.items()}
        for mi, c in enumerate(cands):
            hk = c.h if math.isfinite(c.h) else math.inf
            e[:, mi, :] = d[held] - _solve(fams[hk], sums[hk], int(c.p), c, shape).evaluate(x[held], y[held])
        for ei, xc in enumerate(extra):
            pred = xc.fit(d[train], x[train], y[train], omega[train], train)    # training stars only
            e[:, len(cands) + ei, :] = d[held] - pred.evaluate(x[held], y[held])
        gram[held] = e @ np.transpose(e, (0, 2, 1))                           # batched GEMM
        loss_folds[:, f] = np.einsum("s,smm->m", omega[held], gram[held])
        fp = e.sum(axis=(0, 2))
        for i in range(m):
            for j in range(i + 1, m):
                if same[i, j] and (fp[i] != fp[j] or not np.array_equal(e[:, i], e[:, j])):
                    same[i, j] = same[j, i] = False
    # exact duplicates (bitwise identical held-out residuals in every fold: a candidate listed twice,
    # or candidates the support rule turned into one model) are one model: the stacking problem is
    # defined over the distinct set (ridge and tr/M there) and a group's weight is split equally
    # (design v2.5 3rd ed. 4.1) -- the mixture does not depend on how often a model is listed
    grp = np.array([int(np.argmax(same[i, :i + 1])) for i in range(m)])
    reps = np.unique(grp)
    pos = np.searchsorted(reps, grp)
    counts = np.bincount(pos, minlength=len(reps))
    v_r = local_stack_weights(gram[:, reps][:, :, reps], omega, x, y, mix[0], mix[1], mix[2])
    mix_w = v_r[..., pos] / counts[pos]
    fits = None
    if want_fits:
        full = {hk: _train_sums(fm, None) for hk, fm in fams.items()}
        fits = []
        for c in cands:
            hk = c.h if math.isfinite(c.h) else math.inf
            fits.append(_solve(fams[hk], full[hk], int(c.p), c, shape))
        everyone = np.ones(n, dtype=bool)
        fits += [xc.fit(d, x, y, omega, everyone) for xc in extra]
    return fits, mix_w, gram, loss_folds


def fit_mixture(stamps, star_x, star_y, noise_a, image_shape, *, tau2=None,
                candidates: Sequence[Candidate] | None = None, extra_candidates=(),
                n_folds: int = N_FOLDS, folds: np.ndarray | None = None) -> tuple[Mixture, np.ndarray]:
    """Fit every candidate with K-fold held-out residuals, stack locally, refit on all stars.

    Star weight omega_s = 1 / (lambda a_s + tau^2) (design v2.5 2.3; no Huber weight, 3rd ed. 7.1;
    unit a.1b): ``noise_a`` = a_s, the expected squared norm of the noise of the stamp normalised by its
    box sum (:func:`tract7dt.fit.grid_gpu.psf_from_stars.star_noise_a`).  tau2=None: pass a with weights
    1 / a, then (lambda, tau^2) from the mixed model's held-out residual energies by :func:`tgm_floor`
    (robust, invariant to a common scale of a), pass b with them (two fixed passes).  tau2 = (lambda,
    tau^2) given: one pass with it (PSF(1), PSF(2) reuse the previous iteration's pair) -- four fits per
    band in all.
    extra_candidates: objects with ``fit(d, x, y, omega, train) -> predictor`` whose
    ``evaluate(xs, ys)`` returns (n, S*S) non-negative stamps summing to 1, and optionally
    ``prepare(d, x, y)`` (called once per pass before the folds, with every star) -- the plug-in point
    of unit a.8 (design v2.5 3rd ed. 3).  ``fit`` receives the TRAINING stars only (d, x, y, omega =
    those rows; review_a.1 C-3: held-out pixels never reach it) plus ``train``, the (N,) bool mask of
    those rows in the order of ``stamps``, to index the candidate's own per-star inputs (FWHM_20, F_s,
    sigma_own,s, gain via its constructor) and anything ``prepare`` cached; predictors are pure and
    deterministic.
    Returns (mixture, folds)."""
    t0 = time.perf_counter()
    st = np.asarray(stamps, dtype=np.float64)
    if st.ndim != 3 or st.shape[1] != st.shape[2]:
        raise ValueError(f"stamps must be (N, S, S), got {st.shape}")
    n, s = st.shape[0], st.shape[1]
    x = np.asarray(star_x, dtype=np.float64)
    y = np.asarray(star_y, dtype=np.float64)
    a = np.asarray(noise_a, dtype=np.float64)
    if not (x.shape == y.shape == a.shape == (n,)):
        raise ValueError("star_x, star_y, noise_a must have one value per stamp")
    if not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError("noise_a must be finite and > 0")
    if tau2 is not None:
        try:
            lam_given, t2_given = (float(v) for v in tau2)
        except (TypeError, ValueError):
            raise TypeError(f"tau2 must be None or a (lambda, tau^2) pair (a.1b), got {tau2!r}") from None
        if not (np.isfinite(lam_given) and lam_given > 0.0 and np.isfinite(t2_given) and t2_given >= 0.0):
            raise ValueError(f"tau2 = (lambda, tau^2) needs lambda > 0 and tau^2 >= 0, got {tau2!r}")
    if n < int(n_folds):
        raise ValueError(f"{n} PSF stars; at least {int(n_folds)} are needed (one per fold)")
    sums = st.sum(axis=(1, 2))
    if not np.all(np.isfinite(sums)) or np.any(sums <= 0.0):
        raise ValueError(f"{int(np.sum(~(sums > 0)))} training stamps have a non-positive or "
                         "non-finite sum")
    d = (st / sums[:, None, None]).reshape(n, s * s)
    cands = tuple(candidates) if candidates is not None else default_candidates(image_shape)
    extra = tuple(extra_candidates)
    if folds is None:
        folds = assign_folds(x, y, image_shape, cands, n_folds)
    folds = np.asarray(folds, dtype=np.int64)
    k = int(n_folds)
    mix = mixing_grid(image_shape)
    timing = {}
    omega0 = 1.0 / a
    est: dict = {}
    if tau2 is None:
        tp = time.perf_counter()
        _, mix_a, gram_a, _ = _cv_pass(d, x, y, omega0, folds, k, cands, extra, image_shape, mix, False)
        ws = _weights_at(mix[0], mix[1], mix_a, x, y)
        r2mix = np.einsum("sm,smk,sk->s", ws, gram_a, ws)
        lam, t2, est = tgm_floor(r2mix, a)
        passes = 2
        timing["pass_a"] = time.perf_counter() - tp
    else:
        lam, t2 = lam_given, t2_given
        r2mix = np.zeros(0)
        passes = 1
    omega = 1.0 / (lam * a + t2)
    tp = time.perf_counter()
    fits, mix_w, gram, loss_folds = _cv_pass(d, x, y, omega, folds, k, cands, extra, image_shape, mix, True)
    timing["pass_b"] = time.perf_counter() - tp
    diag = image_diagonal(image_shape)
    rec_c = []
    for mi, (c, ft) in enumerate(zip(cands, fits)):
        rec_c.append({
            "name": f"h={'inf' if c.is_global else f'D/{diag / c.h:.6g}'} p={int(c.p)}",
            "h_px": c.h, "h_over_diag": c.h / diag, "p": int(c.p),
            "loss": float(loss_folds[mi].sum()), "loss_folds": [float(v) for v in loss_folds[mi]],
            "n_nodes": int(ft.coefs.shape[0]), "n_nodes_p0": int(ft.n_p0),
            "n_nodes_singular": int(ft.n_singular), "n_nodes_unreached": int(ft.n_unreached),
            "max_mix_weight": float(mix_w[..., mi].max())})
    for ei, xc in enumerate(extra):
        mi = len(cands) + ei
        rec_c.append({"name": str(getattr(xc, "name", f"extra_{ei}")), "h_px": None, "p": None,
                      "loss": float(loss_folds[mi].sum()), "loss_folds": [float(v) for v in loss_folds[mi]],
                      "n_nodes_p0": 0, "n_nodes_singular": 0,
                      "max_mix_weight": float(mix_w[..., mi].max())})
    med_a = float(np.median(a))
    record = {
        "n_stars": int(n),
        "n_eff_stars": float(omega.sum() ** 2 / np.sum(omega * omega)),
        "n_eff_stars_no_floor": float(omega0.sum() ** 2 / np.sum(omega0 * omega0)),
        "tau2": float(t2), "tau2_over_median_a": float(t2 / med_a), "median_a": med_a,
        "lam": float(lam), "theta": float(t2 / lam),
        "a_min_over_theta": (float(np.min(a)) / float(t2 / lam)) if t2 > 0 else float("inf"),
        "ms_iters": (list(est["ms_iters"]) if est else None), "n_B": est.get("n_B"), "n_F": est.get("n_F"),
        "N50": est.get("N50"), "ms_global_gap": (list(est["ms_global_gap"]) if est else None),
        "tau2_zero": est.get("tau2_zero"), "tau2_fallback": est.get("fallback"),
        "B_excess": est.get("B_excess"), "lambda_identified": est.get("lambda_identified"),
        "lam_hat": est.get("lam_hat"), "F_median_r2_over_a": est.get("F_median_r2_over_a"),
        "tau2_rule": "TGM (design a.1b 3rd ed.) + lambda identification check (prereg_a1b_r2 1)",
        "passes": int(passes), "tau2_given": tau2 is not None,
        "n_folds": k, "fold_sizes": [int(np.sum(folds == f)) for f in range(k)],
        "fold_rule": "finest training-grid cells in raster order, round-robin",
        "mix": {"rule": "local stacking (design v2.5 1.3)", "nodes_h_over_diag": 1.0 / 8.0,
                "kernel_g_over_diag": 1.0 / 8.0, "nx": int(len(mix[0])), "ny": int(len(mix[1])),
                "ridge": MIX_RIDGE,
                "mean_active": float(np.mean(np.sum(mix_w > 0, axis=-1)))},
        "candidates": rec_c,
        "timing_s": dict(timing, total_fit=time.perf_counter() - t0),
    }
    return Mixture(list(fits), mix[0], mix[1], mix_w, s, record, star_r2=r2mix), folds


_SHIFT_NORM: list = []


def _lanczos3_taps(d: float) -> np.ndarray:
    """The engine's normalised 7 Lanczos-3 taps at offsets [-3..3] + d (psf_from_stars)."""
    xx = np.arange(-3, 4) + float(d)
    ax = np.abs(xx)
    sx = np.where(ax < 1e-12, 1.0, xx)
    val = 3.0 * np.sin(np.pi * sx) * np.sin(np.pi * sx / 3.0) / (np.pi * sx) ** 2
    wts = np.where(ax < 1e-12, 1.0, np.where(ax >= 3.0, 0.0, val))
    return wts / wts.sum()


def lanczos_shift_norm_2d(stamp_size: int = 55) -> float:
    """Operator norm of the engine's sub-pixel Lanczos-3 shift on a stamp (computed, not a constant).

    The shift is separable; per axis it is the banded stamp_size x stamp_size matrix of the
    normalised taps (zero outside the stamp).  Its largest singular value, maximised over 201
    shifts in [-0.5, 0.5] px, squared, bounds the 2-D shift's l2 norm (about 1.054, 2.26 x tighter
    than the l1 bound).  It bounds ||S delta||, not the render-error FRACTIONS (their denominator
    ||S phi|| can shrink too), so in the D9-1 rule it is an empirical margin factor: the measured
    ratio of shifted to unshifted fractions is at most 1.012 (review_a.1 C-2)."""
    if not _SHIFT_NORM:
        s = int(stamp_size)
        best = 0.0
        for dd in np.linspace(-0.5, 0.5, 201):
            taps = _lanczos3_taps(dd)
            mtx = np.zeros((s, s))
            for i in range(s):
                for k in range(7):
                    j = i + k - 3
                    if 0 <= j < s:
                        mtx[i, j] = taps[k]
            best = max(best, float(np.linalg.svd(mtx, compute_uv=False)[0]))
        _SHIFT_NORM.append(best * best)
    return _SHIFT_NORM[0]


def _fractions(delta: np.ndarray, phi: np.ndarray, s2: float):
    """Render-error fractions (design v2.5 3rd ed. 1.2): flux |<d, phi>| / <phi, phi> and pixel
    norm ||d|| / ||phi||, both times the shift norm s2.  Arrays (..., P) -> (...), (...)."""
    pp = np.einsum("...p,...p->...", phi, phi)
    with np.errstate(divide="ignore", invalid="ignore"):
        flux = s2 * np.abs(np.einsum("...p,...p->...", delta, phi)) / pp
        pix = s2 * np.linalg.norm(delta, axis=-1) / np.sqrt(pp)
    return flux, pix


def _refine1_errors(mixlike, gx, gy):
    """Exact mixture vs corner-node bilinear interpolation on the refinement-1 cells.

    Returns dict: nodes (ny, nx, P); delta and phi at the cell centres (ncy, ncx, P), at the x-edge
    midpoints (ny, ncx, P) and at the y-edge midpoints (ncy, nx, P)."""
    def ev(px, py):
        return mixlike.evaluate(px.ravel(), py.ravel()).reshape(px.shape + (-1,))
    X, Y = np.meshgrid(gx, gy)
    nodes = ev(X, Y)
    xm, ym = 0.5 * (gx[:-1] + gx[1:]), 0.5 * (gy[:-1] + gy[1:])
    Xc, Yc = np.meshgrid(xm, ym)
    pc = ev(Xc, Yc)
    Xe, Ye = np.meshgrid(xm, gy)
    pxe = ev(Xe, Ye)
    Xf, Yf = np.meshgrid(gx, ym)
    pye = ev(Xf, Yf)
    return {
        "nodes": nodes,
        "d_center": pc - 0.25 * (nodes[:-1, :-1] + nodes[:-1, 1:] + nodes[1:, :-1] + nodes[1:, 1:]),
        "p_center": pc,
        "d_xedge": pxe - 0.5 * (nodes[:, :-1] + nodes[:, 1:]),
        "p_xedge": pxe,
        "d_yedge": pye - 0.5 * (nodes[:-1] + nodes[1:]),
        "p_yedge": pye,
    }


def _verify_cells(mixlike, gx, gy, cells, n: int, s2: float):
    """Actual fractions at every sub-cell centre and sub-cell edge midpoint of the given
    refinement-1 cells at refinement n (edge midpoints: review_a.1 C-1 -- a saddle-shaped error
    concentrates there).  Returns (max flux fraction, max pixel fraction, number of points)."""
    if not len(cells):
        return 0.0, 0.0, 0
    a = np.arange(n + 1) / n
    c = (np.arange(n) + 0.5) / n
    k = len(cells)
    grids = {"nodes": (a, a), "cent": (c, c), "xmid": (c, a), "ymid": (a, c)}
    pts = {key: ([], []) for key in grids}
    for (j, i) in cells:
        lx, ly = gx[i + 1] - gx[i], gy[j + 1] - gy[j]
        for key, (fx_, fy_) in grids.items():
            X, Y = np.meshgrid(gx[i] + lx * fx_, gy[j] + ly * fy_)
            pts[key][0].append(X.ravel())
            pts[key][1].append(Y.ravel())
    ev = {key: mixlike.evaluate(np.concatenate(px), np.concatenate(py))
          for key, (px, py) in pts.items()}
    nodes = ev["nodes"].reshape(k, n + 1, n + 1, -1)
    cent = ev["cent"].reshape(k, n, n, -1)
    xmid = ev["xmid"].reshape(k, n + 1, n, -1)                 # [cell, row (node line), column (mid)]
    ymid = ev["ymid"].reshape(k, n, n + 1, -1)                 # [cell, row (mid), column (node line)]
    fs, qs = [], []
    for val, interp in (
            (cent, 0.25 * (nodes[:, :-1, :-1] + nodes[:, :-1, 1:] + nodes[:, 1:, :-1] + nodes[:, 1:, 1:])),
            (xmid, 0.5 * (nodes[:, :, :-1] + nodes[:, :, 1:])),
            (ymid, 0.5 * (nodes[:, :-1, :] + nodes[:, 1:, :]))):
        f, q = _fractions(val - interp, val, s2)
        fs.append(float(np.max(f)))
        qs.append(float(np.max(q)))
    return max(fs), max(qs), int(k * (n * n + 2 * n * (n + 1)))


def select_render_refine(mixlike, cands, shape, *, refine: int | None = None, image_id: str = ""):
    """D9-1 (design v2.5 3rd ed. 1.2): render refinement n_b of the finest training grid.

    1. On the refinement-1 cells, the exact mixture vs its corner-node bilinear interpolation at the
       cell centre and the four edge midpoints gives the flux fraction |<d, phi>| / <phi, phi> and the
       pixel-norm fraction ||d|| / ||phi|| (both x the shift norm, :func:`lanczos_shift_norm_2d`);
       a cell's value is the maximum over its five points.
    2. Estimate at refinement n = value / n^2; n_est = the smallest n with flux <= 1e-3 and pixel norm
       <= 1e-2 in every cell.
    3. Check: the top 5% of cells by estimate (max of the two fractions over their limits) at every
       sub-cell centre and sub-cell edge midpoint of refinement n with the exact mixture; any excess
       -> n + 1, checked again.  Beyond 16 the run fails (error.kind psf_render_spacing).  The check is
       a sample of the worst cells, not a guarantee for every point (review_a.1 C-1).
    ``refine`` forces n (still estimated and checked for the record).  Returns (n_b, record, nodes1)."""
    s2 = lanczos_shift_norm_2d(int(round(math.sqrt(mixlike.evaluate(np.zeros(1), np.zeros(1)).size))))
    gx, gy = render_grid(cands, shape, 1)
    e = _refine1_errors(mixlike, gx, gy)
    fc, qc = _fractions(e["d_center"], e["p_center"], s2)
    fx, qx = _fractions(e["d_xedge"], e["p_xedge"], s2)
    fy, qy = _fractions(e["d_yedge"], e["p_yedge"], s2)
    f1 = np.maximum.reduce([fc, fx[:-1], fx[1:], fy[:, :-1], fy[:, 1:]])
    q1 = np.maximum.reduce([qc, qx[:-1], qx[1:], qy[:, :-1], qy[:, 1:]])
    fmax, qmax = float(np.max(f1)), float(np.max(q1))
    n_est = max(1, math.ceil(math.sqrt(max(0.0, fmax / FLUX_FRAC_LIMIT - 1e-12))),
                math.ceil(math.sqrt(max(0.0, qmax / PIX_FRAC_LIMIT - 1e-12))))
    score = np.maximum(f1 / FLUX_FRAC_LIMIT, q1 / PIX_FRAC_LIMIT)
    n_top = max(1, math.ceil(VERIFY_TOP_FRACTION * score.size))
    order = np.argsort(-score.ravel(), kind="stable")[:n_top]
    cells = [tuple(int(v) for v in np.unravel_index(o, score.shape)) for o in order]
    verified = {}
    n = n_est if refine is None else int(refine)
    while True:
        if n > MAX_RENDER_REFINE:
            raise RunContractError(
                RENDER_SPACING_ERROR_KIND,
                f"PSF model {image_id!r}: the render error stays above the D9-1 limits (flux "
                f"{FLUX_FRAC_LIMIT}, pixel norm {PIX_FRAC_LIMIT}) up to refinement {MAX_RENDER_REFINE}",
                image_id=image_id, verified=verified)
        vf, vq, npts = _verify_cells(mixlike, gx, gy, cells, n, s2)
        verified[str(n)] = {"flux_frac_max": vf, "pix_frac_max": vq, "n_cells": len(cells), "n_points": npts}
        ok = vf <= FLUX_FRAC_LIMIT and vq <= PIX_FRAC_LIMIT
        if refine is not None or ok:
            break
        n += 1
    # reference only: the a.2 analytic bound with the narrowed shift norm (not used to choose n)
    bx = np.maximum(np.abs(e["d_xedge"][:-1]), np.abs(e["d_xedge"][1:]))
    by = np.maximum(np.abs(e["d_yedge"][:, :-1]), np.abs(e["d_yedge"][:, 1:]))
    nrm = np.linalg.norm(e["nodes"], axis=-1)
    phin = np.minimum(np.minimum(nrm[:-1, :-1], nrm[:-1, 1:]), np.minimum(nrm[1:, :-1], nrm[1:, 1:]))
    dd = s2 * np.linalg.norm(bx + by, axis=-1) / float(n * n)
    with np.errstate(divide="ignore", invalid="ignore"):
        bb = np.where(phin > dd, (dd * phin + dd * dd) / (phin - dd) ** 2, np.inf)
    rec = {"refine": int(n), "n_estimated": int(n_est), "forced": refine is not None,
           "est_flux_frac_max": fmax / (n * n), "est_pix_frac_max": qmax / (n * n),
           "est_flux_frac_max_refine1": fmax, "est_pix_frac_max_refine1": qmax,
           "verified": verified, "shift_norm_2d": s2,
           "limits": {"flux_frac": FLUX_FRAC_LIMIT, "pix_frac": PIX_FRAC_LIMIT},
           "ref_analytic_b_max": float(np.max(bb)),
           "rule": ("coarsest integer refinement of the finest training grid whose estimated render "
                    "error (exact mixture vs bilinear at refinement-1 cell centres and edge midpoints, "
                    "scaled 1/n^2, x shift-norm margin) meets flux <= 1e-3 and pixel norm <= 1e-2, checked at "
                    "every sub-cell centre and edge midpoint of the top 5% cells (design v2.5 3rd ed. "
                    "1.2, review_a.1 C-1; a sample of the worst cells, not a guarantee)")}
    return int(n), rec, e["nodes"]


def build_psf_model(stamps, star_x, star_y, noise_a, image_shape, *, tau2=None,
                    image_id: str = "", band: str = "", frame: str = "",
                    candidates: Sequence[Candidate] | None = None, extra_candidates=(),
                    n_folds: int = N_FOLDS, render_refine: int | None = None,
                    work_frame_offset=(0, 0)) -> PsfModel:
    """Build the model of one image from registered training stamps.

    stamps: (N, S, S) registered stamps (centred on the stamp's central pixel; any positive
    sum -- each is normalised to sum 1 in float64); star_x, star_y: (N,) work-frame positions;
    noise_a, tau2, extra_candidates: see :func:`fit_mixture`; image_shape: (H, W).
    render_refine=None: n_b by :func:`select_render_refine` (D9-1), failing (error.kind
    psf_render_spacing) beyond 16; an integer forces it (tests).
    Pure function: no I/O, no global state.  Raises :class:`PsfModelIntegralError` if a
    render-node stamp does not sum to 1."""
    t0 = time.perf_counter()
    cands = tuple(candidates) if candidates is not None else default_candidates(image_shape)
    mix, folds = fit_mixture(stamps, star_x, star_y, noise_a, image_shape, tau2=tau2,
                             candidates=cands, extra_candidates=extra_candidates, n_folds=n_folds)
    s = mix.stamp_size
    tr = time.perf_counter()
    n_b, rrec, nodes1 = select_render_refine(mix, cands, image_shape, refine=render_refine,
                                             image_id=image_id)
    if n_b == 1:
        node_x, node_y = render_grid(cands, image_shape, 1)
        node_stamps = nodes1.reshape(len(node_y), len(node_x), s, s)
    else:
        node_x, node_y = render_grid(cands, image_shape, n_b)
        gx, gy = np.meshgrid(node_x, node_y)
        node_stamps = mix.evaluate(gx.ravel(), gy.ravel()).reshape(len(node_y), len(node_x), s, s)
    t_render = time.perf_counter() - tr
    dev = check_integral(node_stamps, image_id)
    rec = dict(mix.record)
    rec["integral_max_abs_dev"] = dev
    rec["render"] = dict(rrec, bytes_float32=int(len(node_x) * len(node_y) * s * s * 4))
    rec["timing_s"] = dict(rec["timing_s"], render=t_render, total=time.perf_counter() - t0)
    rec["created"] = datetime.now().astimezone().isoformat(timespec="seconds")
    a_ = np.asarray(noise_a, dtype=np.float64)
    return PsfModel(
        image_id=image_id, band=band, frame=str(frame), image_shape=tuple(image_shape),
        node_x=node_x, node_y=node_y, stamps=node_stamps,
        star_x=np.asarray(star_x, dtype=np.float64), star_y=np.asarray(star_y, dtype=np.float64),
        star_weight=1.0 / (rec["lam"] * a_ + rec["tau2"]), star_fold=folds,
        record=rec, work_frame_offset=tuple(work_frame_offset),
        mix_node_x=mix.mix_node_x, mix_node_y=mix.mix_node_y, mix_weights=mix.mix_weights,
        star_a=a_, star_r2=np.asarray(mix.star_r2, dtype=np.float64))


def render_stamp_bytes(models) -> int:
    """Device bytes of the float32 render-node stamps of the given models (exact)."""
    return int(sum(len(m.node_x) * len(m.node_y) * m.stamp_size * m.stamp_size * 4 for m in models))


def require_render_memory(need_bytes: int, free_bytes: int, *, card, other_users=()) -> None:
    """D9-1 answer 1: fail the run (error.kind psf_render_memory) when a card's free memory is
    short of the need (field stack + node stamps + narrowest kernel, summed by the caller) --
    never coarsen the render grid silently.  The caller measures the free bytes (NVML)."""
    if int(need_bytes) > int(free_bytes):
        raise RunContractError(
            RENDER_MEMORY_ERROR_KIND,
            f"GPU {card}: the PSF render-node stamps do not fit: need {int(need_bytes)} B, free "
            f"{int(free_bytes)} B" + (f" (other users on the card: {', '.join(map(str, other_users))})"
                                       if other_users else ""),
            card=card, need_bytes=int(need_bytes), free_bytes=int(free_bytes),
            other_users=list(map(str, other_users)))


__all__ = [
    "FORMAT_NAME", "FORMAT_VERSION", "INTEGRAL_TOL", "INTEGRAL_ERROR_KIND", "N_FOLDS",
    "FLUX_FRAC_LIMIT", "PIX_FRAC_LIMIT", "RENDER_MEMORY_ERROR_KIND", "RENDER_SPACING_ERROR_KIND",
    "Candidate", "Mixture", "PsfModel", "PsfModelIntegralError",
    "assign_folds", "build_psf_model", "check_integral", "check_unique_image_ids",
    "default_candidates", "fit_mixture", "image_diagonal", "image_id_for", "lanczos_shift_norm_2d",
    "load_psf_model", "local_stack_weights", "mixing_grid", "refine_factor", "render_bilinear",
    "render_grid", "render_stamp_bytes", "require_render_memory", "save_psf_model", "select_render_refine",
    "stack_simplex",
    "tgm_floor", "training_grid", "write_index",
]
