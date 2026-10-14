"""Image-grid divider.

Splits the full per-band image into N x N grids (typically 2 x 2 = 4
grids, one per GPU). Each grid carries:

- Its data + invvar + bad-mask slice (with an optional pixel HALO so
  sources near the boundary still see their full PSF stamp).
- The list of catalog source indices that fall inside the grid (at
  least their centroid; sources outside the grid are not fit there).
- The per-grid pixel offset (x0, y0) into the full image, so the
  spatial polynomial PSF + bg models can use the same global (x, y)
  normalization across all grids.

The HALO size is chosen so that a source at the very edge of the
core grid still has its full PSF stamp inside the grid+halo region.
For 55-px PSF stamps and grids that are 1/4 of the image, halo = 27
covers the ~half-stamp radius cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class GridDef:
    """One grid's geometry + source assignment.

    Attributes
    ----------
    grid_id : int
        0-indexed grid number.
    grid_row, grid_col : int
        Position in the (n_rows x n_cols) grid layout.
    x0_core, x1_core : int
        Core grid bbox in full-image x coords. ``x1_core`` is exclusive.
    y0_core, y1_core : int
        Same for y.
    x0_pad, x1_pad : int
        Padded grid bbox (= core +/- halo, clipped to image bounds).
    y0_pad, y1_pad : int
        Same for y.
    halo : int
        Halo width applied to the core to get the padded bbox.
    source_indices : np.ndarray
        Indices into the FULL input catalog of sources whose
        (x_pix, y_pix) center falls inside the CORE bbox of this grid.
        These are the sources we fit + record in the per-grid output.
    halo_source_indices : np.ndarray
        Indices of sources whose center falls in the HALO region of
        this grid (in the core of a NEIGHBOR grid). These are included
        in the forward model so they don't bias nearby fits, but their
        fitted parameters are NOT recorded here (the neighbor records
        them).
    """
    grid_id: int
    grid_row: int
    grid_col: int
    x0_core: int
    x1_core: int
    y0_core: int
    y1_core: int
    x0_pad: int
    x1_pad: int
    y0_pad: int
    y1_pad: int
    halo: int
    source_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    halo_source_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    @property
    def core_h(self) -> int:
        return self.y1_core - self.y0_core

    @property
    def core_w(self) -> int:
        return self.x1_core - self.x0_core

    @property
    def pad_h(self) -> int:
        return self.y1_pad - self.y0_pad

    @property
    def pad_w(self) -> int:
        return self.x1_pad - self.x0_pad


def divide_image_into_grids(
    image_h: int, image_w: int,
    n_rows: int, n_cols: int,
    halo: int,
) -> list[GridDef]:
    """Build a flat list of ``GridDef`` for an ``n_rows x n_cols``
    grid layout over an image of size ``(image_h, image_w)``."""
    if n_rows < 1 or n_cols < 1:
        raise ValueError(f"n_rows/n_cols must be >= 1; got {n_rows}, {n_cols}")
    if halo < 0:
        raise ValueError(f"halo must be >= 0; got {halo}")

    grids: list[GridDef] = []
    # Core x and y boundaries (inclusive starts, exclusive ends).
    x_edges = np.linspace(0, image_w, n_cols + 1, dtype=np.int64)
    y_edges = np.linspace(0, image_h, n_rows + 1, dtype=np.int64)

    grid_id = 0
    for r in range(n_rows):
        for c in range(n_cols):
            x0c = int(x_edges[c]); x1c = int(x_edges[c + 1])
            y0c = int(y_edges[r]); y1c = int(y_edges[r + 1])
            x0p = max(0, x0c - halo)
            x1p = min(image_w, x1c + halo)
            y0p = max(0, y0c - halo)
            y1p = min(image_h, y1c + halo)
            grids.append(GridDef(
                grid_id=grid_id,
                grid_row=r, grid_col=c,
                x0_core=x0c, x1_core=x1c,
                y0_core=y0c, y1_core=y1c,
                x0_pad=x0p, x1_pad=x1p,
                y0_pad=y0p, y1_pad=y1p,
                halo=halo,
            ))
            grid_id += 1
    return grids


def assign_sources_to_grids(
    grids: list[GridDef],
    src_x: np.ndarray, src_y: np.ndarray,
) -> None:
    """Populate ``GridDef.source_indices`` and ``halo_source_indices``
    in-place from the catalog source positions ``(src_x, src_y)``.

    A source is a CORE source of grid g iff its center is in g's core
    bbox; a source is a HALO source of grid g iff its center is in
    g's padded bbox but not in g's core (i.e., it's a core source of
    a neighbor and we include it in g's forward model).
    """
    for g in grids:
        in_core = (
            (src_x >= g.x0_core) & (src_x < g.x1_core) &
            (src_y >= g.y0_core) & (src_y < g.y1_core)
        )
        in_pad = (
            (src_x >= g.x0_pad) & (src_x < g.x1_pad) &
            (src_y >= g.y0_pad) & (src_y < g.y1_pad)
        )
        in_halo = in_pad & (~in_core)
        g.source_indices = np.where(in_core)[0].astype(np.int64)
        g.halo_source_indices = np.where(in_halo)[0].astype(np.int64)


def slice_band_image(
    band_data: np.ndarray, band_invvar: np.ndarray, band_pix_good: np.ndarray,
    grid: GridDef,
    *,
    copy: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract this grid's (data, invvar, pix_good) slice from a
    full-band image."""
    sl_y = slice(grid.y0_pad, grid.y1_pad)
    sl_x = slice(grid.x0_pad, grid.x1_pad)
    data = band_data[sl_y, sl_x]
    invvar = band_invvar[sl_y, sl_x]
    pix_good = band_pix_good[sl_y, sl_x]
    if copy:
        data = data.copy()
        invvar = invvar.copy()
        pix_good = pix_good.copy()
    return data, invvar, pix_good


__all__ = [
    "GridDef",
    "divide_image_into_grids",
    "assign_sources_to_grids",
    "slice_band_image",
]
