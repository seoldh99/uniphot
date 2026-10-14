"""Gaia-like seeds (per-star POS_ERR) and per-image offset fields Delta_b.

The white frame is the reference pixel frame (where Gaia positions land through the white
WCS).  In image b a star's light is rendered at x_b = x_white + Delta_b(x_white, y_white).
Seeds (what an input catalogue would carry) are x_white + N(0, POS_ERR^2) per axis; a chosen
share of seeds is moved by an extra fixed distance (wrong Gaia position, design A4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OffsetField:
    """Delta_b(xh, yh) = sum_{(p, q)} c[(p, q)] xh^p yh^q, one coefficient dict per axis (px)."""
    shape: tuple
    cx: dict
    cy: dict

    def __call__(self, x, y):
        """Offset (dx, dy) in px at white-frame positions (x, y)."""
        h, w = self.shape
        xh = 2.0 * np.asarray(x, dtype=float) / (w - 1) - 1.0
        yh = 2.0 * np.asarray(y, dtype=float) / (h - 1) - 1.0
        dx = sum(c * xh ** p * yh ** q for (p, q), c in self.cx.items()) + 0.0 * xh
        dy = sum(c * xh ** p * yh ** q for (p, q), c in self.cy.items()) + 0.0 * yh
        return dx, dy


def offset_a1(shape) -> OffsetField:
    """Design A1 offset: constant (-0.134, -0.026) px, 0.05 px linear term and 0.02 px cubic
    term at the field edge (dx = -0.134 + 0.05 xh + 0.02 xh^3, dy = -0.026 + 0.05 yh + 0.02 yh^3)."""
    return OffsetField(tuple(shape), {(0, 0): -0.134, (1, 0): 0.05, (3, 0): 0.02},
                       {(0, 0): -0.026, (0, 1): 0.05, (0, 3): 0.02})


@dataclass(frozen=True)
class Seeds:
    """Seed positions (white frame), per-axis POS_ERR (px) and the wrong-position mask."""
    x: np.ndarray
    y: np.ndarray
    pos_err: np.ndarray
    wrong: np.ndarray


def gaia_seeds(x_true, y_true, rng: np.random.Generator, pos_err_range=(0.002, 0.03),
               wrong_fraction: float = 0.0, wrong_offset: float = 0.5) -> Seeds:
    """Gaia-like seeds: POS_ERR log-uniform in pos_err_range, seed = truth + N(0, POS_ERR^2)
    per axis; round(wrong_fraction n) seeds also move by wrong_offset px in a random direction."""
    x_true = np.atleast_1d(np.asarray(x_true, dtype=float))
    y_true = np.atleast_1d(np.asarray(y_true, dtype=float))
    n = len(x_true)
    lo, hi = pos_err_range
    pos_err = np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))
    x = x_true + rng.normal(0.0, pos_err)
    y = y_true + rng.normal(0.0, pos_err)
    wrong = np.zeros(n, dtype=bool)
    n_wrong = int(round(wrong_fraction * n))
    if n_wrong:
        idx = rng.choice(n, size=n_wrong, replace=False)
        ang = rng.uniform(0.0, 2.0 * np.pi, size=n_wrong)
        x[idx] += wrong_offset * np.cos(ang)
        y[idx] += wrong_offset * np.sin(ang)
        wrong[idx] = True
    return Seeds(x, y, pos_err, wrong)
