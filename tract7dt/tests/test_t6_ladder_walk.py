"""T6/I1 (2026-08-23): the rung search is a pure walk, and the class-level
table fill never changes the production width.

`island_solver.ladder_walk` is the 2026-08-02 inline search factored out;
this pins it against a verbatim copy of the original loop on random probe
oracles (including non-monotone and failing probes). The I1 extension runs
the same walk once more at the ladder ceiling through a MEMOISED oracle,
so: (i) the production width is the tile-limited walk's result, unchanged;
(ii) no rung is probed twice; (iii) the extension only adds rungs.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from tract7dt.fit.grid_gpu.island_solver import (
    _LADDER_MAX_WIDTH, effective_rung_limit, ladder_walk)


def _original_walk(_probe, limit, budget_bytes, c1, dom):
    """Verbatim copy of the pre-T6 inline loop (island_solver.py, 2026-08-02)."""
    best_w, best_c, best_dom = 1, c1, dom
    w = min(limit, 1 << max(0, (budget_bytes // max(c1, 1)).bit_length() - 1))
    while w > best_w:
        c, dm = _probe(w)
        if c is not None and c <= budget_bytes:
            best_w, best_c, best_dom = w, c, dm
            break
        w //= 2
    while best_w * 2 <= limit:
        nxt = best_w * 2
        if best_c * 2 > budget_bytes:
            break
        c, dm = _probe(nxt)
        if c is None or c > budget_bytes:
            break
        best_w, best_c, best_dom = nxt, c, dm
    return best_w, best_c, best_dom


def _oracle(rng, fixed, per_isl, fail_at=None, kink=None):
    def probe(w):
        if fail_at is not None and w >= fail_at:
            return None, "fail"
        c = fixed + per_isl * w
        if kink is not None and w >= kink:          # non-monotone bump
            c += int(0.5 * per_isl * w)
        return int(c), "step_v" if w > 1 else "chi2_v"
    return probe


@pytest.mark.parametrize("seed", range(200))
def test_ladder_walk_is_the_original_loop(seed):
    rng = np.random.default_rng(seed)
    fixed = int(rng.integers(1, 200)) * 2 ** 20
    per_isl = int(rng.integers(1, 50)) * 2 ** 16
    budget = int(rng.integers(1, 24)) * 2 ** 30
    n_isl = int(rng.integers(1, 60000))
    limit = effective_rung_limit(n_isl, _LADDER_MAX_WIDTH, 1)
    fail_at = int(rng.choice([None, 512, 2048, 4096], p=[0.7, 0.1, 0.1, 0.1]) or 0) or None
    kink = int(rng.choice([None, 64, 1024], p=[0.8, 0.1, 0.1]) or 0) or None
    probe = _oracle(rng, fixed, per_isl, fail_at, kink)
    c1, d1 = probe(1)
    calls_new, calls_old = [], []
    new = ladder_walk(lambda w: (calls_new.append(w), probe(w))[1],
                      limit, budget, c1, d1)
    old = _original_walk(lambda w: (calls_old.append(w), probe(w))[1],
                         limit, budget, c1, d1)
    assert new == old
    assert calls_new == calls_old          # same rungs, same order


@pytest.mark.parametrize("seed", range(60))
def test_class_extension_keeps_width_and_never_reprobes(seed):
    rng = np.random.default_rng(seed)
    fixed = int(rng.integers(1, 200)) * 2 ** 20
    per_isl = int(rng.integers(1, 50)) * 2 ** 16
    budget = int(rng.integers(1, 24)) * 2 ** 30
    n_isl = int(rng.integers(1, 5000))
    limit = effective_rung_limit(n_isl, _LADDER_MAX_WIDTH, 1)
    probe = _oracle(rng, fixed, per_isl)
    c1, d1 = probe(1)
    memo, calls = {}, []

    def once(w):
        if w not in memo:
            calls.append(w)
            memo[w] = probe(w)
        return memo[w]

    tile = ladder_walk(once, limit, budget, c1, d1)
    n_tile = len(calls)
    cls = ladder_walk(once, _LADDER_MAX_WIDTH, budget, c1, d1)
    # (i) production decision = the tile walk; the class walk is a superset
    assert tile == _original_walk(probe, limit, budget, c1, d1)
    assert cls[0] >= tile[0] and cls[0] <= _LADDER_MAX_WIDTH
    assert min(cls[0], limit) == tile[0]
    # (ii) no rung probed twice; (iii) extension adds, never removes
    assert len(calls) == len(set(calls))
    assert len(calls) >= n_tile
    assert all(w > tile[0] or w in calls[:n_tile] for w in calls[n_tile:])
