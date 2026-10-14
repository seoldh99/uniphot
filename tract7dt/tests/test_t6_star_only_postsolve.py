"""T6/G2 (2026-08-23): the star-only post-solve render is BITWISE the STAR
rows of the generic mixed renderer.

The blank-response correction (solver_grid) and the confusion error model
(error_model) render every source's unit stamp per band. For an all-STAR
grid they now call :func:`source_model.render_star_stamps` instead of
:func:`source_model.render_source_stamps`. The arc covenant for a perf item
is "no science change", so this test pins exact equality (not closeness) on
the CPU backend with x64 on (the worker's runtime config), for several grid
origins (the helper must reproduce the GRID-LOCAL rounding of
render_source_stamp_at) and both render backends.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from tract7dt.fit.grid_gpu.psf_polynomial import PsfPolynomialJax
from tract7dt.fit.grid_gpu.source_model import (
    SOURCE_EXP, SOURCE_STAR, render_source_stamps, render_star_stamps)

S, NT = 55, 10


def _psf(seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:S, :S] - S // 2
    g = np.exp(-0.5 * (xx ** 2 + yy ** 2) / 2.2 ** 2)
    g /= g.sum()
    coefs = np.zeros((S, S, NT), np.float32)
    coefs[..., 0] = g
    coefs[..., 1:] = (rng.normal(0, 1e-4, (S, S, NT - 1)).astype(np.float32)
                      * g[..., None])
    return PsfPolynomialJax(coefs=jnp.asarray(coefs), x_origin=3300.0,
                            x_scale=3300.0, y_origin=5000.0, y_scale=5000.0,
                            H_psf=S, W_psf=S)


def _sources(seed, n, x0, y0, bw=400, bh=300):
    rng = np.random.default_rng(seed)
    x = (x0 + rng.uniform(0, bw, n)).astype(np.float32)
    y = (y0 + rng.uniform(0, bh, n)).astype(np.float32)
    return jnp.asarray(x), jnp.asarray(y)


@pytest.mark.parametrize("backend", ["direct", "fft"])
@pytest.mark.parametrize("x0,y0", [(0, 0), (1234, 567), (8675, 2573)])
@pytest.mark.parametrize("n", [1, 37, 64])
def test_star_only_equals_full_chain_star_rows_bitwise(backend, x0, y0, n):
    psf = _psf(7)
    px, py = _sources(n * 31 + x0, n, x0, y0)
    st = jnp.full((n,), SOURCE_STAR, jnp.int32)
    zeros = jnp.zeros((n,), jnp.float32)
    full = np.asarray(render_source_stamps(
        psf, px, py, x0, y0, st, zeros, jnp.ones((n,), jnp.float32), zeros,
        sersic_n=jnp.ones((n,), jnp.float32), render_backend=backend))
    star = np.asarray(render_star_stamps(psf, px, py, x0, y0))
    assert star.shape == full.shape == (n, S, S)
    assert star.dtype == full.dtype
    # exact, not allclose: same primitives, same batch, same order
    assert np.array_equal(star, full), (
        f"max|d| = {np.abs(star - full).max():.3e} ({backend}, origin "
        f"{x0},{y0}, n={n})")


def test_helper_is_only_for_all_star_sets():
    """A mixed set must NOT be routed here: an EXP row rendered by the
    star-only helper is the bare PSF, not the EXP model -- the callers
    dispatch on the host-known types (solver_grid `_br_ext.any()`,
    error_model `_all_star_grid`)."""
    psf = _psf(3)
    n = 5
    px, py = _sources(11, n, 100, 100)
    st = jnp.asarray([SOURCE_STAR, SOURCE_EXP, SOURCE_STAR, SOURCE_EXP,
                      SOURCE_STAR], jnp.int32)
    logre = jnp.asarray([0, 1.2, 0, 0.8, 0], jnp.float32)
    full = np.asarray(render_source_stamps(
        psf, px, py, 100, 100, st, logre, jnp.ones((n,), jnp.float32),
        jnp.zeros((n,), jnp.float32), render_backend="direct"))
    star = np.asarray(render_star_stamps(psf, px, py, 100, 100))
    is_star = np.asarray(st) == SOURCE_STAR
    assert np.array_equal(star[is_star], full[is_star])
    assert not np.array_equal(star[~is_star], full[~is_star])
