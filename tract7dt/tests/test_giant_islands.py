"""Giant islands keep the seed geometry and solve nothing (REGISTER R16).

A giant island (bigger than the largest bucket class, more members than
``joint_nsrc_cap``, or measured memory-infeasible one island at a time) gets
no island fit. Its members keep the seed position and shape; pass 2 measures
their fluxes at that geometry like every other source. The removed host
flux-only solve (``_fit_giants``) produced fluxes nothing read, wrong under
``scene_subtract`` (no member add-back), ~480 s of worker time per crowded
anchor run, and an 854 GiB dense canvas on T09469. The reliability reason for
these rows is ``giant_frozen_geometry`` (column ``island_giant``).
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tract7dt.fit.grid_gpu import island_plan
from tract7dt.fit.grid_gpu.island_plan import giant_seed_rows
from tract7dt.merge import compute_reliability_flag

N_BANDS = 4
GEOMETRY = ("pos_x", "pos_y", "logre", "ee1", "ee2", "sersic_n")
UNDEFINED = ("chi2", "chi2_nu", "stall_dchi2", "re_upper_pix",
             "pos_x_err", "pos_y_err", "re_err", "ell_err", "re_prof_lo",
             "re_prof_hi", "theta_err", "sersic_n_err", "pos_x_prior_frac",
             "pos_y_prior_frac", "re_prior_frac", "ell_prior_frac",
             "theta_prior_frac", "sersic_n_prior_frac", "rchisq", "fracflux",
             "fracin", "fracmasked", "fluxes", "flux_err", "flux_win",
             "flux_marg_var")


def _island_set():
    members = [np.array([4, 1, 7]), np.array([0, 2]),
               np.array([3, 5, 6, 8, 9])]
    return SimpleNamespace(members=members,
                           n_src=np.array([m.size for m in members]))


def _seeds(n=10):
    rng = np.random.default_rng(3)
    return dict(pos_x=rng.uniform(0, 100, n), pos_y=rng.uniform(0, 100, n),
                logre=rng.normal(0, 1, n), ee1=rng.uniform(-.3, .3, n),
                ee2=rng.uniform(-.3, .3, n), sersic_n=rng.uniform(1, 4, n))


def test_rows_keep_the_seed_geometry_and_measure_nothing():
    iset, seeds = _island_set(), _seeds()
    rows = giant_seed_rows(np.array([2, 0]), iset, n_bands=N_BANDS, **seeds)
    # one row block per island, sorted by island id (run-independent order)
    assert [int(r["island_id"][0]) for r in rows] == [0, 2]
    for r, g in zip(rows, (0, 2)):
        src = iset.members[g]
        n = src.size
        np.testing.assert_array_equal(r["src"], src)
        for k in GEOMETRY:
            assert r[k].dtype == np.float32
            np.testing.assert_array_equal(r[k], seeds[k][src].astype(np.float32))
        assert r["fluxes"].shape == (n, N_BANDS)
        assert r["flux_win"].shape == r["flux_marg_var"].shape == (n, N_BANDS)
        for k in UNDEFINED:
            assert np.isnan(r[k]).all(), k
        assert r["island_giant"].all()
        assert (r["island_nsrc"] == n).all() and (r["island_id"] == g).all()
        assert r["converged"].all() and (r["n_steps"] == 0).all()
        for k in ("stalled", "diverged", "railed", "free_n"):
            assert not r[k].any(), k


def test_rows_have_the_fitted_island_row_keys():
    # The stage concatenates giant rows and unpack_class_results rows over the
    # keys of the FIRST row, so the two key sets must be identical.
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(island_plan.unpack_class_results)))
    (ret,) = [node for node in ast.walk(tree)
              if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
              and any(kw.arg == "src" for kw in node.value.keywords)]
    fitted_keys = {kw.arg for kw in ret.value.keywords}
    rows = giant_seed_rows([1], _island_set(), n_bands=N_BANDS, **_seeds())
    assert set(rows[0]) == fitted_keys


def test_no_giants_gives_no_rows():
    assert giant_seed_rows(np.zeros(0, np.int64), _island_set(),
                           n_bands=N_BANDS, **_seeds()) == []


def test_the_host_giant_solve_is_gone():
    from tract7dt.fit.grid_gpu import island_solver, island_stage
    assert not hasattr(island_solver, "_fit_giants")
    assert not hasattr(island_stage, "_shard_run_giants")
    assert not hasattr(island_stage, "_GIANT_COST_PER_MEMBER_BAND")


@pytest.mark.parametrize("mode, reliable, tagged", [
    ("fail", False, True), ("warn", True, True), ("ignore", True, False)])
def test_reliability_reason_is_giant_frozen_geometry(mode, reliable, tagged):
    df = pd.DataFrame({"excluded_any": [False, False],
                       "x_pix_white_fit": [1.0, 2.0],
                       "island_giant": [True, False]})
    compute_reliability_flag(df, island_giant_mode=mode)
    assert df["fit_is_reliable"].tolist() == [reliable, True]
    tags = df["fit_quality_issues"].tolist()
    assert ("giant_frozen_geometry" in tags[0].split(",")) == tagged
    assert "island_giant" not in tags[0].split(",")
    assert tags[1] == ""
