"""CPU unit tests for load-time config validation (no GPU / no JAX).

Covers the fail-fast guards in ``_validate_solver_semantics`` and, since card
018 removed ``psf.mode`` per_cell/from_cells, that pinning a removed PSF mode
fails at config LOAD (retired-key rule).

The ``config`` fixture (see conftest.py) yields the ``tract7dt.config`` module.
"""
from __future__ import annotations

import copy

import pytest
import yaml


def _gpu_cfg(position_solver: str, psf_mode: str) -> dict:
    """Minimal merged-style cfg exercising just the solver-semantics validator."""
    return {
        "compute": {
            "mode": "gpu",
            "gpu": {
                "psf": {"mode": psf_mode},
                "solver": {"position_solver": position_solver},
            },
        }
    }


def test_island_with_from_stars_psf_is_accepted(config):
    # The shipped default combo -- must NOT raise.
    config._validate_solver_semantics(_gpu_cfg("island", "from_stars"))


def test_load_config_rejects_retired_psf_mode(config, tmp_path):
    """psf.mode per_cell was removed (card 018): pinning it fails at load_config."""
    seed = config.write_sample_config(tmp_path / "seed.yaml", overwrite=True)
    cfg = copy.deepcopy(yaml.safe_load(seed.read_text()))
    cfg.setdefault("compute", {}).setdefault("gpu", {}).setdefault("psf", {})["mode"] = "per_cell"
    bad = tmp_path / "bad_config.yaml"
    bad.write_text(yaml.safe_dump(cfg, sort_keys=False))
    with pytest.raises(ValueError, match="compute.gpu.psf.mode"):
        config.load_config(bad)

def test_default_config_position_solver_is_island(config):
    assert config.DEFAULT_CONFIG["compute"]["gpu"]["solver"]["position_solver"] == "island"


# ---- dlnp campaign (2026-07-30): island.convergence knob ------------------


def _island_cfg(**island) -> dict:
    return {
        "compute": {
            "mode": "gpu",
            "gpu": {
                "psf": {"mode": "from_stars"},
                "solver": {"position_solver": "island",
                           "island": island},
            },
        }
    }


def test_default_config_convergence_is_legacy(config):
    assert (config.DEFAULT_CONFIG["compute"]["gpu"]["solver"]["island"]
            ["convergence"] == "legacy")


def test_convergence_dlnp_is_accepted(config):
    config._validate_solver_semantics(_island_cfg(convergence="dlnp"))


def test_convergence_bad_value_is_rejected(config):
    with pytest.raises(ValueError, match="convergence must be"):
        config._validate_solver_semantics(_island_cfg(convergence="quantum"))


def test_convergence_dlnp_rejects_trust_region(config):
    with pytest.raises(ValueError, match="requires .?step_mode"):
        config._validate_solver_semantics(
            _island_cfg(convergence="dlnp", step_mode="trust_region"))


def test_convergence_dlnp_rejects_changed_counting(config):
    with pytest.raises(ValueError, match="subsumes"):
        config._validate_solver_semantics(
            _island_cfg(convergence="dlnp", stall_count_mode="changed"))


def test_convergence_legacy_allows_trust_region_and_changed(config):
    # The interaction locks are dlnp-only.
    config._validate_solver_semantics(
        _island_cfg(convergence="legacy", step_mode="trust_region"))
    config._validate_solver_semantics(
        _island_cfg(stall_count_mode="changed"))
