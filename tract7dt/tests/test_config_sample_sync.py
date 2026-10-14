"""Drift guard: ``DEFAULT_CONFIG`` (config.py) <-> shipped ``sample_config.yaml``.

The two ship the pipeline defaults through *independent* channels: the hardcoded
``DEFAULT_CONFIG`` dict is what ``load_config`` merges user YAML onto, while
``tract7dt/data/sample_config.yaml`` is what ``dump-config`` / ``write_sample_config``
hand a new user as a starting template (``write_sample_config`` copies the file
verbatim -- it does NOT serialize the dict). Nothing keeps them in sync
automatically, so they have silently drifted before (a batch of ~19 keys had to be
hand-reconciled). This test makes any future drift a hard CI failure.

Invariants asserted:

1. ZERO value mismatches on keys present in both.
2. ZERO sample-only keys (a key in the sample but not the dict would be *rejected*
   by the strict ``_validate_user_config``, i.e. the shipped template wouldn't load).
3. ZERO DEFAULT-only keys (card 018 removed the hidden ``compute.legacy_gpu`` block).
4. ``position_solver == "island"`` in both (guards the 2026-07-11 default flip).
5. ``write_sample_config`` reproduces the shipped sample byte-for-byte (the
   ``dump-config`` contract).

The ``config`` fixture (see conftest.py) yields the ``tract7dt.config`` module.
"""
from __future__ import annotations

from pathlib import Path

import yaml


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten a nested config dict to dotted-key -> leaf-value."""
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _sample_path(config) -> Path:
    return Path(config.__file__).resolve().parent / "data" / "sample_config.yaml"


def _load_sample(config) -> dict:
    with _sample_path(config).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_sample_config_file_exists(config):
    p = _sample_path(config)
    assert p.is_file(), f"shipped sample config missing: {p}"


def test_no_value_mismatches_on_shared_keys(config):
    fd = _flatten(config.DEFAULT_CONFIG)
    fs = _flatten(_load_sample(config))
    mismatches = {
        k: (fd[k], fs[k]) for k in (set(fd) & set(fs)) if fd[k] != fs[k]
    }
    assert not mismatches, (
        "DEFAULT_CONFIG and sample_config.yaml disagree on shared keys "
        "(DEFAULT, SAMPLE):\n"
        + "\n".join(f"  {k}: DEFAULT={dv!r} SAMPLE={sv!r}"
                    for k, (dv, sv) in sorted(mismatches.items()))
    )


def test_no_sample_only_keys(config):
    fd = _flatten(config.DEFAULT_CONFIG)
    fs = _flatten(_load_sample(config))
    sample_only = sorted(set(fs) - set(fd))
    assert not sample_only, (
        "sample_config.yaml has keys absent from DEFAULT_CONFIG; the strict "
        "validator would REJECT this shipped template:\n"
        + "\n".join(f"  {k} = {fs[k]!r}" for k in sample_only)
    )


def test_no_default_only_keys(config):
    fd = _flatten(config.DEFAULT_CONFIG)
    fs = _flatten(_load_sample(config))
    default_only = sorted(set(fd) - set(fs))
    assert not default_only, (
        "DEFAULT_CONFIG has keys missing from the user-facing sample_config.yaml "
        "(undocumented-to-users defaults -- add them to the sample):\n"
        + "\n".join(f"  {k} = {fd[k]!r}" for k in default_only)
    )


def test_position_solver_is_island_in_both(config):
    """Guard the 2026-07-11 default flip to the island solver."""
    dict_ps = config.DEFAULT_CONFIG["compute"]["gpu"]["solver"]["position_solver"]
    sample_ps = _load_sample(config)["compute"]["gpu"]["solver"]["position_solver"]
    assert dict_ps == "island", f"DEFAULT_CONFIG position_solver = {dict_ps!r}, expected 'island'"
    assert sample_ps == "island", f"sample_config position_solver = {sample_ps!r}, expected 'island'"


def test_write_sample_config_is_byte_identical(config, tmp_path):
    """``dump-config`` (write_sample_config) must reproduce the shipped sample verbatim."""
    out = config.write_sample_config(tmp_path / "dumped_config.yaml", overwrite=True)
    dumped = Path(out).read_bytes()
    shipped = _sample_path(config).read_bytes()
    assert dumped == shipped, (
        "write_sample_config output differs from the shipped "
        "tract7dt/data/sample_config.yaml (dump-config no longer round-trips)."
    )
