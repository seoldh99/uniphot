"""Shared fixtures for the tract7dt test suite.

Provides a ``config`` fixture that yields the ``tract7dt.config`` module. It
imports the package normally in a full environment (dev env / built conda
package), but falls back to loading ``config.py`` **standalone** when the
package's heavy import chain is unavailable.

Why the fallback exists: ``import tract7dt.config`` first runs the package
``__init__.py``, which eagerly does ``from .pipeline import run_pipeline`` --
pulling in numpy / pandas / JAX. ``config.py``
itself only needs ``pathlib`` / ``typing`` / ``yaml``, so the config-sync and
config-validation tests can run in a lightweight CI job (just PyYAML) by loading the module directly from its file.
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest


def _import_config_module():
    try:
        return importlib.import_module("tract7dt.config")
    except Exception:  # noqa: BLE001 -- any heavy-dep failure -> standalone load
        config_path = Path(__file__).resolve().parents[1] / "config.py"
        spec = importlib.util.spec_from_file_location(
            "tract7dt_config_standalone", config_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


@pytest.fixture(scope="session")
def config():
    """The ``tract7dt.config`` module (or a standalone-loaded equivalent)."""
    return _import_config_module()


# ---------------------------------------------------------------------------
# Tests that need the development records (dev_records.py): one marker, one skip reason
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "dev_record: registered test that reads development records (plan-repository files named by "
        "$UNIPHOT_DEV_RECORDS and old git commits); skipped, with the reason, where they are not available "
        "(see dev_records.py)")


def pytest_collection_modifyitems(config, items):
    marked = [it for it in items if it.get_closest_marker("dev_record") is not None]
    if not marked:
        return
    import dev_records

    gone = dev_records.missing()
    if gone:
        skip = pytest.mark.skip(reason=f"{dev_records.REASON}; missing: {', '.join(gone)}")
        for it in marked:
            it.add_marker(skip)
