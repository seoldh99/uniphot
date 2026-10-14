"""The GPU path imports no Tractor (card 018, design_v2 7.1 test 1; card criterion 4).

In a NEW process, import the 14 isolation modules and require that no module whose
name starts with tractor / astrometry / fitsio / emcee is loaded. Statically, the
only Tractor import left in the package is the function-level one in the Sersic
mixture table generator (sersic_mog_gen.py), which no run executes.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import tract7dt

PKG = Path(tract7dt.__file__).resolve().parent
MODULES = [
    "tract7dt.cli", "tract7dt.pipeline", "tract7dt.load_inputs", "tract7dt.zp", "tract7dt.bright_mask",
    "tract7dt.fit.grid_gpu.pipeline", "tract7dt.fit.grid_gpu.runner", "tract7dt.fit.grid_gpu.island_stage",
    "tract7dt.fit.grid_gpu.island_solver", "tract7dt.fit.grid_gpu.solver_grid",
    "tract7dt.fit.grid_gpu.psf_from_stars", "tract7dt.fit.grid_gpu.psf_pipeline_stage",
    "tract7dt.fit.grid_gpu.error_columns", "tract7dt.fit.grid_gpu.error_model",
]
FORBIDDEN = ("tractor", "astrometry", "fitsio", "emcee")


def test_no_tractor_modules_loaded():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PKG.parent) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("JAX_PLATFORMS", "cpu")
    code = ("import importlib, json, sys\n"
            f"for m in {MODULES!r}: importlib.import_module(m)\n"
            f"print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in {FORBIDDEN!r})))\n")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    loaded = json.loads(r.stdout.strip().splitlines()[-1])
    assert loaded == [], loaded


def test_only_the_table_generator_imports_tractor():
    hits = []
    for p in PKG.rglob("*.py"):
        if "tests" in p.parts:
            continue
        for n in ast.walk(ast.parse(p.read_text())):
            names = []
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                names = [n.module]
            if any(x.split(".")[0] in ("tractor", "astrometry") for x in names):
                hits.append(str(p.relative_to(PKG)))
    assert sorted(set(hits)) == ["fit/grid_gpu/sersic_mog_gen.py"], hits


def test_two_pass_code_is_gone():
    """M2 unit 0.1 (card 024; 10-09 decision 6): the engine has one pass. No module, function or
    config key of the two-pass residual-phantom form is left (the keys are retired, test_retired_keys)."""
    import importlib.util

    from tract7dt import config, pipeline
    assert importlib.util.find_spec("tract7dt.fit.grid_gpu.residual_phantoms") is None
    assert not hasattr(pipeline, "_run_residual_phantom_two_pass")
    d = config.DEFAULT_CONFIG
    assert "residual_phantom_pass" not in d["bright_mask"] and "extra_unmatched_csv" not in d["bright_mask"]
    assert "dump_diagnostics" not in d["compute"]["gpu"]["background"] and "calibrate" not in d["zp"]
    # Code only: names, attributes, arguments, imports and string constants (config keys, log
    # text). Docstrings and comments are prose; config.py names the keys once more, in the
    # retired-key table.
    words = ("residual_phantom", "_residual_phantoms_csv", "extra_unmatched", "bg_diag", "dump_diagnostics")
    hits = sorted({f"{p.relative_to(PKG)}: {w}" for p in PKG.rglob("*.py")
                   if "tests" not in p.parts and p.name != "config.py"
                   for t in _code_tokens(ast.parse(p.read_text())) for w in words if w in t})
    assert hits == [], hits


def _code_tokens(tree):
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)
            and isinstance(n.body[0].value.value, str)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            yield n.id
        elif isinstance(n, ast.Attribute):
            yield n.attr
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield n.name
        elif isinstance(n, ast.arg):
            yield n.arg
        elif isinstance(n, ast.keyword) and n.arg:
            yield n.arg
        elif isinstance(n, ast.alias):
            yield n.name
        elif isinstance(n, ast.ImportFrom) and n.module:
            yield n.module
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
            yield n.value
