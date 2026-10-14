"""The synthetic truth generator imports no engine module and no JAX (M2 unit 0.3).

Design 6: the truth PSF is an analytic function independent of the engine's models, and the
expected values come from the generator's own PSF, never from an engine function.  If the
generator imported the engine, an engine function could leak into the truth or the expectations.
Static: every import in tract7dt/tests/synth/*.py names numpy, scipy, the standard library or
the synth package itself.  Dynamic: a fresh isolated interpreter (python -I), with the engine
importable, imports every synth module and reports no tract7dt* / jax* module in sys.modules.
Positive controls show that both checks catch a module that does import the engine.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
SYNTH = TESTS / "synth"
REPO = TESTS.parents[1]
ALLOWED = {"__future__", "argparse", "ast", "concurrent", "dataclasses", "datetime", "hashlib", "json",
           "math", "os", "pathlib", "sys", "time", "typing", "warnings", "numpy", "scipy"}
# importlib: allowed in falsefail.py only, where main() loads a --check MODULE:FUNC named on the
# command line (review B4); every other dynamic import or exec is forbidden (_dynamic_imports).
ALLOWED_IN = {"falsefail.py": {"importlib"}}
DYNAMIC_NAMES = {"__import__", "exec", "eval", "compile", "import_module", "run_module", "run_path"}
DYNAMIC_ATTRS = {"import_module", "__import__", "exec_module", "load_module", "find_spec",
                 "spec_from_file_location", "run_module", "run_path", "reload"}
DYNAMIC_ALLOWED = {("falsefail.py", "main")}
FORBIDDEN = ("tract7dt", "jax", "jaxlib")


def _bad_imports(path: Path) -> list:
    bad = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names = [(0, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.level, node.module or "")]
        else:
            continue
        for level, name in names:
            top = name.split(".")[0]
            if level > 1 or (level == 0 and top not in ALLOWED | ALLOWED_IN.get(path.name, set())):
                bad.append(f"{path.name}: {'.' * level}{name}")
    return bad


def _dynamic_imports(path: Path) -> list:
    """Calls that can import or execute code by name, outside the allowed (file, function) pairs."""
    tree = ast.parse(path.read_text())
    bad = []

    def visit(node, func):
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else func
            if isinstance(child, ast.Call):
                f = child.func
                hit = (isinstance(f, ast.Name) and f.id in DYNAMIC_NAMES) or \
                      (isinstance(f, ast.Attribute) and f.attr in DYNAMIC_ATTRS)
                if hit and (path.name, func) not in DYNAMIC_ALLOWED:
                    bad.append(f"{path.name}:{child.lineno} in {func or '<module>'}: {ast.unparse(f)}")
            visit(child, inner)

    visit(tree, None)
    return bad


def _loaded_forbidden(paths: list, modules: list) -> tuple:
    code = ("import importlib, json, sys\n"
            f"sys.path[:0] = {[str(p) for p in paths]!r}\n"
            f"for m in {modules!r}: importlib.import_module(m)\n"
            f"print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in {FORBIDDEN!r})))\n")
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=600)
    return r.returncode, r.stdout, r.stderr


def test_synth_sources_import_only_allowed_modules():
    files = sorted(SYNTH.glob("*.py"))
    assert len(files) >= 9, files
    bad = [b for p in files for b in _bad_imports(p)]
    assert bad == [], bad


def test_synth_has_no_dynamic_import_outside_the_command_line():
    bad = [b for p in sorted(SYNTH.glob("*.py")) for b in _dynamic_imports(p)]
    assert bad == [], bad


def test_dynamic_import_checker_catches_engine_imports_inside_functions(tmp_path):
    # review B4: the import-statement scan and the fresh-process import both miss these
    probe = tmp_path / "probe_fn.py"
    probe.write_text("def f():\n    import importlib\n    return importlib.import_module('tract7dt')\n"
                     "def g():\n    return __import__('tract7dt')\n"
                     "def h():\n    exec('import tract7dt')\n"
                     "def main():\n    from importlib import import_module\n    return import_module('jax')\n")
    assert _bad_imports(probe) == ["probe_fn.py: importlib", "probe_fn.py: importlib"]
    hits = _dynamic_imports(probe)
    assert len(hits) == 4, hits                       # main() is allowed only in falsefail.py
    allowed = tmp_path / "falsefail.py"
    allowed.write_text(probe.read_text())
    assert len(_dynamic_imports(allowed)) == 3        # f, g, h still caught there


def test_importing_every_synth_module_loads_no_engine_or_jax():
    mods = ["synth"] + [f"synth.{p.stem}" for p in sorted(SYNTH.glob("*.py")) if p.stem != "__init__"]
    rc, out, err = _loaded_forbidden([TESTS, REPO], mods)
    assert rc == 0, err[-3000:]
    assert json.loads(out.strip().splitlines()[-1]) == []


def test_positive_controls_catch_an_engine_import(tmp_path):
    probe = tmp_path / "probe_mod.py"
    probe.write_text("import tract7dt\nfrom jax import numpy\n")
    assert len(_bad_imports(probe)) == 2
    fake = tmp_path / "tract7dt"          # a stand-in package, so the control does not load the engine
    fake.mkdir()
    (fake / "__init__.py").write_text("")
    (tmp_path / "probe_dyn.py").write_text("import tract7dt\n")
    rc, out, err = _loaded_forbidden([tmp_path], ["probe_dyn"])
    assert rc == 0, err[-3000:]
    assert json.loads(out.strip().splitlines()[-1]) == ["tract7dt"]
