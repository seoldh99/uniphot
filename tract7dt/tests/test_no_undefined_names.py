"""No module may reference a name that is never defined.

Why this test exists: on 2026-08-29 the `effarc2` dev4 tree shipped an edit that replaced
`_shard_measure_width` with three new helpers and left its CALL SITE untouched. The whole
config suite (434 tests) passed, `tract7dt doctor` passed, and the module imported fine --
because the missing name is only looked up when that line RUNS. It surfaced 260 s into a GPU
run, as `NameError: name '_shard_measure_width' is not defined`, after the field stacks were
already built. A whole gate arm was wasted.

An import test cannot catch this; a unit test only catches it for paths it exercises, and the
sharded island fit needs 3 GPUs and a real field. Static analysis catches it in milliseconds
for every line of the package, so that is what runs here.

Deliberate lazy forward references (e.g. `coefs: "jax.Array"` in psf_polynomial, kept as a
string so importing the dataclass does not pull jax) are exempted by the `# noqa: F821` marker
the author already put on that line -- the exemption is read from the source, so it cannot
drift from the code and there is no list to maintain here.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent          # .../tract7dt
SKIP_DIRS = {"_vendor", "tests", "__pycache__"}


def _sources():
    return sorted(p for p in PKG.rglob("*.py")
                  if not SKIP_DIRS & set(p.relative_to(PKG).parts))


def test_no_undefined_names():
    pytest.importorskip("pyflakes", reason="pyflakes is the static checker this test drives")
    files = _sources()
    # Sanity: the glob saw the whole package (top level and the GPU subpackage), not an
    # empty or wrong directory. Checked by membership, not by a file count that every
    # legitimate deletion would trip (card 018 removed 20 files).
    must = {PKG / "pipeline.py", PKG / "config.py", PKG / "fit" / "grid_gpu" / "pipeline.py",
            PKG / "fit" / "grid_gpu" / "island_stage.py"}
    assert must <= set(files), f"package glob missed {sorted(map(str, must - set(files)))}"

    proc = subprocess.run([sys.executable, "-m", "pyflakes", *map(str, files)],
                          capture_output=True, text=True)
    bad = []
    for line in proc.stdout.splitlines():
        if "undefined name" not in line:
            continue
        # "path:LINE:COL: undefined name 'x'"
        path, lineno = line.split(":")[0], int(line.split(":")[1])
        try:
            src = Path(path).read_text(encoding="utf-8").splitlines()[lineno - 1]
        except (OSError, IndexError):
            src = ""
        if "noqa: F821" in src:          # author-declared lazy forward reference
            continue
        bad.append(f"{line}\n      source: {src.strip()}")

    assert not bad, (
        "undefined name(s) -- these raise NameError only when the line runs, so they "
        "survive import checks and unit tests:\n  " + "\n  ".join(bad))
