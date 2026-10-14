"""Development records read by some registered tests -- not shipped with uniphot (not a test module).

The registered tests of the M2 PSF units (test_prereg_a3_r5d, test_prereg_a1b_t3_r3d, test_psf_tau2_tgm,
test_psf_tau2_tgm_r2, test_psf_iter0_fit) check the code against fixed files of the team's plan repository (truth
modules and a population JSON, asserted by sha256) and against source files of earlier development commits (read with
`git show`).  Neither is part of the package: the plan files live outside it, and a public release is a single new
commit without the development history.

Every test that needs them carries the marker ``dev_record`` (registered in conftest.py).  When the records are not all
available the marker makes the test SKIP with the reason below -- a skip, not a pass.  The helpers here skip with the
same reason as a backstop.  To run those tests: set UNIPHOT_DEV_RECORDS to the folder that holds the registered files
(a folder of the private development records, not distributed) and run inside a clone with the development history.
"""
from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path

import pytest

ENV = "UNIPHOT_DEV_RECORDS"
# Development commits whose files the registered tests read (c15c72b, 98c87d5: m2-S1; 48a79e4: m2-S1b; f8ae6b7: main).
COMMITS = ("c15c72b", "48a79e4", "f8ae6b7249a9aaf55a0afeecbed8fdbdd0ccda51", "98c87d5")
REASON = ("development record not available (registered test of an M2 PSF unit: needs the plan-repository files "
          f"named by ${ENV} and the development git history; not shipped with uniphot)")
REPO = Path(__file__).resolve().parents[2]


def plan_dir() -> str | None:
    d = os.environ.get(ENV, "")
    return d if d and os.path.isdir(d) else None


@functools.lru_cache(maxsize=None)
def _has_commit(rev: str) -> bool:
    try:
        r = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{rev}^{{commit}}"], capture_output=True)
    except OSError:  # no git
        return False
    return r.returncode == 0


def missing() -> list[str]:
    """What is missing ([] = every development record is available)."""
    out = []
    if plan_dir() is None:
        out.append(f"${ENV} is not set to an existing folder")
    out += [f"commit {c[:7]}" for c in COMMITS if not _has_commit(c)]
    return out


def available() -> bool:
    return not missing()


def require() -> None:
    m = missing()
    if m:
        pytest.skip(f"{REASON}; missing: {', '.join(m)}")


def plan_file(name: str) -> str:
    """Path of a registered file in the plan folder (skips when the records are not available)."""
    require()
    return os.path.join(plan_dir(), name)


def git_show(rev: str, path: str) -> bytes:
    """`git show rev:path` of the development history (skips when the records are not available)."""
    require()
    return subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{path}"], check=True, capture_output=True).stdout
