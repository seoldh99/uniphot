"""Run provenance (plan v3 stage 1, REGISTER R13).

An output catalog alone did not say which code, configuration or environment
produced it: outputs of dev3 to dev7 cannot be told apart, and eight
environment variables that change results were never recorded.
``run_provenance.json`` in the work directory, written at the start of every
pipeline run, records:

* ``schema_version``: the output-contract generation of the catalog;
* ``code``: package version, git commit and dirty flag when the package is a
  git checkout, and a digest of the package source (this also identifies a
  copy launched with ``PYTHONPATH``, which has no git);
* ``config``: the effective configuration, after the defaults were merged
  (``config_used_*.yaml`` is the user's YAML as written);
* ``environment``: the environment variables that can change results;
* ``host``: host name, Python and library versions, GPUs;
* ``missing_values``: what an empty value means in the catalog.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

# Output-contract generation of output_catalog.csv and its side files. dev8
# (plan v3 stage 1): no-data and saturated band cells are NaN, FLAGS_{band}_fit
# and n_bands_usable, MAG_UL5_MARG_{band}_fit, no MAG(ERR)_MARG_FRAC columns,
# frames_used.csv, blank_response_bhat.csv, ZP/zp_model.json.
# m2-a3 (M2 unit a.3, card 024): the PSF training stars of psf_polynomial/<band>/ are
# SEP detections paired with GAIA_SOURCE_ID rows (or TYPE == STAR rows); there
# summary.json n_stars_after_gaia = detections paired with a trainable row (was: after the
# quality cuts) and stars.csv star_origin = "gaia" | "star" (was "sep").
SCHEMA_VERSION = "m2-a3"

# Environment variables that can change results or how they are computed.
ENV_PREFIXES = ("TRACT7DT_", "JAX_", "XLA_", "CUDA_", "NVIDIA_", "TF_",
                "OMP_", "MKL_", "OPENBLAS_", "NUMEXPR_")
ENV_NAMES = ("PYTHONHASHSEED",)

MISSING_VALUES = {
    "FLUX_{band}_fit and FLUXERR*_{band}_fit empty": (
        "no measurement in this band: the band has no information at the "
        "source (FLAGS bit 1), a saturated pixel is near it (bit 8), or the "
        "source was not fitted (fit_quality_issues says why)"),
    "MAG_{band}_fit empty": (
        "FLUX is empty or not positive (a non-detection: use MAG_UL5), or the "
        "band has no zero point"),
    "MAG_UL5_{band}_fit / MAG_UL5_MARG_{band}_fit empty": (
        "the flux error is undefined (no information in the band); "
        "MAG_UL5_MARG is also empty where FLUXERR_MARG_TOT is, i.e. where "
        "rchisq is undefined (island_giant rows)"),
    "FLAGS_{band}_fit": (
        "reason bitmask: 1 no data, 2 PSF model has a negative lobe, 4 zero "
        "point outside its calibrator support (scalar ZP used), 8 saturation. "
        "A band value is usable when FLAGS is 0 and fit_is_reliable is true"),
    "rchisq empty": "no own-profile chi-square (not fitted in an island, or island_giant)",
}

_CODE_IDENTITY: dict | None = None


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_json_safe(v) for v in obj)
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalar
        try:
            return obj.item()
        except (TypeError, ValueError):
            pass
    if isinstance(obj, float) and obj != obj:  # NaN is not valid JSON
        return None
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return repr(obj)


def _source_digest(pkg_dir: Path) -> str:
    """sha256 over the package's Python sources and data tables (paths and
    bytes, sorted), without tests and legacy snapshots."""
    skip = ("tests", "__pycache__")
    files = []
    for p in pkg_dir.rglob("*"):
        rel = p.relative_to(pkg_dir)
        if (p.suffix not in (".py", ".npz") or not p.is_file()
                or any(part in skip or part.startswith((".backup", ".t6-build"))
                       for part in rel.parts)):
            continue
        files.append(rel)
    h = hashlib.sha256()
    for rel in sorted(files, key=lambda r: r.as_posix()):
        h.update(rel.as_posix().encode())
        h.update(b"\0")
        h.update((pkg_dir / rel).read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        r = subprocess.run(["git", "--no-optional-locks", "-C", str(cwd), *args],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def code_identity() -> dict:
    """Which code is running: version, git commit (None outside a checkout),
    whether the package has uncommitted changes, and the source digest."""
    global _CODE_IDENTITY
    if _CODE_IDENTITY is None:
        pkg_dir = Path(__file__).resolve().parent
        from ._version import __version__ as pkg_version
        commit = _git(["rev-parse", "HEAD"], pkg_dir)
        dirty = None
        if commit is not None:
            st = _git(["status", "--porcelain", "--untracked-files=no", "--", "."], pkg_dir)
            dirty = None if st is None else bool(st)
        _CODE_IDENTITY = {
            "package_version": pkg_version,
            "package_dir": str(pkg_dir),
            "git_commit": commit,
            "git_dirty": dirty,
            "source_sha256": _source_digest(pkg_dir),
        }
    return dict(_CODE_IDENTITY)


def science_environment() -> dict:
    """The environment variables that can change results (see ENV_PREFIXES)."""
    return {k: v for k, v in sorted(os.environ.items())
            if k.startswith(ENV_PREFIXES) or k in ENV_NAMES}


def _library_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for name in ("numpy", "scipy", "pandas", "astropy", "jax", "jaxlib", "sep"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _gpus() -> list | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return [[f.strip() for f in line.split(",")] for line in r.stdout.splitlines() if line.strip()]


def write_run_provenance(cfg: dict, work_dir: Path) -> Path:
    """Write ``run_provenance.json`` into ``work_dir`` and return its path."""
    doc = {
        "schema_version": SCHEMA_VERSION,
        "started": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "code": code_identity(),
        "config": _json_safe(cfg),
        "environment": science_environment(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "libraries": _library_versions(),
            "gpus": _gpus(),
        },
        "missing_values": MISSING_VALUES,
    }
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    path = Path(work_dir) / "run_provenance.json"
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1)
    return path
