"""Prerequisite check for uniphot (import name ``tract7dt``).

The ``uniphot doctor`` CLI subcommand (alias ``tract7dt doctor``) prints a one-line status row for
every external dependency the pipeline expects to find, with a copy-
pasteable install hint when something is missing.

Exit code is ``0`` if everything is OK, ``1`` if any check failed (i.e. a
required dependency is missing or unusable).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable


GREEN = "OK"
YELLOW = "WARN"
RED = "MISSING"


class _CheckResult:
    __slots__ = ("status", "name", "details", "fix_hint")

    def __init__(self, status: str, name: str, details: str = "", fix_hint: str = "") -> None:
        self.status = status
        self.name = name
        self.details = details
        self.fix_hint = fix_hint

    @property
    def ok(self) -> bool:
        return self.status == GREEN


def _format_row(r: _CheckResult, name_width: int) -> str:
    tag = f"[{r.status:^7}]"
    name = r.name.ljust(name_width)
    details = r.details
    return f"{tag} {name}  {details}"


def check_python() -> _CheckResult:
    v = sys.version_info
    return _CheckResult(
        GREEN,
        "python",
        f"{v.major}.{v.minor}.{v.micro}  ({sys.executable})",
    )


def check_tract7dt() -> _CheckResult:
    try:
        import tract7dt  # noqa: F401
        version = tract7dt.__version__
    except Exception as exc:
        return _CheckResult(
            RED,
            "uniphot",
            f"import failed: {exc}",
            fix_hint="Re-install with `pip install -e .` from the repo root.",
        )
    return _CheckResult(GREEN, "uniphot", f"{version}  (import name tract7dt, {Path(tract7dt.__file__).parent})")


def check_python_pkg(modname: str, fix_hint: str) -> Callable[[], _CheckResult]:
    def _check() -> _CheckResult:
        try:
            mod = importlib.import_module(modname)
            version = getattr(mod, "__version__", "?")
            return _CheckResult(GREEN, modname, str(version))
        except ImportError as exc:
            return _CheckResult(RED, modname, f"import failed: {exc}", fix_hint=fix_hint)
    return _check


def check_compute_backends() -> list[_CheckResult]:
    out: list[_CheckResult] = []
    # Grid-based GPU mode (compute.mode=gpu, the only mode): check that JAX
    # + at least one CUDA device are visible.
    try:
        import jax
        cuda_devs = jax.devices("cuda")
        if cuda_devs:
            out.append(_CheckResult(
                GREEN, "compute.mode=gpu",
                f"JAX + {len(cuda_devs)} CUDA device(s) (grid pipeline ready)",
            ))
        else:
            out.append(_CheckResult(
                RED, "compute.mode=gpu",
                "JAX imported but no CUDA devices visible "
                "(jax.devices('cuda') is empty)",
            ))
    except Exception as exc:
        out.append(_CheckResult(
            RED, "compute.mode=gpu",
            f"failed: {exc}. Install jax[cuda12] (or equivalent).",
        ))
    return out


def check_gpu_stack() -> list[_CheckResult]:
    """JAX + jaxlib + CUDA device visibility checks.

    Only the JAX path is supported in Phase 1 of the GPU optimizer; PyTorch /
    CuPy variants would be added here when implemented.
    """
    out: list[_CheckResult] = []
    install_hint = (
        "conda install -c conda-forge 'jaxlib=*=*cuda12*' jax\n"
        "  (CPU-only fallback: conda install -c conda-forge jax jaxlib)"
    )
    try:
        import jax  # noqa: F401
        import jaxlib  # noqa: F401
        out.append(_CheckResult(
            GREEN, "jax", f"{jax.__version__} (default backend: {jax.default_backend()})"
        ))
        out.append(_CheckResult(GREEN, "jaxlib", str(jaxlib.__version__)))
    except ImportError as exc:
        out.append(_CheckResult(
            YELLOW,
            "jax",
            f"not installed ({exc}) - GPU optimizer disabled",
            fix_hint=install_hint,
        ))
        out.append(_CheckResult(YELLOW, "jaxlib", "not installed - GPU optimizer disabled", fix_hint=install_hint))
        return out
    except Exception as exc:
        out.append(_CheckResult(RED, "jax", f"unexpected error: {exc}", fix_hint=install_hint))
        return out

    try:
        cuda_devices = jax.devices("cuda")
        if not cuda_devices:
            out.append(_CheckResult(
                YELLOW,
                "cuda devices",
                "0 visible to JAX - GPU optimizer will not have anything to dispatch to",
                fix_hint=(
                    "Install a CUDA-enabled jaxlib: conda install -c conda-forge 'jaxlib=*=*cuda12*'\n"
                    "  Verify NVIDIA driver: nvidia-smi"
                ),
            ))
        else:
            details = ", ".join(
                f"{d.id}:{getattr(d, 'device_kind', '?')}" for d in cuda_devices
            )
            out.append(_CheckResult(
                GREEN, "cuda devices", f"{len(cuda_devices)} visible ({details})"
            ))
    except RuntimeError as exc:
        out.append(_CheckResult(
            YELLOW,
            "cuda devices",
            f"jax.devices('cuda') failed: {exc}",
            fix_hint="Install a CUDA-enabled jaxlib or verify CUDA driver / NVIDIA driver match.",
        ))
    except Exception as exc:
        out.append(_CheckResult(RED, "cuda devices", f"unexpected error: {exc}"))

    return out


def run_doctor() -> int:
    """Run all checks; print a status table; return shell-style exit code."""
    from . import __version__
    print(f"uniphot {__version__} doctor: prerequisite checks", flush=True)

    checks: list[_CheckResult] = []

    # Core stack
    checks.append(check_python())
    checks.append(check_tract7dt())

    # Pure-Python deps
    py_pkgs = [
        ("numpy", "conda install -c conda-forge numpy"),
        ("scipy", "conda install -c conda-forge scipy"),
        ("astropy", "conda install -c conda-forge astropy"),
        ("pandas", "conda install -c conda-forge pandas"),
        ("matplotlib", "conda install -c conda-forge matplotlib"),
        ("sep", "conda install -c conda-forge sep"),
        ("photutils", "conda install -c conda-forge photutils"),
        ("yaml", "conda install -c conda-forge pyyaml"),
    ]
    for name, hint in py_pkgs:
        checks.append(check_python_pkg(name, hint)())

    # Backend checks
    checks.extend(check_compute_backends())
    checks.extend(check_gpu_stack())

    # Print
    width = max(len(c.name) for c in checks)
    print()
    for c in checks:
        print(_format_row(c, width))

    # Hints for any missing items
    missing = [c for c in checks if c.status != GREEN]
    if missing:
        print()
        print("Suggested fixes:")
        for c in missing:
            if c.fix_hint:
                print(f"  - [{c.name}] {c.fix_hint}")

    failed = [c for c in checks if c.status == RED]
    print()
    if failed:
        print(f"FAIL: {len(failed)} required check(s) failed.")
        return 1
    warns = [c for c in checks if c.status == YELLOW]
    if warns:
        print(f"OK with {len(warns)} warning(s) (optional / future-feature items).")
    else:
        print("All checks PASSED.")
    return 0
