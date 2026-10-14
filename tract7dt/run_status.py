"""Run contract (plan R12): every ``tract7dt run`` ends with ``RUN_STATUS.json``.

``status`` is ``"ok"`` only when every stage finished and no stage reported a
contract failure; the process then exits 0. A run that raises -- including a
final-pass grid that still fails after its retry (an error-model failure
inside a grid counts as a grid failure), or a ZP stage that could not
calibrate while ``zp.enabled`` is on -- writes ``status: "failed"`` with the
reason and exits non-zero. ``status: "running"`` is written when the run
starts, so a killed run never leaves an earlier run's ``"ok"`` behind.
Consumers (the lowb campaign gate) read this file; neither the exit code
alone nor the existence of ``output_catalog.csv`` means the run succeeded.

A per-user lock (``/tmp/tract7dt-run-<uid>.lock``) allows one GPU run per user
at a time: runs of one user share the private MPS daemon, and a second run
would also compete for the same cards.
"""
from __future__ import annotations

import datetime
import fcntl
import json
import os
import socket
import tempfile
from pathlib import Path

RUN_STATUS_NAME = "RUN_STATUS.json"
SCHEMA = "run_status/1"
EXIT_FAILED = 1
EXIT_LOCKED = 3


class RunContractError(RuntimeError):
    """A stage finished, but its output would be silently degraded."""

    def __init__(self, kind: str, message: str, **details):
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message
        self.details = details


class RunLockBusy(RuntimeError):
    """Another GPU run of this user holds the run lock."""


_REPORT: dict = {"stage": None, "summary": {}, "warnings": []}


def reset() -> None:
    _REPORT.update(stage=None, summary={}, warnings=[])


def set_stage(name: str) -> None:
    _REPORT["stage"] = name


def note(key: str, value) -> None:
    """Record a summary item (plain JSON types) for RUN_STATUS.json."""
    _REPORT["summary"][key] = value


def warn(message: str) -> None:
    """Record something a consumer should see that does not fail the run."""
    _REPORT["warnings"].append(str(message))


def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def describe(exc: BaseException) -> dict:
    if isinstance(exc, RunContractError):
        return {"kind": exc.kind, "message": exc.message,
                "details": exc.details}
    if isinstance(exc, KeyboardInterrupt):
        return {"kind": "interrupted", "message": "KeyboardInterrupt",
                "details": {}}
    return {"kind": "exception", "message": f"{type(exc).__name__}: {exc}",
            "details": {}}


def write(work_dir, *, status: str, started: str, exit_code=None,
          error: dict | None = None) -> Path:
    """Write ``{work_dir}/RUN_STATUS.json`` atomically and return its path."""
    doc = {
        "schema": SCHEMA,
        "status": status,
        "exit_code": exit_code,
        "started": started,
        "finished": None if status == "running" else now_iso(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "stage": _REPORT["stage"],
        "error": error,
        "summary": _REPORT["summary"],
        "warnings": _REPORT["warnings"],
    }
    path = Path(work_dir) / RUN_STATUS_NAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    os.replace(tmp, path)
    return path


def read(work_dir) -> dict | None:
    """The run status in ``work_dir``, or None when absent or unreadable."""
    try:
        return json.loads((Path(work_dir) / RUN_STATUS_NAME).read_text())
    except (OSError, ValueError):
        return None


def lock_path() -> Path:
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return Path(tempfile.gettempdir()) / f"tract7dt-run-{uid}.lock"


def acquire_run_lock(label: str = ""):
    """Take the per-user run lock for this process's lifetime.

    Returns the open lock file (keep a reference until the run ends; the
    kernel releases the lock when the process exits, however it exits).
    Raises RunLockBusy with the holder's record when another run of this user
    holds it."""
    path = lock_path()
    fh = open(path, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.seek(0)
        holder = fh.read().strip()
        fh.close()
        raise RunLockBusy(
            f"another tract7dt GPU run of this user holds {path}: "
            f"{holder or '(no record)'}") from None
    fh.seek(0)
    fh.truncate()
    fh.write(json.dumps({"pid": os.getpid(), "started": now_iso(),
                         "host": socket.gethostname(), "run": label}) + "\n")
    fh.flush()
    return fh
