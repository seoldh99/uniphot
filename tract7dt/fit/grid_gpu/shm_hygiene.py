"""Crash hygiene for /dev/shm segments and spawn workers.

The grid and PSF stages publish large named SharedMemory segments
(``tract7dt_*_<pid>``, ~600 MB/band) so spawn workers can attach by name
instead of receiving multi-GB pickles. Normal and exception paths unlink
them in-process, but two crash modes defeat every in-process cleanup:

- SIGKILL (kernel OOM killer, ``kill -9``) gives the process no chance to
  run ``finally``/atexit, and can even land inside ``SharedMemory.__init__``
  between ``shm_open`` and ``resource_tracker.register`` -- leaving a file
  that not even the resource_tracker knows about.
- Orphaned spawn workers: a ``ProcessPoolExecutor`` spawn worker holds BOTH
  ends of its call-queue pipe, so when the parent is SIGKILLed the worker
  blocks in ``call_queue.get()`` forever (no EOF can ever arrive). The hung
  workers pin RAM, their semaphores, AND the resource_tracker pipe -- which
  blocks the tracker's at-exit emergency unlink of every registered segment.

Because /dev/shm is tmpfs, leaked segments consume physical RAM; on a
shared box they starve the next run into an earlier OOM kill, which leaks
more segments (observed: 62 GB leaked over 6 runs on the development server, 2026-06).

Two helpers close those holes:

- :func:`sweep_stale_segments`: at stage start, unlink leftover
  ``tract7dt_*_<pid>`` segments owned by the current user whose creator
  PID is provably dead. The only mechanism that also covers SIGKILL.
- :func:`start_parent_watchdog`: in spawn workers, hard-exit when the
  parent dies, so orphans release their RAM and the resource_tracker can
  run its own cleanup.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path

import multiprocessing as mp
from multiprocessing import connection

logger = logging.getLogger("tract7dt.fit.grid_gpu.shm_hygiene")

_SHM_DIR = Path("/dev/shm")
# Segment names are f"tract7dt_{...}_{creator_pid}" (grid + psf stages).
_SEGMENT_RE = re.compile(r"^tract7dt_.+_(\d+)$")


def _creator_is_alive(pid: int) -> bool:
    """True if ``pid`` is alive AND still plausibly the tract7dt python run
    that created a segment. Guards against PID reuse; ambiguous -> True
    (keep the file -- a concurrent run's segments must never be swept)."""
    if pid <= 1:
        # "_0"/"_1" suffixes are not our naming scheme (and kill(0, 0)
        # would probe the whole process group); keep the file.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # PID exists but belongs to another user. We only ever inspect
        # segments WE own, and our runs execute under our uid -- so the
        # creator is gone and the kernel reused its PID for someone else.
        return False
    except (OverflowError, ValueError):
        # Numeric suffix too large to be a real pid -- not our naming
        # scheme; keep the file rather than guess.
        return True
    except OSError:
        return True
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return True
    # Our runs (parent or spawn worker) are always python processes; a
    # live non-python process means the PID was reused after a crash.
    return b"python" in cmdline or b"tract7dt" in cmdline


def sweep_stale_segments(log: logging.Logger | None = None) -> tuple[int, int]:
    """Unlink stale ``/dev/shm/tract7dt_*_<pid>`` segments from crashed runs.

    Only touches regular files owned by the current uid whose embedded
    creator PID is dead (or provably reused). Safe to call concurrently
    with other live runs: their creator PIDs are alive, so their segments
    are kept. Unlinking a file a straggler still has mapped is also safe --
    POSIX frees the pages when the last mapping closes.

    Returns ``(n_removed, bytes_removed)`` (actual tmpfs bytes, st_blocks).
    """
    log = log or logger
    if not _SHM_DIR.is_dir():
        return 0, 0
    uid = os.getuid()
    n_removed = 0
    bytes_removed = 0
    try:
        entries = list(os.scandir(_SHM_DIR))
    except OSError as exc:
        log.warning("shm sweep: cannot scan %s: %s", _SHM_DIR, exc)
        return 0, 0
    for entry in entries:
        m = _SEGMENT_RE.match(entry.name)
        if m is None:
            continue
        try:
            st = entry.stat(follow_symlinks=False)
        except OSError:
            continue  # raced with another cleanup
        if not entry.is_file(follow_symlinks=False) or st.st_uid != uid:
            continue
        if _creator_is_alive(int(m.group(1))):
            continue
        try:
            os.unlink(entry.path)
        except OSError:
            continue
        n_removed += 1
        bytes_removed += st.st_blocks * 512
    if n_removed:
        log.warning(
            "shm sweep: removed %d stale shared-memory segment(s) from "
            "crashed run(s), freeing %.2f GiB of /dev/shm",
            n_removed, bytes_removed / (1024 ** 3),
        )
    return n_removed, bytes_removed


def start_parent_watchdog() -> None:
    """In a spawn worker, hard-exit when the parent process dies.

    Blocks a daemon thread on ``multiprocessing.parent_process().sentinel``
    (readable on parent exit, including SIGKILL -- the kernel closes the
    pipe). Uses ``os._exit`` (not ``sys.exit``): the watchdog must not run
    interpreter teardown concurrently with the blocked main thread, and
    there is nothing worth flushing in a worker whose parent is gone.

    Exiting also closes this worker's inherited call-queue and
    resource_tracker fds, which is what lets the tracker EOF and reclaim
    every registered segment/semaphore of the dead run.

    No-op in the main process (the inline ``n_slots == 1`` path calls the
    worker initializer in-process; ``parent_process()`` is None there).
    """
    parent = mp.parent_process()
    if parent is None or parent.sentinel is None:
        return

    def _watch() -> None:
        try:
            connection.wait([parent.sentinel])
        except Exception:  # noqa: BLE001 - any failure means we can't watch
            pass
        os._exit(1)

    threading.Thread(
        target=_watch, name="tract7dt-parent-watchdog", daemon=True,
    ).start()
