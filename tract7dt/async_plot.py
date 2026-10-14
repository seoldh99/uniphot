"""Background plotting helper.

Saves diagnostic PNGs in a small thread pool so the main pipeline
thread does not block on matplotlib's PNG serialization. Designed for
the pre-EPSF stages (``load_inputs``, ``bright_mask``, ``zp``) where
each stage emits a handful of full-frame plots whose results are not
read by any downstream stage.

Lifecycle
---------

The pool is owned by :func:`tract7dt.pipeline.run_pipeline`:

* :func:`init_plot_pool` is called at the top of the pipeline run.
* Each plot site builds its ``Figure`` as before and then hands it to
  :func:`submit_plot` instead of calling ``fig.savefig(...)`` itself.
* :func:`shutdown_plot_pool` is called in a ``finally`` block at the
  end of the pipeline; it waits for every pending ``savefig`` to
  complete and propagates any background exceptions to the log.

If ``performance.background_plotting`` is ``false`` (or the pool has
not been initialized at all - e.g. a unit test that calls a plot
helper directly), :func:`submit_plot` falls back to the previous
synchronous ``fig.savefig`` + ``plt.close(fig)`` behavior. This means
the helper is safe to wire into every plot site unconditionally.

Threading model
---------------

matplotlib's ``Agg`` backend (which the pipeline forces with
``matplotlib.use("Agg")``) releases the GIL during the actual PNG
encoding step, so background savefig genuinely overlaps with the next
stage's NumPy / SEP / FITS-I/O work. The pool size defaults to 4,
which is enough to keep up with the pre-EPSF plot cadence without
spawning useless threads.

Workers must NOT touch any pyplot global state beyond
``plt.close(fig)`` (which is documented thread-safe for ``Agg``); they
must NOT create new figures, set rcParams, or interact with the
pyplot figure manager.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

logger = logging.getLogger("tract7dt.async_plot")

_POOL: ThreadPoolExecutor | None = None
_FUTURES: list[Future] = []
_POOL_PID: int | None = None


def init_plot_pool(*, max_workers: int = 4) -> None:
    """Initialize the background plot pool. Idempotent."""
    global _POOL, _FUTURES, _POOL_PID
    if _POOL is not None:
        return
    workers = max(1, int(max_workers))
    _POOL = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="t7dt-plot",
    )
    _FUTURES = []
    _POOL_PID = os.getpid()
    logger.debug("Background plot pool started (workers=%d, pid=%d)", workers, _POOL_PID)


def _save_then_close(fig: Any, path: Path, dpi: int, savefig_kwargs: dict) -> str:
    """Worker body. Returns the saved path so logs can report it."""
    import matplotlib.pyplot as plt

    try:
        fig.savefig(str(path), dpi=int(dpi), **savefig_kwargs)
    finally:
        # plt.close(fig) on Agg is safe from worker threads; it removes
        # the figure from pyplot's manager and frees its renderer.
        plt.close(fig)
    return str(path)


def submit_plot(fig: Any, path: Path | str, *, dpi: int, **savefig_kwargs: Any) -> None:
    """Save ``fig`` to ``path`` in the background (or inline if no pool).

    If the background plot pool is initialized (and we're still in the
    PID that created it), the savefig is dispatched to a worker thread
    and the function returns immediately. Otherwise it falls back to a
    synchronous ``fig.savefig`` + ``plt.close(fig)``, matching the
    behavior every plot site had before this helper existed.

    Extra keyword arguments (e.g. ``bbox_inches="tight"``) are passed
    through to ``fig.savefig`` unchanged in both paths.
    """
    global _POOL, _FUTURES, _POOL_PID
    path = Path(path)
    use_pool = _POOL is not None and _POOL_PID == os.getpid()
    if not use_pool:
        # Inline path: same shape as the original code.
        import matplotlib.pyplot as plt

        try:
            fig.savefig(str(path), dpi=int(dpi), **savefig_kwargs)
        finally:
            plt.close(fig)
        return
    fut = _POOL.submit(_save_then_close, fig, path, dpi, dict(savefig_kwargs))
    _FUTURES.append(fut)


def submit_task(fn: Any, *args: Any, **kwargs: Any) -> None:
    """Run an arbitrary diagnostic task in the background plot pool.

    This is for diagnostics where figure construction itself is expensive
    (for example full-frame overlays with many scatter points). If no pool is
    active, the callable runs inline for backward-compatible behavior.
    """
    global _POOL, _FUTURES, _POOL_PID
    use_pool = _POOL is not None and _POOL_PID == os.getpid()
    if not use_pool:
        fn(*args, **kwargs)
        return
    fut = _POOL.submit(fn, *args, **kwargs)
    _FUTURES.append(fut)


def shutdown_plot_pool(*, log_summary: bool = True) -> None:
    """Wait for all pending background saves, surface errors, then close.

    Idempotent. Safe to call from a ``finally`` block even if
    :func:`init_plot_pool` was never called.
    """
    global _POOL, _FUTURES, _POOL_PID
    if _POOL is None:
        return

    pending = list(_FUTURES)
    n_total = len(pending)
    n_failed = 0
    for fut in pending:
        try:
            fut.result()
        except Exception as exc:
            n_failed += 1
            logger.warning("Background plot save failed: %s", exc)

    _POOL.shutdown(wait=True)
    _POOL = None
    _FUTURES = []
    _POOL_PID = None

    if log_summary and n_total > 0:
        logger.info(
            "Background plot pool drained: %d saved, %d failed",
            n_total - n_failed,
            n_failed,
        )


__all__ = [
    "init_plot_pool",
    "submit_plot",
    "submit_task",
    "shutdown_plot_pool",
]
