"""Gaia DR3 quality-flag fetch + caching for the ZP reference-quality filter.

The ZP quality filter (``zp.gaia_quality_filter``) needs ``ruwe`` and
``phot_bp_rp_excess_factor`` per Gaia calibrator. Standard GaiaXP synphot CSVs
usually carry only the synphot mags (+ ``bp_rp``, ``phot_g_mean_mag``), so this
module fetches the two quality columns from the Gaia DR3 archive via TAP
(no astroquery dependency) and caches them next to the run so the cost is paid
once per tile.

Used two ways:
  * automatically by a run when the GaiaXP CSV lacks the columns: the Gaia
    augment stage starts a background fetch of every calibrator candidate
    into ``<work_dir>/ZP/gaia_quality_cache.csv`` and
    :func:`tract7dt.zp.require_gaia_quality_flags` waits for it before the
    fit, inside one time budget (:data:`QUALITY_BUDGET_S`); if the flags
    cannot be obtained the run fails there (R29) -- the ZP is never
    calibrated on the unfiltered calibrator set instead;
  * via the ``tract7dt augment-gaia`` CLI subcommand to pre-augment a CSV
    (then a run needs no network).
"""
from __future__ import annotations

import io
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("tract7dt.gaia_quality")

GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
QUALITY_COLS = ["ruwe", "phot_bp_rp_excess_factor"]

# Total wall-clock budget for obtaining the flags of one run (R29). A
# network constant, not a science one: a healthy archive served the largest
# tile (T08285, 105,649 ids) in 2.1 min (2026-09-27) and 3 min 46 s in the
# slowest recorded case; on 2026-09-27 a fetch with only per-request
# timeouts hung for 27 min before failing.
QUALITY_BUDGET_S = 600.0


class QualityFetchTimeout(TimeoutError):
    """The Gaia quality fetch ran out of its time budget."""


def _fetch_quality_batch(chunk, *, tap_url, timeout, retries,
                         deadline=None) -> pd.DataFrame:
    """Fetch one batch of source_ids from Gaia DR3 TAP (retry w/ backoff).

    ``deadline`` (a ``time.monotonic()`` value) bounds the whole batch: no
    attempt starts after it, each attempt's socket timeout is capped by the
    time left, and the backoff sleep never runs past it."""
    adql = ("SELECT source_id, ruwe, phot_bp_rp_excess_factor "
            "FROM gaiadr3.gaia_source WHERE source_id IN (" + ",".join(chunk) + ")")
    data = urllib.parse.urlencode({
        "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv",
        "MAXREC": str(len(chunk) + 100), "QUERY": adql}).encode()
    last_exc = None
    for attempt in range(retries):
        t_req = timeout
        if deadline is not None:
            left = deadline - time.monotonic()
            if left <= 0:
                raise QualityFetchTimeout(
                    "Gaia quality fetch: time budget exhausted"
                    + (f" (last error: {last_exc})" if last_exc else ""))
            t_req = max(1.0, min(float(timeout), left))
        try:
            txt = urllib.request.urlopen(tap_url, data=data, timeout=t_req).read().decode()
            return pd.read_csv(io.StringIO(txt))
        except Exception as exc:  # noqa: BLE001 - network/HTTP transient
            last_exc = exc
            pause = min(5 * (attempt + 1), 20)
            if deadline is not None:
                pause = min(pause, max(0.0, deadline - time.monotonic()))
            time.sleep(pause)
    raise last_exc


# One concurrent "wave" of the fetch = FETCH_BATCH x FETCH_WORKERS ids in
# flight. Shared by the fetch defaults AND the prefetch's persistence
# segments below, so the two can never drift apart (the T7 lesson: work done
# in two places keeps its parameters in ONE place).
FETCH_BATCH = 2000
FETCH_WORKERS = 8


def fetch_gaia_quality_flags(
    source_ids,
    *,
    batch: int = FETCH_BATCH,
    timeout: int = 120,
    retries: int = 4,
    max_workers: int = FETCH_WORKERS,
    tap_url: str = GAIA_TAP_URL,
    deadline: float | None = None,
) -> pd.DataFrame:
    """Fetch ``ruwe`` + ``phot_bp_rp_excess_factor`` for ``source_ids`` from
    gaiadr3.gaia_source via a batched TAP sync query. Returns a DataFrame with
    columns ``source_id`` (str) + :data:`QUALITY_COLS`.

    Batches are issued CONCURRENTLY (TAP sync is network-latency-bound; urllib
    releases the GIL during socket I/O), with results reassembled in input-id
    order so the output (and the on-disk cache) is identical to the legacy
    sequential fetch. Raises if ANY batch fails after ``retries`` or when
    ``deadline`` (``time.monotonic()``) passes -- all or nothing, so the ZP is
    never calibrated on a partially filtered calibrator set.
    """
    ids = []
    seen = set()
    for s in source_ids:
        s = str(s).strip()
        if not s or s.lower() == "nan" or s in seen:
            continue
        seen.add(s)
        ids.append(s)
    if not ids:
        return pd.DataFrame(columns=["source_id"] + QUALITY_COLS)

    chunks = [ids[i:i + batch] for i in range(0, len(ids), batch)]
    frames: list = [None] * len(chunks)
    workers = max(1, min(max_workers, len(chunks)))
    n_done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_idx = {
            ex.submit(_fetch_quality_batch, chunk, tap_url=tap_url,
                      timeout=timeout, retries=retries, deadline=deadline): bi
            for bi, chunk in enumerate(chunks)
        }
        for fut in as_completed(fut_to_idx):
            frames[fut_to_idx[fut]] = fut.result()  # re-raises on permanent batch failure
            n_done += 1
            if n_done % 10 == 0 or n_done == len(chunks):
                logger.info("Gaia quality fetch: %d/%d batches done (%d sources, %d workers)",
                            n_done, len(chunks), len(ids), workers)
    out = pd.concat(frames, ignore_index=True)
    out["source_id"] = out["source_id"].astype(str)
    for c in QUALITY_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out.drop_duplicates("source_id")[["source_id"] + QUALITY_COLS]


def load_quality_flags(
    source_ids,
    *,
    cache_path: str | Path | None = None,
    **fetch_kwargs,
) -> pd.DataFrame:
    """Return quality flags for ``source_ids``, reusing ``cache_path``.

    T7-(d) (2026-08-15, user decision): a PARTIAL cache is reused too --
    only the ids the cache does not hold are fetched, and the merged set
    (cached rows + new rows) is written back. A prefetch cut short no
    longer wastes its persisted batches, and an inline fetch racing a
    still-running prefetch downloads only the remainder instead of the
    world. Returns a DataFrame ``source_id`` + :data:`QUALITY_COLS`; note
    it may contain MORE ids than requested (the cache's union -- callers
    join on source_id).
    """
    ids = {str(s).strip() for s in source_ids if str(s).strip() and str(s).strip().lower() != "nan"}
    cached = None
    if cache_path is not None and Path(cache_path).exists():
        try:
            cached = pd.read_csv(cache_path, dtype={"source_id": str})
            if not (set(QUALITY_COLS) <= set(cached.columns)):
                cached = None
            elif ids <= set(cached["source_id"]):
                logger.info("Gaia quality: using cache %s (%d sources)", cache_path, len(cached))
                return cached[["source_id"] + QUALITY_COLS]
        except Exception as exc:  # noqa: BLE001
            logger.info("Gaia quality cache %s unreadable (%s); refetching", cache_path, exc)
            cached = None
    missing = ids if cached is None else ids - set(cached["source_id"])
    if cached is not None:
        logger.info(
            "Gaia quality: cache %s covers %d of %d ids -- fetching only the "
            "%d missing", cache_path, len(ids) - len(missing), len(ids),
            len(missing))
    flags = fetch_gaia_quality_flags(sorted(missing), **fetch_kwargs)
    if cached is not None:
        flags = pd.concat(
            [cached[["source_id"] + QUALITY_COLS], flags],
            ignore_index=True).drop_duplicates("source_id")
    if cache_path is not None:
        try:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            # Atomic write (tmp + rename): a concurrent reader -- e.g.
            # compute_zp racing the background prefetch thread -- must never
            # see a torn-but-parseable file and silently take wrong flags.
            tmp = str(cache_path) + ".tmp"
            flags.to_csv(tmp, index=False)
            os.replace(tmp, cache_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not write Gaia quality cache %s: %s", cache_path, exc)
    return flags


def augment_csv(in_path: str | Path, out_path: str | Path | None = None) -> Path:
    """Write a copy of a GaiaXP synphot CSV with :data:`QUALITY_COLS` added
    (fetched from Gaia DR3). Returns the output path. Used by the
    ``tract7dt augment-gaia`` CLI subcommand."""
    in_path = Path(in_path)
    out_path = Path(out_path) if out_path else in_path.with_name(in_path.stem + "_q.csv")
    g = pd.read_csv(in_path, dtype={"source_id": str})
    if all(c in g.columns for c in QUALITY_COLS):
        logger.info("%s already has %s; copying unchanged.", in_path, QUALITY_COLS)
        g.to_csv(out_path, index=False)
        return out_path
    flags = fetch_gaia_quality_flags(g["source_id"].tolist())
    g = g.drop(columns=[c for c in QUALITY_COLS if c in g.columns], errors="ignore")
    g = g.merge(flags, on="source_id", how="left")
    g.to_csv(out_path, index=False)
    logger.info("Wrote %s: %d/%d sources with Gaia quality flags",
                out_path, int(g["ruwe"].notna().sum()), len(g))
    return out_path


def prefetch_quality_flags_async(
    source_ids,
    *,
    cache_path: str | Path,
    on_error=None,
    deadline: float | None = None,
) -> threading.Thread:
    """Warm ``cache_path`` in the background while the PSF stage runs.

    Starts a daemon thread and returns it; ``zp.require_gaia_quality_flags``
    joins it before the fit (the thread's exception, if any, is kept on
    ``thread.error``). ``deadline`` is the run's single time budget, shared
    with that verification call.

    **It deliberately passes NO other fetch parameters**, so the prefetch and
    the verification fetch are the same call. They drifted
    once and the result was the opposite of the intent: the prefetch was pinned
    to ``max_workers=1, timeout=60, retries=2`` while the inline path used the
    defaults ``8 / 120 / 4``, i.e. the warm-up was 8x slower and materially more
    fragile than the fetch it exists to preempt. Measured on one crowded tile,
    same archive, minutes apart: 1 worker did 10 of 53 batches in 373 s (~33 min
    projected) while 8 workers did all 51 in 3 min 46 s. The prefetch never once
    won the race; the inline path did the work both times, and because the
    inline path starts its own fetch while the prefetch is still running, the
    archive saw 9 concurrent connections for one id set and closed the
    connection. Keep the two calls identical.

    The pin's stated reason -- "a single worker bounds any abort-time exit delay
    to one in-flight batch instead of a full parallel drain" -- is false.
    ``concurrent.futures.thread._python_exit`` puts ONE sentinel at the BACK of
    the work queue and joins, so every queued batch is drained at exit whatever
    the worker count; one worker just drains them serially. Measured, 10 x 1 s
    tasks: 1 worker exits in 10.05 s, 8 workers in 2.05 s.
    """
    ids = list(source_ids)
    # T7-(d): the prefetch persists PROGRESS, one fetch wave at a time, and
    # can be asked to stop at a wave boundary (``thread.stop.set()``). Each
    # segment call goes through load_quality_flags, which merges into the
    # cache -- so a prefetch cut short leaves everything it finished on
    # disk, and the inline path then fetches only the remainder. Segment
    # size = one full concurrent wave (FETCH_BATCH x FETCH_WORKERS): smaller
    # would idle workers, larger would just delay persistence.
    stop = threading.Event()
    seg = FETCH_BATCH * FETCH_WORKERS

    def _run() -> None:
        try:
            for i in range(0, len(ids), seg):
                if stop.is_set():
                    logger.info(
                        "Gaia quality prefetch: stop requested -- %d/%d ids "
                        "already persisted to the cache", i, len(ids))
                    return
                load_quality_flags(ids[i:i + seg], cache_path=str(cache_path),
                                   deadline=deadline)
        except Exception as exc:                               # noqa: BLE001
            t.error = exc
            if on_error is not None:
                on_error(exc)

    t = threading.Thread(target=_run, name="gaia-qf-prefetch", daemon=True)
    t.stop = stop
    t.error = None
    t.start()
    return t


__all__ = ["QUALITY_BUDGET_S", "QUALITY_COLS", "QualityFetchTimeout",
           "fetch_gaia_quality_flags", "load_quality_flags",
           "prefetch_quality_flags_async", "augment_csv"]
