"""Shared utilities used across tract7dt modules.

Helpers factored here in Phase 1e to remove duplicated idioms (catalog
column resolution, etc.) from the original monolithic modules. Behavior is
intentionally identical to the inline versions they replace, so byte-identity
against the v0.7.0 baseline is preserved.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("tract7dt.utils")


def case_insensitive_col_map(columns: Iterable[str]) -> dict[str, str]:
    """Return a ``{lowered_stripped_name: original_name}`` map for the given columns.

    Replaces the duplicated idiom

    ``col_map = {str(c).strip().lower(): str(c) for c in <columns>}``

    that was inlined in ``load_inputs``, ``patch_inputs``, ``bright_mask``,
    ``zp``, ``fit.runner``, and ``psf.builder``. Behavior is identical.
    """
    return {str(c).strip().lower(): str(c) for c in columns}


def resolve_ra_dec_cols(
    df: pd.DataFrame,
    *,
    ra_aliases: tuple[str, ...] = ("ra",),
    dec_aliases: tuple[str, ...] = ("dec",),
    type_aliases: tuple[str, ...] = ("type",),
    id_aliases: tuple[str, ...] = ("id",),
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Resolve catalog column names case-insensitively.

    Returns ``(ra_col, dec_col, type_col, id_col)``; each is ``None`` if no
    alias matches a column in ``df``. Aliases are matched after stripping
    and lowercasing the dataframe column names.

    The default aliases match the convention used across the pipeline:
    ``RA`` / ``DEC`` / ``TYPE`` / ``ID`` (in any case).
    """
    col_map = case_insensitive_col_map(df.columns)

    def _first_alias(aliases: tuple[str, ...]) -> Optional[str]:
        for a in aliases:
            v = col_map.get(a)
            if v is not None:
                return v
        return None

    return (
        _first_alias(ra_aliases),
        _first_alias(dec_aliases),
        _first_alias(type_aliases),
        _first_alias(id_aliases),
    )


def _csv_block_to_str(block_df: pd.DataFrame, include_header: bool) -> str:
    """Format one contiguous row-block to a CSV string (module-level so it is
    picklable for spawn workers)."""
    return block_df.to_csv(index=False, header=include_header)


def write_csv_parallel(
    df: pd.DataFrame,
    path,
    *,
    workers: int = 8,
    min_rows: int = 100_000,
) -> None:
    """Write ``df`` to ``path`` BYTE-IDENTICALLY to ``df.to_csv(path, index=False)``
    but format contiguous row-blocks across ``workers`` *spawn* processes.

    pandas ``to_csv`` is single-threaded and CPU-bound on float->str formatting,
    which does not release the GIL, so only processes (not threads) parallelize
    it. Per-cell formatting (incl. quoting of comma-containing fields) is
    row-independent, so concatenating ``block.to_csv(header=(i==0))`` in row order
    reproduces the monolithic write exactly. Spawn (not fork) is mandatory because
    callers (gpu mode) import JAX/CUDA in the parent.

    Falls back to a single-threaded exact ``to_csv`` for small frames, ``workers<=1``,
    or on any worker failure, so the output is always correct.
    """
    n = len(df)
    if workers <= 1 or n < int(min_rows):
        df.to_csv(path, index=False)
        return

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    workers = max(1, min(int(workers), n))
    bounds = np.linspace(0, n, workers + 1).astype(int)
    blocks: list[pd.DataFrame] = []
    headers: list[bool] = []
    for i in range(workers):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        if hi > lo:
            blocks.append(df.iloc[lo:hi])
            headers.append(len(blocks) == 1)  # header only on the first block
    if len(blocks) <= 1:
        df.to_csv(path, index=False)
        return

    try:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(blocks), mp_context=ctx) as ex:
            parts = list(ex.map(_csv_block_to_str, blocks, headers))
    except Exception as exc:  # pragma: no cover - safety net
        logger.warning(
            "write_csv_parallel: parallel write failed (%s); "
            "falling back to single-threaded to_csv", exc,
        )
        df.to_csv(path, index=False)
        return

    # Concatenated block strings are byte-identical to a single to_csv on Linux
    # (os.linesep == '\n'). Write sequentially to avoid materializing a second
    # full-size joined string.
    with open(os.fspath(Path(path)), "w", newline="") as fh:
        for p in parts:
            fh.write(p)
