"""Gate for the Gaia DR3 quality-flag prefetch (efficiency arc T7, 2026-08-03).

`compute_zp` filters its ZP calibrators on RUWE + BP/RP-excess, fetched from the
Gaia archive. `augment_gaia` starts a BACKGROUND prefetch of the same flags so
that fetch is a cache read by the time ZP runs.

The two calls drifted, and the drift inverted the intent: the prefetch was
pinned to `max_workers=1, timeout=60, retries=2` while the inline fetch used the
defaults `8 / 120 / 4`. Measured on one crowded tile, same archive, minutes
apart: the 1-worker prefetch did 10 of 53 batches in 373 s (~33 min projected)
while the inline 8-worker fetch did all 51 in 3 min 46 s. The prefetch never won
the race; worse, the inline path starts its own fetch while the prefetch is
still running, so the archive saw 9 concurrent connections for one id set and
closed the connection -- leaving that pass's quality filter inert.

The contract being gated:
  1. the prefetch issues EXACTLY the same fetch call as the inline path, so the
     two cannot drift again;
  2. the fetch is concurrent by default (the specific regression);
  3. the prefetch is best-effort: a failure is reported, never raised;
  4. a cache that already covers the ids means NO fetch at all.
"""
from __future__ import annotations

import pandas as pd
import pytest

from tract7dt import gaia_quality as gq

IDS = ["100", "200", "300"]


def _fake_flags(ids):
    return pd.DataFrame({"source_id": [str(i) for i in ids],
                         "ruwe": [1.0] * len(ids),
                         "phot_bp_rp_excess_factor": [1.2] * len(ids)})


@pytest.fixture
def recorded(monkeypatch):
    """Capture every fetch the module makes, with its parameters."""
    calls: list[dict] = []

    def _fake_fetch(source_ids, **kwargs):
        calls.append(dict(kwargs))
        return _fake_flags(source_ids)

    monkeypatch.setattr(gq, "fetch_gaia_quality_flags", _fake_fetch)
    return calls


def test_prefetch_issues_the_same_fetch_call_as_the_inline_path(recorded, tmp_path):
    """The invariant that broke: warm-up and inline fetch must be one call.
    Since R29 (2026-10-02) both carry the run's one deadline, and nothing
    else."""
    import time
    inline_cache = tmp_path / "inline.csv"
    pre_cache = tmp_path / "pre.csv"
    deadline = time.monotonic() + 60.0

    # what zp.require_gaia_quality_flags does before the fit
    gq.load_quality_flags(IDS, cache_path=str(inline_cache), deadline=deadline)
    # what augment_gaia does
    gq.prefetch_quality_flags_async(IDS, cache_path=str(pre_cache),
                                    deadline=deadline).join(30)

    assert len(recorded) == 2, recorded
    assert recorded[0] == recorded[1] == {"deadline": deadline}, (
        f"prefetch and inline fetch drifted: {recorded[0]} vs {recorded[1]}")


def test_fetch_is_concurrent_by_default():
    """The specific regression: a serial warm-up cannot win its own race.

    (And a single worker does not even bound abort-time exit as its retired
    comment claimed -- `_python_exit` puts ONE sentinel at the BACK of the work
    queue, so every batch drains at exit whatever the worker count; measured
    10x1 s tasks: 1 worker 10.05 s, 8 workers 2.05 s.)
    """
    import inspect
    dflt = inspect.signature(gq.fetch_gaia_quality_flags).parameters
    assert dflt["max_workers"].default > 1
    assert dflt["retries"].default >= 2
    assert dflt["timeout"].default >= 60


def test_prefetch_is_best_effort_and_never_raises(monkeypatch, tmp_path):
    def _boom(source_ids, **kwargs):
        raise RuntimeError("archive down")

    monkeypatch.setattr(gq, "fetch_gaia_quality_flags", _boom)
    seen: list[BaseException] = []
    t = gq.prefetch_quality_flags_async(
        IDS, cache_path=str(tmp_path / "c.csv"), on_error=seen.append)
    t.join(30)
    assert not t.is_alive()
    assert len(seen) == 1 and "archive down" in str(seen[0])
    # the thread keeps the cause for the pre-fit check (R29)
    assert t.error is seen[0]


def test_covering_cache_means_no_fetch(recorded, tmp_path):
    cache = tmp_path / "c.csv"
    _fake_flags(IDS + ["400"]).to_csv(cache, index=False)
    gq.load_quality_flags(IDS, cache_path=str(cache))
    assert recorded == [], "a covering cache must not hit the archive"


# ---- T7-(d), 2026-08-15 (user decision): fetch only what is missing -------

def test_partial_cache_fetches_only_the_missing(monkeypatch, tmp_path):
    """A partial cache is an asset, not a miss: only the uncovered ids go to
    the archive, and the cache is rewritten as the union."""
    fetched: list[list[str]] = []

    def _fake_fetch(source_ids, **kwargs):
        fetched.append([str(s) for s in source_ids])
        return _fake_flags(source_ids)

    monkeypatch.setattr(gq, "fetch_gaia_quality_flags", _fake_fetch)
    cache = tmp_path / "c.csv"
    _fake_flags(["100", "200"]).to_csv(cache, index=False)

    out = gq.load_quality_flags(["100", "200", "300", "400"],
                                cache_path=str(cache))

    assert fetched == [["300", "400"]], fetched
    assert set(out["source_id"]) >= {"100", "200", "300", "400"}
    on_disk = pd.read_csv(cache, dtype={"source_id": str})
    assert set(on_disk["source_id"]) == {"100", "200", "300", "400"}


def test_prefetch_persists_per_wave_and_honors_stop(monkeypatch, tmp_path):
    """The prefetch works in fetch-sized waves, each merged into the cache,
    and a stop request lands at the next wave boundary -- so an interrupted
    prefetch leaves everything it finished on disk."""
    import threading

    monkeypatch.setattr(gq, "FETCH_BATCH", 1)
    monkeypatch.setattr(gq, "FETCH_WORKERS", 1)
    ready = threading.Event()
    box: dict = {}
    calls: list[list[str]] = []

    def _fake_fetch(source_ids, **kwargs):
        calls.append([str(s) for s in source_ids])
        assert ready.wait(30)
        box["stop"].set()          # ask to stop after THIS wave
        return _fake_flags(source_ids)

    monkeypatch.setattr(gq, "fetch_gaia_quality_flags", _fake_fetch)
    cache = tmp_path / "c.csv"
    t = gq.prefetch_quality_flags_async(IDS, cache_path=str(cache))
    box["stop"] = t.stop
    ready.set()
    t.join(30)
    assert not t.is_alive()
    assert calls == [["100"]], f"exactly one wave should have run: {calls}"
    on_disk = pd.read_csv(cache, dtype={"source_id": str})
    assert set(on_disk["source_id"]) == {"100"}, "the finished wave persists"


def test_stopped_prefetch_plus_inline_covers_all_ids_once(monkeypatch, tmp_path):
    """End-to-end race resolution: prefetch persisted waves + inline
    missing-only fetch together download every id exactly once."""
    fetched: list[str] = []

    def _fake_fetch(source_ids, **kwargs):
        fetched.extend(str(s) for s in source_ids)
        return _fake_flags(source_ids)

    monkeypatch.setattr(gq, "fetch_gaia_quality_flags", _fake_fetch)
    cache = tmp_path / "c.csv"
    _fake_flags(["100"]).to_csv(cache, index=False)  # prefetch got this far

    out = gq.load_quality_flags(IDS, cache_path=str(cache))  # inline path

    assert sorted(fetched) == ["200", "300"], fetched
    assert set(out["source_id"]) == set(IDS)
    assert len(fetched) == len(set(fetched)), "no id downloaded twice"
