"""Island chunk-width REUSE decision (effarc2 K12 + R2).

`_shard_reuse_width` decides whether a chunk should run at a wider rung the worker has already
built instead of compiling its own narrow one. Getting it wrong is expensive in both directions,
and both directions were measured on real anchors:

  * compiling a fresh narrow rung costs a full six-kernel XLA build -- the sparse cold anchor
    logged a 1-island tail at width 1 taking 20.2 s and a 45-island tail at width 64 taking
    17.5 s, against 0.6 s for the same class's already-built 128-wide chunk;
  * padding to a wide rung costs (width / n_rows) times the per-island work -- the crowded anchor,
    under an unconditional rule, logged class (96,32) running 2-island re-queue chunks at width
    256 for 80.5 s and 71.6 s where width 2 costs 32.4 s.

So the decision is made from two numbers the worker measures on the class itself (per-lane-step
cost and fresh-rung compile cost). These tests pin the decision on the exact cases those logs
produced, plus the boundary behaviours (no evidence yet, full chunks, nothing built).

No GPU, no JAX execution -- the functions under test are pure host bookkeeping.
"""
from __future__ import annotations

import numpy as np
import pytest

from tract7dt.fit.grid_gpu import island_stage as isl


@pytest.fixture(autouse=True)
def _clean_shard():
    isl._SHARD.clear()
    yield
    isl._SHARD.clear()


def _setup(ci, observations, built):
    isl._SHARD["built_widths"] = {ci: set(built)}
    for width, steps, wall in observations:
        isl._shard_note_chunk(ci, width, steps, wall)


# ---------------------------------------------------------------- cost model


def test_cost_model_needs_two_observations():
    _setup(0, [(256, 8, 90.0)], [256])
    assert isl._shard_cost_model(0) is None, (
        "with one observation the rate and the compile excess are the same "
        "measurement, so the model must report that it has no evidence")


def test_cost_model_rate_is_the_compile_free_minimum():
    # first chunk of a shape pays the build, the second does not
    _setup(0, [(4096, 3, 30.0), (4096, 3, 1.5)], [4096])
    rate, compile_s = isl._shard_cost_model(0)
    assert rate == pytest.approx(1.5 / (4096 * 3))
    assert compile_s == pytest.approx(28.5, rel=1e-6)


def test_cost_model_compile_is_never_negative():
    _setup(0, [(128, 2, 3.0), (128, 2, 3.0)], [128])
    _, compile_s = isl._shard_cost_model(0)
    assert compile_s >= 0.0


# ------------------------------------------------- decisions from real logs


@pytest.mark.parametrize(
    "tag, obs, built, n_rows, width, phase_a, max_steps, expect",
    [
        # crowded (96,32): 0.039 s per lane-step -- padding 2 islands out to 256
        # for 22 re-queue steps is ~220 s against a ~10 s build. REFUSE.
        ("crowded-96-32-hard", [(256, 8, 90.0), (256, 8, 80.0)], [256],
         2, 2, False, 22, 2),
        # crowded (64,16): same shape of answer at a different rung.
        ("crowded-64-16-hard", [(1024, 8, 77.0), (1024, 8, 60.0)], [1024],
         4, 4, False, 22, 4),
        # crowded (32,1): 1.2e-4 s per lane-step -- padding 1,024 islands out to
        # 4096 is ~11 s against a ~28 s build. WIDEN. (Measured: 1.1 s vs 58.7 s.)
        ("crowded-32-1-hard", [(4096, 3, 30.0), (4096, 3, 1.5)], [4096],
         1024, 1024, False, 22, 4096),
        # crowded (48,8): cheap class but a 3-island chunk padded to 2048 still
        # loses against a 2 s build. REFUSE.
        ("crowded-48-8-hard", [(2048, 6, 13.0), (2048, 6, 11.0)], [2048],
         3, 4, False, 22, 4),
        # sparse (48,1): a 45-island phase-A tail padded to the built 128 costs
        # less than building width 64. WIDEN.
        ("sparse-48-1-tail", [(128, 2, 17.0), (128, 2, 3.4)], [128],
         45, 64, True, 8, 128),
        # crowded (96,32) phase-A tail: the class is expensive per lane, so even
        # at the main-pass step budget padding 28 islands to 256 loses. REFUSE.
        ("crowded-96-32-tail", [(256, 8, 90.0), (256, 8, 80.0)], [256],
         28, 32, True, 8, 32),
    ],
)
def test_reuse_decision_matches_measured_anchors(
    tag, obs, built, n_rows, width, phase_a, max_steps, expect
):
    _setup(3, obs, built)
    assert isl._shard_reuse_width(3, n_rows, width, phase_a, max_steps) == expect, tag


# --------------------------------------------------------------- boundaries


def test_full_phase_a_chunk_is_never_widened():
    # it already runs at the class's measured width, which the probe built:
    # there is no compile to avoid, so extra lanes would be pure waste
    _setup(3, [(256, 8, 90.0), (256, 8, 80.0)], [256, 512])
    assert isl._shard_reuse_width(3, 256, 256, True, 8) == 256


def test_full_phase_b_chunk_may_be_widened():
    # a FULL phase-B chunk sits on a narrow re-queue rung nothing has compiled,
    # so the full-chunk short-circuit must not apply to it
    _setup(3, [(4096, 3, 30.0), (4096, 3, 1.5)], [4096])
    assert isl._shard_reuse_width(3, 1024, 1024, False, 22) == 4096


def test_without_evidence_phase_a_widens_and_phase_b_does_not():
    # phase A is bounded by one ordinary chunk of the class; phase B is not
    _setup(3, [(128, 2, 17.0)], [128])
    assert isl._shard_reuse_width(3, 45, 64, True, 8) == 128
    assert isl._shard_reuse_width(3, 4, 4, False, 22) == 4


def test_nothing_built_means_no_change():
    isl._SHARD["built_widths"] = {}
    assert isl._shard_reuse_width(3, 4, 8, True, 8) == 8


def test_never_narrows():
    _setup(3, [(128, 2, 17.0), (128, 2, 3.4)], [8, 128])
    # the only built rung below the enqueued width must not be selected
    assert isl._shard_reuse_width(3, 100, 128, True, 8) == 128


# --------------------------------------------- halved chunks are not samples
#
# The previous version of this test was VACUOUS: it set `built_widths` and
# asserted the model was empty, without ever driving the code that records a
# sample. It therefore passed while `_shard_run_chunk` was recording halved
# chunks -- found by an independent audit 2026-08-29, see the development ledger
# SS23-21. The tests below drive the real producer and the real consumer.


class _MemErr(RuntimeError):
    """Shaped like the failure `_is_device_mem_error` recognises."""


def _fake_leaf(wall_s, steps=3, n=1):
    return {"wall_s": float(wall_s), "steps_run": int(steps),
            "flux": np.zeros(n, dtype=np.float32)}


def test_resilient_reports_a_clean_run_as_not_halved(monkeypatch):
    from tract7dt.fit.grid_gpu import island_solver as sol
    monkeypatch.setattr(sol, "run_class_chunk",
                        lambda *a, **k: _fake_leaf(1.25, n=len(a[3])))
    out = sol.run_class_chunk_resilient(
        None, None, None, np.arange(8), 8, 3, None, None, 0)
    assert out["widths_ran"] == [8]
    assert out["halved"] is False


def test_resilient_flags_a_halved_run_that_the_width_set_alone_hides(monkeypatch):
    """The defect this fix exists for, asserted from both ends.

    A uniform single-level halving is the ORDINARY out-of-memory case. Its
    `widths_ran` is a deduplicated set, so it collapses to ONE element and is
    indistinguishable by length from a clean run -- while `_merge_chunk_outputs`
    has already SUMMED `wall_s` over the leaves. Recording that as one chunk of
    the halved width makes the per-lane-step rate ~2x too large.
    """
    from tract7dt.fit.grid_gpu import island_solver as sol
    calls = []

    def fake(kern, psf, host, rows, width, *a, **k):
        calls.append(int(width))
        if int(width) > 4:
            raise _MemErr("RESOURCE_EXHAUSTED: Failed to allocate 5.42GiB")
        return _fake_leaf(2.0, n=len(rows))

    monkeypatch.setattr(sol, "run_class_chunk", fake)
    out = sol.run_class_chunk_resilient(
        None, None, None, np.arange(8), 8, 3, None, None, 0)

    assert calls == [8, 4, 4], calls              # one wide try, two halves
    # the trap: length alone cannot see the halving
    assert out["widths_ran"] == [4] and len(out["widths_ran"]) == 1
    # ...and the wall time is the SUM, so the sample would be ~2x wrong
    assert out["wall_s"] == pytest.approx(4.0)
    # ...which is exactly why the producer must say so explicitly
    assert out["halved"] is True


def _prime_shard_for_run_chunk(monkeypatch, resilient_result):
    isl._SHARD.update(payload={"pad": 0, "compute_param_errors": False,
                               "compute_re_profile": False,
                               "compute_source_metrics": False,
                               "compute_flux_marg": False},
                      dstack_dev=None, wstack_dev=None)
    monkeypatch.setattr(isl, "_shard_class_setup",
                        lambda ci: (None, None, None, None))
    from tract7dt.fit.grid_gpu import island_solver as sol
    monkeypatch.setattr(sol, "run_class_chunk_resilient",
                        lambda *a, **k: dict(resilient_result))


def test_shard_run_chunk_records_an_un_halved_chunk(monkeypatch):
    _prime_shard_for_run_chunk(monkeypatch, {
        "widths_ran": [128], "halved": False, "wall_s": 3.0, "steps_run": 2})
    isl._shard_run_chunk(3, np.arange(128), 128, 2, None)
    isl._shard_note_chunk(3, 128, 2, 1.0)     # a second, cheaper sample
    assert isl._shard_cost_model(3) is not None


def test_shard_run_chunk_refuses_to_record_a_halved_chunk(monkeypatch):
    # same shape of result, only `halved` differs -- the model must stay empty
    _prime_shard_for_run_chunk(monkeypatch, {
        "widths_ran": [128], "halved": True, "wall_s": 3.0, "steps_run": 2})
    isl._shard_run_chunk(3, np.arange(200), 256, 2, None)
    assert isl._SHARD.get("chunk_obs", {}).get(3) in (None, [])
    assert isl._shard_cost_model(3) is None
