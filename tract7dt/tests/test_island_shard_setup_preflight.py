"""Pin the island shard-setup field-stack pre-flight and its per-device report.

Two crowded gate runs (2026-08-29 02:14, 2026-08-30 03:05) died in
``_shard_setup`` on ``RESOURCE_EXHAUSTED ... device ordinal 0`` with a log that
named neither the physical card nor the occupant, and the ordered
``fut.result()`` await hid the other two workers' outcomes.

These tests are CPU-only: no GPU, no real device transfer.
"""
import logging
from concurrent.futures import Future
from multiprocessing import shared_memory

import numpy as np
import pytest

from tract7dt.fit.grid_gpu import island_solver as ISOLV
from tract7dt.fit.grid_gpu import island_stage as IST


class _FakeDeviceArray:
    def __init__(self, a):
        self._a = a

    def block_until_ready(self):
        return self


@pytest.fixture()
def payload(monkeypatch):
    """Two tiny REAL shm segments so `_attach` is exercised unchanged."""
    shape, dt = (2, 4, 8), np.float32
    nbytes = int(np.prod(shape)) * np.dtype(dt).itemsize
    segs = [shared_memory.SharedMemory(create=True, size=nbytes)
            for _ in range(2)]
    try:
        yield dict(
            data_spec=(segs[0].name, shape, "float32"),
            invvar_spec=(segs[1].name, shape, "float32"),
            warm_sersic_table=False,
        ), 2 * nbytes
    finally:
        for s in segs:
            s.close()
            s.unlink()


def _no_gpu(monkeypatch):
    import jax.numpy as jnp
    monkeypatch.setattr(jnp, "asarray", lambda a, *a_, **k: _FakeDeviceArray(a))


def test_preflight_passes_when_the_card_has_room(monkeypatch, payload):
    pl, need = payload
    _no_gpu(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(ISOLV, "_device_free_bytes_nvml", lambda: need + 1)
    seen, dt = IST._shard_setup(pl)
    assert seen == "2" and dt >= 0.0
    assert IST._SHARD["dstack_dev"] is not None


def test_preflight_does_not_fail_when_the_meter_is_unreadable(
        monkeypatch, payload):
    """`_device_free_bytes_nvml` returns 0 on any failure (no nvidia-smi, odd
    driver). A missing meter must never convert a working run into a
    failure -- this is the guard that keeps the patch disclosure-only."""
    pl, _ = payload
    _no_gpu(monkeypatch)
    monkeypatch.setattr(ISOLV, "_device_free_bytes_nvml", lambda: 0)
    seen, dt = IST._shard_setup(pl)
    assert IST._SHARD["dstack_dev"] is not None


def test_preflight_names_card_bytes_and_occupants_when_short(
        monkeypatch, payload):
    pl, need = payload
    _no_gpu(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(ISOLV, "_device_free_bytes_nvml", lambda: need - 1)
    monkeypatch.setattr(ISOLV, "_gpu_occupants",
                        lambda idx=None: [(12345, 22602 * 2 ** 20, "tanush")])
    with pytest.raises(RuntimeError) as ei:
        IST._shard_setup(pl)
    msg = str(ei.value)
    assert "PHYSICAL GPU 0" in msg            # the card, not "ordinal 0"
    assert str(need) in msg                   # how much was needed
    assert str(need - 1) in msg               # how much was free
    assert "12345" in msg and "tanush" in msg  # who else was on it


def test_upload_failure_is_reraised_with_the_same_attribution(
        monkeypatch, payload):
    """Free memory looked fine, then the allocation failed anyway (granularity
    / fragmentation / an occupant arriving between the two reads)."""
    pl, need = payload
    import jax.numpy as jnp

    def _boom(a, *a_, **k):
        raise RuntimeError("RESOURCE_EXHAUSTED: Failed to allocate request "
                           "for 5.42GiB (5824000000B) on device ordinal 0")

    monkeypatch.setattr(jnp, "asarray", _boom)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(ISOLV, "_device_free_bytes_nvml", lambda: need * 10)
    monkeypatch.setattr(ISOLV, "_gpu_occupants", lambda idx=None: [])
    with pytest.raises(RuntimeError) as ei:
        IST._shard_setup(pl)
    assert "PHYSICAL GPU 1" in str(ei.value)
    assert "none reported" in str(ei.value)
    assert isinstance(ei.value.__cause__, RuntimeError)


def test_a_non_memory_error_is_not_dressed_up(monkeypatch, payload):
    pl, need = payload
    import jax.numpy as jnp
    monkeypatch.setattr(jnp, "asarray",
                        lambda *a, **k: (_ for _ in ()).throw(
                            ValueError("bad dtype")))
    monkeypatch.setattr(ISOLV, "_device_free_bytes_nvml", lambda: need * 10)
    with pytest.raises(ValueError, match="bad dtype"):
        IST._shard_setup(pl)


# --- the per-device setup report -------------------------------------------

def _fut(result=None, exc=None):
    f = Future()
    f.set_running_or_notify_cancel()
    if exc is not None:
        f.set_exception(exc)
    else:
        f.set_result(result)
    return f


def test_report_logs_every_card_and_reraises_the_first_failure(caplog):
    boom = RuntimeError("island shard worker on PHYSICAL GPU 0: ...")
    futs = [_fut(exc=boom), _fut(("1", 2.8)), _fut(("2", 2.9))]
    with caplog.at_level(logging.INFO, logger=IST.logger.name):
        with pytest.raises(RuntimeError) as ei:
            IST._report_shard_setup([0, 1, 2], futs)
    assert ei.value is boom
    text = caplog.text
    # the two survivors are NOT hidden any more
    assert "shard worker GPU 1 ready" in text
    assert "shard worker GPU 2 ready" in text
    assert "shard worker GPU 0 setup FAILED" in text
    assert "failed GPU(s) [0]" in text and "ready GPU(s) [1, 2]" in text


def test_report_is_silent_about_failures_when_all_succeed(caplog):
    futs = [_fut(("0", 1.0)), _fut(("1", 1.1)), _fut(("2", 1.2))]
    with caplog.at_level(logging.INFO, logger=IST.logger.name):
        IST._report_shard_setup([0, 1, 2], futs)
    assert caplog.text.count("ready") == 3
    assert "FAILED" not in caplog.text


def test_exactly_enough_free_memory_still_uploads(monkeypatch, payload):
    """`free == need` must NOT be refused.

    The guard is `0 < free0 < need`, strictly less. A boundary that refused
    an allocation the card can actually serve would be the same class of
    defect as the missing pre-flight itself -- a healthy run killed by a
    meter -- so it is pinned rather than left to the reader of the operator.
    """
    pl, need = payload
    _no_gpu(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(ISOLV, "_device_free_bytes_nvml", lambda: need)
    seen, dt = IST._shard_setup(pl)
    assert seen == "0"
    assert dt >= 0.0
