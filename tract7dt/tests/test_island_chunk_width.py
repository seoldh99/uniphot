"""T5/#28 -- the island chunk width is MEASURED, not modelled.

These are the host-side guards that need no GPU: the ladder arithmetic, the
retry's error classification, the split-retry merge, and a regression guard
that the deleted byte model does not come back. The measurement itself
(``measure_class_chunk_width``) needs a card and compiled kernels and is gated
by field runs -- see the development ledger 7 (not distributed).
"""

import numpy as np
import pytest

from tract7dt.fit.grid_gpu import island_plan


class TestLadderWidth:
    """Widths are powers of two so the search is O(log w) AND the same
    handful of shapes recur field to field (persistent JIT cache)."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 9, 100, 1000, 4096, 9999])
    def test_is_a_power_of_two(self, n):
        w = island_plan.ladder_width(n, 4096)
        assert w & (w - 1) == 0, f"{w} is not a power of two"

    @pytest.mark.parametrize("n,cap,want", [
        (1, 4096, 1), (2, 4096, 2), (3, 4096, 4), (5, 4096, 8),
        (8, 4096, 8), (9, 4096, 16), (100, 4096, 128), (2234, 4096, 4096),
    ])
    def test_smallest_rung_that_holds_the_rows(self, n, cap, want):
        assert island_plan.ladder_width(n, cap) == want

    def test_never_below_one_and_never_above_cap(self):
        assert island_plan.ladder_width(0, 4096) == 1
        assert island_plan.ladder_width(-5, 4096) == 1
        assert island_plan.ladder_width(10_000, 64) == 64
        assert island_plan.ladder_width(3, 0) == 1

    def test_tail_chunk_pads_to_its_own_rung_not_the_class_width(self):
        # the point of the rule: a 5-island tail of a 4096-wide class must
        # not run 4096 wide.
        assert island_plan.ladder_width(5, 4096) == 8


class TestByteModelStaysDeleted:
    """The hand-calibrated peak-memory model was measured wrong in three
    independent ways (CHANGELOG 2026-08-02). Reintroducing it -- or its GB
    ceiling -- would silently take the width decision away from the
    measurement again."""

    @pytest.mark.parametrize("name", [
        "_island_step_bytes", "_STEP_MEM_BUDGET",
        "_MEM_A_J", "_MEM_A_M", "_MEM_A_T", "_MEM_C_STAR", "_MEM_C_FFT",
        "_MEM_P_STAR", "_MEM_P_FFT", "_MEM_SCAN_DB",
    ])
    def test_gone(self, name):
        assert not hasattr(island_plan, name), (
            f"island_plan.{name} is back -- the chunk width is measured on "
            f"the device now (island_solver.measure_class_chunk_width)")

    def test_env_budget_is_gone(self, monkeypatch):
        import importlib
        monkeypatch.setenv("TRACT7DT_ISLAND_STEP_MEM_GB", "3")
        mod = importlib.reload(island_plan)
        try:
            assert not hasattr(mod, "_STEP_MEM_BUDGET")
        finally:
            monkeypatch.delenv("TRACT7DT_ISLAND_STEP_MEM_GB", raising=False)
            importlib.reload(island_plan)

    def test_debug_chunk_override_still_exists(self):
        # TRACT7DT_ISLAND_CHUNK is the probe/debug forcing knob and must keep
        # winning over the measurement.
        assert hasattr(island_plan, "_CHUNK_OVERRIDE")


class TestDeviceMemErrorClassification:
    """Both failures the crowded tile produced on 2026-08-02 came from one
    cause (a too-wide chunk) and both must trigger the halve-and-retry."""

    def _f(self):
        from tract7dt.fit.grid_gpu.island_solver import _is_device_mem_error
        return _is_device_mem_error

    @pytest.mark.parametrize("msg", [
        "RESOURCE_EXHAUSTED: Failed to allocate request for 19.68GiB",
        "INTERNAL: Autotuning failed for HLO %gemm_fusion_dot.14 ... "
        "NOT_FOUND: No valid config found!",
        "Out of memory while trying to allocate 1234 bytes",
    ])
    def test_retryable(self, msg):
        assert self._f()(RuntimeError(msg))

    @pytest.mark.parametrize("msg", [
        "chunk width 4 < rows 9",
        "ValueError: shapes (3,) and (4,) not aligned",
        "KeyError: 'mmask'",
    ])
    def test_not_retryable(self, msg):
        assert not self._f()(RuntimeError(msg))


class TestSplitRetryMerge:
    def test_rows_concatenate_and_scalars_reduce(self):
        from tract7dt.fit.grid_gpu.island_solver import _merge_chunk_outputs
        a = dict(p=np.zeros((2, 3)), chi2=np.array([1.0, 2.0]), perr=None,
                 wall_s=1.5, steps_run=4)
        b = dict(p=np.ones((3, 3)), chi2=np.array([3.0, 4.0, 5.0]), perr=None,
                 wall_s=2.5, steps_run=7)
        out = _merge_chunk_outputs([a, b])
        assert out["p"].shape == (5, 3)
        assert np.allclose(out["chi2"], [1, 2, 3, 4, 5])
        assert out["perr"] is None            # absent stays absent
        assert out["wall_s"] == pytest.approx(4.0)     # summed
        assert out["steps_run"] == 7                   # max, not sum


class TestFcorrPerPixelIsMeasured:
    """T5 bonus: the raw-sky pass's per-band memory came from a hand-counted
    'about eleven float64 planes'. It is now measured by running the
    estimator once."""

    def test_returns_a_positive_measurement_and_says_which(self):
        from tract7dt.fit.grid_gpu.pipeline import _fcorr_bytes_per_pixel
        bpp, how = _fcorr_bytes_per_pixel()
        assert bpp > 0
        assert isinstance(how, str) and how

    def test_worker_count_is_bounded_by_bands_and_cores(self):
        import os
        from tract7dt.fit.grid_gpu.pipeline import _fcorr_band_workers
        img = {f"b{i}": {"img_scaled": np.zeros((512, 512), np.float32)}
               for i in range(4)}
        n, why = _fcorr_band_workers(list(img), img)
        assert 1 <= n <= min(4, os.cpu_count() or 1)
        assert "thread(s)" in why
