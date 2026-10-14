"""G1 (2026-08-18): measured per-grid GPU peak for the pack decision.

Covers the three moving parts and the contract that must NOT move:

- ``_grid_task_dims`` is the single shape derivation for BOTH paths;
- ``_estimate_grid_peak_mib`` (the STRUCTURAL FALLBACK) computes exactly the
  historical formula;
- ``_grid_resident_bytes`` / ``_measured_grid_peak_mib`` measure, guard, and
  fall back with explicit reasons -- never guess;
- ``_choose_workers_per_gpu`` keeps its override / SAFETY / MAX_WPG / diag
  contract, applies the 2.5x headroom ONLY to the structural fallback, and
  caps worker slots at the grid count (ceil(tasks / devices) per GPU).

The probes run on the CPU backend here (no GPU in CI); the CUDA numbers are
gated by the field runs (development ledger SS19/SS20), per the suite
convention that device-requiring measurements are excluded from pytest.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from tract7dt.fit.grid_gpu import pipeline as P


class _FakeSlice:
    def __init__(self, h, w):
        self.y0, self.y1, self.x0, self.x1 = 0, h, 0, w


def _mk_task(n=40, nb=3, h=128, w=128, ph=25, pw=25, *, single=True,
             conf=False, fcorr=False, shared=True,
             bg_poly_order=2):
    t = types.SimpleNamespace()
    t.source_indices = np.arange(n, dtype=np.int64)
    t.bandnames = [f"b{i}" for i in range(nb)]
    t.psf_shape = (ph, pw)
    t.shared_data_per_band = [_FakeSlice(h, w)] if shared else []
    t.data_per_band = None if shared else [np.zeros((h, w), np.float32)]
    t.bg_poly_order = bg_poly_order
    t.single_stage = single
    t.compute_confusion = conf
    t.compute_fcorr_resid = fcorr
    return t


# ---------------------------------------------------------------------------
# _grid_task_dims: one derivation for both paths
# ---------------------------------------------------------------------------

def test_dims_shared_slice_first():
    t = _mk_task(n=7, nb=2, h=100, w=50, ph=15, pw=17)
    assert P._grid_task_dims(t) == (7, 2, 15, 17, (100, 50))


def test_dims_inline_arrays_fallback():
    t = _mk_task(n=3, nb=1, h=64, w=32, shared=False)
    assert P._grid_task_dims(t)[4] == (64, 32)


def test_dims_unknown_geometry_is_none():
    t = _mk_task()
    t.shared_data_per_band = []
    t.data_per_band = None
    assert P._grid_task_dims(t)[4] is None


# ---------------------------------------------------------------------------
# structural fallback: the historical formula, bit-for-bit
# ---------------------------------------------------------------------------

def _structural_expected(n, nb, ph, pw, pix, chunk=512):
    eff = n if chunk <= 0 else min(n, chunk)
    return (1.0e9 + 95.0 * eff * nb * ph * pw + 16.0 * nb * pix) / 1024.0 ** 2


def test_structural_formula_unchanged(monkeypatch):
    monkeypatch.delenv("TRACT7DT_POS_STEP_CHUNK", raising=False)
    t = _mk_task(n=1555, nb=22, h=2920, w=2920, ph=55, pw=55)
    got = P._estimate_grid_peak_mib(t)
    assert got == pytest.approx(
        _structural_expected(1555, 22, 55, 55, 2920 * 2920))
    # the logged sparse-anchor value the ledger quotes (SS19-4): ~6,902 MiB
    assert 6890 < got < 6915


def test_structural_chunk_zero_uses_full_n(monkeypatch):
    monkeypatch.setenv("TRACT7DT_POS_STEP_CHUNK", "0")
    t = _mk_task(n=1555, nb=22, h=2920, w=2920, ph=55, pw=55)
    assert P._estimate_grid_peak_mib(t) == pytest.approx(
        _structural_expected(1555, 22, 55, 55, 2920 * 2920, chunk=0))


def test_structural_default_pix_when_geometry_unknown(monkeypatch):
    monkeypatch.delenv("TRACT7DT_POS_STEP_CHUNK", raising=False)
    t = _mk_task(n=10, nb=2, ph=55, pw=55)
    t.shared_data_per_band = []
    t.data_per_band = None
    assert P._estimate_grid_peak_mib(t) == pytest.approx(
        _structural_expected(10, 2, 55, 55, 1_250_000))


# ---------------------------------------------------------------------------
# residency arithmetic: terms present, axes monotone, no tuned constants
# ---------------------------------------------------------------------------

def test_resident_terms_and_flags():
    base, detail = P._grid_resident_bytes(_mk_task())
    for key in ("bands", "bg", "stamps", "small", "model"):
        assert key in detail
    assert detail["model"] == 0
    with_flag, d2 = P._grid_resident_bytes(_mk_task(conf=True))
    assert d2["model"] == d2["pix"] * 4
    assert with_flag == base + d2["model"]
    with_fcorr, d3 = P._grid_resident_bytes(_mk_task(fcorr=True))
    assert d3["model"] == d3["pix"] * 4


def test_resident_monotone_axes():
    ref, _ = P._grid_resident_bytes(_mk_task())
    more_bands, _ = P._grid_resident_bytes(_mk_task(nb=6))
    more_pix, _ = P._grid_resident_bytes(_mk_task(h=256, w=256))
    more_src, _ = P._grid_resident_bytes(_mk_task(n=400))
    assert more_bands > ref and more_pix > ref and more_src > ref


def test_resident_none_without_geometry():
    t = _mk_task()
    t.shared_data_per_band = []
    t.data_per_band = None
    assert P._grid_resident_bytes(t) is None


# ---------------------------------------------------------------------------
# measured path: guards fall back with explicit reasons, never guess
# ---------------------------------------------------------------------------

def test_measured_guard_3stage():
    peak, why = P._measured_grid_peak_mib(
        [_mk_task(single=False)], {"jit_cache_dir": "none"})
    assert peak is None and "3-stage" in why


def test_measured_guard_no_geometry(monkeypatch):
    monkeypatch.setattr(P, "_parent_backend_is_gpu", lambda: True)
    t = _mk_task()
    t.shared_data_per_band = []
    t.data_per_band = None
    peak, why = P._measured_grid_peak_mib([t], {"jit_cache_dir": "none"})
    assert peak is None and "geometry" in why


def test_measured_guard_empty():
    peak, why = P._measured_grid_peak_mib([], {"jit_cache_dir": "none"})
    assert peak is None


def test_measured_guard_cpu_backend(monkeypatch):
    # When the parent's JAX backend is not a GPU (silent CPU fallback --
    # e.g. an explicit devices list with CUDA broken), the probe must
    # demote: CPU-backend memory numbers must never be adopted as GPU
    # peaks. Simulated via the helper (the real backend depends on the
    # machine this suite runs on -- asserting on it directly made the test
    # fail on any healthy CUDA host).
    monkeypatch.setattr(P, "_parent_backend_is_gpu", lambda: False)
    peak, why = P._measured_grid_peak_mib(
        [_mk_task()], {"jit_cache_dir": "none"})
    assert peak is None and "backend" in why


def test_measured_probe_cpu_backend(monkeypatch):
    # Real lowering on the CPU backend (backend guard bypassed): plumbing +
    # composition, not CUDA numbers (those are field-gated).
    monkeypatch.setattr(P, "_parent_backend_is_gpu", lambda: True)
    tasks = [_mk_task(n=40), _mk_task(n=20)]
    info, why = P._measured_grid_peak_mib(tasks, {"jit_cache_dir": "none"})
    assert info is not None and info["peak_mib"] > 0
    resident_mib = P._grid_resident_bytes(tasks[0])[0] / 1024.0 ** 2
    assert info["peak_mib"] > resident_mib   # the kernel term contributes
    assert info["peak_mib"] == pytest.approx(
        info["resident_mib"] + info["kernel_mib"])
    assert info["arena_mib"] == pytest.approx(
        info["resident_mib"]
        + (P._GRID_SOLVE_WINDOW + 1) * info["kernel_mib"])
    assert "S1" in why and "resident" in why and "probed" in why


def test_measured_conf_probe_failure_keeps_s1(monkeypatch):
    monkeypatch.setattr(P, "_parent_backend_is_gpu", lambda: True)
    from tract7dt.fit.grid_gpu import error_model
    monkeypatch.setattr(error_model, "confusion_call_bytes",
                        lambda *a, **k: None)
    info, why = P._measured_grid_peak_mib(
        [_mk_task(conf=True)], {"jit_cache_dir": "none"})
    assert info is not None and "conf probe failed" in why


def test_measured_s1_probe_failure_falls_back(monkeypatch):
    monkeypatch.setattr(P, "_parent_backend_is_gpu", lambda: True)
    monkeypatch.setattr(P, "_grid_s1_call_bytes", lambda t: None)
    peak, why = P._measured_grid_peak_mib(
        [_mk_task()], {"jit_cache_dir": "none"})
    assert peak is None and "S1 kernel probe failed" in why


def test_measured_arena_maximand_beats_peak_ordering(monkeypatch):
    # A kernel-heavy candidate with the SMALLER resident+kernel sum must
    # still win the arena sizing: candidates are ranked by
    # resident + (window+1) x kernel, the number the caller provisions.
    monkeypatch.setattr(P, "_parent_backend_is_gpu", lambda: True)
    # Both tasks must be PROBED for the ranking to matter: big_resident is
    # the pix-argmax candidate, kernel_heavy the n_src-argmax candidate.
    big_resident = _mk_task(n=400, h=256, w=256)   # large pix -> resident
    kernel_heavy = _mk_task(n=410, h=64, w=64)
    kb = {id(big_resident): 10 * 1024 ** 2, id(kernel_heavy): 12 * 1024 ** 2}
    monkeypatch.setattr(P, "_grid_s1_call_bytes", lambda t: kb[id(t)])
    info, why = P._measured_grid_peak_mib(
        [big_resident, kernel_heavy], {"jit_cache_dir": "none"})
    r_big = P._grid_resident_bytes(big_resident)[0]
    r_k = P._grid_resident_bytes(kernel_heavy)[0]
    W1 = P._GRID_SOLVE_WINDOW + 1
    # sanity: the constructed case really is peak-vs-arena discordant
    assert r_big + kb[id(big_resident)] > r_k + kb[id(kernel_heavy)]
    assert r_k + W1 * kb[id(kernel_heavy)] > r_big + W1 * kb[id(big_resident)]
    assert info["arena_mib"] == pytest.approx(
        (r_k + W1 * kb[id(kernel_heavy)]) / 1024.0 ** 2)


def test_measured_probe_exception_demotes_not_crashes(monkeypatch):
    monkeypatch.delenv("TRACT7DT_GPU_WORKERS_PER_GPU", raising=False)

    def _boom(tasks, cfg):
        raise ValueError("polynomial_order must be 0..5")

    monkeypatch.setattr(P, "_measured_grid_peak_mib", _boom)
    monkeypatch.setattr(P, "_estimate_grid_peak_mib", lambda t: 4000.0)
    monkeypatch.setattr(P, "_gpu_free_mib",
                        lambda devices: {d: 23000 for d in devices})
    wpg, mode, mem, diag = P._choose_workers_per_gpu(
        [_mk_task() for _ in range(40)], [0, 1], {})
    assert wpg == 2                       # structural formula, run survives
    assert "probe raised ValueError" in diag


# ---------------------------------------------------------------------------
# _choose_workers_per_gpu: the contract that must not move
# ---------------------------------------------------------------------------

def test_chooser_override_env_bypasses_probes(monkeypatch):
    monkeypatch.setenv("TRACT7DT_GPU_WORKERS_PER_GPU", "3")

    def _boom(*a, **k):
        raise AssertionError("probe must not run on the override path")

    monkeypatch.setattr(P, "_measured_grid_peak_mib", _boom)
    monkeypatch.setattr(P, "_estimate_grid_peak_mib", _boom)
    monkeypatch.setattr(P, "_gpu_free_mib", _boom)
    wpg, mode, mem, diag = P._choose_workers_per_gpu(
        [_mk_task()], [0, 1], {})
    assert (wpg, mode) == (3, "bfc")
    assert mem == pytest.approx(0.9 / 3)
    assert diag == "override workers_per_gpu=3 (bfc)"


def test_chooser_override_one_is_platform(monkeypatch):
    monkeypatch.setenv("TRACT7DT_GPU_WORKERS_PER_GPU", "1")
    wpg, mode, mem, diag = P._choose_workers_per_gpu(
        [_mk_task()], [0], {"mem_fraction": 0.5})
    assert (wpg, mode, mem) == (1, "platform", 0.5)


def _measured_info(resident=800.0, kernel=100.0):
    return {"peak_mib": resident + kernel, "resident_mib": resident,
            "kernel_mib": kernel,
            "arena_mib": resident + (P._GRID_SOLVE_WINDOW + 1) * kernel}


def _patch_measured(monkeypatch, resident=800.0, kernel=100.0, ovh=300.0,
                    free=20000, total=24576):
    monkeypatch.delenv("TRACT7DT_GPU_WORKERS_PER_GPU", raising=False)
    monkeypatch.setattr(
        P, "_measured_grid_peak_mib",
        lambda tasks, cfg: (_measured_info(resident, kernel), "probe-diag"))
    monkeypatch.setattr(P, "_parent_cuda_overhead_mib", lambda: ovh)
    monkeypatch.setattr(P, "_gpu_free_total_mib",
                        lambda devices: {d: (free, total) for d in devices})
    monkeypatch.setattr(P, "_gpu_free_mib",
                        lambda devices: {d: free for d in devices})


def test_chooser_measured_formula(monkeypatch):
    # arena = 800 + 9*100 = 1700; eff = 1700 + 300 = 2000
    # int(0.9 * 20000 / 2000) = 9 -> MAX_WPG clamps to 8
    _patch_measured(monkeypatch)
    tasks = [_mk_task() for _ in range(40)]
    wpg, mode, mem, diag = P._choose_workers_per_gpu(tasks, [0, 1], {})
    assert wpg == 8 and mode == "bfc"
    # need-sized arena fraction, NOT the equal split (0.9/8 = 0.1125)
    assert mem == pytest.approx(1700.0 / 24576)
    assert mem < 0.9 / 8
    assert "adaptive packing:" in diag
    assert "measured grid peak ~900MiB" in diag
    assert "arena ~1700MiB" in diag and "overhead ~300MiB" in diag
    assert "probe-diag" in diag
    assert "-> 8 worker(s)/GPU" in diag


def test_chooser_measured_overhead_in_denominator(monkeypatch):
    # Same arena, huge overhead -> wpg collapses: eff = 1700 + 4300 = 6000
    # int(0.9 * 20000 / 6000) = 3
    _patch_measured(monkeypatch, ovh=4300.0)
    tasks = [_mk_task() for _ in range(40)]
    wpg, mode, mem, diag = P._choose_workers_per_gpu(tasks, [0, 1], {})
    assert wpg == 3


def test_chooser_measured_guarantee_min_binds(monkeypatch):
    # Arena bigger than the equal split: mem must clamp to SAFETY/wpg so the
    # collective cap (sum <= 0.9 of the card) survives.
    # arena = 8000 + 9*100 = 8900; eff = 9200 -> wpg = int(18000/9200) = 1?
    # use bigger free so wpg = 2: 0.9*22000/9200 = 2.15 -> 2
    _patch_measured(monkeypatch, resident=8000.0, free=22000)
    tasks = [_mk_task() for _ in range(40)]
    wpg, mode, mem, diag = P._choose_workers_per_gpu(tasks, [0, 1], {})
    assert wpg == 2
    assert mem == pytest.approx(min(8900.0 / 24576, 0.9 / 2))


def test_chooser_overhead_unmeasurable_falls_back(monkeypatch):
    _patch_measured(monkeypatch)
    monkeypatch.setattr(P, "_parent_cuda_overhead_mib", lambda: None)
    monkeypatch.setattr(P, "_estimate_grid_peak_mib", lambda t: 4000.0)
    tasks = [_mk_task() for _ in range(40)]
    wpg, mode, mem, diag = P._choose_workers_per_gpu(tasks, [0, 1], {})
    # structural fallback: int(0.9*20000/(2.5*4000)) = 1 -> platform
    assert wpg == 1 and mode == "platform"
    assert "structural model:" in diag and "unmeasurable" in diag


def test_chooser_card_total_unmeasurable_falls_back(monkeypatch):
    _patch_measured(monkeypatch)
    monkeypatch.setattr(P, "_gpu_free_total_mib", lambda devices: {})
    monkeypatch.setattr(P, "_estimate_grid_peak_mib", lambda t: 1000.0)
    tasks = [_mk_task() for _ in range(40)]
    wpg, mode, mem, diag = P._choose_workers_per_gpu(tasks, [0, 1], {})
    assert "structural model:" in diag
    assert mem == pytest.approx(0.9 / wpg) if wpg > 1 else True


def test_chooser_structural_fallback_keeps_headroom(monkeypatch):
    monkeypatch.delenv("TRACT7DT_GPU_WORKERS_PER_GPU", raising=False)
    monkeypatch.setattr(P, "_measured_grid_peak_mib",
                        lambda tasks, cfg: (None, "why-not"))
    monkeypatch.setattr(P, "_estimate_grid_peak_mib", lambda t: 4000.0)
    monkeypatch.setattr(P, "_gpu_free_mib",
                        lambda devices: {d: 23000 for d in devices})
    tasks = [_mk_task() for _ in range(40)]
    wpg, mode, mem, diag = P._choose_workers_per_gpu(tasks, [0, 1], {})
    # int(0.9 * 23000 / (2.5 * 4000)) = 2  (the historical formula)
    assert wpg == 2
    assert mem == pytest.approx(0.9 / 2)     # legacy equal split preserved
    assert "x2.5 headroom" in diag and "structural model: why-not" in diag
    assert "est max grid peak ~4000MiB" in diag


def test_chooser_fallback_keeps_legacy_clip(monkeypatch):
    # Structural-fallback branch must be behavior-identical to pre-G1: the
    # ceil(tasks/devices) slots cap applies ONLY to the measured branch.
    # 4 tasks / 4 devices with a light structural peak: legacy formula gives
    # wpg = min(8, mem-derived, len(tasks)=4) -> 4 (bfc), NOT ceil(4/4)=1.
    monkeypatch.delenv("TRACT7DT_GPU_WORKERS_PER_GPU", raising=False)
    monkeypatch.setattr(P, "_measured_grid_peak_mib",
                        lambda tasks, cfg: (None, "why-not"))
    monkeypatch.setattr(P, "_estimate_grid_peak_mib", lambda t: 1000.0)
    monkeypatch.setattr(P, "_gpu_free_mib",
                        lambda devices: {d: 23000 for d in devices})
    tasks = [_mk_task() for _ in range(4)]
    wpg, mode, mem, diag = P._choose_workers_per_gpu(
        tasks, [0, 1, 2, 3], {"mem_fraction": 0.5})
    # int(0.9*23000/(2.5*1000)) = 8 -> min(8, 8, 4) = 4 (the legacy clip)
    assert (wpg, mode) == (4, "bfc")
    assert mem == pytest.approx(0.9 / 4)


def test_physical_gpu_index_mapping(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert P._physical_gpu_index(2) == 2          # no restriction: identity
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2,3")
    assert P._physical_gpu_index(0) == 1          # jax id 0 -> physical 1
    assert P._physical_gpu_index(2) == 3
    assert P._physical_gpu_index(7) == 7          # out of range: passthrough


def test_chooser_slots_capped_at_grid_count(monkeypatch):
    _patch_measured(monkeypatch, resident=80.0, kernel=2.0, ovh=100.0,
                    free=23000)
    # 6 grids / 4 devices -> ceil = 2 workers/GPU (8 slots), NOT 6
    tasks = [_mk_task() for _ in range(6)]
    wpg, mode, mem, diag = P._choose_workers_per_gpu(
        tasks, [0, 1, 2, 3], {})
    assert wpg == 2
    # 4 grids / 4 devices -> 1 worker/GPU, solo semantics (platform)
    wpg1, mode1, mem1, _ = P._choose_workers_per_gpu(
        tasks[:4], [0, 1, 2, 3], {"mem_fraction": 0.5})
    assert (wpg1, mode1, mem1) == (1, "platform", 0.5)


def test_chooser_free_probe_failure_uses_12000(monkeypatch):
    _patch_measured(monkeypatch, resident=300.0, kernel=100.0, ovh=0.0)
    monkeypatch.setattr(P, "_gpu_free_mib", lambda devices: {})
    tasks = [_mk_task() for _ in range(40)]
    wpg, _, _, diag = P._choose_workers_per_gpu(tasks, [0], {})
    # arena = 300+9*100 = 1200; int(0.9 * 12000 / 1200) = 9 -> MAX_WPG 8
    assert wpg == 8 and "free~12000MiB" in diag


def test_chooser_no_tasks():
    assert P._choose_workers_per_gpu([], [0], {"mem_fraction": 0.5}) == (
        1, "platform", 0.5, "no grids")


# ---------------------------------------------------------------------------
# the shared persistent-cache helper (one home for both call sites)
# ---------------------------------------------------------------------------

def test_cache_helper_disabled_values():
    from tract7dt.fit.grid_gpu.runner import enable_persistent_jit_cache
    assert enable_persistent_jit_cache(None) is None
    assert enable_persistent_jit_cache("none") is None


def test_cache_helper_explicit_dir(tmp_path):
    from tract7dt.fit.grid_gpu.runner import enable_persistent_jit_cache
    got = enable_persistent_jit_cache(str(tmp_path / "jitcache"))
    assert got == str(tmp_path / "jitcache")
    import jax
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.0
