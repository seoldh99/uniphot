"""Wherever the engine forces float64, it must also refuse the TF32 demotion.

XLA silently lowers every float32 dot/einsum/matmul to TF32 -- 10 mantissa bits
instead of 23 -- unless ``JAX_DEFAULT_MATMUL_PRECISION`` says otherwise. That is
not confined to internals: the shipped JIT cache carries
``mma.sync...f32.tf32.tf32.f32`` PTX inside ``jit_final_fluxes``,
``jit_final_errors`` and ``jit_source_metrics`` -- the kernels that produce the
REPORTED flux, its error, and the reliability metrics
(development ledger SS23-32).

Measured on the sparse anchor T08676 (LEDGER SS24 / effarc2/E10_RESULTS.md):
turning it off removed 1,884 TF32 MMA instructions (to zero), cost no wall time,
moved the reported flux by 0.040 mmag (median) / 0.51 mmag (p99), and took the
cold-run reproducibility from 48.8% bit-identical / 13.65 mmag max / 8
reliability flips to 90.6% / 0.0004 mmag / 0 flips.

The invariant pinned here is deliberately structural rather than behavioural: a
process that needs float64 accumulation is exactly a process whose float32
products must not be truncated either, so the two settings belong at the same
sites. A future entry point that adds one and forgets the other is the failure
mode this test exists to catch -- it cannot be caught by any run-to-run gate,
because the damage is DETERMINISTIC.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


PKG = Path(__file__).resolve().parents[1]

# Source files that set JAX_ENABLE_X64 for a process that will import JAX.
X64_SITES = [
    "cli.py",                              # the parent (runs science JAX inline)
    "fit/grid_gpu/runner.py",              # grid + island shard workers
    "fit/grid_gpu/psf_pipeline_stage.py",  # PSF workers (opt-in epsfbuilder path)
]

_PREC = "JAX_DEFAULT_MATMUL_PRECISION"


def test_every_x64_site_also_sets_full_matmul_precision():
    missing = []
    for rel in X64_SITES:
        src = (PKG / rel).read_text()
        if _PREC not in src:
            missing.append(rel)
    assert not missing, (
        "these entry points configure JAX but do not refuse the TF32 demotion: "
        + ", ".join(missing)
    )


def test_precision_is_highest_and_overridable():
    """`setdefault`, not a hard assignment.

    An explicit operator value has to win: that is how the A/B that justified
    this default was run (development A/B arms e9a vs e10h), and a datacenter GPU
    where TF32 genuinely is faster than FP32 must be able to opt back in.
    On this hardware (GA102) dense TF32 throughput EQUALS FP32 throughput, so
    the demotion buys nothing at all.
    """
    for rel in X64_SITES:
        src = (PKG / rel).read_text()
        for m in re.finditer(r'setdefault\(\s*"%s"\s*,\s*"([a-z0-9]+)"' % _PREC, src):
            assert m.group(1) == "highest", (rel, m.group(1))
        # no hard assignment anywhere -- that would break the override
        assert not re.search(r'\[\s*"%s"\s*\]\s*=' % _PREC, src), rel
        assert _PREC in src and "setdefault" in src, rel


def test_no_stale_override_in_the_test_environment():
    """Guard against a developer shell that pins the variable and therefore
    silently makes every other test in this file meaningless."""
    v = os.environ.get(_PREC)
    assert v in (None, "highest"), (
        f"{_PREC}={v!r} in the environment: unset it before trusting a run")
