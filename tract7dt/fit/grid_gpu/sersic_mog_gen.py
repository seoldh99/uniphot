"""Offline generator for the dense Sersic mixture-of-Gaussians table.

Evaluates The Tractor's ``SersicMixture`` (Hogg & Lang 2013-style
per-range spline fits, SDSS-truncated profiles) on a dense Sersic-index grid
and freezes the result into ``tract7dt/data/sersic_mog_table.npz`` for the
JAX runtime (:mod:`tract7dt.fit.grid_gpu.sersic_mog`).

Run once, with The Tractor installed (it is imported only here; re-run only if Tractor's tables ever change)::

    python -m tract7dt.fit.grid_gpu.sersic_mog_gen

Table layout (all float64 in the npz; the runtime downcasts as it likes):

- ``n_knots``   (N,)        Sersic-index knots. Piecewise-dense inside each of
                            the 8 vendor spline ranges; every internal range
                            boundary appears TWICE (last knot of the left
                            segment and first knot of the right segment carry
                            the same n but that n's mixture evaluated from each
                            side). The mixture itself is continuous across
                            boundaries (new components enter at amplitude 0),
                            but component *slots* are only meaningful within a
                            segment, so runtime interpolation must never span a
                            duplicated knot pair. ``searchsorted(..., 'right')``
                            on a strictly-increasing query lands in the correct
                            segment automatically.
- ``amps``      (N, K_MAX)  Gaussian amplitudes, zero-padded. Include the
                            (1 - core) scaling already applied by Tractor;
                            sum(amps[i]) + core_amp[i] == 1 exactly.
- ``log_vars``  (N, K_MAX)  log variance (units of re**2) per component.
                            Padded slots repeat the segment's first real row so
                            interpolation against a zero-amp slot stays finite.
- ``core_amp``  (N,)        delta-core amplitude (0 for n <= 1); rendered as
                            ``core_amp * PSF`` by the runtime.
- ``k_gauss``   (N,)        number of live Gaussian slots per row.
- ``seg_lo``    (8,)        segment lower n bounds (diagnostic).
- ``n_min, n_max``          scalar hard bounds (0.29, 6.3).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

K_MAX = 15
DN_TARGET = 0.005
# Below this n the Tractor SDSS-truncated fits are used (the profile
# dies within ~2-3 Re there, so truncated == untruncated to <0.5%); at and
# above it the Shajib (2019) UNTRUNCATED decomposition is used, giving total
# fluxes in the ideal-Sersic definition.
SPLICE_N = 0.80

_PKG_ROOT = Path(__file__).resolve().parents[2]   # .../tract7dt/tract7dt
OUT_PATH = _PKG_ROOT / "data" / "sersic_mog_table.npz"


# ---------------------------------------------------------------------------
# Shajib (2019) untruncated Sersic -> MoG (port of pysersic rendering.py)
# ---------------------------------------------------------------------------

def _calculate_etas_betas(P: int = 10):
    from scipy.special import comb
    kes = np.arange(2 * P + 1)
    betas = np.sqrt(2 * P * np.log(10) / 3.0 + 2.0j * np.pi * kes)
    eps = np.zeros(2 * P + 1)
    eps[0] = 0.5
    eps[1:P + 1] = 1.0
    eps[-1] = 1 / 2.0 ** P
    for k in range(1, P):
        eps[2 * P - k] = eps[2 * P - k + 1] + 1 / 2.0 ** P * comb(P, k)
    etas = (-1.0) ** kes * eps * 10.0 ** (P / 3.0) * 2.0 * np.sqrt(2 * np.pi)
    return etas, betas


def _bn_pysersic(n):
    return (2 * n - 1 / 3 + 4 / (405 * n) + 46 / (25515 * n ** 2)) + (
        ((0.00018073182821237496 / (n + np.exp(n))) + 3.7026973571904255e-5)
        / (-0.09149183119702775 - n) ** 2
    ) * ((np.log(n) + (((n * 2.6248718397705195) + -0.9727511612512357)
                       / (n ** (-4) + 94.78011643586419)))
         / (n + -0.006044236674273689))


def _sersic1d(r, n):
    from scipy.special import gammaln
    b = _bn_pysersic(n)
    ie = 1.0 / (2 * np.pi * n * np.exp(b + gammaln(2 * n))) * b ** (2 * n)
    return ie * np.exp(-b * ((r) ** (1.0 / n) - 1.0))


def _s_hi_of(n: float) -> float:
    """Smooth widest-sigma schedule (units of Re): 2.5 at n=0.8 ramping
    linearly to 15 by n=2.0. Low n dies fast; wide sigmas only ring there."""
    return float(np.clip(2.5 + (15.0 - 2.5) * (n - 0.8) / 1.2, 2.5, 15.0))


def shajib_row(n: float, K: int = K_MAX, P: int = 10,
               s_lo: float = 1e-2):
    etas, betas = _ETAS_BETAS
    s_hi = _s_hi_of(n)
    sigmas = np.logspace(np.log10(s_lo), np.log10(s_hi), K)
    f_sig = np.sum(etas * _sersic1d(np.outer(sigmas, betas), n).real, axis=1)
    dls = np.abs(np.diff(np.log(sigmas)).mean())
    amps = f_sig * dls / np.sqrt(2 * np.pi)
    amps[0] *= 0.5
    amps[-1] *= 0.5
    amps = amps * 2 * np.pi * sigmas * sigmas
    return amps, sigmas ** 2


_ETAS_BETAS = _calculate_etas_betas(10)


def build_table():
    sys.path.insert(0, str(_PKG_ROOT / "_vendor"))
    from tractor.sersic import SersicMixture

    sm = SersicMixture()
    tractor_segments = [(lo, hi) for lo, hi, _a, _v in sm.fits]
    n_min = tractor_segments[0][0]
    n_max = tractor_segments[-1][1]
    # Tractor segments below the splice (the segment containing SPLICE_N is
    # cut there); one Shajib segment above.
    segments = []
    for lo, hi in tractor_segments:
        if hi <= SPLICE_N:
            segments.append((lo, hi, "tractor"))
        elif lo < SPLICE_N:
            segments.append((lo, SPLICE_N, "tractor"))
    segments.append((SPLICE_N, n_max, "shajib"))

    n_rows, amp_rows, lv_rows, core_rows, k_rows = [], [], [], [], []
    for si, (lo, hi, family) in enumerate(segments):
        # closed [lo, hi] per segment; interior knots on a uniform grid.
        n_steps = max(2, int(round((hi - lo) / DN_TARGET)) + 1)
        knots = np.linspace(lo, hi, n_steps)
        # Evaluate strictly inside the vendor's half-open [lo, hi) so both
        # duplicated boundary knots get their own segment's spline values.
        eps = 1e-9
        for n in knots:
            n_eval = min(max(float(n), lo), hi - eps if si < len(segments) - 1 else hi)
            if family == "tractor":
                mix = sm._getProfile(n_eval)
                amps = np.asarray(mix.amp, dtype=np.float64)
                varr = np.asarray(mix.var, dtype=np.float64)[:, 0, 0]
                if varr[-1] == 0.0:       # delta core appended by Tractor
                    core = float(amps[-1])
                    amps, varr = amps[:-1], varr[:-1]
                else:
                    core = 0.0
            else:                          # shajib (untruncated)
                amps, varr = shajib_row(n_eval)
                core = 0.0
            k = len(amps)
            if k > K_MAX:
                raise RuntimeError(f"K={k} exceeds K_MAX at n={n_eval}")
            pad_a = np.zeros(K_MAX)
            pad_a[:k] = amps
            pad_lv = np.empty(K_MAX)
            pad_lv[:k] = np.log(varr)
            pad_lv[k:] = pad_lv[k - 1]    # placeholder value for zero-amp slots
            n_rows.append(float(n))
            amp_rows.append(pad_a)
            lv_rows.append(pad_lv)
            core_rows.append(core)
            k_rows.append(k)

    n_knots = np.array(n_rows)
    amps = np.array(amp_rows)
    log_vars = np.array(lv_rows)
    core_amp = np.array(core_rows)
    k_gauss = np.array(k_rows, dtype=np.int32)

    # Sanity: Tractor rows sum to exactly 1 (their own normalization); Shajib
    # rows sum to <= 1 by design -- the deficit is far-wing flux beyond the
    # widest sigma. Do NOT renormalize (in-window SB is what LSQ fitting sees;
    # the fitted amplitude then estimates the untruncated total directly).
    tot = amps.sum(axis=1) + core_amp
    if np.any(tot > 1.01) or np.any(tot < 0.85):
        raise RuntimeError(f"suspicious totals: [{tot.min():.4f}, {tot.max():.4f}]")
    return dict(n_knots=n_knots, amps=amps, log_vars=log_vars, core_amp=core_amp,
                k_gauss=k_gauss, seg_lo=np.array([lo for lo, _hi, _f in segments]),
                n_min=np.float64(n_min), n_max=np.float64(n_max),
                splice_n=np.float64(SPLICE_N),
                family=np.array([f for _lo, _hi, f in segments], dtype="U8"))


def main():
    table = build_table()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, **table)
    n = table["n_knots"]
    print(f"wrote {OUT_PATH}  rows={len(n)}  n=[{n.min():.2f},{n.max():.2f}]  "
          f"K_MAX={K_MAX}  size={OUT_PATH.stat().st_size/1024:.0f} KiB")


if __name__ == "__main__":
    main()
