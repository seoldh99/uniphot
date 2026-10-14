"""Statistical self-checks of the generator, shared by the tests and the false-failure device.

Each check builds synthetic data with the generator, measures statistics and compares them
with expectations derived here from the design's formulas (not from the generator's helpers),
with limits fixed in advance.  ``test_synth_generator.py`` runs each once with a fixed seed;
``falsefail.measure`` runs each many times with independent seeds to record how often a
correct generator would fail it (design 6 sample-size rule: false failure < 1%).

Expectations written out here (independent re-derivations):
- Sky kernel: k = [a, 1, a] with 2a / (1 + 2a^2) = rho1; 1-D correlation r(0) = 1,
  r(1) = rho1, r(2) = a^2 / (1 + 2a^2); 2-D correlation rho(lx, ly) = r(lx) r(ly).
- Poisson: Var(Poisson(g m) / g) = m / g; excess kurtosis 1 / (g m).
- E1 (design 6, ideal PSF fit, white sky): Var(F_hat) = sigma^2 / S2 + F S3 / (g S2^2) with
  S2 = sum(phi^2), S3 = sum(phi^3) of the stamp the generator renders.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import astrometry, ideal, noise, psf, scene, xp


@dataclass
class CheckResult:
    """stats: name -> (value, lo, hi).  Passed when every value is finite and in [lo, hi]."""
    name: str
    stats: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(np.isfinite(v) and lo <= v <= hi for v, lo, hi in self.stats.values())

    def message(self) -> str:
        bad = [f"{k} = {v!r} not in [{lo!r}, {hi!r}]" for k, (v, lo, hi) in self.stats.items()
               if not (np.isfinite(v) and lo <= v <= hi)]
        return f"{self.name}: " + ("; ".join(bad) if bad else "ok")


def kernel_a(rho1: float) -> float:
    """Tap a of k = [a, 1, a] giving lag-1 correlation rho1 (root of 2 rho a^2 - 2a + rho = 0)."""
    if rho1 == 0.0:
        return 0.0
    return (1.0 - np.sqrt(1.0 - 2.0 * rho1 ** 2)) / (2.0 * rho1)


def corr_1d(lag: int, rho1: float) -> float:
    a = kernel_a(rho1)
    r = {0: 1.0, 1: 2.0 * a / (1.0 + 2.0 * a * a), 2: a * a / (1.0 + 2.0 * a * a)}
    return r.get(abs(int(lag)), 0.0)


def corr_2d(lx: int, ly: int, rho1: float) -> float:
    return corr_1d(lx, rho1) * corr_1d(ly, rho1)


def _lag_corr(d: np.ndarray, lx: int, ly: int) -> float:
    d = d - d.mean()
    h, w = d.shape
    a = d[ly:, lx:]
    b = d[:h - ly, :w - lx]
    return float((a * b).mean() / (d * d).mean())


def check_sky_noise(rng: np.random.Generator, shape=(1024, 1024), rho1: float = noise.RHO1,
                    sigma: float = 1.0, nsig: float = 4.0) -> CheckResult:
    """Per-pixel variance, mean and lag correlations of one sky image (and of a white one)."""
    res = CheckResult("sky_noise")
    for tag, r1 in (("corr", rho1), ("white", 0.0)):
        d = noise.sky_noise(shape, rng, sigma=sigma, rho1=r1)
        n = d.size
        s1 = sum(corr_1d(l, r1) for l in (-2, -1, 0, 1, 2)) ** 2          # sum of rho over lags
        s2 = sum(corr_1d(l, r1) ** 2 for l in (-2, -1, 0, 1, 2)) ** 2     # sum of rho^2 over lags
        se_mean = sigma * np.sqrt(s1 / n)
        se_var = sigma ** 2 * np.sqrt(2.0 * s2 / n)
        se_rho = np.sqrt(s2 / n)
        res.stats[f"{tag}_mean_z"] = (float(d.mean()) / se_mean, -nsig, nsig)
        res.stats[f"{tag}_var_z"] = ((float(d.var()) - sigma ** 2) / se_var, -nsig, nsig)
        for lx, ly in ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (3, 0)):
            z = (_lag_corr(d, lx, ly) - corr_2d(lx, ly, r1)) / se_rho
            res.stats[f"{tag}_rho{lx}{ly}_z"] = (z, -nsig, nsig)
    return res


def check_poisson(rng: np.random.Generator, shape=(512, 512), gain: float = 50.0,
                  nsig: float = 4.0) -> CheckResult:
    """Standardised source-Poisson residuals: mean 0 and variance 1; also with sky added."""
    res = CheckResult("poisson")
    m = rng.uniform(0.5, 50.0, size=shape)
    n = m.size
    z = (noise.poisson_counts(m, gain, rng) - m) / np.sqrt(m / gain)
    kurt = float(np.mean(1.0 / (gain * m)))
    res.stats["poisson_mean_z"] = (float(z.mean()) * np.sqrt(n), -nsig, nsig)
    res.stats["poisson_var_z"] = ((float(z.var()) - 1.0) / np.sqrt((2.0 + kurt) / n), -nsig, nsig)
    d = noise.realize(m, rng, sigma=1.0, gain=gain, rho1=noise.RHO1)
    z2 = (d - m) / np.sqrt(m / gain + 1.0)
    s1 = sum(corr_1d(l, noise.RHO1) for l in (-2, -1, 0, 1, 2)) ** 2
    s2 = sum(corr_1d(l, noise.RHO1) ** 2 for l in (-2, -1, 0, 1, 2)) ** 2
    # sky share of each pixel's variance is 1 / (m/g + 1) >= 0.5; correlation only raises the SE,
    # so the bound below uses the all-sky (largest) value.
    res.stats["realize_mean_z"] = (float(z2.mean()) / np.sqrt(s1 / n), -nsig, nsig)
    res.stats["realize_var_z"] = ((float(z2.var()) - 1.0) / np.sqrt((2.0 * s2 + kurt) / n),
                                  -nsig, nsig)
    return res


E1_CONFIGS = (
    # (profile, fwhm px, beta, sky-limited S/N): design 6 E1 isolated stars, S/N 300 to 3000
    ("gaussian", 4.0, 3.0, 300.0),
    ("moffat", 4.0, 3.0, 3000.0),
)


def check_e1(rng: np.random.Generator, n_real: int = 20_000, gain: float = 50.0,
             configs=E1_CONFIGS, offset=(0.21, -0.37), chunk: int = 2_000) -> CheckResult:
    """Design 6 E1 with the ideal fit: empirical Var(F_hat) / expected within +-3%, mean within 4 SE."""
    res = CheckResult("e1_ideal")
    size = psf.BOX
    for prof, fwhm, beta, snr in configs:
        field_ = psf.PSFField((size, size), profile=prof, beta=beta, fwhm=fwhm)
        st = field_.stamp(psf.HALF + offset[0], psf.HALF + offset[1])
        phi = st.data
        s2 = (phi ** 2).sum()          # numpy scalars: a zero stamp gives inf/nan, not an exception
        s3 = (phi ** 3).sum()
        flux = snr / np.sqrt(s2)          # white sky, sigma 1: sky-limited S/N = F sqrt(S2)
        expected = 1.0 / s2 + flux * s3 / (gain * s2 ** 2)
        model = np.zeros((size, size))
        model[:, :] = flux * phi
        est = []
        for k in range(0, n_real, chunk):
            stack = noise.realize(model, rng, n=min(chunk, n_real - k), sigma=1.0, gain=gain, rho1=0.0)
            est.append((stack * phi).sum(axis=(-2, -1)) / s2)
        est = np.concatenate(est)
        tag = f"{prof}{fwhm:g}_snr{snr:g}"
        res.stats[f"{tag}_var_ratio"] = (float(est.var(ddof=1)) / expected, 0.97, 1.03)
        res.stats[f"{tag}_mean_z"] = ((float(est.mean()) - flux) / np.sqrt(expected / n_real), -4.0, 4.0)
        fit = ideal.psf_flux(stack, phi, psf.HALF, psf.HALF)
        ref = (stack * phi).sum(axis=(-2, -1)) / s2
        res.stats[f"{tag}_ideal_fit_maxdiff"] = (float(np.max(np.abs(fit - ref))) / flux, 0.0, 1e-12)
    return res


def check_seeds(rng: np.random.Generator, n: int = 2000, shape=(3000, 4000)) -> CheckResult:
    """Seed pulls (seed - truth) / POS_ERR: RMS in [0.9, 1.1] and |z| > 3 share <= 1% (design rule)."""
    res = CheckResult("gaia_seeds")
    x, y = scene.draw_positions(shape, n, rng)
    s = astrometry.gaia_seeds(x, y, rng, pos_err_range=(0.002, 0.03))
    z = np.concatenate([(s.x - x) / s.pos_err, (s.y - y) / s.pos_err])
    res.stats["pull_rms"] = (float(np.sqrt(np.mean(z ** 2))), 0.9, 1.1)
    res.stats["frac_abs_gt3"] = (float(np.mean(np.abs(z) > 3.0)), 0.0, 0.01)
    return res


def check_xp(rng: np.random.Generator, n: int = 2000, shape=(3000, 4000),
             sigma_mag: float = 0.01) -> CheckResult:
    """XP residuals about ZP - 2.5 log10 F: RMS within 10% of sigma_mag, mean within 4 SE."""
    res = CheckResult("xp")
    x, y = scene.draw_positions(shape, n, rng)
    flux = 10 ** rng.uniform(2.0, 5.0, size=n)
    zp = xp.zp_z1(shape)
    out = xp.synthetic_xp(flux, x, y, zp, rng, sigma_mag=sigma_mag)
    xh = 2.0 * x / (shape[1] - 1) - 1.0
    yh = 2.0 * y / (shape[0] - 1) - 1.0
    truth = 25.0 + 0.05 * xh + 0.02 * yh ** 2 - 2.5 * np.log10(flux)
    r = out.mag - truth
    res.stats["rms_ratio"] = (float(np.sqrt(np.mean(r ** 2))) / sigma_mag, 0.9, 1.1)
    res.stats["mean_z"] = (float(r.mean()) / (sigma_mag / np.sqrt(n)), -4.0, 4.0)
    return res


def check_sky_weighted_sum(rng: np.random.Generator, n_real: int = 20_000,
                           rho1: float = noise.RHO1, chunk: int = 2_000) -> CheckResult:
    """Variance of sum(w * sky) for a PSF-shaped weight equals sum_lags rho(lag) sum_r w(r) w(r + lag)."""
    res = CheckResult("sky_weighted_sum")
    st = psf.PSFField((psf.BOX, psf.BOX), fwhm=4.0).stamp(psf.HALF + 0.3, psf.HALF - 0.1)
    w = st.data / float((st.data ** 2).sum())
    expected = 0.0
    h, wd = w.shape
    for lx in range(-2, 3):
        for ly in range(-2, 3):
            a = w[max(0, ly):h + min(0, ly), max(0, lx):wd + min(0, lx)]
            b = w[max(0, -ly):h + min(0, -ly), max(0, -lx):wd + min(0, -lx)]
            expected += corr_2d(lx, ly, rho1) * float((a * b).sum())
    vals = []
    for k in range(0, n_real, chunk):
        sky = noise.sky_noise((min(chunk, n_real - k), h, wd), rng, sigma=1.0, rho1=rho1)
        vals.append((sky * w).sum(axis=(-2, -1)))
    vals = np.concatenate(vals)
    se = np.sqrt(2.0 / n_real)
    res.stats["var_ratio_z"] = ((float(vals.var(ddof=1)) / expected - 1.0) / se, -4.0, 4.0)
    res.stats["generator_formula_rel"] = (
        abs(noise.linear_estimator_variance(w, sigma=1.0, rho1=rho1) / expected - 1.0), 0.0, 1e-12)
    return res


# Registry for the false-failure device: name -> (check, minimum repeats of its record).
# 300 repeats with 0 failures give a 95% upper bound of 0.99%; E1 (true rate about 0.55%) needs
# about 3,000 repeats for the bound to fall below 1% (falsefail.accepts).
REGISTRY = {
    "synth-sky-noise": (check_sky_noise, 300),
    "synth-poisson": (check_poisson, 300),
    "synth-sky-weighted-sum": (check_sky_weighted_sum, 300),
    "synth-gaia-seeds": (check_seeds, 1000),
    "synth-xp": (check_xp, 1000),
    "E1-ideal": (check_e1, 3000),
}
