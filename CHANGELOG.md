# Changelog

Changes that matter to users of uniphot, newest first.

## 0.1.0 (2026-10-15)

First public release.

- **Origin.** uniphot comes from the GPU photometry engine of tract7dt, the photometry pipeline written for
  7DT data. This release is based on that engine's development version `dev8`, which was never released,
  plus the changes listed below. It is not the code of the last public tract7dt release (0.7.0), which ran
  on the CPU through the Tractor library.
- **No Tractor library.** The CPU path that called the Tractor library has been removed; photometry runs only
  on NVIDIA GPUs, with JAX. The Tractor library is not needed to install or run uniphot. Four pieces derived
  from The Tractor remain; they are listed in `NOTICE` and in the README.
- **Names.** The distribution and the command are called `uniphot`; the `tract7dt` command still works as
  another name. The Python package is still imported as `tract7dt`. Version numbers start again at 0.1.0.
- **License.** GPL-2.0-only.
- **Changes from `dev8` that change the results.** The photometry of 0.1.0 is not the same as that of `dev8`:
  the two PSF changes below change it even with the same inputs and settings.
  - PSF stars are chosen from the input list: among rows with a Gaia identity (`GAIA_SOURCE_ID`) that pass
    `RUWE` ≤ `zp.gaia_max_ruwe` (and `NON_SINGLE_STAR` = 0 when that column exists); a list with Gaia
    identities but no `RUWE` column stops the run. A list without Gaia identities falls back to the
    `TYPE` = `STAR` rows, with a warning in `run.log`. A star is not used when its cutout contains a saturated
    or bad pixel, or when the peak of a brighter detection lies inside its PSF box (55 pixels by default). The
    ellipticity cut and the minimum-separation rule between PSF stars (`min_sep`, 30 pixels in `dev8`) are
    removed.
  - The cutouts of the PSF stars are aligned on their centres with a Fourier sub-pixel shift. `dev8` used
    Lanczos-3 interpolation, which made the mean aligned star sharper, by 0.23% at its peak (for stars with a
    FWHM of 3.7 to 4.5 pixels).
  - The default `crop.margin` is 100 pixels (it was 500).
- **Changes from `dev8` that affect configuration files and input lists:**
  - Removed: the CPU per-patch path and its settings, and the two-pass "residual phantom" mode
    (`bright_mask.residual_phantom_pass.*`, together with `bright_mask.extra_unmatched_csv`,
    `compute.gpu.background.dump_diagnostics` and `zp.calibrate`), and two PSF-star settings,
    `compute.gpu.psf.from_stars.max_ellipticity` and `compute.gpu.psf.from_stars.min_sep`. A configuration file
    that sets a removed key to the value it had in `dev8` still loads, with a note in the log; a value that would
    turn on removed code stops the run when the file is loaded. The exception is
    `compute.gpu.psf.from_stars.min_sep`: only the value 0 (no rule) loads, so a file that sets the `dev8`
    value 30 stops the run; delete that key.
  - The new input columns `GAIA_SOURCE_ID`, `RUWE` and `NON_SINGLE_STAR` choose the PSF stars (see above).
  - The package also contains code for a new PSF model. The photometry in this version does not use it.
