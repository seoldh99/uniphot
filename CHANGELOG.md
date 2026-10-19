# Changelog

Changes that matter to users of uniphot, newest first.

## 0.1.1 (2026-10-19)

Bug-fix release: the bright-object mask no longer masks listed sources by their own light. Apart from that and
one retired setting, the code did not change; the fitting, error, flag and zero-point code is the same as in
0.1.0.

- **Fixed: listed sources masked by their own light.** In 0.1.0, bright objects not in the list were detected on
  the sum of all bands without subtracting the sky and with the header noise (`SKYSIG`), and a detection counted
  as a listed source only if its centre lay within 3 pixels of the row. In the sample run, 99 of the 751 listed
  sources (13%) were masked and not fitted, stars and galaxies alike (59 of 463 `STAR` rows, 40 of 288 `EXP`
  rows). In all 99 the masking detection was the source's own light (in 4 of them merged with nearby sources). In
  92 of them the light is at the listed position, but the detected area spread into the unsubtracted sky, so that
  its centre fell 3.0 to 8.5 pixels away; in the other 7, all stars, the star's light itself lies 3.3 to 10.6
  pixels from its listed position. In addition, 12 fitted listed sources had more than 20% of their model flux on masked pixels, and 11
  of them were marked reliable.
- **What changed in the bright-object mask:**
  - Before detection, a local sky (SEP background: 64-pixel mesh, median filter of 3 meshes) is subtracted from a
    copy of the sum of bands, and the noise is scaled by the scatter measured around each pixel on the same mesh.
    Pixels where the measured scatter is zero or negative (the interpolated scatter can dip below zero next to a
    sharp step in the noise, for example at the edge of a coadd) get no weight in the detection. The images used
    in the fit are not changed.
  - A detection is the light of a listed source when its peak (SEP `xcpeak`, `ycpeak`) lies within
    r = min(`compute.gpu.positions.pos_max_shift_pix`, max(FWHM, 3.72 × scatter)) of a row. The FWHM is that of
    the run's PSF (middle band); the scatter is that between the rows' positions and the detected peaks, measured
    in the run from pairs with S/N ≥ 20 (at least 30 pairs, otherwise the FWHM alone is used), separately for the
    listed rows and for the added Gaia rows. The values are written to `run.log` and to the new file
    `bright_mask/association.json`.
  - Retired setting `bright_mask.match_radius_pix`: the 0.1.0 default 3.0 still loads, with a note in the log;
    any other value stops the run when the file is loaded. Delete the key.
  - A SEP detection cache from a 0.1.0 work folder (`bright_mask/sep_objs.npz`) is not reused.
- **Results that differ from 0.1.0** (same inputs and settings; only through the mask, the bright objects not in
  the list, and the rows flagged as masked):
  - Sample data: listed sources inside the mask 99 → 4 (all stars whose light peaks farther from the listed
    position than the radius, 3.84 pixels in that run); fitted listed sources 652 → 747; reliable 617 → 710 (92
    of the 95 newly fitted, and a net gain of 1 among the 652 fitted in both versions); mask area 1.99% → 1.02%
    of the working frame (the 10,000 × 6,600 pixels left after the 100-pixel border is cropped); fitted listed
    sources with more than 20% of their model flux on masked pixels 12 → 0. Zero points at the rows' positions
    changed by at most 0.78 mmag. In one of the four image tiles of the sample run, the blank-sky level that is
    subtracted from star fluxes (`blank_response_bhat.csv`) rose by 0.22, 0.98, 0.77 and 0.82 times the median
    statistical flux error of the stars there in m400, m500, m600 and m700.
  - Crowded 7DT test field, with the default `bright_mask.subtract_unmatched: true`: listed sources inside the
    mask 148 → 81 (42 of the 67 newly fitted are reliable); masked pixels 49,684 → 35,819; bright objects not in
    the list subtracted as point sources 6,074 → 653. The 3,338 listed stars that had such a point source within
    one PSF width (FWHM) in 0.1.0 and none in 0.1.1 came out brighter by 0.024 mag (median over the stars of
    each star's median over bands; 95% interval 0.021 to 0.026 mag); 10% of them brightened by more than
    0.17 mag, and 10% faded by more than 0.03 mag. The measured scatter of the sum of bands there is about 2.25
    times the header noise (median); we think this is mainly the light of unresolved stars, but have not tested
    it.
  - Sparse 7DT test field: mask unchanged (6,965 pixels); bright objects not in the list subtracted as point
    sources 5,257 → 4,726. Stars more than 40 pixels from every changed point source moved by at most 0.07 times
    their flux error. But 5 stars 0.5 to 5.4 pixels from a changed point source moved by 6 to 143 times their
    flux error in some band (all 5 marked reliable in both versions). Three of them changed by less than
    0.015 mag (median over bands). The other two lie at the edge of the data, and only two of their bands have
    no flag other than bit 4: in those bands, one became 0.66 mag fainter in one band while its flux in the other
    fell from 140 times its error to below zero, and the other became 0.30 and 0.37 mag brighter.
- **Known problems** (five checks that 0.1.1 failed against limits set before the runs, showing four problems;
  problem 4 failed two checks; details in the README under "Known limitations"). In the two test fields, running
  0.1.0 twice on the same input changed no flag and moved the compared fluxes by at most 0.0006 times their flux
  error, so these are responses to the changed detection of bright objects not in the list (the mask and the
  point sources subtracted; in the sparse field only the subtracted point sources changed), not run-to-run noise:
  1. Two list rows on one object (sample list: 8 pairs, 16 rows, also in 0.1.0) can meet at nearly one point with
     an arbitrary split of the flux; 2 pairs (4 rows) are marked reliable with magnitudes off by 0.15 to 2.4 mag
     in both directions, while their reported errors are 0.03 to 0.14 mag.
  2. Sample data: the 19 galaxies checked in one image tile came out about 1% larger and 0.8 to 1.0% brighter
     (medians, m500 to m700); in m500 the median change is 0.17 times their flux error (expected 0.05, limit 0.15).
  3. Sparse field: 23 of the 5,550 compared stars (0.41%), all of them isolated, switched `fit_is_reliable`
     (reason `island_stalled`), while their fluxes moved by at most 0.017 times their flux error.
  4. Crowded field: stars at the edge of the data moved by up to 70 times the 0.1.0 flux error in band values
     with no flag other than bit 4 (rows marked reliable) and up to 267 times in band values with bit 16; in 10
     band values bit 1 (no data) was set in 0.1.0 but not in 0.1.1 (7 of them on pixels without data, with
     meaningless values as large as −2.4 × 10⁷ and bit 16).

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
