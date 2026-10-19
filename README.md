# uniphot

uniphot measures the fluxes of a given list of sources in images of one field taken in several bands.
It fits point-source and galaxy models, convolved with a point-spread function (PSF) built from the
images themselves, to all bands at once on NVIDIA GPUs ("forced PSF photometry").

> **Status: 0.1.1, an early preview release. Not yet recommended for scientific use.** This is research
> software under active development. It has been run only on data from 7DT (the 7-Dimensional Telescope).
> Results, defaults, file formats and command names may change between versions. This version has known
> defects that affect the measured values; see [Known limitations](#known-limitations).

Contents: [What it does](#what-it-does) · [Requirements](#requirements) · [Installation](#installation) ·
[Quick start](#quick-start-with-the-sample-data) · [Inputs](#inputs) · [Outputs](#outputs) ·
[Known limitations](#known-limitations) · [License](#license) · [Citation](#citation) ·
[Contributing](#contributing) · [Contact](#contact)

## What it does

You give uniphot three things: a list of sources (positions and types), images of one field in several
bands that all share one pixel grid, and Gaia DR3 XP synthetic photometry of the stars in the field.
uniphot does not make the source list. It measures the sources you give it.

In one run, uniphot:

1. reads the source list and the images (one image per band; if a band has several, it uses the deepest);
2. builds a PSF model for each image from stars in that image (a polynomial in image position);
3. finds bright objects that are in the images but not in your list, and masks or subtracts them;
4. fits every listed source as a point source (`TYPE` = `STAR`) or a galaxy profile (`EXP`, `DEV`,
   `SERSIC`) convolved with the PSF. Neighbouring sources are fitted together. The position (and the
   galaxy shape) is the same in all bands; fluxes are fitted band by band, with a planar background for
   each tile of the image;
5. calibrates the zero point of each band against the Gaia XP synthetic photometry of stars in the
   field, letting it vary smoothly across the image;
6. writes a catalog with fluxes, magnitudes, errors and quality flags, and a status file that says
   whether the run succeeded.

## Requirements

- Linux on x86-64.
- An NVIDIA GPU with a driver that supports CUDA 12. Photometry runs only on the GPU; there is no CPU mode.
- conda (recommended) or pip, with Python 3.11.
- Internet access to install, to download the sample data, and during a run if your Gaia XP file has no
  `ruwe` and `phot_bp_rp_excess_factor` columns (see [Inputs](#gaia-xp-synthetic-photometry)).

So far uniphot has been tested only on one Linux server with NVIDIA GeForce RTX 3090 cards (24 GiB each,
driver 570.211.01) and 125 GiB of RAM. The minimum GPU memory and RAM have not been measured.

## Installation

Install uniphot into a new environment. Do not install it next to the older `tract7dt` package from PyPI:
both use the Python import name `tract7dt` and would overwrite each other's files.

### With conda (recommended)

```bash
git clone --branch v0.1.1 https://github.com/seoldh99/uniphot.git
cd uniphot
conda env create -f environment.yml
conda activate uniphot
uniphot doctor
```

`environment.yml` installs Python 3.11, JAX with CUDA 12 support and the other dependencies from
conda-forge, and then installs uniphot from the cloned folder (`pip install -e .`), so keep that folder.
The CUDA runtime libraries come with the conda packages; the host needs only the NVIDIA driver.
On our test server, creating the environment took 4 to 5 minutes and downloaded about 12 GB of packages.

### With pip

```bash
python3.11 -m venv uniphot-env
source uniphot-env/bin/activate
pip install --upgrade pip
pip install "jax[cuda12]"
pip install "git+https://github.com/seoldh99/uniphot.git@v0.1.1"
uniphot doctor
```

On our test server this installed JAX 0.10.2 with the CUDA 12 libraries from pip. There, `LD_LIBRARY_PATH`
pointed to an older CUDA toolkit (12.8) installed on the system. Its libraries were loaded instead of the ones
that pip installed, JAX could not use the GPU, and `uniphot doctor` failed ("Unable to load cuSPARSE", no CUDA
device). After `unset LD_LIBRARY_PATH` in that shell, `uniphot doctor` passed and the sample run below finished
with `"status": "ok"`. If `uniphot doctor` finds no CUDA device in a pip environment, check `LD_LIBRARY_PATH` first.

### Check the installation

`uniphot doctor` lists the Python packages, the JAX version and the CUDA devices that JAX can see.
Exit code 0 means the required checks passed. If no CUDA device is visible it prints `FAIL` and exits with 1.

**Names.** The command is `uniphot`; `tract7dt` still works as another name for the same command.
In this version the Python package is still imported as `tract7dt`.

### Run the tests (optional)

The tests are installed with the package. They need neither a GPU nor the sample data; the command below
hides the GPUs so that they run on the CPU:

```bash
pip install pytest pyflakes
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu python -m pytest \
    "$(python -c 'import os, tract7dt; print(os.path.dirname(tract7dt.__file__))')/tests"
```

Give the tests folder by its path as above: `pytest --pyargs tract7dt.tests` finds no tests.
Some tests check records of the development work that are not part of this repository; they are skipped,
each with a message that says so. A few tests are marked as expected failures (`xfail`), with the reason.
In a conda environment made from `environment.yml` on our test server, this command ran 828 tests in about
16 minutes on 4 CPU cores: 732 passed, 92 were skipped (91 that need the development records, and 1 end-to-end
solve that does not work on the CPU backend: a known defect of uniphot's solver there, where a column without
data spoils the whole band), and 4 were expected failures; none failed otherwise. Two of the expected failures depend on thread timing and may pass on
other machines (reported as `xpassed`).

## Quick start with the sample data

The sample data are four 7DT medium-band images (bands m400, m500, m600 and m700; 10200 × 6800 pixels each)
of part of the UDS field, a source list of 751 rows (463 `STAR`, 288 `EXP`) and a Gaia XP synthetic
photometry file. The download is 934 MB (1.1 GB unpacked).

**The sample data are not covered by uniphot's license.** They are © 2026 7DT Team and come with their own
terms (`LICENSE_SAMPLE_DATA.txt` in the download): use them for local test runs only. Do not redistribute
or publish them, use them to produce publications, or distribute products derived from them, without
written permission from the 7DT team (contact in `README_SAMPLE_DATA.txt`).

**1. Download**

```bash
mkdir uniphot-sample && cd uniphot-sample
uniphot download-sample
```

This downloads `tract7dt_sample_data.tar.gz` from release `v1.0.0` of
[seoldh99/tract7dt_sample_data](https://github.com/seoldh99/tract7dt_sample_data), checks its SHA-256
checksum, unpacks it into `./tract7dt_sample_data/`, deletes the archive, and writes
`tract7dt_sample_data/image_list.txt` with the full paths of the four images.
Use `--dir PATH` to choose another folder and `--force` to replace an existing one.
On our test server the download took about 3.5 minutes.

**2. Write a configuration file** `sample.yaml` in the same folder:

```yaml
inputs:
  tile_name: "UDS_sample"
  input_catalog: "tract7dt_sample_data/input_catalog_UDS.csv"
  image_list_file: "tract7dt_sample_data/image_list.txt"
  gaiaxp_synphot_csv: "tract7dt_sample_data/gaiaxp_dr3_synphot_UDS.csv"
outputs:
  work_dir: "sample_run"
bright_mask:
  subtract_unmatched: false   # deep extragalactic field: see "Known limitations"
```

Settings you leave out take their default values. Relative paths are read from the folder of the
configuration file. All outputs go to `work_dir`. To see every setting with its default value and a
comment, run `uniphot dump-config --out full_config.yaml`.
On a shared machine, choose the GPUs to use with:

```yaml
compute:
  gpu:
    devices: [0]   # default "auto": every GPU that JAX can see
```

**3. Run**

```bash
uniphot run --config sample.yaml
```

The sample's Gaia XP file has no `ruwe` column, so the run fetches two Gaia quality columns for the
calibration stars from the ESA Gaia archive (internet needed). To do this in advance, run
`uniphot augment-gaia tract7dt_sample_data/gaiaxp_dr3_synphot_UDS.csv`, which writes
`gaiaxp_dr3_synphot_UDS_q.csv` next to it, and use that file as `gaiaxp_synphot_csv`.

The run succeeded only if `sample_run/RUN_STATUS.json` says `"status": "ok"`. The catalog is
`sample_run/output_catalog.csv`.

On our test server, with the configuration above and `compute.gpu.devices: [1, 2]` (two RTX 3090 cards), a
sample run of this version finished with `"status": "ok"` in about 13 minutes; the step that fits groups of
neighbouring sources took about 10 of those minutes. The JAX compilation cache already held kernels from earlier
runs, so a first run may take longer. The run's memory use peaked at about 24 GiB of RAM (sampled every 0.5 s,
so the true peak can be higher). Of the 751 listed sources, 747 were fitted and 710 have `fit_is_reliable` =
`True`; the other 4 lie inside the bright-object mask and were not fitted (see
[Known limitations](#known-limitations)).

## Inputs

### Source list

`inputs.input_catalog`, a CSV file. Write the column names as shown.

| Column | | Meaning |
|---|---|---|
| `RA`, `DEC` | required | Position in degrees, in the frame of the images' WCS. |
| `ID` | recommended | Unique identifier, kept as text exactly as written. |
| `TYPE` | recommended | `STAR` (point source), or `EXP`, `DEV`, `SERSIC` (galaxy profiles). The galaxy models are not yet validated on real data. |
| `Re`, `ELL`, `THETA` | optional | Starting galaxy shape: half-light radius in pixels; ellipticity 1 − b/a; position angle in degrees (SExtractor convention, counter-clockwise from the image x axis). |
| `POS_ERR` | optional | Width in pixels of a Gaussian prior on the position. Without it the position is fitted freely, up to 5 pixels from the input. |
| `GAIA_SOURCE_ID`, `RUWE`, `NON_SINGLE_STAR` | optional | Gaia DR3 identity of stars. If any row has a `GAIA_SOURCE_ID`, `RUWE` is required, and PSF stars are chosen only among those rows with `RUWE` ≤ 1.4 (and `NON_SINGLE_STAR` = 0 if that column exists). Without `GAIA_SOURCE_ID`, any `TYPE` = `STAR` row can become a PSF star. |

Every input row appears in the output. Rows that cannot be fitted (outside the image, in the 100-pixel
border that is cropped by default, or without usable data) keep empty fit values and `fit_is_reliable` = `False`.
If the list has `FLUX_{band}` columns, their band names must match the images' `FILTER` values exactly,
or the run stops.

### Images

`inputs.image_list_file`, a text file with one FITS path per line. Blank lines and lines starting with `#`
are skipped; relative paths are read from the folder of the configuration file.

- All images must have the same size and the same WCS (one pixel grid). Resample them first if they do not.
- The image is read from the primary HDU. Pixels with no data must be NaN; exact zeros are treated as
  no data only in the padding around the observed area (connected to the image border).
- Required header keywords: `FILTER` (band name, used in the output column names), `ZP_AUTO` (AB zero point
  of the image), `SKYSIG` (sky noise per pixel, in image units), `EGAIN` (gain, e⁻/ADU).
  Without `SATURATE`, no pixel is treated as saturated.
- If a band has several images, only the deepest is used (header `UL5_5` if every image of the band has it,
  otherwise `ZP_AUTO − 2.5 log10(SKYSIG)`). `frames_used.csv` in the output records the choice.

### Gaia XP synthetic photometry

`inputs.gaiaxp_synphot_csv`, a CSV file with one row per Gaia DR3 star in the field. It is used to
calibrate the zero points.

- Required columns: `source_id`, `ra`, `dec`, `phot_g_mean_mag`, `bp_rp` (used by the default Gaia quality
  cut), and one column `mag_{FILTER}` per band with the synthetic magnitude in that band (for example
  `mag_m400`). Output magnitudes are on the system of these magnitudes (AB in our tests).
- Optional columns: `ruwe` and `phot_bp_rp_excess_factor` (fetched from the ESA Gaia archive during the run
  if missing; `uniphot augment-gaia <file>` adds them in advance), and `ra_error`, `dec_error`,
  `pmra_error`, `pmdec_error`.

uniphot does not include a tool to make this file. You can make it from the Gaia DR3 XP spectra and the
transmission curves of your filters, for example with [GaiaXPy](https://gaiaxpy.readthedocs.io/).
Gaia stars in this file that are not in your list are added to the fit as calibration rows; their IDs
start with `gaia_` and their `_is_injected_gaia` column is `True` (`zp.inject_gaia_sources: false` turns this off).

Without such a file you can set `zp.enabled: false`. Then no Gaia calibration is done, and magnitudes use
the header `ZP_AUTO` only. This setting was not part of the checks made for this release.

## Outputs

All files are written to `outputs.work_dir`.

- `RUN_STATUS.json`: `"status": "ok"` only if every step finished. Otherwise `"failed"`, with the step
  (`stage`) and the error. Neither the exit code alone nor the existence of `output_catalog.csv` means that
  the run succeeded. It also holds a short summary (rows fitted, reliable rows, calibrated bands).
- `output_catalog.csv`: the catalog (main columns below).
- `run.log`, `config_used_<time>.yaml`, `run_provenance.json` (package version and settings of the run),
  `frames_used.csv`, `ZP/` (zero-point tables and plots), `psf_polynomial/` (PSF models), `bright_mask/`,
  and diagnostic images.

Only one run per user can be active on a machine. A second `uniphot run` started meanwhile exits with
code 3 and changes nothing. If only the zero-point step fails (for example, the Gaia archive cannot be
reached), the catalog is written without magnitudes, the run is marked `failed`, and
`uniphot compute-zp --config <same file>` finishes it after you fix the cause.

### Main columns of `output_catalog.csv`

`{b}` is a band name (the `FILTER` value). Fluxes are in image units rescaled so that the header zero
point becomes 25, that is, on the header zero point, magnitude = 25 − 2.5 log10(flux).

| Column | Meaning |
|---|---|
| `ID`, `RA`, `DEC`, `TYPE` | From your list. |
| `RA_fit`, `DEC_fit` | Fitted position. |
| `Re_fit`, `ELL_fit`, `THETA_fit` | Fitted galaxy shape. |
| `FLUX_{b}_fit`, `FLUXERR_{b}_fit` | Flux and its statistical error. |
| `FLUXERR_MARG_TOT_{b}_fit` | The total flux error that the code recommends: statistical, confusion, source Poisson and (for galaxies) shape terms, multiplied by √max(1, rchisq), where `rchisq` is the reduced χ² of the source's model. |
| `MAG_{b}_fit`, `MAGERR_{b}_fit`, `MAGERR_MARG_TOT_{b}_fit` | Magnitude with the Gaia-calibrated zero point, and its errors (they include the zero-point error). |
| `MAG_UL5_MARG_{b}_fit` | 5σ limiting magnitude at the source, from `FLUXERR_MARG_TOT_{b}_fit`. |
| `FLAGS_{b}_fit` | Band flags (sum of bits): 1 no data at the source; 2 the PSF model has a negative lobe at the source; 4 the zero point is outside the area supported by calibration stars (the band's single zero point is used); 8 a saturated pixel nearby (values left empty); 16 more than 10% of the PSF falls on pixels without data. |
| `fit_is_reliable`, `fit_quality_issues` | Whether the source's fit can be trusted, and the reasons if not (for example: not fitted, inside a bright-object mask, the fit stalled, in a group too large to fit, no usable band). |
| `n_bands_usable` | Number of usable bands. |

**A band value is usable when `FLAGS_{b}_fit` is 0 and `fit_is_reliable` is `True`.**

## Known limitations

- **Tested only on 7DT data.** uniphot has been run only on 7DT coadds. The header keyword names above are
  fixed. Several defaults are in pixels and suit 7DT images (about 0.5″ per pixel, PSF width of about 5
  pixels), for example the 55-pixel PSF stamps and the 100-pixel cropped border. Other data may need other
  settings; this has not been tested.
- **Galaxies not validated.** Fits of `EXP`, `DEV` and `SERSIC` rows have not been validated on real data.
- **One coadd per band.** uniphot fits one coadded image per band, not individual exposures, and uses one sky
  noise value per image (`SKYSIG`). Near the edges of an image's coverage, where the coadd is shallower, the
  errors are about 10% too small (measured with an earlier development version).
- **PSF model where PSF stars are few.** The PSF model is a polynomial fitted to stars across the image. Where
  there are few PSF stars, often in image corners, it extrapolates and can develop negative lobes, and sources
  there come out too faint. Band values where the PSF model dips below −0.2% of its peak at the source carry
  bit 2 of `FLAGS_{b}_fit`. The size of this bias depends on the field and band and has not yet been measured
  on independent validation data. The package also contains code for a different PSF model; this version does
  not use it.
- **Flux scale.** In development tests on a sparse 7DT field, real stars copied into the images (so that their
  true flux is known) were measured a few percent too faint overall, at both low and high signal-to-noise
  ratios. The cause is not yet established, and the size has not yet been measured on independent validation
  data.
- **Errors.** Errors are propagated from a noise model. Their check against independent truth covers only a
  few 7DT test fields and is not finished. In crowded fields the errors of different bands are correlated, and
  this correlation is not reported. The source Poisson term needs the gain, which uniphot derives from header
  keywords with rules written for 7DT products; for coadds made by an older version of the 7DT pipeline the
  derived gain is about 30 times too large, so that term is too small. For such data and for other instruments, set
  `compute.gpu.error_model.source_poisson.gain_eff` to the effective gain of each band (e⁻/ADU).
- **Crowded fields.** A group of touching sources that is too large to fit (a "giant island") is not
  position-fitted: its sources keep their input positions, their fluxes are measured there, and they are marked
  unreliable (`giant_frozen_geometry` in `fit_quality_issues`). In one dense 7DT field near the Galactic plane
  (|b| < 2°), this was 39% of the rows (measured with an earlier development version).
- **Results depend on the image tiling.** The fluxes are solved in rectangular tiles whose layout is chosen
  automatically from the number of sources, so a small change in the source list can change the layout and move
  fluxes. With an earlier development version, changing only the layout moved 1.4% of band fluxes by more than
  their statistical error (`FLUXERR`).
- **Bright-object mask.** Bright objects that are not in your list are found on a sum of all bands, after
  subtracting a local sky and scaling the noise (`SKYSIG`) to the scatter measured around each pixel. A detection
  is taken as the light of a listed source when its peak lies within a radius of the row: the larger of the PSF
  width (FWHM) and 3.72 times the scatter between the rows' positions and the detected peaks, both measured in the
  run, and at most `compute.gpu.positions.pos_max_shift_pix` (5 pixels by default). The radius is written to `run.log`
  and `bright_mask/association.json`. Listed sources inside the mask are not fitted: their
  `affected_by_bright_mask` is `True` and `fit_is_reliable` is `False` (reason `bright_mask`). In the sample run
  this was 4 of the 751 listed sources (99 in version 0.1.0), all stars whose light peaks farther from the listed
  position than the radius (3.84 pixels in that run). The mask can also cover part of a fitted source, and
  `fit_is_reliable` does not take this into account (column `fracmasked` gives the fraction of the model flux on
  masked pixels); in the sample run no fitted listed source has more than 20% (12 in version 0.1.0). In crowded
  fields the measured scatter is larger than `SKYSIG` (about 2.25 times in a crowded 7DT test field; we think
  mainly from the light of unresolved stars, but have not tested it), so fewer objects are detected and masked
  than with `SKYSIG` alone (see `CHANGELOG.md`).
- **Deep extragalactic fields.** By default, bright objects that are not in your list are subtracted as point
  sources (`bright_mask.subtract_unmatched: true`), which suits star fields. In deep extragalactic fields set it
  to `false`: in a 7DT test of such a field, the default moved 40% of the sources by more than 10 mmag
  compared with `false` (measured with an earlier development version).
- **Five checks that version 0.1.1 failed (four problems).** Before release, 0.1.1 was compared with 0.1.0 on
  the sample data and on a sparse and a crowded 7DT test field, against limits set before the runs. Five checks
  failed; they show the four problems below (problem 4 failed two checks). Only the bright-object mask code
  changed; the failures show how the unchanged fitting and flagging code responds to the changed detection of
  bright objects not in your list (the mask, and the objects subtracted as point sources; in the sparse field
  the mask did not change, only the subtracted point sources). In the two test fields, running 0.1.0 twice on
  the same input changed no flag and moved the compared fluxes by at most 0.0006 times their flux error
  (`FLUXERR_{b}_fit`), so the changes below are not run-to-run noise. Fixes are planned for a later version.
  1. **Two list rows on one object.** If your list has the same object twice (in the sample list, apparently
     saturated stars that the source catalog split in two), the two fitted positions can meet at nearly one point,
     and the split of the flux between the rows is arbitrary: one can be negative, and both can have
     `fit_is_reliable` = `True`. The sample list has 8 such pairs (16 rows), the same in 0.1.0; 2 pairs (4 rows)
     are marked reliable, and their magnitudes are off by 0.15 to 2.4 mag in both directions, while their reported
     errors are 0.03 to 0.14 mag. Such rows have fitted positions within 1 pixel of each other; the sum of their
     fluxes agrees with the Gaia XP synthetic photometry of the star within 0.06 mag in 31 of 32 band values
     (0.16 mag in the other).
  2. **Galaxies in one image tile.** In the sample run, the 19 galaxies checked in one of the four tiles, far from
     where the mask changed, came out about 1% larger (`Re_fit`, median) and 0.8 to 1.0% brighter (median) in
     m500, m600 and m700 than in 0.1.0. In m500 the median change is 0.17 times their flux error, where 0.05 was
     expected and 0.15 was the limit.
  3. **Reliability flag of isolated stars.** In the sparse field, 23 of the 5,550 compared sources (0.41%), all
     isolated stars, switched `fit_is_reliable` (12 to `False`, 11 to `True`; reason `island_stalled`), while
     their fluxes moved by at most 0.017 times their flux error.
  4. **Stars at the edge of the data.** In the crowded field, a group of stars at the edge of the images' coverage
     moved by up to 70 times the flux error of 0.1.0 in band values with no flag other than bit 4, in rows marked
     reliable; band values with bit 16 at the same edge moved by up to 267 times. In 10 band values there, bit 1
     (no data) was set in 0.1.0 but not in 0.1.1; in 7 of them the fitted position lies on pixels without data and
     the value is meaningless (as large as −2.4 × 10⁷), and they carry bit 16. Which version is closer to the true
     fluxes is not known.
- **No validated comparison with other tools yet.** uniphot has been compared with other photometry tools only in
  development tests. No validated comparison on the same data has been completed, and no claim is made here about
  the accuracy or speed of uniphot relative to other tools.

## License

uniphot is released under the GNU General Public License, version 2 only (GPL-2.0-only).
Copyright (C) 2026 Danhyeuk Seol. `LICENSE` holds the text of the license; `NOTICE` holds the copyright
notice and the list of the pieces derived from The Tractor.

uniphot does not use the Tractor library to install or run, but four pieces derived from
[The Tractor](https://github.com/dstndstn/tractor) (Copyright (C) 2011-2023 Dustin Lang & David W. Hogg,
GPL version 2; Lang, Hogg & Mykytyn 2016, [ascl:1604.008](https://ascl.net/1604.008)) remain:

- `tract7dt/fit/optimize/gpu/kernels/lanczos.py` (Lanczos-3 image shift, modelled on Tractor's `lanczos_shift_image`);
- the mixture-of-Gaussians constants in `tract7dt/fit/grid_gpu/source_model.py`;
- `tract7dt/data/sersic_mog_table.npz` and its generator `tract7dt/fit/grid_gpu/sersic_mog_gen.py`
  (the generator imports Tractor when it is run; it is needed only to rebuild the table);
- the alpha grid in `tract7dt/fit/grid_gpu/solver_grid.py`.

The sample data have their own license (see [Quick start](#quick-start-with-the-sample-data)).

## Citation

If you use uniphot, please cite it with the metadata in `CITATION.cff` (on GitHub: "Cite this repository").

## Contributing

Bug reports and questions are welcome as GitHub issues. Pull requests are not accepted at this time;
see `CONTRIBUTING.md`.

## Contact

Danhyeuk Seol, Seoul National University: tjfeksgur@snu.ac.kr
