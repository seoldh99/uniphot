"""Grid-based GPU image-fitting backend.

Backend for ``compute.mode: gpu``. Designed around the structure GPUs
are good at: ONE big linear-algebra-heavy fit per quadrant of the full
image, instead of thousands of tiny per-patch fits with sequential CPU
orchestration around them.

Architecture (per band, per quadrant):

- 2-D POLYNOMIAL PSF (PSFEx-style): each pixel of the PSF stamp is a
  polynomial in (x, y) sky position, built once per band from
  individual stars (``psf_from_stars``).
- 2-D POLYNOMIAL background: low-order polynomial in (x, y).
- N_SOURCES from the input catalog (positions seeded from the catalog,
  fitted with bound clipping). PointSources only in this version.
- ALL fluxes + background coefs solved jointly via sparse Cholesky
  given current positions (linear).
- POSITIONS refined via per-source 2x2 Newton step (vmapped across
  sources).

Modules:

- :mod:`.psf_polynomial`     - fit + evaluate PSFEx-style polynomial PSF
- :mod:`.bg_polynomial`      - 2-D polynomial background basis + eval
- :mod:`.grid`               - split image into quadrants with halos
- :mod:`.forced_phot_sparse` - SPARSE linear flux+bg solve
- :mod:`.source_model`       - STAR + EXP rendering primitives
- :mod:`.solver_grid`        - 3-stage LM (S1 linear / S2 pos newton /
                               S3 alternating); replaces the old
                               single-stage ``.solver`` module
- :mod:`.runner`             - per-grid worker (one per GPU)
- :mod:`.pipeline`           - top-level integration
"""

# Lazy imports - heavy (jax, scipy.sparse) - only loaded when used.

__all__ = [
    "psf_polynomial",
    "bg_polynomial",
    "grid",
    "forced_phot_sparse",
    "source_model",
    "solver_grid",
    "runner",
    "pipeline",
]
