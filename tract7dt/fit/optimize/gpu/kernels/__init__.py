"""JAX kernels used by the grid GPU path.

- :mod:`.lanczos` - Lanczos-3 separable image shift (modelled on Tractor's
  ``lanczos_shift_image``); used for sub-pixel PSF positioning by
  :mod:`tract7dt.fit.grid_gpu`.
"""
