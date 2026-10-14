"""Package shell kept only for the import path of :mod:`.gpu.kernels.lanczos`.

The per-patch optimizer backends that lived here were removed with the
CPU/Tractor path (card 018). The Lanczos kernel moves to
:mod:`tract7dt.fit.grid_gpu` when it is re-derived (see its docstring).
"""
