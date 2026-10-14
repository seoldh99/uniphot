"""SEP process limits, applied right before every SEP extraction in the package.

SEP keeps two process-global limits: the number of sub-objects while deblending
one object (SEP default 1024) and the active-pixel stack of one extraction (SEP
default 300,000). Crowded 7DT tiles exceed both defaults; then ``sep.extract``
raises ("object deblending overflow" / "internal pixel buffer full"), and the PSF
star detection, which catches the error, silently returns no stars.

Up to tract7dt dev8 these limits were set as an import side effect of the CPU
modules ``fit/runner.py`` and ``psf/builder.py``, which every process imported.
Card 018 removed those modules and moved the limits here, with the same values,
so they travel with the calls instead of with an unrelated import. The basis of
the values and whether they should adapt to the image are not addressed here
(recorded in card 018).
"""

from __future__ import annotations

SUB_OBJECT_LIMIT = 200_000
EXTRACT_PIXSTACK = 2_000_000


def apply_sep_limits(sep_module=None) -> None:
    """Set the SEP limits above in this process (idempotent, cheap)."""
    if sep_module is None:
        import sep as sep_module
    sep_module.set_sub_object_limit(SUB_OBJECT_LIMIT)
    sep_module.set_extract_pixstack(EXTRACT_PIXSTACK)
