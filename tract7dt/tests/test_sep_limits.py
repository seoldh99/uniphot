"""SEP process limits travel with the SEP calls, not with an unrelated import (card 018, test 0).

Why: in tract7dt dev8 the SEP limits (sub-object limit 200000, active-pixel stack 2,000,000) were set as a
module-level side effect of two CPU/Tractor modules (``fit/runner.py`` and ``psf/builder.py``, both since removed) that every process
imported through the ``fit/__init__.py`` and ``pipeline.py`` seams. Removing those modules without moving the
limits silently returns SEP to its own defaults (1024 / 300,000): on a crowded image the PSF-star detection
returns an empty list (``detect_sep_objects`` swallows the error) and the bright-star detection raises.
(The residual-speck detection of the two-pass form, the third SEP call, went with that form in M2 unit 0.1.)

Expected values are independent of the implementation: a synthetic image with one connected region of
490,000 pixels above every threshold used here (> 300,000) holding 1,500 point sources (> 1024 sub-objects).
Under dev8's effective limits SEP extracts it without error (>= 1 object); under SEP's defaults it raises.

Each case runs in a NEW process: the limits are process-global, so an import made by this test process (or by
another test) must not be able to set them for the code under test.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


import tract7dt

PKG_ROOT = str(Path(tract7dt.__file__).resolve().parents[1])

# Synthetic crowded image: plateau 50 sigma above zero (above every threshold below), 1,500 Gaussian points.
_MAKE_IMAGE = textwrap.dedent("""
    import numpy as np
    rng = np.random.default_rng(1)
    n = 700
    img = np.full((n, n), 50.0, np.float32)
    yy, xx = np.mgrid[0:n, 0:n]
    for _ in range(1500):
        x0, y0 = rng.uniform(5, n - 5, 2)
        sl = (slice(max(int(y0) - 6, 0), int(y0) + 7), slice(max(int(x0) - 6, 0), int(x0) + 7))
        img[sl] += 200 * np.exp(-((xx[sl] - x0) ** 2 + (yy[sl] - y0) ** 2) / (2 * 1.5 ** 2))
    img += rng.normal(0, 1, img.shape).astype(np.float32)
    err = np.ones_like(img)
""")


def _run_fresh(code: str, tmp_path: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = PKG_ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("JAX_PLATFORMS", "cpu")
    out = tmp_path / "result.json"
    script = _MAKE_IMAGE + textwrap.dedent(code) + textwrap.dedent(f"""
        import json
        with open({str(out)!r}, "w") as fh:
            json.dump(result, fh)
    """)
    r = subprocess.run([sys.executable, "-c", script], cwd=str(tmp_path), env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"subprocess failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}"
    with open(out) as fh:
        return json.load(fh)


def test_psf_star_detection_not_empty_on_crowded_image(tmp_path):
    res = _run_fresh("""
        from tract7dt.fit.grid_gpu.psf_from_stars import detect_sep_objects
        objs = detect_sep_objects(img, err, np.zeros(img.shape, bool))
        result = {"n": int(len(objs))}
    """, tmp_path)
    # detect_sep_objects returns an empty array when sep.extract raises -- non-empty is the key assertion.
    assert res["n"] >= 1


def test_bright_star_detection_does_not_raise_on_crowded_image(tmp_path):
    res = _run_fresh("""
        import copy
        from tract7dt.config import DEFAULT_CONFIG
        from tract7dt.bright_mask import detect_bright_sources
        import sep
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        objs = detect_bright_sources(cfg, {"white": img, "sigma_white": err})
        result = {"n": int(len(objs)), "pixstack": int(sep.get_extract_pixstack()),
                  "sub_object_limit": int(sep.get_sub_object_limit())}
    """, tmp_path)
    assert res["n"] >= 1
    # Since uniphot 0.1.1 the bright detection removes the sky first, which takes this image's 50-sigma plateau
    # away, so the call no longer needs the raised limits to pass; the limits must still be in force in the
    # fresh process right after the call (dev8's effective values).
    assert res["pixstack"] == 2_000_000 and res["sub_object_limit"] == 200_000

