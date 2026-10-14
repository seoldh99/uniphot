"""Retired config keys (card 018, design_v2 5.2 / 7.1 test 2).

Expected behaviour is fixed by the design, not read from the implementation table:
- a production YAML that pins every dev8 key (the dev8s5 T08676 anchor, copied
  verbatim into tests/data) still loads, with ONE summary note on stderr;
- a removed opt-in switched away from its dev8 default fails at load (it would
  otherwise be dropped silently);
- compute.mode other than gpu and psf.mode other than from_stars fail;
- the CPU-only blocks (epsf, ...) are ignored whatever they hold.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

DATA = Path(__file__).resolve().parent / "data"

# design_v2 3.4: the removed switches and their dev8 default values.
REMOVED_SWITCHES = {
    "compute.gpu.psf.from_stars.apply_fwhm_cut": False,
    "compute.gpu.psf.from_stars.use_gaia_match": False,
    "compute.gpu.psf.from_stars.use_segmap": False,
    "compute.gpu.psf.from_stars.per_pixel_weights": False,
    "compute.gpu.solver.use_batched_band_solver": False,
    "compute.gpu.solver.island.subtract_detected": False,
    "compute.gpu.solver.island.subtract_detected_protect": True,
    "compute.gpu.solver.island.confusion_weight": False,
    "compute.gpu.solver.island.blank_response_extended": False,
    "compute.gpu.solver.island.detect_unattributed.enabled": False,
    "compute.gpu.solver.island.model_uncertainty.enabled": False,
    "compute.gpu.solver.island.confusion_whitening.enabled": False,
    "compute.gpu.galaxy_diagnostics.enabled": False,
    "compute.gpu.error_model.re_profile_enabled": False,
}


def _set(d: dict, path: str, value) -> None:
    ks = path.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = value


def _anchor_verbatim() -> dict:
    return yaml.safe_load((DATA / "dev8s5_T08676_r1.yaml").read_text())


def _anchor() -> dict:
    """The dev8s5 anchor in its one-pass form.

    The anchor ran the two-pass form (bright_mask.residual_phantom_pass.enabled: true),
    which M2 unit 0.1 removed; the verbatim file is refused at load
    (test_two_pass_anchor_is_refused). Every other test starts from the one-pass form.
    """
    cfg = _anchor_verbatim()
    cfg["bright_mask"]["residual_phantom_pass"]["enabled"] = False
    # M2 unit a.3 removed the separation rule between PSF training stars; the anchor's
    # dev8 value 30 is refused at load (RETIRED_A3_OTHER), 0 (no rule) is what remains.
    cfg["compute"]["gpu"]["psf"]["from_stars"]["min_sep"] = 0.0
    return cfg


def _write(tmp_path: Path, cfg: dict) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def test_dev8_anchor_yaml_loads_with_one_note(config, tmp_path, capsys):
    cfg = config.load_config(_write(tmp_path, _anchor()))
    err = capsys.readouterr().err
    notes = [ln for ln in err.splitlines() if "retired config key" in ln]
    assert len(notes) == 1, err
    assert cfg["compute"]["gpu"]["solver"]["position_solver"] == "island"
    assert "epsf" not in cfg and "runner" not in cfg and "mode" not in cfg["compute"]


@pytest.mark.parametrize("path, dev8", sorted(REMOVED_SWITCHES.items()))
def test_removed_switch_away_from_dev8_default_fails(config, tmp_path, path, dev8):
    cfg = _anchor()
    _set(cfg, path, not dev8)
    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        config.load_config(_write(tmp_path, cfg))


@pytest.mark.parametrize("path, value", [
    ("compute.mode", "cpu"),
    ("compute.mode", "legacy_gpu"),
    ("compute.gpu.psf.mode", "per_cell"),
    ("compute.gpu.psf.mode", "from_cells"),
    ("compute.gpu.psf.from_stars.psf_normalization", "auto"),
])
def test_single_remaining_value_enforced(config, tmp_path, path, value):
    cfg = _anchor()
    _set(cfg, path, value)
    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        config.load_config(_write(tmp_path, cfg))


def test_cpu_blocks_are_ignored_whatever_they_hold(config, tmp_path):
    cfg = _anchor()
    cfg["epsf"] = {"anything": 1, "max_stars": -5}
    cfg["patch_run"]["psf_model"] = "whatever"
    cfg["runner"] = {}
    loaded = config.load_config(_write(tmp_path, cfg))
    assert "epsf" not in loaded and "runner" not in loaded
    assert set(loaded["patch_run"]) == {"gal_model", "re_fallback_pix"}


def test_unknown_key_still_rejected(config, tmp_path):
    cfg = _anchor()
    cfg["compute"]["gpu"]["no_such_key"] = 1
    with pytest.raises(KeyError, match="no_such_key"):
        config.load_config(_write(tmp_path, cfg))


def test_user_mapping_is_not_modified(config, tmp_path):
    cfg = _anchor()
    before = copy.deepcopy(cfg)
    config.resolve_config(cfg, tmp_path)
    assert cfg == before


# ---------------------------------------------------------------------------
# B-1 (a finding of the review of the CPU-path removal).
# tract7dt@dev8's _migrate_legacy_compute_keys sent some YAMLs to the removed
# legacy per-patch GPU path. The new engine must refuse them at load instead of
# running the grid path silently. Which shapes are refused and which load comes from
# what dev8 did with each shape (probe in b1_fix.md, dev8 config.py run as is), not
# from this implementation:
#   - compute.gpu.framework is refused whatever its value;
#   - compute.mode 'gpu' with a compute.gpu block of per-patch keys only (dev8 ran
#     legacy_gpu) is refused;
#   - the layouts dev8 ran on the grid path load. Their expected values are what the
#     YAML says (the current layout), which is not always what dev8 used: for S08
#     dev8 moved the written devices to its legacy block and ran with "auto".

def _minimal(compute: dict) -> dict:
    return {"outputs": {"work_dir": "wd"}, "compute": copy.deepcopy(compute)}


@pytest.mark.parametrize("value", ["jax", "JAX", "numpy", None, True])
def test_framework_key_is_refused_whatever_its_value(config, tmp_path, value):
    cfg = _anchor()
    cfg["compute"]["gpu"]["framework"] = value
    with pytest.raises(ValueError, match=r"compute\.gpu\.framework"):
        config.load_config(_write(tmp_path, cfg))
    with pytest.raises(ValueError, match=r"compute\.gpu\.framework"):
        config.resolve_config(cfg, tmp_path)


def test_framework_reported_with_other_retired_violations(config, tmp_path):
    cfg = _anchor()
    cfg["compute"]["gpu"]["framework"] = "jax"
    cfg["compute"]["gpu"]["solver"]["use_batched_band_solver"] = True
    with pytest.raises(ValueError) as ei:
        config.load_config(_write(tmp_path, cfg))
    msg = str(ei.value)
    assert "compute.gpu.framework" in msg
    assert "compute.gpu.solver.use_batched_band_solver" in msg


# dev8 resolved compute.mode to 'legacy_gpu' for each of these (b1_fix.md S01, S02, S07, S11).
DEV8_LEGACY_GPU_SHAPES = {
    "S01 framework": ({"mode": "gpu", "gpu": {"framework": "jax"}}, "compute.gpu.framework"),
    "S02 framework+runtime": ({"mode": "gpu", "gpu": {"framework": "jax", "devices": [1, 2],
                                                      "mem_fraction": 0.3, "jit_cache_dir": "auto"}},
                              "compute.gpu.framework"),
    "S07 devices": ({"mode": "gpu", "gpu": {"devices": [1, 2]}}, "compute.mode"),
    "S07 devices+runtime": ({"mode": "gpu", "gpu": {"devices": [0], "mem_fraction": 0.3,
                                                    "jit_cache_dir": "auto"}}, "compute.mode"),
    "S11 devices+dtype": ({"mode": "gpu", "gpu": {"devices": [1, 2], "dtype": "float32"}},
                          "compute.(mode|gpu.dtype)"),
}


@pytest.mark.parametrize("name", sorted(DEV8_LEGACY_GPU_SHAPES))
def test_dev8_legacy_gpu_layout_fails_at_load(config, tmp_path, name):
    compute, key = DEV8_LEGACY_GPU_SHAPES[name]
    with pytest.raises((ValueError, KeyError), match=key.replace(".", r"\.")):
        config.load_config(_write(tmp_path, _minimal(compute)))


# dev8 ran these on the grid path (b1_fix.md S08, S09, S10, S18); they must still load.
# Expected values = what the YAML says (None = the DEFAULT_CONFIG value), not dev8's:
# dev8 ran S08 with devices "auto", this engine with the written [1, 2].
DEV8_GRID_SHAPES = {
    "S08 devices, no mode": ({"gpu": {"devices": [1, 2]}}, [1, 2], None),
    "S09 mode + devices + solver group": ({"mode": "gpu", "gpu": {
        "devices": [1, 2], "solver": {"position_solver": "island"}}}, [1, 2], None),
    "S10 mode + mem_fraction/jit_cache_dir": ({"mode": "gpu", "gpu": {
        "mem_fraction": 0.3, "jit_cache_dir": "auto"}}, None, 0.3),
    "S18 mode + legacy_gpu block": ({"mode": "gpu", "legacy_gpu": {"devices": [0]}}, None, None),
}


@pytest.mark.parametrize("name", sorted(DEV8_GRID_SHAPES))
def test_dev8_grid_layouts_load_with_what_they_say(config, tmp_path, name):
    compute, devices, mem_fraction = DEV8_GRID_SHAPES[name]
    d = config.DEFAULT_CONFIG["compute"]["gpu"]
    cfg = config.load_config(_write(tmp_path, _minimal(compute)))
    gpu = cfg["compute"]["gpu"]
    assert gpu["devices"] == (d["devices"] if devices is None else devices)
    assert gpu["mem_fraction"] == (d["mem_fraction"] if mem_fraction is None else mem_fraction)
    assert "mode" not in cfg["compute"] and "legacy_gpu" not in cfg["compute"]
    assert "framework" not in gpu


# b1_review.md C-1: the refusal text may only claim what dev8 did. dev8 moved a
# per-patch-only compute.gpu block to compute.legacy_gpu whatever compute.mode said,
# but switched the run to the legacy per-patch GPU path only for the exact string
# 'gpu' (its step 3 compares with ==). 'GPU' / ' gpu' are still refused here (the
# 'only' rule reads them as gpu), without the legacy-path claim.
@pytest.mark.parametrize("mode, dev8_switched", [("gpu", True), ("GPU", False), (" gpu", False)])
def test_layout_refusal_claims_only_what_dev8_did(config, tmp_path, mode, dev8_switched):
    user = _minimal({"mode": mode, "gpu": {"devices": [1, 2]}})
    with pytest.raises(ValueError) as ei:
        config.load_config(_write(tmp_path, user))
    msg = str(ei.value)
    assert "compute.mode" in msg and "per-patch GPU layout" in msg
    assert ("legacy per-patch GPU path" in msg) is dev8_switched


# ---------------------------------------------------------------------------
# M2 unit 0.1 (card 024 impl_plan 1.2 = card 018 C6d; 10-09 decision 6: one pass, no opt-in
# left): the two-pass residual-phantom form is removed with the keys that only served it.
# Rule (impl_plan 1.0-6): the value of the surviving behaviour is dropped with the one note;
# any other value fails at load naming the key. The surviving behaviour is what a one-pass
# run did in dev8: no second pass, no external unmatched list, no bg_diag dump, ZP solved.
_MISSING = object()
RETIRED_0_1_SURVIVING = {
    "bright_mask.residual_phantom_pass.enabled": False,
    "bright_mask.extra_unmatched_csv": None,
    "compute.gpu.background.dump_diagnostics": False,
    "zp.calibrate": True,
}
RETIRED_0_1_OTHER = {
    "bright_mask.residual_phantom_pass.enabled": True,
    "bright_mask.extra_unmatched_csv": "extra_unmatched.csv",
    "compute.gpu.background.dump_diagnostics": True,
    "zp.calibrate": False,
}
# knobs of the removed pass: inert while the pass is off, so any value is dropped
RETIRED_0_1_KNOBS = {
    "bright_mask.residual_phantom_pass.nsigma": 9.0,
    "bright_mask.residual_phantom_pass.minarea": 1,
    "bright_mask.residual_phantom_pass.match_radius_pix": 0.5,
}


def _load_error(config, tmp_path, cfg):
    try:
        config.load_config(_write(tmp_path, cfg))
    except (ValueError, KeyError) as e:
        return e
    return None


def _get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return _MISSING
        d = d[k]
    return d


def test_two_pass_anchor_is_refused(config, tmp_path):
    err = _load_error(config, tmp_path, _anchor_verbatim())
    assert isinstance(err, ValueError), err
    assert "bright_mask.residual_phantom_pass.enabled" in str(err), err


@pytest.mark.parametrize("path", sorted(RETIRED_0_1_OTHER))
def test_removed_two_pass_value_fails_at_load(config, tmp_path, path):
    cfg = _anchor()
    _set(cfg, path, RETIRED_0_1_OTHER[path])
    err = _load_error(config, tmp_path, cfg)
    assert isinstance(err, ValueError), err
    assert path in str(err), err


@pytest.mark.parametrize("path, value", sorted({**RETIRED_0_1_SURVIVING, **RETIRED_0_1_KNOBS}.items()))
def test_surviving_two_pass_value_is_dropped_with_the_note(config, tmp_path, capsys, path, value):
    cfg = _anchor()
    _set(cfg, path, value)
    loaded = config.load_config(_write(tmp_path, cfg))
    notes = [ln for ln in capsys.readouterr().err.splitlines() if "retired config key" in ln]
    assert len(notes) == 1 and path in notes[0], notes
    assert _get(loaded, path) is _MISSING
    assert _get(config.DEFAULT_CONFIG, path) is _MISSING


# ---------------------------------------------------------------------------
# M2 unit a.3 (card 024; design 1.5 v2.4, designer answer D9-4; plan 1.0-6): the ellipticity
# cut and the separation rule between PSF training stars are removed.  Rule 1.0-6: the value
# that equals the remaining behaviour loads with the note, any other value fails at load.
#   max_ellipticity: no cut = 10.0 (the dev8 default; a SEP ellipticity is below 1);
#   min_sep: no separation rule = 0 -- the dev8 default 30 is a different behaviour and fails.
RETIRED_A3_SURVIVING = {"compute.gpu.psf.from_stars.max_ellipticity": 10.0,
                        "compute.gpu.psf.from_stars.min_sep": 0.0}
RETIRED_A3_OTHER = {"compute.gpu.psf.from_stars.max_ellipticity": 0.3,
                    "compute.gpu.psf.from_stars.min_sep": 30.0}


@pytest.mark.parametrize("path, value", sorted(RETIRED_A3_SURVIVING.items()))
def test_a3_remaining_value_is_dropped_with_the_note(config, tmp_path, capsys, path, value):
    cfg = _anchor()
    _set(cfg, path, value)
    loaded = config.load_config(_write(tmp_path, cfg))
    notes = [ln for ln in capsys.readouterr().err.splitlines() if "retired config key" in ln]
    assert len(notes) == 1 and path in notes[0], notes
    assert _get(loaded, path) is _MISSING
    assert _get(config.DEFAULT_CONFIG, path) is _MISSING


@pytest.mark.parametrize("path, value", sorted(RETIRED_A3_OTHER.items()))
def test_a3_other_value_fails_at_load(config, tmp_path, path, value):
    cfg = _anchor()
    _set(cfg, path, value)
    err = _load_error(config, tmp_path, cfg)
    assert isinstance(err, ValueError), err
    assert path in str(err), err
