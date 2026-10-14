"""Run contract (REGISTER R12).

Every ``tract7dt run`` ends with ``RUN_STATUS.json``: "ok" and exit 0 only
when nothing failed. Until dev8 a final-pass grid that failed after its retry
was logged and its sources shipped as NaN rows, an error-model failure shipped
errors without their confusion term (x~2.4 too small on crowded fields), and a
ZP stage that could not calibrate shipped a catalog without magnitudes -- all
under exit 0, and the campaign gate trusted the exit code and the catalog's
existence. A per-user lock keeps a second GPU run from starting beside the
first, and the private MPS daemon is replaced when it does not see every card.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from tract7dt import run_status
from tract7dt.run_status import RunContractError, RunLockBusy


@pytest.fixture
def lock_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(run_status, "lock_path", lambda: tmp_path / "run.lock")
    return tmp_path / "run.lock"


@pytest.fixture
def restore_environ():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_status_file_round_trip(tmp_path):
    run_status.reset()
    started = run_status.now_iso()
    run_status.write(tmp_path, status="running", started=started)
    doc = run_status.read(tmp_path)
    assert doc["status"] == "running" and doc["finished"] is None
    run_status.set_stage("run gpu fit")
    run_status.note("final_pass", {"n_grids": 4})
    run_status.warn("[m450] no zero point")
    err = run_status.describe(RunContractError("grid_failed", "1 of 4", grids=[7]))
    run_status.write(tmp_path, status="failed", started=started,
                     exit_code=1, error=err)
    doc = run_status.read(tmp_path)
    assert doc["schema"] == run_status.SCHEMA
    assert doc["status"] == "failed" and doc["exit_code"] == 1
    assert doc["finished"] and doc["stage"] == "run gpu fit"
    assert doc["error"] == {"kind": "grid_failed", "message": "1 of 4",
                            "details": {"grids": [7]}}
    assert doc["summary"] == {"final_pass": {"n_grids": 4}}
    assert doc["warnings"] == ["[m450] no zero point"]
    assert not (tmp_path / "RUN_STATUS.json.tmp").exists()


def test_unreadable_status_reads_as_none(tmp_path):
    assert run_status.read(tmp_path) is None
    (tmp_path / "RUN_STATUS.json").write_text("{broken")
    assert run_status.read(tmp_path) is None


def test_second_run_of_the_same_user_is_refused(lock_in_tmp, restore_environ):
    first = run_status.acquire_run_lock("first.yaml")
    assert first is not None
    assert json.loads(lock_in_tmp.read_text())["pid"] == os.getpid()
    with pytest.raises(RunLockBusy, match="first.yaml"):
        run_status.acquire_run_lock("second.yaml")
    first.close()                                  # the first run ends
    again = run_status.acquire_run_lock("third.yaml")
    assert again is not None
    again.close()


def test_a_run_started_under_a_running_one_is_refused(lock_in_tmp, restore_environ):
    """M2 unit 0.1: the engine no longer starts child runs (the two-pass form is gone), so a
    `tract7dt run` started while another one holds the lock is refused even when it carries the
    marker that dev8 gave its two-pass children (TRACT7DT_RUN_LOCK_OWNER = the parent's pid)."""
    holder = run_status.acquire_run_lock("the running one")
    os.environ["TRACT7DT_RUN_LOCK_OWNER"] = str(os.getppid())
    try:
        run_status.acquire_run_lock("started under it")
        refused = False
    except RunLockBusy:
        refused = True
    finally:
        holder.close()
    assert refused


# ---- the CLI writes RUN_STATUS.json and sets the exit code ------------------

def _config_file(tmp_path, config):
    seed = config.write_sample_config(tmp_path / "seed.yaml", overwrite=True)
    cfg = yaml.safe_load(seed.read_text())
    cfg["outputs"]["work_dir"] = str(tmp_path / "wd")
    p = tmp_path / "run.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p, tmp_path / "wd"


@pytest.mark.parametrize("outcome, code, status, kind", [
    (None, 0, "ok", None),
    (RunContractError("zp_not_computed", "no calibrator"), 1, "failed",
     "zp_not_computed"),
    (ValueError("boom"), 1, "failed", "exception"),
])
def test_cli_run_reports_its_outcome(tmp_path, config, monkeypatch, lock_in_tmp,
                                     restore_environ, outcome, code, status, kind):
    import tract7dt.cli as cli
    cfg_path, wd = _config_file(tmp_path, config)
    seen = {}

    def fake_run_pipeline(cfg):
        seen["status_during_run"] = run_status.read(wd)["status"]
        run_status.set_stage("run gpu fit")
        if outcome is not None:
            raise outcome
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    assert cli.main(["run", "--config", str(cfg_path)]) == code
    assert seen["status_during_run"] == "running"
    doc = run_status.read(wd)
    assert doc["status"] == status and doc["exit_code"] == code
    assert (doc["error"] or {}).get("kind") == kind
    if kind:
        assert doc["stage"] == "run gpu fit"


def test_cli_refuses_a_second_gpu_run_without_touching_its_work_dir(
        tmp_path, config, monkeypatch, lock_in_tmp, restore_environ):
    import tract7dt.cli as cli
    cfg_path, wd = _config_file(tmp_path, config)
    holder = run_status.acquire_run_lock("the running one")
    monkeypatch.setattr(cli, "run_pipeline",
                        lambda cfg: pytest.fail("must not start"))
    try:
        assert cli.main(["run", "--config", str(cfg_path)]) == run_status.EXIT_LOCKED
    finally:
        holder.close()
    assert not wd.exists()


# ---- failure sources ---------------------------------------------------------

def test_a_grid_that_still_fails_fails_the_run():
    from tract7dt.fit.grid_gpu.pipeline import _raise_if_grids_failed
    ok = SimpleNamespace(grid_id=1, error=None)
    bad = SimpleNamespace(grid_id=7, error="RuntimeError: error model failed: x")
    _raise_if_grids_failed([ok, ok])
    with pytest.raises(RunContractError) as ei:
        _raise_if_grids_failed([ok, bad])
    assert ei.value.kind == "grid_failed"
    assert ei.value.details["grids"] == [
        {"grid_id": 7, "error": "RuntimeError: error model failed: x"}]


def _tiny_task(**kw):
    from tract7dt.fit.grid_gpu.psf_polynomial import PsfPolynomial
    from tract7dt.fit.grid_gpu.runner import PerGridTask
    S, NT, H, W = 25, 10, 64, 64
    yy, xx = np.mgrid[:S, :S] - S // 2
    g = np.exp(-0.5 * (xx ** 2 + yy ** 2) / 1.8 ** 2)
    coefs = np.zeros((S, S, NT), np.float32)
    coefs[..., 0] = g / g.sum()
    psf = PsfPolynomial(coefs=coefs, x_origin=W / 2, x_scale=W / 2,
                        y_origin=H / 2, y_scale=H / 2, H_psf=S, W_psf=S)
    rng = np.random.default_rng(5)
    px, py, fl = np.array([20.0, 42.0]), np.array([25.0, 40.0]), np.array([300.0, 200.0])
    data = rng.normal(0.0, 1.0, (H, W))
    for x, y, f in zip(px, py, fl):
        data += f * np.exp(-0.5 * ((np.arange(W)[None, :] - x) ** 2
                                   + (np.arange(H)[:, None] - y) ** 2)
                           / 1.8 ** 2) / g.sum()
    n = len(px)
    return PerGridTask(
        grid_id=3, grid_row=0, grid_col=0, bandnames=["g"],
        data_per_band=[data.astype(np.float32)],
        invvar_per_band=[np.ones((H, W), np.float32)],
        pix_good_per_band=[np.ones((H, W), bool)],
        psfs_per_band=[psf], psf_shape=(S, S), grid_x0=0, grid_y0=0,
        source_indices=np.arange(n, dtype=np.int64),
        pos_x_init=px.astype(np.float32), pos_y_init=py.astype(np.float32),
        pos_err_pix=np.full(n, 0.1, np.float32),
        source_type=np.zeros(n, np.int32),
        logre_init=np.zeros(n, np.float32), ee1_init=np.zeros(n, np.float32),
        ee2_init=np.zeros(n, np.float32), pos_max_shift_pix=1.0,
        bg_x_origin=W / 2, bg_x_scale=W / 2, bg_y_origin=H / 2,
        bg_y_scale=H / 2, bg_poly_order=0, single_stage=True,
        n_opt_iters=0, **kw)


def test_an_error_model_failure_fails_its_grid(monkeypatch):
    jax = pytest.importorskip("jax")
    import tract7dt.fit.grid_gpu.error_model as em
    from tract7dt.fit.grid_gpu.pipeline import _is_oom_error
    from tract7dt.fit.grid_gpu.runner import run_one_grid

    def boom(*a, **k):
        raise RuntimeError("RESOURCE_EXHAUSTED: out of memory in the render")
    monkeypatch.setattr(em, "compute_grid_confusion", boom)
    enable_x64 = getattr(jax, "enable_x64", None)
    if enable_x64 is None:                         # older JAX
        from jax.experimental import enable_x64
    with enable_x64():
        res = run_one_grid(_tiny_task(compute_confusion=True))
    assert res.error and "error model failed" in res.error
    assert np.isnan(np.asarray(res.fluxes, dtype=float)).all()
    assert _is_oom_error(res)            # -> retried with exclusive memory


def test_zp_stage_that_cannot_calibrate_fails_after_writing_the_fit(tmp_path):
    from tract7dt.zp import compute_zp
    cat = pd.DataFrame({"ID": ["a", "b"], "FLUX_g_fit": [1.0, 2.0],
                        "FLUXERR_g_fit": [0.1, 0.1],
                        "fit_is_reliable": [True, True]})
    out = tmp_path / "output_catalog.csv"
    cfg = {"zp": {}, "inputs": {"gaiaxp_synphot_csv": str(tmp_path / "none.csv")},
           "outputs": {"work_dir": str(tmp_path)}}
    with pytest.raises(RunContractError) as ei:
        compute_zp(cfg=cfg, merged_catalog_path=out, catalog_df=cat)
    assert ei.value.kind == "zp_not_computed"
    written = pd.read_csv(out)
    assert list(written["FLUX_g_fit"]) == [1.0, 2.0]
    assert not any(c.startswith("MAG") for c in written.columns)


# ---- MPS daemon: every card visible, or replaced -----------------------------

def _fake_mps(monkeypatch, *, daemon_up, daemon_env):
    import tract7dt.fit.grid_gpu.pipeline as gp
    state = {"up": daemon_up, "calls": [], "started_env": None}

    def fake_control(cmd, pipe_dir, timeout=5):
        state["calls"].append(cmd)
        if cmd == "quit":
            state["up"] = False
            return True, ""
        return state["up"], ""

    def fake_run(argv, env=None, **kw):
        state["started_env"] = dict(env)
        state["up"] = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gp, "_mps_control", fake_control)
    monkeypatch.setattr(gp, "_mps_daemon_env", lambda pipe_dir: daemon_env)
    monkeypatch.setattr(gp, "_install_mps_cleanup", lambda: None)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr("subprocess.run", fake_run)
    return gp, state


def test_mps_daemon_is_started_seeing_every_card(monkeypatch, restore_environ):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    gp, state = _fake_mps(monkeypatch, daemon_up=False, daemon_env=None)
    active, ours, _ = gp._maybe_start_mps(True)
    assert active and ours
    assert "CUDA_VISIBLE_DEVICES" not in state["started_env"]
    assert "quit" not in state["calls"]
    gp._MPS_OWNED_PIPE = None


def test_leftover_daemon_under_a_card_subset_is_replaced(monkeypatch, restore_environ):
    gp, state = _fake_mps(monkeypatch, daemon_up=True,
                          daemon_env={"CUDA_VISIBLE_DEVICES": "1,2"})
    active, ours, _ = gp._maybe_start_mps(True)
    assert active and ours
    assert "quit" in state["calls"] and state["started_env"] is not None
    gp._MPS_OWNED_PIPE = None


def test_leftover_daemon_seeing_every_card_is_reused(monkeypatch, restore_environ):
    gp, state = _fake_mps(monkeypatch, daemon_up=True, daemon_env={})
    active, ours, _ = gp._maybe_start_mps(True)
    assert active and not ours
    assert "quit" not in state["calls"] and state["started_env"] is None
