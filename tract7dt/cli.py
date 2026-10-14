from __future__ import annotations

import argparse
import datetime
import logging
import os
import shutil
import sys
from pathlib import Path

from .config import load_config, write_sample_config
from .logging_utils import setup_logging
from .pipeline import run_pipeline
from .sample_data import download_sample_data


def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", required=True, help="Path to YAML config")


def _suppress_core_dumps() -> None:
    """Disable core dumps for this process and every child (workers inherit).

    A crashing run (e.g. SIGBUS from a full /dev/shm, SIGSEGV) can hold tens
    of GB of anon RAM; with apport-style core collection the kernel then
    writes a same-sized core file, and a crash-retry loop fills the root
    filesystem (observed: 5 x 29 GB in 12 minutes on a shared box).
    RLIMIT_CORE=0 makes the kernel skip core generation entirely, regardless
    of the server's core_pattern/apport configuration. Debugging escape
    hatch: TRACT7DT_ALLOW_COREDUMP=1.
    """
    if os.environ.get("TRACT7DT_ALLOW_COREDUMP") == "1":
        return
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, ValueError, OSError):
        pass  # non-POSIX or restricted environment -- best-effort


def main(argv: list[str] | None = None) -> int:
    _suppress_core_dumps()
    setup_logging()
    log = logging.getLogger("tract7dt.cli")

    from . import __version__

    # Two command names for one entry point: `uniphot` and the alias `tract7dt`.
    prog = Path(sys.argv[0]).name if Path(sys.argv[0]).name in ("uniphot", "tract7dt") else "uniphot"
    ap = argparse.ArgumentParser(prog=prog, description="uniphot: GPU forced PSF photometry")
    ap.add_argument("--version", action="version", version=f"uniphot {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_bump = sub.add_parser("dump-config", help="Write the latest sample config file")
    ap_bump.add_argument("--out", default="sample_config.yaml", help="Output path for sample config")
    ap_bump.add_argument("--force", action="store_true", help="Overwrite if output file exists")

    ap_dl = sub.add_parser("download-sample", help="Download sample image data")
    ap_dl.add_argument(
        "--dir",
        dest="download_dir",
        default=None,
        help="Target dataset directory path (default: ./<original-sample-dir-name>)",
    )
    ap_dl.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing target dataset directory",
    )

    ap_run = sub.add_parser("run", help="Run full pipeline")
    _add_common(ap_run)

    ap_zp = sub.add_parser("compute-zp", help="Compute zero-point calibration from merged catalog")
    _add_common(ap_zp)

    ap_aug = sub.add_parser(
        "augment-gaia",
        help="Add Gaia DR3 quality columns (ruwe, phot_bp_rp_excess_factor) to a "
             "GaiaXP synphot CSV, for zp.gaia_quality_filter (fetched from the Gaia "
             "DR3 archive via TAP)")
    ap_aug.add_argument("input", help="Input GaiaXP synphot CSV (must have a source_id column)")
    ap_aug.add_argument("--out", default=None,
                        help="Output CSV path (default: <input stem>_q.csv)")

    ap_doc = sub.add_parser("doctor", help="Check tract7dt prerequisites and print install hints")
    # No --config required: doctor is a static environment check.

    args = ap.parse_args(argv)

    if args.cmd == "doctor":
        from .doctor import run_doctor
        return run_doctor()

    if args.cmd == "dump-config":
        out = write_sample_config(args.out, overwrite=bool(args.force))
        log.info("Wrote sample config: %s", out)
        return 0

    if args.cmd == "augment-gaia":
        from .gaia_quality import augment_csv
        out = augment_csv(args.input, args.out)
        log.info("Wrote augmented GaiaXP CSV: %s", out)
        return 0

    if args.cmd == "download-sample":
        if args.download_dir:
            target_dir = Path(args.download_dir).expanduser()
            parent_dir = target_dir.parent
        else:
            target_dir = None
            parent_dir = Path.cwd()
        result = download_sample_data(
            path_to_download=parent_dir,
            target_dataset_dir=target_dir,
            overwrite=bool(args.force),
        )
        log.info("Sample data is ready at: %s", result.dataset_dir)
        log.info("Generated image list: %s", result.image_list_file)
        return 0

    cfg = load_config(args.config)

    # R12: one GPU run per user at a time. Taken before the log file and the
    # config snapshot are touched, so a refused run changes nothing in its
    # work_dir (which may be the running one's).
    run_lock = None
    if args.cmd == "run":
        from .run_status import EXIT_LOCKED, RunLockBusy, acquire_run_lock
        try:
            run_lock = acquire_run_lock(str(args.config))
        except RunLockBusy as exc:
            print(f"tract7dt run: not started -- {exc}", file=sys.stderr)
            return EXIT_LOCKED

    log_cfg = cfg.get("logging", {})
    log_file = log_cfg.get("file")
    if isinstance(log_file, str) and log_file.strip().lower() == "auto":
        log_cfg["file"] = Path(cfg["outputs"]["work_dir"]) / f"{args.cmd}.log"
    setup_logging(log_cfg, force=True)

    work_dir = Path(cfg["outputs"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg_src = cfg.get("config_path")
    if cfg_src and Path(cfg_src).exists():
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        snap = work_dir / f"config_used_{ts}.yaml"
        shutil.copy2(str(cfg_src), str(snap))
        log.info("Config snapshot saved: %s", snap)

    # CRITICAL: set the JAX env vars in the
    # PARENT process BEFORE any JAX import happens. JAX reads these at
    # module-import time and caches the allocator decision; setting them
    # later (e.g., inside _worker_init) only affects the WORKER subprocesses,
    # not the parent. Without this, the parent's first jax import would
    # preallocate ~80% of GPU 0's memory (24 GB -> ~19 GB), starving the
    # workers and causing OOM in the per-grid LM solve.
    import os as _os
    _os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    _os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # The parent runs science JAX too (scene-subtract render, the S1
    # packing probe), so it needs the same full-f32 mantissa the workers
    # get. See fit/grid_gpu/runner._worker_init for the measurement
    # (development ledger SS23-25/32 + SS24). Grid solves never run in the
    # parent (R17): x64 is set only in the workers, before their JAX import.
    _os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
    log.info("compute.mode=gpu: forced XLA_PYTHON_CLIENT_ALLOCATOR="
             "platform + XLA_PYTHON_CLIENT_PREALLOCATE=false + "
             "JAX_DEFAULT_MATMUL_PRECISION=%s in parent",
             _os.environ["JAX_DEFAULT_MATMUL_PRECISION"])

    if args.cmd == "run":
        return _run_with_status(cfg, work_dir, log, run_lock)

    from .run_status import EXIT_FAILED, RunContractError
    try:
        return _run_step_command(args, cfg)
    except RunContractError as exc:
        log.error("%s FAILED: %s", args.cmd, exc)
        return EXIT_FAILED


def _run_with_status(cfg: dict, work_dir: Path, log, run_lock) -> int:
    """``tract7dt run`` under the run contract (R12): RUN_STATUS.json is
    "running" while the pipeline runs, then "ok" (exit 0) or "failed" with the
    reason (exit 1). ``run_lock`` (the per-user lock file, or None) is held by
    the caller for the whole call and released when the process exits."""
    from . import run_status
    run_status.reset()
    started = run_status.now_iso()
    run_status.write(work_dir, status="running", started=started)
    try:
        run_pipeline(cfg)
    except BaseException as exc:  # noqa: BLE001 -- recorded, then reported
        error = run_status.describe(exc)
        path = run_status.write(work_dir, status="failed", started=started,
                                exit_code=run_status.EXIT_FAILED, error=error)
        if isinstance(exc, run_status.RunContractError):
            log.error("Run FAILED (%s): %s -- %s", error["kind"],
                      error["message"], path)
        else:
            log.exception("Run FAILED at stage %r -- %s",
                          run_status._REPORT["stage"], path)
        if isinstance(exc, KeyboardInterrupt):
            raise
        return run_status.EXIT_FAILED
    run_status.write(work_dir, status="ok", started=started, exit_code=0)
    return 0


def _run_step_command(args, cfg: dict) -> int:
    if args.cmd == "compute-zp":
        if not bool(cfg.get("zp", {}).get("enabled", True)):
            raise SystemExit(
                "zp.enabled: false in config. Set zp.enabled: true to "
                "run `tract7dt compute-zp`, or omit this subcommand."
            )
        from .zp import compute_zp
        merged_path = Path(cfg["outputs"]["final_catalog"])
        if not merged_path.exists():
            raise SystemExit(f"Catalog not found: {merged_path}. Run 'tract7dt run' first.")
        compute_zp(cfg=cfg, merged_catalog_path=merged_path)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")

    return 0
