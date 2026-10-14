from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


_LOG = logging.getLogger("tract7dt.sample_data")
_CHUNK_SIZE = 1024 * 1024  # 1 MiB
_MIB = 1024 * 1024
_FITS_SUFFIXES = (".fits", ".fit", ".fits.fz")
_BAND_RE = re.compile(r"(m\d{3})", re.IGNORECASE)

# The published sample data: release v1.0.0 of the sample-data repository, with its checksum.
DEFAULT_SAMPLE_REPO = "seoldh99/tract7dt_sample_data"
DEFAULT_SAMPLE_TAG = "v1.0.0"
DEFAULT_SAMPLE_ASSET = "tract7dt_sample_data.tar.gz"
DEFAULT_SAMPLE_SHA256 = "91c906114458cc3372241cedcca61d9786aaf1391a40c136a717a02e5349ac94"


@dataclass(frozen=True)
class DownloadResult:
    dataset_dir: Path
    downloaded_file: Path
    image_list_file: Path


def github_release_asset_url(repo: str, tag: str, asset: str) -> str:
    repo = repo.strip().strip("/")
    if "/" not in repo:
        raise ValueError("repo must be in '<owner>/<repo>' format")
    if not tag.strip():
        raise ValueError("tag cannot be empty")
    if not asset.strip():
        raise ValueError("asset cannot be empty")
    return f"https://github.com/{repo}/releases/download/{tag}/{asset}"


def default_sample_data_url() -> str | None:
    repo = os.environ.get("TRACT7DT_SAMPLE_REPO", DEFAULT_SAMPLE_REPO).strip()
    tag = os.environ.get("TRACT7DT_SAMPLE_TAG", DEFAULT_SAMPLE_TAG).strip()
    asset = os.environ.get("TRACT7DT_SAMPLE_ASSET", DEFAULT_SAMPLE_ASSET).strip()
    if not (repo and tag and asset):
        return None
    return github_release_asset_url(repo, tag, asset)


def default_sample_data_sha256() -> str | None:
    value = os.environ.get("TRACT7DT_SAMPLE_SHA256", DEFAULT_SAMPLE_SHA256).strip()
    return value or None


def _infer_filename(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(parsed.path).name
    return name or "sample_data.bin"


def _dataset_name_from_filename(filename: str) -> str:
    p = Path(filename)
    name = p.name
    for ext in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".zip", ".tar", ".gz"):
        if name.endswith(ext):
            return name[: -len(ext)] or "sample-data"
    return p.stem or "sample-data"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _verify_sha256(path: Path, expected: str | None) -> None:
    if not expected:
        return
    expected_norm = expected.strip().lower()
    got = _sha256(path)
    if got != expected_norm:
        raise RuntimeError(
            f"SHA256 mismatch for {path}. expected={expected_norm}, got={got}"
        )


def _download_url(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "tract7dt/0.2"})

    with urllib.request.urlopen(req) as resp, tmp.open("wb") as f:
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None
        written = 0
        next_log_pct = 10
        next_log_bytes = 50 * _MIB
        logged_completion = False
        use_inline_progress = sys.stderr.isatty()
        _inline_width = max(40, shutil.get_terminal_size((80, 24)).columns - 1)
        last_render = 0.0
        last_pct = -1

        def _progress_bar(pct: int, width: int = 24) -> str:
            clamped = max(0, min(100, pct))
            filled = int((clamped * width) / 100)
            return "#" * filled + "-" * (width - filled)

        while True:
            chunk = resp.read(_CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)
            if total_bytes:
                pct = int((100 * written) / total_bytes)
                if use_inline_progress:
                    now = time.monotonic()
                    # Refresh in-place roughly 5 times/sec or on completion.
                    if pct != last_pct and (now - last_render >= 0.2 or pct >= 100):
                        msg = (
                            f"Download [{_progress_bar(pct)}] {pct:3d}% "
                            f"({written / _MIB:.1f}/{total_bytes / _MIB:.1f} MiB)"
                        )
                        sys.stderr.write("\r" + msg[:_inline_width].ljust(_inline_width))
                        sys.stderr.flush()
                        last_render = now
                        last_pct = pct
                        if pct >= 100:
                            logged_completion = True
                elif pct >= next_log_pct:
                    _LOG.info(
                        "Download [%s] %3d%% (%.1f/%.1f MiB)",
                        _progress_bar(pct),
                        pct,
                        written / _MIB,
                        total_bytes / _MIB,
                    )
                    next_log_pct += 10
                    if pct >= 100:
                        logged_completion = True
            elif written >= next_log_bytes:
                if use_inline_progress:
                    msg = f"Download [unknown size] {written / _MIB:.1f} MiB downloaded"
                    sys.stderr.write("\r" + msg[:_inline_width].ljust(_inline_width))
                    sys.stderr.flush()
                else:
                    _LOG.info("Download [unknown size] %.1f MiB downloaded", written / _MIB)
                next_log_bytes += 50 * _MIB

        if total_bytes and not logged_completion:
            if use_inline_progress:
                msg = (
                    f"Download [{_progress_bar(100)}] 100% "
                    f"({written / _MIB:.1f}/{total_bytes / _MIB:.1f} MiB)"
                )
                sys.stderr.write("\r" + msg[:_inline_width].ljust(_inline_width))
                sys.stderr.flush()
            else:
                _LOG.info(
                    "Download [%s] 100%% (%.1f/%.1f MiB)",
                    _progress_bar(100),
                    written / _MIB,
                    total_bytes / _MIB,
                )

        if use_inline_progress:
            sys.stderr.write("\n")
            sys.stderr.flush()

    tmp.replace(out_path)


def _archive_roots_zip(path: Path) -> set[str]:
    roots: set[str] = set()
    with zipfile.ZipFile(path, "r") as zf:
        for member in zf.namelist():
            clean = member.strip("/")
            if clean:
                roots.add(clean.split("/", 1)[0])
    return roots


def _archive_roots_tar(path: Path) -> set[str]:
    roots: set[str] = set()
    with tarfile.open(path, "r:*") as tf:
        for member in tf.getmembers():
            clean = member.name.strip("/")
            if clean:
                roots.add(clean.split("/", 1)[0])
    return roots


def _extract_archive(archive: Path, extract_root: Path) -> Path:
    extract_root.mkdir(parents=True, exist_ok=True)
    root = extract_root.resolve()

    def _assert_safe_path(candidate: Path) -> None:
        resolved = candidate.resolve()
        if not str(resolved).startswith(str(root)):
            raise RuntimeError(f"Unsafe archive path detected: {candidate}")

    use_inline = sys.stderr.isatty()
    _inline_width = max(40, shutil.get_terminal_size((80, 24)).columns - 1)
    last_non_tty_log = 0.0

    def _progress_bar(done: int, total: int, width: int = 24) -> str:
        if total <= 0:
            return "-" * width
        frac = max(0.0, min(1.0, float(done) / float(total)))
        filled = int(round(frac * width))
        return "#" * filled + "-" * (width - filled)

    def _emit_extract_progress(done: int, total: int, *, force_log: bool = False) -> None:
        nonlocal last_non_tty_log
        msg = (
            f"Unpack [{_progress_bar(done, total)}] "
            f"{done}/{total} files"
        )
        if use_inline:
            sys.stderr.write("\r" + msg[:_inline_width].ljust(_inline_width))
            sys.stderr.flush()
        else:
            now = time.monotonic()
            if force_log or (now - last_non_tty_log) >= 10.0:
                _LOG.info(msg)
                last_non_tty_log = now

    lower = archive.name.lower()
    roots: set[str]
    if lower.endswith(".zip"):
        roots = _archive_roots_zip(archive)
        with zipfile.ZipFile(archive, "r") as zf:
            members = zf.namelist()
            for member in members:
                _assert_safe_path(extract_root / member)
            total = len(members)
            _emit_extract_progress(0, total, force_log=True)
            for i, member in enumerate(members, 1):
                zf.extract(member, extract_root)
                _emit_extract_progress(i, total, force_log=(i == total))
    elif lower.endswith(".tar") or lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        roots = _archive_roots_tar(archive)
        with tarfile.open(archive, "r:*") as tf:
            members = tf.getmembers()
            for member in members:
                _assert_safe_path(extract_root / member.name)
            total = len(members)
            _emit_extract_progress(0, total, force_log=True)
            for i, member in enumerate(members, 1):
                tf.extract(member, extract_root)
                _emit_extract_progress(i, total, force_log=(i == total))
    else:
        raise RuntimeError(
            "Sample data archive must be one of .zip/.tar/.tar.gz/.tgz"
        )

    if use_inline:
        sys.stderr.write("\n")
        sys.stderr.flush()

    if len(roots) == 1:
        ds_name = next(iter(roots))
        ds = (extract_root / ds_name).resolve()
        if ds.exists() and ds.is_dir():
            return ds

    inferred_name = _dataset_name_from_filename(archive.name)
    guessed = (extract_root / inferred_name).resolve()
    if guessed.exists() and guessed.is_dir():
        return guessed

    return extract_root.resolve()


def _planned_dataset_dir(archive: Path, extract_root: Path) -> Path:
    lower = archive.name.lower()
    roots: set[str] = set()
    if lower.endswith(".zip"):
        roots = _archive_roots_zip(archive)
    elif lower.endswith(".tar") or lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        roots = _archive_roots_tar(archive)

    if len(roots) == 1:
        return (extract_root / next(iter(roots))).resolve()

    return (extract_root / _dataset_name_from_filename(archive.name)).resolve()


def _is_fits_path(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(sfx) for sfx in _FITS_SUFFIXES)


def _band_key(band: str) -> int:
    return int(band[1:])


def _extract_band(path: Path) -> str | None:
    m = _BAND_RE.search(path.name)
    return m.group(1).lower() if m else None


def _standardize_image_names(dataset_dir: Path) -> list[Path]:
    images_dir = dataset_dir / "images"
    search_root = images_dir if images_dir.exists() else dataset_dir
    fits_files = [
        p.resolve()
        for p in search_root.rglob("*")
        if p.is_file() and _is_fits_path(p)
    ]
    if not fits_files:
        raise RuntimeError(f"No FITS files found under {search_root}")

    band_to_file: dict[str, Path] = {}
    for p in fits_files:
        band = _extract_band(p)
        if not band:
            continue
        if band in band_to_file:
            raise RuntimeError(f"Duplicate FITS files found for band {band}")
        band_to_file[band] = p

    if not band_to_file:
        raise RuntimeError("Could not infer any band names from FITS filenames")

    ordered_bands = sorted(band_to_file.keys(), key=_band_key)
    out_files: list[Path] = []

    for band in ordered_bands:
        src = band_to_file[band]
        dst = (images_dir / f"7DT_UDS_{band}.fits").resolve()
        images_dir.mkdir(parents=True, exist_ok=True)
        if src != dst:
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
        out_files.append(dst)

    return out_files


def _write_image_list(dataset_dir: Path, image_paths: list[Path]) -> Path:
    out = dataset_dir / "image_list.txt"
    out.write_text("".join(f"{p.resolve()}\n" for p in image_paths))
    return out


def download_sample_data(
    *,
    path_to_download: Path,
    target_dataset_dir: Path | None = None,
    overwrite: bool = False,
) -> DownloadResult:
    url = default_sample_data_url()
    if not url:
        raise RuntimeError(
            "No default sample-data source configured. "
            "Set TRACT7DT_SAMPLE_REPO/TAG/ASSET (or constants in sample_data.py)."
        )
    checksum = default_sample_data_sha256()
    download_root = path_to_download.expanduser().resolve()
    download_root.mkdir(parents=True, exist_ok=True)
    filename = _infer_filename(url)
    archive_path = download_root / filename

    # Pre-download check: fail fast if target already exists.
    if target_dataset_dir is not None:
        _pre_check = target_dataset_dir.expanduser().resolve()
    else:
        _pre_check = (download_root / _dataset_name_from_filename(filename)).resolve()
    if _pre_check.exists() and not overwrite:
        raise RuntimeError(
            f"Target dataset directory already exists: {_pre_check}. "
            "Use --force to overwrite."
        )

    if archive_path.exists():
        if archive_path.is_dir():
            shutil.rmtree(archive_path)
        else:
            archive_path.unlink()
        _LOG.debug("Removed pre-existing archive path: %s", archive_path)

    _LOG.info("Downloading sample data from: %s", url)
    _download_url(url, archive_path)
    _LOG.debug("Downloaded file: %s", archive_path)

    try:
        _verify_sha256(archive_path, checksum)
        if checksum:
            _LOG.debug("SHA256 verified")

        final_target = (
            target_dataset_dir.expanduser().resolve()
            if target_dataset_dir is not None
            else _planned_dataset_dir(archive_path, download_root)
        )
        if target_dataset_dir is not None:
            _LOG.debug("Requested target dataset directory (--dir): %s", final_target)
        if final_target.exists():
            if not overwrite:
                raise RuntimeError(
                    f"Target dataset directory already exists: {final_target}. "
                    "Use --force to overwrite."
                )
            if final_target.is_dir():
                shutil.rmtree(final_target)
            else:
                final_target.unlink()
            _LOG.info("Removed existing target directory due to --force: %s", final_target)

        dataset_dir = _extract_archive(archive_path, download_root)
        _LOG.debug("Extracted archive into staging root: %s", download_root)
        _LOG.debug("Detected extracted dataset directory: %s", dataset_dir)

        if target_dataset_dir is not None:
            target_dir = target_dataset_dir.expanduser().resolve()
            if dataset_dir != target_dir:
                if dataset_dir == download_root:
                    raise RuntimeError(
                        "Cannot rename extracted dataset directory because archive "
                        "does not contain a single top-level folder."
                    )
                if target_dir.exists():
                    raise RuntimeError(f"Target dataset directory already exists: {target_dir}")
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                _LOG.debug(
                    "Moving extracted dataset directory to requested target: %s -> %s",
                    dataset_dir,
                    target_dir,
                )
                shutil.move(str(dataset_dir), str(target_dir))
                dataset_dir = target_dir
            _LOG.debug("Dataset directory ready at requested path: %s", dataset_dir)
        else:
            _LOG.debug("Dataset directory ready at default path: %s", dataset_dir)

        image_paths = _standardize_image_names(dataset_dir)
        image_list_file = _write_image_list(dataset_dir, image_paths)
        _LOG.debug("Generated image list: %s", image_list_file)

        return DownloadResult(
            dataset_dir=dataset_dir,
            downloaded_file=archive_path,
            image_list_file=image_list_file,
        )
    finally:
        if archive_path.exists():
            if archive_path.is_dir():
                shutil.rmtree(archive_path)
            else:
                archive_path.unlink()
            _LOG.debug("Removed temporary archive path: %s", archive_path)
