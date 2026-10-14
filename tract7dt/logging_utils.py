from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


DEFAULT_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def setup_logging(logging_cfg: dict[str, Any] | None = None, *, force: bool = False) -> None:
    cfg = logging_cfg or {}
    level_name = str(cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = str(cfg.get("format", DEFAULT_LOG_FORMAT))

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    file_path = cfg.get("file", None)
    if file_path:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(p, mode="a"))

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=force)

