from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(level: str, directory: str, session_id: str) -> logging.Logger:
    log_dir = Path(directory)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("local_proactive_cli_agent")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / f"session-{session_id}.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

