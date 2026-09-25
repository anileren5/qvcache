"""Small utilities shared by the sample benchmark package."""

from __future__ import annotations

import logging
import os
from typing import Optional


def setupLogging(level: str = "INFO") -> None:
    """Configure root logging once."""

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def ensureDir(path: str) -> None:
    """Create a directory if missing."""

    os.makedirs(path, exist_ok=True)


def normalizePath(path: str) -> str:
    """Return an absolute, normalized path."""

    return os.path.abspath(os.path.expanduser(path))


def parseBool(value: Optional[str], default: bool = False) -> bool:
    """Parse common boolean strings."""

    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    return default
