"""Shared project paths, including locally configured dataset storage."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_datasets_root() -> Path:
    """Return the dataset directory configured for this machine.

    ``PHD_NET_DATASETS`` takes precedence. By default, datasets live in the
    repository's ``datasets`` directory. A sibling ``datasets`` directory is
    also recognized for compatibility with the previous repository layout.
    """
    configured = os.environ.get("PHD_NET_DATASETS")
    if configured:
        return Path(configured).expanduser().resolve()

    repository_datasets = PROJECT_ROOT / "datasets"
    if repository_datasets.is_dir():
        return repository_datasets

    legacy_datasets = PROJECT_ROOT.parent / "datasets"
    if legacy_datasets.is_dir():
        return legacy_datasets

    return repository_datasets
