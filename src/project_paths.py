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


def get_ffpp_root() -> Path:
    """Return the FF++ dataset directory configured for this machine.

    Checks ``PHD_NET_FFPP`` environment variable, then
    ``get_datasets_root() / "FF_Dataset"``, then ``Path("D:/FF_Dataset")``.
    """
    configured = os.environ.get("PHD_NET_FFPP")
    if configured:
        return Path(configured).expanduser().resolve()

    ds_ffpp = get_datasets_root() / "FF_Dataset"
    if ds_ffpp.is_dir():
        return ds_ffpp

    host_d = Path("D:/FF_Dataset")
    if host_d.is_dir():
        return host_d

    return ds_ffpp


def get_celebdf_root() -> Path:
    """Return the Celeb-DF v2 dataset directory configured for this machine.

    Checks ``PHD_NET_CELEBDF`` environment variable, then candidate roots
    (preferring one that contains ``List_of_testing_videos.txt``).
    """
    configured = os.environ.get("PHD_NET_CELEBDF")
    if configured:
        return Path(configured).expanduser().resolve()

    candidates = [
        Path("/datasets_celebdf"),
        get_datasets_root() / "celebdf_v2",
        get_datasets_root() / "celebdf",
        Path("D:/celeb-df_v2"),
    ]
    for cand in candidates:
        if cand.is_dir() and (cand / "List_of_testing_videos.txt").is_file():
            return cand

    for cand in candidates:
        if cand.is_dir():
            return cand

    return get_datasets_root() / "celebdf_v2"
