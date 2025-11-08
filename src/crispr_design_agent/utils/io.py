"""Utility helpers for IO, logging, and hashing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
from typing import Any, Dict

import requests
from requests import Response
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)


def ensure_dir(path: os.PathLike[str] | str) -> pathlib.Path:
    """Create directory (and parents) if it does not yet exist."""
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _raise_for_status(response: Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:  # pragma: no cover - pass through context
        msg = f"Request failed with status {response.status_code}: {response.text[:200]}"
        raise RuntimeError(msg) from exc


def download_file(url: str, destination: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> pathlib.Path:
    """Stream download a file with a progress bar."""
    destination = pathlib.Path(destination)
    ensure_dir(destination.parent)
    LOGGER.info("Downloading %s -> %s", url, destination)
    with requests.get(url, stream=True, timeout=60) as response:
        _raise_for_status(response)
        total = int(response.headers.get("content-length", 0))
        progress = tqdm(total=total, unit="B", unit_scale=True, desc=f"GET {destination.name}")
        with destination.open("wb") as out_handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    out_handle.write(chunk)
                    progress.update(len(chunk))
        progress.close()
    return destination


def write_json(data: Dict[str, Any], path: os.PathLike[str] | str) -> pathlib.Path:
    """Dump a JSON file with UTF-8 encoding."""
    path = pathlib.Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    return path


def hash_file(path: os.PathLike[str] | str, algo: str = "sha256") -> str:
    """Return SHA hash of a file."""
    path = pathlib.Path(path)
    digest = hashlib.new(algo)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
