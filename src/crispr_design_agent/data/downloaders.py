"""Download utilities for curated bio datasets."""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Iterable, List, Optional

import requests

from .sources import DatasetSource, get_source
from ..utils.io import download_file, ensure_dir, write_json

LOGGER = logging.getLogger(__name__)


def download_dataset(
    key: str,
    output_dir: os.PathLike[str] | str,
    *,
    limit: Optional[int] = None,
    fetch_payload: bool = False,
) -> pathlib.Path:
    """Download metadata (and optionally payload) for a dataset."""
    source = get_source(key)
    dataset_dir = ensure_dir(pathlib.Path(output_dir) / source.short_name.replace("/", "_"))
    if source.requires_auth:
        _write_manual_placeholder(source, dataset_dir)
        return dataset_dir

    if key == "mavedb":
        return _download_mavedb(source, dataset_dir, limit=limit, fetch_payload=fetch_payload)
    if key == "uniprot_sprot":
        return _download_simple_file(source, dataset_dir, filename="uniprot_sprot.fasta.gz")
    if key == "alphafold":
        return _download_simple_file(source, dataset_dir, filename="alphafold_human_v4.tar")
    if key == "clinvar":
        return _download_simple_file(source, dataset_dir, filename="clinvar_variant_summary.txt.gz")

    LOGGER.warning("No automated handler for %s. Creating metadata stub.", key)
    _write_manual_placeholder(source, dataset_dir)
    return dataset_dir


def _download_simple_file(source: DatasetSource, dataset_dir: pathlib.Path, filename: str) -> pathlib.Path:
    if not source.download_url:
        raise ValueError(f"{source.name} missing download URL")
    destination = dataset_dir / filename
    if destination.exists():
        LOGGER.info("File %s already exists; skipping download.", destination)
        return destination
    download_file(source.download_url, destination)
    write_json(
        {
            "dataset": source.name,
            "url": source.download_url,
            "license": source.license,
            "notes": source.notes,
        },
        dataset_dir / "metadata.json",
    )
    return destination


def _download_mavedb(
    source: DatasetSource,
    dataset_dir: pathlib.Path,
    *,
    limit: Optional[int],
    fetch_payload: bool,
) -> pathlib.Path:
    manifest_path = dataset_dir / "manifest.json"
    params = {
        "paginate": "false",
        "format": "json",
    }
    LOGGER.info("Fetching MaveDB manifest ...")
    response = requests.get(source.download_url, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    results: List[dict] = payload.get("results") or payload
    if limit:
        results = results[:limit]
    simplified: List[dict] = []
    for entry in results:
        simplified.append(
            {
                "urn": entry.get("urn") or entry.get("uuid") or entry.get("id"),
                "title": entry.get("title"),
                "experiment_type": entry.get("experiment_type"),
                "variant_count": entry.get("variant_count"),
                "score_set_url": entry.get("url"),
                "download_url": _extract_mavedb_download(entry),
            }
        )
    write_json({"dataset": source.name, "items": simplified}, manifest_path)
    if fetch_payload:
        for item in simplified:
            url = item.get("download_url")
            if not url:
                LOGGER.warning("No downloadable asset for %s", item.get("urn"))
                continue
            fname = dataset_dir / f"{item['urn']}.tsv.gz"
            if fname.exists():
                LOGGER.info("Skipping existing %s", fname.name)
                continue
            try:
                download_file(url, fname)
            except Exception as exc:  # pragma: no cover - warn but continue
                LOGGER.warning("Failed downloading %s: %s", url, exc)
    return manifest_path


def _extract_mavedb_download(entry: dict) -> Optional[str]:
    downloads = entry.get("downloads") or entry.get("links") or []
    if isinstance(downloads, dict):
        downloads = downloads.values()
    for candidate in downloads:
        if isinstance(candidate, dict):
            href = candidate.get("href") or candidate.get("url")
        else:
            href = str(candidate)
        if href and href.endswith((".tsv", ".tsv.gz", ".csv", ".csv.gz")):
            return href
    return None


def _write_manual_placeholder(source: DatasetSource, dataset_dir: pathlib.Path) -> None:
    """Create metadata instructions for datasets that require manual acceptance/auth."""
    write_json(
        {
            "dataset": source.name,
            "requires_auth": source.requires_auth,
            "details_url": source.details_url,
            "notes": source.notes or "Download manually and place files in this directory.",
            "license": source.license,
        },
        dataset_dir / "metadata.json",
    )
