#!/usr/bin/env python3
"""Fetch curated datasets needed for the CRISPR design agent."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crispr_design_agent.data.downloaders import download_dataset
from crispr_design_agent.data.sources import list_available_sources
from crispr_design_agent.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default="configs/dataset/default.yaml", help="YAML config.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subset of dataset keys to download. Defaults to config->sources.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional metadata limit for heavy APIs.")
    parser.add_argument("--fetch-payload", action="store_true", help="Also download raw TSV/FASTA payloads when available.")
    parser.add_argument("--list", action="store_true", help="List available dataset keys and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    if args.list:
        for key, meta in list_available_sources().items():
            print(f"{key:>18s} | {meta.description} ({meta.short_name})")
        return
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = pathlib.Path(config["output_dir"])
    sources: List[str] = args.datasets or config["sources"]
    for dataset_key in sources:
        print(f"==> {dataset_key}")
        download_dataset(
            dataset_key,
            output_dir=output_dir,
            limit=args.limit or config.get("limit"),
            fetch_payload=args.fetch_payload or config.get("fetch_payload", False),
        )


if __name__ == "__main__":
    main()
