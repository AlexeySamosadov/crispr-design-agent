#!/usr/bin/env python3
"""Extract structural features from PDB/AlphaFold structures."""

from __future__ import annotations

import argparse
import logging
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm

from crispr_design_agent.features.structural import extract_structure_features
from crispr_design_agent.utils.io import ensure_dir
from crispr_design_agent.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdb-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing PDB files",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("features/structures"),
        help="Output directory for extracted features",
    )
    parser.add_argument(
        "--is-alphafold",
        action="store_true",
        help="Whether structures are AlphaFold predictions",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=8.0,
        help="Distance threshold in Angstroms for contact definition",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdb",
        help="Glob pattern for PDB files",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of structures to process",
    )
    parser.add_argument(
        "--manifest-output",
        type=pathlib.Path,
        default=None,
        help="Optional CSV file to save processing manifest",
    )
    return parser.parse_args()


def process_single_structure(
    pdb_path: pathlib.Path,
    output_dir: pathlib.Path,
    is_alphafold: bool,
    distance_threshold: float,
) -> tuple[str, bool, Optional[str]]:
    """
    Process a single PDB file and extract features.

    Returns:
        Tuple of (pdb_name, success, error_message)
    """
    try:
        extract_structure_features(
            pdb_path=pdb_path,
            is_alphafold=is_alphafold,
            distance_threshold=distance_threshold,
            output_dir=output_dir,
        )
        return pdb_path.stem, True, None
    except Exception as e:
        logger.error(f"Failed to process {pdb_path.name}: {e}")
        return pdb_path.stem, False, str(e)


def main() -> None:
    args = parse_args()
    configure_logging()

    if not args.pdb_dir.exists():
        raise FileNotFoundError(f"PDB directory not found: {args.pdb_dir}")

    pdb_files = sorted(args.pdb_dir.glob(args.pattern))
    if args.limit:
        pdb_files = pdb_files[: args.limit]

    if not pdb_files:
        logger.warning(f"No PDB files found in {args.pdb_dir} with pattern {args.pattern}")
        return

    logger.info(f"Found {len(pdb_files)} PDB files to process")
    ensure_dir(args.output_dir)

    results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_single_structure,
                pdb_path,
                args.output_dir,
                args.is_alphafold,
                args.distance_threshold,
            ): pdb_path
            for pdb_path in pdb_files
        }

        with tqdm(total=len(pdb_files), desc="Extracting features") as pbar:
            for future in as_completed(futures):
                pdb_name, success, error = future.result()
                results.append(
                    {
                        "pdb_name": pdb_name,
                        "success": success,
                        "error": error,
                    }
                )
                pbar.update(1)

    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    logger.info(f"Processing complete: {successful} successful, {failed} failed")

    if args.manifest_output:
        df = pd.DataFrame(results)
        df.to_csv(args.manifest_output, index=False)
        logger.info(f"Saved processing manifest to {args.manifest_output}")


if __name__ == "__main__":
    main()
