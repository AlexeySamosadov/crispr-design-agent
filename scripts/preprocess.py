#!/usr/bin/env python3
"""Convert raw datasets into normalized parquet tables."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crispr_design_agent.data import preprocess
from crispr_design_agent.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=pathlib.Path, default=pathlib.Path("data/raw"), help="Raw data directory.")
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=pathlib.Path("data/processed"), help="Processed data directory."
    )
    parser.add_argument("--depmap-effect-file", type=str, default="CRISPR_gene_effect.csv")
    parser.add_argument("--depmap-expression-file", type=str, default="OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    raw_dir = args.raw_dir
    processed_dir = args.output_dir

    uniprot_dir = raw_dir / "UniProtKB_Swiss-Prot"
    uniprot_fasta = uniprot_dir / "uniprot_sprot.fasta.gz"
    preprocess.prepare_uniprot_sequences(uniprot_fasta, processed_dir / "uniprot_sequences.parquet")

    mavedb_dir = raw_dir / "MaveDB"
    preprocess.prepare_mavedb_tables(mavedb_dir, processed_dir / "dms.parquet")

    clinvar_dir = raw_dir / "ClinVar"
    preprocess.prepare_clinvar_table(clinvar_dir / "clinvar_variant_summary.txt.gz", processed_dir / "clinvar.parquet")

    depmap_dir = raw_dir / "DepMap_CRISPR"
    preprocess.prepare_depmap_tables(
        depmap_dir / args.depmap_effect_file,
        depmap_dir / args.depmap_expression_file,
        processed_dir / "uniprot_sequences.parquet",
        processed_dir / "depmap.parquet",
    )


if __name__ == "__main__":
    main()
