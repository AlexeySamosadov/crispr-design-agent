"""Preprocessing functions converting raw datasets into model-ready tables."""

from __future__ import annotations

import gzip
import logging
import pathlib
from typing import List, Optional

import pandas as pd
from Bio import SeqIO

from ..utils.io import ensure_dir

LOGGER = logging.getLogger(__name__)


def prepare_uniprot_sequences(fasta_path: pathlib.Path, output_path: pathlib.Path) -> Optional[pathlib.Path]:
    if not fasta_path.exists():
        LOGGER.warning("UniProt FASTA %s not found. Skipping sequence extraction.", fasta_path)
        return None
    records = []
    with gzip.open(fasta_path, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            desc = record.description
            gene = _extract_gene_symbol(desc)
            records.append(
                {
                    "uniprot_id": record.id.split("|")[1] if "|" in record.id else record.id,
                    "gene_symbol": gene,
                    "sequence": str(record.seq),
                }
            )
    df = pd.DataFrame.from_records(records)
    ensure_dir(output_path.parent)
    df.to_parquet(output_path, index=False)
    LOGGER.info("Wrote %d UniProt sequences", len(df))
    return output_path


def prepare_mavedb_tables(input_dir: pathlib.Path, output_path: pathlib.Path) -> Optional[pathlib.Path]:
    tables = []
    for file in input_dir.glob("*.tsv*"):
        try:
            df = pd.read_table(file)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to parse %s: %s", file, exc)
            continue
        seq_col = _first_existing(df, ["sequence", "wt_sequence", "wildtype_sequence"])
        effect_col = _first_existing(df, ["score", "effect", "fitness", "scaled_score"])
        variant_col = _first_existing(df, ["variant", "mutant", "mutation", "aa_substitution"])
        if not (seq_col and effect_col and variant_col):
            LOGGER.warning("File %s missing expected columns. Skipping.", file)
            continue
        subset = df[[seq_col, variant_col, effect_col]].rename(
            columns={seq_col: "sequence", variant_col: "variant", effect_col: "effect"}
        )
        subset["source_file"] = file.name
        tables.append(subset)
    if not tables:
        LOGGER.warning("No DMS TSV files found under %s", input_dir)
        return None
    result = pd.concat(tables, ignore_index=True)
    ensure_dir(output_path.parent)
    result.to_parquet(output_path, index=False)
    LOGGER.info("Wrote %s with %d rows", output_path, len(result))
    return output_path


def prepare_clinvar_table(input_file: pathlib.Path, output_path: pathlib.Path) -> Optional[pathlib.Path]:
    if not input_file.exists():
        LOGGER.warning("ClinVar file %s missing", input_file)
        return None
    df = pd.read_table(input_file, sep="\t", low_memory=False)
    df = df.rename(columns={"ClinicalSignificance": "clinical_significance", "GeneSymbol": "gene_symbol"})
    df["is_pathogenic"] = df["clinical_significance"].str.contains("Pathogenic", case=False, na=False).astype(int)
    keep_cols = ["GeneSymbol", "Name", "ClinicalSignificance", "is_pathogenic", "Protein_change"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    subset = df[keep_cols].rename(
        columns={
            "GeneSymbol": "gene_symbol",
            "Name": "variant",
            "Protein_change": "protein_change",
        }
    )
    ensure_dir(output_path.parent)
    subset.to_parquet(output_path, index=False)
    LOGGER.info("Wrote ClinVar table with %d rows", len(subset))
    return output_path


def prepare_depmap_tables(
    effect_file: pathlib.Path,
    expression_file: pathlib.Path,
    uniprot_table: pathlib.Path,
    output_path: pathlib.Path,
) -> Optional[pathlib.Path]:
    if not (effect_file.exists() and expression_file.exists()):
        LOGGER.warning("DepMap inputs missing (effect: %s, expression: %s)", effect_file, expression_file)
        return None
    effect_df = pd.read_csv(effect_file)
    effect_df = effect_df.rename(columns={"Gene": "gene_symbol", "CERES": "gene_effect"})
    expr_df = pd.read_csv(expression_file)
    expr_df = expr_df.rename(columns={"Gene": "gene_symbol"})
    merged = effect_df.merge(expr_df, on=["DepMap_ID", "gene_symbol"], suffixes=("_effect", "_expr"))
    if uniprot_table.exists():
        seq_df = pd.read_parquet(uniprot_table)
        merged = merged.merge(seq_df, on="gene_symbol", how="left")
    else:
        merged["sequence"] = None
    ensure_dir(output_path.parent)
    merged.to_parquet(output_path, index=False)
    LOGGER.info("Wrote DepMap merged table with %d rows", len(merged))
    return output_path


def _extract_gene_symbol(description: str) -> Optional[str]:
    tokens = description.split()
    for token in tokens:
        if token.startswith("GN="):
            return token.split("=", 1)[1]
    return None


def _first_existing(df: pd.DataFrame, columns: List[str]) -> Optional[str]:
    for name in columns:
        if name in df.columns:
            return name
    return None
