# Dataset Blueprint

| Key               | What it provides                            | Download instructions                                                                 | Notes |
|-------------------|----------------------------------------------|---------------------------------------------------------------------------------------|-------|
| `mavedb`          | Deep Mutational Scanning scores (DMS).       | `python scripts/fetch_data.py --datasets mavedb --fetch-payload`                      | Filter manifest via `--limit` to keep metadata light. |
| `uniprot_sprot`   | Curated protein FASTA + annotations.         | `python scripts/fetch_data.py --datasets uniprot_sprot`                               | Powers sequence lookups + UniProt ↔ AlphaFold mapping. |
| `alphafold`       | 3D structure tarballs per proteome.          | `python scripts/fetch_data.py --datasets alphafold`                                   | Default grabs Homo sapiens proteome; swap tarball for other species. |
| `depmap`          | CRISPR knockout gene effect matrices.        | Manual download (DepMap portal → Accept license → gene effect CSV)                    | Place `CRISPR_gene_effect.csv` in `data/raw/DepMap_CRISPR/`. |
| `depmap_expression` | Matched RNA-seq TPM (log1p).               | Manual download with DepMap account                                                   | Place `OmicsExpressionProteinCodingGenesTPMLogp1.csv` in `data/raw/DepMap_CRISPR/`. |
| `clinvar`         | Clinical variant annotations.                | `python scripts/fetch_data.py --datasets clinvar`                                     | Tab-delimited summary, zipped. |

## Storage conventions

```
data/raw/
├── MaveDB/
├── UniProtKB_Swiss-Prot/
├── AlphaFold_DB/
├── DepMap_CRISPR/
└── ClinVar/
```

`download_dataset` automatically creates these directories based on `DatasetSource.short_name`.

## Processing outputs

| File                         | Producer                              | Columns (subset)                               |
|------------------------------|---------------------------------------|------------------------------------------------|
| `data/processed/dms.parquet` | `scripts/preprocess.py` (MaveDB TSV)  | `sequence`, `variant`, `effect`, `source_file` |
| `data/processed/uniprot_sequences.parquet` | same | `uniprot_id`, `gene_symbol`, `sequence` |
| `data/processed/depmap.parquet` | same | `DepMap_ID`, `gene_symbol`, `gene_effect`, `sequence`, ... |
| `data/processed/clinvar.parquet` | same | `gene_symbol`, `variant`, `is_pathogenic`, ... |

These names align with `configs/model/multitask.yaml` so the training script can find them without manual edits.

## Refresh cadence

- **Weekly**: `mavedb`, `clinvar` (data updates frequently).
- **Monthly**: `depmap`, `depmap_expression` (quarterly releases; poll portal).
- **Quarterly**: `uniprot_sprot`, `alphafold` (major releases).

Automate via cron (or GitHub Actions) invoking `scripts/fetch_data.py` and `scripts/preprocess.py`. Publish manifests + hashed artifacts in object storage for reproducibility.
