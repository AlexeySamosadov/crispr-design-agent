# CRISPR Design Agent

Open-source scaffold for building an AI assistant that recommends gene edits, protein mutations, and CRISPR strategies using public datasets.

## Highlights

- 🧬 **Dataset registry** covering MaveDB, UniProt, AlphaFold DB, DepMap CRISPR, and ClinVar with scripted fetch + preprocessing steps.
- 🧠 **Multitask training stack** (DMS effect → regression, DepMap viability → regression, ClinVar pathogenicity → classification) built on Lightning + ProtT5 encoder.
- ⚙️ **Feature generators** for protein embeddings and structural context placeholders.
- 🌐 **FastAPI gateway** wrapping the multitask model for `score` and `design` endpoints.
- 🔁 **Agent-friendly instructions** to automate refreshing data, re-training, and serving updates.

## Repository layout

```
crispr-design-agent/
├── api/                  # FastAPI app
├── configs/              # Dataset + model configs
├── data/                 # Raw/processed placeholders
├── docs/                 # Extended instructions
├── scripts/              # CLI utilities (fetch, preprocess, train, embed)
├── src/crispr_design_agent/
│   ├── data/             # Dataset registry + preprocessing
│   ├── training/         # Lightning module + datamodule
│   └── utils/            # Shared helpers
```

## Quickstart

1. **Environment**
   ```bash
   cd crispr-design-agent
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Fetch datasets** (metadata + downloadable assets where available)
   ```bash
   python scripts/fetch_data.py --list                      # view keys
   python scripts/fetch_data.py                             # uses configs/dataset/default.yaml
   python scripts/fetch_data.py --fetch-payload --datasets mavedb uniprot_sprot
   ```
   DepMap files require creating an account and placing CSVs in `data/raw/DepMap_CRISPR/`.

3. **Preprocess into normalized tables**
   ```bash
   python scripts/preprocess.py \
     --depmap-effect-file CRISPR_gene_effect.csv \
     --depmap-expression-file OmicsExpressionProteinCodingGenesTPMLogp1.csv
   ```
   Outputs Parquet files in `data/processed/` (`dms.parquet`, `depmap.parquet`, `clinvar.parquet`, `uniprot_sequences.parquet`).

4. **Generate embeddings** (optional warm start for downstream models)
   ```bash
   python scripts/embed_features.py --input data/processed/dms.parquet --limit 1000
   ```

5. **Train multitask model**
   ```bash
   python scripts/train_multitask.py --config configs/model/multitask.yaml --limit 20000
   ```
   Edit the config to point at your processed files, batch sizes, and LoRA/T5 checkpoints.

6. **Serve API**
   ```bash
   uvicorn api.app:app --host 0.0.0.0 --port 8000
   # POST /score   {"sequence": "MEEPQ...", "task": "clinvar"}
   # POST /design  {...}
   ```

## Docs for automation agents

- `docs/datasets.md` — canonical sources, licensing, and expected file names per dataset.
- `scripts/fetch_data.py` — idempotent CLI for cron/CI to refresh public assets.
- `scripts/preprocess.py` — single entry point for data normalization; wire this into scheduled jobs post download.
- `scripts/train_multitask.py` — accepts `--limit` to run smoke tests, otherwise trains to convergence using config hyperparameters.
- `api/app.py` — FastAPI app importable by `uvicorn` or embedding inside larger orchestrators.

## Next steps

1. Implement full structural featurization (PDB/AlphaFold contact graphs) and plug into `training/module.py`.
2. Add evaluation notebooks in `notebooks/` for benchmarking on held-out DMS/ClinVar sets.
3. Integrate experiment tracking (Weights & Biases or MLflow) inside `scripts/train_multitask.py`.
4. Extend API with batch scoring and audit logs before exposing to paying users.

## Disclaimer

This scaffold does **not** ship pre-trained medical models. Validate every prediction experimentally and comply with local biosafety regulations before deploying edits.
