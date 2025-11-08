# Agent Playbook

This document enumerates the exact commands an automation agent should execute to keep the CRISPR design system up to date.

## 0. Environment bootstrap

```bash
cd crispr-design-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Data refresh

```bash
python scripts/fetch_data.py --config configs/dataset/default.yaml --fetch-payload --limit 100
```
- For DepMap datasets, pause and request manual files if `data/raw/DepMap_CRISPR/` lacks `CRISPR_gene_effect.csv` or `OmicsExpressionProteinCodingGenesTPMLogp1.csv`.

## 2. Preprocess tables

```bash
python scripts/preprocess.py \
  --depmap-effect-file CRISPR_gene_effect.csv \
  --depmap-expression-file OmicsExpressionProteinCodingGenesTPMLogp1.csv
```

## 3. Optional feature extraction

```bash
python scripts/embed_features.py --input data/processed/dms.parquet --limit 50000 --batch-size 2
```

## 4. Training

```bash
python scripts/train_multitask.py --config configs/model/multitask.yaml
```
- Logs land in `logs/multitask/`.
- Checkpoints in `models/checkpoints/`.

## 5. Serving

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8080
```
- Attach checkpoint path via `CRISPR_AGENT_CHECKPOINT=... uvicorn api.app:app` when extending `create_app` call.

## 6. Release checklist

1. Verify `models/checkpoints/` contains a fresh file whose metric improved.
2. Export `requirements.txt` hash and dataset manifest (from `data/raw/*/metadata.json`).
3. Update changelog/README with date + dataset versions.
4. Tag repository and push container image for deployment.
