# Architecture Overview

## Model stack

1. **Encoder**: ProtT5-XL (`Rostlab/prot_t5_xl_uniref50`) running in fp16 with optional LoRA adapters (configs/model/multitask.yaml → `model.lora`).
2. **Heads**:
   - `dms`: regression predicting quantitative mutation effect.
   - `depmap`: regression predicting cell-line viability impact of gene knockout.
   - `clinvar`: binary classification for pathogenicity labels.
3. **Training**: Lightning `MultiTaskLightningModule` that pools token embeddings, routes batch elements to their respective heads, and optimizes per-task losses.

## Data flow

```
fetch_data.py  --> data/raw/<Dataset>/
preprocess.py  --> data/processed/*.parquet
train_multitask.py --> models/checkpoints/*.ckpt
api/app.py (FastAPI) --> REST surface for score/design
```

## Automation hooks

- Config-driven dataset registry (`configs/dataset/default.yaml`).
- YAML-configurable training hyperparameters + task file paths.
- Feature extraction script decoupled from training for offline embedding caches.
- API loads latest checkpoint when available, otherwise falls back to heuristic encoder scoring.

## Extensibility

- Add new datasets by extending `src/crispr_design_agent/data/sources.py` + `data/preprocess.py`.
- Introduce structure-aware features by dropping Parquet files (per residue) and updating the Lightning module to accept graph inputs.
- Swap encoders (ESM-2, ProtBERT) via config without touching code.
