# Experiment Tracking

This document describes how to use experiment tracking with Weights & Biases (W&B) and MLflow to monitor training runs, compare experiments, and track model performance.

## Overview

The training script supports three logging backends:
1. **CSV Logger** (always enabled): Lightweight local logging
2. **Weights & Biases**: Cloud-based experiment tracking with rich visualizations
3. **MLflow**: Self-hosted or cloud experiment tracking

## Quick Start

### Using CSV Logger (Default)

No configuration needed - CSV logs are automatically saved to `logs/multitask/`:

```bash
python scripts/train_multitask.py --config configs/model/multitask.yaml
```

### Using Weights & Biases

1. **Install and login:**
   ```bash
   pip install wandb
   wandb login
   ```

2. **Enable in config:**
   ```yaml
   tracking:
     use_wandb: true
     wandb_project: crispr-design
     wandb_entity: your-team-name  # optional
     wandb_log_model: false  # set to true to upload model artifacts
   ```

3. **Run training:**
   ```bash
   python scripts/train_multitask.py \
     --config configs/model/multitask.yaml \
     --experiment-name "baseline-v1" \
     --tags experiment baseline protT5
   ```

4. **View results:**
   Visit https://wandb.ai to view your experiments

### Using MLflow

1. **Enable in config:**
   ```yaml
   tracking:
     use_mlflow: true
     mlflow_tracking_uri: file:./mlruns  # or http://mlflow-server:5000
     mlflow_experiment: crispr-design
   ```

2. **Run training:**
   ```bash
   python scripts/train_multitask.py \
     --config configs/model/multitask.yaml \
     --experiment-name "baseline-v1"
   ```

3. **View results:**
   ```bash
   mlflow ui --backend-store-uri ./mlruns
   # Open http://localhost:5000
   ```

## Configuration Reference

### Tracking Section

```yaml
tracking:
  # Weights & Biases
  use_wandb: false                      # Enable W&B logging
  wandb_project: crispr-design          # W&B project name
  wandb_entity: null                    # W&B team/username (optional)
  wandb_log_model: false                # Upload model checkpoints to W&B

  # MLflow
  use_mlflow: false                     # Enable MLflow logging
  mlflow_tracking_uri: file:./mlruns   # MLflow tracking server URI
  mlflow_experiment: crispr-design      # MLflow experiment name

  # General
  experiment_name: multitask-baseline   # Default experiment name
```

### Training Section Additions

```yaml
training:
  # ... other settings ...
  log_every_n_steps: 50                 # Logging frequency
  val_check_interval: null              # Validation frequency (null = every epoch)
  early_stopping: false                 # Enable early stopping
  early_stop_patience: 5                # Early stop patience (epochs)
  save_top_k: 3                         # Number of best checkpoints to keep
```

## Command Line Options

```bash
python scripts/train_multitask.py \
  --config configs/model/multitask.yaml \
  --experiment-name "my-experiment" \
  --tags tag1 tag2 tag3 \
  --disable-tracking  # Disable W&B/MLflow, use CSV only
```

**Options:**
- `--experiment-name`: Override experiment name from config
- `--tags`: Space-separated tags for filtering experiments
- `--disable-tracking`: Force disable W&B/MLflow (use CSV only)

## Logged Metrics

### Training Metrics
- `train_loss`: Overall training loss
- `train_dms_loss`: DMS task loss
- `train_dms_mae`: DMS mean absolute error
- `train_depmap_loss`: DepMap task loss
- `train_depmap_mae`: DepMap mean absolute error
- `train_clinvar_loss`: ClinVar task loss
- `train_clinvar_prob_mean`: ClinVar predicted probability mean

### Validation Metrics
- `val_loss`: Overall validation loss
- `val_dms_loss`, `val_dms_mae`: DMS validation metrics
- `val_depmap_loss`, `val_depmap_mae`: DepMap validation metrics
- `val_clinvar_loss`, `val_clinvar_prob_mean`: ClinVar validation metrics

### System Metrics
- Learning rate (tracked by `LearningRateMonitor`)
- GPU memory usage (W&B only)
- Training time per step

## Example Workflows

### Comparing Different Architectures

```bash
# Baseline sequence-only model
python scripts/train_multitask.py \
  --config configs/model/multitask.yaml \
  --experiment-name "sequence-only" \
  --tags architecture baseline

# Multimodal with structural features
python scripts/train_multitask.py \
  --config configs/model/multimodal.yaml \
  --experiment-name "multimodal-v1" \
  --tags architecture multimodal structure
```

### Hyperparameter Sweeps with W&B

```bash
# Create sweep config
cat > sweep.yaml <<EOF
program: scripts/train_multitask.py
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  config:
    value: configs/model/multitask.yaml
  learning_rate:
    min: 1e-5
    max: 5e-4
  batch_size:
    values: [2, 4, 8]
  grad_accum:
    values: [4, 8, 16]
EOF

# Initialize and run sweep
wandb sweep sweep.yaml
wandb agent <sweep-id>
```

### MLflow Model Registry

```python
import mlflow
from crispr_design_agent.training.module import MultiTaskLightningModule

# Load best checkpoint
model = MultiTaskLightningModule.load_from_checkpoint("models/checkpoints/best.ckpt")

# Register model
with mlflow.start_run():
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name="crispr-multitask",
    )
```

## Best Practices

### Experiment Organization

1. **Use descriptive names:**
   ```bash
   --experiment-name "protT5-xl-dms-only-lr3e5"
   ```

2. **Tag consistently:**
   ```bash
   --tags model-version dataset-config training-phase
   ```

3. **Version your configs:**
   Commit config changes before running experiments

### Resource Management

1. **Disable tracking for quick tests:**
   ```bash
   --disable-tracking --limit 100
   ```

2. **Use CSV logger for CI/CD:**
   Set `use_wandb: false` and `use_mlflow: false` in config

3. **Clean up old runs:**
   ```bash
   # W&B
   wandb sync --clean

   # MLflow
   mlflow gc --backend-store-uri ./mlruns
   ```

### Team Collaboration

1. **Share W&B projects:**
   - Create team workspace
   - Use `wandb_entity: team-name`
   - Share experiment links

2. **Self-host MLflow:**
   ```bash
   # Start MLflow server
   mlflow server \
     --backend-store-uri postgresql://user:pass@localhost/mlflow \
     --default-artifact-root s3://mlflow-artifacts/ \
     --host 0.0.0.0 \
     --port 5000

   # Update config
   mlflow_tracking_uri: http://mlflow-server:5000
   ```

## Troubleshooting

### W&B Not Logging

**Issue:** Experiments not appearing in W&B dashboard

**Solutions:**
1. Check login: `wandb login`
2. Verify project name matches W&B
3. Check network connectivity
4. Review logs for authentication errors

### MLflow Connection Errors

**Issue:** Cannot connect to MLflow tracking server

**Solutions:**
1. Verify tracking URI is correct
2. Check server is running: `mlflow ui`
3. Test connection:
   ```python
   import mlflow
   mlflow.set_tracking_uri("file:./mlruns")
   print(mlflow.list_experiments())
   ```

### Out of Memory with Logging

**Issue:** Training crashes with OOM when logging is enabled

**Solutions:**
1. Reduce `log_every_n_steps`
2. Disable model artifact logging: `wandb_log_model: false`
3. Use gradient checkpointing in model config

## References

- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [PyTorch Lightning Loggers](https://lightning.ai/docs/pytorch/stable/extensions/logging.html)
