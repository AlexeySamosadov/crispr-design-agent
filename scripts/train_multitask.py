#!/usr/bin/env python3
"""Train the multitask CRISPR design model."""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
from typing import Dict, List, Optional

import torch
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, Logger

from crispr_design_agent.training.data_module import MultiTaskDataModule, TaskConfig
from crispr_design_agent.training.module import MultiTaskLightningModule
from crispr_design_agent.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default="configs/model/multitask.yaml", help="Path to YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-task row cap for quick experiments.")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment name for tracking.")
    parser.add_argument("--tags", type=str, nargs="*", default=None, help="Tags for experiment tracking.")
    parser.add_argument("--disable-tracking", action="store_true", help="Disable experiment tracking (W&B/MLflow).")
    return parser.parse_args()


def load_config(path: pathlib.Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_task_configs(config: Dict) -> list[TaskConfig]:
    tasks = []
    for name, spec in config["data"]["tasks"].items():
        tasks.append(TaskConfig(name=name, file=spec["file"], target=spec["target"], problem_type=spec["type"]))
    return tasks


def setup_experiment_loggers(
    config: Dict,
    experiment_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    disable_tracking: bool = False,
) -> List[Logger]:
    """
    Setup experiment tracking loggers (CSV, W&B, MLflow).

    Args:
        config: Configuration dictionary
        experiment_name: Optional experiment name override
        tags: Optional list of tags for the experiment
        disable_tracking: If True, only use CSV logger

    Returns:
        List of configured loggers
    """
    loggers = [CSVLogger(save_dir="logs", name="multitask")]

    if disable_tracking:
        logger.info("Experiment tracking disabled, using CSV logger only")
        return loggers

    tracking_cfg = config.get("tracking", {})
    exp_name = experiment_name or tracking_cfg.get("experiment_name", "crispr-multitask")

    # Weights & Biases
    if tracking_cfg.get("use_wandb", False):
        try:
            from lightning.pytorch.loggers import WandbLogger

            wandb_project = tracking_cfg.get("wandb_project", "crispr-design")
            wandb_entity = tracking_cfg.get("wandb_entity", None)

            wandb_logger = WandbLogger(
                project=wandb_project,
                entity=wandb_entity,
                name=exp_name,
                tags=tags,
                save_dir="logs",
                log_model=tracking_cfg.get("wandb_log_model", False),
            )

            wandb_logger.experiment.config.update(config)
            loggers.append(wandb_logger)
            logger.info(f"Initialized W&B logger: project={wandb_project}, name={exp_name}")

        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B logger: {e}")

    # MLflow
    if tracking_cfg.get("use_mlflow", False):
        try:
            from lightning.pytorch.loggers import MLFlowLogger

            mlflow_tracking_uri = tracking_cfg.get("mlflow_tracking_uri", "file:./mlruns")
            mlflow_experiment = tracking_cfg.get("mlflow_experiment", "crispr-design")

            mlflow_logger = MLFlowLogger(
                experiment_name=mlflow_experiment,
                tracking_uri=mlflow_tracking_uri,
                run_name=exp_name,
                tags={"tags": ",".join(tags)} if tags else None,
            )

            mlflow_logger.log_hyperparams(config)
            loggers.append(mlflow_logger)
            logger.info(f"Initialized MLflow logger: experiment={mlflow_experiment}, uri={mlflow_tracking_uri}")

        except ImportError:
            logger.warning("mlflow not installed, skipping MLflow logging")
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow logger: {e}")

    return loggers


def setup_callbacks(config: Dict) -> List:
    """
    Setup training callbacks.

    Args:
        config: Configuration dictionary

    Returns:
        List of configured callbacks
    """
    trainer_cfg = config["training"]
    checkpoint_dir = pathlib.Path(trainer_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="multitask-{epoch:02d}-{val_loss:.4f}",
            save_top_k=trainer_cfg.get("save_top_k", 3),
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Early stopping
    if trainer_cfg.get("early_stopping", False):
        early_stop_patience = trainer_cfg.get("early_stop_patience", 5)
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=early_stop_patience,
                mode="min",
                verbose=True,
            )
        )
        logger.info(f"Early stopping enabled with patience={early_stop_patience}")

    return callbacks


def main() -> None:
    args = parse_args()
    configure_logging()

    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Build data module
    task_configs = build_task_configs(config)
    trainer_cfg = config["training"]
    datamodule = MultiTaskDataModule(
        task_configs,
        tokenizer_name=config["model"]["encoder"],
        max_length=config["model"]["max_length"],
        batch_size=trainer_cfg["batch_size"],
        num_workers=config["data"]["num_workers"],
        val_split=config["data"]["val_split"],
        limit=args.limit or config["data"]["train_limit"],
    )

    # Build model
    module = MultiTaskLightningModule(config)

    # Setup experiment tracking
    experiment_loggers = setup_experiment_loggers(
        config,
        experiment_name=args.experiment_name,
        tags=args.tags,
        disable_tracking=args.disable_tracking,
    )

    # Setup callbacks
    callbacks = setup_callbacks(config)

    # Setup trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        precision=trainer_cfg["precision"],
        max_steps=trainer_cfg["max_steps"],
        gradient_clip_val=trainer_cfg["gradient_clip_val"],
        accumulate_grad_batches=trainer_cfg["grad_accum"],
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 50),
        val_check_interval=trainer_cfg.get("val_check_interval", None),
        callbacks=callbacks,
        logger=experiment_loggers,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    logger.info("Starting training...")
    trainer.fit(module, datamodule=datamodule)
    logger.info("Training complete!")

    # Log final metrics
    if trainer.checkpoint_callback:
        best_model_path = trainer.checkpoint_callback.best_model_path
        best_score = trainer.checkpoint_callback.best_model_score
        logger.info(f"Best model: {best_model_path}")
        logger.info(f"Best val_loss: {best_score:.4f}")


if __name__ == "__main__":
    main()
