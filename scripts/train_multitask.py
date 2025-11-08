#!/usr/bin/env python3
"""Train the multitask CRISPR design model."""

from __future__ import annotations

import argparse
import logging
import pathlib
from typing import Dict

import torch
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from crispr_design_agent.training.data_module import MultiTaskDataModule, TaskConfig
from crispr_design_agent.training.module import MultiTaskLightningModule
from crispr_design_agent.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default="configs/model/multitask.yaml", help="Path to YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-task row cap for quick experiments.")
    return parser.parse_args()


def load_config(path: pathlib.Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_task_configs(config: Dict) -> list[TaskConfig]:
    tasks = []
    for name, spec in config["data"]["tasks"].items():
        tasks.append(TaskConfig(name=name, file=spec["file"], target=spec["target"], problem_type=spec["type"]))
    return tasks


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_config(args.config)
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
    module = MultiTaskLightningModule(config)
    checkpoint_dir = pathlib.Path(trainer_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="multitask-{step:06d}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        )
    ]
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        precision=trainer_cfg["precision"],
        max_steps=trainer_cfg["max_steps"],
        gradient_clip_val=trainer_cfg["gradient_clip_val"],
        accumulate_grad_batches=trainer_cfg["grad_accum"],
        log_every_n_steps=50,
        callbacks=callbacks,
        logger=CSVLogger(save_dir="logs", name="multitask"),
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
