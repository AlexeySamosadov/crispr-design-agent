"""Lightning DataModule for multitask protein + CRISPR datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


@dataclass
class TaskConfig:
    name: str
    file: str
    target: str
    problem_type: str  # regression | classification


class MultiTaskDataset(Dataset):
    """Concatenates several datasets and keeps track of their task name."""

    def __init__(
        self,
        task_configs: List[TaskConfig],
        tokenizer: AutoTokenizer,
        *,
        split: str,
        val_split: float,
        max_length: int,
        limit: Optional[int],
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        frames: List[pd.DataFrame] = []
        for cfg in task_configs:
            df = pd.read_parquet(cfg.file)
            if limit:
                df = df.head(limit)
            df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            cutoff = max(1, int(len(df) * (1 - val_split)))
            if split == "train":
                df = df.iloc[:cutoff]
            else:
                df = df.iloc[cutoff:]
            df = df.assign(task=cfg.name, target=df[cfg.target], problem_type=cfg.problem_type)
            frames.append(df[["sequence", "target", "task", "problem_type"]])
        self.frame = pd.concat(frames, ignore_index=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        row = self.frame.iloc[idx]
        tokens = self.tokenizer(
            row["sequence"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.squeeze(0) for k, v in tokens.items()}
        target = float(row["target"])
        problem_type = row["problem_type"]
        if problem_type == "classification":
            label = torch.tensor(target, dtype=torch.float32)
        else:
            label = torch.tensor(target, dtype=torch.float32)
        inputs["labels"] = label
        inputs["task_name"] = row["task"]
        inputs["problem_type"] = problem_type
        return inputs


class MultiTaskDataModule(LightningDataModule):
    def __init__(
        self,
        task_configs: List[TaskConfig],
        tokenizer_name: str,
        *,
        max_length: int,
        batch_size: int,
        num_workers: int,
        val_split: float,
        limit: Optional[int],
    ) -> None:
        super().__init__()
        self.task_configs = task_configs
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.val_split = val_split
        self.limit = limit
        self._tokenizer: Optional[AutoTokenizer] = None

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=True)
        return self._tokenizer

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = MultiTaskDataset(
            self.task_configs,
            tokenizer=self.tokenizer,
            split="train",
            val_split=self.val_split,
            max_length=self.max_length,
            limit=self.limit,
        )
        self.val_dataset = MultiTaskDataset(
            self.task_configs,
            tokenizer=self.tokenizer,
            split="val",
            val_split=self.val_split,
            max_length=self.max_length,
            limit=self.limit,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
