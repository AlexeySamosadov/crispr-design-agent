"""Lightning module for multitask protein + CRISPR prediction."""

from __future__ import annotations

from typing import Dict, List

import torch
from lightning.pytorch import LightningModule
from torch import nn
from transformers import AutoModel


class MultiTaskLightningModule(LightningModule):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        model_cfg = config["model"]
        self.encoder_name = model_cfg["encoder"]
        self.encoder = AutoModel.from_pretrained(self.encoder_name, trust_remote_code=True)
        hidden_size = self.encoder.config.hidden_size
        if model_cfg.get("freeze_encoder", False):
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.head_configs = config["heads"]
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 1),
            )
            for name in self.head_configs
        })
        self.loss_fns = {
            "regression": nn.MSELoss(),
            "classification": nn.BCEWithLogitsLoss(),
        }
        self.training_cfg = config["training"]

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        return pooled

    def training_step(self, batch, batch_idx):
        pooled = self.forward(batch["input_ids"], batch["attention_mask"])
        loss, metrics = self._compute_task_losses(pooled, batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch["labels"]))
        for key, value in metrics.items():
            self.log(f"train_{key}", value, prog_bar=False, on_step=True, batch_size=len(batch["labels"]))
        return loss

    def validation_step(self, batch, batch_idx):
        pooled = self.forward(batch["input_ids"], batch["attention_mask"])
        loss, metrics = self._compute_task_losses(pooled, batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=len(batch["labels"]))
        for key, value in metrics.items():
            self.log(f"val_{key}", value, prog_bar=False, on_epoch=True, batch_size=len(batch["labels"]))
        return loss

    def _compute_task_losses(self, pooled, batch):
        task_names: List[str] = batch["task_name"]
        problem_types: List[str] = batch["problem_type"]
        labels = batch["labels"]
        total_loss = torch.tensor(0.0, device=self.device)
        metrics = {}
        for task, head in self.heads.items():
            indices = [idx for idx, name in enumerate(task_names) if name == task]
            if not indices:
                continue
            idx_tensor = torch.tensor(indices, device=pooled.device, dtype=torch.long)
            task_repr = pooled.index_select(0, idx_tensor)
            task_labels = labels.index_select(0, idx_tensor)
            logits = head(task_repr).squeeze(-1)
            problem_type = self.head_configs[task]["type"]
            loss_fn = self.loss_fns[problem_type]
            task_loss = loss_fn(logits, task_labels)
            total_loss = total_loss + task_loss
            metrics[f"{task}_loss"] = task_loss.detach()
            if problem_type == "regression":
                metrics[f"{task}_mae"] = torch.nn.functional.l1_loss(logits, task_labels).detach()
            else:
                preds = torch.sigmoid(logits)
                metrics[f"{task}_prob_mean"] = preds.mean().detach()
        return total_loss, metrics

    def configure_optimizers(self):
        from torch.optim import AdamW

        params = self.parameters()
        optim = AdamW(
            params,
            lr=self.training_cfg["learning_rate"],
            weight_decay=self.training_cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=self.training_cfg.get("max_steps", 1000)
        )
        return {"optimizer": optim, "lr_scheduler": scheduler}
