"""Multimodal Lightning module combining sequence and structural features."""

from __future__ import annotations

import pathlib
from typing import Dict, List, Optional

import torch
from lightning.pytorch import LightningModule
from torch import nn
from transformers import AutoModel


class StructuralEncoder(nn.Module):
    """Graph neural network encoder for protein structural features."""

    def __init__(
        self,
        contact_embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize structural encoder.

        Args:
            contact_embedding_dim: Dimension for edge feature embeddings
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super().__init__()
        self.contact_embedding_dim = contact_embedding_dim
        self.hidden_dim = hidden_dim

        self.edge_encoder = nn.Sequential(
            nn.Linear(3, contact_embedding_dim),
            nn.ReLU(),
            nn.Linear(contact_embedding_dim, contact_embedding_dim),
        )

        self.geom_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gnn_layers = nn.ModuleList(
            [
                GNNLayer(
                    node_dim=hidden_dim,
                    edge_dim=contact_embedding_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        local_geometry: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Forward pass through structural encoder.

        Args:
            edge_index: [2, num_edges] edge connectivity
            edge_attr: [num_edges, 3] edge features (distances + derived)
            local_geometry: [num_nodes, 5] local geometric features
            num_nodes: Number of nodes/residues

        Returns:
            node_embeddings: [num_nodes, hidden_dim] structural embeddings
        """
        edge_features = self.edge_encoder(edge_attr)

        node_features = self.geom_encoder(local_geometry)
        node_features = torch.cat(
            [node_features, torch.zeros(num_nodes, self.hidden_dim // 2, device=node_features.device)],
            dim=-1,
        )

        for gnn_layer, norm_layer in zip(self.gnn_layers, self.norm_layers):
            node_features_new = gnn_layer(node_features, edge_index, edge_features)
            node_features = norm_layer(node_features + node_features_new)

        return self.output_projection(node_features)


class GNNLayer(nn.Module):
    """Single graph neural network layer with edge features."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.message_fn = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )
        self.update_fn = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Message passing step.

        Args:
            node_features: [num_nodes, node_dim]
            edge_index: [2, num_edges]
            edge_features: [num_edges, edge_dim]

        Returns:
            updated_features: [num_nodes, node_dim]
        """
        src, dst = edge_index[0], edge_index[1]
        messages = torch.cat([node_features[src], node_features[dst], edge_features], dim=-1)
        messages = self.message_fn(messages)

        aggregated = torch.zeros_like(node_features)
        aggregated.index_add_(0, dst, messages)

        updated = torch.cat([node_features, aggregated], dim=-1)
        return self.update_fn(updated)


class MultimodalLightningModule(LightningModule):
    """Lightning module combining sequence and structural encoders."""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        model_cfg = config["model"]

        self.encoder_name = model_cfg["encoder"]
        self.sequence_encoder = AutoModel.from_pretrained(self.encoder_name, trust_remote_code=True)
        seq_hidden_size = self.sequence_encoder.config.hidden_size

        if model_cfg.get("freeze_encoder", False):
            for param in self.sequence_encoder.parameters():
                param.requires_grad = False

        self.use_structure = model_cfg.get("use_structure", False)
        if self.use_structure:
            struct_cfg = model_cfg.get("structure", {})
            self.structural_encoder = StructuralEncoder(
                contact_embedding_dim=struct_cfg.get("contact_embedding_dim", 64),
                hidden_dim=struct_cfg.get("hidden_dim", 128),
                num_layers=struct_cfg.get("num_layers", 3),
                dropout=struct_cfg.get("dropout", 0.1),
            )
            struct_hidden = struct_cfg.get("hidden_dim", 128)

            self.fusion_layer = nn.Sequential(
                nn.Linear(seq_hidden_size + struct_hidden, seq_hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
        else:
            self.structural_encoder = None
            self.fusion_layer = None

        self.head_configs = config["heads"]
        final_hidden = seq_hidden_size
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(final_hidden),
                    nn.Linear(final_hidden, 1),
                )
                for name in self.head_configs
            }
        )

        self.loss_fns = {
            "regression": nn.MSELoss(),
            "classification": nn.BCEWithLogitsLoss(),
        }
        self.training_cfg = config["training"]

    def forward(
        self,
        input_ids,
        attention_mask,
        structural_features: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Forward pass with optional structural features.

        Args:
            input_ids: Tokenized sequence
            attention_mask: Attention mask
            structural_features: Optional dict with edge_index, edge_attr, local_geometry

        Returns:
            pooled representation
        """
        outputs = self.sequence_encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_hidden = outputs.last_hidden_state
        seq_pooled = seq_hidden.mean(dim=1)

        if self.use_structure and structural_features is not None:
            struct_repr = self.structural_encoder(
                edge_index=structural_features["edge_index"],
                edge_attr=structural_features["edge_attr"],
                local_geometry=structural_features["local_geometry"],
                num_nodes=structural_features["num_nodes"],
            )
            struct_pooled = struct_repr.mean(dim=0, keepdim=True)

            if struct_pooled.size(0) != seq_pooled.size(0):
                struct_pooled = struct_pooled.expand(seq_pooled.size(0), -1)

            combined = torch.cat([seq_pooled, struct_pooled], dim=-1)
            pooled = self.fusion_layer(combined)
        else:
            pooled = seq_pooled

        return pooled

    def training_step(self, batch, batch_idx):
        structural_features = batch.get("structural_features", None)
        pooled = self.forward(batch["input_ids"], batch["attention_mask"], structural_features)
        loss, metrics = self._compute_task_losses(pooled, batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch["labels"]))
        for key, value in metrics.items():
            self.log(f"train_{key}", value, prog_bar=False, on_step=True, batch_size=len(batch["labels"]))
        return loss

    def validation_step(self, batch, batch_idx):
        structural_features = batch.get("structural_features", None)
        pooled = self.forward(batch["input_ids"], batch["attention_mask"], structural_features)
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.training_cfg.get("max_steps", 1000))
        return {"optimizer": optim, "lr_scheduler": scheduler}
