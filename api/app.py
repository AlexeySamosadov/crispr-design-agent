"""FastAPI surface for the CRISPR design assistant."""

from __future__ import annotations

import pathlib
from typing import Dict, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer

from crispr_design_agent.training.module import MultiTaskLightningModule


class ScoreRequest(BaseModel):
    sequence: str = Field(..., description="Protein sequence (amino acids).")
    task: str = Field("dms", description="Task head to use (dms, depmap, clinvar).")


class ScoreResponse(BaseModel):
    task: str
    score: float
    confidence: float


class DesignRequest(BaseModel):
    sequence: str
    desired_effect: str = Field(..., description="Plain language description of target phenotype.")
    top_k: int = 5


class DesignSuggestion(BaseModel):
    position: int
    variant: str
    predicted_score: float
    rationale: str


class DesignResponse(BaseModel):
    suggestions: list[DesignSuggestion]


class ModelService:
    def __init__(self, checkpoint: Optional[pathlib.Path] = None, encoder_name: str = "Rostlab/prot_t5_xl_uniref50"):
        self.encoder_name = encoder_name
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)
        if checkpoint and checkpoint.exists():
            self.module = MultiTaskLightningModule.load_from_checkpoint(checkpoint, strict=False)
            self.module.eval()
            self.module.freeze()
        else:
            self.module = None
            self.encoder = AutoModel.from_pretrained(encoder_name, trust_remote_code=True)
            self.encoder.eval()

    @torch.inference_mode()
    def score(self, sequence: str, task: str) -> ScoreResponse:
        if not sequence:
            raise HTTPException(status_code=400, detail="Sequence cannot be empty")
        tokens = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding="max_length",
        )
        if self.module:
            pooled = self.module.forward(tokens["input_ids"], tokens["attention_mask"])
            head = self.module.heads.get(task)
            if head is None:
                raise HTTPException(status_code=404, detail=f"Unknown task {task}")
            logits = head(pooled).squeeze(-1)
        else:
            outputs = self.encoder(**tokens)
            pooled = outputs.last_hidden_state.mean(dim=1)
            logits = torch.tanh(pooled.mean(dim=1))
        score = torch.sigmoid(logits).item()
        return ScoreResponse(task=task, score=score, confidence=score)

    def design(self, request: DesignRequest) -> DesignResponse:
        sequence = request.sequence
        if len(sequence) < 10:
            raise HTTPException(status_code=400, detail="Sequence too short for design suggestions.")
        midpoint = len(sequence) // 2
        suggestions = []
        for offset in range(request.top_k):
            pos = min(len(sequence) - 1, midpoint + offset)
            ref_aa = sequence[pos]
            variant = f"{ref_aa}{pos}{ref_aa}"
            suggestions.append(
                DesignSuggestion(
                    position=int(pos),
                    variant=variant,
                    predicted_score=0.5,
                    rationale=f"Heuristic suggestion targeting position {pos} for {request.desired_effect}",
                )
            )
        return DesignResponse(suggestions=suggestions)


def create_app(checkpoint_path: Optional[str] = None) -> FastAPI:
    service = ModelService(pathlib.Path(checkpoint_path) if checkpoint_path else None)
    app = FastAPI(title="CRISPR Design Agent")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/score", response_model=ScoreResponse)
    def score(request: ScoreRequest):
        return service.score(request.sequence, request.task)

    @app.post("/design", response_model=DesignResponse)
    def design(request: DesignRequest):
        return service.design(request)

    return app


app = create_app()
