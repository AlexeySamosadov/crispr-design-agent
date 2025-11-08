"""Extended FastAPI with batch scoring, audit logging, and rate limiting."""

from __future__ import annotations

import os
import pathlib
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from transformers import AutoTokenizer

from crispr_design_agent.api.audit import AuditLogger
from crispr_design_agent.training.module import MultiTaskLightningModule

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


class ScoreRequest(BaseModel):
    sequence: str = Field(..., description="Protein sequence (amino acids).", min_length=1, max_length=10000)
    task: str = Field("dms", description="Task head to use (dms, depmap, clinvar).")
    user_id: Optional[str] = Field(None, description="Optional user identifier for tracking.")

    @validator("sequence")
    def validate_sequence(cls, v):
        if not v.replace(" ", "").isalpha():
            raise ValueError("Sequence must contain only amino acid letters")
        return v.upper().replace(" ", "")

    @validator("task")
    def validate_task(cls, v):
        allowed_tasks = ["dms", "depmap", "clinvar"]
        if v not in allowed_tasks:
            raise ValueError(f"Task must be one of {allowed_tasks}")
        return v


class ScoreResponse(BaseModel):
    request_id: str
    task: str
    score: float
    confidence: float
    metadata: Dict = {}


class BatchScoreRequest(BaseModel):
    sequences: List[str] = Field(..., description="List of protein sequences.", min_items=1, max_items=100)
    task: str = Field("dms", description="Task head to use (dms, depmap, clinvar).")
    user_id: Optional[str] = Field(None, description="Optional user identifier.")

    @validator("sequences")
    def validate_sequences(cls, v):
        cleaned = []
        for seq in v:
            if not seq.replace(" ", "").isalpha():
                raise ValueError("All sequences must contain only amino acid letters")
            cleaned.append(seq.upper().replace(" ", ""))
        return cleaned


class BatchScoreResponse(BaseModel):
    request_id: str
    task: str
    results: List[Dict]
    metadata: Dict = {}


class DesignRequest(BaseModel):
    sequence: str = Field(..., min_length=10)
    desired_effect: str = Field(..., description="Plain language description of target phenotype.")
    top_k: int = Field(5, ge=1, le=20)
    user_id: Optional[str] = None


class DesignSuggestion(BaseModel):
    position: int
    variant: str
    predicted_score: float
    rationale: str


class DesignResponse(BaseModel):
    request_id: str
    suggestions: List[DesignSuggestion]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class StatsResponse(BaseModel):
    total_requests: int
    endpoints: Dict[str, int]
    status_codes: Dict[str, int]
    avg_duration_ms: float
    errors: int


class ModelService:
    """Model service with batch scoring support."""

    def __init__(
        self,
        checkpoint: Optional[pathlib.Path] = None,
        encoder_name: str = "Rostlab/prot_t5_xl_uniref50",
        device: str = "auto",
    ):
        self.encoder_name = encoder_name
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if checkpoint and checkpoint.exists():
            self.module = MultiTaskLightningModule.load_from_checkpoint(checkpoint, strict=False)
            self.module.eval()
            self.module.to(self.device)
            self.module.freeze()
            self.available_tasks = list(self.module.heads.keys())
        else:
            self.module = None
            self.available_tasks = []

    @torch.inference_mode()
    def score_single(self, sequence: str, task: str) -> tuple[float, float]:
        """Score a single sequence."""
        if not sequence:
            raise ValueError("Sequence cannot be empty")

        if self.module and task not in self.available_tasks:
            raise ValueError(f"Unknown task {task}. Available: {self.available_tasks}")

        tokens = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding="max_length",
        ).to(self.device)

        if self.module:
            pooled = self.module.forward(tokens["input_ids"], tokens["attention_mask"])
            head = self.module.heads[task]
            logits = head(pooled).squeeze(-1)
        else:
            raise RuntimeError("No model loaded")

        score = torch.sigmoid(logits).item()
        return score, score

    @torch.inference_mode()
    def score_batch(self, sequences: List[str], task: str, batch_size: int = 8) -> List[tuple[float, float]]:
        """Score multiple sequences in batches."""
        if not sequences:
            raise ValueError("Sequences list cannot be empty")

        if self.module and task not in self.available_tasks:
            raise ValueError(f"Unknown task {task}. Available: {self.available_tasks}")

        results = []

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]

            tokens = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding="max_length",
            ).to(self.device)

            if self.module:
                pooled = self.module.forward(tokens["input_ids"], tokens["attention_mask"])
                head = self.module.heads[task]
                logits = head(pooled).squeeze(-1)
                scores = torch.sigmoid(logits).cpu().numpy()

                for score in scores:
                    results.append((float(score), float(score)))
            else:
                raise RuntimeError("No model loaded")

        return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    app.state.audit_logger = AuditLogger(
        log_dir=pathlib.Path(os.getenv("AUDIT_LOG_DIR", "logs/audit")),
        enable_file_logging=True,
        hash_sequences=os.getenv("HASH_SEQUENCES", "true").lower() == "true",
    )
    yield
    # Shutdown
    pass


def create_app(checkpoint_path: Optional[str] = None) -> FastAPI:
    """Create extended FastAPI application."""
    app = FastAPI(
        title="CRISPR Design Agent - Extended API",
        description="Production-ready API with batch scoring, audit logging, and rate limiting",
        version="2.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Initialize model service
    service = ModelService(
        checkpoint=pathlib.Path(checkpoint_path) if checkpoint_path else None,
        device=os.getenv("DEVICE", "auto"),
    )

    @app.get("/health", response_model=HealthResponse)
    def health():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_loaded=service.module is not None,
            device=service.device,
        )

    @app.post("/score", response_model=ScoreResponse)
    @limiter.limit("100/minute")
    async def score(request_obj: ScoreRequest, req: Request):
        """Score a single sequence."""
        request_id = str(uuid.uuid4())
        audit_logger: AuditLogger = req.app.state.audit_logger

        start_time = audit_logger.log_request(
            request_id=request_id,
            endpoint="/score",
            method="POST",
            request_data=request_obj.dict(),
            user_id=request_obj.user_id,
        )

        try:
            score, confidence = service.score_single(request_obj.sequence, request_obj.task)

            response = ScoreResponse(
                request_id=request_id,
                task=request_obj.task,
                score=score,
                confidence=confidence,
                metadata={"sequence_length": len(request_obj.sequence)},
            )

            audit_logger.log_response(
                request_id=request_id,
                endpoint="/score",
                method="POST",
                request_data=request_obj.dict(),
                response_data=response.dict(),
                status_code=200,
                start_time=start_time,
                user_id=request_obj.user_id,
            )

            return response

        except Exception as e:
            audit_logger.log_response(
                request_id=request_id,
                endpoint="/score",
                method="POST",
                request_data=request_obj.dict(),
                response_data=None,
                status_code=500,
                start_time=start_time,
                user_id=request_obj.user_id,
                error=str(e),
            )
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/batch-score", response_model=BatchScoreResponse)
    @limiter.limit("10/minute")
    async def batch_score(request_obj: BatchScoreRequest, req: Request):
        """Score multiple sequences in a batch."""
        request_id = str(uuid.uuid4())
        audit_logger: AuditLogger = req.app.state.audit_logger

        start_time = audit_logger.log_request(
            request_id=request_id,
            endpoint="/batch-score",
            method="POST",
            request_data=request_obj.dict(),
            user_id=request_obj.user_id,
        )

        try:
            results = service.score_batch(request_obj.sequences, request_obj.task)

            result_dicts = [
                {
                    "sequence_index": idx,
                    "score": score,
                    "confidence": confidence,
                    "sequence_length": len(request_obj.sequences[idx]),
                }
                for idx, (score, confidence) in enumerate(results)
            ]

            response = BatchScoreResponse(
                request_id=request_id,
                task=request_obj.task,
                results=result_dicts,
                metadata={"num_sequences": len(request_obj.sequences)},
            )

            audit_logger.log_response(
                request_id=request_id,
                endpoint="/batch-score",
                method="POST",
                request_data=request_obj.dict(),
                response_data={"num_results": len(result_dicts)},
                status_code=200,
                start_time=start_time,
                user_id=request_obj.user_id,
            )

            return response

        except Exception as e:
            audit_logger.log_response(
                request_id=request_id,
                endpoint="/batch-score",
                method="POST",
                request_data=request_obj.dict(),
                response_data=None,
                status_code=500,
                start_time=start_time,
                user_id=request_obj.user_id,
                error=str(e),
            )
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/stats", response_model=StatsResponse)
    @limiter.limit("10/minute")
    async def get_stats(req: Request):
        """Get API usage statistics."""
        audit_logger: AuditLogger = req.app.state.audit_logger
        stats = audit_logger.get_stats()
        return StatsResponse(**stats)

    return app


# Create app instance
app = create_app(checkpoint_path=os.getenv("CHECKPOINT_PATH"))
