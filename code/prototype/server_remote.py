#!/usr/bin/env python3
# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
DermApIxel Inference Server -- FastAPI application exposing the
DermApIxelPipeline over HTTP and (optionally) via an async RabbitMQ
consumer.

Endpoints:
    POST /analyze           -- full M1-M4 + SigLIP pipeline (base64 JSON)
    POST /analyze/upload    -- same pipeline, multipart file upload
    POST /analyze/zero-shot -- M6 DermLIP v2 open-vocabulary classification
    POST /analyze/unified   -- M7 merged43+TTA hierarchical L1/L2/L3
    GET  /health            -- liveness probe (VRAM, uptime, model status)
    GET  /stats             -- request/error counters + RabbitMQ worker stats
    GET  /v1/models         -- OpenAI-compatible model listing

Usage:
    # Foreground, no RabbitMQ worker (for dev/testing)
    ENABLE_WORKER=false python dermapixel_server.py

    # Foreground with worker
    export RABBITMQ_URL="amqp://USER:PASS@HOST:5672/"
    python dermapixel_server.py

    # As systemd service
    sudo systemctl start dermapixel-server
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from pipeline import DermApIxelPipeline
from rabbitmq_worker import WorkerStats, run_worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dermapixel")

MODEL_NAME = "dermapixel-v1"

# ─── Global state ────────────────────────────────────────────────────────────
pipeline = DermApIxelPipeline()
worker_stats = WorkerStats()

# Request counters (HTTP + worker combined for /stats)
import threading
_counter_lock = threading.Lock()
_requests_total = 0
_requests_ok = 0
_requests_error = 0
_total_inference_ms = 0.0
_server_start_ts: float = 0.0

# Zero-shot counters
_zero_shot_requests = 0
_zero_shot_errors = 0

# Unified merged43+TTA counters
_unified_requests = 0
_unified_errors = 0


# ─── Pydantic models ─────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    image_base64: str = Field(..., max_length=20_000_000, description="Base64-encoded image (jpg/png, max ~15MB)")
    format: str = Field(default="jpg", description="Image format: jpg|jpeg|png")


class ClassificationItem(BaseModel):
    class_name: str
    probability: float
    rank: int


class SegmentationResult(BaseModel):
    mask_base64: str
    lesion_percent: float
    error: Optional[str] = None


class ConceptItem(BaseModel):
    concept_name: str
    score: float
    present: bool


class SimilarCase(BaseModel):
    filename: str
    similarity: float
    disease: str
    concept: str


class AnalyzeResponse(BaseModel):
    status: str
    inference_time_ms: float
    timings_ms: dict[str, float]
    image_size: Optional[list[int]] = None
    classifications: list[ClassificationItem]
    segmentation: SegmentationResult
    concepts: list[ConceptItem]
    similar_cases: list[SimilarCase]
    annotated_image_base64: str
    error: Optional[str] = None
    # Módulos nuevos (compatibilidad retro: los 3 son opcionales)
    siglip_safety: Optional[dict] = None
    m9_spanderm: Optional[dict] = None
    m10_concepts: Optional[dict] = None
    m4bis_rag_es: Optional[dict] = None

    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    vram_used_gb: float
    rabbitmq_connected: bool
    uptime_seconds: float
    dermlip_available: bool = False
    unified_available: bool = False
    siglip_available: bool = False
    m9_available: bool = False
    m10_available: bool = False
    m4bis_available: bool = False


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "panderm"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class StatsResponse(BaseModel):
    requests_total: int
    requests_ok: int
    requests_error: int
    avg_inference_ms: float
    worker_messages_received: int
    worker_messages_ok: int
    worker_messages_error: int
    worker_last_study_id: Optional[str]
    worker_last_error: Optional[str]
    worker_connected: bool
    zero_shot_requests: int
    zero_shot_errors: int
    unified_requests: int = 0
    unified_errors: int = 0


class ZeroShotRequest(BaseModel):
    image_base64: str = Field(..., max_length=20_000_000, description="Base64-encoded image (jpg/png, max ~15MB)")
    format: str = Field(default="jpg", description="Image format: jpg|jpeg|png")
    class_descriptions: list[str] = Field(
        ..., description="List of candidate class text descriptions"
    )


class ZeroShotPrediction(BaseModel):
    class_name: str
    similarity: float
    rank: int


class ZeroShotResponse(BaseModel):
    predictions: list[ZeroShotPrediction]
    model: str
    timing_ms: float


class UnifiedTopK(BaseModel):
    """One entry of the top-k list (class name + probability)."""
    class_name: str = Field(..., alias="class")
    prob: float

    class Config:
        populate_by_name = True


class UnifiedLevel(BaseModel):
    top_class: str
    top_prob: float
    top3: list[UnifiedTopK]
    all_probs: dict[str, float]


class UnifiedRequest(BaseModel):
    image_base64: str = Field(..., max_length=20_000_000, description="Base64-encoded image (jpg/png, max ~15MB)")


class UnifiedResponse(BaseModel):
    level_3: UnifiedLevel
    level_2: UnifiedLevel
    level_1: UnifiedLevel
    model: str
    tta_augmentations: int
    timing_ms: float


# ─── Image decoding helper ───────────────────────────────────────────────────

def _decode_base64_image(b64: str) -> Image.Image:
    """Decode a base64-encoded image string into a PIL RGB Image.

    Accepts raw base64 or ``data:image/...;base64,...`` URI format.
    Raises HTTPException 400 on invalid input.
    """
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(b64, validate=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid base64: {e}")
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image bytes: {e}")
    return img


def _update_counters(ok: bool, inference_ms: float) -> None:
    global _requests_total, _requests_ok, _requests_error, _total_inference_ms
    with _counter_lock:
        _requests_total += 1
        if ok:
            _requests_ok += 1
        else:
            _requests_error += 1
        _total_inference_ms += inference_ms


# ─── App factory ─────────────────────────────────────────────────────────────

def create_app(enable_worker: bool = True) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _server_start_ts
        _server_start_ts = time.time()

        # Load pipeline synchronously (blocks startup until ready)
        logger.info("Loading DermApIxel pipeline...")
        pipeline.load_all()
        logger.info("Pipeline ready")

        worker_task: Optional[asyncio.Task] = None
        if enable_worker:
            logger.info("Starting RabbitMQ worker task")
            worker_task = asyncio.create_task(
                run_worker(pipeline, worker_stats)
            )
        else:
            logger.info("RabbitMQ worker disabled (ENABLE_WORKER=false)")

        yield

        if worker_task:
            logger.info("Cancelling worker task")
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="DermApIxel Inference Server",
        description="PanDerm M1-M4 pipeline for teledermatology",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5178").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ─── Endpoints ────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Liveness/readiness probe.

        Returns model-load status, VRAM usage (GB), uptime, and
        RabbitMQ connection state.  Used by systemd watchdog and
        the Vue frontend polling loop.
        """
        vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        uptime = time.time() - _server_start_ts if _server_start_ts else 0.0
        return HealthResponse(
            status="ok" if pipeline.models_loaded else "loading",
            models_loaded=pipeline.models_loaded,
            vram_used_gb=round(vram, 2),
            rabbitmq_connected=worker_stats.connected,
            uptime_seconds=round(uptime, 1),
            dermlip_available=pipeline.dermlip_available,
            unified_available=pipeline.unified_available,
            siglip_available=getattr(pipeline, "siglip_available", False),
            m9_available=getattr(pipeline, "m9_available", False),
            m10_available=getattr(pipeline, "m10_available", False),
            m4bis_available=getattr(pipeline, "m4bis_available", False),
        )

    @app.get("/v1/models", response_model=ModelList)
    async def list_models() -> ModelList:
        """OpenAI-compatible model listing (GET /v1/models).

        Returns the primary model plus any optional modules that
        loaded successfully (unified, etc.).
        """
        models = [ModelInfo(id=MODEL_NAME)]
        if pipeline.unified_available:
            models.append(ModelInfo(id="unified_merged43_tta"))
        return ModelList(data=models)

    @app.post("/analyze", response_model=AnalyzeResponse)
    async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
        """Full M1-M4 + SigLIP analysis from base64-encoded image.

        Request body: ``{"image_base64": "...", "format": "jpg"}``
        Response: AnalyzeResponse with classifications, segmentation
        mask, SAE concepts, RAG similar cases, and annotated overlay.
        Latency: ~1.5-2.5 s.
        """
        if not pipeline.models_loaded:
            raise HTTPException(status_code=503, detail="models not loaded yet")

        img = _decode_base64_image(request.image_base64)
        return await _run_analyze(img)

    @app.post("/analyze/upload", response_model=AnalyzeResponse)
    async def analyze_upload(file: UploadFile = File(...)) -> AnalyzeResponse:
        """Full M1-M4 + SigLIP analysis from multipart file upload.

        Same pipeline as POST /analyze but accepts a file upload
        instead of base64 JSON.  Useful for curl/Postman testing.
        """
        if not pipeline.models_loaded:
            raise HTTPException(status_code=503, detail="models not loaded yet")

        contents = await file.read()
        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid image: {e}")
        return await _run_analyze(img)

    @app.get("/stats", response_model=StatsResponse)
    async def stats() -> StatsResponse:
        """Prometheus-style counters for HTTP + RabbitMQ worker.

        Returns request totals, error counts, average inference
        latency, and per-module (zero-shot, unified) breakdowns.
        """
        avg = (
            _total_inference_ms / _requests_total if _requests_total > 0 else 0.0
        )
        return StatsResponse(
            requests_total=_requests_total,
            requests_ok=_requests_ok,
            requests_error=_requests_error,
            avg_inference_ms=round(avg, 2),
            worker_messages_received=worker_stats.messages_received,
            worker_messages_ok=worker_stats.messages_ok,
            worker_messages_error=worker_stats.messages_error,
            worker_last_study_id=worker_stats.last_study_id,
            worker_last_error=worker_stats.last_error,
            worker_connected=worker_stats.connected,
            zero_shot_requests=_zero_shot_requests,
            zero_shot_errors=_zero_shot_errors,
            unified_requests=_unified_requests,
            unified_errors=_unified_errors,
        )

    @app.post("/analyze/zero-shot", response_model=ZeroShotResponse)
    async def zero_shot(request: ZeroShotRequest) -> ZeroShotResponse:
        """M6 DermLIP v2 open-vocabulary zero-shot classification.

        Request body: ``{"image_base64": "...", "class_descriptions": [...]}``
        Returns ranked cosine similarities against each candidate text.
        Latency: ~20 ms.  Requires DermLIP v2 to be loaded.
        """
        global _zero_shot_requests, _zero_shot_errors
        if not pipeline.models_loaded:
            raise HTTPException(status_code=503, detail="models not loaded yet")
        if not pipeline.dermlip_available:
            _zero_shot_errors += 1
            raise HTTPException(
                status_code=503, detail="DermLIP v2 not available (load failed)"
            )
        if not request.class_descriptions:
            _zero_shot_errors += 1
            raise HTTPException(
                status_code=400, detail="class_descriptions must be a non-empty list"
            )

        img = _decode_base64_image(request.image_base64)

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                pipeline.classify_zero_shot,
                img,
                request.class_descriptions,
            )
        except Exception as e:
            logger.error("zero-shot classification failed", exc_info=True)
            _zero_shot_errors += 1
            raise HTTPException(status_code=500, detail=f"zero-shot error: {e}")

        _zero_shot_requests += 1
        return ZeroShotResponse(**result)

    @app.post("/analyze/unified", response_model=UnifiedResponse)
    async def analyze_unified(request: UnifiedRequest) -> UnifiedResponse:
        """L3 merged43 (43 classes) classifier with TTA, returning the
        L3 prediction plus L2 (26 classes) and L1 (4 classes) marginal
        views derived by summing softmax probabilities along the Rosa
        ontology hierarchy. ~1.0–1.5 s/image due to TTA ×5.
        """
        global _unified_requests, _unified_errors
        if not pipeline.models_loaded:
            raise HTTPException(status_code=503, detail="models not loaded yet")
        if not pipeline.unified_available:
            _unified_errors += 1
            raise HTTPException(
                status_code=503,
                detail="Unified merged43 not available (load failed)",
            )

        img = _decode_base64_image(request.image_base64)

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None, pipeline.classify_unified, img
            )
        except Exception as e:
            logger.error("unified classification failed", exc_info=True)
            _unified_errors += 1
            raise HTTPException(status_code=500, detail=f"unified error: {e}")

        _unified_requests += 1
        # Re-pack top3 entries to match the alias-aware Pydantic schema
        for level_key in ("level_3", "level_2", "level_1"):
            top3_raw = result[level_key]["top3"]
            result[level_key]["top3"] = [
                {"class": item["class"], "prob": item["prob"]}
                for item in top3_raw
            ]
        return UnifiedResponse(**result)

    return app


async def _run_analyze(img: Image.Image) -> AnalyzeResponse:
    """Run pipeline.analyze() in a threadpool executor and update counters.

    Offloads the synchronous GPU inference to a thread so the FastAPI
    event loop remains responsive for health checks and other requests.
    """
    loop = asyncio.get_running_loop()
    t0 = time.time()
    try:
        result = await loop.run_in_executor(None, pipeline.analyze, img)
    except Exception as e:
        logger.error("pipeline analyze failed", exc_info=True)
        elapsed = (time.time() - t0) * 1000
        _update_counters(ok=False, inference_ms=elapsed)
        raise HTTPException(status_code=500, detail=f"pipeline error: {e}")

    elapsed = (time.time() - t0) * 1000
    ok = result.get("status") == "completed"
    _update_counters(ok=ok, inference_ms=elapsed)
    return AnalyzeResponse(**result)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DermApIxel Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8101, help="Port (default 8101)")
    args = parser.parse_args()

    enable_worker = os.getenv("ENABLE_WORKER", "true").lower() in ("1", "true", "yes")

    app = create_app(enable_worker=enable_worker)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
