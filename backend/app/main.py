"""
ModelVerse Backend - FastAPI application.

Run with: uv run fastapi dev app/main.py
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from .models.api import (
    ResolveRequest,
    ResolveResponse,
    ComputeRequest,
    ComputeResponse,
    EditRequest,
    ChatRequest,
    HealthResponse,
)
from .models.ir import EditResult
from .resolvers.router import resolve
from .resolvers.prebaked import list_prebaked
from .compute.estimator import estimate_compute
from .edit.engine import apply_edit
from .llm.brain import stream_chat

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ModelVerse backend starting up")
    logger.info("Pre-baked models available: %s", list_prebaked())
    yield
    logger.info("ModelVerse backend shutting down")


app = FastAPI(
    title="ModelVerse API",
    description="Architecture Reasoning Engine for ML models",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS - allow the Next.js frontend (and localhost for dev)
_allowed_origins = [
    "http://localhost:3000",
    "https://modelverse.vercel.app",
]
if extra := os.getenv("EXTRA_CORS_ORIGINS"):
    _allowed_origins.extend(extra.split(","))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@app.post("/resolve", response_model=ResolveResponse)
async def resolve_model(req: ResolveRequest):
    """
    Resolve a model name to its Architecture IR.

    Phase 1: checks pre-baked library first, then HuggingFace Hub.
    Returns the IR, the source, and whether it was served from cache.
    """
    hf_token = os.getenv("HF_TOKEN")
    try:
        return await resolve(req.input, hf_token=hf_token)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error resolving '%s'", req.input)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post("/compute", response_model=ComputeResponse)
async def compute_stats(req: ComputeRequest):
    """
    Compute parameter counts, FLOPs per token, and memory for an Architecture IR.
    Pure arithmetic - runs in <1ms.
    """
    try:
        stats = estimate_compute(req.ir)
        return ComputeResponse(compute=stats)
    except Exception as e:
        logger.exception("Error computing stats")
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/edit", response_model=EditResult)
async def edit_architecture(req: EditRequest):
    """
    Apply an edit spec to an Architecture IR.
    Returns the modified IR, a diff, and compute delta.
    """
    try:
        return apply_edit(req.ir, req.edit_spec)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Error applying edit")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Stream a response from the LLM Brain.
    The LLM has the current Architecture IR in its system prompt and
    can call tools: search_huggingface, search_web, estimate_compute,
    apply_edit, explain_layer.

    Returns a Server-Sent Events stream.
    Each event: data: <json>\\n\\n
    Event types: text | tool_call | tool_result | done | error
    """
    return StreamingResponse(
        stream_chat(req.messages, req.ir),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/prebaked", response_model=list[str])
async def get_prebaked_models():
    """List all available pre-baked model IDs."""
    return list_prebaked()
