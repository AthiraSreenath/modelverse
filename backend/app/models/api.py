"""Request and response schemas for the API endpoints."""

from pydantic import BaseModel
from .ir import ArchitectureIR, ComputeStats, EditSpec, EditResult, SourceType


class ResolveRequest(BaseModel):
    input: str  # HF model ID, file path, or NL query


class ResolveResponse(BaseModel):
    ir: ArchitectureIR
    source: SourceType
    cached: bool = False
    resolve_time_ms: float | None = None


class ComputeRequest(BaseModel):
    ir: ArchitectureIR


class ComputeResponse(BaseModel):
    compute: ComputeStats


class EditRequest(BaseModel):
    ir: ArchitectureIR
    edit_spec: EditSpec


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    ir: ArchitectureIR


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
