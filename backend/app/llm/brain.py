"""
LLM Brain — streaming chat with tool calling.

Provider detection order:
  1. ANTHROPIC_API_KEY  → Claude Sonnet
  2. OPENAI_API_KEY     → GPT-4o

The LLM receives the current Architecture IR in its system prompt and
can call 5 tools: search_huggingface, search_web, estimate_compute,
apply_edit, explain_layer.
"""

from __future__ import annotations
import json
import logging
import os
from typing import AsyncIterator

from ..models.ir import ArchitectureIR
from ..models.api import ChatMessage
from .tools import (
    tool_search_huggingface,
    tool_search_web,
    tool_estimate_compute,
    tool_apply_edit,
    tool_explain_layer,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

# Anthropic format
_ANTHROPIC_TOOLS = [
    {
        "name": "search_huggingface",
        "description": (
            "Fetch the config.json and model card for a HuggingFace model. "
            "Returns the full config dict, model card excerpt, and a parsed Architecture IR. "
            "Use this to resolve a model name the user mentions, or to compare architectures."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "The HuggingFace model ID, e.g. 'bert-base-uncased'",
                }
            },
            "required": ["model_id"],
        },
    },
    {
        "name": "search_web",
        "description": (
            "Search the web for architecture information. "
            "Use for models not on HuggingFace (GPT-3, PaLM, etc.), ArXiv papers, "
            "or to verify architectural details from official sources."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Specific search query"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "estimate_compute",
        "description": (
            "Compute parameter count, FLOPs per token, and memory estimates for the current model. "
            "Call this when the user asks about parameters, memory, or compute cost. No input required."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "apply_edit",
        "description": (
            "Apply an architectural edit to the current model and return the modified IR, "
            "a diff, and compute delta. Call this when the user asks to change the architecture. "
            "Supported ops: set_repeat, set_param, add_block, remove_block, replace_block. "
            "Use block IDs exactly as they appear in the architecture_ir (e.g. 'encoder', 'embeddings', 'pooler')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "edit_spec": {
                    "type": "object",
                    "description": (
                        "Edit specification. op is one of: set_repeat, set_param, add_block, remove_block, replace_block. "
                        'Examples: {"op":"set_repeat","target":"encoder","value":6}, '
                        '{"op":"set_param","target":"encoder","key":"num_attention_heads","value":8}, '
                        '{"op":"remove_block","target":"pooler"}'
                    ),
                },
            },
            "required": ["edit_spec"],
        },
    },
    {
        "name": "explain_layer",
        "description": (
            "Return a structured explanation for a layer type including its mathematical formula, "
            "PyTorch pseudocode, and architectural motivation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "layer_type": {"type": "string", "description": "e.g. 'multi_head_attention'"},
                "params": {"type": "object", "description": "Layer params dict from the IR"},
            },
            "required": ["layer_type", "params"],
        },
    },
]

# OpenAI format — same content, different wrapper
_OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in _ANTHROPIC_TOOLS
]


def _build_system_prompt(ir: ArchitectureIR) -> str:
    ir_json = ir.model_dump_json(indent=2)
    return f"""You are ModelVerse, an expert ML architecture assistant.

The user is currently viewing this model architecture:
<architecture_ir>
{ir_json}
</architecture_ir>

You can:
- Answer questions about any part of this architecture
- Explain what layers do and why they exist
- Suggest and apply architectural edits
- Estimate the compute impact of changes
- Look up other models for comparison

Guidelines:
- Be concise and precise. Cite parameter counts and dimensions when relevant.
- When asked about compute impact, call estimate_compute — don't guess.
- When asked to edit the architecture, call apply_edit with the appropriate spec.
- When showing formula breakdowns, use the actual numbers from the IR.
- If source_confidence != "exact", note any uncertainty.
- Never make up architectural details. Use the IR or call a tool to verify.
"""


async def _dispatch_tool(
    tool_name: str, tool_input: dict, current_ir: ArchitectureIR
) -> dict:
    """Execute a tool call and return the result dict."""
    try:
        if tool_name == "search_huggingface":
            return await tool_search_huggingface(**tool_input)
        elif tool_name == "search_web":
            return await tool_search_web(**tool_input)
        elif tool_name == "estimate_compute":
            # IR injected by backend — LLM doesn't need to pass it
            return tool_estimate_compute(ir=current_ir.model_dump())
        elif tool_name == "apply_edit":
            # IR injected by backend — LLM only sends edit_spec
            return tool_apply_edit(
                ir=current_ir.model_dump(),
                edit_spec=tool_input.get("edit_spec", tool_input),
            )
        elif tool_name == "explain_layer":
            return tool_explain_layer(**tool_input)
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    except Exception as e:
        logger.exception("Tool '%s' raised an error", tool_name)
        return {"error": str(e)}


async def _stream_anthropic(
    messages: list[ChatMessage],
    system: str,
    ir: ArchitectureIR,
) -> AsyncIterator[str]:
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.AsyncAnthropic(api_key=api_key)

    anthropic_messages = [{"role": m.role, "content": m.content} for m in messages]

    for _round in range(5):
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=4096,
                system=system,
                messages=anthropic_messages,
                tools=_ANTHROPIC_TOOLS,  # type: ignore[arg-type]
            )
        except anthropic.APIError as e:
            yield f'data: {json.dumps({"type": "error", "error": str(e)})}\n\n'
            return

        for block in response.content:
            if block.type == "text":
                chunk_size = 20
                for i in range(0, len(block.text), chunk_size):
                    yield f'data: {json.dumps({"type": "text", "text": block.text[i:i+chunk_size]})}\n\n'

        if response.stop_reason == "end_turn":
            yield 'data: {"type":"done"}\n\n'
            return

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                yield f'data: {json.dumps({"type": "tool_call", "tool": block.name, "input": block.input})}\n\n'
                result = await _dispatch_tool(block.name, block.input, ir)
                yield f'data: {json.dumps({"type": "tool_result", "tool": block.name, "result": result})}\n\n'
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
            anthropic_messages.append({"role": "assistant", "content": response.content})  # type: ignore
            anthropic_messages.append({"role": "user", "content": tool_results})
            continue

        yield 'data: {"type":"done"}\n\n'
        return

    yield 'data: {"type":"done"}\n\n'


async def _stream_openai(
    messages: list[ChatMessage],
    system: str,
    ir: ArchitectureIR,
) -> AsyncIterator[str]:
    from openai import AsyncOpenAI, APIError

    api_key = os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key)

    openai_messages: list[dict] = [{"role": "system", "content": system}]
    openai_messages += [{"role": m.role, "content": m.content} for m in messages]

    for _round in range(5):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                max_tokens=4096,
                messages=openai_messages,  # type: ignore[arg-type]
                tools=_OPENAI_TOOLS,  # type: ignore[arg-type]
                tool_choice="auto",
            )
        except APIError as e:
            yield f'data: {json.dumps({"type": "error", "error": str(e)})}\n\n'
            return

        message = response.choices[0].message
        stop_reason = response.choices[0].finish_reason

        if message.content:
            chunk_size = 20
            for i in range(0, len(message.content), chunk_size):
                yield f'data: {json.dumps({"type": "text", "text": message.content[i:i+chunk_size]})}\n\n'

        if stop_reason == "stop":
            yield 'data: {"type":"done"}\n\n'
            return

        if stop_reason == "tool_calls" and message.tool_calls:
            tool_results = []
            for tc in message.tool_calls:
                tool_name = tc.function.name
                tool_input = json.loads(tc.function.arguments)
                yield f'data: {json.dumps({"type": "tool_call", "tool": tool_name, "input": tool_input})}\n\n'
                result = await _dispatch_tool(tool_name, tool_input, ir)
                yield f'data: {json.dumps({"type": "tool_result", "tool": tool_name, "result": result})}\n\n'
                tool_results.append({
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "content": json.dumps(result),
                })
            openai_messages.append(message.model_dump())  # type: ignore[arg-type]
            openai_messages.extend(tool_results)
            continue

        yield 'data: {"type":"done"}\n\n'
        return

    yield 'data: {"type":"done"}\n\n'


async def stream_chat(
    messages: list[ChatMessage],
    ir: ArchitectureIR,
) -> AsyncIterator[str]:
    """
    Stream a chat response from the LLM Brain.
    Yields Server-Sent Event lines: data: <json>\\n\\n

    Provider detection order:
      ANTHROPIC_API_KEY → Claude Sonnet
      OPENAI_API_KEY    → GPT-4o
    """
    system = _build_system_prompt(ir)

    if os.getenv("ANTHROPIC_API_KEY"):
        async for chunk in _stream_anthropic(messages, system, ir):
            yield chunk
    elif os.getenv("OPENAI_API_KEY"):
        async for chunk in _stream_openai(messages, system, ir):
            yield chunk
    else:
        yield 'data: {"type":"error","error":"No LLM key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in backend/.env"}\n\n'
