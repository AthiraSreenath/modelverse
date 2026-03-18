"""
LLM Brain — streaming chat with tool calling.

The LLM receives the current Architecture IR in its system prompt.
It can call 5 tools: search_huggingface, search_web, estimate_compute,
apply_edit, explain_layer.
"""

from __future__ import annotations
import json
import logging
import os
from typing import AsyncIterator

import anthropic

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
# Tool schemas (Anthropic format)
# ---------------------------------------------------------------------------

TOOLS = [
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
                    "description": "The HuggingFace model ID, e.g. 'bert-base-uncased' or 'mistralai/Mistral-7B-v0.1'"
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
                "query": {
                    "type": "string",
                    "description": "Specific search query, e.g. 'GPT-3 architecture layers hidden size Brown 2020'"
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "estimate_compute",
        "description": (
            "Compute parameter count, FLOPs per token, and memory estimates for an Architecture IR. "
            "Always call this when the user asks about parameters, memory, or compute cost, "
            "or when showing the impact of an architectural change."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ir": {
                    "type": "object",
                    "description": "The Architecture IR JSON object to estimate compute for"
                }
            },
            "required": ["ir"],
        },
    },
    {
        "name": "apply_edit",
        "description": (
            "Apply an architectural edit to the current model and return the modified IR, "
            "a diff, and compute delta. Always call this when the user asks to change the architecture. "
            "Supported ops: set_repeat (change layer count), set_param (change hidden_size/num_heads/etc), "
            "add_block, remove_block, replace_block."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ir": {"type": "object", "description": "Current Architecture IR"},
                "edit_spec": {
                    "type": "object",
                    "description": (
                        "Edit specification. Examples: "
                        '{"op":"set_repeat","target":"encoder","value":6}, '
                        '{"op":"set_param","target":"encoder","key":"num_attention_heads","value":8}, '
                        '{"op":"remove_block","target":"pooler"}'
                    ),
                },
            },
            "required": ["ir", "edit_spec"],
        },
    },
    {
        "name": "explain_layer",
        "description": (
            "Return a structured explanation for a layer type, including its mathematical formula, "
            "PyTorch pseudocode, and architectural motivation. "
            "Call this when the user clicks a layer or asks what a specific component does."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "layer_type": {
                    "type": "string",
                    "description": "Layer type string, e.g. 'multi_head_attention', 'feed_forward', 'layer_norm', 'ssm'"
                },
                "params": {
                    "type": "object",
                    "description": "The layer's params dict from the Architecture IR"
                },
            },
            "required": ["layer_type", "params"],
        },
    },
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
- When showing formula breakdowns, use the actual numbers from the IR (e.g. "768 × 768 × 3 = 1,769,472").
- If the model in the IR was derived from web search (source_confidence != "exact"), note any uncertainty.
- Never make up architectural details. Use the IR or call a tool to verify.
"""


async def stream_chat(
    messages: list[ChatMessage],
    ir: ArchitectureIR,
) -> AsyncIterator[str]:
    """
    Stream a chat response from the LLM Brain.
    Yields Server-Sent Event lines: data: <json>\\n\\n
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        yield 'data: {"type":"error","error":"ANTHROPIC_API_KEY not set"}\n\n'
        return

    client = anthropic.AsyncAnthropic(api_key=api_key)
    system = _build_system_prompt(ir)

    anthropic_messages = [
        {"role": m.role, "content": m.content}
        for m in messages
    ]

    # Agentic loop: allow up to 5 rounds of tool use
    for _round in range(5):
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=4096,
                system=system,
                messages=anthropic_messages,
                tools=TOOLS,  # type: ignore[arg-type]
            )
        except anthropic.APIError as e:
            yield f'data: {json.dumps({"type": "error", "error": str(e)})}\n\n'
            return

        # Stream text content blocks
        for block in response.content:
            if block.type == "text":
                # Chunk the text for streaming feel
                text = block.text
                chunk_size = 20
                for i in range(0, len(text), chunk_size):
                    chunk = text[i : i + chunk_size]
                    yield f'data: {json.dumps({"type": "text", "text": chunk})}\n\n'

        # Check if we're done
        if response.stop_reason == "end_turn":
            yield 'data: {"type":"done"}\n\n'
            return

        # Handle tool calls
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input

                # Notify frontend a tool is being called
                yield f'data: {json.dumps({"type": "tool_call", "tool": tool_name, "input": tool_input})}\n\n'

                # Execute the tool
                try:
                    if tool_name == "search_huggingface":
                        result = await tool_search_huggingface(**tool_input)
                    elif tool_name == "search_web":
                        result = await tool_search_web(**tool_input)
                    elif tool_name == "estimate_compute":
                        result = tool_estimate_compute(**tool_input)
                    elif tool_name == "apply_edit":
                        result = tool_apply_edit(**tool_input)
                    elif tool_name == "explain_layer":
                        result = tool_explain_layer(**tool_input)
                    else:
                        result = {"error": f"Unknown tool: {tool_name}"}
                except Exception as e:
                    logger.exception("Tool '%s' raised an error", tool_name)
                    result = {"error": str(e)}

                # Notify frontend of the tool result
                yield f'data: {json.dumps({"type": "tool_result", "tool": tool_name, "result": result})}\n\n'

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

            # Add assistant message + tool results to conversation
            anthropic_messages.append({"role": "assistant", "content": response.content})  # type: ignore
            anthropic_messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop reason
        yield 'data: {"type":"done"}\n\n'
        return

    yield 'data: {"type":"done"}\n\n'
