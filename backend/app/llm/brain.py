"""
LLM Brain - streaming chat with tool calling.

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
            "Supported ops: set_repeat, set_param, add_block, remove_block, replace_block.\n\n"
            "Valid block types: embedding, transformer_stack, multi_head_attention, feed_forward, "
            "moe_feed_forward, layer_norm, linear, conv1d, ssm, rnn, pooling, dropout, activation.\n\n"
            "Key params per type:\n"
            "  feed_forward: {hidden_size, intermediate_size, activation}\n"
            "  moe_feed_forward: {hidden_size, intermediate_size, num_experts, num_experts_per_tok, activation}\n"
            "  multi_head_attention: {hidden_size, num_heads, head_dim, num_kv_heads, attention_type, is_causal}\n"
            "  layer_norm: {normalized_shape, norm_type}\n\n"
            "Use block IDs exactly as they appear in the architecture_ir."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "edit_spec": {
                    "type": "object",
                    "description": (
                        "Edit specification. Required field: op. "
                        "op='set_repeat': change layer count - {op, target, value}. "
                        "op='set_param': change one param - {op, target, key, value}. "
                        "op='remove_block': delete a block - {op, target}. "
                        "op='replace_block': swap a block for a new one (use for type changes like FFN→MoE) - "
                        "{op, target, block: {id, label, type, params, repeat}}. "
                        "op='add_block': insert a new block - {op, block: {...}, after: <existing_block_id>}. "
                        "Examples:\n"
                        '  {"op":"set_repeat","target":"encoder","value":6}\n'
                        '  {"op":"set_param","target":"self_attn","key":"num_heads","value":8}\n'
                        '  {"op":"remove_block","target":"pooler"}\n'
                        '  {"op":"replace_block","target":"ffn","block":{"id":"ffn","label":"MoE FFN","type":"moe_feed_forward","params":{"hidden_size":768,"intermediate_size":3072,"num_experts":8,"num_experts_per_tok":2,"activation":"gelu"},"repeat":1}}'
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

# OpenAI format - same content, different wrapper
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

    # Build a flat list of block IDs for quick LLM reference
    def collect_blocks(blocks: list) -> list:
        result = []
        for b in blocks:
            result.append(b)
            result.extend(collect_blocks(b.children))
        return result

    all_blocks = collect_blocks(ir.blocks)
    block_id_list = ", ".join(f'"{b.id}"' for b in all_blocks)

    # Find the transformer stack block for a concrete example
    stack_block = next(
        (b for b in all_blocks if b.type == "transformer_stack"), None
    )
    example_id = stack_block.id if stack_block else (all_blocks[0].id if all_blocks else "encoder")

    return f"""You are ModelVerse, an expert ML architecture assistant.

The user is currently viewing this model architecture:
<architecture_ir>
{ir_json}
</architecture_ir>

Available block IDs: {block_id_list}

You can:
- Answer questions about any part of this architecture
- Explain what layers do and why they exist
- Suggest and apply architectural edits using apply_edit
- Estimate the compute impact of changes using estimate_compute
- Look up other models for comparison using search_huggingface

IMPORTANT - when calling apply_edit:
- Use ONLY block IDs listed above. Never invent IDs.
- To change the number of layers/blocks, use op="set_repeat" with the transformer_stack block ID.
- To convert a feed-forward block to MoE, use op="replace_block" with type="moe_feed_forward".
  Example for this model: {{"op":"replace_block","target":"ffn","block":{{"id":"ffn","label":"MoE FFN","type":"moe_feed_forward","params":{{"hidden_size":768,"intermediate_size":3072,"num_experts":8,"num_experts_per_tok":2,"activation":"gelu"}},"repeat":1}}}}
- To convert GQA/MQA attention, use op="replace_block" targeting the attention block with updated num_kv_heads.
- Concrete set_repeat example for this model: {{"op": "set_repeat", "target": "{example_id}", "value": 6}}
- Call apply_edit exactly ONCE per user request with a single edit_spec dict.
- If apply_edit returns an error with "available_block_ids", use one of those IDs and retry immediately.

Valid block types and their key params:
  feed_forward       → hidden_size, intermediate_size, activation
  moe_feed_forward   → hidden_size, intermediate_size, num_experts, num_experts_per_tok, activation
  multi_head_attention → hidden_size, num_heads, head_dim, num_kv_heads, attention_type, is_causal
  layer_norm         → normalized_shape, norm_type ("layer_norm" or "rms_norm")
  linear             → in_features, out_features, bias

Guidelines:
- Be concise. Cite parameter counts and dimensions when relevant.
- When asked about compute impact, call estimate_compute - don't calculate manually.
- When asked to edit the architecture, always call apply_edit - don't just describe the edit in text.
- After a successful apply_edit, summarise what changed using the compute_delta in the tool result.
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
            # IR injected by backend - LLM doesn't need to pass it
            return tool_estimate_compute(ir=current_ir.model_dump())
        elif tool_name == "apply_edit":
            # IR injected by backend - LLM only sends edit_spec
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
