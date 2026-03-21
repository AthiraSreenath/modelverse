"""
Multimodal Architecture Parser (LLM-based)
==========================================
For any HuggingFace model whose config.json contains nested sub-configs
(vision_config, connector_config, image_config, etc.), autoconfig_mapper
only sees the top-level language backbone and misses the rest.

This module:
1. Detects multimodal configs by scanning for known sub-config keys.
2. Pre-analyzes the config to extract signals (decoder vs encoder, token
   interleaving, vision projection, GQA) and injects them into the prompt.
3. Sends the enriched prompt to the LLM to generate a full block list.
4. Computes param counts from config math (not LLM guesses) wherever possible.
5. Caches by SHA-256(config + prompt version) so prompt improvements
   automatically invalidate stale results.

This is NOT model-specific. The LLM can reason about CogViT, CLIP, SAM,
CogVLM, InternViT, SigLIP, etc. without any hardcoded parser per model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path

from ..models.ir import (
    ArchBlock, ArchitectureIR, BlockType, SourceType, SourceConfidence,
    ComputeStats,
)

logger = logging.getLogger(__name__)

# Bump this string whenever the prompt or cache schema changes.
# Old cached block lists (keyed on a different version) are silently superseded.
_PROMPT_VERSION = "v5"

# Canonical block IDs the LLM is required to use for known components.
# _compute_stats_from_config matches on these exact IDs, bypassing fragile
# keyword/type matching entirely.
CANONICAL_IDS = {
    "vision_patch_embed",   # patch embedding before the vision stack
    "vision_encoder",       # the main vision transformer stack
    "vision_projection",    # linear projection of vision features → LM hidden dim
    "embed_tokens",         # text token embedding table
    "multimodal_merge",     # merge / token-interleaving step
    "lm_decoder",           # language model decoder transformer stack
    "final_norm",           # final RMSNorm/LayerNorm before LM head
    "lm_head",              # output projection to vocab logits
}

_CACHE_DIR = (
    Path(os.getenv("MODELVERSE_CACHE_DIR", Path.home() / ".cache" / "modelverse"))
    / "multimodal_llm"
)
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Config keys that indicate a nested sub-model component
_MULTIMODAL_KEYS: set[str] = {
    "vision_config",
    "visual_config",
    "image_config",
    "encoder_config",
    "decoder_config",
    "connector_config",
    "projector_config",
    "vit_config",
    "cogvit_config",
    "sam_config",
    "clip_config",
    "audio_config",
    "speech_config",
    "text_config",     # used by GLM-OCR, CogVLM, etc.
    "mm_vision_tower",
    "vision_tower",
}


def is_multimodal(config: dict) -> bool:
    """Return True if the config has any nested sub-model component."""
    return any(k in config for k in _MULTIMODAL_KEYS)


# ---------------------------------------------------------------------------
# Cache helpers  (key includes prompt version so upgrades auto-invalidate)
# ---------------------------------------------------------------------------

def _cache_key(config: dict) -> str:
    blob = _PROMPT_VERSION + json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _load_cache(key: str) -> list[dict] | None:
    p = _CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return None


def _save_cache(key: str, blocks: list[dict]) -> None:
    p = _CACHE_DIR / f"{key}.json"
    try:
        p.write_text(json.dumps(blocks))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Config signal extractor
# Reads the config before calling the LLM and produces a plain-English
# "hints" block that is injected into the prompt. This prevents the LLM from
# having to infer these signals itself — smaller/faster models like Haiku
# miss them reliably without explicit guidance.
# ---------------------------------------------------------------------------

def _extract_config_signals(config: dict) -> str:
    """
    Scan the config and return a bullet-list of architecture signals
    that the LLM must respect when generating the block list.
    """
    signals: list[str] = []

    # ── Locate text/LM sub-config ──────────────────────────────────────────
    lm_cfg = (
        config.get("text_config")
        or config.get("language_config")
        or config.get("llm_config")
        or config
    )

    h       = lm_cfg.get("hidden_size", 0)
    n       = lm_cfg.get("num_hidden_layers", 0)
    vocab   = lm_cfg.get("vocab_size", 0)
    heads   = lm_cfg.get("num_attention_heads", 0)
    kv_h    = lm_cfg.get("num_key_value_heads", heads)
    use_cache = lm_cfg.get("use_cache", False)
    tie_emb = lm_cfg.get("tie_word_embeddings", True)

    # Decoder signal: GQA (kv_heads < heads) or use_cache=true are decoder-only markers
    is_decoder = use_cache or (kv_h and heads and kv_h < heads)
    if is_decoder:
        signals.append(
            f"TEXT SIDE IS A DECODER (not encoder): use_cache={use_cache}, "
            f"num_key_value_heads={kv_h} < num_attention_heads={heads} (GQA). "
            "Label the text transformer block as 'Language Model Decoder', never 'Encoder'."
        )
    else:
        signals.append("Text transformer block is encoder-style.")

    if h and n:
        signals.append(
            f"LM backbone: hidden_size={h}, num_hidden_layers={n}, vocab_size={vocab}, "
            f"intermediate_size={lm_cfg.get('intermediate_size', '?')}, "
            f"num_attention_heads={heads}, num_key_value_heads={kv_h}."
        )

    if not tie_emb:
        signals.append(
            "tie_word_embeddings=false: the LM head is a SEPARATE weight matrix "
            f"(vocab_size={vocab} x hidden_size={h}). Add it as a distinct 'LM Head' block "
            "after the decoder with type='linear'."
        )
    else:
        signals.append("tie_word_embeddings=true: LM head shares embedding weights, no separate block needed.")

    # ── Vision sub-config ──────────────────────────────────────────────────
    vis_cfg = (
        config.get("vision_config")
        or config.get("visual_config")
        or config.get("vit_config")
        or {}
    )
    vis_h       = vis_cfg.get("hidden_size", 0)
    vis_depth   = vis_cfg.get("depth") or vis_cfg.get("num_hidden_layers", 0)
    vis_out     = vis_cfg.get("out_hidden_size", 0)
    patch_size  = vis_cfg.get("patch_size", 0)
    img_size    = vis_cfg.get("image_size", 0)

    if vis_h and vis_depth:
        signals.append(
            f"Vision encoder: hidden_size={vis_h}, depth={vis_depth}, "
            f"intermediate_size={vis_cfg.get('intermediate_size', '?')}, "
            f"image_size={img_size}, patch_size={patch_size}. "
            "Use type='transformer_stack'."
        )

    if vis_out and h and vis_out == h:
        signals.append(
            f"vision_config.out_hidden_size={vis_out} matches LM hidden_size={h}. "
            "This means the vision encoder output is projected into the LM hidden space. "
            "Add a 'Vision Output Projection' block (type='linear') between the vision "
            "encoder and the multimodal merge step."
        )
    elif vis_out and vis_h and vis_out != vis_h:
        signals.append(
            f"vision_config.out_hidden_size={vis_out} differs from vision hidden_size={vis_h}. "
            "Add a 'Vision Output Projection' block after the vision encoder."
        )

    # ── Token-interleaving signal ──────────────────────────────────────────
    has_image_token_id = any(
        k in config for k in ("image_token_id", "image_start_token_id", "img_token_id")
    )
    if has_image_token_id:
        signals.append(
            "Config has image_token_id / image_start_token_id: visual features are "
            "INSERTED INLINE into the token sequence (token interleaving), NOT concatenated "
            "as a separate stream. The merge step should be labelled "
            "'Multimodal Token Sequence (token interleaving)', type='add', param_count=0."
        )

    # ── Connector / projector sub-config ──────────────────────────────────
    conn_cfg = config.get("connector_config") or config.get("projector_config") or {}
    if conn_cfg:
        signals.append(
            f"Connector config present: {json.dumps(conn_cfg)[:200]}. "
            "Add a connector/projector block between vision encoder and merge."
        )

    return "\n".join(f"  - {s}" for s in signals)


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert ML engineer specialising in multimodal vision-language
architectures. Given a HuggingFace config.json and a set of pre-extracted
architecture signals, produce the correct full forward-pass architecture as
a JSON block list.

CRITICAL RULES (must not be violated):
1. Follow the pre-extracted signals EXACTLY — they are derived from the config.
2. Use the EXACT canonical block IDs listed below for known components.
   These IDs are used by downstream code to override your param estimates with
   exact computed values, so wrong IDs cause wrong parameter counts.
3. The forward-pass order is FIXED (vision first, merge before decoder):
     vision_patch_embed → vision_encoder → vision_projection
               ↓
     multimodal_merge  ←── embed_tokens (parallel independent input)
               ↓
     lm_decoder → [final_norm] → lm_head
   The LM decoder ALWAYS comes AFTER the multimodal merge. Never before."""

_USER_TMPL = """\
Model ID: {model_id}

Pre-extracted architecture signals (ground truth from config):
{signals}

Canonical block IDs you MUST use (exact strings, no variations):
  "vision_patch_embed"  — vision patch / token embedding (type=embedding)
  "vision_encoder"      — vision transformer stack (type=transformer_stack)
  "vision_projection"   — vision output linear projection (type=linear)
  "embed_tokens"        — text token embedding table (type=embedding)
  "multimodal_merge"    — multimodal token merge/interleaving (type=add, param_count=0)
  "lm_decoder"          — language model decoder transformer (type=transformer_stack)
  "final_norm"          — final layer norm before LM head (type=layer_norm, if present)
  "lm_head"             — output projection to vocab logits (type=linear)

Full config JSON:
```json
{config_json}
```

Produce the architecture as a JSON array of blocks in STRICT forward-pass order.
Each block has exactly these fields:
{{
  "id": "canonical_id_or_unique_snake_case",
  "label": "Human-Readable Name",
  "type": one of ["embedding","transformer_stack","linear","add","layer_norm","unknown"],
  "param_count": integer (best estimate; 0 for param-free ops),
  "merge_from": null OR list of source block IDs,
  "notes": "one sentence"
}}

Layout / merge_from rules:
- Vision branch root (vision_patch_embed): merge_from=[]
- Text token embeddings (embed_tokens): merge_from=[]
  Also add "layout_align_y_with": "vision_projection" so it renders alongside
  the vision output at the correct vertical level.
- Multimodal merge (multimodal_merge): merge_from=["vision_projection","embed_tokens"]
- lm_decoder comes IMMEDIATELY after multimodal_merge (merge_from=null, auto-connects).
- All other blocks: merge_from=null.

Return ONLY the JSON array, no prose, no markdown fences.
"""


def _build_prompt(config: dict, model_id: str) -> str:
    signals = _extract_config_signals(config)
    config_str = json.dumps(config, indent=2)
    if len(config_str) > 4000:
        config_str = config_str[:4000] + "\n  ... (truncated)"
    return _USER_TMPL.format(model_id=model_id, signals=signals, config_json=config_str)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def _call_anthropic(system: str, user: str) -> str:
    import anthropic
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.AsyncAnthropic(api_key=key)
    msg = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


async def _call_openai(system: str, user: str) -> str:
    from openai import AsyncOpenAI
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = AsyncOpenAI(api_key=key)
    resp = await client.chat.completions.create(
        model="gpt-4o",
        max_tokens=2048,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


async def _call_llm(system: str, user: str) -> str:
    if os.getenv("OPENAI_API_KEY"):
        return await _call_openai(system, user)
    if os.getenv("ANTHROPIC_API_KEY"):
        return await _call_anthropic(system, user)
    raise RuntimeError("No LLM API key set (ANTHROPIC_API_KEY or OPENAI_API_KEY)")


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

_BLOCK_TYPE_MAP: dict[str, BlockType] = {
    "embedding":         BlockType.EMBEDDING,
    "transformer_stack": BlockType.TRANSFORMER_STACK,
    "linear":            BlockType.LINEAR,
    "add":               BlockType.ADD,
    "layer_norm":        BlockType.LAYER_NORM,
    "unknown":           BlockType.UNKNOWN,
}


def _extract_json_array(text: str) -> list[dict]:
    """Pull the first JSON array out of the LLM response."""
    text = re.sub(r"```[a-z]*\n?", "", text).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        result = json.loads(text[start : end + 1])
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def _blocks_from_llm(raw_blocks: list[dict]) -> list[ArchBlock]:
    blocks: list[ArchBlock] = []
    for b in raw_blocks:
        bid = str(b.get("id", f"block_{len(blocks)}"))
        label = str(b.get("label", bid))
        type_str = str(b.get("type", "unknown")).lower()
        btype = _BLOCK_TYPE_MAP.get(type_str, BlockType.UNKNOWN)
        param_count = b.get("param_count")
        if param_count is not None:
            try:
                param_count = int(param_count)
            except (TypeError, ValueError):
                param_count = None

        merge_from: list[str] | None = b.get("merge_from")
        if merge_from is not None and not isinstance(merge_from, list):
            merge_from = None

        layout_align_y_with: str | None = b.get("layout_align_y_with")

        blocks.append(ArchBlock(
            id=bid,
            label=label,
            type=btype,
            params={},
            repeat=1,
            children=[],
            param_count=param_count,
            notes=b.get("notes"),
            merge_from=merge_from,
            layout_align_y_with=(
                layout_align_y_with if isinstance(layout_align_y_with, str) else None
            ),
        ))
    return blocks


# ---------------------------------------------------------------------------
# Param computation  (config math, not LLM guesses)
# ---------------------------------------------------------------------------

def _vision_encoder_params(vis_cfg: dict) -> int:
    """
    Compute approximate vision encoder params from vision_config fields.
    Uses the standard ViT formula: patch_embed + pos_embed + L*(attn+FFN+norms).
    """
    vis_h   = vis_cfg.get("hidden_size", 0)
    depth   = vis_cfg.get("depth") or vis_cfg.get("num_hidden_layers", 0)
    vis_ffn = vis_cfg.get("intermediate_size", 0)
    patch   = vis_cfg.get("patch_size", 0)
    img     = vis_cfg.get("image_size", 0)
    heads   = vis_cfg.get("num_heads") or vis_cfg.get("num_attention_heads", 0)

    if not (vis_h and depth):
        return 0

    # Patch embedding: 3 channels × patch² × hidden
    patch_emb = 3 * patch * patch * vis_h if patch else 0
    # CLS + position embeddings
    n_patches = (img // patch) ** 2 if (img and patch) else 196
    pos_emb   = (n_patches + 1) * vis_h

    # Per-layer: standard ViT MHA + SwiGLU/MLP
    attn = 4 * vis_h * vis_h   # Q+K+V+O projections (square)
    ffn_p = (3 * vis_h * vis_ffn) if vis_ffn else (8 * vis_h * vis_h)   # SwiGLU
    norms = 2 * 2 * vis_h      # 2 norms × (weight + bias)
    per_layer = attn + ffn_p + norms

    return patch_emb + pos_emb + depth * per_layer


def _compute_stats_from_config(blocks: list[ArchBlock], config: dict) -> ComputeStats:
    """
    Compute param stats from config math, not LLM estimates.

    Strategy:
    1. Canonical-ID match (exact, no type check) — overrides LLM estimate completely.
    2. Keyword fallback for blocks with non-canonical IDs.
    3. Raw LLM estimate for everything else.

    This makes correctness independent of whatever type the LLM assigns.
    """
    # ── Locate LM sub-config ───────────────────────────────────────────────
    lm_cfg = (
        config.get("text_config")
        or config.get("language_config")
        or config.get("llm_config")
        or config
    )
    h     = lm_cfg.get("hidden_size", 0)
    n     = lm_cfg.get("num_hidden_layers", 0)
    # vocab_size may live in lm_cfg or at the top level (some models put it there)
    vocab = lm_cfg.get("vocab_size") or config.get("vocab_size", 0)
    ffn   = lm_cfg.get("intermediate_size", 0)
    heads = lm_cfg.get("num_attention_heads", 0)
    kv_h  = lm_cfg.get("num_key_value_heads", heads)
    tie   = lm_cfg.get("tie_word_embeddings", True)
    head_dim = lm_cfg.get("head_dim") or (h // heads if heads else 0)

    # ── Exact LM decoder params (GQA-aware) ───────────────────────────────
    exact_lm_params: int | None = None
    if h and n and heads and head_dim:
        q_proj  = h * (heads * head_dim)
        kv_proj = 2 * h * (kv_h * head_dim)
        o_proj  = (heads * head_dim) * h
        attn    = q_proj + kv_proj + o_proj
        if ffn:
            act   = lm_cfg.get("hidden_act", "silu")
            ffn_p = (3 * h * ffn) if act in ("silu", "gelu_new") else (2 * h * ffn)
        else:
            ffn_p = 4 * h * h
        norm_p = 2 * h    # single RMSNorm per layer (weight only, no bias)
        exact_lm_params = n * (attn + ffn_p + norm_p)

    # ── Exact embedding params ─────────────────────────────────────────────
    exact_emb_params: int | None = (vocab * h) if (vocab and h) else None

    # ── Separate LM head (when not tied) ──────────────────────────────────
    lm_head_params: int | None = (vocab * h) if (not tie and vocab and h) else None

    # ── Vision encoder params ─────────────────────────────────────────────
    vis_cfg = (
        config.get("vision_config")
        or config.get("visual_config")
        or config.get("vit_config")
        or {}
    )
    exact_vis_params: int | None = _vision_encoder_params(vis_cfg) or None

    vis_h_in  = vis_cfg.get("hidden_size", 0)
    vis_h_out = vis_cfg.get("out_hidden_size", 0)
    exact_vis_proj: int | None = None
    if vis_h_in and vis_h_out and vis_h_in != vis_h_out:
        exact_vis_proj = vis_h_in * vis_h_out + vis_h_out  # linear + bias

    # ── Canonical-ID → computed value map ─────────────────────────────────
    # We match by exact block id so correctness doesn't depend on the LLM
    # choosing a particular type string or label phrasing.
    canonical_override: dict[str, int | None] = {
        "embed_tokens":       exact_emb_params,
        "lm_decoder":         exact_lm_params,
        "vision_encoder":     exact_vis_params,
        "lm_head":            lm_head_params,
        "vision_projection":  exact_vis_proj,
        "multimodal_merge":   0,
        "final_norm":         0,
        "vision_patch_embed": None,   # let LLM estimate stand (varies widely)
    }

    # ── Assign params per block ────────────────────────────────────────────
    total = 0
    emb   = 0

    for b in blocks:
        bid_lower = b.id.lower()
        lbl_lower = b.label.lower()

        # 1. Canonical-ID exact match (highest priority)
        if b.id in canonical_override:
            p_override = canonical_override[b.id]
            p = p_override if p_override is not None else (b.param_count or 0)

        # 2. Keyword fallback for non-canonical IDs (handles model-specific naming)
        elif (b.type == BlockType.EMBEDDING and exact_emb_params is not None):
            p = exact_emb_params

        elif (b.type == BlockType.TRANSFORMER_STACK
              and any(kw in bid_lower or kw in lbl_lower
                      for kw in ("decoder", "lm_", "language model", "language_model",
                                 "text model", "text_model"))
              and exact_lm_params is not None):
            p = exact_lm_params

        elif (b.type == BlockType.TRANSFORMER_STACK
              and any(kw in bid_lower or kw in lbl_lower
                      for kw in ("vision", "visual", "vit", "image encoder",
                                 "image_encoder"))
              and exact_vis_params is not None):
            p = exact_vis_params

        elif (b.type == BlockType.LINEAR
              and any(kw in bid_lower or kw in lbl_lower
                      for kw in ("lm_head", "lm head", "output proj", "output_proj",
                                 "output logits"))
              and lm_head_params is not None):
            p = lm_head_params

        elif (b.type == BlockType.LINEAR
              and any(kw in bid_lower or kw in lbl_lower
                      for kw in ("vision_proj", "vision proj", "vision_output_proj",
                                 "visual_proj"))
              and exact_vis_proj is not None):
            p = exact_vis_proj

        # 3. Raw LLM estimate
        else:
            p = b.param_count or 0

        total += p
        if b.type == BlockType.EMBEDDING:
            emb += p

    return ComputeStats(
        params_total=total,
        params_embedding=emb,
        params_encoder=total - emb,
        params_decoder=0,
        flops_per_token=None,
        memory_bytes_fp16=total * 2 if total else None,
        kv_cache_bytes_per_token=None,
        context_window=lm_cfg.get("max_position_embeddings"),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def parse_multimodal_with_llm(
    config: dict,
    model_id: str,
) -> ArchitectureIR:
    """
    Generate a full multimodal ArchitectureIR using the LLM.

    Cached by SHA-256(prompt_version + config JSON). Bumping _PROMPT_VERSION
    automatically invalidates all prior cached block lists.
    """
    key    = _cache_key(config)
    cached = _load_cache(key)

    if cached is not None:
        logger.debug("Multimodal LLM parser: cache hit for %s", model_id)
        raw_blocks = cached
    else:
        prompt = _build_prompt(config, model_id)
        try:
            response   = await _call_llm(_SYSTEM, prompt)
            raw_blocks = _extract_json_array(response)
            if not raw_blocks:
                raise ValueError("LLM returned no blocks")
            _save_cache(key, raw_blocks)
            logger.info("Multimodal LLM parser: generated %d blocks for %s",
                        len(raw_blocks), model_id)
        except Exception as exc:
            logger.warning("Multimodal LLM parser failed for %s: %s — falling back",
                           model_id, exc)
            fallback_block = ArchBlock(
                id="unknown_multimodal",
                label="Multimodal Architecture (parsing failed)",
                type=BlockType.UNKNOWN,
                params={},
                repeat=1,
                children=[],
                param_count=None,
                notes=(
                    f"Could not generate architecture via LLM ({exc}). "
                    "Check that ANTHROPIC_API_KEY or OPENAI_API_KEY is set."
                ),
            )
            return ArchitectureIR(
                name=model_id,
                display_name=model_id.split("/")[-1],
                family="multimodal",
                task="unknown",
                blocks=[fallback_block],
                compute=ComputeStats(params_total=0, params_embedding=0,
                                     params_encoder=0, params_decoder=0),
                source=SourceType.LLM_KNOWLEDGE,
                source_confidence=SourceConfidence.LOW,
            )

    blocks  = _blocks_from_llm(raw_blocks)
    compute = _compute_stats_from_config(blocks, config)

    name       = config.get("_name_or_path") or model_id.split("/")[-1]
    model_type = config.get("model_type", "multimodal")

    return ArchitectureIR(
        name=model_id,
        display_name=name,
        family=model_type,
        task="image-to-text",
        blocks=blocks,
        compute=compute,
        source=SourceType.LLM_KNOWLEDGE,
        source_confidence=SourceConfidence.MEDIUM,
    )
