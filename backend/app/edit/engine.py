"""
Architecture Edit Engine.
Applies an EditSpec to an Architecture IR and returns the result + diff + compute delta.
"""

from __future__ import annotations
import copy
import logging
from typing import Any

from ..models.ir import (
    ArchBlock,
    ArchitectureIR,
    BlockType,
    ComputeDelta,
    DiffEntry,
    EditOp,
    EditResult,
    EditSpec,
)
from ..compute.estimator import estimate_compute

logger = logging.getLogger(__name__)


def _build_moe_children(block: ArchBlock) -> list[ArchBlock]:
    """Generate the standard internal sub-blocks for a moe_feed_forward block.

    These are display-only children that reflect the real MoE forward pass:
      Router → top-k selection → parallel expert FFNs → weighted combine.
    """
    p = block.params
    h = p.get("hidden_size", 768)
    num_experts = p.get("num_experts", 8)
    top_k = p.get("num_experts_per_tok", p.get("top_k", 2))
    inter = p.get("intermediate_size", h * 4)
    act = p.get("activation", "gelu")
    bid = block.id
    return [
        ArchBlock(
            id=f"{bid}_router",
            label="Router",
            type=BlockType.LINEAR,
            params={"in_features": h, "out_features": num_experts, "bias": False},
            notes=(
                f"Gating network (Linear {h}→{num_experts} + softmax). "
                f"Scores every expert and selects the top-{top_k} for this token."
            ),
        ),
        ArchBlock(
            id=f"{bid}_experts",
            label=f"Expert FFN",
            type=BlockType.FEED_FORWARD,
            params={"hidden_size": h, "intermediate_size": inter, "activation": act},
            repeat=num_experts,
            notes=(
                f"Pool of {num_experts} independent FFNs. "
                f"Only {top_k} activate per token — chosen by the router. "
                f"All {num_experts} sets of weights are stored; only {top_k} compute."
            ),
        ),
        ArchBlock(
            id=f"{bid}_combine",
            label="Weighted Combine",
            type=BlockType.UNKNOWN,
            params={},
            notes=(
                f"output = Σ softmax_weight_i × expert_i(x)  for i in top-{top_k}. "
                "Weights come from the router softmax over selected experts."
            ),
        ),
    ]


def _find_block(blocks: list[ArchBlock], target_id: str) -> tuple[list[ArchBlock], int] | None:
    """Find a block by id. Returns (parent_list, index) or None."""
    for i, block in enumerate(blocks):
        if block.id == target_id:
            return blocks, i
        if block.children:
            result = _find_block(block.children, target_id)
            if result:
                return result
    return None


def apply_edit(ir: ArchitectureIR, spec: EditSpec) -> EditResult:
    """Apply an edit spec to an IR and return the edit result."""
    new_ir = ir.model_copy(deep=True)
    diff: list[DiffEntry] = []

    match spec.op:
        case EditOp.SET_REPEAT:
            result = _find_block(new_ir.blocks, spec.target)
            if not result:
                raise ValueError(f"Block '{spec.target}' not found")
            parent_list, idx = result
            block = parent_list[idx]
            old_val = block.repeat
            new_val = int(spec.value)
            block.repeat = new_val
            diff.append(DiffEntry(path=f"{spec.target}.repeat", old=old_val, new=new_val))

        case EditOp.SET_PARAM:
            result = _find_block(new_ir.blocks, spec.target)
            if not result:
                raise ValueError(f"Block '{spec.target}' not found")
            parent_list, idx = result
            block = parent_list[idx]
            old_val = block.params.get(spec.key)
            block.params[spec.key] = spec.value
            diff.append(DiffEntry(path=f"{spec.target}.params.{spec.key}", old=old_val, new=spec.value))

        case EditOp.REMOVE_BLOCK:
            result = _find_block(new_ir.blocks, spec.target)
            if not result:
                raise ValueError(f"Block '{spec.target}' not found")
            parent_list, idx = result
            removed = parent_list.pop(idx)
            diff.append(DiffEntry(path=spec.target, old=removed.model_dump(), new=None))

        case EditOp.ADD_BLOCK:
            new_block = ArchBlock.model_validate(spec.block)
            if new_block.type == BlockType.MOE_FEED_FORWARD and not new_block.children:
                new_block.children = _build_moe_children(new_block)
            if spec.after:
                result = _find_block(new_ir.blocks, spec.after)
                if result:
                    parent_list, idx = result
                    parent_list.insert(idx + 1, new_block)
                else:
                    new_ir.blocks.append(new_block)
            else:
                new_ir.blocks.append(new_block)
            diff.append(DiffEntry(path=new_block.id, old=None, new=new_block.model_dump()))

        case EditOp.REPLACE_BLOCK:
            result = _find_block(new_ir.blocks, spec.target)
            if not result:
                raise ValueError(f"Block '{spec.target}' not found")
            parent_list, idx = result
            old_block = parent_list[idx]
            new_block = ArchBlock.model_validate(spec.block)
            if new_block.type == BlockType.MOE_FEED_FORWARD and not new_block.children:
                new_block.children = _build_moe_children(new_block)
            parent_list[idx] = new_block
            diff.append(DiffEntry(path=spec.target, old=old_block.model_dump(), new=new_block.model_dump()))

        case _:
            raise ValueError(f"Unknown op: {spec.op}")

    # Recompute stats
    old_compute = ir.compute
    new_compute = estimate_compute(new_ir)
    new_ir.compute = new_compute

    params_delta = new_compute.params_total - (old_compute.params_total if old_compute else 0)
    old_total = old_compute.params_total if old_compute and old_compute.params_total else 1
    params_pct = round(params_delta / old_total * 100, 2)

    mem_delta = None
    if old_compute and new_compute.memory_fp16_gb and old_compute.memory_fp16_gb:
        mem_delta = round(new_compute.memory_fp16_gb - old_compute.memory_fp16_gb, 3)

    flops_delta = None
    if old_compute and new_compute.flops_per_token and old_compute.flops_per_token:
        flops_delta = new_compute.flops_per_token - old_compute.flops_per_token

    compute_delta = ComputeDelta(
        params_delta=params_delta,
        params_delta_pct=params_pct,
        memory_fp16_delta_gb=mem_delta,
        flops_delta=flops_delta,
    )

    return EditResult(new_ir=new_ir, diff=diff, compute_delta=compute_delta)
