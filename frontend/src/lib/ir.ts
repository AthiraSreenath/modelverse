/**
 * Architecture IR - TypeScript types.
 * Mirrors the Pydantic models in backend/app/models/ir.py exactly.
 */

export type SourceType =
  | "hf_config"
  | "file_header"
  | "prebaked"
  | "paper"
  | "web_synthesis"
  | "llm_knowledge";

export type SourceConfidence = "exact" | "high" | "medium" | "low";

export type BlockType =
  | "embedding"
  | "transformer_stack"
  | "multi_head_attention"
  | "feed_forward"
  | "moe_feed_forward"
  | "layer_norm"
  | "linear"
  | "conv1d"
  | "ssm"
  | "rnn"
  | "pooling"
  | "dropout"
  | "activation"
  | "add"
  | "unknown";

export interface ArchBlock {
  id: string;
  label: string;
  type: BlockType;
  params: Record<string, unknown>;
  repeat: number;
  children: ArchBlock[];
  param_count: number | null;
  notes: string | null;
  unknown_fields: string[];
}

export interface ComputeStats {
  params_total: number;
  params_embedding: number;
  params_encoder: number;
  params_head: number;
  flops_per_token: number | null;
  memory_fp32_gb: number | null;
  memory_fp16_gb: number | null;
  memory_bf16_gb: number | null;
  memory_int8_gb: number | null;
  memory_int4_gb: number | null;
}

export interface SourceRef {
  title: string;
  url: string;
  excerpt: string | null;
}

export interface ArchitectureIR {
  schema_version: string;
  name: string;
  display_name: string | null;
  family: string | null;
  task: string | null;
  /** Raw HuggingFace architectures list, e.g. ["BertForMaskedLM"]. Empty for non-HF sources. */
  architectures: string[];
  source: SourceType;
  source_confidence: SourceConfidence;
  source_refs: SourceRef[];
  blocks: ArchBlock[];
  compute: ComputeStats | null;
}

// ---------------------------------------------------------------------------
// Edit types
// ---------------------------------------------------------------------------

export type EditOp =
  | "set_repeat"
  | "set_param"
  | "add_block"
  | "remove_block"
  | "replace_block";

export interface EditSpec {
  op: EditOp;
  target: string;
  value?: unknown;
  key?: string;
  block?: Partial<ArchBlock>;
  after?: string;
}

export interface DiffEntry {
  path: string;
  old: unknown;
  new: unknown;
}

export interface ComputeDelta {
  params_delta: number;
  params_delta_pct: number;
  memory_fp16_delta_gb: number | null;
  flops_delta: number | null;
}

export interface EditResult {
  new_ir: ArchitectureIR;
  diff: DiffEntry[];
  compute_delta: ComputeDelta;
}

// ---------------------------------------------------------------------------
// API response types
// ---------------------------------------------------------------------------

export interface ResolveResponse {
  ir: ArchitectureIR;
  source: SourceType;
  cached: boolean;
  resolve_time_ms: number | null;
}

// ---------------------------------------------------------------------------
// Chat types
// ---------------------------------------------------------------------------

export type ChatEventType = "text" | "tool_call" | "tool_result" | "done" | "error";

export interface ChatEvent {
  type: ChatEventType;
  text?: string;
  tool?: string;
  input?: Record<string, unknown>;
  result?: unknown;
  error?: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  /** Accumulated tool call events for this message */
  toolEvents?: ChatEvent[];
  /** If this message triggered an IR edit, the result */
  editResult?: EditResult;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export function isExactSource(ir: ArchitectureIR): boolean {
  return ir.source_confidence === "exact" || ir.source_confidence === "high";
}

export function getBlockById(
  blocks: ArchBlock[],
  id: string
): ArchBlock | null {
  for (const block of blocks) {
    if (block.id === id) return block;
    const found = getBlockById(block.children, id);
    if (found) return found;
  }
  return null;
}
