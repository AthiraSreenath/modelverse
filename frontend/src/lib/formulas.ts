/**
 * Human-readable parameter formula strings for each layer type.
 * Used in the detail panel hover tooltips.
 */

import { formatParams } from "./utils";

export interface FormulaLine {
  label: string;
  value: number;
  formula: string;
}

type Params = Record<string, unknown>;

function num(p: Params, key: string, fallback = 0): number {
  return typeof p[key] === "number" ? (p[key] as number) : fallback;
}

export function getParamFormulas(
  blockType: string,
  params: Params
): FormulaLine[] {
  switch (blockType) {
    case "multi_head_attention": {
      const h = num(params, "hidden_size", 768);
      const heads = num(params, "num_heads", 12);
      const kvHeads = num(params, "num_kv_heads", heads);
      const headDim = num(params, "head_dim", h / heads);
      const q = h * heads * headDim;
      const k = h * kvHeads * headDim;
      const v = h * kvHeads * headDim;
      const o = heads * headDim * h;
      return [
        { label: "Q projection", value: q, formula: `${h} × ${heads} × ${headDim}` },
        { label: "K projection", value: k, formula: `${h} × ${kvHeads} × ${headDim}` },
        { label: "V projection", value: v, formula: `${h} × ${kvHeads} × ${headDim}` },
        { label: "Output projection", value: o, formula: `${heads * headDim} × ${h}` },
      ];
    }
    case "feed_forward": {
      const h = num(params, "hidden_size", 768);
      const inter = num(params, "intermediate_size", h * 4);
      const act = params["activation"] as string ?? "gelu";
      if (["swiglu", "geglu", "silu"].includes(act)) {
        return [
          { label: "Gate projection", value: h * inter, formula: `${h} × ${inter}` },
          { label: "Up projection", value: h * inter, formula: `${h} × ${inter}` },
          { label: "Down projection", value: inter * h, formula: `${inter} × ${h}` },
        ];
      }
      return [
        { label: "Up projection", value: h * inter + inter, formula: `${h} × ${inter} + bias` },
        { label: "Down projection", value: inter * h + h, formula: `${inter} × ${h} + bias` },
      ];
    }
    case "embedding": {
      const vocab = num(params, "vocab_size", 30522);
      const h = num(params, "hidden_size", 768);
      const maxPos = num(params, "max_position_embeddings", 512);
      const typeVocab = num(params, "type_vocab_size", 0);
      const lines: FormulaLine[] = [
        { label: "Token embeddings", value: vocab * h, formula: `${formatParams(vocab)} vocab × ${h}` },
      ];
      if (maxPos > 0 && params["position_embedding_type"] === "absolute") {
        lines.push({ label: "Position embeddings", value: maxPos * h, formula: `${maxPos} pos × ${h}` });
      }
      if (typeVocab > 0) {
        lines.push({ label: "Segment embeddings", value: typeVocab * h, formula: `${typeVocab} × ${h}` });
      }
      lines.push({ label: "Embedding LayerNorm", value: h * 2, formula: `${h} × 2` });
      return lines;
    }
    case "layer_norm": {
      const shape = num(params, "normalized_shape", 768);
      const normType = params["norm_type"] as string ?? "layer_norm";
      if (normType === "rms_norm") {
        return [{ label: "Weight (γ)", value: shape, formula: `${shape}` }];
      }
      return [
        { label: "Weight (γ)", value: shape, formula: `${shape}` },
        { label: "Bias (β)", value: shape, formula: `${shape}` },
      ];
    }
    case "linear": {
      const inF = num(params, "in_features");
      const outF = num(params, "out_features");
      const hasBias = params["bias"] !== false;
      const lines: FormulaLine[] = [
        { label: "Weight matrix", value: inF * outF, formula: `${inF} × ${outF}` },
      ];
      if (hasBias) {
        lines.push({ label: "Bias", value: outF, formula: `${outF}` });
      }
      return lines;
    }
    case "moe_feed_forward": {
      const h = num(params, "hidden_size", 4096);
      const inter = num(params, "intermediate_size", 14336);
      const experts = num(params, "num_experts", 8);
      const topK = num(params, "num_experts_per_tok", 2);
      return [
        { label: `${experts} SwiGLU experts`, value: experts * 3 * h * inter, formula: `${experts} × 3 × ${h} × ${inter}` },
        { label: "Router linear", value: h * experts, formula: `${h} × ${experts}` },
        { label: `Active params (top-${topK})`, value: topK * 3 * h * inter, formula: `${topK} of ${experts} experts active` },
      ];
    }
    default:
      return [];
  }
}
