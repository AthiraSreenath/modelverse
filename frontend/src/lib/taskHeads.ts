/**
 * Task-specific head block definitions for encoder-only (BERT-like) models.
 * Used by the TaskSwitcher to swap the head without a backend round-trip.
 */

import type { ArchBlock } from "./ir";

export type TaskId = "mlm" | "classification" | "ner" | "qa";

export const TASK_LABELS: Record<TaskId, string> = {
  mlm: "MLM",
  classification: "Classification",
  ner: "NER",
  qa: "QA",
};

export const TASK_DESCRIPTIONS: Record<TaskId, string> = {
  mlm: "Predict masked tokens",
  classification: "CLS → Linear → Softmax",
  ner: "Per-token labelling",
  qa: "Span start / end prediction",
};

/** Detect whether the current IR is an encoder-only (BERT-like) model. */
export function isEncoderOnly(blocks: ArchBlock[]): boolean {
  const hasEncoder = blocks.some(
    (b) => b.type === "transformer_stack" && b.params?.is_causal === false
  );
  const hasDecoder = blocks.some((b) => b.id === "decoder");
  return hasEncoder && !hasDecoder;
}

/**
 * Infer the active task from the last non-transformer-stack block label/id.
 * Falls back to "mlm" for BERT-like models.
 */
export function inferTask(blocks: ArchBlock[]): TaskId {
  const last = [...blocks].reverse().find((b) => b.type !== "transformer_stack");
  if (!last) return "mlm";
  const id = last.id.toLowerCase();
  const label = last.label.toLowerCase();
  if (id.includes("ner") || label.includes("ner") || label.includes("token class")) return "ner";
  if (id.includes("qa") || label.includes("span") || label.includes("qa")) return "qa";
  if (
    id.includes("classif") ||
    label.includes("classif") ||
    id.includes("cls_head")
  )
    return "classification";
  return "mlm";
}

/** Build head blocks for the given task. */
export function buildTaskHead(
  task: TaskId,
  h: number,
  vocabSize: number
): ArchBlock[] {
  const base: Omit<ArchBlock, "id" | "label" | "type" | "params" | "param_count" | "notes"> = {
    repeat: 1,
    children: [],
    unknown_fields: [],
  };

  switch (task) {
    case "mlm":
      return [
        {
          ...base,
          id: "lm_head",
          label: "MLM Head",
          type: "linear",
          params: { in_features: h, out_features: vocabSize },
          param_count: vocabSize * h,
          notes:
            `Linear(${h} → ${vocabSize.toLocaleString()}) applied to every token position. ` +
            "Predicts the probability distribution over the vocabulary for each masked position. " +
            "Weights are typically tied to the embedding matrix (no extra storage cost).",
        },
      ];

    case "classification":
      return [
        {
          ...base,
          id: "cls_extract",
          label: "CLS Token Extract",
          type: "unknown",
          params: {},
          param_count: 0,
          notes:
            "Takes the hidden state of the [CLS] token (position 0). " +
            "This single vector summarises the full input sequence for classification tasks.",
        },
        {
          ...base,
          id: "pooler",
          label: "Pooler (CLS projection)",
          type: "linear",
          params: { in_features: h, out_features: h, activation: "tanh" },
          param_count: h * h + h,
          notes:
            `Linear(${h} → ${h}) + Tanh. ` +
            "Projects the CLS vector into a 'pooled' representation. " +
            "Often fine-tuned or discarded in modern practice.",
        },
        {
          ...base,
          id: "classifier",
          label: "Classifier Head",
          type: "linear",
          params: { in_features: h, out_features: 2 },
          param_count: h * 2 + 2,
          notes:
            `Linear(${h} → num_classes) + Softmax. ` +
            "Shown here for binary classification (positive / negative). " +
            "Change out_features for multi-class tasks (e.g. 5 for SST-5 sentiment).",
        },
      ];

    case "ner": {
      const numLabels = 9; // CoNLL-2003 BIO tags
      return [
        {
          ...base,
          id: "ner_head",
          label: "Token Classifier",
          type: "linear",
          params: { in_features: h, out_features: numLabels },
          param_count: h * numLabels + numLabels,
          notes:
            `Linear(${h} → ${numLabels}) applied at every token position independently. ` +
            `${numLabels} labels = CoNLL-2003 BIO NER schema ` +
            "(B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC, O).",
        },
      ];
    }

    case "qa":
      return [
        {
          ...base,
          id: "qa_head",
          label: "Span Predictor",
          type: "linear",
          params: { in_features: h, out_features: 2 },
          param_count: h * 2 + 2,
          notes:
            `Linear(${h} → 2) outputs two logit vectors over sequence positions: ` +
            "start_logits and end_logits. " +
            "Answer = tokens[argmax(start) : argmax(end)+1]. " +
            "Used in SQuAD-style extractive QA.",
        },
      ];
  }
}
