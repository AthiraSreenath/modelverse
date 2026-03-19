/**
 * Zustand store - global client state for ModelVerse.
 */

import { create } from "zustand";
import type {
  ArchBlock,
  ArchitectureIR,
  ChatMessage,
  DiffEntry,
} from "./ir";
import { type TaskId, buildTaskHead, inferTask, isBaseEncoderModel } from "./taskHeads";
import { DEFAULT_HARDWARE_ID } from "./hardware";

interface ModelVerseState {
  // Current Architecture IR
  ir: ArchitectureIR | null;
  setIR: (ir: ArchitectureIR) => void;

  // IR history for undo/redo
  irHistory: ArchitectureIR[];
  pushIRHistory: (ir: ArchitectureIR) => void;
  undoIR: () => void;
  canUndo: boolean;

  // Which block is selected in the graph
  selectedBlockId: string | null;
  setSelectedBlockId: (id: string | null) => void;

  // Which blocks are expanded (transformer_stack nodes)
  expandedBlockIds: Set<string>;
  toggleBlockExpanded: (id: string) => void;

  // Latest diff (from an edit) - used to highlight changed nodes
  latestDiff: DiffEntry[];
  setLatestDiff: (diff: DiffEntry[]) => void;
  clearDiff: () => void;

  // Chat history
  chatMessages: ChatMessage[];
  addChatMessage: (msg: ChatMessage) => void;
  updateLastAssistantMessage: (text: string) => void;
  clearChat: () => void;

  // Task head switching (encoder-only models)
  activeTask: TaskId | null;
  switchTask: (task: TaskId) => void;

  // Hardware selector for latency estimation
  selectedHardwareId: string;
  setSelectedHardwareId: (id: string) => void;

  // Loading states
  isResolving: boolean;
  setIsResolving: (v: boolean) => void;
  isChatStreaming: boolean;
  setIsChatStreaming: (v: boolean) => void;
  resolveError: string | null;
  setResolveError: (e: string | null) => void;
}

export const useStore = create<ModelVerseState>((set, get) => ({
  ir: null,
  setIR: (ir) => set({
    ir,
    latestDiff: [],
    resolveError: null,
    activeTask: isBaseEncoderModel(ir) ? inferTask(ir.blocks) : null,
  }),

  irHistory: [],
  pushIRHistory: (ir) => {
    const { ir: current, irHistory } = get();
    if (!current) return;
    set({ irHistory: [...irHistory.slice(-19), current], ir });
  },
  undoIR: () => {
    const { irHistory } = get();
    if (!irHistory.length) return;
    const prev = irHistory[irHistory.length - 1];
    set({ ir: prev, irHistory: irHistory.slice(0, -1), latestDiff: [] });
  },
  get canUndo() {
    return get().irHistory.length > 0;
  },

  selectedBlockId: null,
  setSelectedBlockId: (id) => set({ selectedBlockId: id }),

  expandedBlockIds: new Set(),
  toggleBlockExpanded: (id) => {
    const { expandedBlockIds } = get();
    const next = new Set(expandedBlockIds);
    if (next.has(id)) {
      next.delete(id);
    } else {
      next.add(id);
    }
    set({ expandedBlockIds: next });
  },

  latestDiff: [],
  setLatestDiff: (diff) => set({ latestDiff: diff }),
  clearDiff: () => set({ latestDiff: [] }),

  chatMessages: [],
  addChatMessage: (msg) =>
    set((s) => ({ chatMessages: [...s.chatMessages, msg] })),
  updateLastAssistantMessage: (text) =>
    set((s) => {
      const msgs = [...s.chatMessages];
      const last = msgs[msgs.length - 1];
      if (last?.role === "assistant") {
        msgs[msgs.length - 1] = { ...last, content: last.content + text };
      } else {
        msgs.push({ role: "assistant", content: text });
      }
      return { chatMessages: msgs };
    }),
  clearChat: () => set({ chatMessages: [] }),

  activeTask: null,
  switchTask: (task) => {
    const { ir } = get();
    if (!ir) return;

    const encBlock = ir.blocks.find((b) => b.type === "transformer_stack");
    const h = (encBlock?.params?.hidden_size as number) ?? 768;
    const embBlock = ir.blocks.find((b) => b.type === "embedding");
    const vocabSize = (embBlock?.params?.vocab_size as number) ?? 30522;

    // Everything up to and including the last transformer_stack is the backbone
    let stackIdx = -1;
    for (let i = ir.blocks.length - 1; i >= 0; i--) {
      if (ir.blocks[i].type === "transformer_stack") { stackIdx = i; break; }
    }
    const backbone = ir.blocks.slice(0, stackIdx + 1);
    const newHead = buildTaskHead(task, h, vocabSize);
    const newBlocks = [...backbone, ...newHead];

    // Recalculate total params
    const allParams = newBlocks.reduce((s, b) => s + (b.param_count ?? 0), 0);
    const fp16 = (allParams * 2) / 1e9;

    // Task heads (linear classifiers) add negligible FLOPs vs the encoder stack.
    // Reuse backbone flops_per_token unchanged — it's dominated by attention + FFN.
    const newIR: ArchitectureIR = {
      ...ir,
      task,
      blocks: newBlocks,
      compute: ir.compute
        ? {
            ...ir.compute,
            params_total: allParams,
            params_head: newHead.reduce((s, b) => s + (b.param_count ?? 0), 0),
            memory_fp16_gb: fp16,
            memory_bf16_gb: fp16,
            memory_fp32_gb: (allParams * 4) / 1e9,
            memory_int8_gb: (allParams * 1) / 1e9,
            memory_int4_gb: (allParams * 0.5) / 1e9,
            // flops_per_token: backbone flops dominate; head adds 2*h*num_labels ≈ negligible
            flops_per_token: ir.compute.flops_per_token,
          }
        : ir.compute,
    };
    set({ ir: newIR, activeTask: task, latestDiff: [] });
  },

  selectedHardwareId: DEFAULT_HARDWARE_ID,
  setSelectedHardwareId: (id) => set({ selectedHardwareId: id }),

  isResolving: false,
  setIsResolving: (v) => set({ isResolving: v }),
  isChatStreaming: false,
  setIsChatStreaming: (v) => set({ isChatStreaming: v }),
  resolveError: null,
  setResolveError: (e) => set({ resolveError: e }),
}));
