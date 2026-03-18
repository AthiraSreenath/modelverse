/**
 * Zustand store — global client state for ModelVerse.
 */

import { create } from "zustand";
import type {
  ArchBlock,
  ArchitectureIR,
  ChatMessage,
  DiffEntry,
} from "./ir";

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

  // Latest diff (from an edit) — used to highlight changed nodes
  latestDiff: DiffEntry[];
  setLatestDiff: (diff: DiffEntry[]) => void;
  clearDiff: () => void;

  // Chat history
  chatMessages: ChatMessage[];
  addChatMessage: (msg: ChatMessage) => void;
  updateLastAssistantMessage: (text: string) => void;
  clearChat: () => void;

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
  setIR: (ir) => set({ ir, latestDiff: [], resolveError: null }),

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

  isResolving: false,
  setIsResolving: (v) => set({ isResolving: v }),
  isChatStreaming: false,
  setIsChatStreaming: (v) => set({ isChatStreaming: v }),
  resolveError: null,
  setResolveError: (e) => set({ resolveError: e }),
}));
