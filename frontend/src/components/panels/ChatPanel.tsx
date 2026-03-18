"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Loader2, Wrench, AlertCircle } from "lucide-react";
import { useStore } from "@/lib/store";
import { streamChat } from "@/lib/api";
import type { ChatMessage, ChatEvent } from "@/lib/ir";
import { cn } from "@/lib/utils";

const SUGGESTION_QUERIES = [
  "What does the feed-forward layer do?",
  "Why does this model have this many attention heads?",
  "What if I halved the number of layers?",
  "How does this compare to BERT-large?",
];

function ToolCallBadge({ event }: { event: ChatEvent }) {
  const names: Record<string, string> = {
    search_huggingface: "Searching HuggingFace",
    search_web: "Searching the web",
    estimate_compute: "Estimating compute",
    apply_edit: "Applying edit",
    explain_layer: "Explaining layer",
  };

  return (
    <div className="flex items-center gap-1.5 text-[11px] text-violet-400/80 bg-violet-400/5 border border-violet-400/20 rounded px-2 py-1 w-fit">
      <Wrench className="w-3 h-3" />
      <span>{names[event.tool ?? ""] ?? event.tool}</span>
    </div>
  );
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === "user";

  return (
    <div className={cn("flex flex-col gap-1", isUser && "items-end")}>
      {msg.toolEvents && msg.toolEvents.length > 0 && (
        <div className="flex flex-col gap-1 mb-1">
          {msg.toolEvents.map((e, i) => (
            <ToolCallBadge key={i} event={e} />
          ))}
        </div>
      )}
      <div
        className={cn(
          "max-w-[90%] text-sm leading-relaxed rounded-xl px-3 py-2 whitespace-pre-wrap",
          isUser
            ? "bg-indigo-600 text-white rounded-br-sm"
            : "bg-slate-800 text-slate-200 rounded-bl-sm"
        )}
      >
        {msg.content || <span className="text-slate-500 animate-pulse">▌</span>}
      </div>
      {msg.editResult && (
        <div className="text-[11px] text-emerald-400/80 bg-emerald-400/5 border border-emerald-400/20 rounded px-2 py-1">
          Architecture updated · {msg.editResult.diff.length} changes ·{" "}
          {msg.editResult.compute_delta.params_delta_pct > 0 ? "+" : ""}
          {msg.editResult.compute_delta.params_delta_pct.toFixed(1)}% params
        </div>
      )}
    </div>
  );
}

export default function ChatPanel() {
  const {
    ir,
    chatMessages,
    addChatMessage,
    updateLastAssistantMessage,
    isChatStreaming,
    setIsChatStreaming,
    pushIRHistory,
    setLatestDiff,
  } = useStore();

  const [input, setInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  const send = useCallback(
    async (text: string) => {
      if (!ir || !text.trim() || isChatStreaming) return;

      setError(null);
      const userMsg: ChatMessage = { role: "user", content: text.trim() };
      addChatMessage(userMsg);
      setInput("");
      setIsChatStreaming(true);

      const toolEvents: ChatEvent[] = [];
      let assistantContent = "";
      let editResult: import("@/lib/ir").EditResult | undefined = undefined;

      addChatMessage({ role: "assistant", content: "" });

      try {
        const history = [...useStore.getState().chatMessages.slice(0, -1)];
        for await (const event of streamChat(history, ir)) {
          if (event.type === "text" && event.text) {
            assistantContent += event.text;
            updateLastAssistantMessage(event.text);
          } else if (event.type === "tool_call") {
            toolEvents.push(event);
          } else if (event.type === "tool_result") {
            if (event.tool === "apply_edit" && event.result) {
              const result = event.result as {
                new_ir?: unknown;
                diff?: unknown;
                compute_delta?: unknown;
              };
              if (result.new_ir) {
                editResult = result as import("@/lib/ir").EditResult;
                pushIRHistory(result.new_ir as import("@/lib/ir").ArchitectureIR);
                setLatestDiff((result.diff ?? []) as import("@/lib/ir").DiffEntry[]);
              }
            }
          } else if (event.type === "error") {
            setError(event.error ?? "Something went wrong");
          }
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Connection error");
      } finally {
        setIsChatStreaming(false);
        // Attach tool events to the last assistant message
        useStore.setState((s) => {
          const msgs = [...s.chatMessages];
          const last = msgs[msgs.length - 1];
          if (last?.role === "assistant") {
            msgs[msgs.length - 1] = {
              ...last,
              toolEvents,
              editResult,
            };
          }
          return { chatMessages: msgs };
        });
      }
    },
    [ir, isChatStreaming, addChatMessage, updateLastAssistantMessage, setIsChatStreaming, pushIRHistory, setLatestDiff]
  );

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {chatMessages.length === 0 && (
          <div className="flex flex-col gap-2 pt-2">
            <p className="text-xs text-slate-500 text-center mb-2">
              {ir ? "Ask anything about this model" : "Load a model first"}
            </p>
            {ir &&
              SUGGESTION_QUERIES.map((q) => (
                <button
                  key={q}
                  onClick={() => send(q)}
                  disabled={isChatStreaming}
                  className="text-left text-xs text-slate-400 hover:text-white bg-slate-800/60 hover:bg-slate-800 border border-slate-700/50 rounded-lg px-3 py-2 transition-colors"
                >
                  {q}
                </button>
              ))}
          </div>
        )}
        {chatMessages.map((msg, i) => (
          <MessageBubble key={i} msg={msg} />
        ))}
        {error && (
          <div className="flex items-center gap-2 text-xs text-red-400 bg-red-400/5 border border-red-400/20 rounded px-2 py-1.5">
            <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
            {error}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="flex-shrink-0 p-3 border-t border-slate-800">
        <div className="flex gap-2 items-end">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={ir ? "Ask or edit…" : "Load a model first"}
            disabled={!ir || isChatStreaming}
            rows={1}
            className="flex-1 resize-none bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-40 max-h-32 overflow-y-auto"
            style={{ minHeight: "2.5rem" }}
          />
          <button
            onClick={() => send(input)}
            disabled={!ir || !input.trim() || isChatStreaming}
            className="flex-shrink-0 p-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-lg transition-colors"
          >
            {isChatStreaming ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
