"use client";

import { useState, useEffect, useRef } from "react";
import { Search, Loader2, X, Undo2 } from "lucide-react";
import { useStore } from "@/lib/store";
import { resolveModel, listPrebaked } from "@/lib/api";
import { cn } from "@/lib/utils";

const PLACEHOLDER_CYCLE = [
  "bert-base-uncased",
  "mistralai/Mistral-7B-v0.1",
  "google/flan-t5-xl",
  "meta-llama/Llama-3.1-8B",
  "gpt2",
];

export default function SearchBar() {
  const {
    setIR,
    setIsResolving,
    isResolving,
    resolveError,
    setResolveError,
    ir,
    undoIR,
    canUndo,
    irHistory,
  } = useStore();

  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showSugg, setShowSugg] = useState(false);
  const [placeholderIdx, setPlaceholderIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Cycle through placeholder examples
  useEffect(() => {
    const id = setInterval(
      () => setPlaceholderIdx((i) => (i + 1) % PLACEHOLDER_CYCLE.length),
      3000
    );
    return () => clearInterval(id);
  }, []);

  // Load prebaked model list for suggestions
  useEffect(() => {
    listPrebaked().then(setSuggestions).catch(() => {});
  }, []);

  const filteredSugg = query.length > 1
    ? suggestions.filter((s) =>
        s.toLowerCase().includes(query.toLowerCase())
      ).slice(0, 6)
    : [];

  const resolve = async (input: string) => {
    const q = input.trim();
    if (!q) return;
    setResolveError(null);
    setIsResolving(true);
    setShowSugg(false);

    try {
      const resp = await resolveModel(q);
      setIR(resp.ir);
    } catch (e) {
      setResolveError(e instanceof Error ? e.message : "Failed to load model");
    } finally {
      setIsResolving(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    resolve(query);
  };

  const handleSugg = (s: string) => {
    setQuery(s);
    resolve(s);
  };

  return (
    <div className="relative flex items-center gap-2 w-full max-w-xl">
      <form onSubmit={handleSubmit} className="flex-1 relative">
        <div
          className={cn(
            "flex items-center gap-2 rounded-xl border bg-slate-800/80 backdrop-blur px-3 h-10 transition-all",
            resolveError
              ? "border-red-500/60"
              : "border-slate-700 focus-within:border-indigo-500/60"
          )}
        >
          {isResolving ? (
            <Loader2 className="w-4 h-4 text-slate-400 flex-shrink-0 animate-spin" />
          ) : (
            <Search className="w-4 h-4 text-slate-500 flex-shrink-0" />
          )}
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setShowSugg(true);
            }}
            onFocus={() => setShowSugg(true)}
            onBlur={() => setTimeout(() => setShowSugg(false), 150)}
            placeholder={`e.g. ${PLACEHOLDER_CYCLE[placeholderIdx]}`}
            className="flex-1 bg-transparent text-sm text-white placeholder-slate-500 focus:outline-none"
          />
          {query && (
            <button
              type="button"
              onClick={() => { setQuery(""); setResolveError(null); }}
              className="text-slate-500 hover:text-white"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Suggestions dropdown */}
        {showSugg && filteredSugg.length > 0 && (
          <div className="absolute top-full mt-1 left-0 right-0 bg-slate-800 border border-slate-700 rounded-xl shadow-xl overflow-hidden z-50">
            {filteredSugg.map((s) => (
              <button
                key={s}
                type="button"
                onMouseDown={() => handleSugg(s)}
                className="w-full text-left px-3 py-2 text-sm text-slate-300 hover:bg-slate-700 hover:text-white flex items-center gap-2 transition-colors"
              >
                <Search className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
                {s}
              </button>
            ))}
          </div>
        )}
      </form>

      {/* Undo */}
      {canUndo && (
        <button
          onClick={undoIR}
          title={`Undo (${irHistory.length} states)`}
          className="flex-shrink-0 p-2 text-slate-400 hover:text-white bg-slate-800 border border-slate-700 rounded-xl transition-colors"
        >
          <Undo2 className="w-4 h-4" />
        </button>
      )}

      {/* Error */}
      {resolveError && (
        <div className="absolute top-full mt-1 left-0 right-0 text-xs text-red-400 bg-red-400/5 border border-red-400/20 rounded-lg px-3 py-2 z-50">
          {resolveError}
        </div>
      )}
    </div>
  );
}
