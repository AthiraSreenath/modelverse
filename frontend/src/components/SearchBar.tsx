"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  Search, Loader2, X, Undo2, Upload, CloudDownload, FileText,
} from "lucide-react";
import { useStore } from "@/lib/store";
import { resolveModel, uploadModel, listPrebaked } from "@/lib/api";
import { cn } from "@/lib/utils";

type Mode = "hub" | "upload";

const PLACEHOLDER_CYCLE = [
  "bert-base-uncased",
  "mistralai/Mistral-7B-v0.1",
  "google/flan-t5-xl",
  "meta-llama/Llama-3.1-8B",
  "gpt2",
];

const SUPPORTED_EXTS = [".safetensors", ".gguf", ".json", ".bin", ".pt", ".pth"];

export default function SearchBar() {
  const {
    setIR,
    setIsResolving,
    isResolving,
    resolveError,
    setResolveError,
    undoIR,
    canUndo,
    irHistory,
  } = useStore();

  // ── Hub mode state ────────────────────────────────────────────────────────
  const [query, setQuery]                 = useState("");
  const [suggestions, setSuggestions]     = useState<string[]>([]);
  const [showSugg, setShowSugg]           = useState(false);
  const [placeholderIdx, setPlaceholderIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // ── Upload mode state ─────────────────────────────────────────────────────
  const [mode, setMode]           = useState<Mode>("hub");
  const [dragOver, setDragOver]   = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadedName, setUploadedName] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Cycle placeholders
  useEffect(() => {
    const id = setInterval(
      () => setPlaceholderIdx((i) => (i + 1) % PLACEHOLDER_CYCLE.length),
      3000
    );
    return () => clearInterval(id);
  }, []);

  // Load prebaked suggestions
  useEffect(() => {
    listPrebaked().then(setSuggestions).catch(() => {});
  }, []);

  const filteredSugg =
    query.length > 1
      ? suggestions
          .filter((s) => s.toLowerCase().includes(query.toLowerCase()))
          .slice(0, 6)
      : [];

  // ── Hub resolve ───────────────────────────────────────────────────────────
  const resolveHub = async (input: string) => {
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
    resolveHub(query);
  };

  // ── File upload ───────────────────────────────────────────────────────────
  const handleFile = useCallback(
    async (file: File) => {
      setUploadError(null);
      setUploadedName(null);
      setIsUploading(true);
      try {
        const resp = await uploadModel(file);
        setUploadedName(file.name);
        setIR(resp.ir);
      } catch (e) {
        setUploadError(e instanceof Error ? e.message : "Failed to parse file");
      } finally {
        setIsUploading(false);
      }
    },
    [setIR]
  );

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    // Reset so the same file can be re-uploaded
    e.target.value = "";
  };

  const busy = isResolving || isUploading;
  const error = mode === "hub" ? resolveError : uploadError;

  return (
    <div className="relative flex items-center gap-2 w-full max-w-2xl">

      {/* ── Mode tabs ──────────────────────────────────────────────────── */}
      <div className="flex flex-shrink-0 rounded-lg bg-slate-800 border border-slate-700 p-0.5 gap-0.5">
        <button
          type="button"
          onClick={() => { setMode("hub"); setUploadError(null); }}
          className={cn(
            "flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap",
            mode === "hub"
              ? "bg-indigo-600 text-white shadow-sm"
              : "text-slate-400 hover:text-white"
          )}
        >
          <CloudDownload className="w-3 h-3" />
          HF Hub
        </button>
        <button
          type="button"
          onClick={() => { setMode("upload"); setResolveError(null); }}
          className={cn(
            "flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap",
            mode === "upload"
              ? "bg-indigo-600 text-white shadow-sm"
              : "text-slate-400 hover:text-white"
          )}
        >
          <Upload className="w-3 h-3" />
          Upload File
        </button>
      </div>

      {/* ── Hub search ─────────────────────────────────────────────────── */}
      {mode === "hub" && (
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
                if (resolveError) setResolveError(null);
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
                  onMouseDown={() => { setQuery(s); resolveHub(s); }}
                  className="w-full text-left px-3 py-2 text-sm text-slate-300 hover:bg-slate-700 hover:text-white flex items-center gap-2 transition-colors"
                >
                  <Search className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
                  {s}
                </button>
              ))}
            </div>
          )}
        </form>
      )}

      {/* ── File upload ─────────────────────────────────────────────────── */}
      {mode === "upload" && (
        <div className="flex-1 relative">
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept={SUPPORTED_EXTS.join(",")}
            onChange={handleFileInput}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            disabled={busy}
            className={cn(
              "w-full flex items-center gap-3 rounded-xl border px-3 h-10 text-sm transition-all cursor-pointer",
              dragOver
                ? "border-indigo-400 bg-indigo-500/10 text-indigo-300"
                : uploadError
                ? "border-red-500/60 bg-slate-800/80 text-red-400"
                : uploadedName
                ? "border-emerald-500/50 bg-slate-800/80 text-emerald-400"
                : "border-slate-700 bg-slate-800/80 text-slate-400 hover:border-indigo-500/50 hover:text-slate-300"
            )}
          >
            {isUploading ? (
              <Loader2 className="w-4 h-4 flex-shrink-0 animate-spin" />
            ) : uploadedName ? (
              <FileText className="w-4 h-4 flex-shrink-0" />
            ) : (
              <Upload className="w-4 h-4 flex-shrink-0" />
            )}
            <span className="flex-1 text-left truncate">
              {isUploading
                ? "Parsing file…"
                : uploadedName
                ? uploadedName
                : dragOver
                ? "Drop to parse"
                : "Drop file or click to browse"}
            </span>
            {/* Format badges */}
            {!uploadedName && !isUploading && (
              <span className="flex-shrink-0 text-xs text-slate-600 hidden sm:block">
                {SUPPORTED_EXTS.slice(0, 3).join(" · ")} …
              </span>
            )}
          </button>
        </div>
      )}

      {/* ── Undo button ─────────────────────────────────────────────────── */}
      {canUndo && (
        <button
          onClick={undoIR}
          title={`Undo (${irHistory.length} states)`}
          className="flex-shrink-0 p-2 text-slate-400 hover:text-white bg-slate-800 border border-slate-700 rounded-xl transition-colors"
        >
          <Undo2 className="w-4 h-4" />
        </button>
      )}

      {/* ── Error banner ─────────────────────────────────────────────────── */}
      {error && !(mode === "hub" && showSugg && filteredSugg.length > 0) && (
        <div className="absolute top-full mt-1 left-0 right-0 text-xs text-red-400 bg-red-400/5 border border-red-400/20 rounded-lg px-3 py-2 z-50">
          {error}
        </div>
      )}
    </div>
  );
}
