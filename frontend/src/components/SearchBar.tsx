"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Search, Loader2, X, Undo2, Upload, FileText } from "lucide-react";
import { useStore } from "@/lib/store";
import { resolveModel, uploadModel, listPrebaked } from "@/lib/api";
import { cn } from "@/lib/utils";

const PLACEHOLDER_CYCLE = [
  "bert-base-uncased",
  "mistralai/Mistral-7B-v0.1",
  "google/flan-t5-xl",
  "meta-llama/Llama-3.1-8B",
  "gpt2",
];

const SUPPORTED_EXTS = [".safetensors", ".gguf", ".onnx", ".json", ".bin", ".pt", ".pth"];

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

  const [query, setQuery]                   = useState("");
  const [suggestions, setSuggestions]       = useState<string[]>([]);
  const [showSugg, setShowSugg]             = useState(false);
  const [placeholderIdx, setPlaceholderIdx] = useState(0);
  const inputRef    = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [dragOver, setDragOver]         = useState(false);
  const [isUploading, setIsUploading]   = useState(false);
  const [uploadError, setUploadError]   = useState<string | null>(null);
  const [uploadedName, setUploadedName] = useState<string | null>(null);

  useEffect(() => {
    const id = setInterval(
      () => setPlaceholderIdx((i) => (i + 1) % PLACEHOLDER_CYCLE.length),
      3000
    );
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    listPrebaked().then(setSuggestions).catch(() => {});
  }, []);

  const filteredSugg =
    query.length > 1
      ? suggestions.filter((s) => s.toLowerCase().includes(query.toLowerCase())).slice(0, 6)
      : [];

  const resolveHub = async (input: string) => {
    const q = input.trim();
    if (!q) return;
    setResolveError(null);
    setUploadError(null);
    setIsResolving(true);
    setShowSugg(false);
    try {
      const resp = await resolveModel(q);
      setIR(resp.ir);
      setUploadedName(null);
    } catch (e) {
      setResolveError(e instanceof Error ? e.message : "Failed to load model");
    } finally {
      setIsResolving(false);
    }
  };

  const handleFile = useCallback(
    async (file: File) => {
      setUploadError(null);
      setUploadedName(null);
      setResolveError(null);
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
    [setIR, setResolveError]
  );

  const error = resolveError || uploadError;

  return (
    <div className="relative flex items-center gap-2 w-full max-w-2xl">

      {/* ── Single joined bar ─────────────────────────────────────────── */}
      <div
        className={cn(
          "flex-1 flex items-stretch rounded-xl border bg-slate-800/80 backdrop-blur h-10 overflow-hidden transition-all",
          resolveError || uploadError
            ? "border-red-500/60"
            : "border-slate-700 focus-within:border-indigo-500/60"
        )}
      >
        {/* HF Hub label */}
        <span className="flex items-center px-3 text-xs font-medium text-slate-400 border-r border-slate-700 flex-shrink-0 whitespace-nowrap select-none">
          HF Hub
        </span>

        {/* Search input */}
        <form
          onSubmit={(e) => { e.preventDefault(); resolveHub(query); }}
          className="flex-1 flex items-center px-3 gap-2 min-w-0"
        >
          {isResolving
            ? <Loader2 className="w-3.5 h-3.5 text-slate-400 flex-shrink-0 animate-spin" />
            : <Search className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
          }
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
            className="flex-1 bg-transparent text-sm text-white placeholder-slate-500 focus:outline-none min-w-0"
          />
          {query && (
            <button
              type="button"
              onClick={() => { setQuery(""); setResolveError(null); }}
              className="text-slate-500 hover:text-white flex-shrink-0"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </form>

        {/* Upload section */}
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          accept={SUPPORTED_EXTS.join(",")}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
            e.target.value = "";
          }}
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            const f = e.dataTransfer.files[0];
            if (f) handleFile(f);
          }}
          disabled={isUploading}
          title={`Upload model file (${SUPPORTED_EXTS.join(", ")})`}
          className={cn(
            "flex items-center gap-1.5 px-3 text-xs font-medium border-l border-slate-700 flex-shrink-0 transition-colors whitespace-nowrap",
            dragOver
              ? "bg-indigo-500/10 text-indigo-300"
              : uploadError
              ? "text-red-400"
              : uploadedName
              ? "text-emerald-400"
              : "text-slate-400 hover:text-white hover:bg-slate-700/40"
          )}
        >
          {isUploading
            ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
            : uploadedName
            ? <FileText className="w-3.5 h-3.5" />
            : <Upload className="w-3.5 h-3.5" />
          }
          <span className="hidden sm:inline">
            {isUploading ? "Parsing…" : uploadedName ?? "Upload"}
          </span>
        </button>
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

      {/* ── Undo ─────────────────────────────────────────────────────── */}
      {canUndo && (
        <button
          onClick={undoIR}
          title={`Undo (${irHistory.length} states)`}
          className="flex-shrink-0 p-2 text-slate-400 hover:text-white bg-slate-800 border border-slate-700 rounded-xl transition-colors"
        >
          <Undo2 className="w-4 h-4" />
        </button>
      )}

      {/* ── Error ────────────────────────────────────────────────────── */}
      {error && !(showSugg && filteredSugg.length > 0) && (
        <div className="absolute top-full mt-1 left-0 right-0 text-xs text-red-400 bg-red-400/5 border border-red-400/20 rounded-lg px-3 py-2 z-50">
          {error}
        </div>
      )}
    </div>
  );
}
