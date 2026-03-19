"use client";

import { Cpu, MemoryStick, Zap, Database, Clock } from "lucide-react";
import { useStore } from "@/lib/store";
import { formatParams, formatBytes, formatFlops } from "@/lib/utils";
import { HARDWARE_PRESETS, getHardware, decodeLatencyMs } from "@/lib/hardware";
import TaskSwitcher from "@/components/TaskSwitcher";

function formatLatency(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(0)} µs`;
  if (ms < 100) return `${ms.toFixed(1)} ms`;
  return `${ms.toFixed(0)} ms`;
}

function formatThroughput(ms: number): string {
  const tps = 1000 / ms;
  if (tps >= 1000) return `${(tps / 1000).toFixed(1)}k tok/s`;
  return `${tps.toFixed(0)} tok/s`;
}

export default function ComputeBar() {
  const { ir, selectedHardwareId, setSelectedHardwareId } = useStore();
  const c = ir?.compute;

  if (!c) return null;

  const hw = getHardware(selectedHardwareId);
  const latencyMs = decodeLatencyMs(c.params_total, hw.memBwGbs);

  // Determine whether this is an encoder-only model (latency label differs)
  const isEncoder = ir?.blocks.some(
    (b) => b.type === "transformer_stack" && b.params?.is_causal === false
  ) && !ir?.blocks.some((b) => b.id === "decoder");

  const stats = [
    {
      icon: <Cpu className="w-3.5 h-3.5" />,
      label: "Parameters",
      value: formatParams(c.params_total),
      sub: `${formatParams(c.params_embedding)} emb · ${formatParams(c.params_encoder)} enc`,
      tooltip:
        "Exact - derived directly from model config. " +
        "MLM heads weight-tied to embeddings are excluded to avoid double-counting.",
    },
    {
      icon: <MemoryStick className="w-3.5 h-3.5" />,
      label: "Memory (fp16)",
      value: c.memory_fp16_gb != null ? formatBytes(c.memory_fp16_gb) : "-",
      sub: c.memory_int4_gb != null ? `${formatBytes(c.memory_int4_gb)} @ int4` : "",
      tooltip:
        "Estimated - model weight memory only (params × 2 bytes for fp16). " +
        "Excludes KV cache, activations, and optimizer state.",
    },
    {
      icon: <Zap className="w-3.5 h-3.5" />,
      label: "FLOPs",
      value: c.flops_per_token != null ? formatFlops(c.flops_per_token) : "-",
      sub: "",
      tooltip:
        "Estimated - forward-pass FLOPs per token. " +
        "Counts attention QKV/O projections + FFN matrix multiplications. " +
        "LayerNorm, activations, and embeddings excluded (<1% of total).",
    },
  ];

  return (
    <div className="flex items-center gap-5 px-4 py-2 bg-slate-900/80 border-b border-slate-800 flex-shrink-0 overflow-x-auto">
      <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex-shrink-0">
        {ir?.display_name ?? ir?.name}
      </span>
      <div className="h-4 w-px bg-slate-700 flex-shrink-0" />

      {/* Core stats */}
      {stats.map((s) => (
        <div key={s.label} className="flex items-center gap-2 flex-shrink-0">
          <span className="text-slate-500">{s.icon}</span>
          <div>
            <div className="flex items-baseline gap-1.5">
              <span className="text-xs text-slate-400">{s.label}</span>
              <span className="text-sm font-mono font-semibold text-white">
                {s.value}
              </span>
              {s.tooltip && (
                <span
                  title={s.tooltip}
                  className="text-[10px] text-slate-600 hover:text-slate-400 cursor-help select-none leading-none"
                >
                  ⓘ
                </span>
              )}
            </div>
            {s.sub && (
              <div className="text-[10px] text-slate-500">{s.sub}</div>
            )}
          </div>
        </div>
      ))}

      <div className="h-4 w-px bg-slate-700 flex-shrink-0" />

      {/* KV Cache - only for decoder / encoder-decoder models */}
      {c.kv_cache_fp16_gb != null && (
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-slate-500">
            <Database className="w-3.5 h-3.5" />
          </span>
          <div>
            <div className="flex items-baseline gap-1.5">
              <span className="text-xs text-slate-400">KV Cache</span>
              <span className="text-sm font-mono font-semibold text-white">
                {formatBytes(c.kv_cache_fp16_gb)}
              </span>
              <span
                title={
                  "Estimated - key/value cache memory at fp16. " +
                  "Formula: 2 (K+V) × layers × kv_heads × head_dim × seq_len × 2 bytes. " +
                  "For GQA models (LLaMA, Mistral) uses the reduced num_kv_heads. " +
                  "For T5 includes both decoder self-attention and cross-attention KV."
                }
                className="text-[10px] text-slate-600 hover:text-slate-400 cursor-help select-none leading-none"
              >
                ⓘ
              </span>
            </div>
            <div className="text-[10px] text-slate-500">
              fp16 @ {(c.kv_cache_ref_seq_len ?? 2048).toLocaleString()} tokens
            </div>
          </div>
        </div>
      )}

      {/* Latency estimate with hardware selector */}
      <div className="flex items-center gap-2 flex-shrink-0">
        <span className="text-slate-500">
          <Clock className="w-3.5 h-3.5" />
        </span>
        <div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-xs text-slate-400">Latency</span>
            <span className="text-sm font-mono font-semibold text-white">
              ~{formatLatency(latencyMs)}
            </span>
            <span className="text-[10px] text-slate-500">
              {isEncoder ? "/seq" : "/tok"}
            </span>
            <span
              title={
                "Estimated lower bound - memory-bandwidth-bound roofline model (batch=1, fp16). " +
                "Formula: model_bytes / memory_bandwidth. " +
                "Assumes 100% VRAM bandwidth utilization. " +
                "Real latency is typically 1.5-3× higher due to software overhead and compute constraints. " +
                (isEncoder ? "For encoder models: time per full-sequence forward pass." : "For decoder models: time per output token.")
              }
              className="text-[10px] text-slate-600 hover:text-slate-400 cursor-help select-none leading-none"
            >
              ⓘ
            </span>
          </div>
          <div className="flex items-center gap-1 text-[10px] text-slate-500">
            <span>{formatThroughput(latencyMs)} ·</span>
            {/* Hardware selector */}
            <select
              value={selectedHardwareId}
              onChange={(e) => setSelectedHardwareId(e.target.value)}
              className="bg-transparent text-indigo-400 border-none outline-none cursor-pointer hover:text-indigo-300 text-[10px] font-medium"
              title="Select reference hardware"
            >
              {HARDWARE_PRESETS.map((h) => (
                <option key={h.id} value={h.id} className="bg-slate-900 text-white">
                  {h.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Task head switcher - only renders for base encoder-only models */}
      <div className="ml-auto flex-shrink-0">
        <TaskSwitcher />
      </div>
    </div>
  );
}
