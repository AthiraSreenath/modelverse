"use client";

import { Cpu, MemoryStick, Zap } from "lucide-react";
import { useStore } from "@/lib/store";
import { formatParams, formatBytes, formatFlops } from "@/lib/utils";

export default function ComputeBar() {
  const { ir } = useStore();
  const c = ir?.compute;

  if (!c) return null;

  const stats = [
    {
      icon: <Cpu className="w-3.5 h-3.5" />,
      label: "Parameters",
      value: formatParams(c.params_total),
      sub: `${formatParams(c.params_embedding)} emb · ${formatParams(c.params_encoder)} enc`,
    },
    {
      icon: <MemoryStick className="w-3.5 h-3.5" />,
      label: "Memory (fp16)",
      value: c.memory_fp16_gb != null ? formatBytes(c.memory_fp16_gb) : "—",
      sub: c.memory_int4_gb != null ? `${formatBytes(c.memory_int4_gb)} @ int4` : "",
    },
    {
      icon: <Zap className="w-3.5 h-3.5" />,
      label: "FLOPs",
      value: c.flops_per_token != null ? formatFlops(c.flops_per_token) : "—",
      sub: "",
    },
  ];

  return (
    <div className="flex gap-6 px-4 py-2 bg-slate-900/80 border-b border-slate-800 flex-shrink-0">
      <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider self-center">
        {ir?.display_name ?? ir?.name}
      </span>
      <div className="h-4 w-px bg-slate-700 self-center" />
      {stats.map((s) => (
        <div key={s.label} className="flex items-center gap-2">
          <span className="text-slate-500">{s.icon}</span>
          <div>
            <div className="flex items-baseline gap-1.5">
              <span className="text-xs text-slate-400">{s.label}</span>
              <span className="text-sm font-mono font-semibold text-white">
                {s.value}
              </span>
            </div>
            {s.sub && (
              <div className="text-[10px] text-slate-500">{s.sub}</div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
