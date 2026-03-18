"use client";

import dynamic from "next/dynamic";

const ArchGraph = dynamic(() => import("./ArchGraph"), {
  ssr: false,
  loading: () => (
    <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
      Loading graph engine…
    </div>
  ),
});

export default function ArchGraphClient() {
  return <ArchGraph />;
}
