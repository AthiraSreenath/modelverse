import SearchBar from "@/components/SearchBar";
import ComputeBar from "@/components/panels/ComputeBar";
import DetailPanel from "@/components/panels/DetailPanel";
import ChatPanel from "@/components/panels/ChatPanel";
import ArchGraphClient from "@/components/graph/ArchGraphClient";
import ResizablePanel from "@/components/layout/ResizablePanel";

export default function Home() {
  return (
    <div className="h-screen flex flex-col bg-slate-950 overflow-hidden">
      {/* Top bar - z-50 ensures the search dropdown floats above the graph */}
      <header className="relative z-50 flex-shrink-0 flex items-center gap-4 px-4 h-14 border-b border-slate-800 bg-slate-900/80 backdrop-blur">
        <div className="flex items-center gap-2 flex-shrink-0">
          {/* eslint-disable-next-line @next/next/no-img-element -- static SVG mark */}
          <img
            src="/modelverse-mark.svg"
            alt=""
            width={32}
            height={32}
            className="h-8 w-8 flex-shrink-0 rounded-lg"
          />
          <span className="font-brand text-base font-semibold text-white tracking-tight hidden sm:block">
            ModelVerse
          </span>
        </div>

        <div className="flex-1 flex justify-center">
          <SearchBar />
        </div>

        <a
          href="https://github.com/athira/modelverse"
          target="_blank"
          rel="noreferrer"
          className="text-xs text-slate-500 hover:text-white transition-colors flex-shrink-0"
        >
          GitHub
        </a>
      </header>

      {/* Compute bar - shown only when model is loaded */}
      <ComputeBar />

      {/* Three-panel layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Graph panel - center / largest */}
        <main className="flex-1 flex overflow-hidden">
          <ArchGraphClient />
        </main>

        {/* Right side panels - draggable to expand leftward */}
        <ResizablePanel>
          {/* Detail panel - top half */}
          <div className="flex-1 overflow-hidden border-b border-slate-800">
            <DetailPanel />
          </div>

          {/* Chat panel - bottom half */}
          <div className="flex-1 overflow-hidden">
            <ChatPanel />
          </div>
        </ResizablePanel>
      </div>
    </div>
  );
}
