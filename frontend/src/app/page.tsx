import SearchBar from "@/components/SearchBar";
import ComputeBar from "@/components/panels/ComputeBar";
import DetailPanel from "@/components/panels/DetailPanel";
import ChatPanel from "@/components/panels/ChatPanel";
import ArchGraphClient from "@/components/graph/ArchGraphClient";

export default function Home() {
  return (
    <div className="h-screen flex flex-col bg-slate-950 overflow-hidden">
      {/* Top bar */}
      <header className="flex-shrink-0 flex items-center gap-4 px-4 h-14 border-b border-slate-800 bg-slate-900/80 backdrop-blur">
        <div className="flex items-center gap-2 flex-shrink-0">
          <div className="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center">
            <span className="text-white text-xs font-bold">MV</span>
          </div>
          <span className="text-sm font-semibold text-white hidden sm:block">
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

      {/* Compute bar — shown only when model is loaded */}
      <ComputeBar />

      {/* Three-panel layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Graph panel — center / largest */}
        <main className="flex-1 flex overflow-hidden">
          <ArchGraphClient />
        </main>

        {/* Right side panels */}
        <aside className="flex-shrink-0 w-72 flex flex-col border-l border-slate-800 overflow-hidden">
          {/* Detail panel — top half */}
          <div className="flex-1 overflow-hidden border-b border-slate-800">
            <DetailPanel />
          </div>

          {/* Chat panel — bottom half */}
          <div className="flex-1 overflow-hidden">
            <ChatPanel />
          </div>
        </aside>
      </div>
    </div>
  );
}
