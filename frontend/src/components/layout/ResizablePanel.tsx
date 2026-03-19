"use client";

import { useRef, useState, useCallback, useEffect } from "react";

const MIN_WIDTH = 240;
const MAX_WIDTH = 640;
const DEFAULT_WIDTH = 288; // w-72

interface Props {
  children: React.ReactNode;
}

/**
 * Right-side panel that can be dragged to expand leftward.
 * A 4 px drag handle sits on the left edge of the panel.
 */
export default function ResizablePanel({ children }: Props) {
  const [width, setWidth] = useState(DEFAULT_WIDTH);
  const isResizing = useRef(false);

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!isResizing.current) return;
    const next = window.innerWidth - e.clientX;
    setWidth(Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, next)));
  }, []);

  const stopResize = useCallback(() => {
    isResizing.current = false;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }, []);

  useEffect(() => {
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", stopResize);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", stopResize);
    };
  }, [onMouseMove, stopResize]);

  const startResize = () => {
    isResizing.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  };

  return (
    <aside
      style={{ width }}
      className="flex-shrink-0 flex border-l border-slate-800 overflow-hidden"
    >
      {/* Drag handle */}
      <div
        onMouseDown={startResize}
        className="flex-shrink-0 w-1 cursor-col-resize group relative hover:bg-indigo-500/40 transition-colors"
        title="Drag to resize"
      >
        {/* Visual line that highlights on hover */}
        <div className="absolute inset-y-0 left-0 w-px bg-slate-800 group-hover:bg-indigo-500/60 transition-colors" />
      </div>

      {/* Panel content */}
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        {children}
      </div>
    </aside>
  );
}
