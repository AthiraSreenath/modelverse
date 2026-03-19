"use client";

import { useEffect, useRef } from "react";
import { useStore } from "@/lib/store";
import { TASK_LABELS, TASK_DESCRIPTIONS, type TaskId, isBaseEncoderModel } from "@/lib/taskHeads";
import { cn } from "@/lib/utils";

const TASKS: TaskId[] = ["mlm", "classification", "ner", "qa"];
const SEEN_KEY = "mv_task_switcher_seen";

export default function TaskSwitcher() {
  const { ir, activeTask, switchTask } = useStore();
  const isNew = useRef(
    typeof window !== "undefined" && !localStorage.getItem(SEEN_KEY)
  );

  // Mark as seen after first user interaction
  const handleSwitch = (task: TaskId) => {
    switchTask(task);
    if (isNew.current) {
      localStorage.setItem(SEEN_KEY, "1");
      isNew.current = false;
    }
  };

  // Reset "new" flag when a base encoder model loads so the hint reappears
  // (only if the user has never seen it at all)
  useEffect(() => {
    if (ir && isBaseEncoderModel(ir)) {
      isNew.current = !localStorage.getItem(SEEN_KEY);
    }
  }, [ir]);

  if (!ir || !isBaseEncoderModel(ir)) return null;

  return (
    <div className="flex items-center gap-2 px-2.5 py-1 rounded-lg border border-indigo-500/30 bg-indigo-500/5">
      {/* Section label */}
      <span className="text-[10px] text-indigo-400 uppercase tracking-widest font-bold flex-shrink-0 select-none">
        Fine-tune task
      </span>

      <div className="w-px h-3.5 bg-indigo-500/30 flex-shrink-0" />

      {/* Task pills */}
      <div className="flex items-center gap-0.5">
        {TASKS.map((task) => {
          const active = activeTask === task;
          return (
            <button
              key={task}
              onClick={() => handleSwitch(task)}
              title={TASK_DESCRIPTIONS[task]}
              className={cn(
                "relative px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-150",
                active
                  ? "bg-indigo-600 text-white shadow-sm shadow-indigo-900"
                  : "text-slate-400 hover:text-white hover:bg-slate-700/70"
              )}
            >
              {/* Pulse ring shown on active pill until user has clicked once */}
              {active && isNew.current && (
                <span className="absolute inset-0 rounded-md ring-2 ring-indigo-400 animate-ping opacity-60 pointer-events-none" />
              )}
              {TASK_LABELS[task]}
            </button>
          );
        })}
      </div>
    </div>
  );
}
