"use client";

import { useStore } from "@/lib/store";
import { TASK_LABELS, TASK_DESCRIPTIONS, type TaskId, isEncoderOnly } from "@/lib/taskHeads";
import { cn } from "@/lib/utils";

const TASKS: TaskId[] = ["mlm", "classification", "ner", "qa"];

export default function TaskSwitcher() {
  const { ir, activeTask, switchTask } = useStore();

  if (!ir || !isEncoderOnly(ir.blocks)) return null;

  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold flex-shrink-0">
        Task
      </span>
      <div className="flex items-center gap-1">
        {TASKS.map((task) => {
          const active = activeTask === task;
          return (
            <button
              key={task}
              onClick={() => switchTask(task)}
              title={TASK_DESCRIPTIONS[task]}
              className={cn(
                "px-2 py-0.5 rounded text-[11px] font-medium transition-all",
                active
                  ? "bg-indigo-600 text-white"
                  : "text-slate-400 hover:text-white hover:bg-slate-700"
              )}
            >
              {TASK_LABELS[task]}
            </button>
          );
        })}
      </div>
    </div>
  );
}
