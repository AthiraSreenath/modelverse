/**
 * Backend API client.
 * All calls go through this module - never call fetch directly.
 */

import type {
  ArchitectureIR,
  ChatMessage,
  ComputeStats,
  EditResult,
  EditSpec,
  ResolveResponse,
} from "./ir";

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BACKEND}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

/** Resolve a model name → Architecture IR */
export async function resolveModel(input: string): Promise<ResolveResponse> {
  return post<ResolveResponse>("/resolve", { input });
}

/** Compute stats for an IR */
export async function computeStats(
  ir: ArchitectureIR
): Promise<{ compute: ComputeStats }> {
  return post("/compute", { ir });
}

/** Apply an edit spec to an IR */
export async function editArchitecture(
  ir: ArchitectureIR,
  edit_spec: EditSpec
): Promise<EditResult> {
  return post<EditResult>("/edit", { ir, edit_spec });
}

/** List pre-baked model IDs */
export async function listPrebaked(): Promise<string[]> {
  const res = await fetch(`${BACKEND}/prebaked`);
  return res.json();
}

/** Upload a local model file (.safetensors, .gguf, .json, .bin, .pt, .pth) */
export async function uploadModel(file: File): Promise<{ ir: ArchitectureIR }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BACKEND}/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  // /upload returns the IR dict directly (not wrapped in {ir: ...})
  const ir = await res.json();
  return { ir };
}

/**
 * Stream a chat response from the LLM Brain.
 * Yields parsed ChatEvent objects.
 */
export async function* streamChat(
  messages: ChatMessage[],
  ir: ArchitectureIR
): AsyncGenerator<import("./ir").ChatEvent> {
  const res = await fetch(`${BACKEND}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
      ir,
    }),
  });

  if (!res.ok || !res.body) {
    yield { type: "error", error: `HTTP ${res.status}` };
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const json = line.slice(6).trim();
      if (!json) continue;
      try {
        yield JSON.parse(json);
      } catch {
        // malformed line - skip
      }
    }
  }
}
