/**
 * Hardware presets for latency estimation.
 *
 * Latency model: memory-bandwidth-bound, batch=1.
 * At batch=1 the bottleneck for LLM inference is loading model weights from VRAM
 * for every token. This gives a tight lower bound on decode latency:
 *
 *   decode_latency_ms = model_bytes / (mem_bw_gb_s * 1e6)
 *
 * where model_bytes = params_total * dtype_bytes (2 for fp16).
 *
 * For encoder-only models (BERT) this represents the time per full-sequence
 * forward pass, not per output token.
 *
 * Sources:
 *   - NVIDIA H100 datasheet: 3.35 TB/s HBM3, 989 TFLOPS fp16 (non-sparse)
 *   - NVIDIA A100 80GB datasheet: 2.0 TB/s HBM2e, 312 TFLOPS fp16 (non-sparse)
 *   - NVIDIA RTX 4090: 1.008 TB/s GDDR6X, 82.6 TFLOPS fp32 -> 165 TFLOPS fp16
 *   - NVIDIA RTX 3090: 936 GB/s GDDR6X, 35.6 TFLOPS fp32 -> 71 TFLOPS fp16
 *   - NVIDIA T4: 300 GB/s HBM2, 65 TFLOPS fp16 (Tensor Core)
 */

export interface HardwarePreset {
  id: string;
  label: string;
  /** HBM / GDDR bandwidth in GB/s */
  memBwGbs: number;
  /** fp16 peak TFLOPS (non-sparse Tensor Core) */
  tflopsF16: number;
  /** VRAM capacity in GB */
  vramGb: number;
}

export const HARDWARE_PRESETS: HardwarePreset[] = [
  { id: "h100",    label: "H100 SXM",  memBwGbs: 3350, tflopsF16: 989, vramGb: 80 },
  { id: "a100",    label: "A100 80GB", memBwGbs: 2000, tflopsF16: 312, vramGb: 80 },
  { id: "rtx4090", label: "RTX 4090",  memBwGbs: 1008, tflopsF16: 165, vramGb: 24 },
  { id: "rtx3090", label: "RTX 3090",  memBwGbs: 936,  tflopsF16: 71,  vramGb: 24 },
  { id: "t4",      label: "T4",        memBwGbs: 300,  tflopsF16: 65,  vramGb: 16 },
];

export const DEFAULT_HARDWARE_ID = "a100";

export function getHardware(id: string): HardwarePreset {
  return HARDWARE_PRESETS.find((h) => h.id === id) ?? HARDWARE_PRESETS[1];
}

/**
 * Estimate decode latency (ms) using the memory-bandwidth-bound roofline model.
 *
 * Formula: (params_total * dtype_bytes) / (mem_bw_gb_s * 1e6)
 *   - params_total * dtype_bytes = model size in bytes (fp16 -> dtype_bytes=2)
 *   - mem_bw_gb_s * 1e6 = bandwidth in bytes/ms
 *
 * Interpretation:
 *   - Decoder models (GPT-2, LLaMA): latency per OUTPUT token, batch=1
 *   - Encoder models (BERT): latency per full-sequence forward pass, batch=1
 *   - This is a theoretical lower bound assuming 100% memory bandwidth utilization.
 *
 * @param paramsTotal  Total parameter count
 * @param memBwGbs     Hardware memory bandwidth in GB/s
 * @param dtypeBytes   Bytes per parameter (default 2 for fp16)
 */
export function decodeLatencyMs(
  paramsTotal: number,
  memBwGbs: number,
  dtypeBytes = 2
): number {
  // memBwGbs * 1e6 converts GB/s -> bytes/ms
  return (paramsTotal * dtypeBytes) / (memBwGbs * 1e6);
}
