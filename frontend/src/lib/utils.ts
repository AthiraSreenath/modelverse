import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return String(n);
}

export function formatBytes(gb: number): string {
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  return `${(gb * 1024).toFixed(0)} MB`;
}

export function formatFlops(flops: number): string {
  if (flops >= 1e12) return `${(flops / 1e12).toFixed(1)}T FLOPs/tok`;
  if (flops >= 1e9) return `${(flops / 1e9).toFixed(1)}B FLOPs/tok`;
  if (flops >= 1e6) return `${(flops / 1e6).toFixed(0)}M FLOPs/tok`;
  if (flops >= 1e3) return `${(flops / 1e3).toFixed(0)}K FLOPs/tok`;
  return `${flops} FLOPs/tok`;
}
