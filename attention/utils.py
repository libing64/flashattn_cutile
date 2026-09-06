"""Shared helpers: tensor generation, CUDA timing, error metrics."""

from __future__ import annotations

import statistics
from typing import Callable

import torch


def make_qkv(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    q = torch.randn(batch, heads, seq_len, head_dim, generator=g, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, generator=g, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, generator=g, device=device, dtype=dtype)
    return q, k, v


def sm_scale(head_dim: int) -> float:
    return 1.0 / (head_dim**0.5)


@torch.inference_mode()
def benchmark_cuda(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int = 5,
    iters: int = 20,
) -> float:
    """Return median kernel latency in milliseconds (CUDA events)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return float(statistics.median(times))


def error_metrics(out: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """Compare `out` against `ref` in float32."""
    a = out.detach().float()
    b = ref.detach().float()
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-6)
    rel = diff / denom
    mse = (a - b).pow(2).mean().item()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rel_rmse": float(mse**0.5),
        "max_rel": float(rel.max().item()),
    }
