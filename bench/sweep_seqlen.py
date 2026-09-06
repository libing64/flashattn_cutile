"""Sweep sequence length S in [512, 1M] and plot latency / TFLOPS curves.

Usage:
  python -m bench.sweep_seqlen
  python -m bench.sweep_seqlen --max-s 65536 --quick
  python -m bench.sweep_seqlen --causal --out results/seqlen_sweep
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import torch

from attention import (
    attention_flash_attn,
    attention_torch_flash,
    flash_attention_v1,
    flash_attention_v2,
    flash_attn_available,
    flash_attn_status,
    standard_attention,
)
from attention.utils import benchmark_cuda, error_metrics, make_qkv

# Log-spaced points covering [512, 1_048_576]
DEFAULT_SEQLENS = [
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
]

SKIP_STANDARD_ABOVE = 8192
SKIP_ACCURACY_ABOVE = 8192  # fp32 math materializes SxS


def _estimate_bytes(batch: int, heads: int, seq_len: int, head_dim: int) -> int:
    """Q+K+V+O bf16 footprint (approx)."""
    return 4 * batch * heads * seq_len * head_dim * 2


def _flops(batch: int, heads: int, seq_len: int, head_dim: int, is_causal: bool) -> float:
    """Approx attention FLOPs (QK^T + PV)."""
    flops = 2.0 * batch * heads * seq_len * seq_len * head_dim
    if is_causal:
        flops *= 0.5
    return flops


def _build_impls(seq_len: int):
    impls = []
    if seq_len <= SKIP_STANDARD_ABOVE:
        impls.append(("cutile_standard", standard_attention))
    impls.append(("cutile_fa_v1", flash_attention_v1))
    impls.append(("cutile_fa_v2", flash_attention_v2))
    impls.append(("torch_flash_sdpa", attention_torch_flash))
    if flash_attn_available():
        impls.append(("flash_attn", attention_flash_attn))
    return impls


def _adaptive_iters(seq_len: int, base_warmup: int, base_iters: int) -> tuple[int, int]:
    if seq_len >= 524288:
        return max(1, base_warmup // 5), max(2, base_iters // 10)
    if seq_len >= 131072:
        return max(1, base_warmup // 3), max(3, base_iters // 5)
    if seq_len >= 32768:
        return max(2, base_warmup // 2), max(5, base_iters // 2)
    return base_warmup, base_iters


def run_seqlen_sweep(
    seq_lens: list[int],
    *,
    batch: int = 1,
    heads: int = 8,
    head_dim: int = 64,
    is_causal: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 5,
    iters: int = 20,
) -> list[dict]:
    free, total = torch.cuda.mem_get_info()
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {free / 1024**3:.2f} / {total / 1024**3:.2f} GiB free")
    print(f"flash_attn: {flash_attn_status()}")
    print(
        f"Config: B={batch} H={heads} D={head_dim} causal={is_causal} "
        f"S={seq_lens[0]}..{seq_lens[-1]} ({len(seq_lens)} points)"
    )

    records: list[dict] = []
    for s in seq_lens:
        need = _estimate_bytes(batch, heads, s, head_dim)
        free, _ = torch.cuda.mem_get_info()
        print(f"\n--- S={s} (est tensors ~{need / 1024**3:.2f} GiB, free {free / 1024**3:.2f} GiB) ---")
        if need > free * 0.85:
            print(f"  SKIP: estimated footprint exceeds free VRAM")
            for name, _ in _build_impls(s):
                records.append(
                    {
                        "seq_len": s,
                        "impl": name,
                        "ms": None,
                        "tflops": None,
                        "max_abs_flash": None,
                        "error": "skipped_oom_estimate",
                    }
                )
            continue

        w, n = _adaptive_iters(s, warmup, iters)
        try:
            q, k, v = make_qkv(batch, heads, s, head_dim, dtype=dtype, seed=0)
        except torch.OutOfMemoryError as exc:
            print(f"  SKIP alloc OOM: {exc}")
            torch.cuda.empty_cache()
            for name, _ in _build_impls(s):
                records.append(
                    {
                        "seq_len": s,
                        "impl": name,
                        "ms": None,
                        "tflops": None,
                        "max_abs_flash": None,
                        "error": "alloc_oom",
                    }
                )
            continue

        lib_ref = None
        if s <= SKIP_ACCURACY_ABOVE:
            try:
                lib_ref = attention_torch_flash(q, k, v, is_causal=is_causal)
            except Exception:
                lib_ref = None

        flops = _flops(batch, heads, s, head_dim, is_causal)
        for name, fn in _build_impls(s):
            row = {
                "seq_len": s,
                "impl": name,
                "ms": None,
                "tflops": None,
                "max_abs_flash": None,
                "error": None,
            }
            try:
                _ = fn(q, k, v, is_causal=is_causal)
                torch.cuda.synchronize()
                ms = benchmark_cuda(
                    lambda fn=fn: fn(q, k, v, is_causal=is_causal),
                    warmup=w,
                    iters=n,
                )
                tflops = (flops / (ms * 1e-3)) / 1e12
                row["ms"] = ms
                row["tflops"] = tflops
                if lib_ref is not None:
                    out = fn(q, k, v, is_causal=is_causal)
                    torch.cuda.synchronize()
                    row["max_abs_flash"] = error_metrics(out, lib_ref)["max_abs"]
                print(f"  {name:<18} {ms:10.3f} ms  {tflops:8.2f} TFLOPS")
            except torch.OutOfMemoryError as exc:
                row["error"] = "oom"
                print(f"  {name:<18} OOM: {exc}")
                torch.cuda.empty_cache()
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  {name:<18} ERROR: {row['error'][:80]}")
            records.append(row)

        del q, k, v, lib_ref
        gc.collect()
        torch.cuda.empty_cache()

    return records


def save_csv(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["seq_len", "impl", "ms", "tflops", "max_abs_flash", "error"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    print(f"Wrote {path}")


def plot_curves(records: list[dict], out_prefix: Path, *, title_suffix: str):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    impls = []
    for r in records:
        if r["impl"] not in impls:
            impls.append(r["impl"])

    colors = {
        "cutile_standard": "#4C78A8",
        "cutile_fa_v1": "#F58518",
        "cutile_fa_v2": "#E45756",
        "torch_flash_sdpa": "#54A24B",
        "flash_attn": "#B279A2",
    }
    markers = {
        "cutile_standard": "o",
        "cutile_fa_v1": "s",
        "cutile_fa_v2": "D",
        "torch_flash_sdpa": "^",
        "flash_attn": "v",
    }

    # Latency
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in impls:
        xs, ys = [], []
        for r in records:
            if r["impl"] == name and r["ms"] is not None:
                xs.append(r["seq_len"])
                ys.append(r["ms"])
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            label=name,
            color=colors.get(name),
            marker=markers.get(name, "o"),
            linewidth=2,
            markersize=6,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length S")
    ax.set_ylabel("Latency (ms, median)")
    ax.set_title(f"Attention latency vs S {title_suffix}")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    lat_path = Path(str(out_prefix) + "_latency.png")
    fig.tight_layout()
    fig.savefig(lat_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {lat_path}")

    # TFLOPS
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in impls:
        xs, ys = [], []
        for r in records:
            if r["impl"] == name and r["tflops"] is not None:
                xs.append(r["seq_len"])
                ys.append(r["tflops"])
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            label=name,
            color=colors.get(name),
            marker=markers.get(name, "o"),
            linewidth=2,
            markersize=6,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length S")
    ax.set_ylabel("Effective TFLOPS")
    ax.set_title(f"Attention throughput vs S {title_suffix}")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    tflops_path = Path(str(out_prefix) + "_tflops.png")
    fig.tight_layout()
    fig.savefig(tflops_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {tflops_path}")

    # Speedup vs torch flash (when both present)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    torch_ms = {
        r["seq_len"]: r["ms"]
        for r in records
        if r["impl"] == "torch_flash_sdpa" and r["ms"] is not None
    }
    plotted = False
    for name in impls:
        if name == "torch_flash_sdpa":
            continue
        xs, ys = [], []
        for r in records:
            if r["impl"] != name or r["ms"] is None:
                continue
            base = torch_ms.get(r["seq_len"])
            if base is None or base <= 0:
                continue
            xs.append(r["seq_len"])
            ys.append(base / r["ms"])
        if not xs:
            continue
        plotted = True
        ax.plot(
            xs,
            ys,
            label=f"{name} / torch_flash",
            color=colors.get(name),
            marker=markers.get(name, "o"),
            linewidth=2,
            markersize=6,
        )
    if plotted:
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="parity")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Sequence length S")
        ax.set_ylabel("Speedup vs torch_flash_sdpa")
        ax.set_title(f"Speedup vs Torch Flash {title_suffix}")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend()
        sp_path = Path(str(out_prefix) + "_speedup.png")
        fig.tight_layout()
        fig.savefig(sp_path, dpi=160)
        print(f"Wrote {sp_path}")
    plt.close(fig)


def main():
    # Make progress visible when redirected / backgrounded
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Sweep S and plot attention curves")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--no-causal", action="store_true")
    parser.add_argument("--min-s", type=int, default=512)
    parser.add_argument("--max-s", type=int, default=1_048_576)
    parser.add_argument(
        "--seq-lens",
        type=str,
        default="",
        help="Comma-separated S list; default log2 points in [min-s, max-s]",
    )
    parser.add_argument("--quick", action="store_true", help="Fewer points, shorter timing")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--out",
        type=str,
        default="results/seqlen_sweep",
        help="Output prefix for csv/png (under repo root unless absolute)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        sys.exit(1)

    is_causal = not args.no_causal
    if args.seq_lens:
        seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    else:
        seq_lens = [s for s in DEFAULT_SEQLENS if args.min_s <= s <= args.max_s]
    if args.quick:
        seq_lens = [s for s in seq_lens if s <= 16384]
        args.warmup = min(args.warmup, 3)
        args.iters = min(args.iters, 8)

    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out

    records = run_seqlen_sweep(
        seq_lens,
        batch=args.batch,
        heads=args.heads,
        head_dim=args.head_dim,
        is_causal=is_causal,
        warmup=args.warmup,
        iters=args.iters,
    )
    save_csv(records, Path(str(out) + ".csv"))
    title = f"(B={args.batch}, H={args.heads}, D={args.head_dim}, causal={is_causal})"
    plot_curves(records, out, title_suffix=title)


if __name__ == "__main__":
    main()
