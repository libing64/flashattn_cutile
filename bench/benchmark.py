"""Speed and accuracy benchmark across cuTile kernels and library backends."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python -m bench.benchmark` from repo root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from attention import (
    attention_flash_attn,
    attention_math,
    attention_torch_flash,
    flash_attention_v1,
    flash_attention_v2,
    flash_attn_available,
    flash_attn_status,
    standard_attention,
)
from attention.utils import benchmark_cuda, error_metrics, make_qkv


def _build_impls(include_standard: bool = True):
    impls = []
    if include_standard:
        impls.append(("cutile_standard", standard_attention))
    impls.append(("cutile_fa_v1", flash_attention_v1))
    impls.append(("cutile_fa_v2", flash_attention_v2))
    impls.append(("torch_flash_sdpa", attention_torch_flash))
    if flash_attn_available():
        impls.append(("flash_attn", attention_flash_attn))
    return impls


def run_one(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    is_causal: bool,
    *,
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 5,
    iters: int = 20,
    skip_standard_above: int = 4096,
) -> list[dict]:
    q, k, v = make_qkv(batch, heads, seq_len, head_dim, dtype=dtype, seed=0)
    ref = attention_math(q, k, v, is_causal=is_causal)
    try:
        lib_ref = attention_torch_flash(q, k, v, is_causal=is_causal)
    except Exception:
        lib_ref = None

    include_standard = seq_len <= skip_standard_above
    rows = []
    for name, fn in _build_impls(include_standard=include_standard):
        try:
            # Compile / first-launch warmup outside timing for cuTile
            _ = fn(q, k, v, is_causal=is_causal)
            torch.cuda.synchronize()
            ms = benchmark_cuda(
                lambda fn=fn: fn(q, k, v, is_causal=is_causal),
                warmup=warmup,
                iters=iters,
            )
            out = fn(q, k, v, is_causal=is_causal)
            torch.cuda.synchronize()
            vs_math = error_metrics(out, ref)
            vs_lib = error_metrics(out, lib_ref) if lib_ref is not None else None
            rows.append(
                {
                    "impl": name,
                    "ms": ms,
                    "max_abs_math": vs_math["max_abs"],
                    "mean_abs_math": vs_math["mean_abs"],
                    "rel_rmse_math": vs_math["rel_rmse"],
                    "max_abs_flash": None if vs_lib is None else vs_lib["max_abs"],
                    "mean_abs_flash": None if vs_lib is None else vs_lib["mean_abs"],
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "impl": name,
                    "ms": None,
                    "max_abs_math": None,
                    "mean_abs_math": None,
                    "rel_rmse_math": None,
                    "max_abs_flash": None,
                    "mean_abs_flash": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def _fmt(x, width=10, prec=4):
    if x is None:
        return f"{'—':>{width}}"
    if isinstance(x, float):
        return f"{x:>{width}.{prec}f}"
    return f"{str(x):>{width}}"


def print_table(cfg: dict, rows: list[dict]):
    base_ms = next((r["ms"] for r in rows if r["impl"] == "cutile_fa_v1" and r["ms"]), None)
    if base_ms is None:
        base_ms = next((r["ms"] for r in rows if r["ms"]), None)

    header = (
        f"B={cfg['batch']} H={cfg['heads']} S={cfg['seq_len']} D={cfg['head_dim']} "
        f"causal={cfg['is_causal']} dtype={cfg['dtype']}"
    )
    print("\n" + "=" * 100)
    print(header)
    print("-" * 100)
    print(
        f"{'impl':<18} {'ms':>10} {'speedup':>10} "
        f"{'max|Δmath|':>12} {'mean|Δmath|':>12} {'max|Δflash|':>12}"
    )
    for r in rows:
        if r["error"]:
            print(f"{r['impl']:<18} ERROR: {r['error'][:70]}")
            continue
        sp = (base_ms / r["ms"]) if (base_ms and r["ms"]) else None
        print(
            f"{r['impl']:<18} {_fmt(r['ms'])} {_fmt(sp)} "
            f"{_fmt(r['max_abs_math'], 12, 5)} {_fmt(r['mean_abs_math'], 12, 5)} "
            f"{_fmt(r['max_abs_flash'], 12, 5)}"
        )


def main():
    parser = argparse.ArgumentParser(description="cuTile attention benchmark")
    parser.add_argument("--quick", action="store_true", help="Smaller sweep for smoke test")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Capability: {torch.cuda.get_device_capability()}")
    print(f"flash_attn: {flash_attn_status()}")

    if args.quick:
        configs = [
            dict(batch=1, heads=8, seq_len=512, head_dim=64, is_causal=False),
            dict(batch=1, heads=8, seq_len=512, head_dim=64, is_causal=True),
            dict(batch=1, heads=8, seq_len=1024, head_dim=64, is_causal=True),
        ]
    else:
        configs = []
        for batch in (1, 2):
            for heads in (8, 16):
                for seq_len in (512, 1024, 2048, 4096):
                    for head_dim in (64, 128):
                        for is_causal in (False, True):
                            configs.append(
                                dict(
                                    batch=batch,
                                    heads=heads,
                                    seq_len=seq_len,
                                    head_dim=head_dim,
                                    is_causal=is_causal,
                                )
                            )

    for cfg in configs:
        rows = run_one(
            cfg["batch"],
            cfg["heads"],
            cfg["seq_len"],
            cfg["head_dim"],
            cfg["is_causal"],
            warmup=args.warmup,
            iters=args.iters,
        )
        print_table({**cfg, "dtype": "bf16"}, rows)


if __name__ == "__main__":
    main()
