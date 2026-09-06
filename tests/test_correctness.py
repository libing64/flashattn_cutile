"""Correctness tests for cuTile attention kernels vs FP32 math reference."""

from __future__ import annotations

import pytest
import torch

from attention import (
    attention_math,
    flash_attention_v1,
    flash_attention_v2,
    standard_attention,
)
from attention.utils import error_metrics, make_qkv


@pytest.mark.parametrize("seq_len", [64, 128, 256])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize(
    "fn",
    [standard_attention, flash_attention_v1, flash_attention_v2],
    ids=["standard", "fa_v1", "fa_v2"],
)
def test_matches_math(fn, seq_len, head_dim, is_causal):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    q, k, v = make_qkv(1, 4, seq_len, head_dim, dtype=torch.bfloat16, seed=42)
    ref = attention_math(q, k, v, is_causal=is_causal)
    out = fn(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()
    m = error_metrics(out, ref)
    # bf16 fused attention vs fp32 math — allow modest absolute error
    assert m["max_abs"] < 5e-2, m
    assert m["mean_abs"] < 1e-2, m


def test_non_divisible_seq():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    q, k, v = make_qkv(2, 2, 100, 64, dtype=torch.bfloat16, seed=7)
    ref = attention_math(q, k, v, is_causal=True)
    for fn in (standard_attention, flash_attention_v1, flash_attention_v2):
        out = fn(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        m = error_metrics(out, ref)
        assert m["max_abs"] < 8e-2, (fn.__name__, m)
