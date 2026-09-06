"""cuTile FlashAttention-v2: online softmax + causal loop split + fast math epilogue."""

from __future__ import annotations

import math

import cuda.tile as ct
import torch
from cuda.tile import RoundingMode as RMd

INV_LOG_2 = 1.0 / math.log(2)
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


def _default_tiles() -> tuple[int, int]:
    """sm_120 prefers 64x64; larger GPUs can use 128."""
    if not torch.cuda.is_available():
        return 64, 64
    cap = torch.cuda.get_device_capability()
    if cap[0] >= 10 and cap not in ((12, 0), (12, 1)):
        return 128, 128
    return 64, 64


@ct.kernel(occupancy=2)
def _flash_v2_kernel(
    Q,
    K,
    V,
    Out,
    qk_scale: float,
    TILE_D: ConstInt,
    H: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    CAUSAL: ConstBool,
    EVEN_K: ConstBool,
):
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H

    qk_scale = qk_scale * INV_LOG_2

    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    offs_m = offs_m[:, None]
    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)[None, :]

    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    q = ct.load(
        Q,
        index=(batch_idx, head_idx, bid_x, 0),
        shape=(1, 1, TILE_M, TILE_D),
        padding_mode=ct.PaddingMode.ZERO,
    ).reshape((TILE_M, TILE_D))

    k_seqlen = K.shape[2]
    if CAUSAL:
        m_end = (bid_x + 1) * TILE_M
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
        mask_start = min(bid_x * TILE_M // TILE_N, k_seqlen // TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
        mask_start = k_seqlen // TILE_N

    # Phase 1: fully unmasked KV tiles (FA-v2 causal split — skip mask FLOPs).
    for j in range(0, mask_start):
        k = ct.load(
            K,
            index=(batch_idx, head_idx, 0, j),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
        ).reshape((TILE_D, TILE_N))
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij
        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        v = ct.load(
            V,
            index=(batch_idx, head_idx, j, 0),
            shape=(1, 1, TILE_N, TILE_D),
            latency=4,
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((TILE_N, TILE_D))
        acc = ct.mma(p.astype(Q.dtype), v, acc)
        m_i = m_ij

    # Phase 2: diagonal / OOB tiles that need masking.
    for j in range(mask_start, Tc):
        k = ct.load(
            K,
            index=(batch_idx, head_idx, 0, j),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
        ).reshape((TILE_D, TILE_N))
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        offs_n = j * TILE_N + offs_n_tile
        mask = ct.full((TILE_M, TILE_N), True, dtype=ct.bool_)
        if not EVEN_K:
            mask = mask & (offs_n < k_seqlen)
        if CAUSAL:
            mask = mask & (offs_m >= offs_n)
        qk = qk + ct.where(mask, 0.0, -math.inf)

        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij
        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        v = ct.load(
            V,
            index=(batch_idx, head_idx, j, 0),
            shape=(1, 1, TILE_N, TILE_D),
            latency=4,
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((TILE_N, TILE_D))
        acc = ct.mma(p.astype(Q.dtype), v, acc)
        m_i = m_ij

    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    ct.store(
        Out,
        index=(batch_idx, head_idx, bid_x, 0),
        tile=acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype),
    )


def flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    scale: float | None = None,
    tile_m: int | None = None,
    tile_n: int | None = None,
) -> torch.Tensor:
    """Launch FA-v2 cuTile kernel. Tensors: [B, H, S, D]."""
    assert q.shape == k.shape == v.shape
    batch, heads, seq_len, head_dim = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    if tile_m is None or tile_n is None:
        dm, dn = _default_tiles()
        tile_m = tile_m or dm
        tile_n = tile_n or dn

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)
    even_k = (seq_len % tile_n) == 0

    ct.launch(
        torch.cuda.current_stream(),
        (ct.cdiv(seq_len, tile_m), batch * heads, 1),
        _flash_v2_kernel,
        (q, k, v, out, scale, head_dim, heads, tile_m, tile_n, is_causal, even_k),
    )
    return out
