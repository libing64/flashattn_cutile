"""cuTile standard (two-pass) attention: safe softmax without online rescale."""

from __future__ import annotations

import math

import cuda.tile as ct
import torch

INV_LOG_2 = 1.0 / math.log(2)
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


@ct.kernel
def _standard_attn_kernel(
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
    """Two-pass attention per Q tile: (1) max+sum (2) P@V / sum."""
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H

    qk_scale = qk_scale * INV_LOG_2

    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    offs_m = offs_m[:, None]
    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)[None, :]

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

    # ---- Pass 1: row-max and row-sum of exp2(qk*scale - max) ----
    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
    for j in range(0, Tc):
        k = ct.load(
            K,
            index=(batch_idx, head_idx, 0, j),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
        ).reshape((TILE_D, TILE_N))
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        if (CAUSAL or not EVEN_K) and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            mask = ct.full((TILE_M, TILE_N), True, dtype=ct.bool_)
            if not EVEN_K:
                mask = mask & (offs_n < k_seqlen)
            if CAUSAL:
                mask = mask & (offs_m >= offs_n)
            qk = qk + ct.where(mask, 0.0, -math.inf)

        m_i = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)

    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    for j in range(0, Tc):
        k = ct.load(
            K,
            index=(batch_idx, head_idx, 0, j),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
        ).reshape((TILE_D, TILE_N))
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        if (CAUSAL or not EVEN_K) and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            mask = ct.full((TILE_M, TILE_N), True, dtype=ct.bool_)
            if not EVEN_K:
                mask = mask & (offs_n < k_seqlen)
            if CAUSAL:
                mask = mask & (offs_m >= offs_n)
            qk = qk + ct.where(mask, 0.0, -math.inf)

        p = ct.exp2(qk * qk_scale - m_i, flush_to_zero=True)
        l_i = l_i + ct.sum(p, axis=-1, keepdims=True)

    # ---- Pass 2: accumulate P@V, then normalize ----
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)
    for j in range(0, Tc):
        k = ct.load(
            K,
            index=(batch_idx, head_idx, 0, j),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
        ).reshape((TILE_D, TILE_N))
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        if (CAUSAL or not EVEN_K) and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            mask = ct.full((TILE_M, TILE_N), True, dtype=ct.bool_)
            if not EVEN_K:
                mask = mask & (offs_n < k_seqlen)
            if CAUSAL:
                mask = mask & (offs_m >= offs_n)
            qk = qk + ct.where(mask, 0.0, -math.inf)

        p = ct.exp2(qk * qk_scale - m_i, flush_to_zero=True).astype(Q.dtype)
        v = ct.load(
            V,
            index=(batch_idx, head_idx, j, 0),
            shape=(1, 1, TILE_N, TILE_D),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((TILE_N, TILE_D))
        acc = ct.mma(p, v, acc)

    acc = ct.truediv(acc, l_i, flush_to_zero=True)
    ct.store(
        Out,
        index=(batch_idx, head_idx, bid_x, 0),
        tile=acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype),
    )


def standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    scale: float | None = None,
    tile_m: int = 64,
    tile_n: int = 64,
) -> torch.Tensor:
    """Launch two-pass cuTile attention. Tensors: [B, H, S, D]."""
    assert q.shape == k.shape == v.shape
    batch, heads, seq_len, head_dim = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)
    even_k = (seq_len % tile_n) == 0

    ct.launch(
        torch.cuda.current_stream(),
        (ct.cdiv(seq_len, tile_m), batch * heads, 1),
        _standard_attn_kernel,
        (q, k, v, out, scale, head_dim, heads, tile_m, tile_n, is_causal, even_k),
    )
    return out
