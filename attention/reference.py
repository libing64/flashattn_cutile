"""Reference attention backends: math SDPA, Torch Flash SDPA, optional flash-attn."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _HAS_SDPA_KERNEL = True
except ImportError:  # pragma: no cover
    _HAS_SDPA_KERNEL = False

_FLASH_ATTN = None
_FLASH_ATTN_ERR: str | None = None
try:
    from flash_attn import flash_attn_func as _FLASH_ATTN  # type: ignore
except Exception as exc:  # pragma: no cover
    _FLASH_ATTN_ERR = f"{type(exc).__name__}: {exc}"


def flash_attn_available() -> bool:
    return _FLASH_ATTN is not None


def flash_attn_status() -> str:
    if _FLASH_ATTN is not None:
        return "available"
    return f"unavailable ({_FLASH_ATTN_ERR or 'not installed'})"


@torch.inference_mode()
def attention_math(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Numerically stable FP32 reference via SDPA MATH backend (or manual)."""
    scale = (1.0 / (q.size(-1) ** 0.5)) if scale is None else scale
    qf, kf, vf = q.float(), k.float(), v.float()
    if _HAS_SDPA_KERNEL:
        with sdpa_kernel(SDPBackend.MATH):
            out = F.scaled_dot_product_attention(
                qf, kf, vf, is_causal=is_causal, scale=scale
            )
        return out
    # Manual fallback
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    if is_causal:
        s = scores.size(-1)
        mask = torch.ones(s, s, device=scores.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, vf)


@torch.inference_mode()
def attention_torch_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Torch SDPA FLASH_ATTENTION backend (library baseline when flash-attn missing)."""
    if not _HAS_SDPA_KERNEL:
        raise RuntimeError("torch.nn.attention.sdpa_kernel is required")
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, scale=scale
        )


@torch.inference_mode()
def attention_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Dao et al. flash-attn package. Expects [B,H,S,D]; converts to [B,S,H,D]."""
    if _FLASH_ATTN is None:
        raise RuntimeError(f"flash_attn not available: {_FLASH_ATTN_ERR}")
    # flash_attn_func wants [B, S, H, D]
    q_ = q.transpose(1, 2).contiguous()
    k_ = k.transpose(1, 2).contiguous()
    v_ = v.transpose(1, 2).contiguous()
    out = _FLASH_ATTN(q_, k_, v_, softmax_scale=scale, causal=is_causal)
    return out.transpose(1, 2).contiguous()
