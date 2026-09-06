# cuTile FlashAttention

用 [cuTile Python](https://docs.nvidia.com/cuda/cutile-python)（`cuda.tile`）实现并对比：

| 实现 | 说明 |
|------|------|
| **Standard** | 两遍 KV 扫描：先算 row-max / row-sum，再算 `P@V`（无 online rescale） |
| **FlashAttention-v1** | 经典 online softmax，每 tile rescale `O`/`L`，统一 KV 循环 |
| **FlashAttention-v2** | online softmax + 因果循环拆分（无 mask 前缀 / 对角线）+ `exp2`/`APPROX` 收尾 |
| **Torch Flash SDPA** | `torch.nn.functional.scaled_dot_product_attention` + `FLASH_ATTENTION` 后端（库基准） |
| **flash-attn**（可选） | Dao et al. 官方包；在 sm_120 上可能无法安装 |

正确性以 **FP32 math SDPA** 为 ground truth。

## 环境

```bash
conda activate cutile
# 已验证: cuda-tile 1.5.0, torch 2.11+cu128, GPU sm_120 (RTX 5060 Ti)
cd /path/to/flashattn_cutile
```

可选安装官方 flash-attn（失败不影响主流程；本机 sm_120 上曾因 GitHub wheel 404/超时未能装上，对比默认使用 Torch Flash SDPA）：

```bash
pip install psutil
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

## 布局与 API

张量布局均为 `[B, H, S, D]`，dtype 推荐 `bfloat16`。

```python
from attention import (
    standard_attention,
    flash_attention_v1,
    flash_attention_v2,
    attention_math,
    attention_torch_flash,
)

out = flash_attention_v2(q, k, v, is_causal=True)
```

## 测试

```bash
python -m pytest tests/test_correctness.py -v
```

## 基准测试

```bash
# 冒烟（少量 shape）
python -m bench.benchmark --quick

# 完整扫表: B∈{1,2} H∈{8,16} S∈{512..4096} D∈{64,128} causal∈{T,F}
python -m bench.benchmark --warmup 5 --iters 20
```

输出列：

- `ms`：CUDA event 中位延迟
- `speedup`：相对 `cutile_fa_v1`
- `max|Δmath|` / `mean|Δmath|`：相对 FP32 math 的绝对误差
- `max|Δflash|`：相对 Torch Flash SDPA 的绝对误差

## 目录

```
attention/
  standard.py      # 两遍 standard
  flash_v1.py      # FA-v1
  flash_v2.py      # FA-v2
  reference.py     # math / torch flash / flash_attn
  utils.py
bench/benchmark.py
tests/test_correctness.py
```

## 说明

- 仅 **forward**；首版为标准 MHA（`H_q == H_kv`），无 GQA/varlen/backward。
- sm_120 上 TileGym / NVIDIA Blog 推荐默认 tile `64×64`；FA-v2 使用 `occupancy=2`。
- bf16 下相对 FP32 math 的 `max_abs` 约 `1e-3`–`1e-2` 量级属正常范围。
