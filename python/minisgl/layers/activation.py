"""
========================================================================
文件名: layers/activation.py
所属模块: Layers - FFN 激活函数（融合"门控 + 激活 + 元素相乘"）
========================================================================

【这个文件是做什么的】
LLM 的 FFN（Feed-Forward Network）通常长这样：
    x = down_proj(silu(gate_proj(input)) * up_proj(input))
                  └──── act_and_mul ────┘
"silu_and_mul" 就是把"激活 + 逐元素相乘"融合成一个 CUDA kernel，
避免分两步分配中间 tensor。

【几种激活函数】
- silu (SwiGLU): LLaMA/Qwen 系列用，silu(x) = x * sigmoid(x)
- gelu (GeGLU): 部分模型用，gelu(x) = x * Phi(x)

【输入约定】
x 形状 [N, 2D]，前 D 维是"门控" gate，后 D 维是"上投影" up。
silu_and_mul(x) 返回 [N, D] = silu(gate) * up
========================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    """
    SwiGLU 激活：把 x 分成前后两半（gate, up），返回 silu(gate) * up。
    用 flashinfer 的 CUDA kernel 实现，比 PyTorch 单独算快很多。
    """
    from flashinfer import silu_and_mul

    return silu_and_mul(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    """GeGLU 激活：把 x 分成前后两半（gate, up），返回 gelu(gate) * up。"""
    from flashinfer import gelu_and_mul

    return gelu_and_mul(x, out=out)


__all__ = ["silu_and_mul", "gelu_and_mul"]
