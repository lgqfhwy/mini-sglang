"""
========================================================================
文件名: layers/norm.py
所属模块: Layers - 归一化层 (RMSNorm)
========================================================================

【这个文件是做什么的】
封装 RMSNorm (Root Mean Square Normalization)——LLaMA、Qwen 等现代
LLM 都用它代替传统的 LayerNorm。

【RMSNorm 是什么】
传统 LayerNorm：x = (x - mean) / sqrt(var + eps) * weight + bias
RMSNorm：       x = x / sqrt(mean(x^2) + eps) * weight
区别：去掉中心化（减 mean）和 bias，只做 RMS 缩放。
优点：少一次 reduce 运算，更快；效果几乎不变。

【两个变体】
- RMSNorm:      普通的 RMSNorm
- RMSNormFused: 带"加 residual"融合——典型用法 `out = norm(x + residual)`
                把 add 和 norm 融合成一个 kernel，省一次显存读写

【实现】flashinfer 提供的 CUDA kernel，比 PyTorch 实现快 2-3 倍。
========================================================================
"""

from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    """
    标准 RMSNorm。
    weight 是可学习参数；eps 防除 0。
    forward_inplace 版本直接写回 x，省一次 alloc。
    """

    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        """原地版：写回 x，省一个临时 tensor。"""
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    """
    融合"加 residual"的 RMSNorm。
    典型用法：在 Transformer block 里 `x, residual = norm(x, residual)`，
    把 `x = x + residual; x = norm(x)` 融合成一个 kernel。
    """

    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        【功能】融合：x = norm(x + residual)；同时返回更新后的 residual（= 原 x + residual）。
        【参数】residual=None 表示第一层没有 residual，退化为普通 norm。
        """
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        # fused: x = norm(x + residual); residual 也被写成 x+residual（共享内存）
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
