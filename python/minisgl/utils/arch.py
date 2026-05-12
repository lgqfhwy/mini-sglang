"""
========================================================================
文件名: utils/arch.py
所属模块: Utils - GPU 架构（SM）能力探测
========================================================================

【这个文件做什么】
查询当前 GPU 的"计算能力"（compute capability），判断是不是 H100（SM 9.0+）
或 Blackwell（SM 10.0+）。其他模块据此自动选最优的 kernel：
- SM 10+: 用 TensorRT-LLM backend（最快）
- SM 9+:  可以用 fused PDL kernel 等 Hopper 特性
- 更老: 退回到通用 kernel

【为什么用 functools.cache】
查询 device_capability 不便宜——缓存一次以后所有调用都是 O(1)。
"""

from __future__ import annotations

import functools
from typing import Tuple


@functools.cache
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    """返回 (major, minor) compute capability；没 CUDA 时返回 None。"""
    import torch
    import torch.version

    if not torch.cuda.is_available() or not torch.version.cuda:
        return None
    return torch.cuda.get_device_capability()


def is_arch_supported(major: int, minor: int = 0) -> bool:
    """当前 GPU 的能力是否 >= (major, minor)。"""
    arch = _get_torch_cuda_version()
    if arch is None:
        return False
    return arch >= (major, minor)


def is_sm90_supported() -> bool:
    """是否 SM 9.0+（H100 / H200 / Hopper 系列）。"""
    return is_arch_supported(9, 0)


def is_sm100_supported() -> bool:
    """是否 SM 10.0+（B100 / B200 / Blackwell 系列）。"""
    return is_arch_supported(10, 0)
