"""
========================================================================
文件名: kernel/radix.py
所属模块: Kernel - Radix 树节点匹配的快速比较 kernel
========================================================================

fast_compare_key(a, b):
  比较两个 1-D 张量 a, b 的最长公共前缀长度（返回从 0 开始连续相同的元素数）。

用于 kvcache/radix_cache.py 的 _tree_walk：每次到一个节点要看"我和它的 key
能匹配多长"。Python 循环逐元素比慢，CUDA kernel 一次扫完。
========================================================================
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@functools.cache
def _load_radix_module() -> Module:
    return load_aot("radix", cpp_files=["radix.cpp"])


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    # compare 2 1-D int cpu tensors for equality
    return _load_radix_module().fast_compare_key(x, y)
