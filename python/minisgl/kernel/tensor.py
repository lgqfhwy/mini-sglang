"""
========================================================================
文件名: kernel/tensor.py
所属模块: Kernel - tensor 相关 helper / 测试函数
========================================================================

提供 test_tensor 等小型 CUDA kernel 用于单元测试 / 基础 sanity check。
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
def _load_test_tensor_module() -> Module:
    return load_aot("test_tensor", cpp_files=["tensor.cpp"])


def test_tensor(x: torch.Tensor, y: torch.Tensor) -> int:
    return _load_test_tensor_module().test(x, y)
