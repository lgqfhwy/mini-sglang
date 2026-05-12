"""
========================================================================
文件名: kernel/store.py
所属模块: Kernel - 把 K/V 写入 KV pool 的指定 slot（CUDA kernel 封装）
========================================================================

store_cache(k_cache, v_cache, indices, k, v):
  把本步算出的 [num_tokens, num_kv_heads, head_dim] 形状的 k 和 v 张量，
  按 indices 给的 slot 编号写到 k_cache / v_cache 的对应位置。

用 CUDA kernel 实现（csrc/jit/store.cu）而不是 PyTorch 索引——少一次显存
拷贝、对向量化 store 友好，性能比 k_cache[indices] = k 提升明显。
========================================================================
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache
def _jit_store_module(
    element_size: int,
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, *config)
    return load_jit(
        "store",
        *args,
        cuda_files=["store.cu"],
        cuda_wrappers=[("launch", f"StoreKernel<{args}>::run")],
    )


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    num_tokens = k_cache.shape[0]
    k_cache = k_cache.view(num_tokens, -1)
    v_cache = v_cache.view(num_tokens, -1)
    element_size = k_cache.shape[1] * k_cache.element_size()
    module = _jit_store_module(element_size)
    module.launch(k_cache, v_cache, indices, k, v)
