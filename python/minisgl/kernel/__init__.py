"""
========================================================================
文件名: kernel/__init__.py
所属模块: 自定义 CUDA / Triton kernel 入口
========================================================================

【这个模块包含什么】
项目自己写的几个 CUDA / Triton kernel：
- store.py:     把 K/V 张量写入 KV pool 的特定 slot（CUDA）
- index.py:     高效的 fancy indexing kernel
- radix.py:     fast_compare_key——比较两个 token 序列的最长公共前缀（CUDA）
- pynccl.py:    Python 包装的 NCCL collective 通信
- tensor.py:    一些 tensor 测试 helper
- moe_impl.py:  fused MoE 的 Triton 实现（fallback / 多平台）
- utils.py:     JIT 编译辅助
- store.py + 等其它: 通过 csrc/ 下的 C++/CUDA 源码 JIT 编译

【为什么自己写 kernel】
- 部分功能 PyTorch 没原生支持（如把 KV 写到 paged slot）
- PyTorch 通用算子开销大；定制 kernel 更快
- 集中维护，方便统一编译策略
"""

from .index import indexing
from .moe_impl import fused_moe_kernel_triton, moe_sum_reduce_triton
from .pynccl import PyNCCLCommunicator, init_pynccl
from .radix import fast_compare_key
from .store import store_cache
from .tensor import test_tensor

__all__ = [
    "indexing",
    "fast_compare_key",
    "store_cache",
    "test_tensor",
    "init_pynccl",
    "PyNCCLCommunicator",
    "fused_moe_kernel_triton",
    "moe_sum_reduce_triton",
]
