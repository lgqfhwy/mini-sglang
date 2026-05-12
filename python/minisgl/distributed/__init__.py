"""
========================================================================
文件名: distributed/__init__.py
所属模块: Distributed - 张量并行通信入口
========================================================================

【这个模块负责什么】
- info.py: TP rank/size 单例
- impl.py: 真正的 all-reduce / all-gather 通信封装（基于 pynccl 或 nccl）

【为什么不直接用 torch.distributed】
torch.distributed 在某些 NCCL 调用上有性能开销/边界情况。pynccl 是更
精细的 NCCL 自定义封装，对 mini-sglang 关心的几种 collective 操作做了
专门优化。
"""

from .impl import DistributedCommunicator, destroy_distributed, enable_pynccl_distributed
from .info import DistributedInfo, get_tp_info, set_tp_info, try_get_tp_info

__all__ = [
    "DistributedInfo",
    "get_tp_info",
    "set_tp_info",
    "enable_pynccl_distributed",
    "DistributedCommunicator",
    "try_get_tp_info",
    "destroy_distributed",
]
