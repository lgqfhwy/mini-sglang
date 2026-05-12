"""
========================================================================
文件名: utils/__init__.py
所属模块: 通用工具模块入口
========================================================================

【utils 模块包含什么（各文件简介）】
- arch.py:        判断当前 GPU 架构（SM90 / SM100 等）
- hf.py:          HuggingFace 模型/tokenizer 加载（缓存 config / 下载权重）
- logger.py:      日志（按 rank 输出、彩色）
- misc.py:        小工具（div_ceil / div_even / align_ceil 等数学辅助）
- mp.py:          ZMQ 队列封装（同步 + 异步 PULL/PUSH/PUB/SUB 四种模式）
- registry.py:    Registry 类（注册表模式：字符串 → 工厂函数）
- torch_utils.py: PyTorch 相关（dtype context manager、nvtx 标注等）

【哪些是最常被其他模块用到的】
- init_logger:    几乎每个文件都用它打日志
- get/set_tp_info: 在 distributed/info.py，但被这里间接导出
- ZmqXxxQueue:    跨进程通信的核心
- Registry:       attention / moe / kvcache 模块都用它做"按字符串选实现"
"""

from .arch import is_arch_supported, is_sm90_supported, is_sm100_supported
from .hf import cached_load_hf_config, download_hf_weight, load_tokenizer
from .logger import init_logger
from .misc import UNSET, Unset, align_ceil, align_down, call_if_main, div_ceil, div_even
from .mp import (
    ZmqAsyncPullQueue,
    ZmqAsyncPushQueue,
    ZmqPubQueue,
    ZmqPullQueue,
    ZmqPushQueue,
    ZmqSubQueue,
)
from .registry import Registry
from .torch_utils import nvtx_annotate, torch_dtype

__all__ = [
    "cached_load_hf_config",
    "download_hf_weight",
    "load_tokenizer",
    "init_logger",
    "is_arch_supported",
    "is_sm90_supported",
    "is_sm100_supported",
    "call_if_main",
    "div_even",
    "div_ceil",
    "align_ceil",
    "align_down",
    "UNSET",
    "Unset",
    "torch_dtype",
    "nvtx_annotate",
    "Registry",
    "ZmqPushQueue",
    "ZmqPullQueue",
    "ZmqPubQueue",
    "ZmqSubQueue",
    "ZmqAsyncPushQueue",
    "ZmqAsyncPullQueue",
]
