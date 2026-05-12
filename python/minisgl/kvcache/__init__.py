"""
========================================================================
文件名: kvcache/__init__.py
所属模块: KV Cache 模块的包入口
========================================================================

【这个文件是做什么的】
1. 重新导出对外公开的接口（BasePrefixCache、BaseKVCachePool 等）；
2. 提供两个工厂函数：
   - create_kvcache_pool: 创建 KV 显存池（目前只支持 MHA 一种）
   - create_prefix_cache: 创建前缀缓存（"radix" 或 "naive"）；
3. 用 Registry 模式让"新增前缀缓存实现"只需 @SUPPORTED_CACHE_MANAGER.register("xxx")。

【为什么用 Registry 模式】
让"添加新缓存类型"对 import 顺序友好：
  - 注册函数只在被工厂调用时才真正 import 具体实现；
  - 避免模块加载时就把所有实现拉进来，省启动时间。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry

if TYPE_CHECKING:
    import torch
    from minisgl.models import ModelConfig

from .base import (
    BaseCacheHandle,
    BaseKVCachePool,
    BasePrefixCache,
    MatchResult,
    SizeInfo,
)


class CacheManagerCreator(Protocol):
    """
    【Protocol】描述"前缀缓存工厂函数"的签名：
        接受一个 device，返回一个 BasePrefixCache 实例
    用 Protocol 是因为我们想用函数（不是类）做 Registry 的值。
    """
    def __call__(self, device: torch.device) -> BasePrefixCache: ...


# Registry 模式：模块加载时只是登记，调用 SUPPORTED_CACHE_MANAGER["xxx"]
# 才真的拿到对应工厂函数。
SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> BaseKVCachePool:
    """
    【功能】根据模型配置和资源限制创建一个 KV 显存池。
    【目前实现】只支持 MHA（包括 MQA/GQA）模型；以后可扩展 MLA（DeepSeek）等。
    """
    from .mha_pool import MHAKVCache  # TODO: support other variants (e.g. MLA)

    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(device: torch.device):
    """工厂：朴素缓存（不共享前缀）。"""
    from .naive_cache import NaivePrefixCache

    return NaivePrefixCache(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device):
    """工厂：Radix Tree 缓存（多请求共享前缀，LRU 驱逐）。"""
    from .radix_cache import RadixPrefixCache

    return RadixPrefixCache(device=device)


def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    """
    【功能】根据 type 字符串（"naive" / "radix"）创建对应的前缀缓存实例。
    【调用方】CacheManager.__init__
    """
    return SUPPORTED_CACHE_MANAGER[type](device)


__all__ = [
    "create_kvcache_pool",
    "create_prefix_cache",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
    "SUPPORTED_CACHE_MANAGER",
]
