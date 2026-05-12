"""
========================================================================
文件名: kvcache/naive_cache.py
所属模块: KV Cache - 朴素前缀缓存（无共享）实现
========================================================================

【这个文件是做什么的 - 一句话总结】
这个文件是 BasePrefixCache 的最简实现——"什么都不缓存、什么都不共享"。
所有的 match 都返回"命中 0 个 token"，所有的 insert 也是空操作。
用于不想要前缀共享的场景（调试 / 基线对比 / 单用户）。

【为什么需要这个文件 / 这个模块存在的原因】
1. 提供基线参照：测 Radix 缓存的加速效果时，对比 naive 模式很方便；
2. 简化调试：怀疑 Radix 出 bug 时，临时切到 naive 排除嫌疑；
3. 充当"接口示范"：让读者用最少的代码看清 BasePrefixCache 要实现哪些方法。

【这个文件在整个推理流程中的位置】
   SchedulerConfig.cache_type = "naive"
     ↓
   create_prefix_cache 工厂返回 NaivePrefixCache 实例
     ↓
   CacheManager 拿它当 prefix_cache 用
     ↓
   每次 match 都返回 "miss"（cached_len=0），所以请求总是从头算 prompt
"""

import torch

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo


# ════════════════════════════════════════════════════════════════════
# NaiveCacheHandle: 朴素实现的"句柄"
# ────────────────────────────────────────────────────────────────────
# 它永远表示"命中长度为 0"——所以本身没有任何指向某段缓存的信息。
# get_matched_indices() 返回一个空张量。
#
# 注意：empty_tensor 是类变量（不是实例变量），由 NaivePrefixCache 在
# 初始化时统一设置。所有 handle 共享这一个空 tensor，省内存。
# ════════════════════════════════════════════════════════════════════
class NaiveCacheHandle(BaseCacheHandle):
    """
    【类名】NaiveCacheHandle
    【一句话描述】朴素前缀缓存的句柄——永远表示"命中 0 个 token"。
    """

    # 类级别共享的空 tensor，避免每个 handle 各自创建一份
    empty_tensor: torch.Tensor  # should be set by NaivePrefixCache

    def __init__(self):
        # cached_len 固定为 0：朴素模式永远命中 0 个 token
        super().__init__(cached_len=0)

    def get_matched_indices(self) -> torch.Tensor:
        """返回类级别的空 tensor。"""
        return self.empty_tensor


# ════════════════════════════════════════════════════════════════════
# NaivePrefixCache: 朴素前缀缓存
# ────────────────────────────────────────────────────────────────────
# 所有方法都是"占位实现"：
#   - lock_handle: 啥也不做（没东西可锁）
#   - match_prefix: 永远返回新的空 handle
#   - insert_prefix: 不真的插入，返回 cached_len=0 + 空 handle
#   - evict: 0 时返回空；非 0 抛 NotImplementedError（没东西可驱逐）
# ════════════════════════════════════════════════════════════════════
class NaivePrefixCache(BasePrefixCache):
    """
    【类名】NaivePrefixCache
    【一句话描述】最简的前缀缓存——不存任何条目，每次 match 都 miss。
    【用途】调试 / 测试 / 不需要前缀共享的场景。
    """

    def __init__(self, device: torch.device):
        self.device = device
        # 创建一个全局共享的空 tensor 给所有 handle 用
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        # 把 empty_tensor 灌到 NaiveCacheHandle 类的类变量里
        NaiveCacheHandle.empty_tensor = self.empty_tensor
        super().__init__()

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """朴素模式没有真实条目可锁——空操作。"""
        pass

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """永远 miss——返回一个 cached_len=0 的空 handle。"""
        return MatchResult(NaiveCacheHandle())

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """不真的插入——返回 cached_len=0 + 空 handle，表示"没有任何之前已存的部分要释放"。"""
        return InsertResult(0, NaiveCacheHandle())

    def evict(self, size: int) -> torch.Tensor:
        """
        【功能】朴素模式不支持驱逐。
        【唯一允许情况】size==0（调用者其实不需要驱逐）→ 返回空 tensor；
        【其它】抛 NotImplementedError——上游应避免在 naive 模式下需要驱逐。
        """
        if size == 0:
            return self.empty_tensor
        raise NotImplementedError("NaiveCacheManager does not support eviction.")

    def reset(self) -> None:
        """空操作——本来就没什么要清的。"""
        pass

    @property
    def size_info(self) -> SizeInfo:
        """两边都是 0——朴素模式从不占缓存空间。"""
        return SizeInfo(evictable_size=0, protected_size=0)

    def check_integrity(self) -> None:
        """空操作——朴素模式本来就没有可校验的状态。"""
        pass
