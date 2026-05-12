"""
========================================================================
文件名: kvcache/base.py
所属模块: KV Cache 模块的抽象接口层
========================================================================

【这个文件是做什么的 - 一句话总结】
这个文件就像 KV Cache 系统的"插座标准"——定义了几个抽象基类（ABC）
和数据类，规定"任何一个 KV 缓存实现 / 任何一个前缀缓存实现长什么样
（有哪些方法、参数、返回什么）"。具体实现（NaivePrefixCache / RadixPrefixCache /
MHAKVCache）只需照着这个标准实现一遍即可。

【为什么需要这个文件 / 这个模块存在的原因】
mini-sglang 设计上让 KV Cache 是"可插拔"的：可以用朴素的（无共享）、
也可以用 Radix Tree（带前缀共享）。上层（如 CacheManager）不应关心
"具体用了哪种实现"，只面向抽象接口编程。
本文件就是这个抽象接口，下面 4 个抽象类各自代表一种角色：
  - BaseKVCachePool:  KV 显存池（真正存 K/V 张量的那块大显存）
  - BaseCacheHandle:  指向前缀缓存里"我们用了哪一段"的句柄
  - BasePrefixCache:  前缀缓存接口（match/insert/evict）
另外几个 NamedTuple 是这些方法的返回值打包。

【核心概念速览】

- KV Cache Pool（KV 缓存池）:
    真正在 GPU 显存里"放 K/V 张量"的那块巨大缓冲区。
    形状大致是 [layers, slots, kv_heads, head_dim]——每层每个 slot 存
    一个 token 的 K（或 V）。slot 是分配单位。

- Prefix Cache（前缀缓存）:
    一种"逻辑层"——它本身不存 K/V 张量数据，但它记录"哪段 token 序列
    在 KV pool 的哪些 slot 已经存好了，可以复用"。
    类比：图书馆的索引卡片——卡片不是书本身，而是"哪本书在第几号架"。

- handle（句柄）:
    "我用了前缀缓存的某一段"的引用计数凭证。持有 handle 期间，
    被引用的那段缓存不会被驱逐（evict）。

- evict（驱逐）:
    缓存空间不够时，把最久没用过的条目踢出去腾位置。

【关键设计决策】

1. 抽象基类 + 工厂函数模式：
   __init__.py 里的 create_prefix_cache(device, type) 根据 type 字符串
   返回不同实现。上层永远只 import BasePrefixCache。

2. handle 是 frozen dataclass：
   不可变——保证 ref_count 增减总是配对、不会被外部偷偷改字段。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import torch


# ════════════════════════════════════════════════════════════════════
# 抽象类: BaseKVCachePool
# ────────────────────────────────────────────────────────────────────
# 真正"在 GPU 显存里持有 K/V 张量"的池子。形状概念上是：
#   K: [num_layers, num_pages * page_size, num_kv_heads, head_dim]
#   V: 同上
# 它提供 4 个核心能力：
#   - k_cache(layer): 取第 layer 层的 K 张量（注意力 kernel 用）
#   - v_cache(layer): 取第 layer 层的 V 张量
#   - store_kv(k, v, out_loc, layer_id): 把本步算出的 k/v 写到指定 slot
#   - device/dtype/num_layers: 元信息
# ════════════════════════════════════════════════════════════════════
class BaseKVCachePool(ABC):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used.

    【抽象基类】定义 KV 缓存池的统一接口。
    具体实现见 kvcache/mha_pool.py 中的 MHAKVCache。
    """

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor:
        """
        【功能】取第 index 层的 K 缓存张量。
        【参数】index: Transformer 第几层（0 起步）
        【返回】GPU 张量，注意力 kernel 直接用 page_table 索引读它
        """
        ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor:
        """取第 index 层的 V 缓存张量。"""
        ...

    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        """
        【功能】把本步算出的 k 和 v 张量写到 KV pool 里 out_loc 指定的 slot。

        【参数】
        - k, v: 形状 [num_tokens, num_kv_heads, head_dim] 的 GPU 张量
        - out_loc: [num_tokens] 整数张量，每个元素是"该 token 的 KV 写到第几号 slot"
        - layer_id: 当前是 Transformer 的第几层

        【调用时机】每层的注意力子层在算完 k/v projection 后调用，
                    把这一层的 k/v 落盘到 KV pool。
        """
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """KV pool 所在的 GPU 设备"""
        ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """KV pool 的数据类型（通常 float16/bfloat16，省一半显存）"""
        ...

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Transformer 总层数"""
        ...


# ════════════════════════════════════════════════════════════════════
# 抽象类: BaseCacheHandle
# ────────────────────────────────────────────────────────────────────
# 句柄——"我用了前缀缓存的某一段"的引用凭证。
# 不同实现的 handle 内部可能持有不同对象（NaiveCacheHandle 啥也没有，
# RadixCacheHandle 持有一个 RadixTreeNode），但都要提供两个能力：
#   - cached_len: 这段命中的长度（token 数）
#   - get_matched_indices(): 这段对应在 KV pool 的 slot 编号张量
# ════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    """
    【抽象基类】前缀缓存的句柄——"我引用了缓存的哪一段"的凭证。
    frozen=True 让它不可变，避免 ref_count 配对错乱。
    """

    # 这段缓存对应的 token 数（也是上层判断"我跳过了多少 prompt"的依据）
    cached_len: int

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor:
        """
        【功能】返回这段缓存对应在 KV pool 的 slot 编号张量。
        【返回】形状 [cached_len] 的整数张量，每元素是一个 slot 编号
        【用途】调度器要把这些 slot 编号写到本请求在 page_table 中的对应行
        """
        ...


# ════════════════════════════════════════════════════════════════════
# 数据类: SizeInfo（缓存的"剩余空间"快照）
# ────────────────────────────────────────────────────────────────────
# 前缀缓存里的条目分两类：
#   - evictable（可驱逐）: ref_count==0，没人在用，可以踢
#   - protected（被锁住）: ref_count>0，有请求正在用，不能踢
# 总占用 = 两者之和。
# 调度器用 evictable_size 算"还能挪出多少空间"。
# ════════════════════════════════════════════════════════════════════
class SizeInfo(NamedTuple):
    """缓存空间分布快照。"""

    # 可驱逐的总 token 数（这些可以被踢出去为新请求腾空间）
    evictable_size: int
    # 被锁住保护的总 token 数（不能驱逐——有请求正持有 handle）
    protected_size: int

    @property
    def total_size(self) -> int:
        """缓存里总共占了多少 token 的空间。"""
        return self.evictable_size + self.protected_size


# ════════════════════════════════════════════════════════════════════
# 数据类: InsertResult（insert_prefix 的返回值）
# ────────────────────────────────────────────────────────────────────
# 把一段 token+slot 插入前缀缓存后返回：
#   - cached_len: 插入前已经存在于缓存里的那部分长度
#                 → 这部分对应的 slot 是"冗余的"（别人已经存了一份），
#                   调用者必须释放掉自己手里这份避免显存泄漏。
#   - handle: 新插入完成后的句柄（指向这段缓存）
# ════════════════════════════════════════════════════════════════════
class InsertResult(NamedTuple):
    cached_len: int  # length already in cache before insertion (should be freed)
    handle: BaseCacheHandle  # cache handle for the inserted prefix


# ════════════════════════════════════════════════════════════════════
# 数据类: MatchResult（match_prefix 的返回值）
# ────────────────────────────────────────────────────────────────────
# 仅含一个 cuda_handle —— 表示在 GPU 端的命中句柄。
# 注释 "TODO: support HiCache" 说未来想加 CPU 侧的二级缓存（HiCache）
# 那时会扩展为同时返回 cpu_handle 等。
# ════════════════════════════════════════════════════════════════════
class MatchResult(NamedTuple):
    cuda_handle: BaseCacheHandle
    # TODO: support HiCache


# ════════════════════════════════════════════════════════════════════
# 抽象类: BasePrefixCache
# ────────────────────────────────────────────────────────────────────
# 前缀缓存的核心接口。一个完整的前缀缓存实现要提供：
#   - lock_handle: 锁定/解锁某段缓存（保护不被驱逐）
#   - match_prefix: 给一段 token 序列，找到它能命中的最长前缀
#   - insert_prefix: 把一段 (token, slot) 序列插入缓存
#   - evict: 驱逐一定量的可驱逐条目（LRU）
#   - reset: 清空缓存
#   - size_info: 当前空间使用情况
#   - check_integrity: 自检（调试用）
# ════════════════════════════════════════════════════════════════════
class BasePrefixCache(ABC):
    """【抽象基类】前缀缓存的统一接口。"""

    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        Lock or unlock a cache handle.
        This operation will not modify the cache, but change the size info only.
        When a handle is locked, it cannot be evicted.
        Handles must be locked before the previously-returned tensor of `match_prefix` is used.
        Otherwise it may be evicted by calling evict.

        Args:
            handle (BaseCacheHandle): The cache handle to lock or unlock.
            unlock (bool): Whether to unlock the handle. Defaults to False.

        【功能】锁定或解锁一个句柄。
        【含义】lock 期间该段缓存的 ref_count > 0，不会被驱逐。
        【关键约束】调用 match_prefix 后必须立即 lock，否则 evict 时可能
                    把它踢出去，handle 持有的 slot 编号就指向了不存在的数据！
        """

    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """
        Match prefix and return the indices of the matched prefix in the cache.
        This operation will not modify the cache.
        The returned indices is only safe to use when the handle is locked.

        Args:
            input_ids (torch.Tensor): The input ids to match. Shape: (seq_len,)
        Returns:
            MatchResult: The match result containing the cache handles.

        【功能】给定一段 token id 序列，从缓存里找它能命中的最长前缀。
        【副作用】只读——不修改缓存。
        【返回的句柄安全期】只有在 lock 之后才能放心用，否则 evict 可能让它失效。
        """

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """
        Insert a new prefix into the cache.
        This operation will modify the cache.
        Args:
            input_ids (torch.Tensor): The input ids to insert. Shape: (seq_len,)
            indices (torch.Tensor): The indices to store the new prefix. Shape: (seq_len,)

        Returns:
            InsertResult: The result of the insertion.

        【功能】把 (token, slot) 序列插入缓存，供后续请求复用。
        【常见调用】Scheduler._process_last_data → cache_manager.cache_req →
                    本方法。
        """

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """
        Evict some prefixes from the cache to free up space.
        This operation will modify the cache.
        Note that evict 0 is always safe and does nothing.
        Note that the actual evict size may be larger than the requested size.
        Args:
            size (int): The size to evict.

        Returns:
            torch.Tensor: The indices evicted. Shape: (evict_size,)
        Raises:
            RuntimeError: If the requested size is larger than the evictable size.

        【功能】按 LRU 把"size 个 token 的可驱逐条目"踢出去，腾空间。
        【返回】被驱逐 token 对应的 slot 编号张量（调用者把这些 slot 放回 free 池）。
        【注意】实际驱逐数可能 ≥ size（一次必须按"叶子节点"为粒度驱逐，
                可能多踢一点）。
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the cache manager and the underlying cache.

        【功能】清空所有缓存条目。当前实现中很少调用。
        """

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        """Get the size information of the cache.

        【功能】返回当前可驱逐/被保护各占多少 token 的 SizeInfo。
        """

    @abstractmethod
    def check_integrity(self) -> None:
        """Check the integrity of the cache. Raise an error if the cache is corrupted.

        【功能】自检：验证内部数据结构一致（如树节点引用计数 == 实际持有数）。
        【用途】调度器空闲时调用做调试性校验。
        """
