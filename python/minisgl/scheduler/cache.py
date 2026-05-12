"""
========================================================================
文件名: scheduler/cache.py
所属模块: 调度器 - KV Cache 资源分配 + 前缀缓存协调层
========================================================================

【这个文件是做什么的 - 一句话总结】
这个文件是调度器与 KV Cache 池之间的"承包商"——负责按需向 KV pool
申请显存 slot、把分配信息写入页表、请求结束时把"用过的 KV"插入
前缀缓存以便后续请求复用。它把"分配/回收/缓存复用"这些杂事统一管
起来，让上层调度逻辑只用调几个方法。

【为什么需要这个文件 / 这个模块存在的原因】
直接操作 KV Cache pool 太底层，需要：
  1. 把 token 数换算成 page 数；
  2. 把 page 编号"展开"成每个 token 的 slot 编号；
  3. 把这些 slot 编号写回 page_table 对应行；
  4. 请求结束时把它的 prompt + 生成结果"喂"给 RadixCache，下次有
     相同前缀的请求就能直接复用；
  5. 显存不够时通过 LRU 驱逐部分缓存腾位置。
要让 PrefillManager / Scheduler 直接做这些会让它们变得臃肿——
CacheManager 把这套接口收拢起来。

【这个文件在整个推理流程中的位置】
   PrefillAdder 看上某个 PendingReq
     ↓
   ★ cache_manager.match_req() 查询前缀缓存命中多少 ★
     ↓ 命中部分的 KV 已经在显存
   PrefillManager 把请求加入本轮 batch
     ↓
   Scheduler._prepare_batch:
     ★ cache_manager.allocate_paged(reqs) 为每个请求新算的部分分页 ★
     ↓
   GPU 跑前向，K/V 写入 page_table 指向的 slot
     ↓ 一轮 decode 完成 / 整个请求完成
   ★ cache_manager.cache_req(req) 把已生成的内容插入前缀缓存 ★

【核心概念速览】

- KV Cache pool：
    显存上分配的一大块区域，切成等大的 slot（每个 slot 存 1 个或多个
    token 的 K/V）。CacheManager 不直接持有这块显存，但管"哪些 slot
    空闲/被谁占着"。

- page / slot：
    "slot" 是最小存储单位（存 1 个 token 的 KV）。
    "page" 是分配单位（一页含 page_size 个 slot）。
    page_size=1 时两者等价；page_size>1 时一次至少分配一页（page_size
    个 slot）。

- prefix cache（前缀缓存）：
    一种"过去算过的 KV 留着"的复用机制。比如用户都给 GPT 发 "你是
    一个有帮助的助手..." 这种 system prompt，第一个请求算完这些 token
    的 KV 后，把它插入 RadixCache；第二个请求来时发现前缀完全一样，
    直接复用之前的 KV，省下一大段 prefill 计算。
    类比：图书馆畅销书一个本一个本来借太慢，干脆把热门书复印好放
    "公共阅览区"，谁要直接拿。

- evict（驱逐）：
    缓存满了，要把一些"最近用得最少"的条目踢出去给新请求腾位置。
    RadixCache 实现 LRU 驱逐。

- handle（缓存句柄）：
    BaseCacheHandle 是 RadixCache 给请求的"占位证"。请求在使用某段
    缓存时拿着它，缓存就不能被驱逐（叫 lock）。请求用完了 unlock。

- lazy_free（懒释放）：
    一轮调度里可能有多个请求要回收 KV slot。如果每次释放都立刻改
    free_slots 张量，会产生大量小 tensor 拼接。所以用 context manager
    "暂存"所有要释放的 slot，最后一次性 cat 合并，大大减少开销。

【关键设计决策】

1. free_slots 用 GPU tensor 而不是 CPU list：
   分配出来的 slot 编号要写到 GPU 上的 page_table，如果是 CPU 列表，
   每次还要 host→device 拷贝。直接放 GPU 上省一次同步。

2. allocate_paged 一次给一批请求分配：
   amortize tensor 操作开销，比逐 req 调用快很多。

3. cache_req 注释里有"valid cache region"那张图非常关键：
   一段 token 区间在 prefill 后可能出现 4 种"片段语义"，要分别处理
   （见 cache_req 函数内部注释）。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Req
from minisgl.kvcache import BaseCacheHandle, MatchResult, create_prefix_cache
from minisgl.utils import div_ceil

if TYPE_CHECKING:
    from .utils import PendingReq


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：CacheManager（读代码之前请先读完这段）███
# ════════════════════════════════════════════════════════════════════
#
# 一、这个类要解决什么问题？
#
#   调度器需要的 KV 缓存层包括两块：
#     (A) 显存 slot 池：物理资源，固定大小，每个 slot 存 1 个 token 的 KV
#     (B) 前缀缓存：逻辑层，记录"哪些 token 序列的 KV 还在显存里、能复用"
#
#   CacheManager 是 (A)(B) 的统一门面，对外暴露:
#     - 查询: "我有 X 个 token 想算，能用上多少前缀缓存？"
#     - 分配: "给本批请求新算的 token 划页"
#     - 缓存/释放: 请求阶段性结束时把它的 KV 插入前缀缓存（或释放）
#
# 二、用具体例子走一遍（page_size = 1，简单起见）
#
#   假设系统有 16 个 slot，初始全空：
#     free_slots = tensor([0,1,2,...,15])
#     prefix_cache 是空的
#
#   【请求 A 来，prompt = [10, 20, 30, 40, 50]】
#     match_req(A) → 前缀缓存空，命中长度 = 0，handle.cached_len = 0
#     allocate_paged([A])（假设 A.cached_len=0, A.device_len=5）:
#       需要分配的页 = ceil(5/1) - ceil(0/1) = 5 页 = 5 个 slot
#       _allocate(5) → free_slots[:5] = [0,1,2,3,4]
#       free_slots 变成 [5,6,...,15]
#       page_table[A.table_idx, 0:5] = [0,1,2,3,4]
#         (把 A 的第 0~4 个 token 的 KV 存到 slot 0~4)
#     GPU 跑 prefill，把 A 的 5 个 token 的 K/V 写入 slot 0~4
#
#   【prefill 结束，调用 cache_req(A, finished=False)】
#     A.cached_len 此时已经更新到 5（5 个 token 都在显存里）
#     insert_prefix(A.input_ids[:5], page_indices=[0,1,2,3,4]):
#       前缀缓存把 (token序列 [10,20,30,40,50], slot [0,1,2,3,4])
#       记录下来。返回 (cached_len=5, new_handle)
#     - 这一段没释放任何 slot（finished=False，且没被别人提前缓存过）
#     - A.cache_handle 更新成 new_handle，并 lock 起来（防被驱逐）
#
#   【请求 B 来，prompt = [10, 20, 30, 99, 88]】
#     match_req(B) 跟前缀缓存匹配：[10,20,30] 命中！handle.cached_len=3
#     allocate_paged([B])（cached_len=3, device_len=5）:
#       需要分配 = ceil(5/1) - ceil(3/1) = 2 页
#       _allocate(2) → 取 free_slots[:2] = [5,6]
#       page_table[B.table_idx, 3:5] = [5,6]
#         （前 3 个位置不分配，因为复用了 A 的 [0,1,2]——已经在 cache 里）
#     prefill 只算 [99, 88] 这两个 token 的 K/V（省掉 60%）
#
#   【显存满了，新请求 C 需要 12 个 slot 但只剩 9 个】
#     _allocate(12) 发现 free_pages=9 不够：
#       evict( (12-9) * 1 = 3 个 slot )
#       前缀缓存按 LRU 驱逐 3 个 slot（被驱逐的那些 token 之后不能复用）
#       free_slots 增加 3 个 → 现在足够 12 个
#     继续分配
#
# 三、为什么有 "valid cache region" 那段长注释？
#
#   prefill 过程中一段 token 区间会经过 4 种语义片段：
#     [0, old_handle.cached_len)              — 前缀缓存里已有，复用
#     [old_handle.cached_len, req.cached_len) — 本次新分配并算出来的
#   插入到 prefix_cache 后:
#     [0, new_handle.cached_len)              — 现在被 prefix_cache 持有
#     [new_handle.cached_len, req.cached_len) — 没插进缓存（比如剩个零头不够一页）
#   还有一种特殊情况:
#     [old_handle.cached_len, cached_len)     — 我们刚算时缓存里没有，
#                                                 但与此同时别的请求也算了
#                                                 同样的前缀，已经插了。
#                                                 我们这份就是冗余，必须释放。
#   cache_req 必须正确处理这 4 类片段才不会显存泄漏。
#
# ════════════════════════════════════════════════════════════════════
class CacheManager:
    """
    【类名】CacheManager
    【一句话描述】调度器与 KV pool 之间的协调层：分配 slot、写页表、
                  把已算 KV 喂给 prefix cache 用作复用。
    【生活类比】仓库管理员：东西放哪个货架（slot 分配）、谁来取
                （cache 命中）、过期了就清掉（evict）都他管。
    """

    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor, type: str):
        """
        【功能】初始化分配器与前缀缓存。

        【参数】
        - num_pages (int): KV pool 总共切成多少页（资源上限）。
            page_size=1 时就是 slot 数。
        - page_size (int): 每页几个 slot。默认 1，部分注意力后端可能用 16 等。
        - page_table (torch.Tensor): 全局页表（GPU 上）。本类需要把
            分配出的 slot 编号写进去。
        - type (str): 前缀缓存类型："radix"（带前缀共享）或 "naive"（无）。

        【内部初始化】
        - free_slots: 全部 num_pages 个页的"首 slot 编号"列表，按页对齐。
            例如 page_size=2 时 free_slots = [0, 2, 4, 6, ...]
            每个元素代表一页，按页粒度分配，再展开成 slot。
        - prefix_cache: 通过工厂函数创建（具体类型由 type 决定）。
        """
        # NOTE: free_slots 是"按页对齐"的。每个元素是"一页的第一个 slot 的编号"。
        # 例如 page_size = 2，则 free_slots 形如 [0, 2, 4, 6, ...]——
        # 每个数代表"从这里开始往后 page_size 个 slot 都是同一页"。
        device = page_table.device
        # 用 arange 一次性在 GPU 上生成所有可用页编号
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        # 通过工厂函数创建具体的前缀缓存实现（Radix / Naive 等）
        self.prefix_cache = create_prefix_cache(device=device, type=type)
        # 保存一些便利字段
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size

    def match_req(self, req: PendingReq) -> MatchResult:
        """
        【功能】查询前缀缓存：本请求的 prompt 能复用多少 KV？

        【参数】req: 一个待办请求（PendingReq）
        【返回】MatchResult，里面含 cuda_handle.cached_len（命中长度）
                和 cuda_handle（用于后续 lock/复用）

        【为什么用 input_ids[:input_len-1] 而非 input_ids[:input_len]?】
        ⚠️ 注意：最后一个 token 不能匹配——因为我们至少要喂模型 1 个
        新 token 来产生 logits 做采样。如果整个 prompt 都"命中"了，
        就没有任何新计算，模型不会输出新 token。所以匹配上限 = len-1，
        保证至少有 1 个 token 要做真正的 prefill。
        """
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        """
        【功能】当前还有多少 slot 可以"拿来用"（不含已经被锁住的）。
        【公式】= prefix_cache 中可驱逐部分 + free_slots 中真正空着的
        【含义】前缀缓存里有些条目没被任何请求 lock，理论上可以为新请求驱逐
                出来。所以"实际可用"含这部分。
        """
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        """
        【功能】锁住一个前缀缓存句柄——锁住期间它的 slot 不会被驱逐。
        【调用时机】请求被接受、把它的 cache_handle 记下来时立即 lock。
        """
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        """
        【功能】解锁一个前缀缓存句柄——之后这部分可被驱逐。
        【调用时机】请求生成完毕、或被 abort 时。
        """
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: List[Req]) -> None:
        """
        【功能】给一批请求批量分配 KV slot 并把分配结果写入 page_table。

        【参数】reqs: 本轮 Batch 里的请求列表（含真实和 padding 占位）

        【内部逻辑】
        1. 遍历所有请求，按页粒度计算每个请求需要新分配多少页（first_page
           到 last_page 之间）。如果两者相等说明本轮不需要新分配。
        2. 把这些"分配需求"打包成 allocation_info（table_idx, first, last）。
        3. 一次性调用 _allocate(needed_pages) 拿到这么多页。
        4. 把"页编号"展开成"每个 token 的 slot 编号"（_page_to_token）。
        5. 调用 _write_page_table 把 slot 编号写入 page_table 对应位置。

        【具体例子（page_size=1）】
          req A: cached_len=3, device_len=5 → 需要分配 [3,4) 的页 = 2 页
          req B: cached_len=0, device_len=4 → 需要分配 [0,4) 的页 = 4 页
          allocation_info = [(A.table_idx, 3, 5), (B.table_idx, 0, 4)]
          needed_pages = 6
          _allocate(6) → e.g. [10,11,12,13,14,15]
          _write_page_table 把 page_table[A.table_idx, 3:5] = [10,11]
                          把 page_table[B.table_idx, 0:4] = [12,13,14,15]
        """
        # 统计总共需要多少页 + 每个请求的分配信息
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            # 把 token 长度向上取整到页（每页 page_size 个 token）
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                # 本请求新增了若干页（如果不变说明本轮不需要新分配）
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            # 拿到 needed_pages 个页（必要时会触发驱逐）
            # 然后把页编号展开成 slot 编号，写入 page_table
            allocated = self._page_to_token(self._allocate(needed_pages))
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        """
        【功能】请求阶段性"卸载"——把它已经算好的 KV 喂给前缀缓存（供复用），
                并视情况释放部分 slot。

        【调用时机】
        - 每完成 prefill 后（finished=False）：把 prompt 那段 KV 插入缓存；
        - 请求最终完成时（finished=True）：把所有 KV 插入缓存并释放掉
          末尾不能成页的零头 slot。

        【参数】
        - req: 要处理的请求
        - finished: 是否是该请求的"最后一次"（True = 请求结束）

        【4 种 token 区间的处理（见下方注释里的图）】
        这是本类最复杂的逻辑。注释里画了 6 个端点：
            [0, old_handle.cached_len)
              ─ prefill 开始前就在前缀缓存里的部分
            [old_handle.cached_len, cached_len)
              ─ 本次 prefill 新算并写入了 slot 的部分
            [cached_len, new_handle.cached_len)
              ─ 调用 insert_prefix 后，新被前缀缓存收下的部分
            [new_handle.cached_len, req.cached_len)
              ─ 没被收（比如不够一页对齐），尾巴部分

        要点：
        (a) old_handle 在请求接收时被 lock 住了，这里要 unlock；
        (b) [old_handle.cached_len, cached_len) 这段如果 cached_len <
            old_handle.cached_len 说明别人也算过同样前缀并已插入，
            我们这份是冗余，要释放；
        (c) 如果 finished：尾巴 [new_handle.cached_len, req.cached_len) 释放；
            否则保留并 lock 新 handle 等下一轮 decode 用。
        """
        # ==================================== valid cache region ====================================
        # [0, req.cached_len)                       This part is valid for attention kernel read/write.
        # [0, old_handle.cached_len)                This part is in the prefix cache before prefill.
        # [old_handle.cached_len, req.cached_len)   This part is allocated by cache manager for this request.
        # ================================== allocated cache region ==================================
        # [old_handle.cached_len, cached_len)       This part was not in the prefix cache when prefill,
        #                                           but later cached by other requests.
        #                                           We must free them to avoid memory leak.
        # [cached_len, new_handle.cached_len)       This part is newly inserted into the prefix cache.
        # [new_handle.cached_len, req.cached_len)   This part is tailing part that can not inserted into the prefix cache.
        #                                           We should free it if the request has finished.
        #
        # 用 req.input_ids[: req.cached_len] 作为要插入前缀缓存的 token 序列
        # 用 page_table[req.table_idx, :req.cached_len] 作为对应的 slot 编号
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle

        # 把这段 token+slot 插入前缀缓存；返回 (实际被缓存收下到第几位, 新句柄)
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)

        # 旧 handle 不再需要被锁住——释放它（注意：unlock 不会真的释放显存，
        # 只是把它从"不可驱逐"变回"可驱逐"）
        # unlock until all operations on handle is done
        self.unlock(old_handle)

        # 释放"我算了一遍但别人已经先插入缓存的"那段——避免显存泄漏
        # this part is already in the prefix cache, free it
        self._free(page_indices[old_handle.cached_len : cached_len])

        if finished:
            # 请求结束了——把尾巴零头释放（不进缓存的部分）
            self._free(page_indices[new_handle.cached_len :])
        else:
            # 还要继续 decode——把 new_handle 设为请求的新 handle 并 lock
            req.cache_handle = new_handle
            self.lock(new_handle)

    def check_integrity(self) -> None:
        """
        【功能】调试 / 自检：验证 "缓存里的页数 + 空闲页数 == 总页数"
                以及 free_slots 全部按 page_size 对齐。
        【调用时机】run_when_idle（调度器空闲时）做一次完整性检查。
        """
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            # 所有空闲槽都应该是某页的首 slot（按 page_size 对齐）
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self):
        """
        【功能】上下文管理器，让区域内所有 _free 调用"暂存"起来，
                离开时一次性合并到 free_slots，减少 tensor 操作次数。

        【用法】
            with cache_manager.lazy_free_region():
                ...    # 这里多次调用 self._free(...) 都不会立刻 cat
            # 离开 with 后 self.free_slots 才被一次性更新

        【为什么】
        ⚡ 性能关键: 单次 torch.cat 在 GPU 上有几十微秒开销；一轮调度
        可能要释放十几个请求的 slot，逐个 cat 会显著拖慢。
        懒释放把所有要还的 slot tensor 攒到一个 list，最后一次性合并。
        """
        def lazy_free(indices: torch.Tensor) -> None:
            # 注意 [:: page_size]：indices 里是"每个 token 的 slot 编号"，
            # 我们要把"每页的首 slot"放进 free_slots（保持页对齐）
            lazy_free_list.append(indices[:: self.page_size])

        lazy_free_list: List[torch.Tensor] = []
        try:
            # 临时替换掉真实的 _free 方法
            self._free = lazy_free
            yield
        finally:
            # 离开 with：恢复 _free 方法并把暂存列表一次性合并到 free_slots
            del self._free
            self.free_slots = torch.cat([self.free_slots] + lazy_free_list)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        """
        【功能】底层分配：拿出 needed_pages 个页编号；不够就 evict 出来。
        【参数】needed_pages: 想要分配多少页
        【返回】tensor，含 needed_pages 个页编号（每个是"首 slot" 编号）
        【内部逻辑】
        1. 看 free_slots 够不够；
        2. 不够：调 prefix_cache.evict 释放 (needed - free) 页；
        3. 然后从 free_slots 头部取 needed_pages 个、剩下的留作 free。
        """
        if needed_pages > (free_pages := len(self.free_slots)):
            # 需要驱逐 (needed - free) 个页 = (needed - free) * page_size 个 slot
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            # evicted 是"每个被驱逐 token 的 slot"——按 page_size 取首 slot 放回
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        # 从 free_slots 头部切出 needed_pages 个
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        """
        【功能】把若干 slot 还回 free_slots。
        【参数】indices: 每个元素是 token 的 slot 编号
        【处理】只保留"每页首 slot"（[::page_size]）放回 free_slots，
                避免在 page_size>1 时把同一页的多个 slot 重复登记。
        """
        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        """
        【功能】把"页编号"展开成"每个 token 的 slot 编号"。
        【参数】pages: 每元素是"页的首 slot 编号"，例如 [0, 8, 16]（page_size=8）
        【返回】展开后的 slot 编号，例如 [0,1,...,7, 8,9,...,15, 16,17,...,23]

        【为什么需要展开】
        page_table 的语义是"每个 token 一个 slot 编号"，所以分配出的页
        必须展开。page_size=1 时直接返回 pages 即可。
        """
        if self.page_size == 1:
            return pages
        # offsets = [0, 1, ..., page_size-1]
        # pages.unsqueeze(1) + offsets → [[首+0, 首+1, ...], ...]
        # flatten → 一维：每页 page_size 个连续编号
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    """
    【功能】把刚分配出的 slot 编号写入 page_table 的对应位置。

    【参数】
    - page_table: 全局页表，GPU 张量
    - allocated: 刚分配的 slot 编号（按 allocation_info 顺序拼接）
    - allocation_info: 列表 of (table_idx, first_page, last_page)
        告诉我们 allocated 里前 (last-first)*page_size 个属于第一个请求，
        接着的若干属于下一个请求，依此类推。
    - page_size: 页大小

    【实现技巧】
    ⚡ 性能关键: 一次 fancy index 写入比 for-loop 逐行赋值快几十倍。
    所以先在 CPU 上用 pinned memory 构造好 (row_indices, col_indices) 两个数组，
    然后一次性拷到 GPU 用 fancy indexing 写：
        page_table[table_idxs, offsets] = allocated

    【pinned_memory 的作用】
    torch.empty(..., pin_memory=True) 在锁页内存上创建张量，
    后续的 .to(device, non_blocking=True) 拷贝可以异步执行（不阻塞 CPU），
    充分利用 PCIe 带宽。

    【例子】
    allocation_info = [(7, 3, 5), (2, 0, 4)]，page_size=1
    allocated = [10, 11, 12, 13, 14, 15]
    展开后:
      第一个请求 (table=7, page 3..5 → token 位置 3,4)
      第二个请求 (table=2, page 0..4 → token 位置 0,1,2,3)
    table_idxs = [7, 7, 2, 2, 2, 2]
    positions  = [3, 4, 0, 1, 2, 3]
    page_table[table_idxs, positions] = [10, 11, 12, 13, 14, 15]
    """
    needed_tokens = len(allocated)
    # 在 pinned memory 上分配 host 张量（用于异步 host→device 拷贝）
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        # 把 page 编号转回 token 位置：每页 page_size 个连续 token
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        # 该请求要写的所有位置 row = 同一个 table_idx
        table_idx_host[offset : offset + length].fill_(table_idx)
        # column = [first_pos, first_pos+1, ..., last_pos-1]
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    # 异步拷贝到 GPU
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    # 一次性 fancy indexing 写入
    page_table[table_idxs, offsets] = allocated
