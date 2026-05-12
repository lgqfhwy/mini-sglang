"""
========================================================================
文件名: kvcache/radix_cache.py
所属模块: KV Cache - 基于 Radix Tree（基数树）的前缀缓存实现
========================================================================

【这个文件是做什么的 - 一句话总结】
这个文件实现了"前缀共享"——多个请求如果开头的 token 序列相同
（比如都用了相同的 system prompt），就让它们共用同一份 KV Cache，
不需要每个请求都重新算一遍 prompt 的 K/V。
内部用一种叫 Radix Tree（压缩前缀树）的数据结构来管理。

【为什么需要这个文件 / 这个模块存在的原因】
在真实多用户场景下，请求往往有大量相同前缀：
  - 客服系统：每个对话开头都是同一段角色设定；
  - Few-shot 任务：所有请求都贴一长串 example；
  - 多轮对话：同一会话的后续请求开头都包含之前的对话历史。

如果不共享，每个请求都重新跑一遍这些 token 的 prefill——浪费 GPU 算力
也浪费显存。Radix 缓存能让命中前缀的部分"秒级"复用，加速 5-10 倍以上。

【这个文件在整个推理流程中的位置】
   用户发请求
     ↓
   CacheManager.match_req(req) → 调用本文件的 match_prefix
     ↓ 命中 N 个 token → 前 N 个不必再算
   prefill 只算后续部分
     ↓ 完成
   CacheManager.cache_req → 调用本文件的 insert_prefix 把整个 prompt+生成
   存进去，给后续请求继续复用

【核心概念速览】

- Radix Tree（基数树 / 压缩字典树）：
    一种树形数据结构，把多个字符串的公共前缀压缩到同一条边上。
    类比："看树枝就知道大家共享了哪一段开头"。
    例：序列 [1,2,3,4] 和 [1,2,5,6] 在 Radix Tree 中表示为：
                root
                 │ [1,2]
                 *
                ╱ ╲
            [3,4] [5,6]
                 *      *
    [1,2] 是公共前缀，存在树边上；[3,4] 和 [5,6] 是分叉。

- LRU 驱逐：
    显存满了要踢掉一些条目。优先踢"最近最少使用"的（timestamp 最旧）。
    用最小堆按 timestamp 排序，每次弹出最旧的叶子节点。

- ref_count（引用计数）：
    一个节点被多少请求 lock 着。> 0 表示有人在用，不能驱逐；
    == 0 表示可驱逐。这是经典的 "锁定/可回收" 两态切换机制。

【关键设计决策】

1. 用 timestamp（monotonic_ns）做 LRU 排序：
   单调递增的纳秒时间戳，简单可靠，避免引入复杂的双向链表。

2. fast_compare_key kernel：
   节点匹配长度的核心运算（"我这条边的 key 和 input_ids 前几位相同？"）
   用 CUDA kernel 实现，避免 Python 循环逐 token 比较。

3. 按 page 对齐插入（align_down）：
   page_size > 1 时，只插入"完整页"的部分，零头部分留着不存——
   这是 PagedAttention 的对齐要求。

4. key_fn 把"一段 tokens"变成 dict 的 hash key：
   page_size=1 时直接用第一个 token 的 int 值；
   page_size>1 时用 tuple——dict 查找时 O(1) 跳到对应子节点。
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from minisgl.core import get_global_ctx
from minisgl.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

# 类型别名：key_fn 接受一段 token tensor，返回可哈希的 key（用作子节点 dict 的索引）
KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：RadixTreeNode（读代码之前请先读完这段） ███
# ════════════════════════════════════════════════════════════════════
#
# 一、节点存什么？
#   每个节点（除根外）代表"一条边" = "一段连续的 token 序列 + 对应的
#   slot 编号序列"。具体字段：
#     _key:    这段 tokens 本身（torch tensor）——用于精确比较
#     _value:  对应的 slot 编号 tensor——上层要用
#     _length: 这段长度（token 数）
#     children: 子节点字典 {子段的首部 key -> 子节点}
#     ref_count: 多少请求正持有"我"作为 handle.node
#     timestamp: 最后被访问的时间（LRU 用）
#
# 二、举例说明 Radix Tree 结构
#   假设我们陆续插入了以下 (tokens, slots) 对：
#     ([10,20,30,40,50], [100,101,102,103,104])
#     ([10,20,30,99,88], [200,201,202,203,204])
#     ([10,20,30],       [300,301,302])    ← 短一些，是上面的公共前缀
#
#   构建出的树：
#       root
#        │ tokens=[10,20,30]  values=[300,301,302]    ← node A
#        ├─ tokens=[40,50]    values=[103,104]        ← node B
#        └─ tokens=[99,88]    values=[203,204]        ← node C
#
#   含义：所有 3 个请求都"经过"节点 A（共享前 3 个 token 的 KV），
#         然后第 1、2 个请求分叉到 B / C。第 3 个请求只到 A 就结束了。
#
# 三、split_at 的用途
#   假如已有节点 A: tokens=[10,20,30,40,50]
#   现在来一个请求 [10,20,30,77,88]——和 A 的前 3 个匹配，第 4 个开始不同。
#   这时要"拆"A：
#     原 A 变成两个节点 A'(tokens=[10,20,30]) 和 A''(tokens=[40,50])，
#     A' 是 A'' 的父节点。
#     然后新请求挂在 A' 下：A'.children[77] = 新节点(tokens=[77,88])。
#
#   这就是 split_at(3) 的作用——在位置 3 切开当前节点。
#
# ════════════════════════════════════════════════════════════════════
class RadixTreeNode:
    """
    【类名】RadixTreeNode（基数树节点）
    【一句话描述】Radix 树的一个节点，存一段 (tokens, slot indices) 边。
    """

    # 类变量：用作生成 UUID（仅调试用）
    counter: int = 0

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        """
        【功能】创建一个空节点（key/value 等字段后续通过 set_key_value 填）。
        【参数】
        - key_fn: 把 tokens 切片转成 dict 可哈希 key 的函数
        - tic: 可选的初始 timestamp；None 时取当前时间
        """
        self.key_fn = key_fn
        # children: 子段首部 key -> 子节点。dict 查找 O(1)。
        self.children: Dict[Any, RadixTreeNode] = {}
        # 父节点引用——根节点的 _parent 为 None
        self._parent: RadixTreeNode | None = None
        # 引用计数——多少个请求 lock 了本节点
        self.ref_count: int = 0
        # 唯一 id（调试用）
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        # timestamp（LRU 用）—— monotonic_ns 永远递增，不受系统时间调整影响
        self.timestamp = tic or time.monotonic_ns()

        # 这些字段在 set_key_value 时填写
        # these fields should be updated later
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """【功能】填好本节点的 (tokens, slots) 和长度。"""
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def set_parent(self, parent: RadixTreeNode) -> None:
        """
        【功能】设置父节点；同时把自己挂到父节点的 children 字典里。
        【key 选择】用 key_fn 算出本节点 key 的"首部哈希"作为 dict 索引——
                   这样在 children 里查找 O(1)。
        """
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        """返回父节点；如果是根（无父），断言失败。"""
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        """是否是根节点（根的 _parent 为 None）。"""
        return self._parent is None

    def is_leaf(self) -> bool:
        """是否是叶子（没有子节点）。"""
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        """
        【功能】比较自己的 _key 和传入的 input_ids，返回从开头算的"最长公共前缀长度"。
        【实现】调 CUDA kernel fast_compare_key 加速。
        ⚡ 性能关键：Python 逐 token 比较慢 100 倍以上；用 kernel 一次性比较。
        """
        from minisgl.kernel import fast_compare_key

        # compare key and input_ids, find the first diff
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        """
        【功能】在位置 pos 把本节点"切开"为两段，返回新创建的"父段"节点。

        【参数】pos: 切割点（0 < pos < length）

        【实现】
        旧 self: [tokens, values, length, ref_count, ...]
                  │
                  ▼ 切割
        新节点 new_node = [self._key[:pos], self._value[:pos], ...]
                           │
                           ▼
        旧 self (剩下部分): [self._key[pos:], self._value[pos:], ...]

        新节点接到原 self 的父节点；原 self 接到 new_node 之下。
        ref_count 复制到 new_node（因为之前持有 self 的请求等于也持有它的祖先）。
        """
        assert 0 < pos < self.length
        parent = self.parent

        # 创建一个"前半部分"的新节点
        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        # 引用计数原样复制（持有 self 的请求实际上也持有它的祖先链）
        new_node.ref_count = self.ref_count

        # self 收缩为"后半部分"
        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        """
        让 heapq（最小堆）能按 timestamp 排序——timestamp 最小（最旧）排前面，
        LRU 驱逐时优先被弹出。
        """
        return self.timestamp < other.timestamp


# ════════════════════════════════════════════════════════════════════
# RadixCacheHandle: 指向某个 RadixTreeNode 的句柄
# ────────────────────────────────────────────────────────────────────
# handle.node 指向"匹配到的最深节点"——cached_len 个 token 的 KV 全部
# 在从 root 到 node 这条路径上。
# ════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    """
    【类名】RadixCacheHandle
    【一句话描述】Radix 缓存的句柄——内部持有一个树节点指针。
    """

    node: RadixTreeNode

    def get_matched_indices(self) -> torch.Tensor:
        """
        【功能】沿"node → root"路径把所有 values 拼起来，得到完整的 slot 编号张量。

        【实现】
        从 node 开始往上走父节点（除根外），把每个节点的 value 收集起来，
        最后反转（让 root 那端排前面）再 cat 成一个张量。
        例:
          root → A(values=[1,2,3]) → B(values=[4,5])
          handle.node = B
          走 B → A → root：value_list = [[4,5], [1,2,3]]
          reverse: [[1,2,3], [4,5]]
          cat: [1,2,3,4,5]
        """
        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        # 反转让 root 端在前
        value_list.reverse()
        return torch.cat(value_list)


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：RadixPrefixCache（读代码之前请先读完这段）███
# ════════════════════════════════════════════════════════════════════
#
# 一、整体工作流程（用具体例子）
#
#   初始: 树里只有 root（永远 protected）
#         evictable_size = 0, protected_size = 0
#
#   【请求 A: prompt = [10,20,30,40,50]，slots = [100..104]】
#     1. match_prefix([10,20,30,40,50])
#        _tree_walk: 从 root 找子节点 key_fn([10,...]) → 没找到
#        return (root, 0) → match 0 个
#     2. CacheManager.allocate_paged 给请求 A 分配 5 个 slot
#     3. prefill 完成，cache_req → insert_prefix:
#        创建节点 A: key=[10,20,30,40,50], value=[100..104]
#        A.parent = root
#        evictable_size += 5  → 5
#        返回 (cached_len=0, handle=A)
#     4. handle 被 lock → ref_count=1, evictable -=5, protected +=5
#
#   【请求 B: prompt = [10,20,30,99,88]，slots 暂未分配】
#     1. match_prefix([10,20,30,99,88])
#        _tree_walk:
#          从 root 找 key_fn([10,...]) → 找到节点 A
#          走进 A: get_match_len([10,20,30,99,88]) → 3（前 3 个相同）
#          3 != 5（A.length）→ 触发 split_at(3)
#            A 被切成 A'(key=[10,20,30]) + A''(key=[40,50])
#          return (A', 3) → match 到 3 个 token
#     2. 分配 2 个新 slot（[200,201]）做请求 B 的 [99,88]
#     3. cache_req → insert_prefix([10,20,30,99,88], [...,200,201]):
#        _tree_walk 又匹配到 A'，长度 3，对齐 page_size=1 不变
#        prefix_len=3 != insert_len=5 → 创建新节点:
#          new_node: key=[99,88], value=[200,201], parent=A'
#        evictable_size += 2 → 总共多了 2
#        返回 (cached_len=3, handle=new_node)
#
#   【显存吃紧，要驱逐 6 个 slot】
#     evict(6):
#       _collect_leave_nodes_for_evict 找所有 ref_count==0 的叶子节点
#       按 timestamp 排序（heapq）→ 最旧的优先
#       弹出叶子，删它在父节点 children 里的入口，把 value 收集起来
#       如果父变成新叶子且 ref_count==0 → push 进堆继续考虑
#       累积到 evicted_size >= 6 → 停
#       返回所有被驱逐 token 的 slot 编号 tensor
#
# 二、为什么 root 始终 protected？
#
#   设计上简化——root 是"虚根"，所有路径都从它出发。如果 root 也可
#   驱逐就会导致整棵树消失，逻辑复杂。所以 root.ref_count=1 永久。
#
# 三、size 统计（evictable_size + protected_size）
#
#   lock_handle 时：把 node 到 root 路径上 ref_count==0 的节点的 length
#                    从 evictable 移到 protected。
#   unlock_handle 时：反向操作。
#
# ════════════════════════════════════════════════════════════════════
class RadixPrefixCache(BasePrefixCache):
    """
    【类名】RadixPrefixCache
    【一句话描述】基于 Radix Tree 的前缀缓存——支持多请求共享前缀 KV、LRU 驱逐。
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        # 从全局 Context 拿 page_size（用于对齐插入长度）
        self.page_size = get_global_ctx().page_size
        # 构造 key_fn（依赖 page_size）
        self.key_fn = _get_key_fn(self.page_size)
        # 全局共享的空 tensor（evict(0) 返回它）
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        # 当前可驱逐 / 被锁住保护的总 token 数
        self.evictable_size = 0
        self.protected_size = 0
        # 创建根节点；永远 protected（ref_count=1）
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        【功能】锁定/解锁一个 handle 指向的路径——把路径上所有节点的
                ref_count ±1，同时调整 evictable / protected 统计。

        【参数】
        - handle: RadixCacheHandle 实例
        - unlock: False=lock（ref_count+1），True=unlock（ref_count-1）

        【实现细节】
        从 handle.node 一路往上走父节点（不到 root），逐个节点修改
        ref_count。如果某个节点的 ref_count 从 0 变正 / 从正变 0，
        相应地把它的 length 在 evictable_size 和 protected_size 之间转移。
        """
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    # 从 "被保护" 变回 "可驱逐"
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    # 从 "可驱逐" 变成 "被保护"
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """
        【功能】给定 input_ids，找到最长匹配前缀对应的节点。
        【返回】MatchResult(cuda_handle=RadixCacheHandle(prefix_len, node))
        【副作用】不修改树结构，但会更新被访问节点的 timestamp（LRU 用）。
        """
        node, prefix_len = self._tree_walk(input_ids)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """
        【功能】把 (input_ids, indices) 插入树；如果已有部分前缀存在则复用。

        【参数】
        - input_ids: 完整的 token 序列
        - indices: 对应的 slot 编号

        【实现】
        1. 按 page_size 向下对齐（不足一页的零头不插入）；
        2. _tree_walk 先看已有的最长前缀；
        3. 如果 prefix_len < insert_len，说明有"新内容"——创建新节点
           挂在匹配到的节点下，存剩余的 (key, value)；
        4. evictable_size 增加新增长度；
        5. 返回 (cached_len=prefix_len, handle=最深节点)
            cached_len 是"插入前已有的部分"长度，调用者需释放冗余 slot。

        【.clone() 的用意】
        indices[prefix_len:] 是一个 view（共享父张量内存）。如果父张量
        后续被释放，view 就悬空了。所以这里 clone 出独立的 tensor 让
        节点持有，确保数据生命周期独立。
        """
        # 按 page_size 对齐：只插完整页的部分
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:  # NOTE: prefix_len < insert_len
            # 把"新内容"做成新节点挂上去
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> torch.Tensor:
        """
        【功能】驱逐至少 size 个 token 的可驱逐条目，返回它们对应的 slot 编号。

        【算法】LRU
        1. 收集所有"叶子且 ref_count==0"的节点；
        2. 用最小堆按 timestamp 排序；
        3. 不断弹最旧的、累积 length，达到 size 就停；
        4. 弹出的节点要从父节点的 children 字典里删除；
        5. 父节点可能因此变成新的可驱逐叶子，再 push 进堆参与排序。

        【注意】实际驱逐量可能 > size（只能整节点驱逐，不能切半）。
        """
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        # 收集所有可驱逐叶子节点
        leave_nodes = self._collect_leave_nodes_for_evict()
        # 转成最小堆（heapq 默认按 < 操作排序，即 timestamp 升序）
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            # 至少要够驱逐——前面 assert 已检查 evictable_size >= size
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            # 弹出最旧的叶子
            node = heapq.heappop(leave_nodes)
            # 安全性检查：必须是 ref_count=0 的叶子且非 root
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            # 从父节点 children 字典里删除自己
            del parent.children[self.key_fn(node._key)]
            # 父节点可能因此变成新的可驱逐叶子
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    def reset(self) -> None:
        """重置——当前未实现（项目暂时不需要）。"""
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        """返回当前可驱逐/被保护各占多少 token。"""
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        """完整性检查——目前未实现（占位）。"""
        pass

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        """
        【功能】遍历整棵树，收集所有"叶子且 ref_count==0"的节点。
        【用途】evict 时的候选池。
        """
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                # 中间节点：把子节点全部入栈继续遍历
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        """
        【功能】从 root 出发，沿树尽量深地匹配 input_ids 的前缀。

        【返回】(最深匹配到的节点, 已匹配的 token 长度)

        【关键步骤】
        1. 用 key_fn 算"剩余 input_ids 头部"的 key，看 children dict 里有没有；
        2. 没有 → 当前位置就是分叉点，返回 (当前节点, 已匹配长度)；
        3. 有 → 走进子节点，调 get_match_len 比较"它的 key"和"input_ids 剩余"
           ↳ 没匹配完整子节点的 key → split_at 切开节点，返回前半部分；
           ↳ 完全匹配子节点的 key → 继续往下走；
        4. 每访问一个节点就更新 timestamp（LRU）。
        """
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        # 本次 walk 用同一个 tic 给所有访问到的节点更新 timestamp
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            # 看 children 里有没有以 key_fn(剩余开头) 为 key 的子节点
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                # 没有匹配的子节点——这就是分叉点
                return node, prefix_len
            node = child_node  # walk to child node

            # 比较"子节点的 key"和"input_ids 剩余"，返回最长公共前缀
            # NOTE: at least 1 page is matched, so match_len >= page_size
            match_len = node.get_match_len(input_ids[prefix_len:])
            # 按 page_size 向下对齐（不能匹配半页）
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # 如果只匹配了 node 的一部分（不是整个 key），要切开 node
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len

            # 完整匹配整个子节点，继续往下走
            node.timestamp = tic

        return node, prefix_len


def _get_key_fn(page_size: int) -> KEY_FN:
    """
    【功能】根据 page_size 返回一个 key_fn——把"一段 tokens"切片转成 dict key。

    【实现】
    - page_size == 1：直接用首 token 的 int 值（O(1) 最快）；
    - page_size > 1：用前 page_size 个 tokens 的 tuple（hash 仍然 O(1)，
                     但要遍历 page_size 个 token，略慢）。

    【为什么不直接用 tensor 做 key】
    tensor 不可哈希——不能作为 dict 的 key。必须转成原生 Python 类型。
    """
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
