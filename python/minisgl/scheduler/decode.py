"""
========================================================================
文件名: scheduler/decode.py
所属模块: 调度器 - "解码阶段"请求管理器
========================================================================

【这个文件是做什么的 - 一句话总结】
这个文件维护"正处于 decode 阶段（一个 token 一个 token 生成中）"的
所有请求集合。每轮调度它把这堆请求一股脑打包成 decode batch 送去
GPU 算一个 token。

【为什么需要这个文件 / 这个模块存在的原因】
LLM 推理有两个阶段：
  - prefill：处理 prompt（一次性算很多 token 的 K/V）
  - decode：逐 token 生成（每次只算 1 个 token）

两阶段在调度策略上很不一样：
  - prefill 阶段要做 token 预算控制、要可能拆 chunk；
  - decode 阶段所有正在跑的请求"一起再生成一个 token"，结构最简单。

所以让 decode 有自己的 manager，分担调度器的复杂度，让两边各自专注。

【这个文件在整个推理流程中的位置】
   PrefillManager 处理完一个请求的 prompt（不再是 ChunkedReq）
     ↓
   该请求的 can_decode == True → 自动并入 DecodeManager.running_reqs
     ↓
   每个调度循环:
     ★ DecodeManager.schedule_next_batch() ★
       把 running_reqs 打包成一个 decode batch
     ↓
   batch 算完，根据采样结果：
     - 如果生成完了 / 触到 EOS → 从 running_reqs 移除
     - 否则继续留在 running_reqs，下轮再算

【核心概念速览】

- decode 阶段:
    模型每次只算"1 个新 token"。每个请求每轮提供 1 个 input token、
    输出 1 个 logits → 采样出 1 个 next_token。
    所以 decode batch 里 batch size = 请求数，每个请求贡献 1 个 token。

- inflight_tokens（"在飞行中"的 token 数）:
    所有正在 decode 的请求"未来还要消耗的 KV slot"总数。
    用作 prefill 调度时的"保留预算"——必须留够空间给已经在跑的请求
    继续生成，否则它们可能突然没显存了。

- page_size:
    一页存几个 token 的 K/V。当 page_size > 1 时，每个请求最后一页
    可能没填满 → 实际预留 = 已用 + (page_size - 1) 的"取整冗余"。

【关键设计决策】

1. 使用 set 而非 list 存储 running_reqs：
   - O(1) 的 discard（remove）；
   - 不在乎顺序——decode batch 内顺序对结果无影响（每个请求独立采样）。

2. filter_reqs 一次性同时做"合并 + 过滤"：
   prefill 完成的请求会被传过来并入，已生成完的请求会被过滤掉。
   合并到一个调用里减少重复遍历。

3. 每轮都新建 Batch 对象：
   Batch 是轻量数据类，新建开销可忽略；保持每轮"无状态"更简单。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：DecodeManager（读代码之前请先读完这段）███
# ════════════════════════════════════════════════════════════════════
#
# 一、这个类要解决什么问题？
#
#   正在做 decode（逐 token 生成）的请求集合需要被维护：
#     - 哪些请求还在生成？
#     - 它们一共还要再吃多少 token 的显存？（供 prefill 预算参考）
#     - 怎么一键打包成一个 decode batch？
#
#   DecodeManager 就这点事。
#
# 二、用具体例子走一遍
#
#   假设 page_size = 1（最常见配置）。
#
#   【初始】running_reqs = {}
#
#   【prefill_manager 完成了请求 A 的 prefill，把 A 并入】
#     调度器调用 decode_manager.filter_reqs([A])
#     A.can_decode = True（还需要继续生成）→ 加入
#     running_reqs = {A}
#
#   【接着请求 B、C 也完成 prefill 并入】
#     running_reqs = {A, B, C}
#
#   【某一轮调度循环：要做 decode】
#     schedule_next_batch()
#       → runnable = True（有 3 个请求）
#       → 返回 Batch(reqs=[A,B,C], phase="decode")
#     调度器拿这个 batch 去 GPU 跑
#
#   【GPU 跑完，得到 3 个 next_token】
#     调度器对每个请求：
#       - 把 next_token append 到 req.input_ids
#       - 调 req.complete_one()（cached_len 推进 1）
#       - 检查是否完成生成（达到 max_tokens 或采到 EOS）
#         若完成 → decode_manager.remove_req(req)
#
#   【某个请求 D 刚被 prefill 完，本步同时要并入 D】
#     调度器调用 decode_manager.filter_reqs([D])
#     遍历 (running_reqs ∪ {D}) 中所有 can_decode 仍 True 的留下
#     运行中已经 done 的请求会在这一步被自动剔除
#
# 三、inflight_tokens 的用途
#
#   假设当前 running_reqs = {A, B, C}，各自 remain_len = 10, 5, 20
#   inflight_tokens = 10 + 5 + 20 + (page_size-1)*3 = 35（若 page_size=1）
#
#   prefill_manager 在尝试接收新请求 D 时，会把 inflight_tokens 当作
#   "已经预定要消耗的显存"——必须确保给 D 分配后，还能容下这 35 个
#   未来要生成的 token。否则 A/B/C 可能在中途因没空间被驱逐。
#
# ════════════════════════════════════════════════════════════════════
@dataclass
class DecodeManager:
    """
    【类名】DecodeManager（解码阶段请求管理器）
    【一句话描述】持有"正在生成中"的所有请求，并能一键打包成 decode batch。
    【生活类比】餐厅里"正在用餐的桌子"列表——每轮服务员去给所有还
                没吃完的桌子上下一道菜（生成下一个 token）。

    【它和其他类的关系】
    - 由 Scheduler 创建并持有；
    - PrefillManager 与它共享请求"接力棒"——prefill 完的请求并入它；
    - Scheduler 在每个循环里看它 .runnable 决定是否做 decode；
    - 它没有"分配资源"职责（页表和 KV 都由 TableManager/CacheManager 管），
      只关心"哪些请求处于 decode 中"。
    """

    # ----------------------------------------------------------------
    # 字段: page_size
    # 类型: int
    # 含义: KV cache 的页大小（一页存几个 token 的 K/V）。
    # 用途: inflight_tokens 计算时考虑"每个请求至少要为下一页预留空间"
    #       (page_size - 1) 个 token 的余量。
    # ----------------------------------------------------------------
    page_size: int

    # ----------------------------------------------------------------
    # 字段: running_reqs
    # 类型: Set[Req]
    # 含义: 当前正在 decode 阶段（still generating）的请求集合。
    # 注意:
    #   - 用 set 不用 list 因为不在意顺序、要 O(1) 的 add/remove；
    #   - dataclass 字段用 field(default_factory=set) 而不是 = set()
    #     避免所有实例共享同一个空集合的陷阱。
    # ----------------------------------------------------------------
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        """
        【功能】"合并 + 过滤"二合一：
          1. 把传入的新请求并入 running_reqs；
          2. 同时把不再需要 decode 的请求过滤掉。

        【调用时机】每个调度循环的末尾，由 Scheduler 调用：
                    self.decode_manager.filter_reqs(batch.reqs)
                    传入的是上一轮跑完的 batch 的请求列表
                    （包含可能完成了的、可能继续的、可能是新 prefill 完的）

        【参数】
        - reqs (Iterable[Req]): 一批"可能需要 decode 的请求"

        【内部逻辑】
        把已有的 running_reqs 和新传入的 reqs 取并集，再筛选
        can_decode 仍为 True 的留下，构造一个新 set 整体替换。

        【为什么用并集 + 过滤而不是 add+remove】
        - 一次性走完一个 pass，比来回判断更简洁；
        - set 推导式天然处理"已经存在"的重复请求（set 去重）。
        """
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        """
        【功能】明确把一个请求从 running 集合移除（不论它当前是否还
                 can_decode）。
        【调用时机】调度器判断该请求已经"自然完成"——达到 max_tokens
                    或采到 EOS——时调用。
        【为什么用 discard 而非 remove】
        discard 在 req 不存在于集合时不会抛异常，更"宽容"。
        """
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        """
        【功能】根据用户 uid 查找并强制移除一个 running 请求（用于用户
                取消请求 / 客户端断连）。
        【参数】uid: 要 abort 的请求的 uid
        【返回】找到并移除的 Req（调用者负责释放它的显存资源）；
                如果没找到（请求可能还在 pending、或已经结束），返回 None。
        【实现】线性扫描 running_reqs；O(n) 但 n 通常很小（几十~几百），
                而且 abort 不是热路径。
        """
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        """
        【功能】估算所有"正在 decode"的请求未来还要消耗多少 token 的 KV 空间。

        【返回】整数，单位是"token 数"。

        【公式拆解】
          tokens_reserved = (page_size - 1) * 请求数
              ← 每个请求假设最后一页有 page_size-1 个浪费/未填充槽位
          sum(req.remain_len) = 所有请求各自还要生成的 token 数
        最终 = 两者之和。

        【为什么要这么算 / 谁用它】
        PrefillManager 在尝试接受新请求时，必须把已经"在跑的请求未来要
        占的空间"也算上，否则它们可能跑到一半没空间。
        所以 PrefillAdder 用 self.cache_manager.available_size 减去
        本字段（reserved_size）作为新请求可用的预算。

        【为什么是 (page_size - 1) 而不是 page_size？】
        每个请求至少已占了 1 个 slot 在某页里，所以"那一页的剩余位置"
        只可能再吃 page_size-1 个 token 就要开新页。
        """
        # 保守预留：每请求 page_size-1 token，避免精细计算每个请求落在页内的偏移
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        """
        【功能】把当前所有 running 请求打包成一个 decode batch 返回。
        【返回】Batch（phase="decode"）；如果没有 running 请求返回 None。

        【内部逻辑】
        1. 没有 running 请求 → 直接返回 None（让上层去看 prefill）；
        2. 否则用 set 转 list 构造一个 Batch；
           reqs 的顺序由 set 的迭代顺序决定（Python 实现里近似插入序，
           但语义上认为"无序"——不依赖此顺序）。

        【为什么这里只构造 Batch 而不填 input_ids/positions 等？】
        填这些字段的是 Scheduler._prepare_batch（涉及对 padded_reqs、
        CUDA graph padding、注意力元数据的统一处理）。
        DecodeManager 只关心"哪些请求要跑"，不关心张量怎么拼。
        """
        if not self.runnable:
            return None
        return Batch(reqs=list(self.running_reqs), phase="decode")

    @property
    def runnable(self) -> bool:
        """
        【功能】是否有 running 请求可以做 decode。
        【用法】调度器主循环用它判断要不要走 decode 路径。
        """
        return len(self.running_reqs) > 0
