"""
========================================================================
文件名: scheduler/prefill.py
所属模块: 调度器 - "预填阶段"调度子模块
========================================================================

【这个文件是做什么的 - 一句话总结】
这个文件管"还没开始处理的等待队列"——决定每一轮调度循环里，应该
从等待队列中挑哪些请求、用多少 token 预算把它们的 prompt 做 prefill。
还负责把过长的 prompt 切成"分块"（chunked prefill）。

【为什么需要这个文件 / 这个模块存在的原因】
prefill 阶段的调度比 decode 复杂得多：
  - 需要管 token 预算（一次最多算多少 token）；
  - 需要查前缀缓存看命中率；
  - 需要给请求分配页表行和 KV slot；
  - 如果 prompt 太长，要切成多次小段，分多轮 prefill 完；
  - 还要给后续 decode 留够空间（reserved_size）。
把这堆逻辑独立成 PrefillManager + PrefillAdder，让上层调度器只调用
一个 schedule_next_batch 即可。

【这个文件在整个推理流程中的位置】
   tokenizer 把 UserMsg 发给 scheduler
     ↓
   Scheduler._process_one_msg 收到 → 调用 prefill_manager.add_one_req
     ↓
   PendingReq 进入 prefill_manager.pending_list 排队
     ↓ 每个调度循环
   ★ prefill_manager.schedule_next_batch(budget) ★
     ↓
   PrefillAdder 一个一个尝试从 pending_list 里挑请求，看资源够不够
     ↓
   够 → 构造 Req（或 ChunkedReq）放入本轮 batch
     ↓ 一次性返回 Batch
   GPU 跑 prefill → 完整请求并入 DecodeManager；ChunkedReq 留在 pending_list

【核心概念速览】

- chunked prefill（分块预填）：
    prompt 长度可能远超单步 token 预算。比如预算 8192，prompt 32000，
    就要拆成 4 段。前 3 段每段处理 8192 个 token（用 ChunkedReq 表示），
    最后一段处理剩下的 7424 个 token 然后转为正式 Req 进入 decode。

- token_budget（token 预算）：
    本轮 prefill 整批最多算多少 token = SchedulerConfig.max_extend_tokens。
    PrefillAdder 把它当"还能塞多少 token"的剩余配额，每加一个请求就扣。

- reserved_size（预留空间）：
    decode 中那些请求未来还要消耗的显存（DecodeManager.inflight_tokens）+
    本轮 prefill 中已经决定要进 batch 的请求未来要消耗的显存。
    PrefillAdder 在评估是否能接收一个新请求时，把 reserved_size 当
    "不能动的部分"。

【关键设计决策】

1. ChunkedReq 继承自 Req，禁用 append_host 和 can_decode：
   - ChunkedReq 在中途状态，绝对不能采样新 token（因为 prompt 还没读完）
     → 重写 append_host 抛异常防止误用；
   - can_decode 返回 False → 调度器不会误把它并入 DecodeManager。

2. try_add_one 失败立即返回 None：
   等待队列里第一个塞不下，就不再尝试后面的（避免乱序，按 FIFO 公平）。

3. _try_allocate_one 先 lock 再二次检查：
   多个调度循环之间可能有别的活动让 available_size 变化；lock 之后
   重新 assert 一次，确保我们真的还能用上这块缓存。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：ChunkedReq ███
# ════════════════════════════════════════════════════════════════════
#
# ChunkedReq 表示"一个被切成多段的长 prompt 中、目前处理到中间某段
# 的状态"——它已经分到了 table_idx 和 cache_handle，已经算过部分
# token 的 KV，但还有后续 token 未算完。
#
# 例子: prompt 长 30000，token_budget = 8192
#   第 1 轮: 创建 ChunkedReq(cached_len=0, device_len=8192)
#   第 2 轮: ChunkedReq(cached_len=8192, device_len=16384)
#   第 3 轮: ChunkedReq(cached_len=16384, device_len=24576)
#   第 4 轮: 创建普通 Req(cached_len=24576, device_len=30000)
#            → 进入 decode 队列
#
# 因为 ChunkedReq 还没读完 prompt：
#   - 这一轮虽然在 batch 里跑了，但最后一位"token"还不是它要预测的；
#   - 所以采样器看到它会跳过；append_host 不能被调用。
#
# can_decode 返回 False → 在 decode_manager.filter_reqs 里被过滤掉，
# 不会误进 running set。
# ════════════════════════════════════════════════════════════════════
class ChunkedReq(Req):
    """
    【类名】ChunkedReq（分块预填中的请求）
    【一句话描述】继承 Req，但表示"prompt 还没读完、不能被采样"的中间态。

    【为什么用继承】
    复用 Req 的所有字段和属性（table_idx, cached_len, device_len 等）；
    只重写两处行为禁止"被当成正常 Req 用"。
    """

    def append_host(self, next_token: torch.Tensor) -> None:
        """
        【功能】明确禁止——ChunkedReq 还没产生 logits，不应该被采样。
        如果误调用，立刻报错（防御性编程）。
        """
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        """
        【功能】返回 False 让 decode_manager.filter_reqs 自动过滤掉本对象。
        否则 ChunkedReq 会被误加入 running set 并参与下一轮 decode batch。
        """
        return False  # avoid being added to decode manager


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：PrefillAdder（读代码之前请先读完这段）███
# ════════════════════════════════════════════════════════════════════
#
# 一、这个类要解决什么问题？
#
#   每一轮 prefill 调度里要做"逐个尝试把 pending 请求加入本轮 batch"
#   的工作：
#     - token_budget 还够吗？
#     - 显存还够吗？（考虑到已经 running 的请求未来还要吃多少）
#     - 页表还有空行吗？
#     - 命中前缀缓存的比例多少？需要分块吗？
#
#   把这个"加入一个请求的尝试逻辑"封装成 PrefillAdder，便于在循环里
#   一次次调用 try_add_one。
#
# 二、用具体例子走一遍
#
#   配置: token_budget=8192, page_size=1, cache_manager 当前
#         available_size=10000，table_manager 还有空行。
#   running 中的请求未来还需 inflight_tokens=2000 token 的显存
#   → 初始 reserved_size=2000
#
#   pending_list = [
#     P1: prompt 长 3000, output_len 200
#     P2: prompt 长 100, output_len 50
#     P3: prompt 长 20000, output_len 100  ← 超长，会触发分块
#   ]
#
#   --- 尝试 P1 ---
#   _try_allocate_one(P1):
#     match → 命中 0（假设）→ extend_len = 3000
#     estimated_len = 3000 + 200 = 3200
#     3200 + reserved_size(2000) = 5200 <= 10000 ✓
#     lock 一个空 handle，分配 table_idx=15
#     return (handle, 15)
#   _add_one_req(P1, handle, 15, cached_len=0):
#     remain_len = 3000 - 0 = 3000
#     chunk_size = min(8192, 3000) = 3000 ← 不分块
#     is_chunked = False
#     token_budget = 8192 - 3000 = 5192
#     reserved_size = 2000 + (3000 + 200) = 5200
#     创建 Req(cached_len=0, device_len=3000, ...)
#
#   --- 尝试 P2 ---
#   类似：chunk_size = 100，不分块，创建 Req
#   token_budget = 5192 - 100 = 5092
#   reserved_size = 5200 + (100 + 50) = 5350
#
#   --- 尝试 P3 ---
#   _try_allocate_one(P3):
#     match → 命中 0 → extend_len = 20000
#     estimated_len = 20000 + 100 = 20100
#     20100 + reserved_size(5350) = 25450 > 10000 ✗
#     返回 None → P3 这一轮加入失败
#
#   退出循环。batch.reqs = [Req(P1), Req(P2)]
#
#   --- 假设 token_budget 充足、显存也够，P3 也能加入但 prompt 太长 ---
#     remain_len = 20000, chunk_size = min(5092, 20000) = 5092
#     is_chunked = True
#     创建 ChunkedReq(cached_len=0, device_len=5092)
#     P3.chunked_req 设为这个 ChunkedReq
#     P3 留在 pending_list（不被消费）
#   下一轮接着算 P3 的 5092..10184
#
# ════════════════════════════════════════════════════════════════════
@dataclass
class PrefillAdder:
    """
    【类名】PrefillAdder（"逐个尝试加入"的辅助类）
    【一句话描述】围绕 token_budget 和 reserved_size 维护"还能塞多少"的
                  状态，对 pending list 做一次"逐请求 try_add"的扫描。

    【它和其他类的关系】
    - 每轮 prefill 调度时由 PrefillManager 新建一次，用完丢弃；
    - 内部依赖 cache_manager / table_manager 查询资源；
    - 不直接修改 pending_list（这是 PrefillManager 的工作）。
    """

    # ----------------------------------------------------------------
    # 字段: token_budget
    # 类型: int
    # 含义: 本轮还能算多少 token（每加一个请求就扣它的 chunk_size）。
    # 初始值 = SchedulerConfig.max_extend_tokens
    # ----------------------------------------------------------------
    token_budget: int

    # ----------------------------------------------------------------
    # 字段: reserved_size
    # 类型: int
    # 含义: "已被预定要消耗"的显存（token 数）：
    #       = DecodeManager.inflight_tokens（已在跑的请求）
    #       + 本轮已加入 batch 的请求未来要消耗的（动态累加）
    # 用途: 评估新请求时确保不会挤占 reserved 部分
    # ----------------------------------------------------------------
    reserved_size: int

    # cache_manager / table_manager 引用——供查询资源
    cache_manager: CacheManager
    table_manager: TableManager

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        """
        【功能】尝试为一个尚未分到资源的 PendingReq 找显存 + 页表行。

        【返回】
        - 成功: (cache_handle, table_idx) —— 调用者拿这两个去构造 Req；
        - 失败: None —— 当前资源不足以接收这个请求。

        【内部逻辑】
        1. 没空行就直接拒；
        2. 查前缀缓存命中情况；
        3. 估算请求最终要占多少 token 的 KV（extend + output_len）；
        4. 估算超过 available_size 就拒；
        5. lock handle 防被驱逐 → 二次验证（lock 操作本身可能改变可用空间）；
        6. 通过则分配 table_idx，并把已缓存部分的 token_pool / page_table 写好。
        """
        # 1. 页表行检查
        if self.table_manager.available_size == 0:
            return None

        # 2. 前缀缓存匹配（cuda_handle 是命中的句柄；如果没命中 cached_len=0）
        # TODO: consider host cache match case
        handle = self.cache_manager.match_req(req).cuda_handle
        cached_len = handle.cached_len

        # 3. 估算新算多少 token + 未来要生成多少 token
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        # 4. 是否塞得下
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        # 5. lock 防止 handle 在我们用之前被驱逐
        self.cache_manager.lock(handle)
        # ⚠️ lock 之后再二次检查——lock 把 handle 标为不可驱逐，会改变
        # available_size。如果突然不够了要立刻 unlock 退还。
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        # 6. 分配页表行
        table_idx = self.table_manager.allocate()
        if cached_len > 0:
            # 把"已命中部分"的 token id 和 slot 编号写到 token_pool / page_table
            # 之后注意力 kernel 直接能读到这些 KV（不必再算）
            # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            # input_ids[:cached_len] → host 上的命中部分 token id，
            # .pin_memory() 加速 host→device 异步拷贝
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            # 命中部分的 slot 编号直接从 handle 取出，写入 page_table 行
            page_entry.copy_(handle.get_matched_indices())

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        """
        【功能】把一个已经分到资源的 PendingReq 转成正式的 Req/ChunkedReq，
                并更新预算/预留。

        【参数】
        - pending_req: 待办请求
        - cache_handle: _try_allocate_one 或上一个 ChunkedReq 拿到的 handle
        - table_idx: 已分配的页表行号
        - cached_len: 命中或上次已处理到的位置（即下一轮 prefill 起点）

        【返回】Req（如果一次能算完）或 ChunkedReq（如果要分块）

        【内部逻辑】
        1. remain_len = input_len - cached_len，还要算的 token 数
        2. chunk_size = min(token_budget, remain_len)
            - 如果 token_budget 足以一次性算完 → chunk_size = remain_len
            - 否则只算到预算上限 → 触发分块
        3. is_chunked = chunk_size < remain_len （还有剩没算完）
        4. 扣预算、累加 reserved（注意：reserved 加的是 remain_len + output_len，
            因为整个请求未来要消耗的总量是固定的，不论本轮算多少）
        5. 把本轮要算的 token id 拷到 token_pool 的对应位置
        6. 创建 Req 或 ChunkedReq
        """
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        # 根据是否分块选择类型
        CLS = ChunkedReq if is_chunked else Req
        # 扣本轮预算
        self.token_budget -= chunk_size
        # 累加 reserved（请求一生中还要占的 KV token 数）
        self.reserved_size += remain_len + pending_req.output_len

        # 把本轮要算的 token id 拷贝到 token_pool 的对应区间
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)

        # 构造 Req（input_ids 只保留到 cached_len + chunk_size，因为 device 端
        # 这一轮只算到这里；下一轮 ChunkedReq 才会扩到下一段）
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        """
        【功能】统一入口：尝试把一个 PendingReq 加入本轮 batch。

        【返回】Req 或 ChunkedReq（成功），None（资源不够 / 预算用完）

        【两种路径】
        1. pending_req 上次已经被分块预填过 → 直接复用它的 chunked_req 资源
        2. 全新请求 → 走 _try_allocate_one 流程申请资源
        """
        # token 预算已经用尽——直接拒
        if self.token_budget <= 0:
            return None

        # 路径 1：复用上次的 chunked_req（已经分过资源了）
        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        # 路径 2：全新请求，尝试申请新资源
        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：PrefillManager（读代码之前请先读完这段）███
# ════════════════════════════════════════════════════════════════════
#
# 一、这个类要解决什么问题？
#
#   prefill 阶段的"宏观调度"：
#     - 维护 pending_list 队列（来一个 UserMsg 就追加）；
#     - 每轮被调度器调一次 schedule_next_batch，按预算尽量多地接收请求；
#     - 处理"分块预填"——chunked_req 不消费 pending_list 条目，让它
#       下一轮接着处理。
#
# 二、用具体例子走一遍
#
#   pending_list = [P1, P2, P3]   ← 三个待办，按 FIFO 排
#   schedule_next_batch(prefill_budget=8192) 被调用
#
#   构造 PrefillAdder（budget=8192, reserved_size=inflight_tokens）
#   遍历:
#     P1 → try_add_one → 返回 Req(P1)
#       加入 reqs。P1.chunked_req=None。
#     P2 → try_add_one → 返回 ChunkedReq(P2)  （P2 prompt 很长被分块）
#       加入 reqs。P2.chunked_req = 那个 ChunkedReq → 加入 chunked_list。
#       注意：因为分块了，predict P2 这一轮预算被吃光了，下一次 try_add_one 大概率 None
#     P3 → try_add_one → None  （预算已用完）
#       break，停止扫描
#
#   pending_list 更新规则：
#     chunked_list = [P2]
#     被消费数 = 2（P1 和 P2 各处理了一次，但 P2 还要继续）
#     pending_list = [P2] + pending_list[2:] = [P2, P3]
#       ← P1 真正被消费（生成完会自动回不到 pending）；
#         P2 因为 chunked 还要继续，被放回队首；
#         P3 留在原位。
#
#   返回 Batch(reqs=[Req(P1), ChunkedReq(P2)], phase="prefill")
#
# 三、abort_req 的处理
#
#   用户在请求还在 pending 时取消（断连等）→ 调度器调 abort_req(uid)。
#   从 pending_list 里删除该条；如果它已经有 chunked_req（之前分块过、
#   占着资源），返回它供调度器释放显存/页表行。
#
# ════════════════════════════════════════════════════════════════════
@dataclass
class PrefillManager:
    """
    【类名】PrefillManager（预填阶段调度管理器）
    【一句话描述】管 pending_list 队列、用 PrefillAdder 决定每轮 prefill batch。

    【它和其他类的关系】
    - 由 Scheduler 创建并持有；
    - 持有对 cache_manager / table_manager / decode_manager 的引用；
    - 输出 Batch 给 Scheduler._prepare_batch 进一步加工。
    """

    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager

    # pending_list: 待办请求队列。default_factory=list 避免共享默认值。
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        """
        【功能】把一个 UserMsg 包装成 PendingReq 放进等待队列末尾。
        【调用时机】Scheduler._process_one_msg 在收到新 UserMsg 时调用。
        """
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        """
        【功能】本轮 prefill 调度——按 FIFO 顺序尽量多地接收请求。

        【参数】prefill_budget: 本轮总 token 预算

        【返回】Batch（含 prefill 请求列表）或 None（没有可调度的请求）

        【内部逻辑】
        1. pending_list 空 → 直接返回 None；
        2. 构造 PrefillAdder（reserved_size 初始为 DecodeManager.inflight_tokens）；
        3. 按 FIFO 顺序遍历 pending_list 调 try_add_one：
            - 返回 Req → 加入 reqs，把 pending_req.chunked_req 清空（消费完）；
            - 返回 ChunkedReq → 加入 reqs，并把 pending_req.chunked_req 设为它
              （这条不消费，下轮继续）；
            - 返回 None → break（按 FIFO，前面塞不下就别试后面）；
        4. 更新 pending_list：
            前面消费了 len(reqs) 条，但其中 chunked 的还要放回前面，
            所以新 pending_list = chunked_list + 剩下的；
        5. 返回 Batch（phase="prefill"）。

        【为什么 break 而不是继续尝试后面的请求？】
        保证 FIFO 公平。否则一个"小请求"可能不停插队前面的"大请求"，
        导致大请求饿死。
        """
        if len(self.pending_list) == 0:
            return None

        # 构造每轮专用的 Adder（estimated offset 来自 decode 中正在跑的请求）
        # estimated offset due to in-flight decode
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []  # 这些 pending 还要留在队列里继续处理
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                # 默认清空：通常表示这一轮就把它处理完了
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    # 这是分块预填的某一片，记下来让下一轮接着算
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                # 资源/预算不够，按 FIFO 公平起见停止扫描
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        # 更新 pending_list：把还要继续的 chunked 放回前面 + 没碰过的尾部
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        """
        【功能】用户取消某个请求时，从 pending_list 里把它删掉。

        【参数】uid: 要 abort 的请求 uid
        【返回】
        - 如果该请求之前已经分到资源（有 chunked_req）→ 返回那个 Req 让
          调度器去释放显存/页表行；
        - 否则（还纯粹是 PendingReq，没分资源）→ 返回 None（什么也不用释放）。
        """
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    @property
    def runnable(self) -> bool:
        """是否有待办请求等着调度。"""
        return len(self.pending_list) > 0
