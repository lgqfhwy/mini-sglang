"""
========================================================================
文件名: scheduler/utils.py
所属模块: 调度器（Scheduler）模块的工具类容器
========================================================================

【这个文件是做什么的 - 一句话总结】
这个文件就像调度器的"小工具抽屉"——里面放着两个很短小的数据类，
分别用来表示"还没开始处理的待办请求"和"一次调度循环的结果摘要"。
它们本身不做任何复杂逻辑，只是给上层调度逻辑提供干净的数据载体。

【为什么需要这个文件 / 这个模块存在的原因】
当用户的请求消息（UserMsg）从其他进程（tokenizer）传到调度器进程后，
还不能立刻进入 Req（核心运行态对象），因为：
  - 需要先看显存够不够（prefix cache 命中多少 / 还要分配多少 slot）；
  - 需要看页表里还有没有空行；
  - 在没真正"被接受"进入运行队列之前，它就是个"待办事项"。

PendingReq 就是这个"中间形态"——已经从消息里解码出来，但还没被
具体调度到本轮 batch 的请求。

【这个文件在整个推理流程中的位置】
   UserMsg（来自 tokenizer 进程）
     ↓
   ★ PendingReq（本文件定义） ★   ← 等在 prefill_manager.pending_list 里
     ↓ 某一轮 schedule_next_batch 里"通过资源检查"
   Req（core.py 定义）          ← 真正参与 GPU 前向的运行态对象

【核心概念速览】

- UserMsg vs PendingReq vs Req（三种"请求"的区别！很容易混淆）：
    UserMsg:      跨进程 IPC 消息体（序列化用），只装最简单的字段。
    PendingReq:   调度器进程内的"待办队列条目"——还没分到显存资源。
    Req:          已经分到 table_idx 和 cache_handle，准备/正在跑 GPU。

- chunked_req（分块预填）：
    如果用户输入太长（比如 prompt 有 1 万 token），一次性 prefill 会
    把单步 token 数推爆（kernel 占太多显存）。所以会"切片"成多段，
    每段几千 token，分多轮 prefill 完。chunked_req 就是"这条 PendingReq
    上一轮已经处理到一半，下一轮接着处理"的占位句柄。

【关键设计决策】
- 把这两个数据类抽到独立文件而不是塞进 scheduler.py：
    保持 scheduler.py 聚焦于"调度逻辑"，数据类放工具文件里更整洁。
- 用 TYPE_CHECKING 做类型导入：
    避免 utils.py 反向 import scheduler 模块的真实类型，防止循环依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

# 只在静态类型检查时导入，运行时不实际加载——避免循环依赖
# （prefill.py 依赖 utils.py 的 PendingReq；utils.py 仅在类型上提到
# prefill.py 的 ChunkedReq，所以这里只做"前向声明"）。
if TYPE_CHECKING:
    from minisgl.core import SamplingParams

    from .prefill import ChunkedReq


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：PendingReq（读代码之前请先读完这段） ███
# ════════════════════════════════════════════════════════════════════
#
# 一、这个类要解决什么问题？
#
#   当用户的请求消息到达调度器时，它还需要排队等显存资源——可能要
#   等几个毫秒到几秒不等（看队列长度、显存压力）。在这段"等待期"，
#   调度器需要一个简单的容器来存它，PendingReq 就是。
#
# 二、用具体例子走一遍
#
#   假设 tokenizer 把用户输入 "你好，今天天气怎么样？" 切成 8 个 token：
#     input_ids = tensor([12, 34, 56, 78, 90, 11, 22, 33])
#     uid = 100001
#     sampling_params = SamplingParams(temperature=0.7, max_tokens=200)
#
#   tokenizer 把这些打包成 UserMsg 发给调度器。调度器收到后，
#   立刻包装成一个 PendingReq：
#     PendingReq(
#         uid=100001,
#         input_ids=tensor([12, 34, 56, 78, 90, 11, 22, 33]),
#         sampling_params=...,
#         chunked_req=None     ← 还没开始处理，自然没有"上次切到一半"
#     )
#   塞进 prefill_manager.pending_list 排队。
#
#   【场景1：一次性 prefill 全部 8 个 token】
#     调度器某一轮发现显存足够、页表有空行
#     → 构造 Req（cached_len=0, device_len=8, output_len=200）
#     → 把 PendingReq 从 pending_list 移除
#
#   【场景2：prompt 很长，触发分块（chunked prefill）】
#     假设单步预填上限 token_budget = 1024，本请求 prompt 有 5000 token
#     第 1 轮：prefill 处理 0..1023（这 1024 个 token）
#       → 构造一个 ChunkedReq（cached_len=0, device_len=1024）
#       → PendingReq.chunked_req = 这个 ChunkedReq
#       → PendingReq 仍然留在 pending_list（前移）
#     第 2 轮：prefill 接着处理 1024..2047
#       → 在 _add_one_req 里读取 chunked_req 的 table_idx 和 cache_handle
#       → 创建新的 ChunkedReq（cached_len=1024, device_len=2048）
#       → 更新 PendingReq.chunked_req
#     ...
#     第 N 轮：处理最后一段 → 这次构造的是 Req（而非 ChunkedReq）
#       → PendingReq 才真正从 pending_list 移除，进入 decode 阶段
#
# 三、各字段的来源与去向
#
#   字段              来自                           去向
#   uid              UserMsg.uid                    复制到 Req.uid
#   input_ids        UserMsg.input_ids              复制到 Req.input_ids
#   sampling_params  UserMsg.sampling_params        复制到 Req.sampling_params
#   chunked_req      scheduler 在分块时写入        下轮 try_add_one 里读取
#
# ════════════════════════════════════════════════════════════════════
@dataclass
class PendingReq:
    """
    【类名】PendingReq（待办请求）
    【一句话描述】调度器内部"还没拿到显存资源"的请求等待条目。
    【生活类比】餐厅门口的"等位号牌"——已经登记了几位、有什么忌口
                （sampling_params），但还没拿到桌子（table_idx）。

    【它和其他类的关系】
    - 由 PrefillManager.add_one_req 在收到 UserMsg 时创建；
    - 留在 PrefillManager.pending_list 里排队；
    - 在 PrefillAdder.try_add_one 里被转换为 Req 或 ChunkedReq；
    - 若发生 abort（用户取消），就直接从 pending_list 弹出丢弃。
    """

    # ----------------------------------------------------------------
    # 变量: uid
    # 类型: int
    # 含义: 用户请求的全局唯一标识（贯穿 API → tokenizer → scheduler →
    #       detokenizer → API 全流程）。
    # ----------------------------------------------------------------
    uid: int

    # ----------------------------------------------------------------
    # 变量: input_ids
    # 类型: torch.Tensor（CPU 上的 1-D long 张量）
    # 含义: 用户输入文本经过 tokenizer 切分后的 token id 序列。
    # 例: 文本 "Hello" → tensor([15496])
    # ----------------------------------------------------------------
    input_ids: torch.Tensor

    # ----------------------------------------------------------------
    # 变量: sampling_params
    # 类型: SamplingParams
    # 含义: 用户为本请求指定的采样规则（temperature、top_k、top_p 等）。
    # ----------------------------------------------------------------
    sampling_params: SamplingParams

    # ----------------------------------------------------------------
    # 变量: chunked_req
    # 类型: ChunkedReq | None
    # 含义:
    #   - None：本请求还没开始 prefill / 已经一次性 prefill 完；
    #   - 非 None：本请求处于"分块预填"中途——值就是上一轮残留下来的
    #     ChunkedReq（里面记着 table_idx、cache_handle、已处理到第几个 token）。
    # 用途: 让下一轮 prefill 知道"接着 chunked_req.device_len 往下处理"
    #       而不是从头开始（也不重复分配页表行）。
    # ----------------------------------------------------------------
    chunked_req: ChunkedReq | None = None

    @property
    def input_len(self) -> int:
        """prompt 的 token 数量（等同于 len(input_ids)，提供别名以提高可读性）。"""
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        """
        用户希望最多生成多少个新 token。
        来自 sampling_params.max_tokens（可能在 scheduler 处理 UserMsg 时
        被 clip 到不超过 max_seq_len - input_len）。
        """
        return self.sampling_params.max_tokens


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：ScheduleResult ███
# ════════════════════════════════════════════════════════════════════
#
# 这是一个返回值容器，目前在调度器主路径里并未直接使用（属于扩展点
# 或测试代码的需要）。保留它是为了未来可能的"批量返回调度信息"接口。
#
# - reqs:           本轮决定要处理的 PendingReq 列表
# - output_indices: 它们各自最终拿到的输出位置（GPU 端 tensor 列表）
#
# 当前 mini-sglang 的调度器主路径直接返回 Batch 对象，这个类几乎没被
# 用到，可视为预留接口。
# ════════════════════════════════════════════════════════════════════
@dataclass
class ScheduleResult:
    """【类名】ScheduleResult（一轮调度的结果摘要，预留接口）"""

    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
