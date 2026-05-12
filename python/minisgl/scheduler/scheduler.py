"""
========================================================================
文件名: scheduler/scheduler.py
所属模块: 调度器（Scheduler）模块的"主入口" - 整个推理后端的大脑
========================================================================

【这个文件是做什么的 - 一句话总结】
这个文件就是整个推理系统的"指挥中心"——它有一个永远不停止的主循环
（run_forever），不断地：(1) 收新请求 (2) 决定本轮要算哪些 (3) 喂给
GPU 跑前向 (4) 把结果发回去 (5) 回收资源。所有其他模块都围着它转。

【为什么需要这个文件 / 这个模块存在的原因】
LLM 推理服务的核心问题：怎么"既高吞吐又低延迟"地服务一群同时来的
用户？答案就是高效的 batching + scheduling。Scheduler 把这个问题分解：
  - PrefillManager 负责 prefill 的细节；
  - DecodeManager 负责 decode 的简单情形；
  - CacheManager / TableManager 负责显存资源；
  - Engine 负责 GPU 计算；
  - SchedulerIOMixin 负责跨进程通信。
而 Scheduler 这个文件用一个清晰的主循环，把它们串起来：
    while True:
        收消息 → 决定 batch → 跑前向 → 处理结果

【这个文件在整个推理流程中的位置】
   用户 HTTP 请求 → API server → tokenizer → ZMQ →
     ↓
   ★ Scheduler.run_forever 主循环 ★   ← 本文件
     ↓
   每轮:
     1. receive_msg 拉 UserMsg
     2. _process_one_msg 包装成 PendingReq
     3. _schedule_next_batch 决定本轮 batch
     4. _forward 喂给 engine.forward_batch
     5. _process_last_data 取采样结果发给 detokenizer
     ↑
   ZMQ → detokenizer → API server → 用户

【核心概念速览】

- overlap scheduling（重叠调度）:
    一个调度循环里"调度（CPU）"和"前向（GPU）"是串行的：
      调度a → GPU 跑a → 调度b → GPU 跑b → ...
    GPU 等调度时空转，浪费算力。
    Overlap 方案：在 GPU 跑 a 的同时，CPU 已经开始调度 b：
      [GPU: 跑a       ][跑b       ]
      [CPU: 调度b][调度c       ]
    实现关键：用两个 CUDA stream + 把"上一步结果的处理"和"下一步的调度"
    分离。本文件的 overlap_loop 就是这个模式的实现。

- CUDA Stream（CUDA 流）:
    GPU 上的"任务队列"。同一 stream 内任务严格按顺序；不同 stream 之间
    可以并行。通过 wait_stream 做同步点。

- inference_mode:
    torch.inference_mode() 是比 torch.no_grad() 更高效的"推理专用"上下文，
    完全关掉 autograd 相关的开销。

- ForwardInput / ForwardOutput:
    Scheduler 内部约定的"一步前向的输入输出"打包结构，用于 overlap 时
    把上一步的处理推迟到下一步进行（隐藏延迟）。

【关键设计决策】

1. 同时支持 overlap_loop 和 normal_loop：
   overlap 模式默认（更快）；ENV.DISABLE_OVERLAP_SCHEDULING=true 时
   走 normal 模式，便于调试。

2. Scheduler 继承 SchedulerIOMixin：
   把"跨进程消息收发"的复杂代码抽成 mixin，主类 scheduler.py 聚焦
   调度逻辑。

3. 用 NamedTuple 而不是 dataclass 装 ForwardInput：
   性能优化——NamedTuple 是 C 实现的轻量对象。

4. _free_req_resources 的两步释放：
   先 free table_idx（页表行马上能给别人用），再 cache_req(finished=True)
   把 KV 喂给前缀缓存（其他请求可能能复用）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

# Indice2D 类型别名：（行索引, 列索引/位置）的二元组，用于在 GPU 上做
# fancy indexing 取/写 page_table、token_pool。
Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# ════════════════════════════════════════════════════════════════════
# ForwardInput / ForwardData 解说
# ════════════════════════════════════════════════════════════════════
#
# ForwardInput 装着一个 batch 在送 GPU 前需要的所有信息：
#   - batch:         本轮的 Batch 对象本身
#   - sample_args:   采样器所需参数（temperature/top_k 等已预处理）
#   - input_tuple:   (token_mapping, positions)
#                    告诉前向 kernel：每个 token 在 token_pool 的哪个位置取 id，
#                    以及它的"序列位置"（用作 RoPE）
#   - write_tuple:   (req_mapping, seq_lens or -1)
#                    告诉采样器：把采样出来的 next_token 写到 token_pool 的哪行哪列
#                    -1 表示该请求不需要写（如 ChunkedReq）
#
# ForwardData = (ForwardInput, ForwardOutput) —— 一步的完整记录
# overlap 模式把它存到下轮再"处理结果"，从而和下轮前向重叠。
# ════════════════════════════════════════════════════════════════════
# For overlap scheduling, we also need to cache some other data to avoid IMA
# (Illegal Memory Access)
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：Scheduler（读代码之前请先读完这段） ███
# ════════════════════════════════════════════════════════════════════
#
# 一、这个类要解决什么问题？
#
#   把零散的"接收/调度/计算/回收"四件事拧成一个永远运行的主循环。
#   既要高吞吐（多人共享 GPU），又要低延迟（隐藏调度开销）。
#
# 二、主循环 overlap_loop 的执行流程（用具体例子走一遍）
#
#   假设上一轮 last_data = (上一轮的 ForwardInput, 上一轮的 ForwardOutput)
#   也就是 GPU 上一轮算的结果。本轮要做：
#
#   ① 拉新消息（receive_msg）
#     从 ZMQ 拉所有等待处理的 UserMsg，逐个 _process_one_msg：
#       - UserMsg → prefill_manager.add_one_req
#       - AbortBackendMsg → 终止某个请求
#       - ExitMsg → KeyboardInterrupt 退出
#     blocking 条件: 如果没有任何要做的事，就阻塞等消息；否则立即返回。
#
#   ② 调度下一轮 batch（_schedule_next_batch）
#     先看 prefill 有没有可调度的请求；没有再看 decode。
#     调度成功 → 得到一个 ForwardInput；否则 None。
#
#   ③ 启动本轮前向（_forward）
#     在 engine 的 stream 上下文里调 engine.forward_batch。
#     注意 stream.wait_stream(self.stream) ——本轮前向必须等前面的
#     metadata 准备完毕。
#     得到 ForwardOutput 后，包成 ongoing_data 返回。
#
#   ④ 处理上一轮的结果（_process_last_data）
#     在 GPU 跑本轮的同时，CPU 这边处理 last_data：
#       - 等 D2H 拷贝完成（拿到 next_token 在 CPU 上）
#       - 对每个真实请求:
#         · append_host 新 token
#         · 判断是否结束（达到 max_tokens / EOS）
#         · 结束 → 释放资源；未结束 → 调 cache_req(finished=False)
#       - 把 DetokenizeMsg 列表发给 detokenizer
#
#   这一轮返回 ongoing_data，作为下一轮的 last_data。如此循环。
#
# 三、为什么 receive_msg 在最前面？
#   先把消息全收下来（包括用户 abort），避免半途调度了一个已经被
#   abort 的请求。
#
# 四、ChunkedReq 不参加采样
#   _process_last_data 里 isinstance(req, ChunkedReq) 就 continue——它
#   还在读 prompt，没有 logits。
#
# ════════════════════════════════════════════════════════════════════
class Scheduler(SchedulerIOMixin):
    """
    【类名】Scheduler（推理后端的调度器主类）
    【一句话描述】整个推理系统的指挥中心，运行永不停止的主循环。
    【生活类比】快餐店的"厨房调度员"——盯着取餐口（pending list），
                安排锅炉（GPU）按节奏出餐，让 N 个顾客的菜都能尽量快上桌。
    """

    def __init__(self, config: SchedulerConfig):
        """
        【功能】初始化整个调度器——建立 engine、各管理器、流、tokenizer 等。

        【参数】config: SchedulerConfig，包含所有启动参数

        【内部初始化步骤】
        1. 创建 Engine（载入模型、KV pool、attention 后端）；
        2. 创建两条 CUDA stream：
           - self.stream: 元数据处理（数据拷贝、采样准备等）
           - engine.stream: 真正跑前向的流
           两条 stream 异步并行 → overlap scheduling 的基础；
        3. 创建 4 个子管理器：TableManager、CacheManager、DecodeManager、PrefillManager；
        4. 载入 tokenizer 拿 EOS token id（用于判断生成结束）；
        5. 调用 mixin 的 __init__ 设置 ZMQ 通信。
        """
        from minisgl.engine import Engine

        # 1. 创建引擎（这是最重的初始化——载模型权重、分配 KV pool 等）
        self.engine = Engine(config)

        # 2. 创建 CPU 端用于"准备工作"的 CUDA stream
        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        # 新建一个 stream，让 metadata 操作（host→device 拷贝、prepare 等）
        # 不阻塞 engine.stream 上的真正前向计算
        self.stream = torch.cuda.Stream(device=self.device)
        # 一个上下文管理器，进入它后默认 stream 切到 engine.stream
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        # 默认让本进程在 self.stream 上执行
        torch.cuda.set_stream(self.stream)

        # 3. 初始化各子管理器
        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # 4. 便利字段
        # some alias for easy access
        # 已完成的请求集合——避免 overlap 模式下重复释放
        self.finished_reqs: Set[Req] = set()
        # tokenizer 用来取 EOS token id（采到这个 token 默认要停下来）
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        # self.config = config

        # 5. 初始化 I/O mixin（ZMQ 通信）
        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """
        【功能】调度器没活干、要阻塞等消息时调用——顺便做点后台维护工作。
        【当前实现】只跑一次缓存完整性检查（catch 内部 bug）。
        """
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        【功能】重叠调度版本的"一次循环"——把"处理上一步结果"与"启动
                下一步前向"重叠执行，最大化 GPU 利用率。

        【参数】last_data: 上一轮的 (ForwardInput, ForwardOutput)。
                第一轮调用时为 None。

        【返回】本轮的 (ForwardInput, ForwardOutput)；如果本轮没有 batch
                就返回 None（下一轮 last_data 也是 None）。

        【内部流程】
        1. 判断要不要 blocking 等消息：
           - 如果有 last_data（要处理）/ 有 pending prefill / 有 running decode
             → 不阻塞（立即返回所有已到达的消息）
           - 否则真的没事干 → 阻塞等下一条消息
        2. 把所有拉到的消息处理掉
        3. _schedule_next_batch 决定本轮 batch
        4. 启动 GPU 前向（异步，立即返回 ForwardOutput 句柄）
        5. 处理上一轮结果（这时 GPU 正在跑本轮，CPU 处理 last_data 不阻塞）

        【为什么这样能 overlap？】
        forward 启动后立即返回（CUDA 异步），CPU 继续往下做
        _process_last_data。而 GPU 那边继续跑本轮 batch，两边并行。
        """
        # 判断是否阻塞：只有所有队列都空时才阻塞
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        # 收所有消息（blocking 时至少等到一条；非 blocking 时拉所有已到达的）
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # 决定本轮 batch
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            # 切到 engine.stream 跑前向；先等 self.stream 上的 metadata 准备完
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        # 处理上一轮结果（与本轮 GPU 前向并行）
        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        """
        【功能】非重叠版本：调度 → 前向 → 处理结果，严格串行。
        【用途】DISABLE_OVERLAP_SCHEDULING 时使用；调试用。
        """
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        # 立即处理本轮结果（不延迟到下一轮）
        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        """
        【功能】调度器进程的"永远运行"入口。
        【返回类型 NoReturn】表示永不正常返回——只能通过 KeyboardInterrupt 退出。
        【inference_mode】关掉所有 autograd 开销，纯推理性能更好。
        """
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            # 调试模式：所有计算都在 engine.stream 上
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            # 生产模式：CPU 跑 self.stream，GPU 跑 engine.stream
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                # 上一轮返回的 data 作为本轮的 last_data，形成滚动重叠
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        """
        【功能】优雅关闭：等 GPU 把所有未完成任务跑完、同步各 TP rank、关闭 engine。
        """
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        """
        【功能】处理上一轮前向的结果：
          - 取采样出的 next_token；
          - 更新每个请求的状态；
          - 判定是否完成；
          - 完成 → 释放资源；未完成 → 把 prompt KV 喂给前缀缓存；
          - 把 DetokenizeMsg 发给 detokenizer。

        【参数】last_data: 上一轮的 (ForwardInput, ForwardOutput)；None 表示首轮。
        """
        if last_data is None:
            return

        # 解开 last_data：拿到 batch 和 (next_tokens_gpu, next_tokens_cpu, copy_done_event)
        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        # 等 D2H 拷贝完成（next_tokens 从 GPU 拷到 CPU 才能读）
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        # 用 lazy_free 区域把本轮所有 _free 调用合并成一次性 cat（性能优化）
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                # ChunkedReq 跳过——它没产生有效采样
                if isinstance(req, ChunkedReq):
                    continue
                # 拿到本请求采样得到的 next_token（CPU 上的标量）
                next_token = next_tokens_cpu[i]
                # 把它追加到请求的 input_ids（host 侧）
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                # 判定是否完成：(a) 达到 max_tokens (b) 采到 EOS（且没忽略 EOS）
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                # 构造发给 detokenizer 的消息
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # 处理资源释放/缓存
                # NOTE: overlap scheduling may make the request freed twice, skip second free
                # ⚠️ overlap 模式下，可能同一个请求在两个滚动周期里都被处理，要去重
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:
                    # prefill 后非完成的请求：把 prompt 那段 KV 插入前缀缓存
                    # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        # 更新已完成集合（用于下一轮去重）
        self.finished_reqs = new_finished_reqs
        # 把这一批结果发给 detokenizer
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        """
        【功能】处理一条从外部进程发来的消息。

        【参数】msg: 任意 BaseBackendMsg 的子类实例

        【支持的消息类型】
        - BatchBackendMsg: 一批消息（递归处理）
        - ExitMsg: 退出命令（抛 KeyboardInterrupt）
        - UserMsg: 用户请求（核心路径）
        - AbortBackendMsg: 用户取消请求
        - 其他: 当前未实现，抛 NotImplementedError
        """
        if isinstance(msg, BatchBackendMsg):
            # 批消息：拆开逐个处理
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            # 退出信号：用 KeyboardInterrupt 触发 finally 清理
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            # 防御性处理：input_ids 超长 → 丢弃
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            # 用户要的 max_tokens 太大 → 自动 clip 到剩余空间
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            # 加入 pending 队列
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            # 先在 pending 里找；找到了它的 chunked_req 才需要释放资源
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            # 没在 pending 里 → 看看是不是在 decode 里
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        """
        【功能】请求生命终结：释放它占的页表行 + 把 KV 喂给前缀缓存（finished=True）。
        """
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        """
        【功能】把 Manager 返回的"骨架 Batch"补齐为可以送给 GPU 的 ForwardInput。

        【内部步骤】
        1. graph_runner.pad_batch：补齐到 CUDA Graph 档位（padded_reqs）；
        2. cache_manager.allocate_paged：为本轮新算的 token 分配 KV slot；
        3. _make_positions：构造每个 token 的 position id（GPU 张量）；
        4. _make_input_tuple：构造 token id 取址的索引；
        5. _make_write_tuple：构造 next_token 写回的索引；
        6. batch.out_loc：根据 input_mapping 取出对应的 page_table 槽位；
        7. attn_backend.prepare_metadata：注意力后端构造自己的元数据；
        8. sampler.prepare：采样器根据 sampling_params 做预处理；
        9. 打包成 ForwardInput。
        """
        # CUDA Graph 需要固定形状 → 用 dummy 请求把 batch size 补到固定档位
        self.engine.graph_runner.pad_batch(batch)
        # 为本轮新算的 token 分配 KV slot 并写入 page_table
        self.cache_manager.allocate_paged(batch.reqs)
        # 构造 positions 张量（每个 token 在自己序列里的位置）
        batch.positions = _make_positions(batch, self.device)
        # 构造索引（取 token id 的 source 索引）
        input_mapping = _make_input_tuple(batch, self.device)
        # 构造索引（采样结果写回 token_pool 的目标索引）
        write_mapping = _make_write_tuple(batch, self.device)
        # 通过 input_mapping fancy-index 取出每个 token 对应的 slot 编号
        batch.out_loc = self.engine.page_table[input_mapping]
        # 让 attention backend 准备自己的元数据（如 page_table、seq_lens 等）
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        """
        【功能】决定本轮跑什么 batch：prefill 优先，否则 decode。

        【为什么 prefill 优先】
        prefill 用满预算的 token 数远超 decode（decode 每请求 1 token），
        所以先做 prefill 让吞吐量更高。但也要注意：长 prompt 的请求会
        通过 chunked prefill 分摊到多轮，不会持续饿死 decode。
        """
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        """
        【功能】执行一次前向：构造 input_ids、调 engine.forward_batch、把
                采样结果写回 token_pool、更新 decode_manager。

        【内部步骤】
        1. 从 token_pool 用 input_mapping 取出本轮要算的 token id；
        2. 调 engine.forward_batch（异步，GPU 上算 logits + 采样）；
        3. 把 next_tokens_gpu 通过 write_mapping 写回 token_pool（供下轮用）；
        4. filter_reqs：把刚 prefill 完的请求并入 decode 队列。
        """
        batch, sample_args, input_mapping, output_mapping = forward_input
        # 取 token id 张量（GPU 上的 long）
        batch.input_ids = self.token_pool[input_mapping]
        # GPU 跑前向 + 采样
        forward_output = self.engine.forward_batch(batch, sample_args)
        # 把采样得到的 next_token 写回 token_pool（下一轮的 input 就在这里）
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        # prefill 完的请求自动并入 decode 池（在 forward_input.batch.reqs 里）
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


# ════════════════════════════════════════════════════════════════════
# 下面三个辅助函数构造 GPU 端"位置/取址/写址"张量
# 它们都遵循同一个模式：先在 pin_memory 上构造 host 张量，再异步拷到 GPU。
# ════════════════════════════════════════════════════════════════════


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    """
    【功能】构造每个 token 的 position id（在自己序列中的位置）。

    【例子】batch.padded_reqs = [
              Req(cached=2, device=5, extend=3),  → positions += [2,3,4]
              Req(cached=4, device=6, extend=2),  → positions += [4,5]
            ]
            结果: tensor([2, 3, 4, 4, 5])

    【用途】Transformer 用 RoPE 时需要每个 token 的位置编码。
    """
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        # [cached_len, cached_len+1, ..., device_len-1]
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    """
    【功能】构造取 token id 用的二维索引 (row, col)。

    【例子】batch.padded_reqs = [
              Req(table_idx=7, extend=3),  → mapping += [7,7,7], positions += [2,3,4]
              Req(table_idx=2, extend=2),  → mapping += [2,2],   positions += [4,5]
            ]
            返回:
              row = tensor([7,7,7, 2,2])
              col = batch.positions（前面已经构造好）
            后续 token_pool[row, col] 就能取出每个 token 的 id。

    【为什么 row 不能直接用 cumsum 计算？】
    每个请求 extend_len 不同，需要重复 table_idx 填充对应段。
    """
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        # 同一个请求的所有 token 都用同一个 table_idx 作为行
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    """
    【功能】构造把采样结果写回 token_pool 用的索引。

    【说明】
    - row = 每个请求的 table_idx（每个请求只写 1 个采样结果——下一个 token）
    - col = 该请求的 device_len（下一个 token 要写的位置）
             如果该请求不需要写（如 ChunkedReq.can_decode=False），col=-1
             表示"丢弃这一项"——后续 fancy index 会处理 -1 索引。
    - 注意：用 batch.reqs（不是 padded_reqs）——padding 的 dummy 不写。

    【-1 哨兵】write_host 里 -1 表示"这个采样结果不要写"——常见于
    ChunkedReq（它还没读完 prompt，下一轮才有有效的预测）。
    ⚠️ 设计上：被设为 -1 的位置，对应的 next_token 也不会被使用，
    所以哪怕 token_pool[row, -1] 会写到最后一列也无害（但事实上代码会避免）。
    """
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
