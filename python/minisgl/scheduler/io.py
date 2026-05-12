"""
========================================================================
文件名: scheduler/io.py
所属模块: 调度器 - 跨进程消息收发与多卡同步
========================================================================

【这个文件是做什么的 - 一句话总结】
这个文件是一个 Mixin（"混入类"），专门负责调度器与外部世界的通信：
  - 从 tokenizer 进程拉用户请求（UserMsg）；
  - 把采样结果（DetokenizeMsg）推给 detokenizer 进程；
  - 多 GPU 张量并行时，把请求广播给其它 rank 让所有 GPU 看到同样的消息。

【为什么需要这个文件 / 这个模块存在的原因】
mini-sglang 把整个推理系统拆成多个独立进程：
  - API server 进程：处理 HTTP
  - tokenizer 进程：文本 ↔ token id
  - scheduler 进程：批调度 + GPU 前向
  - detokenizer 进程：token id ↔ 增量文本
为什么拆？
  - 各进程可以用不同 CPU 核，避免 GIL 争抢；
  - 每个进程职责单一，崩了不会拖累整体；
  - 多 GPU TP（张量并行）时，scheduler 是多份（每 GPU 一份）。

进程之间用 ZMQ 通信。这个文件把"怎么收/怎么发/怎么广播到其它 GPU 的
scheduler"封装成 Mixin，让主 Scheduler 类聚焦在调度逻辑上。

【这个文件在整个推理流程中的位置】
   API/tokenizer 进程
     ↓ ZMQ PUSH
   ★ SchedulerIOMixin.receive_msg → 主 scheduler 拿到 UserMsg ★
     ↓ 主 scheduler 调度
   ★ SchedulerIOMixin.send_result → ZMQ PUSH ★
     ↓
   detokenizer 进程

【核心概念速览】

- TP（Tensor Parallelism / 张量并行）：
    一个大模型的权重切分到多个 GPU 上，每个 GPU 算一部分，
    最后汇总。需要 N 个 scheduler 进程（每 GPU 一个），它们必须看到
    完全相同的消息序列才能保持步调一致。本文件处理这种"广播给所有
    rank"的逻辑。

- rank 0（主 rank）与其他 rank：
    多卡场景下，约定 rank 0 是"主"——它负责对外通信（接 ZMQ 消息）。
    其他 rank 不直接连 ZMQ，而是通过 rank 0 转发（rank0 → 用另一个
    ZMQ PUB 把原始字节广播给 rank 1..N）。

- ZMQ 消息模式：
    PULL/PUSH: 单工，多对一/一对多；非阻塞读；本项目 scheduler ←
      tokenizer 用 PULL/PUSH。
    PUB/SUB:   广播。本项目 rank0 → 其他 ranks 用 PUB/SUB。

- offline_mode（离线模式）：
    跑 benchmark 时不需要 ZMQ，直接 in-process 调用接口拿消息。
    所以 __init__ 里如果 offline_mode=True，直接把 receive_msg /
    send_result 替换成抛 NotImplementedError 的占位（实际实现在子类）。

【关键设计决策】

1. 用 Mixin 而不是组合：
   Scheduler 既要持有自己的状态，也要通信功能"直接挂在 self 上"——
   Mixin 模式让 self.receive_msg / self.send_result 能像本类方法一样用，
   而不必写成 self.io.receive_msg。

2. receive_msg 在多卡场景下做"主从同步"：
   - rank0: 从 ZMQ 拉消息，同时通过 PUB 广播给其它 rank；
   - rank1+: 不连 ZMQ，从 PUB 接收。
   所有 rank 通过 broadcast(int) 把"本轮收到几条消息"先对齐，确保
   不会出现 rank0 收到 3 条而 rank1 等到第 4 条这种死锁。

3. send_result 只在 rank0 执行：
   答复给 detokenizer 是单向输出，没必要让每个 rank 都发一遍。
   rank1+ 的 send_result 是 no-op。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, List

import torch
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg, DetokenizeMsg
from minisgl.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：SchedulerIOMixin（读代码之前请先读完这段）███
# ════════════════════════════════════════════════════════════════════
#
# 一、这个类要解决什么问题？
#
#   调度器在多卡 TP 场景下的通信会变得复杂：
#     - 只有 rank0 连 ZMQ，但所有 rank 都需要看到相同的消息；
#     - rank0 收到一批消息后要先广播条数，所有 rank 同步后再各自取消息；
#     - 否则就可能出现：rank0 收到 3 条 → 调度了 3 个请求；
#       rank1 因为还没收到只调度 1 个 → 两边 batch 不一致 → NCCL 死锁。
#
#   SchedulerIOMixin 把这套"看似简单实则容易死锁"的协议封装好。
#
# 二、TP=1（单卡）情形：最简
#
#   recv = self._recv_msg_single_rank
#     blocking=True: 先调 run_when_idle，再 _recv_from_tokenizer.get() 阻塞拿一条
#     blocking=False: 把队列里所有现有的拿出来
#   send = self._reply_tokenizer_rank0
#     1 条就直接 put；多条就打包成 BatchTokenizerMsg 一次性发
#
# 三、TP>1（多卡）情形：rank0 主导广播
#
#   假设 TP=2，rank0 和 rank1 都跑 Scheduler。
#
#   recv 在 rank0:
#     blocking=True:
#       run_when_idle → 阻塞拉一条 raw bytes →
#       通过 ZMQ PUB 转发给所有 rank → 自己 decode 一份
#     接着把队列里剩余的也都拿出来转发
#     最后 broadcast(本批条数, root=0) 让 rank1 知道要收几条
#     然后逐条 decode（最后这步也可以省略，因为 raw 已经发过去了）
#
#   recv 在 rank1:
#     blocking=True: 通过 SUB 阻塞拉一条
#     broadcast(条数, root=0) 收到 rank0 通知的总条数
#     从 SUB 队列再拿剩下的 N-1 条
#
#   send 在 rank0: 正常发给 detokenizer
#   send 在 rank1: no-op（rank0 已经代发了）
#
# 四、offline_mode
#
#   __init__ 检测到 offline_mode 时把 receive_msg 和 send_result 替换成抛
#   NotImplementedError 的占位——子类（如 OfflineScheduler）必须重写。
#
# ════════════════════════════════════════════════════════════════════
class SchedulerIOMixin:
    """
    Mixin class for Scheduler I/O operations.

    This class handles the communication between the scheduler and the tokenizer.

    Public Utilities:
        receive_msg: Function to receive messages from the tokenizer.
        send_result: Function to send results back to the tokenizer.
        sync_all_ranks: Function to synchronize all ranks on CPU side.
    """

    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        """
        【功能】根据配置（单卡/多卡/离线）选择合适的 receive_msg 和 send_result 实现。

        【参数】
        - config: SchedulerConfig，包含 zmq 地址、tp_info、offline_mode 等
        - tp_cpu_group: torch.distributed 的 CPU 进程组（用作 barrier、broadcast）

        【关键决策点】
        1. offline_mode → 用占位实现（子类必须覆写）
        2. 单 TP rank → 简单的 PULL/PUSH，rank0 自己收发
        3. 多 TP rank:
           - rank 0: 走 _recv_msg_multi_rank0 + _reply_tokenizer_rank0
           - rank 1+: 走 _recv_msg_multi_rank1 + _reply_tokenizer_rank1 (no-op)
        """
        tp_info = config.tp_info
        # tp_cpu_group 用 Final 标注表示"启动后不再改"，提升 mypy 检查精度
        self.tp_cpu_group: Final = tp_cpu_group

        if config.offline_mode:
            # 离线模式：通信方法由子类提供
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return  # early exit

        if tp_info.is_primary():
            # rank0：负责连 tokenizer 进程的 ZMQ 队列
            self._recv_from_tokenizer: Final = ZmqPullQueue(
                config.zmq_backend_addr,
                create=True,
                decoder=BaseBackendMsg.decoder,
            )
            # rank0：负责发结果给 detokenizer
            self._send_into_tokenizer: Final = ZmqPushQueue(
                config.zmq_detokenizer_addr,
                create=config.backend_create_detokenizer_link,
                encoder=BaseTokenizerMsg.encoder,
            )

        # 默认走单 rank 路径
        recv = self._recv_msg_single_rank
        send = self._reply_tokenizer_rank0
        if tp_info.size > 1:
            # 多 rank 场景
            if tp_info.is_primary():
                # rank0：除接 ZMQ 外，还要 PUB 广播给其它 rank
                recv = self._recv_msg_multi_rank0
                self._send_into_ranks: Final = ZmqPubQueue(
                    config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder
                )
            else:
                # rank1+：不连 ZMQ，仅 SUB 从 rank0 接收
                recv = self._recv_msg_multi_rank1
                send = self._reply_tokenizer_rank1  # no-op
                self._recv_from_rank0: Final = ZmqSubQueue(
                    config.zmq_scheduler_broadcast_addr,
                    create=False,
                    decoder=BaseBackendMsg.decoder,
                )

        # 把选定的方法绑定到 self（这是 Mixin 的精髓——后续用 self.receive_msg）
        self.receive_msg = recv
        self.send_result = send

    def run_when_idle(self):
        """
        【功能】等消息时的空闲钩子。子类（Scheduler）实现具体内容
                （比如完整性自检）。
        """
        raise NotImplementedError("should be implemented")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """离线模式占位——子类必须重写。"""
        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        """离线模式占位——子类必须重写。"""
        raise NotImplementedError("should be implemented")

    def sync_all_ranks(self) -> None:
        """
        【功能】CPU 端 barrier——让所有 rank 在这里"对齐"再继续。
        【用途】shutdown 时、需要确保所有 rank 都跑到同一点的关键节点。
        """
        self.tp_cpu_group.barrier().wait()

    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """
        【功能】单卡场景下的消息收取。

        【参数】blocking: True=至少拿一条（阻塞等）；False=拿当前队列里所有

        【内部逻辑】
        1. blocking=True 时先调 run_when_idle，再阻塞 .get() 拿一条；
        2. 然后非阻塞地把队列里剩下的也都拿出来；
        3. 返回所有拿到的消息列表。
        """
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_tokenizer.get())
        # 把现有的全收下来——一轮调度尽量多消费消息，避免延迟堆积
        while not self._recv_from_tokenizer.empty():
            pending_msgs.append(self._recv_from_tokenizer.get())
        return pending_msgs

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """
        【功能】多卡场景下 rank0 的消息收取——除了自己收，还要广播给其它 rank。

        【内部逻辑（关键！）】
        1. blocking=True 时阻塞拿一条 raw bytes，立即通过 PUB 转发；
        2. 非阻塞地把剩余 raw bytes 全拿出来（先不转发）；
        3. 用 broadcast(int) 把"本批条数"发给所有 rank（这一步是关键同步点）；
        4. 然后才逐条转发并 decode——确保 rank1 知道要等几条；
        5. 返回 decode 后的消息列表。

        【为什么先 broadcast 条数再转发？】
        broadcast 是 NCCL 阻塞调用——必须所有 rank 都到达才能继续。
        这样 rank1 在等条数广播时，rank0 已经收完所有消息，避免：
          - rank1 误以为还有 X 条要等 → 调度死循环；
          - rank0 还没收完 → batch 大小不一致。

        【为什么 raw 转发然后再 decode？】
        rank1 也需要 decode 同样的字节流；rank0 自己也 decode 一份，
        这样两边逻辑对称（都拿到结构化的 BaseBackendMsg 列表）。
        """
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            # 阻塞拿一条 raw bytes，立刻转发给其它 rank，然后自己 decode
            raw = self._recv_from_tokenizer.get_raw()
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))

        # 非阻塞地收下队列里剩余的 raw
        pending_raw_msgs: List[bytes] = []
        while not self._recv_from_tokenizer.empty():
            pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

        # 关键同步点：让所有 rank 一致知道本批要接收多少条
        # broadcast the number of raw messages to all ranks
        src_tensor = torch.tensor(len(pending_raw_msgs))
        self.tp_cpu_group.broadcast(src_tensor, root=0).wait()

        # 然后把剩余 raw 一条条转发出去并 decode
        for raw in pending_raw_msgs:
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))
        return pending_msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """
        【功能】多卡场景下 rank1+ 的消息收取——只从 SUB 拿，不连 ZMQ 外部。

        【内部逻辑】
        1. blocking=True 时阻塞从 SUB 拿一条；
        2. broadcast(int) 等 rank0 告知本批剩余条数；
        3. 按条数从 SUB 拿剩下的。

        【为什么不能直接 while not empty】
        SUB 队列里可能"暂时空"但 rank0 还在转发中——必须按 rank0 通知的
        条数等够。这就是为什么 _recv_msg_multi_rank0 要先 broadcast 条数。
        """
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_rank0.get())

        # 等 rank0 通知本批剩余多少条
        # ensure all ranks have the same number of raw messages
        dst_tensor = torch.tensor(-1)
        self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()
        dst_length = int(dst_tensor.item())

        for _ in range(dst_length):
            pending_msgs.append(self._recv_from_rank0.get())
        return pending_msgs

    def _reply_tokenizer_rank0(self, reply: List[DetokenizeMsg]) -> None:
        """
        【功能】rank0 把一批 DetokenizeMsg 发给 detokenizer。

        【优化】
        - 0 条不发；
        - 1 条直接 put；
        - 多条打包成 BatchTokenizerMsg 一次发——减少 ZMQ 消息数量、减少
          反序列化开销。
        """
        num_reply = len(reply)
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
        if num_reply == 1:
            self._send_into_tokenizer.put(reply[0])
        elif num_reply > 1:
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))  # type: ignore

    def _reply_tokenizer_rank1(self, reply: List[DetokenizeMsg]) -> None:
        """
        【功能】rank1+ 的 send_result 是 no-op——rank0 已经代发了。
        【为什么参数还在】保持接口一致，可以无差别地调 self.send_result(reply)。
        """
        _ = reply  # do nothing for non-primary ranks
