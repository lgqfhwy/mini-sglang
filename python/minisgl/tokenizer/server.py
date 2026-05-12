"""
========================================================================
文件名: tokenizer/server.py
所属模块: Tokenizer 进程主循环 (worker)
========================================================================

【这个文件是做什么的 - 一句话总结】
定义 tokenizer 子进程的"主循环函数"——一个无限循环，
从 ZMQ 拉消息（tokenize 请求 + detokenize 请求 + abort 通知混在一起），
分别处理后再分别发给 scheduler 和 frontend。

【为什么用一个 worker 处理两种角色】
tokenize 和 detokenize 都基于同一个 tokenizer 实例，又都是 CPU 操作；
合并成一个 worker 进程能：
  - 复用 tokenizer 实例，省内存；
  - 减少进程数（每个进程都有 IPC overhead）。

【消息流向】
   frontend(API)  → TokenizeMsg  → 本 worker → UserMsg → scheduler
   scheduler      → DetokenizeMsg → 本 worker → UserReply → frontend
   frontend(API)  → AbortMsg     → 本 worker → AbortBackendMsg → scheduler

【local_bs（局部批大小）】
worker 一次从 ZMQ 拉到第一条消息后，不立刻处理——再尽量多拉几条
（凑齐 local_bs 条或队列空了为止），然后批量 tokenize/detokenize。
理由：tokenizer.batch_decode 批量比单条快很多。

【进程通信参数】
- addr:          本 worker 的 PULL 地址（接收消息）
- backend_addr:  发给 scheduler 的 PUSH 地址
- frontend_addr: 发给 frontend 的 PUSH 地址
- ack_queue:     进程间同步用——worker 就绪后往里 put 一个字符串，
                  父进程读到即可继续。
"""

from __future__ import annotations

import multiprocessing as mp
from typing import List

import torch
from minisgl.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger, load_tokenizer


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    """
    【功能】把可能是批消息的"信封"拆开成多条小消息。
    BatchTokenizerMsg 是"批"，普通消息就是单条——这里统一返回 list。
    """
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


# ════════════════════════════════════════════════════════════════════
# tokenize_worker - tokenizer 子进程的入口函数
# ────────────────────────────────────────────────────────────────────
# 用 @torch.inference_mode() 关掉 autograd 开销（虽然本函数不直接动
# tensor 计算，但保险起见）。
#
# 主循环每轮：
#   1. 从 ZMQ 拉一条阻塞消息；
#   2. 然后尽量再多拉几条直到攒够 local_bs 或队列空；
#   3. 按类型分组（detokenize / tokenize / abort）；
#   4. 各组分别处理：
#      - detokenize → 转字符串 → UserReply → 发给 frontend
#      - tokenize → 转 token id → UserMsg → 发给 scheduler
#      - abort → AbortBackendMsg → 发给 scheduler
#   5. 单条/多条自动选择是否打包成 BatchMsg。
# ════════════════════════════════════════════════════════════════════
@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    model_source: str = "huggingface",
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    """tokenizer 子进程的主循环。

    【参数】
    - tokenizer_path: 加载 tokenizer 的路径（通常和模型同目录）
    - addr: 本 worker 拉消息的 ZMQ 地址
    - create: 是否由本 worker 负责 bind socket（True）还是 connect（False）
    - backend_addr: 发给 scheduler 的 ZMQ 地址
    - frontend_addr: 发给 frontend 的 ZMQ 地址
    - local_bs: 单次循环最多攒多少条消息再批处理
    - tokenizer_id: 这是第几个 tokenizer 进程（日志用）
    - model_source: 模型来源（"huggingface" 或 "modelscope"）
    - ack_queue: 启动同步用，就绪后往里 put 一个字符串
    """
    # 创建 ZMQ 队列
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    # 加载 tokenizer（HF 的 PreTrainedTokenizer）
    tokenizer = load_tokenizer(tokenizer_path)
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    # 延迟 import 避免循环依赖
    from .detokenize import DetokenizeManager
    from .tokenize import TokenizeManager

    tokenize_manager = TokenizeManager(tokenizer)
    detokenize_manager = DetokenizeManager(tokenizer)

    # 通知父进程"本 worker 已就绪"
    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            # 阻塞拉第一条消息，然后尽量再拉到 local_bs 条或队列空
            pending_msg = _unwrap_msg(recv_listener.get())
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            # 按类型分组
            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
            abort_msg = [m for m in pending_msg if isinstance(m, AbortMsg)]
            assert len(detokenize_msg) + len(tokenize_msg) + len(abort_msg) == len(pending_msg)

            # 处理 detokenize：token id → 增量文本 → 给前端
            if len(detokenize_msg) > 0:
                replies = detokenize_manager.detokenize(detokenize_msg)
                batch_output = BatchFrontendMsg(
                    data=[
                        UserReply(
                            uid=msg.uid,
                            incremental_output=reply,
                            finished=msg.finished,
                        )
                        for msg, reply in zip(detokenize_msg, replies, strict=True)
                    ]
                )
                # 只有一条时不打包（省一层封装）
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_frontend.put(batch_output)

            # 处理 tokenize：文本 → token id → 给后端 scheduler
            if len(tokenize_msg) > 0:
                tensors = tokenize_manager.tokenize(tokenize_msg)
                batch_output = BatchBackendMsg(
                    data=[
                        UserMsg(
                            uid=msg.uid,
                            input_ids=t,
                            sampling_params=msg.sampling_params,
                        )
                        for msg, t in zip(tokenize_msg, tensors, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)

            # 处理 abort：用户取消 → 转发给后端
            if len(abort_msg) > 0:
                batch_output = BatchBackendMsg(
                    data=[AbortBackendMsg(uid=msg.uid) for msg in abort_msg]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)
    except KeyboardInterrupt:
        # 父进程发 SIGINT 让 worker 退出
        pass
