"""
========================================================================
文件名: server/launch.py
所属模块: 进程启动器 - 多子进程协调入口
========================================================================

【这个文件是做什么的 - 一句话总结】
启动整个推理服务的"总开关"——
1. 解析命令行；
2. fork 出 N 个 scheduler 子进程（多 TP rank）；
3. fork 出 tokenizer 子进程 + detokenizer 子进程；
4. 启动 FastAPI HTTP server（主进程跑）；
5. 用 ack_queue 等所有子进程"准备就绪"再开始接收用户请求。

【为什么用多进程而不是多线程】
- Python GIL 限制：多线程不能真正并行；
- 进程隔离：scheduler 崩溃不会带垮 API server；
- 不同进程可以用不同 GPU（每个 scheduler 一张卡）。

【进程拓扑示例】
TP=2, num_tokenizer=2 时的进程：
  - 主进程：FastAPI API server
  - 子进程 1: scheduler rank 0 (GPU 0)
  - 子进程 2: scheduler rank 1 (GPU 1)
  - 子进程 3: detokenizer (1 个)
  - 子进程 4: tokenizer 0
  - 子进程 5: tokenizer 1
  全部通过 ZMQ IPC 通信。
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    """
    【功能】scheduler 子进程的入口函数（被 mp.Process 调用）。

    【流程】
    1. 用 inference_mode 创建 Scheduler 实例（这是最重的初始化）；
    2. 调 sync_all_ranks 等所有 TP rank 都初始化好；
    3. 主 rank 往 ack_queue 报告就绪；
    4. shell 模式下关掉 INFO 日志；
    5. run_forever 进入主循环；
    6. 收到 KeyboardInterrupt 时优雅 shutdown。
    """
    import torch
    from minisgl.scheduler import Scheduler

    with torch.inference_mode():
        # 重的初始化：加载模型、KV pool、注意力后端、CUDA Graph...
        scheduler = Scheduler(args)
        # 等所有 TP rank 都准备好
        scheduler.sync_all_ranks()

        if args.tp_info.is_primary():
            # 主 rank 通知父进程
            ack_queue.put("Scheduler is ready")

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


def launch_server(run_shell: bool = False) -> None:
    """
    【功能】启动整个服务的入口（外部 main 调用此函数）。

    【参数】run_shell: True = 启动交互式 shell；False = 启动 HTTP API。

    【流程】
    1. parse_args 解析 sys.argv；
    2. 定义内部函数 start_subprocess（不立即执行）：
       - 起 N 个 scheduler 进程（每 TP rank 一个）
       - 起 1 个 detokenizer 进程
       - 起 num_tokenizer 个 tokenizer 进程
       - 等 ack_queue 收齐所有就绪通知
    3. 调用 run_api_server，把 start_subprocess 作为 callback 传入
       （api_server 会在适当时机调用它启动子进程）。
    """
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> None:
        """启动所有子进程并等它们就绪。"""
        import multiprocessing as mp

        from minisgl.tokenizer import tokenize_worker

        # spawn 模式确保子进程是全新的 Python 解释器（避免 fork 复制 CUDA 状态）
        mp.set_start_method("spawn", force=True)

        world_size = server_args.tp_info.size
        # a multiprocessing queue to receive ack from subprocesses
        # so that we can guarantee all subprocesses are ready
        ack_queue: mp.Queue[str] = mp.Queue()

        # 起 N 个 scheduler 进程（每个 TP rank 一个）
        for i in range(world_size):
            # 用 replace（dataclass 的浅拷贝）改 rank
            new_args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size),
            )
            mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,
                name=f"minisgl-TP{i}-scheduler",
            ).start()

        num_tokenizers = server_args.num_tokenizer
        # 起 1 个 detokenizer 进程（不论 num_tokenizer 是几）
        # DeTokenizer, only 1
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="minisgl-detokenizer-0",
        ).start()
        # 起 num_tokenizer 个 tokenizer 进程
        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"minisgl-tokenizer-{i}",
            ).start()

        # Wait for acknowledgments from all worker processes:
        # - world_size schedulers (but only primary rank sends ack)
        # - num_tokenizers tokenizers
        # - 1 detokenizer
        # Total acks expected: 1 + num_tokenizers + 1 = num_tokenizers + 2
        # 等所有子进程都报告就绪
        for _ in range(num_tokenizers + 2):
            logger.info(ack_queue.get())

    # 启动 API server，里面会触发 start_subprocess
    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
