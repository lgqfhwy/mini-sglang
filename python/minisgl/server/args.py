"""
========================================================================
文件名: server/args.py
所属模块: 命令行参数解析 + ServerArgs（最完整的配置类）
========================================================================

【这个文件是做什么的 - 一句话总结】
1. 定义 ServerArgs—— ServerArgs ⊃ SchedulerConfig ⊃ EngineConfig，
   最完整的配置类，包括 HTTP 服务地址、tokenizer 数量等；
2. 提供 parse_args() —— 把 sys.argv 解析成 ServerArgs。

【ServerArgs 在配置类继承链中的位置】
   EngineConfig（model_path / tp_info / dtype 等）
       ↑ 继承
   SchedulerConfig（+ max_extend_tokens / cache_type / zmq 地址）
       ↑ 继承
   ★ ServerArgs（+ server_host/port / num_tokenizer / 等）★
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from minisgl.distributed import DistributedInfo
from minisgl.scheduler import SchedulerConfig
from minisgl.utils import init_logger


# ════════════════════════════════════════════════════════════════════
# ServerArgs - 完整的服务器配置
# ────────────────────────────────────────────────────────────────────
# 继承 SchedulerConfig，加入"服务器特有"字段：
#   - server_host/port: HTTP 监听地址
#   - num_tokenizer:    起几个 tokenizer 进程（0=和 detokenizer 共用一个）
#   - silent_output:    静默模式（shell 用）
#
# share_tokenizer 模式 (num_tokenizer == 0):
#   只起 1 个 worker 同时承担 tokenize + detokenize。
#   省进程数，但单进程可能成为瓶颈。
#
# 否则 (num_tokenizer > 0):
#   起 num_tokenizer 个 tokenize 专职 worker + 1 个 detokenize 专职 worker。
# ════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    """完整的服务器配置——含 HTTP + 进程 + 调度器 + 引擎所有参数。"""

    server_host: str = "127.0.0.1"   # HTTP 服务监听 host
    server_port: int = 1919          # HTTP 服务监听 port
    # tokenizer 进程数（0 表示和 detokenizer 共用一个 worker）
    num_tokenizer: int = 0
    silent_output: bool = False      # shell 模式下关掉日志

    @property
    def share_tokenizer(self) -> bool:
        """True = tokenizer 和 detokenizer 共用一个进程。"""
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        """detokenizer → API server 的 ZMQ 地址。"""
        return "ipc:///tmp/minisgl_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        """
        API server → tokenizer 的 ZMQ 地址。
        共享模式时复用 detokenizer 的地址（同一个 worker 收两种消息）。
        """
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/minisgl_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        """tokenizer 是否负责 bind 自己的 socket（vs connect）。"""
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        """覆盖 SchedulerConfig 同名属性。share 模式下由 tokenizer 创建。"""
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        """frontend 是否负责 bind 到 tokenizer 的 socket。"""
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        """torch.distributed 用的地址——和 HTTP server_port 错开 1。"""
        return f"tcp://127.0.0.1:{self.server_port + 1}"


def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return an EngineConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        EngineConfig instance with parsed arguments

    【功能】把 sys.argv 解析成 ServerArgs，返回 (args, run_shell)。

    【流程】
    1. 用 argparse 定义所有参数；
    2. 解析；
    3. 一系列后处理：
       - run_shell 模式下强制 cuda_graph_max_bs=1（避免占太多显存）
       - ~ 路径展开
       - modelscope 模型下载
       - dtype 字符串 → torch.dtype
       - 构造 DistributedInfo 包装 rank/size
    4. 构造 ServerArgs 返回。
    """
    # 延迟 import：argparse 的 choices 依赖这些模块，但运行时才用得到
    from minisgl.attention import validate_attn_backend
    from minisgl.kvcache import SUPPORTED_CACHE_MANAGER
    from minisgl.moe import SUPPORTED_MOE_BACKENDS

    parser = argparse.ArgumentParser(description="MiniSGL Server Arguments")

    # 必填：模型路径
    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    # 精度
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    # TP 大小
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    # 并发上限
    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    # 序列长度上限覆盖
    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    # 显存比例
    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,
        help="The fraction of GPU memory to use for KV cache.",
    )

    # 用 dummy 权重（测试）
    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    # 禁用 pynccl
    assert ServerArgs.use_pynccl == True
    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    # HTTP 服务地址
    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    # CUDA Graph 上限
    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    # tokenizer 进程数
    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    # 单次 prefill 上限
    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    # KV pool 页数手动设置
    parser.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    # 页大小
    parser.add_argument(
        "--page-size",
        type=int,
        default=ServerArgs.page_size,
        help="Set the page size for system management.",
    )

    # 注意力后端
    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_attn_backend,
        default=ServerArgs.attention_backend,
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    # 模型下载源
    parser.add_argument(
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="The source to download model from. Either 'huggingface' or 'modelscope'.",
    )

    # 前缀缓存类型
    parser.add_argument(
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="The KV cache management strategy.",
    )

    # MoE 后端
    parser.add_argument(
        "--moe-backend",
        default=ServerArgs.moe_backend,
        choices=["auto"] + SUPPORTED_MOE_BACKENDS.supported_names(),
        help="The MoE backend to use.",
    )

    # shell 模式
    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    # Parse arguments
    kwargs = parser.parse_args(args).__dict__.copy()

    # ----- 后处理 -----
    # shell 模式优化：单请求、超小 graph
    run_shell |= kwargs.pop("shell_mode")
    if run_shell:
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    # ~ 路径展开
    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    # modelscope 自动下载
    if kwargs["model_source"] == "modelscope":
        model_path = kwargs["model_path"]
        if not os.path.isdir(model_path):
            from modelscope import snapshot_download

            ignore_patterns = []
            if kwargs["use_dummy_weight"]:
                # dummy weights 不需要真正下载权重文件
                ignore_patterns = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"]
            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)
            kwargs["model_path"] = model_path
    del kwargs["model_source"]

    # dtype: "auto" → 从 HF config 推断；其它 → 字符串映射成 torch.dtype
    if (dtype_str := kwargs["dtype"]) == "auto":
        from minisgl.utils import cached_load_hf_config

        dtype_str = cached_load_hf_config(kwargs["model_path"]).dtype

    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs["dtype"] = DTYPE_MAP[dtype_str] if isinstance(dtype_str, str) else dtype_str
    # 这里默认 rank=0；多 TP 时 launch.py 会在每个子进程里覆盖 rank
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell
