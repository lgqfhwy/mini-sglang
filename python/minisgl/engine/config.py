"""
========================================================================
文件名: engine/config.py
所属模块: 引擎（Engine）配置参数定义
========================================================================

【这个文件是做什么的 - 一句话总结】
定义"引擎（GPU 计算部分）"启动需要的所有参数：模型路径、TP 信息、
精度（dtype）、并发上限、注意力后端选择等。SchedulerConfig 也继承自它。

【这些参数中比较关键的几个】
- model_path: 模型权重文件夹路径（含 config.json、tokenizer 等）
- tp_info: 张量并行信息（当前进程 rank、总 size）
- dtype: 推理精度（bfloat16/float16/float8）
- max_running_req: 最大并发请求数（决定 page_table 的行数）
- attention_backend: 注意力 kernel 实现（fa = FlashAttention / fi = FlashInfer / trtllm）
- moe_backend: MoE 后端（"fused" / "auto"）
- cuda_graph_bs: 哪些 batch size 要预捕获 CUDA Graph（如 [1,2,4,8,16,...]）
- page_size: KV cache 的页大小
- memory_ratio: 占用显存比例（0.9 表示 90%）
- distributed_timeout: 分布式初始化超时
- use_dummy_weight: 是否用随机权重（测试用）
- use_pynccl: 是否启用 pynccl 加速通信
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：EngineConfig ███
# ════════════════════════════════════════════════════════════════════
#
# 这是一个 frozen=True 的 dataclass——启动后不可变，到处只读传递。
# 每个字段都有合理默认值，调用方只需要传 model_path / tp_info / dtype。
#
# 例子: 在 H200 上用 bf16 跑 Qwen2-7B 单卡：
#   EngineConfig(
#       model_path="/models/qwen2-7b",
#       tp_info=DistributedInfo(rank=0, size=1),
#       dtype=torch.bfloat16,
#   )
#
# 然后这套配置会传给：
#   - Engine: 用全部字段
#   - SchedulerConfig（子类）: 加上 max_extend_tokens / cache_type / zmq 地址等
# ════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class EngineConfig:
    """【类名】EngineConfig - 引擎启动的全部参数。"""

    # 必填参数
    model_path: str                  # 模型权重目录（HuggingFace 风格）
    tp_info: DistributedInfo         # 张量并行：本进程是第几号 rank、总共几个
    dtype: torch.dtype               # 推理精度

    # 并发与资源
    max_running_req: int = 256       # 最大并发请求数（决定 page_table 行数）

    # 后端选择
    attention_backend: str = "auto"  # 注意力 kernel: "fa", "fi", "trtllm", "auto"
    moe_backend: str = "auto"        # MoE 后端: "fused", "auto"

    # CUDA Graph 配置
    # cuda_graph_bs: 显式指定要捕获的 batch size 列表（如 [1,2,4,8,16,32]）；
    #                None 时自动选择。
    cuda_graph_bs: List[int] | None = None
    # cuda_graph_max_bs: 仅指定最大值，自动生成等差序列。
    cuda_graph_max_bs: int | None = None

    # KV Cache
    page_size: int = 1               # 每页几个 token 的 KV
    memory_ratio: float = 0.9        # 启动后用多少比例的剩余显存做 KV cache

    # 分布式
    distributed_timeout: float = 60.0  # 多 rank 初始化的超时（秒）

    # 调试 / 测试
    use_dummy_weight: bool = False     # True = 用随机权重（跳过加载，加快测试）
    use_pynccl: bool = True            # 用 pynccl 自定义 all-reduce

    # 序列长度上限
    # 默认从模型 config（rotary.max_position）读取；可手动覆盖
    max_seq_len_override: int | None = None

    # KV pool 页数手动覆盖（一般留 None 让引擎自己根据显存算）
    num_page_override: int | None = None  # if not None, will override the number of pages

    @cached_property
    def hf_config(self):
        """加载 HuggingFace 的 config.json（缓存避免重复读盘）。"""
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        """从 hf_config 构造 mini-sglang 内部的 ModelConfig。"""
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        """模型支持的最大上下文长度（用户 override > 模型默认）。"""
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        """单次前向最多处理多少 token——基类里等同于 max_seq_len（被 SchedulerConfig 重写）。"""
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        """torch.distributed 初始化用的 master 地址。"""
        return "tcp://127.0.0.1:2333"
