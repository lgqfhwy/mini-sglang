"""
========================================================================
文件名: engine/engine.py
所属模块: 引擎主类 - 真正在 GPU 上跑模型前向的"发动机"
========================================================================

【这个文件是做什么的 - 一句话总结】
这个文件就是 LLM 推理的"动力心脏"——它在 GPU 上加载模型权重、预分配
KV cache 显存、初始化注意力 / MoE 后端、CUDA Graph、采样器，然后对外
暴露 forward_batch() 给调度器调用。

【为什么需要这个文件】
调度器把"什么时候跑什么"管好了，但"怎么跑"是另一个学问：
  - 怎么把模型权重切分到多个 GPU（TP）？
  - 怎么算出当前还能给 KV cache 用多少显存？
  - 怎么选最优的注意力 kernel？
  - 怎么把所有这些组件协调起来一次前向？
Engine 把这些放在一个地方，对调度器只暴露 forward_batch 这个简单接口。

【这个文件在整个推理流程中的位置】
   Scheduler.__init__
     ↓ 创建一个 Engine 实例
   ★ Engine 加载模型、KV pool、attention/MoE backend、CUDA Graph ★
     ↓
   每个调度循环：
   Scheduler._forward
     ↓ 调用 engine.forward_batch(batch, sample_args)
   ★ Engine 在 GPU 上跑前向 + 采样，返回 ForwardOutput ★
     ↓
   调度器拿到 next_tokens 处理后续

【核心概念速览】

- ForwardOutput:
    一次前向的输出包，含 next_tokens（GPU + CPU 两份）+ CUDA Event。
    GPU 拷到 CPU 是异步的，event 用来等拷贝完成（同步点）。

- TP rank 间通信（all-reduce）:
    多卡 TP 时，每层做完局部计算后需要 all-reduce 把分片结果汇总。
    本类初始化 NCCL/pynccl 通信组。

- CUDA Graph:
    一种把"一段 GPU 操作"录下来反复重放的机制，省 kernel 启动开销。
    具体见 graph.py。

【关键设计决策】

1. 启动时一次性预分配所有大显存：
   模型权重 + KV pool + page_table + CUDA graph buffer 全在 __init__
   阶段确定，运行时不再 alloc 大块显存——避免碎片和 OOM。

2. 用 "meta" device 先建空模型再加载权重：
   `with torch.device("meta")` 让 nn.Module 不真的分配显存，只建结构；
   load_state_dict 时再 to_empty 到真实 device。这样可以避免"先在 CPU
   构造完整模型再搬到 GPU"的 2 倍峰值显存。

3. dummy_req 用于 CUDA Graph 占位：
   page_table 多分配 1 行（max_running_req + 1），多分配 1 页给 dummy。
   dummy_req 总指向那一行/那一页，确保录 graph 时读到的是合法数据。
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from minisgl.kvcache import create_kvcache_pool
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.moe import create_moe_backend
from minisgl.utils import div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


# ════════════════════════════════════════════════════════════════════
# ForwardOutput - 一次前向的输出
# ────────────────────────────────────────────────────────────────────
# 字段:
#   next_tokens_gpu: GPU 上的采样结果（int32 张量）→ 写回 token_pool 供下轮用
#   next_tokens_cpu: CPU 上的同样结果（异步拷过来）→ 调度器读它做后处理
#   copy_done_event: CUDA event，标记"D2H 拷贝完成"，调度器等它再读 cpu 张量
# ════════════════════════════════════════════════════════════════════
class ForwardOutput(NamedTuple):
    """一次前向的产出："""
    next_tokens_gpu: torch.Tensor    # GPU 端的下一个 token id
    next_tokens_cpu: torch.Tensor    # 异步拷到 CPU 的同样数据
    copy_done_event: torch.cuda.Event  # 同步点：等它完成才能读 CPU 张量


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：Engine ███
# ════════════════════════════════════════════════════════════════════
#
# 一、初始化阶段做的事情（按顺序）
#   1. 设置 TP 信息和 CUDA 设备；
#   2. 自动调整配置（auto 选 attention/moe backend，page_size 对齐等）；
#   3. 初始化分布式通信（gloo 用于 CPU barrier、nccl/pynccl 用于 GPU all-reduce）；
#   4. 记录初始显存（用于后续算 KV pool 大小）；
#   5. 用 meta device 创建模型骨架，然后加载真实权重；
#   6. 根据剩余显存决定 KV pool 页数；
#   7. 创建 KV pool 和 page_table；
#   8. 创建注意力后端、MoE 后端、采样器；
#   9. 创建 dummy_req 和 GraphRunner（录 CUDA Graph）。
#
# 二、forward_batch 的执行流程
#   1. assert 当前 stream 正确；
#   2. forward_batch context 切换 (设置 ctx.batch)；
#   3. 看能不能用 CUDA Graph：
#      - 能用 → graph_runner.replay()（快路径）
#      - 不能 → model.forward()（普通路径，多用于 prefill）；
#   4. 每个请求 .complete_one()（推进 cached_len / device_len）；
#   5. 采样：sampler.sample(logits, args) → 得到 next_tokens (GPU);
#   6. 异步拷贝到 CPU；
#   7. 记录 CUDA event；
#   8. 返回 ForwardOutput。
#
# ════════════════════════════════════════════════════════════════════
class Engine:
    """【类名】Engine - GPU 推理引擎主类。"""

    def __init__(self, config: EngineConfig):
        """根据 config 初始化整个引擎栈。"""
        # 断言：进入 __init__ 前 CUDA 不能被初始化（否则 set_device 等可能行为异常）
        assert not torch.cuda.is_initialized()
        # 设置全局 TP 信息（rank/size），供其他模块通过 get_tp_info 拿到
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        # 根据 GPU 能力自动调整若干配置（attention/moe backend、page_size 等）
        _adjust_config(config)

        # 绑定当前进程到自己的 GPU
        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)  # 固定种子保证可复现
        # 创建自己的 CUDA stream（与调度器的 metadata stream 分开）
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        # 创建 Context 并注册为全局单例
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        # 初始化分布式通信，得到 CPU 端 process group
        self.tp_cpu_group = self._init_communication(config)
        # 记录初始 free memory（后面算 KV pool 大小要用）
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        # 让 RoPE 等位置编码缓存知道 device
        set_rope_device(self.device)
        # 用 meta device 先建一个"空骨架"模型——不分配显存，只建结构
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        # 加载真实权重（或随机权重）
        self.model.load_state_dict(self._load_weight_state_dict(config))

        # ======================= KV cache initialization ========================
        # 根据剩余显存反推 KV pool 能开多少页
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        # 创建 KV pool（多分配 1 页给 dummy 用）
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )

        # ======================= Page table initialization ========================
        # max_seq_len 取配置和能装得下的 KV 数量两者的最小值
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        # 对齐到 32 的倍数（kernel 友好）
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        # page_table 多分配 1 行给 dummy 请求
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # ======================= Attention & MoE backend initialization ========================
        # 根据 config.attention_backend 字符串创建具体后端实例
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_moe:
            # MoE 模型才需要 MoE 后端
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        # 创建 dummy_req：CUDA Graph 录制时用它补齐 batch
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,   # 用预留的最后一行
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        # 让 dummy 请求的页表行全部指向 dummy 页（num_tokens 是第一个 dummy slot）
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        # 创建 GraphRunner（录制所有档位的 CUDA Graph）
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        """
        【功能】初始化 torch.distributed 通信组。

        【两种模式】
        - 单卡 / use_pynccl=True: 用 gloo 作为通信后端（CPU），
          再额外起 pynccl 做 GPU 上的快速 all-reduce；
        - 多卡 + 不用 pynccl: 直接用 nccl 后端。

        【为什么有 pynccl】
        pynccl 是更精细的 NCCL 封装，能更好处理某些边界情况和性能优化。
        """
        if config.tp_info.size == 1 or config.use_pynccl:
            # gloo: CPU 后端，用于 barrier 和小张量广播
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            # 计算 pynccl 缓冲区上限：一次最大 forward 的隐藏状态大小
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            # 标准 NCCL
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            # 额外起一个 gloo 组做 CPU 端 barrier
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        """
        【功能】返回 state_dict 供 model.load_state_dict 使用。

        【两种模式】
        - use_dummy_weight=True: 用 randn_like 生成随机权重（快速测试用）
        - 否则: 调 minisgl.models.load_weight 从磁盘读真实权重并转 dtype
        """
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            return {k: v.to(self.dtype) for k, v in load_weight(config.model_path, self.device)}

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        """
        【功能】根据"加载模型后剩余显存"反推 KV cache 能开多少页。

        【公式】
          model_memory = old_free_memory - new_free_memory  ← 模型占了多少
          available    = memory_ratio * old_free_memory - model_memory
                         ↑ 用初始的 90% 减去模型占用 = 可给 KV cache 的预算
          cache_per_page = 2 (K + V) × head_dim × local_kv_heads × page_size × dtype_size × num_layers
                         ↑ 每页的字节数
          num_pages = available / cache_per_page

        【为什么是 90% 而不是 100%】
        留 10% 余量给：CUDA Graph buffer、中间激活张量、临时通信缓冲等。

        【allow_replicate=True】
        某些模型 num_kv_heads 不能整除 TP size——这时让所有卡都存全部
        KV heads（复制），而不是切分。
        """
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.dtype.itemsize
            * config.model_config.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """
        Get the min and max free memory across TP ranks.

        【功能】同步获取所有 TP rank 的最小/最大剩余显存。
        【为什么要同步】保证每个 rank 看到"统一的剩余显存"，避免不同 rank
                        各自算出不同的 num_pages 导致后续不一致。
        【实现】all_reduce(min) 同时计算 min 和 max（把 free_memory 和
                -free_memory 各发一遍，对 -x 取 min 等价于对 x 取 max）。
        【失败条件】各 rank 间显存差异 > 2GB 就报错（说明 TP 切分不均）。
        """
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        """
        【功能】跑一次前向 + 采样，返回 ForwardOutput。

        【参数】
        - batch: 已经被 Scheduler._prepare_batch 填好所有字段的 Batch
        - args: 采样器预处理好的 GPU 张量

        【流程】
        1. 进入 ctx.forward_batch 上下文（设置 ctx.batch）；
        2. 决定走 CUDA Graph 快路径还是 model.forward 普通路径；
        3. 每个真实请求 .complete_one() 推进 cached_len/device_len；
        4. 采样得到 next_tokens；
        5. 异步拷贝到 CPU 并记录 event；
        6. 返回 ForwardOutput。

        【为什么 logits[:batch.size]】
        decode 走 CUDA Graph 时 logits 形状是 [padded_size, vocab]，
        但我们只关心前 batch.size 个（真实请求）的输出。
        """
        # 必须在 engine.stream 上调用本方法
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()

        # 每个请求推进一格（cached_len → device_len，device_len += 1）
        for req in batch.reqs:
            req.complete_one()

        # 采样：从 logits 选 next_token；只用真实请求那部分
        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        # 异步 D2H 拷贝（调度器要在 CPU 上判断 EOS 等）
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        """
        【功能】优雅关闭：销毁 CUDA Graph、关闭分布式通信组。
        【顺序很关键】先 destroy_cuda_graphs 再销毁 NCCL，否则程序会挂起。
        """
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _align_up_32(num: int) -> int:
    """向上取整到 32 的倍数（kernel 友好的对齐）。"""
    return (num + 31) // 32 * 32


def _adjust_config(config: EngineConfig):
    """
    【功能】根据 GPU 能力自动调整 EngineConfig 的部分字段。

    【⚠️ 危险操作】config 是 frozen=True 的不可变 dataclass，
    这里用 object.__setattr__ 绕过 frozen 限制——只能在初始化阶段用一次！

    【自动调整】
    - attention_backend == "auto":
       SM100（B200）→ trtllm
       SM90（H100）  → "fa,fi"（FlashAttention + FlashInfer 组合）
       其他          → "fi"（FlashInfer）
    - trtllm 后端要求 page_size ∈ {16, 32, 64}：自动改成 64
    - MoE 模型 + moe_backend == "auto" → "fused"
    """
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        # 绕过 frozen 的"暴力"赋值——只能在 __init__ 阶段用！
        object.__setattr__(config, attr, value)

    if config.attention_backend == "auto":
        backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")
