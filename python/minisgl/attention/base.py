"""
========================================================================
文件名: attention/base.py
所属模块: 注意力（Attention）后端的抽象接口
========================================================================

【这个文件是做什么的 - 一句话总结】
定义所有"注意力后端"必须实现的接口（基类）。LLM 推理里 attention 是
最核心也最耗时的操作，有多种实现（FlashAttention / FlashInfer / TRT-LLM），
本文件用抽象基类让上层代码不关心具体用哪种。

【为什么需要这个文件】
不同的注意力 kernel 各有优缺点：
  - FlashAttention 2: prefill 强，decode 一般
  - FlashInfer: decode 强，对 paged KV 支持好
  - TRT-LLM (Blackwell): 旗舰卡上最快
我们想"prefill 用一种，decode 用另一种"——所以提供 HybridBackend
把两个后端组合起来按 batch 阶段自动切换。

【核心概念速览】

- attention metadata:
    注意力 kernel 不只是 Q/K/V 张量，还需要一堆元数据：
    - 每个请求的 seq_len（已有多长）
    - 每个 token 在 KV pool 里的 slot 编号（page_table 行）
    - 累积偏移 (cu_seqlens) 以便 kernel 寻址变长序列
    这些都打包在 BaseAttnMetadata 里。

- Hybrid Backend:
    prefill 和 decode 用不同后端。本文件提供 HybridBackend 把两个
    后端"封装成一个"，对外暴露统一接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch
    from minisgl.core import Batch


# ════════════════════════════════════════════════════════════════════
# BaseAttnMetadata - 注意力后端的元数据基类
# ────────────────────────────────────────────────────────────────────
# 每种后端有自己的 metadata 子类（FA / FI / TRT-LLM 各自需要的字段不同），
# 但都要支持 get_last_indices——告诉采样器"每个请求的最后一个 token 在
# 拼接后的 logits 张量里的位置"（prefill 时需要，每个请求只取最后一个
# token 的 logits 用来采样下一个 token）。
# ════════════════════════════════════════════════════════════════════
@dataclass
class BaseAttnMetadata(ABC):
    """注意力后端元数据基类。"""

    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor:
        """
        【功能】返回每个请求"最后一个 token 的全局位置索引"。
        【用途】prefill 阶段：每个请求贡献多个 token 到 batch，但采样
                只用每个请求的最后一个 token 的 logits。
        【参数】bs: 真实 batch size（不含 padding）
        【返回】[bs] 整数张量，每元素是该请求最后 token 在拼接 logits 里的下标
        """
        ...


# ════════════════════════════════════════════════════════════════════
# BaseAttnBackend - 注意力后端基类
# ────────────────────────────────────────────────────────────────────
# 5 个抽象方法：
#   - forward: 实际跑注意力计算（Q,K,V → output）
#   - prepare_metadata: 在前向前构造元数据
#   - init_capture_graph: CUDA Graph 录制前的初始化
#   - prepare_for_capture: 每个 graph 录制前的准备
#   - prepare_for_replay: 每次 graph 回放前的准备
# ════════════════════════════════════════════════════════════════════
class BaseAttnBackend(ABC):
    """注意力后端的统一接口。"""

    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """
        【功能】跑一层的注意力计算。
        【参数】
        - q: [num_tokens, num_q_heads, head_dim]
        - k, v: [num_tokens, num_kv_heads, head_dim]（本层新算的 K/V）
        - layer_id: 第几层
        - batch: 当前 Batch（含 metadata、page_table 等所有需要的信息）
        【返回】注意力输出，形状 [num_tokens, hidden_size]
        """
        ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None:
        """在跑前向前调用，构造本批次需要的注意力元数据，写入 batch.attn_metadata。"""
        ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """CUDA Graph 录制前的一次性初始化（预分配元数据张量）。"""
        ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None:
        """每个 graph 录制前的准备（设置元数据指向预分配的固定张量）。"""
        ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None:
        """每次 graph 回放前的准备（更新元数据张量的内容）。"""
        ...


# ════════════════════════════════════════════════════════════════════
# HybridBackend - 组合两个后端（prefill 一种、decode 一种）
# ────────────────────────────────────────────────────────────────────
# 例: HybridBackend(FlashAttention, FlashInfer)
#   - prefill 阶段走 FlashAttention（变长序列效率高）
#   - decode 阶段走 FlashInfer（单 token 多请求效率高）
#
# 实现：forward / prepare_metadata 根据 batch.is_prefill 路由到对应后端。
# CUDA Graph 相关只走 decode_backend（prefill 不用 graph）。
# ════════════════════════════════════════════════════════════════════
class HybridBackend(BaseAttnBackend):
    """组合两个后端：prefill 用 prefill_backend，decode 用 decode_backend。"""

    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """根据 batch 阶段路由到对应后端。"""
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch: Batch) -> None:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        # CUDA Graph 只用于 decode（prefill 形状变化大不录）
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_replay(batch)
