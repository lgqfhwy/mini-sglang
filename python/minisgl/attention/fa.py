"""
========================================================================
文件名: attention/fa.py
所属模块: 注意力 - FlashAttention 后端
========================================================================

【这个文件是做什么的 - 一句话总结】
封装 FlashAttention（FA2/FA3）作为 mini-sglang 的注意力后端。
FlashAttention 是 Tri Dao 2022 论文里提出的"显存高效注意力" kernel，
通过分块计算 + 在线 softmax 把 O(seq_len^2) 的显存降到 O(seq_len)，
同时也比 PyTorch 原生快很多。本文件主要工作是把"Batch 元数据"翻译成
FA kernel 能直接吃的张量格式。

【为什么用 FlashAttention 做后端】
- 变长序列友好：prefill 时每个请求长度都不同，FA 用 cu_seqlens 直接处理
- 性能极佳：H100 上几乎打满 BF16 算力
- 支持 paged KV：FA3 直接接受 page_table 输入，无需 gather

【核心数据结构】
- FACaptureData: CUDA Graph 录制用的固定张量（继承 BaseCaptureData）
- FAMetadata: 每个 batch 算前向时的元数据（cu_seqlens_k/q、page_table 等）
- FlashAttentionBackend: 后端主类，实现 prepare_metadata / forward 等

【cu_seqlens 解释】
"累积 seqlen 偏移"——对变长序列必备。
例: 3 个请求长度 [5, 3, 7]
    cu_seqlens = [0, 5, 8, 15]
    含义: 第 i 个请求的 token 在拼接张量里的 slice [cu[i], cu[i+1])

【SM100 / Blackwell 特别处理】
某些 kernel 调用要根据 SM 版本切换 API（is_sm100_supported）。
========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.utils import is_sm100_supported

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class FACaptureData(BaseCaptureData):
    pass


@dataclass
class FAMetadata(BaseAttnMetadata):
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int

    page_table: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class FlashAttentionBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        ctx = get_global_ctx()
        self.config = config
        self.kvcache = ctx.kv_cache
        self.page_size = ctx.page_size
        self.capture: FACaptureData | None = None
        self.max_graph_bs = 0
        self.capture_bs: List[int] = []
        self.scale = config.head_dim**-0.5
        self.version = 4 if is_sm100_supported() else 3

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        return _fa_sgl_impl(
            q=q,
            k_cache=self.kvcache.k_cache(layer_id),
            v_cache=self.kvcache.v_cache(layer_id),
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            softmax_scale=self.scale,
            version=self.version,
        )

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs

        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_k = max(seqlens_k)
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.kvcache.device
        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS)
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        if max_seqlen_q == 1:
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q = cu_seqlens_k
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)

        page_table = get_global_ctx().page_table
        new_page_table = torch.stack(  # NOTE: global page table treat page_size = 1, we need slice
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )
        if self.page_size > 1:
            new_page_table.div_(self.page_size, rounding_mode="floor")
        batch.attn_metadata = FAMetadata(
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FACaptureData.create(max_bs, max_seq_len // self.page_size, self.kvcache.device)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        metadata = FAMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )
        batch.attn_metadata = metadata

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FAMetadata)
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)
        table_len = metadata.page_table.size(1)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)


def _fa_sgl_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    version: int,
    sm_margin: int = 0,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    softcap: float = 0.0,  # 0.0 means deactivated
    num_splits: int = 0,  # Can be tuned for speed
    pack_gqa: bool | None = None,  # Can be tuned for speed
    causal: bool = True,
) -> torch.Tensor:
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
    except ImportError as e:
        raise ImportError(
            "sgl_kernel.flash_attn is not found. Please install it with `pip install sgl-kernel`.\n"
            "If you're sure it's correctly installed, try `apt update && apt install libnuma1`."
        ) from e

    return flash_attn_with_kvcache(  # type: ignore
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale,
        sm_margin=sm_margin,
        window_size=window_size,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        causal=causal,
        ver=version,  # TODO: support FA4 on blackwell
    )
