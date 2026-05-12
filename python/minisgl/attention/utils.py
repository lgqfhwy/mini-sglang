"""
========================================================================
文件名: attention/utils.py
所属模块: 注意力后端的通用工具结构
========================================================================

【这个文件是做什么的】
提供 BaseCaptureData——CUDA Graph 录制时用的"固定地址元数据张量"基类。
每种后端（FA / FI / TRT-LLM）的 CaptureData 都继承它，再加上自己特殊的字段。

【为什么需要】
CUDA Graph 要求每次 replay 时所有 tensor 地址不变。所以注意力 metadata
也要预先 alloc 固定大小的张量，每次 replay 前往里写数据，而不是新建。

【字段含义】
- seq_lens:     每个请求当前的序列长度
- positions:    每个 token 的位置 id（RoPE 用）
- cu_seqlens_k: 累积 K 长度——cu_seqlens_k[i] = 前 i 个请求 K 总数
- cu_seqlens_q: 累积 Q 长度（同上）
- page_table:   决定 KV 寻址的页表（kernel 读 KV 时按它索引）
"""

from dataclasses import dataclass

import torch


@dataclass
class BaseCaptureData:
    """CUDA Graph 录制时的"固定地址"元数据张量基类。"""
    seq_lens: torch.Tensor       # 每个请求的序列长度
    positions: torch.Tensor      # 每个 token 的 position id
    cu_seqlens_k: torch.Tensor   # 累积 K seqlen 偏移
    cu_seqlens_q: torch.Tensor   # 累积 Q seqlen 偏移
    page_table: torch.Tensor     # 页表（KV 寻址用）

    @classmethod
    def create(cls, max_bs: int, max_seq_len: int, device: torch.device, **kwargs):
        """
        【功能】按"最大 batch size + 最大 seq len"预分配所有张量。
        【kwargs】传给子类的额外字段（每种后端有自己的扩展字段）。
        """
        return cls(
            # seq_lens 初值全 1：dummy 请求至少有 1 个 token 的 KV
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            positions=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            # cu_seqlens 初值是 [0, 1, 2, ..., max_bs]——表示每个请求 1 个 token
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),
            **kwargs,
        )
