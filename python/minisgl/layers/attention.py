"""
========================================================================
文件名: layers/attention.py
所属模块: Layers - 注意力子层（Transformer 中的"Self-Attention"）
========================================================================

【这个文件做什么】
封装 Transformer 中"注意力子层"——把已经通过 Q/K/V projection 算出的
qkv 张量切开 → 应用 RMS norm（如果配置了）→ 应用 RoPE → 调 attention
后端做注意力 → 返回输出。

【为什么是 StateLessOP】
注意力子层本身没有权重——所有的权重都在 Q/K/V/O 的线性投影里（那些是
Linear 层）。本层只做"算"，不存"参"。

【调用链】
   Transformer Layer.forward()
     ↓ Linear (q_proj, k_proj, v_proj) 一起算成一个大 qkv
   ★ AttentionLayer.forward(qkv) ★
     ↓
   1. split 成 q, k, v
   2. (可选) q_norm / k_norm 做 RMS norm（QK-Norm 模型如 Qwen3 用）
   3. rotary (RoPE) 应用位置编码到 q, k
   4. ctx.attn_backend.forward 真的做注意力（FA/FI/TRT-LLM）
   5. 返回输出 → 后面会接 o_proj Linear

【RoPE (Rotary Position Embedding)】
一种位置编码：把 q 和 k 中每对维度按某个频率旋转，使得 q·k 自然包含
相对位置信息。比绝对位置编码效果好得多。

【MHA / GQA】
num_qo_heads >= num_kv_heads，GQA 模型 num_kv_heads < num_qo_heads。
TP 切分时 num_qo_heads 必须整除 TP_size；num_kv_heads 不一定能整除，
allow_replicate=True 表示不能整除时复制到所有卡。
========================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import StateLessOP
from .rotary import get_rope

if TYPE_CHECKING:
    from minisgl.layers import RMSNorm
    from minisgl.models import RotaryConfig


class AttentionLayer(StateLessOP):
    """
    【类名】AttentionLayer - 注意力子层
    【一句话描述】对输入的 qkv（已经 Q/K/V projection 过）做注意力计算。
    """
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
    ):
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
        )
        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return o.view(-1, self.qo_attn_dim)
