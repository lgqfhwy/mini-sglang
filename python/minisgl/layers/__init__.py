"""
========================================================================
文件名: layers/__init__.py
所属模块: Layers - Transformer 各种"组件层"的统一入口
========================================================================

【layers 模块包含什么】
Transformer 模型的所有基础组件:
  - base.py:      模块基类（自实现的 mini nn.Module）
  - linear.py:    线性层（含 TP 切分版本：Row/Col Parallel）
  - attention.py: 注意力子层
  - embedding.py: 词表 embedding + LM head
  - norm.py:      RMSNorm
  - activation.py: silu_and_mul / gelu_and_mul（FFN 用的激活）
  - rotary.py:    RoPE 位置编码
  - moe.py:       Mixture-of-Experts 层

【为什么自己实现这些】
比 torch.nn 简洁很多——只保留推理需要的功能；并且专门为 TP 切分、
KV cache 写入等推理场景定制。
"""

from .activation import gelu_and_mul, silu_and_mul
from .attention import AttentionLayer
from .base import BaseOP, OPList, StateLessOP
from .embedding import ParallelLMHead, VocabParallelEmbedding
from .linear import (
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
)
from .moe import MoELayer
from .norm import RMSNorm, RMSNormFused
from .rotary import get_rope, set_rope_device

__all__ = [
    "silu_and_mul",
    "gelu_and_mul",
    "AttentionLayer",
    "BaseOP",
    "StateLessOP",
    "OPList",
    "VocabParallelEmbedding",
    "ParallelLMHead",
    "LinearColParallelMerged",
    "LinearRowParallel",
    "LinearOProj",
    "LinearQKVMerged",
    "RMSNorm",
    "RMSNormFused",
    "get_rope",
    "set_rope_device",
    "LinearReplicated",
    "MoELayer",
]
