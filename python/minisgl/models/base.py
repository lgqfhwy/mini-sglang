"""
========================================================================
文件名: models/base.py
所属模块: Models - 模型基类
========================================================================

BaseLLMModel: 所有具体模型（LLaMA/Qwen2/Qwen3/Mistral 等）的共同父类。
唯一抽象方法 forward()——返回最后一层的 logits。
为什么 forward 没有参数？因为输入信息（input_ids、positions、batch）
都通过 minisgl.core.get_global_ctx() 拿到，避免到处传参。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from minisgl.layers import BaseOP

if TYPE_CHECKING:
    import torch


class BaseLLMModel(ABC, BaseOP):
    """LLM 模型基类——子类必须实现 forward 返回 logits。"""

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """
        【返回】最后一层的 logits 张量，形状 [num_tokens, vocab_size]
        【输入】从 get_global_ctx().batch 拿（input_ids / positions / attn_metadata）
        """
        ...
