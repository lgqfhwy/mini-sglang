"""
========================================================================
文件名: moe/base.py
所属模块: MoE - 后端抽象接口
========================================================================

BaseMoeBackend：所有 MoE 后端实现的统一接口。
唯一方法 forward 接受：
- hidden_states:           [num_tokens, hidden_dim] 当前层输入
- w1:                       [num_experts, 2*intermediate, hidden] 上+门控投影
- w2:                       [num_experts, hidden, intermediate] 下投影
- gating_output:            [num_tokens, num_experts] router 打分
- topk:                     每个 token 选几个专家
- renormalize:              选出 top-k 后是否重新归一化权重
- activation:               "silu" 或 "gelu"
- apply_router_weight_on_input: router 权重作用于输入还是输出
"""

from abc import ABC, abstractmethod

import torch


class BaseMoeBackend(ABC):
    """MoE 计算后端的统一接口。"""

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor: ...
