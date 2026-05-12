"""
========================================================================
文件名: layers/moe.py
所属模块: Layers - Mixture-of-Experts 层
========================================================================

【这个文件做什么】
实现 MoE（混合专家）的 forward——这是 Qwen3-MoE、Mixtral 等模型的核心层。

【MoE 是什么】
传统 FFN 对每个 token 都跑同一个大矩阵；MoE 把这个矩阵切成 N 个"专家"，
每个 token 通过一个"门控网络"（router）选 top-k 个专家计算。
好处：参数量大但激活量小——例如 8 个专家、每 token 选 2 个，参数是 8 倍
但实际计算量只有 2 倍。

【主要步骤】
1. router 算出每个 token 选哪 top-k 专家、各专家的权重；
2. 把 token 按所选专家分组；
3. 对每组 token 分别送到对应专家的 FFN；
4. 把各专家的输出按权重加权求和回到原 token 位置。

【MoE 后端】
具体的"按专家分组 + 并行算 FFN"由 moe/fused.py 的 fused MoE kernel 实现，
本层只是个调度封装。
========================================================================
"""

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_even

from .base import BaseOP


class MoELayer(BaseOP):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        self.tp_size = tp_size = tp_info.size
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        intermediate_size_per_partition = div_even(intermediate_size, tp_size)
        self.gate_up_proj = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
        )
        self.down_proj = torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
        )

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        ctx = get_global_ctx()
        final_hidden_states = ctx.moe_backend.forward(
            hidden_states=hidden_states,
            w1=self.gate_up_proj,
            w2=self.down_proj,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )
        if self.tp_size > 1:
            final_hidden_states = self._comm.all_reduce(final_hidden_states)
        return final_hidden_states
