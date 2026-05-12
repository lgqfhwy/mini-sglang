"""
========================================================================
文件名: moe/__init__.py
所属模块: MoE 后端注册与工厂
========================================================================

【模块包含什么】
- base.py:   BaseMoeBackend 抽象接口
- fused.py:  FusedMoe 实现（用 vLLM 的 fused MoE kernel）

【目前支持的后端】
"fused"——把"router → 分组 → 多专家 FFN → 聚合"融合成少量 CUDA kernel。

【MoE 后端做什么】
不直接被模型代码调用——而是被 layers/moe.py 的 MoELayer 调用：
   forward(hidden, router_logits, expert_weights, ...)
就能返回 MoE 的输出。
"""

from __future__ import annotations

from typing import Protocol

from minisgl.utils import Registry, init_logger

from .base import BaseMoeBackend

logger = init_logger(__name__)


class MoeBackendCreator(Protocol):
    """MoE 后端工厂协议：无参，返回一个 BaseMoeBackend 实例。"""
    def __call__(self) -> BaseMoeBackend: ...


SUPPORTED_MOE_BACKENDS = Registry[MoeBackendCreator]("MoE Backend")


@SUPPORTED_MOE_BACKENDS.register("fused")
def create_fused_moe_backend():
    """工厂：fused MoE 后端。"""
    from .fused import FusedMoe

    return FusedMoe()


def create_moe_backend(backend: str) -> BaseMoeBackend:
    """按名字创建 MoE 后端。"""
    return SUPPORTED_MOE_BACKENDS[backend]()


__all__ = [
    "BaseMoeBackend",
    "create_moe_backend",
    "SUPPORTED_MOE_BACKENDS",
]
