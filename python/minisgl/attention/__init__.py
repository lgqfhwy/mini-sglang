"""
========================================================================
文件名: attention/__init__.py
所属模块: 注意力后端模块入口 + 工厂函数
========================================================================

【这个模块负责什么】
mini-sglang 支持多种注意力 kernel，本文件用 Registry 模式把它们注册起来：
  - "fa":     FlashAttention（前向、Triton/CUDA 实现）
  - "fi":     FlashInfer（PagedAttention 友好、decode 快）
  - "trtllm": TensorRT-LLM 后端（Blackwell GPU 最快）

【两种使用方式】
- 单后端: --attn fi          全程用 FlashInfer
- 混合后端: --attn fa,fi      prefill 用 FA，decode 用 FI
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata, HybridBackend

if TYPE_CHECKING:
    from minisgl.models import ModelConfig

logger = init_logger(__name__)


class BackendCreator(Protocol):
    """后端工厂函数协议：传入 ModelConfig，返回一个 BaseAttnBackend 实例。"""
    def __call__(self, config: ModelConfig) -> BaseAttnBackend: ...


# 注册表：字符串名字 → 工厂函数
SUPPORTED_ATTENTION_BACKENDS = Registry[BackendCreator]("Attention Backend")


@SUPPORTED_ATTENTION_BACKENDS.register("trtllm")
def create_trtllm_backend(config: ModelConfig):
    """TensorRT-LLM 后端（要求 SM100 / Blackwell）。"""
    from .trtllm import TensorRTLLMBackend

    return TensorRTLLMBackend(config)


@SUPPORTED_ATTENTION_BACKENDS.register("fi")
def create_fi_backend(config: ModelConfig):
    """FlashInfer 后端（PagedAttention 友好）。"""
    from .fi import FlashInferBackend

    return FlashInferBackend(config)


@SUPPORTED_ATTENTION_BACKENDS.register("fa")
def create_fa_backend(config: ModelConfig):
    """FlashAttention 后端（变长 prefill 效率高）。"""
    from .fa import FlashAttentionBackend

    return FlashAttentionBackend(config)


def validate_attn_backend(backend: str, allow_auto: bool = True):
    """
    【功能】校验 --attn 参数是否合法。
    【支持格式】
    - "auto"
    - "fi"（单后端）
    - "fa,fi"（混合：prefill,decode）
    """
    if backend != "auto":
        required_backends = backend.split(",") if "," in backend else [backend]
        SUPPORTED_ATTENTION_BACKENDS.assert_supported(required_backends)
    else:
        assert allow_auto, "auto is not allowed here"
    return backend


def create_attention_backend(
    backend: str,
    config: ModelConfig,
) -> BaseAttnBackend:
    """
    【功能】根据字符串名字创建后端实例。
    【混合后端】"fa,fi" → 创建两个后端、用 HybridBackend 包起来。
    """
    validate_attn_backend(backend, allow_auto=False)
    if "," in backend:
        assert backend.count(",") == 1, "Only one comma is allowed in hybrid backend"
        p_backend, d_backend = backend.split(",", 1)
        if p_backend != d_backend:
            logger.info(f"Using hybrid attention backend: prefill={p_backend}, decode={d_backend}")
            # 递归创建两个后端实例
            p_backend = create_attention_backend(p_backend, config)
            d_backend = create_attention_backend(d_backend, config)
            return HybridBackend(p_backend, d_backend)
        backend = p_backend  # both are the same, fall through to single backend
        logger.warning(f"P/D attention backends are the same: {backend}, using single backend.")

    return SUPPORTED_ATTENTION_BACKENDS[backend](config)


__all__ = [
    "BaseAttnMetadata",
    "BaseAttnBackend",
    "create_attention_backend",
    "SUPPORTED_ATTENTION_BACKENDS",
    "validate_attn_backend",
]
