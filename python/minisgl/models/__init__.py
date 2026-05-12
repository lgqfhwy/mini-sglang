"""
========================================================================
文件名: models/__init__.py
所属模块: Models - LLM 模型实现的统一入口
========================================================================

【这个模块包含什么】
- base.py:     BaseLLMModel 抽象基类
- config.py:   ModelConfig / RotaryConfig（从 HF config 转化而来）
- register.py: 模型类的注册表（按 architecture 名查类）
- weight.py:   从磁盘读 safetensors / bin 权重文件
- llama.py / qwen2.py / qwen3.py / qwen3_moe.py / mistral.py:
              具体模型的实现
- utils.py:   构建 Transformer block 的通用工具

【create_model 的作用】
根据 ModelConfig.architectures[0]（如 "Qwen2ForCausalLM"）从注册表
找到对应的模型类并实例化。
"""

from .base import BaseLLMModel
from .config import ModelConfig, RotaryConfig
from .register import get_model_class
from .weight import load_weight


def create_model(model_config: ModelConfig) -> BaseLLMModel:
    """根据 architectures 字段创建对应的模型实例。"""
    return get_model_class(model_config.architectures[0], model_config)


__all__ = ["create_model", "load_weight", "RotaryConfig"]
