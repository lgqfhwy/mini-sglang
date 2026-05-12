"""
========================================================================
文件名: models/register.py
所属模块: Models - 模型类注册表
========================================================================

【这个文件做什么】
把 HuggingFace config.json 里 architectures 字段（如 "Qwen2ForCausalLM"）
映射到 mini-sglang 的具体模型实现类。

【延迟 import】
用 importlib.import_module 在 get_model_class 被调用时才真的加载模块——
避免 import minisgl 时把所有模型代码都拉进来（占内存且慢）。
"""

import importlib

from .config import ModelConfig

# 注册表：HF 模型名 → (模块相对路径, 类名)
_MODEL_REGISTRY = {
    "LlamaForCausalLM": (".llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": (".qwen2", "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM": (".qwen3", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": (".qwen3_moe", "Qwen3MoeForCausalLM"),
    "MistralForCausalLM": (".mistral", "MistralForCausalLM"),
    # Mistral3 多模态走纯文本部分也用同一个类
    "Mistral3ForConditionalGeneration": (".mistral", "MistralForCausalLM"),
}


def get_model_class(model_architecture: str, model_config: ModelConfig):
    """
    【功能】根据 architecture 字符串找到对应模型类并实例化。
    【参数】
    - model_architecture: 如 "Qwen2ForCausalLM"
    - model_config: 实例化时传给类的构造参数
    """
    if model_architecture not in _MODEL_REGISTRY:
        raise ValueError(f"Model architecture {model_architecture} not supported")
    module_path, class_name = _MODEL_REGISTRY[model_architecture]
    # 延迟 import，加快启动
    module = importlib.import_module(module_path, package=__package__)
    model_cls = getattr(module, class_name)
    return model_cls(model_config)


__all__ = ["get_model_class"]
