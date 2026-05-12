"""
========================================================================
文件名: utils/hf.py
所属模块: Utils - HuggingFace 集成（tokenizer / config / 权重下载）
========================================================================

【这个文件做什么】
封装 HuggingFace 的三个核心入口：
- load_tokenizer:       加载 tokenizer（处理 Mistral 的 chat_template 特殊情况）
- cached_load_hf_config: 加载并缓存 config.json
- download_hf_weight:   从 HF Hub 下载模型权重（只下 safetensors）

【为什么 cached_load_hf_config 还要 type(config)(...) 复制一份】
@functools.cache 会缓存 config 对象——如果调用者修改了它，下次别人拿到
的也是改后的版本。所以每次返回一个新副本，保证 immutability。
"""

import functools
import json
import os
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase


class DisabledTqdm(tqdm):
    """关闭进度条显示的 tqdm 子类——下载时不打扰 stdout。"""

    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    """
    加载 tokenizer。
    特殊处理: 部分 Mistral 模型把 chat_template 单独存在 chat_template.json，
    AutoTokenizer 不会自动读取，这里手动加载。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Some Mistral models store chat_template in a separate JSON file
    if not getattr(tokenizer, "chat_template", None):
        try:
            path = hf_hub_download(repo_id=model_path, filename="chat_template.json")
            with open(path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = json.load(f)["chat_template"]
        except Exception:
            pass
    return tokenizer


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    """内部函数：加载 HF config 并缓存（避免重复读盘）。"""
    return AutoConfig.from_pretrained(model_path)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    """
    加载 HF config。返回的是缓存对象的"浅拷贝"——调用方可以放心修改，
    不会影响缓存。
    """
    config = _load_hf_config(model_path)
    return type(config)(**config.to_dict())


def download_hf_weight(model_path: str) -> str:
    """
    如果 model_path 是本地目录就直接返回；否则当成 HF repo id 下载。
    只下 safetensors 文件（忽略 .bin），更安全更快。
    """
    if os.path.isdir(model_path):
        return model_path
    try:
        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],
            tqdm_class=DisabledTqdm,
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )
