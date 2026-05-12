"""
========================================================================
文件名: utils/torch_utils.py
所属模块: Utils - PyTorch 相关辅助
========================================================================

提供两个小工具：
- torch_dtype: 上下文管理器，临时改变 torch.set_default_dtype。
- nvtx_annotate: 装饰器，给 GPU profile（nsys）打 NVTX 标签，方便看
                每段代码的耗时分布。
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@contextmanager
def torch_dtype(dtype: torch.dtype):
    """
    临时把 torch 默认 dtype 改成指定值（with 块结束自动恢复）。
    主要用在 Engine 初始化时——用 meta device + bfloat16 默认 dtype 创建
    模型骨架，离开 with 块后恢复全局默认 dtype。
    """
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    """
    给方法加 NVTX 标签的装饰器——方便用 nsight systems 看每段 kernel 的耗时。

    例: @nvtx_annotate("Attn_L{}", layer_id_field="layer_id")
        会被注解为 "Attn_L0", "Attn_L1", ...
    """
    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            # 如果指定了 layer_id_field，把 self.layer_id 填进 {} 占位
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            with nvtx.range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator
