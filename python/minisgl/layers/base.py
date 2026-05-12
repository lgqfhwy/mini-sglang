"""
========================================================================
文件名: layers/base.py
所属模块: Layers - 自实现的轻量"模块基类"（mini 版的 nn.Module）
========================================================================

【这个文件是做什么的 - 一句话总结】
mini-sglang 没用 torch.nn.Module（自带很多 autograd / hook 开销），
而是自实现了一套"推理专用"的最小模块基类 BaseOP，只提供两个能力：
forward() 和 state_dict()/load_state_dict()。

【为什么不用 nn.Module】
nn.Module 设计为训练 + 推理通用——自带 parameters、autograd hooks、
buffers 注册等机制。推理时不需要这些，反而拖慢启动和增加内存。
mini-sglang 把模块层数大幅简化，只保留权重存取的能力。

【三个核心类】
- BaseOP:     最基础的模块基类（要实现 forward）
- StateLessOP: 没有可训练权重的模块（norm 之类）——load_state_dict 是 no-op
- OPList:     列表形式的容器（类似 nn.ModuleList），按下标存权重

【state_dict 的层级前缀】
和 nn.Module 一样，按属性名拼前缀:
  Model.encoder.layer0.attn.q_proj.weight
  → state_dict key = "encoder.layer0.attn.q_proj.weight"
========================================================================
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Generic, List, TypeAlias, TypeVar

import torch

# 类型别名：state_dict 就是 {名字: tensor} 字典
_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]


def _concat_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


class BaseOP:
    """
    【类名】BaseOP - 模块基类（mini 版的 nn.Module）

    【两个核心能力】
    - forward: 子类必须实现的前向计算
    - state_dict / load_state_dict: 通过 __dict__ 反射自动序列化所有
      torch.Tensor 字段和嵌套的 BaseOP 字段
    """

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}

        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                result[_concat_prefix(prefix, name)] = param
            elif isinstance(param, BaseOP):
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)

        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                item = state_dict.pop(_concat_prefix(prefix, name))
                assert isinstance(item, torch.Tensor)
                assert param.shape == item.shape and param.dtype == item.dtype
                setattr(self, name, item)
            elif isinstance(param, BaseOP):
                param.load_state_dict(
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                )

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


class StateLessOP(BaseOP):
    def __init__(self):
        super().__init__()

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        return result if result is not None else {}


T = TypeVar("T", bound=BaseOP)


class OPList(BaseOP, Generic[T]):
    def __init__(self, ops: List[T]):
        super().__init__()
        self.op_list = ops

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}
        for i, op in enumerate(self.op_list):
            op.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)
        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for i, op in enumerate(self.op_list):
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True)

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
