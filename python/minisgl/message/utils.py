"""
========================================================================
文件名: message/utils.py
所属模块: 跨进程消息的通用序列化 / 反序列化
========================================================================

【这个文件是做什么的 - 一句话总结】
提供两对函数把 Python 对象（含 dataclass / torch.Tensor / 嵌套结构）
变成 JSON-like dict（可被 msgpack/json 编码），以及反过来还原。
所有 message/* 的 encoder/decoder 都依赖这里。

【为什么需要这个文件】
ZMQ 传递的只能是字节。要把 Python 对象（含 dataclass、嵌套、tensor）
变成字节，需要：
  1. 把对象变成 JSON-like dict（本文件的工作）；
  2. 再用 msgpack/json 编码成字节（ZMQ queue 内部完成）。
反过来同理。

【关键挑战】
- torch.Tensor 不是原生 JSON 类型 → 特殊处理（按 numpy 字节存）
- 嵌套 dataclass → 递归处理
- 反序列化时要根据 "__type__" 找到正确的类 → 通过 cls_map 查表

【序列化格式】
普通 dataclass 序列化结果举例：
  UserMsg(uid=1, input_ids=tensor([2,3]), sampling_params=SamplingParams(...))
   ↓
  {
    "__type__": "UserMsg",
    "uid": 1,
    "input_ids": {"__type__": "Tensor", "buffer": b"...", "dtype": "torch.int32"},
    "sampling_params": {"__type__": "SamplingParams", "temperature": 0.0, ...}
  }
"""

from __future__ import annotations

from typing import Any, Dict, Type

import numpy as np
import torch


def _serialize_any(value: Any) -> Any:
    """
    【功能】递归序列化任意 Python 值。

    【分类处理】
    - dict: 对 value 递归
    - list/tuple: 对每个元素递归（保持原类型）
    - 原子类型（int/float/str/None/bool/bytes）: 直接返回
    - 其它（dataclass、Tensor 等）: 委托给 serialize_type
    """
    if isinstance(value, dict):
        return {k: _serialize_any(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return type(value)(_serialize_any(v) for v in value)
    elif isinstance(value, (int, float, str, type(None), bool, bytes)):
        return value
    else:
        return serialize_type(value)


def serialize_type(self) -> Dict:
    """
    【功能】把一个对象（dataclass 或 Tensor）序列化成 dict。

    【特殊处理】
    - torch.Tensor: 只支持 1-D；序列化成 {__type__: "Tensor", buffer: bytes, dtype: str}
    - dataclass: 用 __dict__ 拿到所有字段，递归序列化每个值

    【为什么 buffer 用 numpy 字节】
    torch.Tensor 没有可序列化的 wire-format；转 numpy 后 tobytes()
    拿到原始字节流，反序列化时再 frombuffer 还原。
    """
    # find all member variables
    serialized = {}

    if isinstance(self, torch.Tensor):
        # 张量目前只支持 1-D（message 模块里用到的 input_ids 等都是 1-D）
        assert self.dim() == 1, "we can only serialize 1D tensor for now"
        serialized["__type__"] = "Tensor"
        # 转 numpy 再 tobytes 拿原始字节流
        serialized["buffer"] = self.numpy().tobytes()
        # dtype 存字符串形式（如 "torch.int32"），反序列化时解析
        serialized["dtype"] = str(self.dtype)
        return serialized

    # 普通 dataclass：__dict__ 包含所有实例字段
    # normal type
    serialized["__type__"] = self.__class__.__name__
    for k, v in self.__dict__.items():
        serialized[k] = _serialize_any(v)
    return serialized


def _deserialize_any(cls_map: Dict[str, Type], data: Any) -> Any:
    """
    【功能】递归反序列化任意值。

    【参数】cls_map: 类型名 → 类 的映射，通常是调用方文件的 globals()

    【分类处理】
    - dict 且含 "__type__": 委托给 deserialize_type（构造一个具体对象）
    - dict 不含 "__type__": 普通 dict，递归 value
    - list/tuple: 每元素递归
    - 原子类型: 直接返回
    """
    if isinstance(data, dict):
        if "__type__" in data:
            return deserialize_type(cls_map, data)
        else:
            return {k: _deserialize_any(cls_map, v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    elif isinstance(data, (int, float, str, type(None), bool, bytes)):
        return data
    else:
        raise ValueError(f"Cannot deserialize type {type(data)}")


def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    """
    【功能】根据 dict 里的 "__type__"，从 cls_map 查到对应类并实例化。

    【参数】
    - cls_map: 调用方传入的类型查找表（一般是文件级 globals()）
    - data: 待反序列化的 dict（必须含 "__type__"）

    【特殊处理 Tensor】
    - "__type__" == "Tensor"：从 buffer + dtype 还原 torch.Tensor。
    - 注意 dtype 字符串如 "torch.int32" 要去掉 "torch." 前缀去 numpy 找。

    【普通对象】
    - 用 cls_map[type_name] 拿到类；
    - 递归 _deserialize_any 处理每个字段；
    - 用 kwargs 调用 cls(**kwargs) 构造实例。

    【注意 numpy buffer 的 .copy()】
    np.frombuffer 创建的数组是 read-only 视图，torch.from_numpy 会要求
    可写——所以 .copy() 出独立可写副本。
    """
    type_name = data["__type__"]
    # we can only serialize 1D tensor for now
    if type_name == "Tensor":
        buffer = data["buffer"]
        # 把 "torch.int32" 去掉前缀变成 numpy 的 "int32"
        dtype_str = data["dtype"].replace("torch.", "")
        np_dtype = getattr(np, dtype_str)
        assert isinstance(buffer, bytes)
        np_tensor = np.frombuffer(buffer, dtype=np_dtype)
        # .copy() 让数组可写，torch.from_numpy 不接受 read-only
        return torch.from_numpy(np_tensor.copy())

    # 普通 dataclass：查类、递归处理字段、调用构造函数
    cls = cls_map[type_name]
    kwargs = {}
    for k, v in data.items():
        if k == "__type__":
            continue
        kwargs[k] = _deserialize_any(cls_map, v)
    return cls(**kwargs)
