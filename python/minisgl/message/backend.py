"""
========================================================================
文件名: message/backend.py
所属模块: 发给 Scheduler（后端）的消息定义
========================================================================

【这个文件是做什么的 - 一句话总结】
定义所有"从其他进程发给 scheduler 进程"的消息格式：
  - UserMsg:        新用户请求（最常见）
  - AbortBackendMsg: 用户取消请求
  - ExitMsg:        让 scheduler 退出
  - BatchBackendMsg: 把上述消息批量打包（减少 ZMQ 帧开销）

【为什么需要这个文件】
进程间通信必须有明确约定的"信件格式"，否则 scheduler 收到字节流不知
道该当 user message 处理还是 abort 处理。dataclass + encoder/decoder
让两边代码以同一份 schema 收发消息。

【典型流程】
   tokenizer 进程把 (uid, input_ids, sampling_params) 封装成 UserMsg
       → encoder() 序列化成 dict → ZMQ 发字节
   scheduler 进程收到字节
       → decoder() 反序列化回 UserMsg 对象
       → isinstance(msg, UserMsg) → 走 _process_one_msg 路径

【序列化策略】
所有消息都通过 utils.py 里的 serialize_type / deserialize_type
（递归处理 dataclass、tensor、list、dict 等）转成 JSON-like dict，
再交给 ZMQ 的 msgpack 编码层。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


# ════════════════════════════════════════════════════════════════════
# BaseBackendMsg —— 所有"后端消息"的基类
# ────────────────────────────────────────────────────────────────────
# 提供两个静态方法 encoder / decoder。
# - encoder: 实例 → dict（通用序列化）
# - decoder: dict → 实例（基于 globals() 查到对应的子类）
# 注意 decoder 是 @staticmethod 而不是 classmethod：
#   它要根据 dict 里的 "__type__" 找到正确的子类（UserMsg/ExitMsg/...），
#   不能用 cls，所以传入 globals() 作为查找表。
# ════════════════════════════════════════════════════════════════════
@dataclass
class BaseBackendMsg:
    """所有发给后端的消息基类。"""

    def encoder(self) -> Dict:
        """把自己序列化成 dict（递归处理嵌套字段）。"""
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        """
        【功能】根据 dict 里的 "__type__" 字段，从本模块的 globals() 里
                找到对应的类并实例化。
        【为什么用 globals()】
        每种消息只在自己的文件里定义。globals() 包含本文件的所有类，
        所以可以查到 UserMsg / ExitMsg / AbortBackendMsg / BatchBackendMsg。
        """
        return deserialize_type(globals(), json)


# ════════════════════════════════════════════════════════════════════
# BatchBackendMsg —— 一批消息打包成一条
# ────────────────────────────────────────────────────────────────────
# tokenizer 在一个 tick 内可能同时把多个用户消息送出。把它们打成一个
# BatchBackendMsg 一次性发，减少 ZMQ 帧数（每帧都有开销）。
# scheduler 收到后递归拆开逐个处理。
# ════════════════════════════════════════════════════════════════════
@dataclass
class BatchBackendMsg(BaseBackendMsg):
    """批量消息容器——内含多条 BaseBackendMsg。"""
    data: List[BaseBackendMsg]


# ════════════════════════════════════════════════════════════════════
# ExitMsg —— 让 scheduler 优雅退出
# ────────────────────────────────────────────────────────────────────
# 没有任何字段——纯粹的信号。scheduler 收到后 raise KeyboardInterrupt
# 让 run_forever 退出循环、最终走到 shutdown。
# ════════════════════════════════════════════════════════════════════
@dataclass
class ExitMsg(BaseBackendMsg):
    """退出信号消息——无字段。"""
    pass


# ════════════════════════════════════════════════════════════════════
# UserMsg —— 一个新用户请求（最常见的消息）
# ────────────────────────────────────────────────────────────────────
# 包含 3 个字段：
#   uid: 唯一请求 id，用于追踪和回复
#   input_ids: 已经被 tokenizer 切好的 token id 序列
#   sampling_params: 用户指定的采样规则
# scheduler 收到后包装成 PendingReq 入 pending_list。
# ════════════════════════════════════════════════════════════════════
@dataclass
class UserMsg(BaseBackendMsg):
    """一个用户请求——已经过 tokenize、可直接给 scheduler 用。"""
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams


# ════════════════════════════════════════════════════════════════════
# AbortBackendMsg —— 用户取消请求
# ────────────────────────────────────────────────────────────────────
# 用户 HTTP 断连或主动 cancel 时，API 让 tokenizer 发这条消息给 scheduler。
# scheduler 找到对应 uid 的请求，释放它占的显存和页表行。
# ════════════════════════════════════════════════════════════════════
@dataclass
class AbortBackendMsg(BaseBackendMsg):
    """用户取消请求的通知。"""
    uid: int
