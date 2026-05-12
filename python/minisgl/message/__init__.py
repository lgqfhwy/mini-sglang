"""
========================================================================
文件名: message/__init__.py
所属模块: 跨进程消息（IPC 通信）的统一入口
========================================================================

【这个文件是做什么的 - 一句话总结】
mini-sglang 把推理系统拆成多个进程（API server / tokenizer / scheduler /
detokenizer）。message 模块定义这些进程之间传递的所有"信件格式"。
本文件汇总三类信件并对外暴露。

【三类消息】
- BackendMsg (backend.py): 发给 scheduler 的"后端消息"（含用户请求、abort、退出）
- TokenizerMsg (tokenizer.py): tokenizer 和 detokenizer 进程间的消息
- FrontendMsg (frontend.py): 发给 API server 的"前端消息"（最终给用户的回复）

【消息流向】
   用户 HTTP 请求
     ↓
   API server (前端进程)
     ↓ TokenizeMsg
   tokenizer 进程
     ↓ UserMsg (BackendMsg)
   scheduler 进程
     ↓ DetokenizeMsg (TokenizerMsg)
   detokenizer 进程
     ↓ UserReply (FrontendMsg)
   API server → 用户

【为什么所有消息都用 dataclass + 自定义 encoder/decoder】
- dataclass 让字段一目了然；
- 自定义序列化处理了 torch.Tensor 这种 JSON 默认不认识的类型；
- 反序列化用 cls_map（globals()）把 "__type__" 字符串映射回类——
  各文件 globals() 不同，所以每类消息各自管自己的解码。
"""

from .backend import AbortBackendMsg, BaseBackendMsg, BatchBackendMsg, ExitMsg, UserMsg
from .frontend import BaseFrontendMsg, BatchFrontendMsg, UserReply
from .tokenizer import AbortMsg, BaseTokenizerMsg, BatchTokenizerMsg, DetokenizeMsg, TokenizeMsg

__all__ = [
    "AbortMsg",
    "AbortBackendMsg",
    "BaseBackendMsg",
    "BatchBackendMsg",
    "ExitMsg",
    "UserMsg",
    "BaseTokenizerMsg",
    "BatchTokenizerMsg",
    "DetokenizeMsg",
    "TokenizeMsg",
    "BaseFrontendMsg",
    "BatchFrontendMsg",
    "UserReply",
]
