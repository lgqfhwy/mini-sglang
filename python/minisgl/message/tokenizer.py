"""
========================================================================
文件名: message/tokenizer.py
所属模块: tokenizer / detokenizer 进程间消息定义
========================================================================

【这个文件是做什么的 - 一句话总结】
定义 tokenizer 和 detokenizer 进程"接收/发出"的消息。
  - TokenizeMsg:   API server → tokenizer，"请把这段文字切成 token"
  - DetokenizeMsg: scheduler → detokenizer，"采样出了这个 token，请转回文字"
  - AbortMsg:      用户取消请求
  - BatchTokenizerMsg: 批量打包

【为什么这些消息单独成一类】
和 backend.py（发给 scheduler 的消息）类似，但发往不同进程。
分文件让每类消息的 globals() 解析独立、相互不污染。

【消息流向】
   API server → TokenizeMsg → tokenizer 进程 → UserMsg → scheduler
                                                              ↓
   用户 ← UserReply ← API server ← detokenizer ← DetokenizeMsg ← scheduler
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


# ════════════════════════════════════════════════════════════════════
# BaseTokenizerMsg —— tokenizer 系列消息基类
# ════════════════════════════════════════════════════════════════════
@dataclass
class BaseTokenizerMsg:
    """tokenizer 系列消息的基类。"""

    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)


# 批量打包：把多条小消息合并发送（减少帧数）
@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]


# ════════════════════════════════════════════════════════════════════
# DetokenizeMsg —— scheduler 通知 detokenizer "采样出了下一个 token"
# ────────────────────────────────────────────────────────────────────
# scheduler 每生成一个 token 就发一条这种消息。
# detokenizer 维护每个 uid 的累计 token 列表，逐步还原成增量文本。
# 收到 finished=True 时清理该 uid 的状态并通知 frontend。
# ════════════════════════════════════════════════════════════════════
@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    """scheduler → detokenizer：本轮某请求采样到的新 token。"""
    uid: int          # 请求标识
    next_token: int   # 本轮采样出的 token id
    finished: bool    # 该请求是否生成完毕（达到 max_tokens / EOS / abort）


# ════════════════════════════════════════════════════════════════════
# TokenizeMsg —— API server 让 tokenizer 把文本切成 token
# ────────────────────────────────────────────────────────────────────
# text 可以是：
#   - str: 纯文本（用基础 chat template 或直接 tokenize）
#   - List[Dict]: chat messages（OpenAI 格式），tokenizer 会应用 chat template
# ════════════════════════════════════════════════════════════════════
@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    """API server → tokenizer：把这段输入文本切成 token id。"""
    uid: int
    # 文本可以是纯字符串，也可以是 OpenAI 格式的 chat messages 列表
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams


# ════════════════════════════════════════════════════════════════════
# AbortMsg —— 用户取消请求（API → tokenizer/detokenizer）
# ────────────────────────────────────────────────────────────────────
# API server 检测到用户断连时发这条消息。
# tokenizer 把它转发为 AbortBackendMsg 给 scheduler。
# ════════════════════════════════════════════════════════════════════
@dataclass
class AbortMsg(BaseTokenizerMsg):
    """用户取消请求的通知。"""
    uid: int
