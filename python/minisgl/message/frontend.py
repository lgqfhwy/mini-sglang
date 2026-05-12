"""
========================================================================
文件名: message/frontend.py
所属模块: 发给前端（API server）的消息定义
========================================================================

【这个文件是做什么的 - 一句话总结】
定义"detokenizer → API server"方向的消息——核心是 UserReply（增量
输出 + 完成标志）。

【为什么需要这个文件】
推理是流式的（用户 HTTP/SSE 长连接），每生成一个 token 就要把对应
的文本片段发回给用户。这里的 UserReply 就是"一小段返回的文本"的载体。

【典型消息内容】
   UserReply(uid=12345, incremental_output=" hello", finished=False)
   UserReply(uid=12345, incremental_output=" world", finished=False)
   ...
   UserReply(uid=12345, incremental_output="!", finished=True)

API server 收到后通过 SSE 把 incremental_output 流式推给浏览器。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .utils import deserialize_type, serialize_type


@dataclass
class BaseFrontendMsg:
    """前端消息基类。"""

    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        return deserialize_type(globals(), json)


# 批量打包多条 UserReply（减少 ZMQ 帧）
@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    data: List[BaseFrontendMsg]


# ════════════════════════════════════════════════════════════════════
# UserReply —— detokenizer → API server 的核心消息
# ────────────────────────────────────────────────────────────────────
# 每生成一个 token、且经过 detokenizer 转回文字后，发一条 UserReply：
#   - uid:                标识是哪个请求
#   - incremental_output: 本轮新增的那段文本（增量）
#   - finished:           是否是该请求的最后一条消息
# ════════════════════════════════════════════════════════════════════
@dataclass
class UserReply(BaseFrontendMsg):
    """detokenizer → API server：一段增量的文本输出。"""
    uid: int
    incremental_output: str  # 本轮新增的文本片段（流式输出）
    finished: bool           # 是否已生成完毕
