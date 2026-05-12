"""
========================================================================
文件名: server/api_server.py
所属模块: 前端 HTTP API Server (FastAPI + uvicorn)
========================================================================

【这个文件是做什么的 - 一句话总结】
对外暴露 HTTP API（兼容 OpenAI 风格）：
  - POST /generate         —— mini-sglang 原生格式
  - POST /v1/chat/completions —— OpenAI Chat Completions 兼容
  - GET  /v1/models        —— 列出可用模型
还提供一个交互式 shell 模式（无 HTTP）。

【为什么需要这个文件】
LLM 推理框架要能接到生态——常见的客户端（openai-python / LangChain /
vllm benchmark / 前端 ChatBot）都是按 OpenAI API 风格写的。提供兼容
接口能直接对接现有客户端。

【请求生命周期】
   用户 HTTP POST /v1/chat/completions
     ↓
   FastAPI 解析 → 构造 TokenizeMsg → send_one 推给 tokenizer
     ↓
   返回 StreamingResponse → SSE 流式响应
     ↓ (异步)
   recv_tokenizer 不停接收 UserReply → 唤醒对应 uid 的 event
     ↓
   stream_xxx_completions 协程消费 UserReply → yield SSE 块
     ↓
   用户浏览器收到流式输出

【关键技术点】

- FastAPI + uvicorn:
    Python 异步 Web 框架（基于 Starlette），用 uvicorn 作为 ASGI server。

- StreamingResponse / SSE (Server-Sent Events):
    HTTP 的"服务器推"技术。每条消息 `data: ...\\n\\n`，浏览器或客户端
    通过 EventSource 接收。

- asyncio.Event:
    每个用户 uid 对应一个 Event，detokenizer 来消息时 set() 它，
    用户的协程在 wait() 阻塞，被唤醒后取走累积的消息。

- 取消机制 (stream_with_cancellation):
    if request.is_disconnected() → 用户断连 → 抛 CancelledError →
    自动发 AbortMsg 给后端释放资源。
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    TokenizeMsg,
    UserReply,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")

# 全局 FrontendManager 单例（FastAPI 路由通过 get_global_state 拿到它）
_GLOBAL_STATE = None


def get_global_state() -> FrontendManager:
    """拿到全局 FrontendManager 单例。"""
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    """把 BatchFrontendMsg 信封拆开成 UserReply 列表。"""
    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)
    return [msg]


# ════════════════════════════════════════════════════════════════════
# Pydantic 请求/响应模型 - FastAPI 自动校验/序列化
# ════════════════════════════════════════════════════════════════════
class GenerateRequest(BaseModel):
    """mini-sglang 原生 /generate 请求体。"""
    prompt: str
    max_tokens: int
    ignore_eos: bool = False


class Message(BaseModel):
    """OpenAI chat 消息格式。"""
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions.

    支持两种用法：
    - messages: 标准 chat 格式
    - prompt: 直接传 prompt 字符串
    """

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None

    max_tokens: int = 16
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: List[str] = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False


class ModelCard(BaseModel):
    """OpenAI /v1/models 返回中的一条模型记录。"""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str


class ModelList(BaseModel):
    """OpenAI /v1/models 返回体。"""
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：FrontendManager（读代码之前请先读完这段） ███
# ════════════════════════════════════════════════════════════════════
#
# 一、它做什么？
#   维护"从用户请求到 SSE 流式输出"的全部状态：
#     - 自增的 uid 计数器
#     - 每个 uid 对应一个 asyncio.Event + UserReply 累积列表
#     - 一个后台协程 listen() 不停从 tokenizer 进程拉消息分发
#     - 流式输出协程：wait_for_ack / stream_generate / stream_chat_completions
#     - 用户断连时的 abort 协程
#
# 二、单次请求时序
#   1. 路由收到 POST /v1/chat/completions
#   2. state.new_user() 拿到一个 uid
#   3. state.send_one(TokenizeMsg(...)) 把 prompt 推给 tokenizer
#   4. 返回 StreamingResponse(state.stream_chat_completions(uid))
#   5. FastAPI 开始流式调用 stream_chat_completions
#      - 内部 wait_for_ack(uid) 不断 await event.wait()
#      - 后台 listen() 协程从 tokenizer 拉到 UserReply 时，
#        往 ack_map[uid] 追加并 set event
#      - wait_for_ack 被唤醒，yield 累积的 reply
#      - stream_chat_completions 把 reply 格式化成 SSE chunk
#   6. 用户断连 → request.is_disconnected() True → 抛 CancelledError →
#      触发 abort_user(uid) 通知后端
#
# ════════════════════════════════════════════════════════════════════
@dataclass
class FrontendManager:
    """前端 API server 的全局状态管理器。"""

    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]  # → tokenizer
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]   # ← detokenizer
    uid_counter: int = 0
    initialized: bool = False  # listen 协程是否已经启动
    # uid → 该用户累积但还没消费的 UserReply
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    # uid → 唤醒消费协程的 Event
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)

    def new_user(self) -> int:
        """分配一个新 uid 并初始化对应的累积列表和 Event。"""
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    async def listen(self):
        """
        【功能】后台协程：不停从 detokenizer 拉 UserReply，分发到对应 uid 的累积列表。
        【生命周期】整个 server 运行期间一直在跑。
        """
        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                if msg.uid not in self.ack_map:
                    # 用户已经断连/abort，丢弃
                    continue
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    def _create_listener_once(self):
        """第一次有请求时启动 listen 协程（懒启动）。"""
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        """把一条消息发给 tokenizer 进程。"""
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        """
        【功能】异步生成器：每次有新 UserReply 时 yield 之；直到 finished=True。

        【实现模式】event-driven：
        1. await event.wait()
        2. event.clear() 重置
        3. 取走 ack_map[uid] 的累积消息逐个 yield
        4. 如果最后一条 finished=True，跳出循环
        5. 清理本 uid 的状态
        """
        event = self.event_map[uid]

        while True:
            await event.wait()
            event.clear()

            pending = self.ack_map[uid]
            self.ack_map[uid] = []
            ack = None
            for ack in pending:
                yield ack
            if ack and ack.finished:
                break

        del self.ack_map[uid]
        del self.event_map[uid]

    async def stream_generate(self, uid: int):
        """
        【功能】mini-sglang 原生 /generate 接口的 SSE 流。
        【格式】data: 文本块\\n
        【结束】data: [DONE]\\n
        """
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int):
        """
        【功能】OpenAI Chat Completions SSE 兼容流。

        【格式】每个 chunk 是 JSON：
            {"id": "...", "object": "...", "choices": [{"delta": {...}, ...}]}
        【流程】
        - 首块带 delta.role = "assistant"
        - 中间块带 delta.content = 增量文本
        - 末块带 finish_reason = "stop"
        - 终止符 data: [DONE]
        """
        first_chunk = True
        async for ack in self.wait_for_ack(uid):
            delta = {}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            if ack.incremental_output:
                delta["content"] = ack.incremental_output

            chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

            if ack.finished:
                break

        # send final finish_reason
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        """
        【功能】包一层，每个 yield 前检查用户是否断连；断连了发 abort。

        【实现】
        - 在每次 yield 前 await request.is_disconnected()
        - 如果断连：raise CancelledError → except 里发 AbortMsg
        - 这是一种典型的"协程取消传播"模式。
        """
        try:
            async for chunk in generator:
                # 检测用户是否断连
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            # 发 abort 给后端，让 scheduler 释放该请求的显存
            asyncio.create_task(self.abort_user(uid))
            raise

    async def abort_user(self, uid: int):
        """
        【功能】向后端发 AbortMsg，并清理本 uid 的本地状态。
        【sleep 0.1】给一个短暂窗口让最后几条 reply 收到再清理（防止竞态）。
        """
        await asyncio.sleep(0.1)
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        logger.warning("Aborting request for user %s", uid)
        await self.send_one(AbortMsg(uid=uid))

    def shutdown(self):
        """关闭 ZMQ 队列。"""
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    【功能】FastAPI 应用的生命周期钩子——退出时调用 shutdown。
    【yield 前】启动逻辑（这里不需要）
    【yield 后】退出逻辑（关闭 ZMQ）
    """
    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


# 创建 FastAPI app 实例（模块级，uvicorn 会通过路径找到它）
app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)


# ════════════════════════════════════════════════════════════════════
# HTTP 路由
# ════════════════════════════════════════════════════════════════════
@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    """mini-sglang 原生 /generate 接口（纯 prompt → 增量文本流）。"""
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    uid = state.new_user()
    # 推送 TokenizeMsg 到 tokenizer 进程
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        )
    )

    # 返回流式响应（带取消传播）
    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    """OpenAI v1 根路径——返回简单 ok 表示存活。"""
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    """OpenAI Chat Completions 兼容接口。"""
    state = get_global_state()
    if req.messages:
        # 以 chat messages 格式传给 tokenizer，它会应用 chat template
        prompt = [msg.model_dump() for msg in req.messages]
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_chat_completions(uid), request, uid),
        media_type="text/event-stream",
    )


@app.get("/v1/models")
async def available_models():
    """OpenAI /v1/models 兼容——返回当前加载的单个模型。"""
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


async def shell_completion(req: OpenAICompletionRequest):
    """shell 模式下的 chat 完成——内部不走 HTTP 而是直接调函数。"""
    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    prompt = [msg.model_dump() for msg in req.messages]

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    async def _abort():
        await state.abort_user(uid)

    return StreamingResponse(
        state.stream_generate(uid),
        media_type="text/event-stream",
        background=BackgroundTask(lambda: _abort),
    )



async def shell():
    """
    【功能】交互式 shell 模式（无 HTTP，直接在终端聊天）。
    【操作】
    - / 开头是命令：/exit 退出，/reset 清空历史
    - 其它输入作为用户消息发给模型，流式打印回复
    - Ctrl+D 退出，会自动 kill 所有子进程
    """
    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)

    try:
        history: List[Tuple[str, str]] = []
        while True:
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")
            # 历史消息（user/assistant 交替）
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # send to server
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
            )
            cur_msg = ""
            async for chunk in (await shell_completion(req)).body_iterator:
                msg = chunk.decode()  # type: ignore
                assert msg.startswith("data: "), msg
                msg = msg[6:]
                assert msg.endswith("\n"), msg
                msg = msg[:-1]
                if msg == "[DONE]":
                    continue
                cur_msg += msg
                print(msg, end="", flush=True)
            print("", flush=True)
            history.append((cmd, cur_msg))
    except EOFError:
        # user pressed Ctrl-D
        pass
    finally:
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        get_global_state().shutdown()
        # 把所有子进程 kill 干净
        # then kill all the subprocesses
        import psutil

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()


def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.

    【功能】启动 FastAPI 主进程 + 触发 backend 子进程启动。

    【流程】
    1. 初始化 FrontendManager（连 ZMQ）；
    2. 调 start_backend() 启动所有 backend 子进程（scheduler + tokenizer）；
    3. uvicorn.run(app) 或 asyncio.run(shell())。
    """

    global _GLOBAL_STATE

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # start the backend here
    # 启动后端 schedulers + tokenizers/detokenizer 并等他们就绪
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())
