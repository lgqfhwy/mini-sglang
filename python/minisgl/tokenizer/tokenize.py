"""
========================================================================
文件名: tokenizer/tokenize.py
所属模块: Tokenizer - 文本切成 token id 的"前处理"
========================================================================

【这个文件是做什么的 - 一句话总结】
把用户的文本（普通字符串或 OpenAI 风格的 chat messages）转成 GPU 模型
能吃的 token id 序列。

【为什么需要这个文件】
GPU 模型只懂数字（token id），不懂文字。tokenizer 把
"今天天气真好" → [123, 456, 789]
"Hello world" → [15496, 1917]
这种转换在每个用户请求进入系统时必做。

【为什么单独跑一个 tokenizer 进程】
tokenize 是 CPU 操作（HuggingFace tokenizer 是纯 Python/Rust）。
独立进程：
  - 避开 GIL 影响 scheduler 的 Python 主循环；
  - 多核 CPU 时可以起多个 tokenizer 进程并行处理多用户输入。

【chat template】
ChatGPT/Claude 类 API 接收的是 `[{"role": "system", "content": "..."},
{"role": "user", "content": "..."}]` 这种结构。tokenizer.apply_chat_template
按照模型规定的模板把它拼成一个长字符串再 tokenize。
"""

from __future__ import annotations

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


# ════════════════════════════════════════════════════════════════════
# TokenizeManager - 批量 tokenize 的小工具
# ────────────────────────────────────────────────────────────────────
# 当前实现是逐条 tokenize（TODO 注释说将来想做 batch tokenization 加速）。
# 关键步骤:
#   1. 如果 text 是 list (chat messages)，应用 chat template；
#   2. tokenizer.encode 把字符串转 token id；
#   3. 拍平成 1-D int32 张量返回。
# ════════════════════════════════════════════════════════════════════
class TokenizeManager:
    """文本 → token id 的转换器。"""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # tokenizer 由外部加载好传进来（load_tokenizer 在 utils 里）
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        """
        【功能】把一批 TokenizeMsg 转成一批 token id 张量。

        【参数】msgs: 待处理的消息列表，每个消息含 text 和 sampling_params
        【返回】每个消息对应的 1-D int32 张量列表

        【流程】
        1. 看 text 是不是 list（chat messages）：
           - 是: 用 apply_chat_template 拼成一段字符串
           - 否: 直接当字符串用
        2. tokenizer.encode 切 token id；
        3. .view(-1) 拍平成 1-D；
        4. 转 int32（模型常用 dtype）。
        """
        results: List[torch.Tensor] = []
        # TODO: batch tokenization
        for msg in msgs:
            if isinstance(msg.text, list):
                # chat messages → 字符串
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,  # 在末尾加"assistant 该说话了"的提示
                )
                assert isinstance(prompt, str)
            else:
                # 普通字符串直接用
                prompt = msg.text
            # encode 返回 [1, seq_len] 形状的张量
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="pt")
            )
            # 拍平成 1-D，转 int32
            results.append(input_ids.view(-1).to(torch.int32))
        return results
