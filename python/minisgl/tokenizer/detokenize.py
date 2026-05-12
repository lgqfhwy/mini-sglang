"""
========================================================================
文件名: tokenizer/detokenize.py
所属模块: Tokenizer - token id 还原成增量文本
========================================================================

【这个文件是做什么的 - 一句话总结】
把 scheduler 一个一个采样出来的 token id 还原成"增量文本"——也就是
每收到一个新 token，就吐出"在已发送基础上新增的那一小段文字"。
这是流式输出 (streaming) 的核心。

【为什么 detokenize 不能简单地一个 token 还原一个字】
原因：很多 token 是"半个字"或"半个 emoji"。
例：UTF-8 编码下一个中文字 "好" 是 3 字节，可能被分到多个 token：
  token_a → b"\xe5"     ← 单独看是无效字节
  token_b → b"\xa5\xbd" ← 拼起来才是 "好"
所以如果每收到一个 token 立刻解码，会看到 "�" 这种乱码。
detokenize 必须"积累几个 token 再尝试解码"。

【DecodeStatus 的三个 offset】
每个请求维护一个状态：
  - decoded_ids: 这个请求迄今为止采样到的所有 token id
  - decoded_str: 已经确认能解码出的完整文本
  - read_offset: 上一轮读取 (read_ids) 的截止位置
  - surr_offset: 当作"上下文"包围的起点
  - sent_offset: 已经发给用户的字符长度

【find_printable_text 的作用】
某些"还没收齐的"token 末尾不要立刻发——比如英文单词只发了一半。
- 末尾是换行: 发；
- 末尾是 CJK 字符: 全发（中日韩单字是完整的）；
- 倒数第二位是 CJK: 发到倒数第二位；
- 其它（英文）: 发到最后一个空格——下半个词留着等下一轮。

【流程例子】
   token 序列采样进来: A, B, C, D ...
   每次都做:
     read_ids = decoded_ids[surr_offset:]
     surr_ids = decoded_ids[surr_offset : read_offset]
     batch_decode 这两组 → read_text, surr_text
     new_text = read_text[len(surr_text):]   ← 这一轮真正新增的文本

   如果 new_text 非空且不以 � 结尾 → 确认完整：更新 decoded_str
   否则用 find_printable_text 截一段安全文本发出
"""

from dataclasses import dataclass
from typing import Dict, List

from minisgl.message import DetokenizeMsg
from transformers import PreTrainedTokenizerBase

# Borrowed from sglang


def _is_chinese_char(cp: int):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    #
    # 【功能】判断一个 Unicode 码点是不是 CJK（中日韩汉字）。
    # 【为什么】CJK 字符通常一个 token 就是一个完整字（不像英文要拼词），
    #            所以在 find_printable_text 里可以"放心"发出。
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def find_printable_text(text: str):
    """
    Returns the longest printable substring of text that contains only entire words.

    【功能】从 text 末尾"裁掉"可能还没收完整的部分，只返回可以放心发出的前缀。

    【规则】
    1. 末尾是 \\n：整段发；
    2. 末尾是 CJK 字符：整段发（CJK 一个 token 一个字，不会半字）；
    3. 倒数第二位是 CJK：发到倒数第二位（最后那个可能是不全的标点等）；
    4. 否则（英文/拉丁系）：截到最后一个空格——半个词留着等下一轮。
    """
    # Borrowed from https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99

    # After the symbol for a new line, we flush the cache.
    if text.endswith("\n"):
        return text
    # If the last token is a CJK character, we print the characters.
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # Otherwise if the penultimate token is a CJK character, we print the characters except for the last one.
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
    # which may change with the subsequent token -- there are probably smarter ways to do this!)
    else:
        return text[: text.rfind(" ") + 1]


# ════════════════════════════════════════════════════════════════════
# DecodeStatus - 每个请求的 detokenize 进度
# ────────────────────────────────────────────────────────────────────
# decoded_ids: 已采样到的所有 token id 序列
# decoded_str: 已确认可输出的字符串
# read_offset / surr_offset: tokenizer.batch_decode 的两个滑动窗口
# sent_offset: 已发给用户的字符长度（保证流式只发增量）
# ════════════════════════════════════════════════════════════════════
@dataclass
class DecodeStatus:
    """单个请求的 detokenize 状态。"""
    decoded_ids: List[int]
    decoded_str: str
    read_offset: int  # length of read ids
    surr_offset: int  # length of surr ids
    sent_offset: int  # length of sent out string


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：DetokenizeManager ███
# ════════════════════════════════════════════════════════════════════
#
# 一、为什么不能一个 token 解码成一个字？
#   多字节字符（中文、emoji 等）会被切成多个 token。一个 token 单独
#   解码出来可能是 "�"（无效字节）。所以需要"上下文 + 增量"的滑动
#   解码模式。
#
# 二、batch_decode 的双窗口技巧
#   一次解码 read_ids = decoded_ids[surr_offset:]（包含一些上下文）
#   再解码 surr_ids   = decoded_ids[surr_offset : read_offset]（不含本轮新 token）
#   new_text = read_text[len(surr_text):] —— 本轮真正新增的文本
#
#   为什么要 surr_offset 这层上下文？
#   因为 tokenizer 解码"零散 token 序列"和"完整序列的尾部"可能产生
#   不同结果（空格、合并字符等）。把前文做"陪练"能让 token 间的边界
#   字符正确生成。
#
# 三、流程例子（中文）
#   sample 1: token = X1（不完整字节）
#     decoded_ids = [X1]
#     read_text = "好" 的半个 → "�"
#     检测到 �，不更新 decoded_str；用 find_printable_text 截
#     incremental 是空（或之前的尾巴）
#
#   sample 2: token = X2（凑齐完整字 "好"）
#     decoded_ids = [X1, X2]
#     read_text = "好" → 不含 �
#     更新 decoded_str = "好"，surr_offset/read_offset 推进
#     incremental = "好" 发给 frontend
#
# ════════════════════════════════════════════════════════════════════
class DetokenizeManager:
    """token id → 增量文本的转换器（流式）。"""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # uid -> DecodeStatus
        # 每个请求独立维护一个状态（请求结束时清除）
        self.decode_map: Dict[int, DecodeStatus] = {}
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id

    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
        """
        【功能】给定一批 DetokenizeMsg，返回每条对应的"本轮增量文本"。

        【内部流程】
        1. 收集所有 read_ids / surr_ids（需要 batch_decode）
        2. tokenizer.batch_decode 两组 → 拿到 read_texts / surr_texts
        3. 对每个 msg:
           - new_text = read - surr
           - 如果 new_text 非空且没 �: 确认完整，推进 offset
           - 否则: 用 find_printable_text 安全截取
           - 增量 = 完整输出从 sent_offset 之后那段
        4. 完成的请求清理掉 decode_map[uid]
        """
        read_ids: List[List[int]] = []
        surr_ids: List[List[int]] = []
        for msg in msgs:
            # 第一次见到这个 uid，初始化状态
            if msg.uid not in self.decode_map:
                self.decode_map[msg.uid] = DecodeStatus(
                    decoded_ids=[],
                    decoded_str="",
                    read_offset=0,
                    surr_offset=0,
                    sent_offset=0,
                )
            s = self.decode_map[msg.uid]
            # 收到的 next_token：如果是结束的 EOS，不加入 decoded_ids（避免输出 <|endoftext|>）
            if not (msg.finished and msg.next_token == self.eos_token_id):
                s.decoded_ids.append(msg.next_token)
            # 准备本轮 batch_decode 的两个窗口
            read_ids.append(s.decoded_ids[s.surr_offset :])
            surr_ids.append(s.decoded_ids[s.surr_offset : s.read_offset])

        # 一次性 batch_decode（比逐个 decode 快很多）
        read_texts = self.tokenizer.batch_decode(read_ids)
        surr_texts = self.tokenizer.batch_decode(surr_ids)

        incremental_strs: List[str] = []
        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):
            s = self.decode_map[msg.uid]
            # 本轮"真正新增"的文本
            new_text = read_str[len(surr_str) :]
            # Streaming chunk: update the decode status
            if len(new_text) > 0 and not new_text.endswith("�"):
                # new_text 完整、可信 → 确认进入 decoded_str
                output_str = s.decoded_str + new_text
                s.decoded_str = output_str
                # 滑动两个 offset
                s.surr_offset = s.read_offset
                s.read_offset = len(s.decoded_ids)
            else:
                # 还没收齐（末尾 �）或为空 → 安全截取
                new_text = find_printable_text(new_text)
                output_str = s.decoded_str + new_text

            # 计算本轮要发给前端的"增量"（在 sent_offset 之后的部分）
            incremental_output = output_str[s.sent_offset :]
            s.sent_offset = len(output_str)
            incremental_strs.append(incremental_output)
            # 请求结束，清理状态
            if msg.finished:
                del self.decode_map[msg.uid]

        return incremental_strs
