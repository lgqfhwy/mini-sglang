"""
========================================================================
文件名: tokenizer/__init__.py
所属模块: Tokenizer / Detokenizer 进程入口
========================================================================

本模块定义"独立运行的 tokenizer 进程"——它同时承担 tokenize（文本→token）
和 detokenize（token→文本）两个角色：
  - 从 frontend 进程拿 TokenizeMsg → 切成 token → 发给 scheduler
  - 从 scheduler 进程拿 DetokenizeMsg → 把 token 转回增量文本 → 发给 frontend
"""

from .server import tokenize_worker

__all__ = ["tokenize_worker"]
