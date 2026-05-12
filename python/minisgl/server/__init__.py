"""
========================================================================
文件名: server/__init__.py
所属模块: API Server / 进程启动器入口
========================================================================

【这个模块负责什么】
1. 用 argparse 解析命令行 (args.py)
2. 启动各子进程 (launch.py): scheduler × N + tokenizer × M + detokenizer × 1
3. 启动 FastAPI HTTP server (api_server.py)

【对外暴露】
launch_server() - 启动整个推理服务的"开关"
"""

from .launch import launch_server

__all__ = ["launch_server"]
