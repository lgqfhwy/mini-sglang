"""
========================================================================
文件名: engine/__init__.py
所属模块: 引擎模块的包入口
========================================================================

【这个文件是做什么的】
对外暴露 Engine 主类、EngineConfig、ForwardOutput、BatchSamplingArgs。
Scheduler 通过这几个名字访问引擎能力。
"""

from .config import EngineConfig
from .engine import Engine, ForwardOutput
from .sample import BatchSamplingArgs

__all__ = ["Engine", "EngineConfig", "ForwardOutput", "BatchSamplingArgs"]
