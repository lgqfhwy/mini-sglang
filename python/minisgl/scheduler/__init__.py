"""
========================================================================
文件名: scheduler/__init__.py
所属模块: 调度器模块的 Python 包入口
========================================================================

【这个文件是做什么的】
Python 包的"门面文件"。让外部代码可以写
    from minisgl.scheduler import Scheduler, SchedulerConfig
而不用关心它们内部到底在哪个子模块里实现。

【__all__ 的作用】
显式声明"本包对外公开的符号"，让 `from minisgl.scheduler import *`
只导入这两个名字，其它都是实现细节。
"""

from .config import SchedulerConfig
from .scheduler import Scheduler

__all__ = ["Scheduler", "SchedulerConfig"]
