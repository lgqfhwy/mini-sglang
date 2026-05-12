"""
========================================================================
文件名: distributed/info.py
所属模块: Distributed - TP（张量并行）信息单例
========================================================================

【这个文件做什么】
定义 DistributedInfo（包装 rank/size 的小 dataclass）和一个全局单例
用于"任何代码都能拿到当前进程的 TP rank/size"。

【为什么用全局单例】
模型代码（layers/models）到处需要知道"我是 rank 几""TP 一共几张卡"
来正确做切分。如果每次都从参数链一层层传太烦——用模块级单例
get_tp_info() 直接拿。

【生命周期】
- 进程启动后 set_tp_info(rank, size) 一次（在 Engine.__init__ 里）
- 后续到处 get_tp_info() 都拿到这一份
- try_get_tp_info()：未设置时返回 None（极少数代码路径需要）
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    """TP 信息：本进程是第几号 rank、总共多少 rank。"""
    rank: int
    size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        """是否是主 rank（rank 0）——一般用于"只让主 rank 做日志/写文件"。"""
        return self.rank == 0


# 全局单例（每进程一份）
_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    """启动时调用一次，注册本进程的 rank/size。重复调用会报错。"""
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    """获取当前 TP 信息；未设置时报错。"""
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    """获取当前 TP 信息；未设置时返回 None（不报错）。"""
    return _TP_INFO


__all__ = ["DistributedInfo", "set_tp_info", "get_tp_info", "try_get_tp_info"]
