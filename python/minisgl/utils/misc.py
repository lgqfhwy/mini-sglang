"""
========================================================================
文件名: utils/misc.py
所属模块: Utils - 杂项数学/装饰器工具
========================================================================

提供几个项目里到处都用的小工具：
- div_even: 整除（强制不能有余数），可选 allow_replicate 处理 GQA 的 KV head 复制
- div_ceil: 向上取整除法
- align_ceil/align_down: 对齐到 N 的倍数（向上/向下）
- Unset / UNSET: "未设置"哨兵值（区别于 None）
- call_if_main: 装饰器，让函数仅在脚本作为 __main__ 运行时调用
"""

from __future__ import annotations


def call_if_main(name: str = "__main__", discard: bool | None = None):
    """Decorator to ensure a function will call when the script is run as main."""
    if name != "__main__":
        discard = False if discard is None else discard
        if discard:
            return lambda _: None
        else:
            return lambda f: f
    else:
        discard = True if discard is None else discard
        if discard:
            return lambda f: (f() or True) and None
        else:
            return lambda f: (f() and None) or f


def div_even(a: int, b: int, allow_replicate: bool = False) -> int:
    """
    Divides two integers. If allow_replicate=True, allows b > a when b % a == 0, returning 1.

    【功能】整除（要求 a 必须被 b 整除）。
    【allow_replicate】TP 切 KV head 时，如果卡数比 head 数还多，
        要求卡数能整除 head 数——此时返回 1（每卡复制一份完整 KV）。
    """
    if allow_replicate and b > a:
        assert b % a == 0, f"{b = } must be divisible by {a = } for KV head replication"
        return 1
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b


def div_ceil(a: int, b: int) -> int:
    """Divides two integers, rounding up"""
    return (a + b - 1) // b


def align_ceil(a: int, b: int) -> int:
    """Aligns a to the next multiple of b"""
    return div_ceil(a, b) * b


def align_down(a: int, b: int) -> int:
    """Aligns a to the previous multiple of b"""
    return (a // b) * b


class Unset:
    """"未设置"哨兵类型——区别于 None（None 是合法值，Unset 表示真没设过）。"""
    pass


# 全局唯一的 Unset 实例（"哨兵值"）
UNSET = Unset()
