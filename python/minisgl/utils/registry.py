"""
========================================================================
文件名: utils/registry.py
所属模块: Utils - 注册表模式（按字符串选实现）
========================================================================

【这个文件做什么】
Registry 模式的通用实现——让"按字符串选不同实现"的代码更整洁。

【典型用法】
    ATTENTION_BACKENDS = Registry[BackendCreator]("Attention Backend")

    @ATTENTION_BACKENDS.register("fi")
    def create_fi(): return FlashInferBackend(...)

    backend = ATTENTION_BACKENDS["fi"]()   # 拿到工厂调用

【优点 vs if-else 长链】
- 添加新实现只需 @register("xxx")，不用改任何工厂代码
- supported_names() 自动列出所有已注册名字（用于 argparse choices）
- 各种 attention/MoE/cache 后端都用同一套机制，统一风格
"""

from typing import Callable, Generic, Iterable, List, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """通用注册表——把字符串 name 映射到对象 T。"""

    def __init__(self, type: str):
        self._registry = {}
        # 类型描述字符串（用于错误信息）
        self._type = type

    def register(self, name: str) -> Callable[[T], None]:
        """装饰器：@registry.register("xxx") 把函数/类注册为 name。"""
        if name in self._registry:
            raise KeyError(f"{self._type} '{name}' is already registered.")

        def decorator(item: T) -> None:
            self._registry[name] = item

        return decorator

    def __getitem__(self, name: str) -> T:
        """按 name 查注册项；找不到时报错。"""
        if name not in self._registry:
            raise KeyError(f"Unsupported {self._type}: {name}")
        return self._registry[name]

    def supported_names(self) -> List[str]:
        """返回所有已注册的名字（用于命令行 choices）。"""
        return list(self._registry.keys())

    def assert_supported(self, names: str | Iterable[str]) -> None:
        """校验给定名字是否都已注册；用于参数解析阶段。"""
        if isinstance(names, str):
            names = [names]
        for name in names:
            if name not in self._registry:
                from argparse import ArgumentTypeError

                raise ArgumentTypeError(
                    f"Unsupported {self._type}: {name}. "
                    f"Supported items: {self.supported_names()}"
                )
