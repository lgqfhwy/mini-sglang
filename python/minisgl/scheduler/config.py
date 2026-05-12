"""
========================================================================
文件名: scheduler/config.py
所属模块: 调度器（Scheduler）模块的配置定义
========================================================================

【这个文件是做什么的 - 一句话总结】
这个文件就像调度器的"参数面板"——把调度器在启动时需要的所有可调
参数（最大 prefill 一次能处理多少 token、用哪种 cache、ZMQ 通信地址
怎么算等）打包成一个不可变的 dataclass。

【为什么需要这个文件 / 这个模块存在的原因】
调度器本身已经够复杂，如果再把"上限多少、地址多少"这些配置散落
在代码里，改一处忘一处就是 bug。统一收集到 SchedulerConfig：
  1. 一处定义、各处只读，避免散弹枪式修改；
  2. dataclass(frozen=True) 强制不可变，启动后绝不会被偷偷改掉；
  3. 通过继承 EngineConfig，自动包含模型路径、TP 大小等引擎参数。

【这个文件在整个推理流程中的位置】
   命令行参数 (server/args.py)
     ↓
   ★ SchedulerConfig（本文件定义） ★
     ↓
   Scheduler.__init__ 接收并保存
     ↓
   各子管理器（Cache/Table/Prefill/Decode）从中读取自己需要的字段

【核心概念速览】

- ZMQ（ZeroMQ）：
    一个轻量级、高性能的进程间消息库。本项目用它在
    API server / tokenizer / scheduler / detokenizer 这几个独立进程
    之间传 UserMsg、DetokenizeMsg 等消息。
    类比：进程间的"传纸条"系统，比直接走 TCP 简单很多。

- IPC（Inter-Process Communication）：
    ZMQ 的 ipc:// 协议走 unix domain socket（用 /tmp 下的特殊文件名做
    "地址"），不走网络，比 tcp:// 快得多——本机进程通信首选。

- max_extend_tokens（最大扩展 token 数）：
    单次 prefill batch 里所有请求加起来最多算多少 token 的 KV。
    这是调度的核心预算指标——越大吞吐越高，但单步延迟也越大、显存
    峰值也越高。8192 是常用的中等取值。

- "extend"（扩展长度）：
    回顾 core.py：每个请求每轮要新算的 token 数 = device_len - cached_len。
    一批请求的 extend 之和就是本步要算的总 token 数，必须不超过
    max_extend_tokens 这个预算。

- offline mode（离线模式）：
    跑离线 benchmark 时不需要走 ZMQ 收发，调度器直接从内存中拿请求、
    把结果存回内存——offline_mode=True 就走这条简化路径。

【关键设计决策】

1. frozen=True 让配置不可变：
   防止运行时被意外修改。要改？重新构造一个新的 SchedulerConfig。

2. ZMQ 地址里带 PID 后缀（_get_pid_suffix）：
   同一台机器可能同时跑多个 mini-sglang 实例（开发时、测试时常见）。
   地址里加 ".pid=12345" 这种后缀就能避免不同实例抢同一个 IPC 文件。

3. 地址作为 @property 而不是 dataclass 字段：
   字段只存唯一的"种子"_unique_suffix，三个地址按规则派生出来。
   后续修改地址格式只需改一处。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    """
    【功能】生成当前进程的"PID 后缀"字符串，用于隔离 ZMQ 地址。
    【返回】例如当前进程 PID = 12345，返回 ".pid=12345"
    【为什么】同一台机器可能同时存在多个 mini-sglang 实例，必须给
              ipc:// 地址加唯一后缀，避免互相串线。
    """
    import os  # 延迟 import，避免文件加载时无谓开销

    return f".pid={os.getpid()}"


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：SchedulerConfig ███
# ════════════════════════════════════════════════════════════════════
#
# 一、这个类要解决什么问题？
#
#   调度器跑起来要知道一堆参数：
#     - 单次 prefill 预算多少 token（max_extend_tokens）
#     - 用哪种前缀缓存（cache_type: "radix" 或 "naive"）
#     - 是不是离线模式（offline_mode）
#     - 几个 ZMQ 地址（用于跨进程通信）
#   外加从 EngineConfig 继承来的：
#     - 模型路径、TP（张量并行）大小、attention 后端类型等
#
#   全打包成一个 frozen dataclass，启动时一次性传入。
#
# 二、典型使用流程
#
#   1. 用户运行: python -m minisgl ... --max-extend-tokens 16384
#      → server/args.py 解析参数构造 SchedulerConfig(max_extend_tokens=16384)
#   2. SchedulerConfig 实例传给 launch_scheduler_process
#   3. Scheduler.__init__ 收到 config，把它的字段分发给各管理器：
#      - cache_manager 用 config.cache_type
#      - prefill_manager 用 config.max_extend_tokens
#      - io_mixin 用 config.zmq_* 几个地址
#
# 三、ZMQ 地址例子
#
#   假设当前进程 PID = 99999：
#     zmq_backend_addr         = "ipc:///tmp/minisgl_0.pid=99999"
#       ← API/tokenizer → scheduler 的消息通道（用户请求进入）
#     zmq_detokenizer_addr     = "ipc:///tmp/minisgl_1.pid=99999"
#       ← scheduler → detokenizer 的消息通道（生成结果出去）
#     zmq_scheduler_broadcast_addr = "ipc:///tmp/minisgl_2.pid=99999"
#       ← TP 多卡时：rank 0 把消息广播给其他 rank
#
# ════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    """
    【类名】SchedulerConfig
    【一句话描述】调度器启动时需要的所有配置参数（继承 EngineConfig）。
    """

    # ----------------------------------------------------------------
    # 字段: max_extend_tokens
    # 类型: int
    # 含义: 单次前向能"扩展"的总 token 数上限——也就是单步 prefill batch
    #       里所有请求 extend_len 之和不能超过这个值。
    # 取值: 默认 8192；常见 4096~32768。越大吞吐越好但单步延迟越大。
    # 调度依据: PrefillManager 用它做 token_budget。
    # ----------------------------------------------------------------
    max_extend_tokens: int = 8192

    # ----------------------------------------------------------------
    # 字段: cache_type
    # 类型: str
    # 含义: 前缀缓存（Prefix Cache）的实现类型。
    # 取值:
    #   - "radix"  : 基于 radix 树的前缀共享（多请求若有相同 prompt 前缀，
    #                共用同一份 KV，能省大量显存和时间）
    #   - "naive"  : 朴素实现（无前缀共享，每个请求独立分配）
    # 默认 "radix"：多用户场景下显著提速。
    # ----------------------------------------------------------------
    cache_type: str = "radix"

    # ----------------------------------------------------------------
    # 字段: offline_mode
    # 类型: bool
    # 含义: 是否运行在"离线模式"（不走 ZMQ，直接 in-process 拿数据）。
    # 用途: benchmark、单元测试时把 server/tokenizer 都省掉，直接调度器
    #       接受 Python 函数传进来的请求。
    # ----------------------------------------------------------------
    offline_mode: bool = False

    # ----------------------------------------------------------------
    # 字段: _unique_suffix
    # 类型: str
    # 含义: 用于隔离同机多实例的 ZMQ 地址后缀，默认是 ".pid=当前进程号"。
    # 注意: 加下划线前缀表示"内部字段"，外部不应直接使用，
    #       而是通过下面三个 @property 派生出实际地址。
    # 注意: default_factory=_get_pid_suffix 让每次构造时动态取 PID，
    #       而不是模块加载时固定一次（多进程模型里 PID 会变）。
    # ----------------------------------------------------------------
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        """
        外部（API server / tokenizer）→ 调度器 的 ZMQ 通道地址。
        用 PULL/PUSH 模式，调度器是"拉"端。
        """
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        """
        调度器 → 反 tokenizer（detokenizer）的 ZMQ 通道地址。
        调度器把"采样出来的 next_token"发出去，detokenizer 拿到后还原成文本。
        """
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        """
        TP（张量并行）多卡时，rank 0（主调度器）→ 其他 rank 的广播通道。
        所有 rank 必须看到相同的请求序列，才能各自一致地做计算。
        """
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        """
        单次 forward 处理的最大 token 数。
        当前等于 max_extend_tokens——表示我们假设 decode 比 prefill 短得多，
        所以以 prefill 预算作为整体上限。
        """
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        """
        是否让"backend 一侧"主动创建到 detokenizer 的 ZMQ socket。
        True 表示由调度器进程 bind socket、detokenizer 来 connect。
        """
        return True
