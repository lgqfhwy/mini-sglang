"""
========================================================================
文件名: kvcache/mha_pool.py
所属模块: KV Cache - MHA（多头注意力）模型的 KV 显存池实现
========================================================================

【这个文件是做什么的 - 一句话总结】
这个文件创建并管理一大块 GPU 显存——存所有正在跑的请求的 K（键）
和 V（值）张量。它本身不管"哪个 slot 属于谁"（那是 CacheManager 的事），
只提供 3 个核心能力：
  1. 创建出这块大显存（__init__）
  2. 把本步算的 k/v 写到指定 slot（store_kv）
  3. 给注意力 kernel 读用的 k/v 张量（k_cache / v_cache）

【为什么需要这个文件 / 这个模块存在的原因】
KV Cache 是 LLM 推理的最大显存消费方（可达模型权重的几倍）。必须：
  - 启动时一次性预分配（避免运行时反复 alloc/free 引起的显存碎片）；
  - 用张量切片高效访问（注意力 kernel 直接通过 page_table 读它）；
  - 支持张量并行（每个 GPU 只存自己负责的 kv_heads 的 KV，省 1/TP 显存）。

【KV pool 的形状解析】
   _kv_buffer: shape = [2, num_layers, num_pages, page_size, local_kv_heads, head_dim]
   维度含义:
     维 0: 2          — 索引 0 是 K，索引 1 是 V
     维 1: num_layers — Transformer 层数
     维 2: num_pages  — 总共多少页
     维 3: page_size  — 每页几个 slot
     维 4: local_kv_heads — 本 GPU 负责的 KV head 数（= num_kv_heads / TP）
     维 5: head_dim   — 每个 head 的向量维度

   显存占用 = 2 × num_layers × (num_pages × page_size) × local_kv_heads × head_dim × dtype_size

【核心概念速览】

- MHA / MQA / GQA：
    MHA (Multi-Head Attention)：每个 head 都有自己的 Q/K/V，标准做法。
    MQA (Multi-Query)：所有 Q head 共享一组 K/V。
    GQA (Grouped Query)：折中——每组 Q head 共享一组 K/V，num_kv_heads < num_q_heads。
    本类适用于 MHA/MQA/GQA（都用 num_kv_heads 控制 K/V 头数）。

- TP（张量并行 / Tensor Parallelism）：
    把 num_kv_heads 平均切给 TP_size 个 GPU，每 GPU 只存
    num_kv_heads/TP_size 个 head 的 KV。例如 num_kv_heads=8、TP=2，
    每个 GPU 存 4 个 head 的 KV，显存对半省。

- page_size：
    每页存几个 token 的 K/V。page_size=1 时一个 slot 一个 token；
    page_size>1 时多个 token 紧密存在一起，对部分注意力 kernel 友好。
"""

from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：MHAKVCache（读代码之前请先读完这段） ███
# ════════════════════════════════════════════════════════════════════
#
# 一、这个类要解决什么问题？
#
#   推理时显存 = 模型权重 + KV Cache + 激活值。其中 KV Cache 在长上下文
#   场景下是最大头。必须高效管理。MHAKVCache 就是一个大显存池子，
#   预分配后用 page_table[req, pos] 索引访问。
#
# 二、具体例子（数值化）
#
#   假设:
#     num_kv_heads = 8, head_dim = 128, num_layers = 32
#     num_pages = 10000, page_size = 1
#     dtype = torch.bfloat16 (2 bytes)
#     TP size = 2 (即 2 张卡并行)
#
#   每张卡上:
#     local_kv_heads = 8 / 2 = 4
#     _kv_buffer shape = [2, 32, 10000, 1, 4, 128]
#     总大小 = 2 × 32 × 10000 × 1 × 4 × 128 × 2 bytes
#            = 655,360,000 bytes ≈ 624 MB
#
#   能装多少 token？= num_pages × page_size = 10000 个 token 的 KV。
#   够 50 个 200-token 的并发请求，或 10 个 1000-token 的请求等。
#
# 三、store_kv 的工作流程
#
#   一层注意力子层算完 k, v 后（形状 [num_tokens, local_kv_heads, head_dim]）:
#     out_loc = [3, 5, 7, ...] （每个 token 的目标 slot 编号）
#     layer_id = 这是第几层
#
#   1. 取出本层的 k/v 缓存视图: _k_buffer[layer_id], _v_buffer[layer_id]
#      形状 [num_pages, page_size, local_kv_heads, head_dim]
#   2. 通过 .view(_storage_shape) 把它"压平"成
#      [num_pages * page_size, local_kv_heads, head_dim]
#      —— 现在第一维就是"slot 编号"了。
#   3. 调用 kernel store_cache 把 (k, v) 按 out_loc 写到对应位置。
#
# ════════════════════════════════════════════════════════════════════
class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.

    【类名】MHAKVCache（Multi-Head Attention 用的 KV Cache 池）
    【一句话描述】预分配的 GPU 大缓冲区，按 (layer, slot) 索引存所有
                  正在跑的请求的 K/V。
    【生活类比】一个超大的"鞋柜"，每个格子（slot）存一只鞋（一个 token
                的 K/V），多层架子（layers）×多列（slots）。
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """
        【功能】预分配 KV 缓存的大显存块。

        【参数】
        - num_kv_heads (int): 模型的 KV head 总数（GQA/MQA 模型会比 Q head 少）
        - num_layers (int): Transformer 层数
        - head_dim (int): 每个 head 的向量维度
        - num_pages (int): 总共要预分配多少页
        - page_size (int): 每页几个 slot
        - dtype: KV 缓存的数据类型（通常 bf16/fp16 省显存）
        - device: 在哪个 GPU 上分配

        【关键操作】
        1. 拿到 TP 信息，把 num_kv_heads 按 TP 切分；
        2. torch.empty 一次性分配整块显存（不初始化具体值——反正写入前不会读）；
        3. 切出 _k_buffer 和 _v_buffer 视图（共享底层显存，不复制）；
        4. 计算 _storage_shape：把 (num_pages, page_size) 这两维 view 成
           (num_pages * page_size) 一维，方便用 slot 编号直接索引。
        """
        # 取当前进程的 TP（张量并行）信息
        tp_info = get_tp_info()
        # 把 num_kv_heads 按 TP 平均切分（allow_replicate=True 允许在某些
        # 模型上 num_kv_heads 不能整除 TP size 时复制到所有卡）
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)

        # 一次性预分配整块 GPU 显存
        # ⚡ 性能关键：用 torch.empty 而非 zeros——节省初始化时间。
        # 反正本块显存被读之前一定会先被 store_kv 写过，初值无所谓。
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._num_layers = num_layers
        # K 缓存和 V 缓存的视图（共享内存，没有额外开销）
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        # 为 store_kv 时 view 用的形状：把 (num_pages, page_size) 合并成一维
        # 这样 slot 编号可以直接作为第 0 维索引
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        """
        【功能】返回第 index 层的 K 缓存张量（共享内存视图）。
        【返回形状】[num_pages, page_size, local_kv_heads, head_dim]
        【调用方】注意力 kernel 在做 attention 时用 page_table 索引读这个张量。
        """
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        """返回第 index 层的 V 缓存张量。"""
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        """
        【功能】把本步算出的 k, v 张量写到 KV pool 里 out_loc 指定的 slot。

        【参数】
        - k, v: 形状 [num_tokens, local_kv_heads, head_dim] 的 GPU 张量
        - out_loc: [num_tokens] 整数张量，每元素是目标 slot 编号
        - layer_id: 当前是第几层

        【实现】
        调用 minisgl.kernel.store_cache（CUDA kernel）做高效写入。
        内部相当于:
            k_view[out_loc] = k
            v_view[out_loc] = v
        但用 CUDA kernel 可以更好地处理 vectorized write，比 PyTorch 索引快。
        """
        # 延迟 import 避免循环依赖（kernel 模块依赖 torch）
        from minisgl.kernel import store_cache

        store_cache(
            # 把 [num_pages, page_size, ...] view 成 [num_pages*page_size, ...]
            # 这样 out_loc 里的"slot 编号"可以直接作为第 0 维索引
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
