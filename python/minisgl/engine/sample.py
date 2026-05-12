"""
========================================================================
文件名: engine/sample.py
所属模块: 引擎 - 采样器（从 logits 选出下一个 token）
========================================================================

【这个文件是做什么的 - 一句话总结】
模型每步前向算出 logits（每个 token 的"打分向量"），采样器根据每个
请求的 SamplingParams（temperature/top_k/top_p）从中"抽"出下一个 token。

【为什么需要这个文件 / 这个模块存在的原因】
- 不同请求有不同采样策略 → 必须支持 batch 内逐请求异化处理；
- 贪心和随机采样性能差距大 → 提供贪心快路径；
- top-k / top-p 采样涉及排序、归一化、随机数生成等专用算法 → 用
  flashinfer 的 CUDA kernel 加速。

【核心概念速览】

- logits:
    模型最后一层输出的"未归一化对数概率"，形状 [batch_size, vocab_size]。
    每个 token 一个分数，分数越高越可能被选中。

- softmax + temperature:
    把 logits 转成概率：probs = softmax(logits / temperature)。
    temperature 越大分布越平坦；越小越尖锐。

- top-k sampling:
    只在概率前 k 个 token 里采样，避免选到尾巴上的低概率词。

- top-p (nucleus) sampling:
    把概率从大到小排序累加，累计到 p 时停下，只在这部分采样。

- 贪心采样:
    直接 argmax 取概率最大那个，输出确定可复现。

【关键设计决策】

1. 把 batch 内所有请求的 sampling params 收集成 GPU 张量（prepare 方法）：
   而不是在 sample 时逐个判断——这样 kernel 可以一次性向量化处理。

2. 用 flashinfer.sampling：
   它实现了高效的 top-k/top-p 采样 CUDA kernel，避免 PyTorch 通用算子
   带来的开销。

3. 全 batch greedy 走 argmax 快路径：
   如果所有请求都是贪心，跳过 flashinfer 调用直接 argmax，更快。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from minisgl.core import Batch


# ════════════════════════════════════════════════════════════════════
# BatchSamplingArgs - 一批请求的采样参数（已转 GPU 张量形式）
# ────────────────────────────────────────────────────────────────────
# - temperatures: None 表示全 batch 贪心，可走快路径；否则是长度=batch_size 的张量
# - top_k:        None 表示不启用 top-k；否则张量
# - top_p:        None 表示不启用 top-p；否则张量
# ════════════════════════════════════════════════════════════════════
@dataclass
class BatchSamplingArgs:
    """打包 batch 内每个请求的采样参数为 GPU 张量。"""
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    【功能】把 Python list 异步拷到 GPU。
    【实现技巧】先在 pinned memory 创建，再 .to(device, non_blocking=True)
                让 host→device 拷贝异步进行，不阻塞主线程。
    """
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    """
    【功能】调 flashinfer 的采样 kernel，根据 (top_k, top_p) 的组合走不同路径。

    【路径选择】
    - 都 None: 纯 temperature 采样（softmax + 普通抽样）
    - 只 top_k: top-k 采样
    - 只 top_p: top-p 采样
    - 都有:   top-k → top-p 复合采样

    【enable_pdl】SM90+（H100）GPU 可启用 Producer-Consumer Data Layout 优化，加速 softmax。
    """
    import flashinfer.sampling as sampling

    # logits → 概率分布（含 temperature 缩放）
    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    # 都启用：组合采样
    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


# ════════════════════════════════════════════════════════════════════
# Sampler 主类
# ────────────────────────────────────────────────────────────────────
# 两个核心方法：
# - prepare(batch): 把每个请求的 sampling params 收集成 GPU 张量
# - sample(logits, args): 用 GPU 张量执行采样
#
# 拆开是因为：
#   - prepare 是 host 侧操作（构 tensor、host→device 拷贝），可以提前做；
#   - sample 是 GPU 上的核心计算，可以重叠执行。
# ════════════════════════════════════════════════════════════════════
@dataclass
class Sampler:
    """采样器：把 logits 转成下一个 token id。"""

    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        """
        【功能】把 batch 内每个请求的 SamplingParams 收集成 GPU 张量。
        【返回】BatchSamplingArgs（如果全是贪心，temperatures=None）

        【实现细节】
        - 全贪心 → 直接返回 temperatures=None 走快路径；
        - 否则：
          - 把 temperature/top_k/top_p 转成 GPU 张量；
          - 对贪心的请求把 temperature 设为很小值（避免除 0）；
          - top_k=-1（不启用）映射到 vocab_size 表示"全部参与"；
          - 如果所有 top_k 都是 vocab_size（即没有任何请求启用 top_k），
            就把 top_k 整个设为 None，省一次 kernel 调用；
          - top_p 同理。
        """
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            # 所有都是贪心 → 走 argmax 快路径
            return BatchSamplingArgs(temperatures=None)

        # 防止数值问题：temperature 和 top_p 不能取 0
        MIN_P = MIN_T = 1e-6
        # 贪心请求设很小温度（数学上等价于 argmax）
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        # top_k=-1 → vocab_size（即"全部 token 都候选"）
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        # 只有至少一个请求真的启用 top_k 才传张量
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        """
        【功能】执行采样：从 logits 选出 next_token。
        【参数】
        - logits: [batch_size, vocab_size] 浮点张量
        - args: prepare 返回的 BatchSamplingArgs
        【返回】[batch_size] 整数张量，每元素是采样到的 token id

        【路径】
        - 全贪心 → torch.argmax；
        - 否则 → 调 sample_impl 走 flashinfer。
        """
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
