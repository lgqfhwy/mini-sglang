"""
========================================================================
文件名: engine/graph.py
所属模块: 引擎 - CUDA Graph 捕获与回放
========================================================================

【这个文件是做什么的 - 一句话总结】
这个文件用 CUDA Graph 把"一次 decode 前向"录制成可以高速重放的"剧本"。
此后做 decode 时不再发起几百次 kernel 启动，而是 graph.replay() 一次
搞定，单步延迟从几毫秒降到几百微秒。

【为什么需要 CUDA Graph】
CUDA kernel 启动有 host 侧固定开销（约 5-10 微秒）。Transformer 一层
有几十个 kernel，32 层就是 1000+ kernel 启动 → host 侧光启动开销就
要几毫秒，占整个 decode 步骤的大头。
CUDA Graph 把所有 kernel 启动"录"下来，回放时整个图作为单个 GPU
任务调度，几乎消除 host 侧开销。

【核心概念速览】

- CUDA Graph:
    CUDA 提供的"录制 + 重放"机制。一次性录下整段 GPU 工作，之后
    cuda graph.replay() 即可重放。要求形状固定（重放期间张量 shape
    不能变），所以本类把多种 batch size 各预录一份图。

- batch padding:
    本类对外提供 pad_batch(batch)——把真实 batch size 补齐到最近的
    "已录制档位"（比如真实 7 个请求 → 补成 8）。补出来的是 dummy_req
    占位。

- dummy_req:
    一个永远存在的伪请求，用来填补 padding 位。它有自己的 table_idx
    指向"dummy page"，那一页里全是 token id=0 的数据，跑前向时安全无害。

【关键设计决策】

1. 只为 decode 录 graph（不为 prefill）：
   decode 的每步都是"每请求 1 token"——形状容易归一到固定档位；
   而 prefill 每个请求长度不同（extend_len 从 1 到几千都有可能），
   做 graph 性价比不高（要录的形状组合爆炸）。

2. 共用一个 graph pool（pool = first_graph.pool()）：
   不同 batch size 的图共享底层显存池，节约显存。

3. 录制时 forward 两遍：
   第一遍预热（warmup），第二遍才真的录入 graph。这是 CUDA Graph 的
   一个常见 pitfall——首次 forward 可能触发一些 lazy 初始化（cuBLAS
   handle 等），不预热的话这些 op 会被错误录入。
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.models import BaseLLMModel

logger = init_logger(__name__)


# ════════════════════════════════════════════════════════════════════
# GraphCaptureBuffer - CUDA Graph 录制/回放用的固定张量缓冲区
# ────────────────────────────────────────────────────────────────────
# CUDA Graph 要求张量"地址固定"（每次 replay 读写同一块显存）。
# 所以我们预先 alloc 几个最大尺寸的张量（按 max_graph_bs 大小）：
#   - input_ids: 每次 replay 前把真实 batch 的 input_ids 拷进来
#   - out_loc:   同上
#   - positions: 同上
#   - logits:    输出位置（replay 后从这里读结果）
# 每次 replay 用 set_batch / copy_from 让 batch 字段指向这块固定显存。
# ════════════════════════════════════════════════════════════════════
@dataclass
class GraphCaptureBuffer:
    """CUDA Graph 录制/回放用的固定地址张量。"""
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        """按"最大可能 batch size"预分配各张量。"""
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        """
        【功能】把 batch 的输入字段指向 buffer 切片（这样 model.forward
                读 batch.input_ids 时读的就是 buffer 的固定地址）。
        【用途】录制 graph 前调用一次，让 graph 学会"读这些固定地址"。
        """
        _slice = slice(batch.padded_size)
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]

    def copy_from(self, batch: Batch) -> None:
        """
        【功能】replay 前把真实 batch 的内容拷进固定 buffer。
        【为什么】graph 录的时候硬编码读 buffer 地址，replay 时输入必须
                  在那里——所以每次 replay 都先把当前 batch 数据拷过去。
        """
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    """
    【功能】决定要录哪些 batch size 的 CUDA Graph。

    【策略】
    - 用户显式给了 cuda_graph_bs → 直接用；
    - 否则按显存情况选 cuda_graph_max_bs：
      - > 80GB（H200）: 256
      - 否则: 160
    - 生成 [1, 2, 4, 8, 16, 24, 32, ...] 这种序列
    """
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))


def mem_GB(size: int) -> str:
    """字节数转 GB 字符串，用于日志输出。"""
    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    """查询当前 GPU 上的剩余显存（字节）。"""
    return torch.cuda.mem_get_info(device)[0]


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：GraphRunner ███
# ════════════════════════════════════════════════════════════════════
#
# 一、它做什么？
#   把每种支持的 decode batch size（如 1/2/4/8/16/.../256）各录制一份
#   CUDA Graph。运行时根据真实 batch size 选最接近的档位（pad_batch）
#   然后 replay。
#
# 二、典型流程
#   __init__:
#     1. 决定要录哪些 bs（如 [1,2,4,8,16,24,32]）
#     2. 准备 buffer（按最大 bs 32 预分配）
#     3. 从大到小（先 32 后 1）依次：
#        a. 创建 CUDAGraph()
#        b. 用 dummy_req 凑 bs 个请求构造 Batch
#        c. attn_backend.prepare_for_capture（让注意力后端准备好录制需要的元数据）
#        d. buffer.set_batch（让 batch 字段指向固定地址）
#        e. forward 一次预热（不录入）
#        f. with cuda.graph(graph): forward 一次（这次录入）
#        g. 把 graph 存到 graph_map[bs]
#   运行时:
#     - pad_batch(batch): 真实 size → 找最近档位 → padded_reqs 补齐
#     - can_use_cuda_graph: decode 且 size <= max_graph_bs
#     - replay(batch): copy_from → graph.replay() → 从 buffer.logits 读结果
#
# ════════════════════════════════════════════════════════════════════
class GraphRunner:
    """【类名】GraphRunner - 多档位 CUDA Graph 的录制 + 回放器。"""

    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        # 最大档位（决定 buffer 的大小、pad_batch 的上界）
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        # 排好序的档位列表，pad_batch 用它从小到大找第一个 >= 当前 size 的档位
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        """
        【功能】真正录制 CUDA Graph 的主循环。

        【为什么从大到小录制】
        - 大 bs 用的显存最多——先录它能知道极限够不够；
        - 共用同一个 pool 减少显存碎片。

        【主要步骤】见类逻辑全景解说。
        """
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        # 注意力后端预先做录制需要的初始化（如预分配 metadata 张量）
        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        # 同步 + 清理碎片，让显存测量更准
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        # 为最大档位 alloc 一份 buffer，所有档位都用它的切片
        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)

        # 进度条（只在 rank0 显示）
        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            # 用 dummy_req 凑出 bs 个请求构造一个假的 decode batch
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            with get_global_ctx().forward_batch(batch):
                # 预热前向：让 cuBLAS 等 lazy 初始化都跑完
                self.buffer.logits[:bs] = model.forward()
                # 在 cuda.graph 上下文里真正录入
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
            if pool is None:
                # 第一个图建立后，把 pool 拿出来供后续图复用
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph

        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        """
        【功能】本 batch 能不能走 CUDA Graph 快路径？
        【条件】是 decode 阶段，且真实 batch size 不超过最大档位。
        """
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        """
        【功能】回放对应档位的 graph，返回 logits（已切到真实 batch_size 长度）。
        【步骤】
        1. copy_from: 把 batch 输入拷到 buffer；
        2. 查 graph_map[padded_size] 拿到对应 graph；
        3. attn_backend.prepare_for_replay（设置注意力元数据）；
        4. graph.replay() 真正跑；
        5. 返回 buffer.logits[:真实size]。
        """
        assert self.can_use_cuda_graph(batch)
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.buffer.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        """
        【功能】把真实 batch 补齐到最近的 graph 档位。

        【例子】
        - 真实 size=7, graph_bs_list=[1,2,4,8,16,32]
        - 选 8 作为 padded_size
        - padded_reqs = batch.reqs + [dummy] * 1
        """
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        """
        【功能】析构所有 CUDA Graph，释放显存。
        【⚠️ 重要】必须在销毁 NCCL 资源之前调用，否则程序会挂起。
        """
        del self.graph_map
        gc.collect()
