"""
========================================================================
文件名: scheduler/table.py
所属模块: 调度器 - "页表行" 资源分配器
========================================================================

【这个文件是做什么的 - 一句话总结】
这个文件管理一张大表（page_table）的"行号资源"——每个并发请求占
一行，本文件就是 "拿一行、还一行" 的简单分配器。可以想成 GPU 显存
里"请求席位"的座位管理员。

【为什么需要这个文件 / 这个模块存在的原因】
GPU 端有一张全局二维张量 page_table：
   page_table[i, j] = 第 i 号请求的第 j 个 token 的 KV 存在哪个 slot
i 这一维（行号）就是"请求 id"。运行时要并发处理几十~几百个请求，
就要一个简单的管理器来分配/回收这些行号。

【这个文件在整个推理流程中的位置】
   PrefillManager 想接收一个新请求
     ↓
   ★ TableManager.allocate() → 拿到一个空闲行号 table_idx ★
     ↓
   把 table_idx 写到 Req 对象里
     ↓ 请求生成完毕（或被 abort）
   ★ TableManager.free(table_idx) → 行号回收 ★

【核心概念速览】

- page_table:
    一个 GPU 上的大二维张量，形状 [max_running_reqs, max_seq_len]。
    第 i 行第 j 列存的整数表示"第 i 号请求的第 j 个 token 的 KV 在
    KV cache pool 的哪个 slot"。注意力 kernel 靠它来寻址 KV。

- token_pool:
    形状和 page_table 一模一样，但存的是 input_ids（token 的整数 id）
    而不是 KV slot 编号。
    为什么需要：CUDA Graph 录制时要预先知道"输入 token 张量"在哪，
    而我们希望调度器只用做 host→device 拷贝（写到 token_pool 的对应
    位置），不用每步重新 alloc 一个新 tensor 喂 kernel。所以 token_pool
    起到"输入缓冲区"作用。

- max_running_reqs:
    系统支持的最大并发请求数（页表行数）。如所有行都占满，新请求
    必须等待。这是 SchedulerConfig 里的一个上限。

【关键设计决策】

1. 用 list 的 pop/append 实现空闲行管理：
   规模小（几十到几百），简单列表足够；如果是几万级别才需要更复杂
   的数据结构。

2. token_pool 初始化为 0：
   "dummy 请求"（CUDA Graph padding 用的占位）会从 token_pool 读
   input_ids。必须保证那部分是合法值（token id 0 通常对应 <pad> 或
   一个无害的 token），否则模型可能算出 NaN 或越界。
"""

import torch


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：TableManager（读代码之前请先读完这段）███
# ════════════════════════════════════════════════════════════════════
#
# 一、这个类要解决什么问题？
#
#   page_table 上有 max_running_reqs 行（比如 64 行）。
#   新请求来了要占一行，请求结束要还行。如果没有任何管理：
#     - 不知道哪行空着；
#     - 容易"行号冲突"——两个请求同时拿到同一行 → 数据互相覆盖。
#
#   TableManager 就是这张表的"前台"：发牌、收牌、记着剩多少张。
#
# 二、用具体例子走一遍
#
#   假设 max_running_reqs = 4：
#     初始化:
#       _free_slots = [0, 1, 2, 3]   ← 4 个空闲行号
#       page_table = 形状 [4, max_seq_len] 的 GPU 张量
#       token_pool = 形状 [4, max_seq_len] 的 GPU 张量，全 0
#
#     【请求 A 进来】
#       allocate() → pop 出 3 (列表尾)，返回 3
#       _free_slots = [0, 1, 2]
#       Req_A.table_idx = 3
#       → 之后 page_table[3, :] 就是请求 A 的 KV 索引
#
#     【请求 B 进来】
#       allocate() → 返回 2
#       _free_slots = [0, 1]
#
#     【请求 C、D 也进来】
#       _free_slots = []
#
#     【请求 E 想进来】
#       available_size = 0 → 调度器知道"满了"，让 E 等
#
#     【请求 A 生成完，释放】
#       free(3) → _free_slots = [3]
#       这一行可以分配给下一个新请求 F
#
# 三、为什么 pop() 默认从尾部？
#
#   pop() 不带索引就是从尾部弹，O(1)。
#   append() 也是 O(1)。
#   不在意"哪一行先用哪一行"，所以用栈式 LIFO 最简单。
#
# 四、为什么 token_pool 要和 page_table 同形状？
#
#   两者按"请求 id × 序列位置"二维索引，是天然平行的：
#     page_table[req_idx, pos] = 该 token KV 存哪个 slot
#     token_pool[req_idx, pos] = 该 token 的 id 是多少
#   一一对应，方便 CUDA Graph 复用同一套索引访问。
#
# ════════════════════════════════════════════════════════════════════
class TableManager:
    """
    【类名】TableManager（页表行号管理器）
    【一句话描述】管理 page_table 的行号资源：分配、回收、查询剩余。
    【生活类比】停车场入口的"剩余车位牌"+ 取车票/还车票流程。

    【它和其他类的关系】
    - 由 Scheduler.__init__ 创建并持有；
    - PrefillManager 在准备接收新请求前调用 allocate()；
    - Scheduler 在请求结束时调用 free()；
    - 字段 page_table 和 token_pool 也开放给其他模块直接读写
      （是设计上的"共享内存"）。
    """

    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        """
        【功能】初始化分配器，把所有行号都标记为空闲。

        【参数】
        - max_running_reqs (int): 系统支持的最大并发请求数（即 page_table 行数）。
            例子: 64 表示最多 64 个请求可同时在 GPU 上进行。
        - page_table (torch.Tensor):
            已经在 GPU 上创建好的页表张量，形状 [max_running_reqs, max_seq_len]。
            本类只持有引用，不负责创建/销毁。

        【关键操作】
        1. _free_slots = [0, 1, ..., max_running_reqs-1]：所有行号都空闲。
        2. token_pool 与 page_table 同形同设备，dtype=int32（token id 整型）。
           初值全 0：dummy 请求读到的也是合法 token id。
        """
        # 总行数（也是并发上限），只在内部用做合法性参考
        self._max_running_reqs = max_running_reqs

        # 空闲行号列表，初始全部可用
        # 用 list 而不是 set/queue：规模小，pop/append 的 O(1) 足够快
        self._free_slots = list(range(max_running_reqs))

        # 持有 page_table 引用——给外部直接读写
        # ⚠️ 注意：本类不会自己改写 page_table 的内容（行号分配只是"标记"
        # 哪行属于谁，至于行里写什么 KV slot 由 cache_manager 负责）。
        self.page_table = page_table

        # token_pool：和 page_table 形状一样的"token id 缓冲区"
        # 含义：每个请求每个位置上的 token id 暂存在这里，前向时 kernel
        #       读它做输入。
        # ⚠️ dummy 请求（CUDA Graph padding）也读这个 pool，所以必须用全 0
        # 初始化以保证读到合法值（不是未初始化的随机内存）。
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        """
        【功能】当前还有多少空闲行号可分配。
        【返回】整数，范围 0..max_running_reqs
        【典型用法】调度器在尝试接收新请求前调用：
                     if table_manager.available_size > 0: ...
        """
        return len(self._free_slots)

    def allocate(self) -> int:
        """
        【功能】拿走一个空闲行号供新请求使用。

        【返回】一个整数 table_idx，调用者负责把它写入 Req 对象。

        【副作用】_free_slots 减少一个元素。

        【调用前必须确认 available_size > 0】，否则 pop() 会抛 IndexError。
        当前实现没有做防护（性能优先），调度器自己保证不在空时调用。
        """
        return self._free_slots.pop()  # 从尾部弹，O(1)

    def free(self, slot: int) -> None:
        """
        【功能】把一个不再使用的行号还回空闲池。

        【参数】slot (int): 要回收的行号（之前通过 allocate 拿到的）。

        【调用时机】请求生成完毕、被 abort、或迁移走时由 Scheduler 调用。

        【注意】不会清空 page_table[slot] 的内容——反正下次分配出去时
                会被新请求覆盖，提前清是浪费。
        """
        self._free_slots.append(slot)
