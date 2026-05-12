"""
========================================================================
文件名: core.py
所属模块: minisgl 项目根模块（最核心、最底层的数据结构定义文件）
========================================================================

【这个文件是做什么的 - 一句话总结】
这个文件就像整个 LLM 推理系统的"通用零件库"——它定义了所有其他模块都
要使用的最基础的数据类型（请求、批次、采样参数、全局上下文）。可以把
它理解为整栋大楼的"地基蓝图"：地基不复杂，但所有楼层都依赖它。

【为什么需要这个文件 / 这个模块存在的原因】
LLM 推理系统涉及很多模块（调度器、引擎、KV 缓存、注意力后端、采样器
等），它们之间需要传递"一个请求"或"一批请求"这样的数据。如果每个模块
都自己定义一遍 Request 类，就会重复且容易不一致。
所以我们把所有人都要用的数据结构抽离出来放在这里，保证：
  1. 所有模块对"什么是一个请求"有统一理解；
  2. 修改一处即可影响全局，避免不同步。

【这个文件在整个推理流程中的位置】
   用户发请求(HTTP)
     ↓
   API Server (server/api_server.py)        ← 收到 HTTP 请求
     ↓
   Tokenizer 进程 (tokenizer/server.py)     ← 文本切成 token id
     ↓
   ★ 构造 Req 对象（本文件定义） ★
     ↓
   Scheduler 进程 (scheduler/scheduler.py)  ← 把多个 Req 打成 Batch（本文件定义）
     ↓
   Engine (engine/engine.py)               ← 把 Batch 交给 GPU 跑前向
     ↓
   Sampler                                  ← 用 SamplingParams（本文件定义）采样下个 token
     ↓
   Detokenizer                              ← token id 还原成文本
     ↓
   返回结果给用户

   ★本文件★ 几乎在每一步都被读取或修改。

【核心概念速览 - 读这个文件之前你需要了解的背景知识】

- token（词元）:
    LLM 不能直接处理文字，它先要把文字切成"token"。例如英文
    "Hello world" 可能被切成 ["Hello", " world"] 两个 token；
    中文"今天天气好"可能切成 ["今天", "天气", "好"] 等。每个 token
    会被映射到一个整数 id（例如 "Hello" → 15496）。
    类比：就像快递分拣时，每件包裹都贴一个数字编号方便扫描。

- prompt（提示词）/ input_ids:
    用户输入的那句话（被切分后的 token id 列表）。
    例如用户输入 "你好"，可能变成 input_ids = [123, 456]。

- prefill 阶段:
    模型第一次"读懂"用户的全部输入。GPU 需要一次性把整段 prompt
    跑一遍前向计算，生成对应的 KV Cache（见下文）。
    类比：你做阅读理解前，要先把文章读一遍。
    特点：一次处理很多 token（整个 prompt），计算密集。

- decode 阶段:
    prefill 完成后，模型开始一个 token 一个 token 地"写答案"。
    每生成 1 个 token 就要做一次前向计算。
    类比：写作文时一个字一个字地写。
    特点：每次只算 1 个 token，访存密集（瓶颈在显存带宽）。

- KV Cache（键值缓存）:
    Transformer 的注意力机制需要每个 token 看到它前面的所有 token。
    如果每生成一个新 token 都要重新算前面所有 token 的 K（Key）和
    V（Value），开销巨大。因此把已经算过的 K、V 缓存在显存里，下次
    直接读用，这就是 KV Cache。
    类比：解数学题时把中间结果记下来，下一步直接用，不用重算。
    KV Cache 占用大量显存，是 LLM 推理的最大资源瓶颈之一。

- batch / batching（批处理）:
    把多个用户的请求"打包"一次性扔给 GPU 处理。GPU 擅长并行，单独
    跑一个请求和打包跑 16 个请求耗时差不多，所以批处理能显著提高
    吞吐量。
    类比：餐厅一次给 5 桌上菜，比跑 5 趟单独上菜效率高。

- sampling（采样）:
    模型每一步前向输出的是"下一个 token 是某个词的概率分布"（叫
    logits）。从这个分布里选出一个具体 token 的过程叫采样。
    - 贪心（greedy）: 直接选概率最高那个。
    - top-k: 只在概率前 k 个里选。
    - top-p (nucleus): 累积概率达到 p 的最小集合里选。
    - temperature: 温度越高分布越平坦，输出越随机。

- 页表 (page_table) / table_idx:
    Mini-SGLang 用类似操作系统"页表"的方式管理 KV Cache 在显存里
    的位置。每个请求占用一行页表（一个 table_idx），这一行记录
    "我占了 KV Cache 的哪几页（哪几个 slot）"。
    类比：图书馆借书证：每个读者一张证（table_idx），上面写他
    借了哪几本书（KV Cache 的哪些 slot）。

【关键设计决策】

1. 使用 @dataclass 而不是手写 __init__：
   减少样板代码，让字段一目了然，符合"代码即文档"的理念。

2. SamplingParams 默认 temperature=0.0（贪心）：
   贪心采样是确定性的，调试和压测时输出可复现，对开发更友好。

3. Req 中的 input_ids 必须是 CPU tensor（见 __post_init__ 的 assert）：
   Req 在调度器进程里被反复读写、追加（每生成一个 token 都 append
   一次），而调度器主要跑在 CPU 上；KV Cache 才在 GPU 上。
   把 input_ids 留在 CPU 避免频繁 GPU↔CPU 拷贝。

4. 全局只有一个 Context，用模块级单例 _GLOBAL_CTX：
   注意力后端、KV Cache、当前 batch 等是"GPU 端的全局状态"，
   到处通过函数参数传递太繁琐，所以做成单例，通过
   get_global_ctx() 在需要的地方拿到。
   缺点：单例不利于单进程内跑多个模型实例（但本项目不需要）。

5. Batch 的多个字段标注 init=False：
   Batch 创建时只填 reqs 和 phase，其它字段（input_ids、positions、
   out_loc、attn_metadata 等）由调度器和注意力后端**后续逐步填充**。
   这是一种"对象逐步建造"模式：先有骨架，再被不同模块加工。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch

# TYPE_CHECKING 是 Python 的一个特殊常量：
# - 静态类型检查工具（如 mypy、pyright）看到它时，值为 True，会导入下面的类型；
# - 实际运行时，值为 False，下面的 import 不会真的执行。
# 这样做的目的：
#   1. 避免循环依赖（attention/kvcache/moe 模块反过来也会 import core.py）；
#   2. 加快启动速度（运行时不必加载这些大模块）。
# 在下面的类型注解中，凡是用到这些类型的地方，都要写成字符串形式（向前引用）
# 或借助 from __future__ import annotations（本文件最顶端已 import）。
if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from minisgl.moe import BaseMoeBackend


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：SamplingParams（读代码之前请先读完这段）███
# ════════════════════════════════════════════════════════════════════
#
# 一、这个类要解决什么问题？
#
#   模型每一步前向计算最终输出的是一个长度为 vocab_size（词表大小，
#   通常 30000~150000）的概率分布——告诉你下一个 token 应该是哪个
#   词的概率。但具体"从这个分布里挑哪个"取决于用户/任务的需求：
#     - 要"确定、稳定"的回答 → 选概率最高那个（贪心）；
#     - 要"有创意、多样化"的回答 → 引入随机性、温度等。
#
#   SamplingParams 就是装这些用户偏好的小盒子——一个请求一个盒子。
#
# 二、字段含义（先看例子再看代码）
#
#   假设模型预测下一个 token 的概率（仅取前 5 个，实际是几万维）：
#     "猫" → 0.40    "狗" → 0.25    "鸟" → 0.15    "鱼" → 0.10    "牛" → 0.05
#
#   场景1: 用 SamplingParams(temperature=0.0)
#     → is_greedy = True
#     → 直接选概率最高的 "猫"
#     → 每次运行结果都相同（确定性）
#
#   场景2: 用 SamplingParams(temperature=1.0, top_k=3, top_p=1.0)
#     → 把分布除以 temperature 后做 softmax（temperature=1 等于不变）
#     → top_k=3 → 只保留前 3 个 "猫/狗/鸟"，重新归一化
#       变成 "猫"=0.5, "狗"=0.3125, "鸟"=0.1875
#     → 从这个新分布里按概率随机抽一个
#     → 这次可能选到 "狗"，下次可能选到 "猫"
#
#   场景3: 用 SamplingParams(temperature=2.0)（高温）
#     → logits / 2.0 → 分布变得更平坦
#     → 原本 "猫" 0.40 → 变成约 0.30，差距缩小
#     → 输出更随机、更有"创造性"
#
#   场景4: max_tokens=1024, ignore_eos=False
#     → 最多生成 1024 个新 token 就停止
#     → 中途如果模型生成了 EOS（结束符），也立刻停止
#     → 如果 ignore_eos=True，就忽略 EOS 一直生成直到 max_tokens
#
# 三、为什么 is_greedy 这么定义？
#
#   只要满足 (temperature<=0 OR top_k==1) AND top_p==1.0，输出就是
#   确定的：
#     - temperature<=0：除零退化成 argmax（实现上特殊处理）；
#     - top_k==1：只允许第一名进入候选，没得选；
#     - top_p==1.0：不截断，但前两个条件已经决定了确定性。
#   贪心的好处：可以跳过 softmax 之类的计算，直接 argmax 提速。
#
# ════════════════════════════════════════════════════════════════════
@dataclass
class SamplingParams:
    """
    【类名】SamplingParams（采样参数）
    【一句话描述】保存"如何从模型输出的概率分布里挑下一个 token"的用户偏好。
    【生活类比】如果模型给你一份选项菜单（每道菜的诱人程度），
                这个类就是你的"点菜规则"：是要最爱的那道、还是随机选、
                是不是只看前 3 道等。

    【它为什么存在】
    每个请求可以有不同的采样需求（GPT 接口里的 temperature/top_k/top_p
    就是这些参数）。把它们打包成一个对象，能整齐地附在 Req 里传递。

    【它和其他类的关系】
    - 由 API server 根据用户传入的 JSON 创建；
    - 作为 Req 的一个字段长期持有；
    - 被采样器 (engine/sample.py) 读取以决定如何采样。

    【重要属性】
    - temperature: 温度，控制随机性强弱（0 = 完全确定）
    - top_k: 只在概率最高的前 k 个 token 里选；-1 表示不启用
    - top_p: 累积概率达到 p 的最小集合里选；1.0 表示不启用
    - ignore_eos: 是否忽略模型生成的"结束符"
    - max_tokens: 最多生成多少个新 token 就停止
    """

    # ----------------------------------------------------------------
    # 字段: temperature
    # 类型: float
    # 含义: 采样温度，控制输出的"随机性"。
    # 详细解释:
    #   模型输出 logits（原始打分）后，会做 softmax 变成概率。
    #   在 softmax 前会先 logits / temperature：
    #     - temperature 越大，分布越平坦 → 越随机；
    #     - temperature 越小，分布越尖锐 → 越确定；
    #     - temperature == 0 → 退化为贪心（直接选最大）。
    # 默认值 0.0：默认贪心，输出可复现，方便测试。
    # ----------------------------------------------------------------
    temperature: float = 0.0

    # ----------------------------------------------------------------
    # 字段: top_k
    # 类型: int
    # 含义: 只在概率前 k 个候选里采样；-1 表示不启用。
    # 例子: top_k=5 表示只看 vocab 中概率最高的 5 个 token。
    # ----------------------------------------------------------------
    top_k: int = -1

    # ----------------------------------------------------------------
    # 字段: top_p
    # 类型: float
    # 含义: "核采样"——把概率从大到小累加，凑够 top_p 就停，
    #       只在这个最小集合里采样。1.0 表示不启用。
    # 例子: top_p=0.9，token 排序后概率累加超过 0.9 的那一刻停下，
    #       只在累加进来的这几个里采。
    # ----------------------------------------------------------------
    top_p: float = 1.0

    # ----------------------------------------------------------------
    # 字段: ignore_eos
    # 类型: bool
    # 含义: 遇到结束符（EOS, End-Of-Sequence）时是否停止生成。
    #   - False（默认）: 一旦采到 EOS，立即停止——这是正常对话模式。
    #   - True: 忽略 EOS，继续生成直到 max_tokens——常用于做基准测试，
    #     需要每个请求严格生成指定长度。
    # ----------------------------------------------------------------
    ignore_eos: bool = False

    # ----------------------------------------------------------------
    # 字段: max_tokens
    # 类型: int
    # 含义: 本次请求最多生成多少个新 token 后停下。
    # 为什么需要：防止模型"无限输出"耗尽显存或卡死调度器。
    # 默认 1024：对大部分对话场景够用。
    # ----------------------------------------------------------------
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        """
        【功能】判断本次采样是否"贪心"（输出确定性、无随机性）。

        【通俗解释】
        如果用户既没要随机性（temperature 极低 / top_k 只取第一名）
        又没要核采样截断（top_p=1.0），那本次就是贪心模式——
        每次输入相同，输出也相同。

        【为什么要这个属性】
        贪心模式可以走快路径：直接 argmax 拿最大概率的 token，
        不必做 softmax、不必生成随机数，速度更快。

        【返回值】True=贪心；False=随机采样
        """
        # 条件 1：temperature<=0 或 top_k==1，二者任一都使候选无随机性可言
        # 条件 2：top_p==1.0，没有 nucleus 截断（截断本身不引入随机，但
        #         结合其他参数也不影响这里的判断）
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：Req（读代码之前请先读完这段） ███
# ════════════════════════════════════════════════════════════════════
#
# 一、这个类要解决什么问题？
#
#   一个用户请求从进入系统到结束，需要被反复处理（先 prefill，再一轮
#   一轮 decode），中间要追踪：
#     - 用户原本输入了什么；
#     - 已经生成到第几个 token；
#     - KV Cache 已经算到哪了（哪些 token 的 K/V 已经在显存里）；
#     - 它在显存页表里占哪一行；
#     - 它配的采样规则；
#     - 它的唯一标识（用于返回结果给对的用户）。
#   Req 就是装这些"请求状态"的容器，调度器主要就是在操作一个个 Req。
#
# 二、关键字段的含义和它们之间的关系（用具体例子走一遍）
#
#   假设用户发来 prompt "今天天气真好"，分词后 5 个 token：
#     input_ids = [101, 234, 567, 890, 102]
#   用户希望最多生成 10 个新 token，max_tokens = 10。
#
#   【刚进入系统时】
#     input_ids = [101, 234, 567, 890, 102]   ← 共 5 个
#     cached_len = 0          ← 还没有任何 token 的 KV 进入显存
#     device_len = 5          ← 但我们打算让这 5 个 token 都跑前向（prefill）
#     output_len = 10         ← 还需要生成 10 个
#     max_device_len = 5 + 10 = 15   ← 一生中最长会有 15 个 token
#     table_idx = 3           ← 调度器给它分配了页表第 3 行
#
#   【prefill 完成后】
#     模型把这 5 个 token 的 K/V 都算好存进显存了
#     cached_len = 5          ← 这 5 个的 KV 已就绪
#     device_len = 6          ← 加上刚刚 prefill 完最后一步采样得到的下一个 token
#     input_ids 还是 5 个（host 侧），但 device_len 已经记 6
#
#     注意：device_len 总是 >= len(input_ids) - 中间有个时间窗口
#     KV 算完了但新 token 还没 append 到 input_ids（host 侧）。
#
#   【第一轮 decode 完成后（生成了 1 个新 token）】
#     append_host(new_token) 把新 token 加到 input_ids（host）
#     input_ids = [101, 234, 567, 890, 102, 新token]   ← 6 个
#     complete_one() 被调用:
#       cached_len = 6        ← 之前的 device_len 变成 cached_len
#       device_len = 7        ← 准备下一轮再算一个
#
#   【迭代到生成够 10 个 token 时】
#     device_len = 15 = max_device_len
#     remain_len = 0 → can_decode = False
#     调度器把它从 running 队列移除，释放页表那行
#
# 三、为什么要分 cached_len 和 device_len？
#
#   - cached_len: 已经在显存里有 KV Cache 的 token 数量（之前算完了的）
#   - device_len: 本轮目标：让多少 token 的 KV 都在显存里（包括本轮即将算的）
#   - extend_len = device_len - cached_len = 本轮需要新算多少个 token 的 KV
#
#   - 在 prefill 时：cached_len=0，device_len=prompt 长度，extend_len 很大；
#   - 在 decode 时：每轮 extend_len = 1（只算新生成的那一个）；
#   - 在 chunked prefill 时（prompt 很长，分段做 prefill）：
#       第一段：cached_len=0, device_len=100, extend=100
#       第二段：cached_len=100, device_len=200, extend=100
#       ...
#
# 四、append_host vs complete_one 的关系
#
#   - append_host(t): 把刚采样出的 token 加到 host 侧 input_ids；
#   - complete_one(): 通知 Req"上一轮 KV 计算成功了"，把 cached_len
#     推进到 device_len，并把 device_len 再+1 为下一轮做准备。
#   两个动作通常在 scheduler 的 decode 步骤里前后脚发生。
#
# ════════════════════════════════════════════════════════════════════
@dataclass(eq=False)
class Req:
    """
    【类名】Req（Request 的缩写，单个推理请求）
    【一句话描述】一个用户请求在整个生命周期里的"档案夹"——记录它的
                  输入、当前进度、显存占用位置、采样规则等所有状态。
    【生活类比】医院的病历卡：从挂号到出院都在更新，记着病人姓名
                （uid）、住几号床（table_idx）、检查到哪一步（cached_len /
                device_len）、医生开了什么治疗方案（sampling_params）。

    【它为什么存在】
    调度器要同时管理成百上千个请求，必须有一个统一对象来描述每个请求
    的状态。没有这个类，每个模块都会重复造轮子定义"什么是请求"。

    【它和其他类的关系】
    - 被调度器（scheduler/prefill.py、scheduler/decode.py）创建和持有；
    - 被打包进 Batch（见下面的 Batch 类）一起送给 GPU；
    - cache_handle 字段链接到 KV Cache 池里它占用的 slot；
    - table_idx 指向显存页表里它占的那一行。

    【设计说明】
    @dataclass(eq=False) 关掉 dataclass 默认的 __eq__：
    我们希望两个 Req 即使字段值相同也算"不同请求"（按对象身份比较），
    避免 dict/set 里发生意外冲突。
    """

    # ----------------------------------------------------------------
    # 变量: input_ids
    # 类型: torch.Tensor（CPU 上，1-D long 张量）
    # 含义: 该请求当前已知的全部 token id 序列（包括原始 prompt
    #       + 已经生成的若干新 token）。
    # 为什么放 CPU 而不是 GPU:
    #   1. input_ids 主要被调度器读写（CPU 程序），频繁追加新 token；
    #   2. GPU 真正需要算的只是当前这一步的"新增片段"，调度器会从
    #      input_ids 里切出片段拷到 GPU，开销可控。
    #   3. KV Cache（真正大头）已经在 GPU 上，input_ids 只是元数据。
    # 例: prompt 5 个 token + 已生成 3 个 → 这里就是长度为 8 的 tensor。
    # ----------------------------------------------------------------
    input_ids: torch.Tensor  # cpu tensor

    # ----------------------------------------------------------------
    # 变量: table_idx
    # 类型: int
    # 含义: 在全局"页表"（page_table）里，本请求占用的那一行的下标。
    # 详细解释:
    #   系统启动时分配一个固定大小的页表（例如最多支持 256 个并发请求，
    #   就是 256 行）。每个请求挑空闲的一行用，把它的 KV slot 索引
    #   写到这一行里。table_idx 就是"我占第几行"。
    # 例: table_idx=7 表示本请求占页表第 7 行（从 0 开始）。
    # ----------------------------------------------------------------
    table_idx: int

    # ----------------------------------------------------------------
    # 变量: cached_len
    # 类型: int
    # 含义: 已经把多少个 token 的 KV Cache 算好并存进了显存。
    # 例: cached_len=5 → 第 0~4 个 token 的 K/V 已在 KV pool 里。
    # 取值范围: 0 <= cached_len < device_len <= max_device_len
    # ----------------------------------------------------------------
    cached_len: int

    # ----------------------------------------------------------------
    # 变量: output_len
    # 类型: int
    # 含义: 用户希望（或允许）本请求**额外生成**多少个新 token。
    # 注意：这里是"输出长度"，不包括 prompt 长度。
    # 通常和 sampling_params.max_tokens 一致，但调度器初始化时
    # 可能根据剩余显存做一些 clip。
    # ----------------------------------------------------------------
    output_len: int

    # ----------------------------------------------------------------
    # 变量: uid
    # 类型: int
    # 含义: 用户请求的全局唯一标识。
    # 用途: 多个进程间（API server / tokenizer / scheduler / detokenizer）
    #       靠 uid 找到"是哪个用户的请求"，把结果回传给对的连接。
    # 类比: 快递单号——同一单贯穿整个流转。
    # ----------------------------------------------------------------
    uid: int

    # ----------------------------------------------------------------
    # 变量: sampling_params
    # 类型: SamplingParams
    # 含义: 用户为本请求指定的采样规则（见上面的类）。
    # ----------------------------------------------------------------
    sampling_params: SamplingParams

    # ----------------------------------------------------------------
    # 变量: cache_handle
    # 类型: BaseCacheHandle（在 kvcache 模块定义）
    # 含义: "本请求占了 KV Cache 池的哪些 slot"的句柄。
    # 详细解释:
    #   KV Cache 池把显存切成很多小块（slot），按需分配给请求。
    #   cache_handle 就是分配给本请求的那些 slot 的"凭证"——它内部
    #   记着 slot 编号列表。请求结束时，凭这个 handle 归还 slot。
    # 类比: 寄存柜钥匙——上面写着你存的是哪几个柜子。
    # ----------------------------------------------------------------
    cache_handle: BaseCacheHandle

    def __post_init__(self) -> None:
        """
        【功能】dataclass 的"二次构造"钩子——所有字段填好后自动调用。
        【作用】做合法性校验，并衍生出两个常用属性 device_len/max_device_len。
        """
        # input_ids 必须在 CPU 上（设计决策见类注释）。
        assert self.input_ids.is_cpu
        # ⚠️ 注意：device_len 不是写入字段，是动态属性。
        # 但这里直接 self.device_len = ... 仍然合法——Python 会把它
        # 当作普通实例属性，不参与 dataclass 自动生成的字段列表。
        #
        # 含义：本请求目前在 GPU 这边"已经/即将处理到第几个 token"。
        # 初始 = len(input_ids)：意味着"我们打算让 prompt 这些 token
        # 都进入 device 端"——也就是 prefill 阶段的目标长度。
        self.device_len = len(self.input_ids)
        # max_device_len = prompt 长度 + 计划生成长度
        #   = 这条请求最终一生中 device 端 token 数会达到的最大值
        # 用来判断是否还能继续 decode（见 can_decode）。
        self.max_device_len = len(self.input_ids) + self.output_len
        # 健全性校验：三个长度必须满足排序关系，否则上层逻辑错了。
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        """
        【功能】还剩多少个 token 可以生成（不算已生成的）。
        【公式】max_device_len - device_len
        【例子】prompt 5 个 + max_tokens 10 → max_device_len=15
                现在 device_len=8（生成了 3 个）→ remain_len=7
        """
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        """
        【功能】本轮需要新算多少个 token 的 KV（"扩展长度"）。
        【公式】device_len - cached_len
        【含义】
          - prefill 一次到位时：等于 prompt 长度（cached_len=0）；
          - decode 阶段：始终 = 1；
          - chunked prefill 阶段：等于本段的长度（通常几十到几百）。
        【为什么叫 "extend"】
          因为它是"在已缓存的基础上向前扩展多少 token"。
        """
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        """
        【功能】通知本请求"上一轮 device_len 范围内的 KV 已经算完
                并落盘到 KV Cache 了，请把状态向前推进一格"。
        【调用时机】每轮 decode 后、确认采样结果有效后被调度器调用。

        【内部逻辑】
        1. 把 cached_len 推到当前 device_len（之前在算的现在算好了）
        2. device_len += 1（为下一轮"再算 1 个 token 的 KV" 做准备）
        """
        self.cached_len = self.device_len  # 先前在算的 → 现在已缓存
        self.device_len += 1               # 下一轮目标推进 1 步

    def append_host(self, next_token: torch.Tensor) -> None:
        """
        【功能】把刚采样出的新 token 追加到 host 侧的 input_ids 末尾。
        【参数】next_token: 1-D 长度为 1 的 CPU long 张量
        【为什么】device 端只算 KV，但 host 侧也要知道"完整的 token 序列"
                  用于：1) 给 detokenizer 拼字；2) 后续可能要用到全文。
        """
        # torch.cat 会创建一个新的 tensor，input_ids 引用更新到新对象。
        # 频繁 cat 的开销在 host 是 O(N)，对单条请求来说可以接受；
        # 真正的性能瓶颈在 GPU 上。
        self.input_ids = torch.cat([self.input_ids, next_token])

    @property
    def can_decode(self) -> bool:
        """
        【功能】判断本请求是否还需要/还能继续生成下一个 token。
        【返回】True = 还能 decode；False = 该请求生成完毕，可以收尾。
        【判断标准】remain_len > 0 即还有"未生成的预算"。

        注意：这里只看"长度预算"。是否生成到 EOS（提前结束）由调度器
        在拿到采样结果后单独判断（结合 sampling_params.ignore_eos）。
        """
        return self.remain_len > 0

    def __repr__(self) -> str:
        """
        【功能】打印 Req 时输出便于调试的字符串。
        只列出最关心的字段：table_idx 和三个长度。
        """
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：Batch（读代码之前请先读完这段） ███
# ════════════════════════════════════════════════════════════════════
#
# 一、这个类要解决什么问题？
#
#   GPU 喜欢"一次干很多活"——同一种计算同时跑 N 份比串行跑 N 次只
#   慢一点点（吞吐量大幅提升）。LLM 推理就利用这一点：把几十/上百个
#   请求打包送进 GPU，称为"批处理"（batching）。Batch 就是这个"打包后
#   的工作单元"。
#
# 二、用具体例子走一遍
#
#   假设当前有 3 个请求要一起处理（都在 decode 阶段，每个只算 1 个新 token）：
#     Req A: input_ids=[10,20,30,40], device_len=5, cached_len=4
#            → 本轮 GPU 要算第 5 个 token 的 K/V（即位置 4 处）
#     Req B: input_ids=[7,8,9],       device_len=4, cached_len=3
#     Req C: input_ids=[100,200],     device_len=3, cached_len=2
#
#   组装成 Batch 后：
#     reqs = [A, B, C]
#     phase = "decode"
#     input_ids = [40, 9, 200]   ← 把每个请求"本轮要送给 GPU 的那一个 token"拼起来
#     positions = [4, 3, 2]      ← 每个 token 在自己序列里的位置编号
#                                   （供位置编码 RoPE 使用）
#     out_loc = [...]             ← 告诉 KV cache pool "把算出来的 K/V 存到哪个 slot"
#     attn_metadata = <由注意力后端填>
#                                ← 注意力机制需要的辅助张量（如 page table、
#                                   每个请求的 cached_len 等）
#     padded_reqs = [A, B, C, dummy, dummy, ...]
#                                ← 为了使用 CUDA Graph，常需要把 batch size
#                                  补齐到一个固定档位（如 4/8/16/32）。
#                                  这些填充进来的"占位请求"放在这里，真实请求仍在 reqs。
#
#   prefill 模式下：每个请求贡献多个 token（extend_len 个），input_ids 是
#   多个段拼起来的长向量。
#
# 三、为什么很多字段 init=False？
#
#   Batch 在不同模块手里要走好几道工序：
#     1. 调度器 (prefill.py/decode.py) 决定 reqs 和 phase → Batch 诞生；
#     2. 调度器再调用 _prepare_batch 填好 input_ids/positions/out_loc/padded_reqs；
#     3. 注意力后端读取上述字段，构造 attn_metadata；
#     4. 引擎拿着完全填满的 Batch 跑前向。
#   先空着是为了让每一步只关心自己负责的字段，分工明确。
#
# ════════════════════════════════════════════════════════════════════
@dataclass
class Batch:
    """
    【类名】Batch（一批请求）
    【一句话描述】把同一阶段（prefill 或 decode）的多个请求打包成一个
                  GPU 工作单元。
    【生活类比】食堂的"送餐推车"：把同一波要送的餐盘一起推过去，
                避免来回多次跑厨房。

    【它和其他类的关系】
    - 由 Scheduler 在每个调度步骤创建一次；
    - 内部装着多个 Req；
    - 被 attention backend 读取并填充 attn_metadata；
    - 最终被 Engine 拿去做前向计算；
    - 一步完成后被丢弃（每步都是新 Batch）。
    """

    # ----------------------------------------------------------------
    # 变量: reqs
    # 类型: List[Req]
    # 含义: 本批次里所有"真实"的请求（不含 padding 占位）。
    # ----------------------------------------------------------------
    reqs: List[Req]

    # ----------------------------------------------------------------
    # 变量: phase
    # 类型: Literal["prefill", "decode"]
    # 含义: 本批次处于哪个推理阶段。
    # 重要性: 注意力机制、KV Cache 写入方式、采样方式在两种阶段下
    #         实现细节不同，必须靠 phase 区分代码路径。
    # ----------------------------------------------------------------
    phase: Literal["prefill", "decode"]

    # ----------------------------------------------------------------
    # 下面这些字段都标注 init=False —— 创建 Batch 时不传，
    # 后续由调度器/注意力后端按顺序填上。
    # ----------------------------------------------------------------

    # 变量: input_ids（GPU 上的 1-D long 张量）
    # 含义: 本批次本轮要喂进模型的所有 token id 拼起来的长向量。
    # 例:
    #   decode 模式 3 个请求 → 长度 3 的张量；
    #   prefill 模式 2 个请求各 100 token → 长度 200 的张量。
    input_ids: torch.Tensor = field(init=False)

    # 变量: positions（GPU 上的 1-D long 张量）
    # 含义: 每个 token 在自己请求序列里的位置编号（0 起步）。
    # 用途: Transformer 的位置编码（RoPE/绝对位置编码）需要这个。
    # 例: decode 时 [4, 3, 2] —— 各自序列里这个新 token 是第 4/3/2 位。
    positions: torch.Tensor = field(init=False)

    # 变量: out_loc（GPU 上的 1-D long 张量）
    # 含义: 模型算出的每个 token 的 K/V，要写到 KV cache pool 的哪个 slot。
    # 详细: kv_cache_pool 提前给每个请求按需分配了 slot 编号，
    #       out_loc 就是"按本批次 token 顺序排列"的目标 slot 列表。
    out_loc: torch.Tensor = field(init=False)

    # 变量: padded_reqs
    # 类型: List[Req]
    # 含义: 为了 CUDA Graph 把 batch 大小补齐到固定档位（如 32），
    #       这里是补齐后的完整请求列表（reqs + 填充占位）。
    # 注意: 真实请求仍以 reqs 为准；padded_reqs 多出来的元素是 dummy。
    # ⚡ 性能关键: CUDA Graph 需要每次形状固定才能复用预编译的图，
    #              所以即使本步只有 7 个真实请求，也要补到 8/16/32 等档位。
    padded_reqs: List[Req] = field(init=False)

    # 变量: attn_metadata
    # 类型: BaseAttnMetadata（注意力后端实现各自的元数据类）
    # 含义: 注意力机制要用的所有辅助张量，比如：
    #       - 各请求的 cached_len、device_len；
    #       - 各请求在 page_table 中占的 slot 索引；
    #       - 起止位置偏移量等。
    # 由具体的注意力后端 (FlashAttention / FlashInfer 等) 自行填充。
    # 📄 算法背景: 现代 LLM 用 PagedAttention 思想——KV 像 OS 内存页
    #              一样分块管理，所以注意力 kernel 必须收到 page_table
    #              才能知道每个请求的 KV 散落在哪些显存块里。
    attn_metadata: BaseAttnMetadata = field(init=False)

    @property
    def is_prefill(self) -> bool:
        """是否是 prefill 阶段（处理 prompt）"""
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        """是否是 decode 阶段（生成新 token）"""
        return self.phase == "decode"

    @property
    def size(self) -> int:
        """本批次中真实请求的数量。"""
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        """补齐到 CUDA Graph 档位后的总数量（含 dummy 占位）。"""
        return len(self.padded_reqs)


# ════════════════════════════════════════════════════════════════════
# ███ 类逻辑全景解说：Context（读代码之前请先读完这段） ███
# ════════════════════════════════════════════════════════════════════
#
# 一、这个类要解决什么问题？
#
#   在 GPU 侧跑前向时，有些组件是"全局存在、各处都要用"的：
#     - KV Cache 池（kv_cache）
#     - 显存页表（page_table）
#     - 注意力后端（attn_backend）
#     - MoE 后端（moe_backend）
#     - 当前正在跑的 batch
#
#   如果通过函数参数一层层传递太烦。所以放进一个全局上下文对象
#   _GLOBAL_CTX，谁需要谁通过 get_global_ctx() 直接拿。
#
# 二、为什么需要 forward_batch 这个 contextmanager？
#
#   "当前 batch"只在前向计算的那段时间有意义。出了前向就应该清空，
#   防止旧 batch 被误用。contextmanager 配合 with 语法刚好能做到：
#
#     with ctx.forward_batch(batch):
#         # 此时 ctx.batch 可用
#         model.forward(...)
#     # 离开 with 后，ctx.batch 自动变回 None
#
#   并且它在进入时 assert _batch is None，防止嵌套调用导致状态错乱。
#
# 三、为什么有 page_size 字段，但有注释说"页表始终按 page_size=1 处理"？
#
#   PagedAttention 允许 page_size > 1（一个页存多个 token 的 KV）。但
#   本项目当前的 page_table 实现简化为"一行存的是每个 token 的 slot"
#   （等价于 page_size=1）。page_size 字段保留为将来扩展。
#
# ════════════════════════════════════════════════════════════════════
@dataclass
class Context:
    """
    【类名】Context（GPU 全局上下文）
    【一句话描述】把跑前向时所有"全局存在的组件"装在一起的容器。
    【生活类比】像一个工厂的"车间总控柜"——电源、原料、机器、当前
                正在加工的工件都挂在它身上，任何工序都可以来问它。
    """

    # ----------------------------------------------------------------
    # 变量: page_size
    # 类型: int
    # 含义: KV Cache 的"页大小"——每页存几个 token 的 K/V。
    # ⚠️ 注意: 当前 page_table 实现简化为 page_size=1。该字段保留供
    #          未来扩展或部分 kernel 内部使用。
    # ----------------------------------------------------------------
    page_size: int

    # ----------------------------------------------------------------
    # 变量: page_table
    # 类型: torch.Tensor (GPU, 2-D long)
    # 形状: [max_requests, max_seq_len]
    # 含义: 一张大表——第 i 行第 j 列是"第 i 个请求的第 j 个 token 存
    #       在 KV Cache pool 的哪个 slot"。
    # 例: page_table[3, 5] = 17 表示第 3 号请求的第 5 个 token 的 K/V
    #     在 KV pool 的 17 号 slot。
    # 这是注意力 kernel 寻址 KV 的关键索引。
    # ----------------------------------------------------------------
    page_table: torch.Tensor = field(init=False)

    # 变量: attn_backend —— 实际的注意力计算后端
    # （FlashAttention/FlashInfer/TRT-LLM 等），可插拔。
    attn_backend: BaseAttnBackend = field(init=False)

    # 变量: moe_backend —— Mixture-of-Experts 模型用的专家路由后端
    # （非 MoE 模型也持有一个 dummy，用于统一接口）。
    moe_backend: BaseMoeBackend = field(init=False)

    # 变量: kv_cache —— KV Cache 池本身（管所有 slot 的分配与读写）。
    kv_cache: BaseKVCachePool = field(init=False)

    # 变量: _batch —— 当前正在前向的 batch；不在前向时为 None。
    # 加下划线表示"内部字段"，外部应通过 .batch 属性访问（有 assert 保护）。
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        """
        【功能】拿到当前正在跑的 batch。
        【约束】必须在 forward_batch(...) with 块内调用，否则报错。
        """
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        """
        【功能】把传入的 batch 设为"当前 batch"，离开 with 时自动清空。
        【用法】
            with ctx.forward_batch(batch):
                # 在这块代码里 ctx.batch 就是 batch
                model.forward(...)
            # 出来后 ctx.batch 又变回 None

        【为什么这样设计】
        - 自动清理：即使中间抛异常，finally 也会把 _batch 设回 None；
        - 防嵌套：进入时 assert 上一次已经清空，避免状态混乱。
        """
        # 防止嵌套调用导致 _batch 被覆盖丢失
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield  # 把控制权交给 with 块内部
        finally:
            # 不论 with 块正常退出还是抛异常，都要清掉 _batch
            self._batch = None


# ════════════════════════════════════════════════════════════════════
# 全局上下文单例 —— 进程内只有一个 Context
# ════════════════════════════════════════════════════════════════════
#
# 为什么用模块级变量做单例？
#   - 简单直接：Python 模块本身就是单例（每个模块只加载一次）；
#   - 各处方便访问：通过 get_global_ctx() 在任何函数里都能拿到；
#   - 多 GPU/多进程场景下：每个进程有自己的 _GLOBAL_CTX 副本，互不干扰。
#
# 为什么不用类方法做单例？
#   - 避免 Context.instance() 这种写法太冗长；
#   - 模块变量 + 两个简单函数已经足够清晰。
# ════════════════════════════════════════════════════════════════════
_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    """
    【功能】在系统启动阶段调用一次，把构造好的 Context 注册为全局单例。
    【调用时机】engine 初始化完成后、开始接收请求之前。
    【保护】重复调用会 assert 失败——一个进程只允许一份全局上下文。
    """
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    """
    【功能】拿到全局 Context；要求 set_global_ctx 已经调用过。
    【调用时机】层、注意力、采样等代码内部需要 KV cache/page_table 时。
    """
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
