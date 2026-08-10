# vLLM 并发估算与显存/算力综合分析

> 本文是 [vLLM参数与推理机制详解](vLLM参数与推理机制详解.md) 的延伸补充，聚焦于一个更具体的实战问题：**日志里打印的"理论最大并发数"是怎么算出来的？为什么实际线上跑的并发经常远超这个理论值？** 涉及 KV Cache 容量计算、block 管理机制、Prefix Cache、Chunked Prefill 等多个概念如何"综合"影响最终的并发能力。
>
> 关联笔记：
> - Prefill / Decode / PIECEWISE / CUDA Graph 等推理阶段基础概念 → [vLLM参数与推理机制详解](vLLM参数与推理机制详解.md)
> - PagedAttention 原理辨析 → [VLLM.md](../HuaggingFace/VLLM.md)
> - GQA/MQA 与 KV Cache 大小的关系 → [MiniMind从0到1构建大模型 · 3.2.3](MiniMind从0到1构建大模型.md)
> - vLLM 除 HTTP API 外的调用方式、OpenAI 协议生态、LangChain tool calling 兼容性 → [vLLM调用方式与OpenAI协议生态](vLLM调用方式与OpenAI协议生态.md)

## 目录

1. [一个真实启动日志案例](#一一个真实启动日志案例)
2. [三个数字是怎么算出来的](#二三个数字是怎么算出来的)
3. [为什么实际并发经常远超理论值](#三为什么实际并发经常远超理论值)
4. [显存里到底分给了谁：不止 KV Cache](#四显存里到底分给了谁不止-kv-cache)
5. [block 只是管理粒度，不是新资源](#五block-只是管理粒度不是新资源)
6. [几种"并发计算公式"其实是同一件事](#六几种并发计算公式其实是同一件事)
7. [Chunked Prefill 与 mixed prefill-decode 的绑定关系](#七chunked-prefill-与-mixed-prefill-decode-的绑定关系)
8. [运行时监控指标：usage 与 hit rate](#八运行时监控指标usage-与-hit-rate)
9. [实战演练：结合运行时指标反推当前真实并发数](#九实战演练结合运行时指标反推当前真实并发数)
10. [总结：两个维度综合来看](#十总结两个维度综合来看)

---

## 一、一个真实启动日志案例

Qwen3-VL-2B 模型，单卡启动，`max_model_len=8192`，日志片段：

```
INFO gpu_worker.py:298] Available KV cache memory: 66.89 GiB
INFO kv_cache_utils.py:1087] GPU KV cache size: 626,240 tokens
INFO kv_cache_utils.py:1091] Maximum concurrency for 8,192 tokens per request: 76.45x
```

以及后续实测中，运行时打印的监控指标：

```
GPU KV cache usage: 26.6%, Prefix cache hit rate: 89.1%
```

本文接下来围绕这几行日志，把背后的计算逻辑和"为什么实际能跑得比 76.45x 更高"讲清楚。

---

## 二、三个数字是怎么算出来的

### 核心公式链条

```
GPU 总显存 × gpu_memory_utilization
        ↓ 减去模型权重、激活值等占用
Available KV cache memory (66.89 GiB)
        ↓ 除以「每 token 的 KV Cache 体积」
GPU KV cache size (626,240 tokens)
        ↓ 除以 max_model_len (8192)
Maximum concurrency (76.45x)
```

### 第一步：`Available KV cache memory` —— 显存做减法

```
Available KV cache memory
    = GPU 总显存 × gpu_memory_utilization
      - 模型权重占用
      - 激活值/中间缓冲区峰值占用（profiling 阶段实测出来的）
      - 其他固定开销（CUDA context、碎片预留等）
```

例如 80GB 显存卡，`gpu_memory_utilization=0.95`：

```
80 × 0.95 ≈ 76 GiB 可用总量
76 GiB - 4.24 GiB(模型权重) - 其他杂项 ≈ 66.89 GiB
```

### 第二步：`GPU KV cache size` —— 换算成"能装多少 token"

```
每个 token 的 KV Cache 大小 = 2（K和V） × num_layers × num_kv_heads × head_dim × dtype_bytes
GPU KV cache size = Available KV cache memory ÷ 每个 token 的 KV Cache 大小
```

- `2`：同时存 K 和 V
- `num_layers`：模型总层数，每层都要单独存
- `num_kv_heads`：**GQA/MQA 压缩后**的 KV 头数（不是 Query 头数），头数越少，KV Cache 越小
- `head_dim`：每个注意力头的维度
- `dtype_bytes`：`bf16` 为 2 bytes

以 Qwen3-VL-2B 典型配置估算（`num_layers≈28`，`num_kv_heads≈8`，`head_dim≈128`）：

```
每 token KV Cache = 2 × 28 × 8 × 128 × 2 bytes ≈ 112 KB
626,240 tokens × 112 KB ≈ 66.9 GiB ✅ 与日志基本吻合
```

> 💡 这也是为什么 GQA/MQA/MLA 这类"压缩 KV 头数"的架构优化，对推理并发能力提升明显——`num_kv_heads` 越小，同样显存能装的 token 越多。详见 [MiniMind从0到1构建大模型 · 3.2.3 GQA、MQA](MiniMind从0到1构建大模型.md)。

### 第三步：`Maximum concurrency` —— 一个除法

```
Maximum concurrency = GPU KV cache size ÷ max_model_len
                     = 626,240 ÷ 8192
                     ≈ 76.45x
```

含义：**如果每个请求都用满 `max_model_len` 的长度，理论上最多能同时支撑约 76 个请求处于 Decode 阶段**。

---

## 三、为什么实际并发经常远超理论值

`Maximum concurrency: 76.45x` 是**最坏情况下的悲观估算**，前提假设是"每个请求都刚好用满 `max_model_len`"。真实业务中这个假设通常不成立，主要有两个原因：

### 原因一：真实请求长度远小于 `max_model_len`

`max_model_len` 只是一个上限保护值，不代表请求真的会用到这么长。

```
假设真实请求平均长度只有 1000 token
真实并发上限 ≈ 626,240 ÷ 1000 ≈ 626x  （而不是按 8192 算出的 76.45x）
```

### 原因二：PagedAttention 按需分配，不是预先整块锁死

vLLM 用分块（block）方式管理 KV Cache，**请求实际生成到多长，就占用多少 block**，而不是一开始就预留满 `max_model_len` 的空间。详见 [VLLM.md · PagedAttention 解析](../HuaggingFace/VLLM.md)。

### 一个类比

```
餐厅仓库共有 626,240 "食材单位"（GPU KV cache size）
餐厅规定每桌最多点一份"满汉全席"，耗费 8192 单位（max_model_len）

按"每桌都点满汉全席"估算 → 最多接待 76 桌（Maximum concurrency）

现实中大部分客人只点了一份小炒（平均消耗远小于 8192 单位）
→ 仓库能同时供应的桌数远超过 76 桌
```

---

## 四、显存里到底分给了谁：不止 KV Cache

一张 GPU 卡的显存，vLLM 启动时依次划分给：

```
GPU 总显存
  ├── 【常驻】模型权重（Weights）
  │     Attention 的 Q/K/V/O 投影矩阵、FFN 的 Linear 层、Embedding、Norm 参数等
  │
  ├── 【瞬时】激活值（Activations）
  │     FFN 中间输出、Attention 计算过程中间结果、Softmax 中间值
  │     Prefill 阶段因一次处理 token 多，这部分峰值最大，是 OOM 主因之一
  │
  ├── 【VL模型特有】Encoder Cache（视觉 token 缓存）
  │     图像/视频过 Vision Encoder 编码出来的视觉 token 也占显存
  │
  ├── 【常驻，推理独有】KV Cache
  │     只存 Attention 里的 K、V（不含 FFN！）
  │
  └── 【固定开销】CUDA context、CUDA Graph 录制占用、显存碎片预留
```

### 为什么只有 Attention 需要 Cache，FFN 不需要？

Attention 的计算需要当前 token 的 Query 跟**所有历史 token**的 Key、Value 做运算：

```
Attention(Q_当前token, K_所有历史token, V_所有历史token)
```

不缓存历史 K、V 的话，每生成一个新 token 就要把从头到尾所有历史 token 重新算一遍，代价随生成长度平方增长。

而 FFN 是逐 token 独立计算：

```
FFN(某个token的向量) → 输出
```

只依赖这个 token 自己，跟别的 token 没关系，算完即释放，没有"缓存历史"的需求。

```
FFN 相关显存占用      = 权重（常驻）+ 激活值（瞬时，算完释放）
Attention 相关显存占用 = 权重（常驻）+ 激活值（瞬时）+ KV Cache（常驻，随生成增长）
```

所以 `Available KV cache memory` 这个数字，本质上是：

```
总可用显存 - 模型权重(含Attention和FFN全部权重) - 激活值峰值(含Attention和FFN中间结果) - Encoder Cache - 固定开销
= 剩下专门留给 KV Cache 的显存
```

FFN 的权重和激活值已经在"减法"里扣掉了，不会重复出现在 KV Cache 的计算里，两者不是同一维度、不需要放在一起比。

---

## 五、block 只是管理粒度，不是新资源

"KV Cache block × block 数量"和"总容量 ÷ 单 token 大小"是**同一件事，只是换了个单位**。

```
不分 block 的朴素想法：
KV Cache 总容量（token数）= 显存 ÷ 单 token 的 KV Cache 大小

vLLM 实际用 block 管理：
1个 block = 固定数量的 token（block_size，启动时配置好，如 16）
KV Cache 总容量（token数）= block_size × 总 block 数
```

两者数学上等价：

```
总 block 数 = 显存 ÷ (block_size × 单 token KV Cache 大小)

→ block_size × 总 block 数 = 显存 ÷ 单 token KV Cache 大小
```

block 只是显存的"管理粒度"（类似操作系统内存分页），目的是实现 PagedAttention——**按需一块一块地分配显存，而不是一次性预留 `max_model_len` 那么长的连续空间**，避免长度不一的请求互相产生显存碎片浪费。

日志里 `num_gpu_blocks is: 39140` 对应：

```
39,140 个 block × block_size(默认16) = 626,240 token
```

正好对上 `GPU KV cache size: 626,240 tokens`——不是两种算法，是同一个总量的两种单位表达。

---

## 六、几种"并发计算公式"其实是同一件事

```
写法A（整包估算）：
最大并发数 = KV Cache 总 token 容量 ÷ max_model_len

写法B（block 版本，等价写法）：
最大并发数 = 总 block 数 ÷ 每个请求需要的 block 数
          = 总 block 数 ÷ ceil(max_model_len ÷ block_size)

写法C（考虑真实长度）：
真实并发数 ≈ KV Cache 总 token 容量 ÷ 每个请求的真实平均长度（而非 max_model_len）

写法D（考虑 Prefix Cache 命中后）：
真实并发数 ≈ KV Cache 总 token 容量 ÷ 每个请求"新增"的 token 数（命中前缀部分不重复占用）
```

这不是四种互相矛盾的算法，而是**同一个"总容量 ÷ 每个请求占用量"公式**，随着考虑的因素越贴近真实场景（从"用 block 单位算"到"用真实长度算"再到"扣掉前缀命中"），估算结果从"理论保守值"逐步逼近"实际能达到的值"。

### 概念定位速查表

| 概念 | 属于哪一层 | 作用 |
|---|---|---|
| 模型权重 | 显存分配第一部分 | 常驻，训练好后固定大小 |
| 激活值（含 FFN 中间结果） | 显存分配第二部分 | 瞬时，随 token 数、batch size 变化 |
| KV Cache | 显存分配第三部分 | 只存 Attention 的 K/V，常驻且随生成增长 |
| block / block_size | KV Cache 的**管理粒度**，不是新资源 | 决定显存怎么按需切块分配（PagedAttention） |
| Prefix Cache 命中率 | 影响"每个请求实际占用多少 KV Cache" | 命中部分不重复占用，等效降低分母 |
| Chunked Prefill / mixed prefill-decode | 独立于 KV Cache 显存这条线，是**算力调度**层面的东西 | 决定 GPU 每个 step 先干谁的活，不影响 KV Cache 总量怎么算 |

### 一句话理清

```
显存被切成"权重 + 激活值(含FFN) + KV Cache(只含Attention的K,V)"三大块
   ↓
KV Cache 这一块，内部用 block 做精细化管理（block数 × block_size = 总token容量）
   ↓
"能撑多少并发" = 这个总token容量 ÷ 每个请求实际占用的token数
   （用 max_model_len 算是"最悲观估算"，用真实长度、扣掉前缀命中算才是"贴近实际"）
   ↓
Chunked Prefill / mixed prefill-decode 跟上面这条链完全是另一件事，
它管的是"GPU算力这一秒该先算谁"，不影响"KV Cache能装下多少请求"这个数字
```

---

## 七、Chunked Prefill 与 mixed prefill-decode 的绑定关系

> 完整原理见 [vLLM参数与推理机制详解 · 四、mixed prefill-decode 与 PIECEWISE](vLLM参数与推理机制详解.md#四mixed-prefill-decode-与-piecewise)，这里只梳理结论。

1. **是绑定的**：`mixed prefill-decode` 是 Chunked Prefill 开启后才可能出现的调度结果，不开 Chunked Prefill 就不可能出现"一个 step 里 Prefill 和 Decode 混在一起"的情况。

2. **不开 Chunked Prefill 时，一个 step 只能是纯 Prefill 或纯 Decode，二选一**，不能"纵向"混类型。

3. **但"一个 step 多个 Prefill"是可以的，且和 Chunked Prefill 无关**——只要这些 Prefill 请求的 token 总数不超过 `max_num_batched_tokens`，vLLM 可以把多个不同请求的 Prefill 打包进同一个纯 Prefill step 一起算（batched prefill，"横向"拼多个同类型任务）：

```
不开 Chunked Prefill：
step1（纯Prefill）: [请求A的全部Prefill] + [请求B的全部Prefill] = 一起算，只要没超 max_num_batched_tokens ✅
step2（纯Decode）:  [A的decode token] + [B的decode token] + [C的decode token]
→ 两种 step 泾渭分明，不会出现在同一个 step 里
```

### 两个维度的区分

```
"能不能多个请求的 Prefill 挤在一个 step"  → 可以，不需要 Chunked Prefill（横向，同类型拼批）
"能不能 Prefill 和 Decode 挤在同一个 step" → 需要 Chunked Prefill 才行（纵向，不同类型混合）
```

不开 Chunked Prefill 时真正被禁止的只有"同一个请求的 Prefill 被切块跨多个 step"以及"纵向混类型"，并不禁止"一个 step 里塞入多个不同请求的完整 Prefill"。

---

## 八、运行时监控指标：usage 与 hit rate

启动日志打印的是**静态理论值**（只算一次，之后不变），而下面两个指标是 vLLM **运行时**周期性打印的**动态实况**，衡量完全不同的两件事。

```
GPU KV cache usage: 26.6%, Prefix cache hit rate: 89.1%
```

### `GPU KV cache usage`：显存"仓库"当前用了多少

```
GPU KV cache usage = 当前已分配给活跃请求的 KV Cache token 数 ÷ KV Cache 总容量（token数）

usage = 26.6% → 当前约 626,240 × 26.6% ≈ 166,580 个 token 的 KV Cache 正被占用
```

含义：

```
usage 低（如 26.6%）→ 显存维度还有很大余量，不是当前瓶颈
usage 接近 100%     → KV Cache 快装不下，vLLM 会开始"抢占"（preempt）某些请求
                       （释放其 KV Cache，请求排队重新等，造成明显延迟毛刺）
```

### `Prefix cache hit rate`：这次请求的输入里，有多少是"抄来的"

```
Prefix cache hit rate = 命中缓存、免于重新计算的 prompt token 数 ÷ 总 prompt token 数

hit rate: 89.1% → 平均每个请求的输入 token 里，89.1% 复用了之前请求算好的 KV Cache
                   只有约 10.9% 是真正需要重新做 Prefill 计算的"新内容"
```

高命中率通常意味着业务里大量请求共享同一个很长的 system prompt（固定任务说明、few-shot 示例等），只有末尾一小段用户输入/图片每次不同。

> ⚠️ **必须明确的边界**：Prefix Cache **只能命中 Prompt（输入）部分，不可能命中 Decode（输出/生成）部分**。因为输入是"发请求前就写死的文本"，只要多个请求前缀字符完全一致，算出来的 K、V 必然一致，才有得复用；而输出是"模型运算后才产生"的结果，依赖采样参数、随机性，哪怕输入完全相同，生成内容通常也不同，天然没有"预先算好可复用"这回事。所以 `Prefix cache hit rate` 统计的永远是 **prompt token 的命中比例**，跟这次请求生成了多少 token、生成速度无关。详细原理见 [vLLM参数与推理机制详解 · Prefix Cache 只能命中 Prompt 部分](vLLM参数与推理机制详解.md#prefix-cache-只能命中-prompt输入部分不可能命中-decode输出部分)。

### 两者合起来说明了什么

```
GPU KV cache usage: 26.6%     → 仓库当前堆了多少货，还剩多少空位（结果，实时的"库存水位"）
Prefix cache hit rate: 89.1%  → 为什么货能堆得这么省地方：
                                 大部分货物是"共用的公共部分"，不用每单都单独囤一份（原因）
```

这也解释了"为什么实际并发能远超理论 76.45x"的另一个关键原因：

```
理论估算 76.45x 的前提是"每个请求都要重新用满 max_model_len=8192 的 KV Cache"

实际情况：
若一个请求总长度 8192 token（输入 prompt + 输出 response），
且这 8192 里假设有 7000 是 prompt、1200 是 response，其中 prompt 命中前缀缓存 89.1%

⚠️ 注意：不能直接把整个 8192 乘以 (1-89.1%)，因为 response 是 Decode 阶段全新生成的内容，
天生不可能命中任何缓存（命中率恒为 0%），必须把 prompt 和 response 分开计算：

  prompt 新增占用  = 7000 × (1 - 89.1%) ≈ 763 token
  response 新增占用 = 1200 × (1 - 0%)    = 1200 token（全额，无法打折）
  → 这个请求真正"新增占用"的 KV Cache ≈ 763 + 1200 = 1963 token

→ 仍然明显小于理论值 8192，相当于每个请求实际消耗的显存远小于理论估算
→ 同样的显存池子，自然能撑更多并发（具体折扣幅度取决于 prompt/response 的长度占比，
   response 占比越高，Prefix Cache 带来的节省效果越有限）
```

> ⚠️ 完整的、按 prompt/response 分开计算的实战演练见 [九、实战演练：结合运行时指标反推当前真实并发数](#九实战演练结合运行时指标反推当前真实并发数)。

### 实践建议

- `GPU KV cache usage`：判断"要不要担心 OOM / 抢占"的直接指标，持续接近 100% 才需要警惕，可配合压测的 P99 TTFT 一起看（见 [vLLM参数与推理机制详解 · 七、压测指标详解](vLLM参数与推理机制详解.md#七压测指标详解吞吐量ttfttpot)）
- `Prefix cache hit rate`：判断"值不值得针对性优化 prompt 结构"的指标。命中率低时，检查是不是把易变内容（时间戳、随机 id 等）不小心放在了 prompt 靠前的位置，导致本该相同的公共前缀因为前面变了而"整体错位"，无法命中缓存

---

## 九、实战演练：结合运行时指标反推当前真实并发数

前面几节讲的都是"启动时的静态理论值"，这里演示一次**完全用运行时动态指标反推真实并发数**的完整计算，把前面所有知识点串起来用一遍。

### 已知条件

```
GPU KV cache size(总容量，来自启动日志，固定值)  = 626,240 tokens
GPU KV cache usage(运行时)                      = 26.6%
Prefix cache hit rate(运行时，文本前缀命中率)      = 89.1%
MM cache hit rate(运行时，多模态/图像前缀命中率)   = 0.0%
max_token_len(单请求 输入+输出 总长度)            = 9500
response(这次请求的输出长度)                     = 1400
```

> 💡 `MM cache hit rate` 是 Prefix Cache 针对**多模态输入**（图像/视频编码出来的视觉 token）单独统计的命中率，和纯文本的 `Prefix cache hit rate` 是两个独立指标——原因同样是"只有输入侧才谈得上缓存复用"（详见 [八、Prefix Cache 只能命中 Prompt 部分](#八运行时监控指标usage-与-hit-rate)），只不过 VL 模型的输入分文本和图像两类，各自的内容重复模式不同，所以拆成两个指标分别统计。这次 `MM cache hit rate = 0.0%` 说明**这一批请求里的图像/视频内容完全没有重复**（比如每次都是不同的图片），一次都没命中过。

### 第一步：算出当前"已占用"的 token 数

```
已占用 token 数 = GPU KV cache size × usage
              = 626,240 × 26.6%
              ≈ 166,580 tokens
```

### 第二步：反推单个请求的输入(prompt)长度

```
max_token_len = 9500（输入+输出总长度，这里假设这次请求正好用满）
response      = 1400
→ prompt = 9500 - 1400 = 8100 tokens
```

### 第三步：结合命中率，算出这个请求"新增"占用了多少 KV Cache

**关键原则**：命中缓存的部分不重复占用显存，只有"新增"部分才会消耗当前的 usage；而 response（Decode 生成的输出）天生不可能命中任何缓存，全部都是新增。

```
prompt 里新增的 token = 8100 × (1 - 89.1%) ≈ 883 tokens
                       （这里简化处理：假设 89.1% 是对整个 prompt 的综合命中率；
                        如果能拆分出 prompt 里文本、图像各自的 token 数，
                        应该分别用 Prefix cache hit rate 和 MM cache hit rate 精确计算，
                        即 文本新增 = 文本token×(1-89.1%)，图像新增 = 图像token×(1-0.0%)）

response 全部新增 = 1400（Decode 输出，无法命中任何缓存）

单个请求新增占用的 KV Cache ≈ 883 + 1400 = 2283 tokens
```

### 第四步：反推当前真实并发数

```
当前并发数 ≈ 已占用 token 数 ÷ 单个请求新增占用的 token 数
           ≈ 166,580 ÷ 2283
           ≈ 73x
```

### 与启动时理论估算对比

```
启动日志的理论估算（假设人人写满 max_model_len、完全不命中缓存）  ≈ 76.45x
这次结合真实 usage + 真实命中率反推出的"当前实际并发数"          ≈ 73x
```

两者数值恰好接近，但含义完全不同：前者是"最坏情况下的悲观静态估算"，后者是"某一时刻真实业务负载下反推出来的真实并发数"——之所以这次反推值没有像前面章节讨论的那样"远超"理论值，是因为这次 `MM cache hit rate = 0.0%`（图像部分完全没有复用），拉低了整体的"新增占用"节省效果，说明**命中率对并发能力的提升幅度，取决于业务里到底有多少内容是真正可复用的**，不能一概而论地认为"有了 Prefix Cache 并发就一定能远超理论值"。

### 这套反推方法依赖的关键假设（务必核对）

1. **`max_token_len` 和 `response` 是这批请求的代表值（如平均值），而不是偶然的单次极端值**，否则反推出的并发数不能代表整体真实水平。
2. **`Prefix cache hit rate` 的统计口径**是对全部 prompt token 综合计算，还是仅针对文本 token（不含图像）——如果 vLLM 版本对文本与图像分开统计命中率（且各自有独立的 token 计数），应该把 prompt 拆成"文本 token 数 + 图像 token 数"分别代入对应的命中率计算，结果会比这里的简化算法更精确。

### 通用计算公式总结

```
反推实际并发数 = (GPU KV cache size × GPU KV cache usage)
              ÷ [ prompt × (1 - 综合/加权命中率) + response ]

其中"综合/加权命中率"理想情况下应按下式精确计算（如果文本、图像token数可拆分）：
加权命中率 = (文本token数×Prefix cache hit rate + 图像token数×MM cache hit rate) ÷ prompt总token数
```

---

## 十、总结：两个维度综合来看

所有这些概念，最终只影响两种资源：

```
资源一：显存里的 KV Cache 容量（"仓库能装多少菜"）
资源二：GPU 算力/调度时间片（"厨房一次能做多少道菜"）

Prefix Cache Hit      → 影响资源一（省显存 + 省计算）
Chunked Prefill        → 影响资源二（怎么切分/调度算力）
mixed prefill-decode   → 是 Chunked Prefill 带来的"调度结果现象"，不是独立机制
```

```
┌─────────────────────────────────────────────────────────┐
│  维度一：显存够不够（KV Cache 容量）                        │
│  受影响于：max_model_len、请求真实长度、Prefix Cache 命中率  │
│  → 决定"能不能同时装下这么多请求的历史"                      │
│  → 撞到上限的表现：请求被"抢占"（preempted）、排队等 KV Cache  │
├─────────────────────────────────────────────────────────┤
│  维度二：算力/调度够不够（GPU 计算时间片）                    │
│  受影响于：max_num_batched_tokens、Chunked Prefill 开关     │
│  → 决定"每个 step 能处理多少活儿，Prefill 和 Decode 怎么分配" │
│  → 撞到上限的表现：TTFT/TPOT 变高（这次 step 没轮到你）      │
└─────────────────────────────────────────────────────────┘
```

实际观察到"并发数远超理论值"，通常是这两个维度共同作用的结果：

1. 真实请求长度 << `max_model_len`，且/或 Prefix Cache 命中率高 → **显存维度**压力比理论估算小得多
2. Chunked Prefill 让 GPU 算力被更精细地切分利用，没有"一个大 Prefill 卡住所有人"的浪费 → **算力维度**利用率更高

### 记忆口诀

```
KV Cache 容量（含 Prefix Cache 命中率）→ 决定"能同时记住多少个请求的历史"（显存）
Chunked Prefill / mixed prefill-decode  → 决定"GPU 这一秒钟该先干谁的活"（算力调度）

理论并发值(76.45x) 只算了显存这一半账
真实体验好不好，还要看算力调度这一半账，
以及真实请求是不是比 max_model_len 短很多、前缀重复率高不高
```

判断自己业务当前是显存瓶颈还是算力瓶颈，最直接的方式是对照运行时日志中的 `GPU KV cache usage`（显存维度）和 TTFT/TPOT 随并发变化的曲线（算力调度维度），而不是只看启动时打印的静态理论并发值。
