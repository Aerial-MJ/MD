# vLLM 参数与推理机制详解

## 三句话总结

**`max_num_batched_tokens`**：一次 forward（一个 step）里，所有请求（不管是在 Prefill 还是 Decode）加起来最多能处理的 token 总数上限，控制的是"这一步计算量有多大"，直接决定显存里激活值的峰值大小。

**`mixed prefill-decode`**：因为开了 Chunked Prefill，一个 step 里可以**同时**塞入"某些请求的 Prefill 片段"和"另一些请求的 Decode token"，两者混在一起算，不用等 Prefill 全部做完才轮到 Decode。

**CUDA Graph 加速手段**：把 GPU 要执行的一连串固定指令**提前录好、之后直接整体回放**，省掉 CPU 和 GPU 每一步来回请示汇报的通信开销，从而提速（PIECEWISE 就是"只把形状固定、能录的那部分层录成回放，形状不固定的 Prefill 部分照常现场算"）。

## 一、两个核心参数的区别

### `--max-model-len`

**针对**：单条请求  
**控制**：输入 token + 输出 token 的**总长度上限**

```
请求A: 输入 15000 token + 输出 1000 token = 16000 ✅
请求B: 输入 16000 token + 输出 1000 token = 17000 ❌ 直接拒绝
```

**影响的是 KV Cache 的分配**：

```
max_model_len 越大
→ 每个请求预留的 KV Cache 槽位越多
→ 总 KV Cache 能容纳的并发请求数越少
→ 显存占用越高
```

---

### `--max-num-batched-tokens`

**针对**：一个 batch（所有并发请求的总和）  
**控制**：每次 forward 的**计算量上限**

```
例子（假设当前有 5 个并发请求，每个当前 step 需处理 2000 token）：
  请求A: 2000 token
  请求B: 2000 token
  请求C: 2000 token
  请求D: 2000 token
  请求E: 2000 token
  合计: 10000 token > 8192

→ vLLM 会把请求 E 推迟到下一个 step
→ 保证每次 forward 不超过 8192 token
```

**影响的是激活值 buffer 的大小**：

```
max_num_batched_tokens 越大
→ 每次 forward 的激活值越大
→ 瞬时显存峰值越高
→ 越容易 OOM
```

> ⚠️ **常见误区**："一次 forward 不应该只处理一个 token 吗？"——这个说法只在**纯 Decode、单个请求**时成立。上面例子里 A~E 贡献的 2000 token 并不是 Decode 生成的，而是它们的 **Prefill**（详见下一节）。一次 forward 的总 token 数，其实是"这个 step 里所有请求各自贡献的 token 数之和"：
>
> ```
> 一次 forward 的 token 组成 =
>     多个请求各自的 Decode token（每个只贡献 1 个）
>   + 多个请求各自的 Prefill chunk（每个可能贡献几百~几千个）
> ```
>
> 如果把上面的例子换成 A~E 全部处于 Decode 阶段，那合计只有 5 token，远小于 8192，根本不会触发这个限制。真正会让 `max_num_batched_tokens` 起作用的场景，几乎都是有 Prefill（或 Prefill+Decode 混合）参与的 step。这也是为什么该参数叫 `max_num_batched_tokens`（限制的是整个 batch 叠加后的总量），而不是限制单个请求。

> 🤔 **追问**：那有没有可能纯 Decode 场景也刚好凑够 8192，撞到这个上限？
>
> **理论上可能，但现实中几乎不会先撞到这里，会先被别的限制拦住**：
>
> ```
> 8192 个请求 × 1 token/请求（Decode）= 8192 token
> → 数学上确实等于 max_num_batched_tokens=8192
> ```
>
> 但在此之前，通常会先被以下两个限制拦住：
>
> 1. **`--max-num-seqs`（最大并发序列数）**：vLLM 还有一个参数专门限制同时处理的请求数量。比如日志里 `max_num_seqs=3`，意味着同一时刻最多只有 3 个请求在跑，根本凑不到 8192 个并发。
> 2. **KV Cache 显存容量**：想让 8192 个请求同时处于 Decode 阶段，KV Cache 必须能装下这 8192 个请求各自的历史 K、V，这个显存需求通常是几十上百 GB 起步，远超单卡甚至多卡显存。比如日志里 `GPU KV cache size: 21,056 tokens`，`max_model_len=7250` 时最多只能撑 2.9 个并发，离 8192 差了几个数量级。
>
> 一个类比：
>
> ```
> max_num_batched_tokens  → 餐厅每一轮上菜最多端 8192 道菜（计算上限）
> max_num_seqs            → 餐厅同时最多接待 3 桌客人（并发上限）
> KV Cache 显存            → 餐厅仓库最多备 21,056 份食材（显存上限）
>
> 想让"端菜数"撞到 8192 上限，前提是先凑齐 8192 桌客人同时点餐
> 但餐厅最多只能接待 3 桌 → 根本凑不齐这个数
> ```
>
> 只有在**并发量特别大、KV Cache 也配得很大**（如多卡大显存、`max_num_seqs` 调到几百上千，常见于短输入短输出的高并发场景）时，纯 Decode 才有可能撞到 `max_num_batched_tokens`。实践中更常见的做法是让 `max_num_batched_tokens` 明显大于 `max_num_seqs`，这样 Decode 阶段的 token 总数（约等于并发数）通常远小于上限，真正吃满上限的还是 Prefill 阶段。

---

### 两者关系对比

```
max_model_len    → 纵向限制（单个请求有多深）
                        ↓
              [请求A: ████████████ 16000 tokens]
              [请求B: ████████████ 16000 tokens]

max_num_batched_tokens → 横向限制（一次 forward 处理多宽）
                        ↓
              step1: [A 的前 4096 token + B 的前 4096 token] = 8192 ✅
              step2: [A 的后 4096 token + B 的后 4096 token] = 8192 ✅
```

> 💡 **一句话**：`max_model_len` 决定单个请求能有多长，`max_num_batched_tokens` 决定一次 forward 能有多重；前者影响 KV Cache，后者影响激活值，都需要根据显存余量来调。

---

## 二、推理的两个阶段：Prefill 与 Decode

### Prefill（预填充）阶段

**含义**：把输入的所有 token 一次性全部喂给模型，计算并"填充"每个 token 对应的 K、V，存入 KV Cache。

```
用户输入："请判断这张图片是否有效。[图片]巴拉巴拉...系统 prompt..."
= 6000 个 token

Prefill：把这 6000 个 token 一次性全部计算
         一个 step 处理 6000 token
         
为什么一次性处理？
因为这 6000 个 token 是已知的，可以并行计算
GPU 最擅长并行 → 一次处理完效率最高
```

**为什么叫 Prefill（预先填充）？**

Attention 机制需要 K（Key）和 V（Value）矩阵，Prefill 阶段就是把所有输入 token 的 K、V "预先填充"到 KV Cache 里，供后续 Decode 阶段使用：

```
┌──────────────────────────────┐
│ KV Cache                     │
│ token1 的 K,V  ████          │
│ token2 的 K,V  ████          │
│ ...                          │
│ token6000 的 K,V  ████       │  ← prefill 完成
│ token6001 的 K,V  [空]       │  ← decode 阶段逐步填入
└──────────────────────────────┘
```

### Decode（解码）阶段

**含义**：模型开始逐个生成输出 token，每个 step 只生成一个 token。

```
step1 → 生成 "{"          (处理 1 个 token)
step2 → 生成 "label"      (处理 1 个 token)
step3 → 生成 ":"           (处理 1 个 token)
step4 → 生成 "无效图"      (处理 1 个 token)
...
```

每次生成一个新 token，就往 KV Cache 里追加一条 K,V，并读取所有历史 K,V 来计算 Attention。

### 完整请求生命周期

```
用户请求
    ↓
┌─────────────────────────────────────┐
│  Prefill（1 个 step，处理 N 个输入 token）│
│  6000 token → 一次并行计算完           │
│  同时生成 KV Cache                   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Decode（每个 step 生成 1 个 token）  │
│  step1: 读 KV Cache → 生成 token1   │
│  step2: 读 KV Cache → 生成 token2   │
│  ...重复约 1250 次（max_tokens）     │
└─────────────────────────────────────┘
    ↓
输出结果
```

> 💡 **显存压力分布**：
> - **Prefill 阶段**：重计算，激活值峰值高，是 OOM 的主要触发点
> - **Decode 阶段**：轻计算，但 KV Cache 随生成长度增长

---

## 三、三个"Pre"概念区分

### 名词速查

| 概念 | 层次 | 含义 |
|------|------|------|
| **Prefill** | 推理阶段 | 处理输入 token 的阶段 |
| **Prefix Cache** | 缓存复用技术 | 复用相同前缀的 KV Cache |
| **Chunked Prefill** | 优化方式 | 把大 Prefill 切成小块执行 |

---

### Prefix Cache（前缀缓存）

**场景**：大量请求共享相同的 system prompt。

```
请求1: [system prompt 5000 token] + [图片A 300 token] + [问题 100 token]
请求2: [system prompt 5000 token] + [图片B 300 token] + [问题 100 token]
请求3: [system prompt 5000 token] + [图片C 300 token] + [问题 100 token]
         ↑ 这 5000 个 token 每次都一样！
```

**没有 Prefix Cache 时**：每次请求都要重复计算 5000 token 的 KV Cache，浪费。

**有 Prefix Cache 时**：

```
请求1: 计算全部 5400 token，把前 5000 token 的 KV Cache 缓存起来
请求2: 前 5000 token 命中缓存！直接复用 ✅ 只需计算新的 400 token
请求3: 同上，只计算 400 token
```

> ⚠️ **注意**：缓存的 KV Cache 占显存，若淘汰不及时，会导致显存持续增长。

---

### Chunked Prefill（分块预填充）

**问题背景**：没有 Chunked Prefill 时，请求 A 的 Prefill（6000 token）会长时间占用 GPU，请求 B 只能干等，用户体验差。

**解决方案**：把大的 Prefill 切成小块，和 Decode 交替执行：

```
配置: chunked_prefill_enabled=True, max_num_batched_tokens=2048
（B 全程在 Decode，每个 step 占 1 个 token 的预算，剩下 2047 才是留给 A 的 Prefill 预算）

时间线：
step1: [请求A Prefill 前 2047 token ████] + [请求B Decode token1]  = 2048 ✅
step2: [请求A Prefill 中 2047 token ████] + [请求B Decode token2]  = 2048 ✅
step3: [请求A Prefill 后 1906 token ███ ] + [请求B Decode token3]  = 1907 ✅（A 的 6000 token 已处理完）
step4: [请求A Decode token1]              + [请求B Decode token4]  = 2   ✅
```

> 注意：每个 step 里，A 的 Prefill chunk + B 的 Decode token 之和**不能超过** `max_num_batched_tokens`。因为 B 一直占着 1 个名额，A 每次只能拿到 `2048 - 1 = 2047` 个 token，而不是整整 2048——这也是 Chunked Prefill 调度器实际做的事：先满足所有 Decode 请求（保证已经在生成的请求不被饿死），剩下的预算才分给 Prefill。

**好处**：
- ✅ 请求 B 不用等请求 A 的 Prefill 全部完成
- ✅ 每个 step 的 token 数不超过 2048，激活值峰值可控
- ✅ 整体吞吐量更高

---

### Prefill 阶段一般需要多少次 forward？

**不开 Chunked Prefill（或输入长度 < max_num_batched_tokens）**：1 次 forward 就能完成。因为 Prefill 的 token 全部已知，可以在 Attention 里并行计算，理论上"一口吃下"：

```
输入 6000 token，max_num_batched_tokens = 8192
6000 < 8192，一次就能装下 → 1 次 forward 完成整个 Prefill
```

**开启 Chunked Prefill（vLLM 默认行为），输入超过 max_num_batched_tokens**：会被切成多个 chunk，分摊到多个 step：

```
forward 次数 ≈ ceil(输入总 token 数 / 每个 step 分给该请求的 chunk 大小)

例：max_num_batched_tokens = 2048，输入 6000 token（无并发抢占时）
ceil(6000 / 2048) = 3 次

step1: Prefill chunk1 [token 0~2047]    → 2048 token
step2: Prefill chunk2 [token 2048~4095] → 2048 token
step3: Prefill chunk3 [token 4096~5999] → 1904 token
→ 3 次 forward 才能完成这一个请求的 Prefill
```

**多个请求并发时**：`max_num_batched_tokens` 这个预算不会被一个请求独占，还要给其他请求的 Decode token 留位置，所以实际能分给 Prefill 的 chunk 会更小，次数可能略增：

```
假设 max_num_batched_tokens = 2048，当前有 3 个请求正在 Decode（各占 1 token）
→ 剩余可用于新请求 Prefill 的 budget = 2048 - 3 = 2045

请求F 输入 6000 token：
step1: [3个Decode token] + [F的Prefill chunk 2045 token] = 2048
step2: [3个Decode token] + [F的Prefill chunk 2045 token] = 2048
step3: [3个Decode token] + [F的剩余 1910 token]           = 1913
→ 依然接近 3 次，但因为要分摊给别的请求，次数可能比理想值略多
```

> 💡 **一句话**：Prefill 的 forward 次数取决于「输入长度」和「`max_num_batched_tokens` 减去并发占用后剩余的 budget」的比值，budget 越大、并发 Decode 请求越少，Prefill 完成得越快。

---

## 四、mixed prefill-decode 与 PIECEWISE

### mixed prefill-decode

这是 Chunked Prefill 的直接体现：开启 Chunked Prefill 后，调度器**被允许**把一个 step 同时安排给 Prefill chunk 和 Decode token，即"混合执行"。

```
日志：Chunked prefill is enabled with max_num_batched_tokens=2048
                ↓
每个 step 最多 2048 token，里面可以混合 Prefill 和 Decode
                ↓
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)
```

> 🤔 **追问**：不开 Chunked Prefill 就一定不会 mixed？如果 `max_num_batched_tokens` 设得够大，是不是也能 mixed？
>
> **关键点在于"调度规则"，而不是"budget 大小"**：
>
> - **不开 Chunked Prefill**：调度规则规定"一个请求的 Prefill 必须一次性、独占一个 step 做完，中途不能被打断、也不能塞入别的请求"。哪怕 `max_num_batched_tokens` 设置得非常大，这个 step 依然只会是纯 Prefill 或纯 Decode，**天生不可能 mixed**——不是因为 budget 不够，而是规则本身不允许混。
> - **开启 Chunked Prefill**：调度规则变成"允许把 Prefill 切块，见缝插针地跟 Decode 混进同一个 step"。但即便开了，也**不代表每个 step 都一定会 mixed**——如果当前系统里只有 Prefill 请求（还没有请求进入 Decode 阶段）或者只有 Decode 请求（没有新请求在排队做 Prefill），这个 step 依然是纯的，不会 mixed。
>
> 所以更准确的说法是：
>
> ```
> 开启 Chunked Prefill        → 只是"解锁"了 mixed 的可能性（调度规则允许了）
> 是否真的发生 mixed          → 取决于当前是否同时存在"待 Prefill"和"正在 Decode"的请求
> max_num_batched_tokens 大小 → 只决定 mixed 时这个 step 能装多少 token，不决定能不能 mixed
> ```

### CUDA Graph 简介（先抛开术语，从生活化例子理解）

GPU 每次计算，CPU 都要发一堆"指令"给它，比如一层模型就有几十条指令（矩阵乘法、加法、激活函数……），几十层叠起来就是上千条：

**没有 CUDA Graph**：每条指令都是"CPU 发一条 → GPU 执行 → GPU 汇报执行完了 → CPU 再发下一条"，这一来一回的沟通本身就要花时间，哪怕 GPU 算得很快，沟通开销也不能忽略。

**用 CUDA Graph**：把这上千条指令**预先录好**，做成一份"剧本"。以后 CPU 只需要说一句"演这个剧本"，GPU 就按录好的顺序自己往下走，不用每一步都跟 CPU 请示汇报，省掉了大量来回沟通的时间。

### PIECEWISE（分段式 CUDA Graph）到底是什么

**先抛开术语，从"为什么需要 CUDA Graph"讲起**

GPU 每次计算，CPU 都要发一堆"指令"给它，比如：
- 指令1: 算矩阵乘法
- 指令2: 算加法
- 指令3: 算激活函数
- ...（一个模型一层就有几十条指令，几十层叠起来就是上千条）

没有 CUDA Graph：每条指令都是 CPU 现发一条、GPU 执行一条、再汇报"我执行完了"，然后 CPU 才发下一条。这个"发指令 → 等回复"的来回沟通，本身就要花时间（哪怕 GPU 算得很快，沟通开销也不能忽略）。

用 CUDA Graph：把这上千条指令"预先录好"，做成一个"剧本"。以后 CPU 只需要说一句"演这个剧本"，GPU 就按录好的顺序自己往下走，不用每一步都跟 CPU 汇报请示。省掉了大量来回沟通的时间。

**关键限制**：CUDA Graph 录的是"具体的操作步骤"，这些步骤是绑定"数据形状（shape）"的。剧本里录的是"处理 3 个 token 的计算"，下次来了 5 个 token，这份剧本就不适用了，得重新录。

**对照 Decode 和 Prefill 的 shape 特点：**

```
Decode 阶段：每次都是"1 个新 token 去查历史" → 形状永远是 1
             → 形状固定，可以放心录成剧本 ✅

Prefill 阶段：这次来 2000 个 token，下次来 1500 个，再下次 3800 个
             → 形状每次都不一样，没法录成一个固定剧本 ❌
```

**PIECEWISE 的思路**：模型的一次计算其实是很多层"小模块"叠起来的（Attention 层、FFN 层……）。与其"整个模型录一个剧本，要么全录要么全不录"，不如**拆成一段一段（piece by piece），能录的段落单独录好，不能录的段落就照旧现场执行**：

```
一次 forward 拆成很多小段依次执行：

段1: Attention 计算 ← Prefill 时 token 数不固定，shape 会变
                      → 不录剧本，每次现场按需执行
段2: FFN 计算       ← 跟 token 数关系不大，shape 相对固定
                      → 提前录好剧本，直接播放，很快 ⚡
段3: Attention 计算 ← 同段1，现场执行
段4: FFN 计算       ← 同段2，播放剧本
...（每一层重复这个模式）
```

所以 **PIECEWISE（分段式）** 的意思是：**不是"一整块 all-or-nothing"地用 CUDA Graph，而是把计算拆成一小块一小块，每一小块单独判断能不能用 CUDA Graph**——能用的地方享受加速，不能用的地方老实动态执行，两者混在同一次 forward 里。

> 💡 一句话记忆：CUDA Graph 是"录好的剧本"，PIECEWISE 是"只把能录的片段录成剧本，剩下的照旧现场演"。它只是一个加速手段，跟要不要开 Chunked Prefill、会不会 OOM 没有直接关系，知道它是"能提速的地方提速，不能提速的地方正常算"这么个折中方案就够了。

### 日志解读

```
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 0/3 → 1/3 → 2/3 → 3/3
```

**为什么要按 batch_size 分别录，而不是按"段"分？**

回顾一下之前讲的：CUDA Graph 要求 shape 固定。这里的"shape"不仅指 token 数，还包括同时处理几个请求（batch_size）——batch_size=4 和 batch_size=1 是两种不同的输入形状，各自要单独录一份剧本，运行时来了几个并发请求，就播放对应那张图。

而"分段（PIECEWISE）"说的是另一件事：每一张图内部，也不是把模型从头到尾整个录成一张图，而是把 Attention 部分摘出来动态算、把 FFN 等部分录成图——这是"横向"上对模型结构的切分，跟"纵向"上按 batch_size 录 3 张图是两个不同维度的概念，可以同时存在：

捕获了 3 张图，对应配置 `cudagraph_capture_sizes: [4, 2, 1]`，即 batch_size = 4、2、1 时各一张图：

```
实际推理时 batch_size 会变化：
  只有 1 个请求在 Decode → 用 batch_size=1 的图
  有 2 个请求在 Decode  → 用 batch_size=2 的图
  有 4 个请求在 Decode  → 用 batch_size=4 的图
  超过 4 个             → 退回动态执行（不用 CUDA Graph）
```

### 一张图总结

```
一个 step 的执行过程：

┌─────────────────────────────────────────────┐
│              一个 step                       │
│                                             │
│  请求A(Prefill chunk): [████ 2048 token]   │
│         +                                   │  ← mixed prefill-decode
│  请求B(Decode):        [█ 1 token]         │
│                                             │
│  执行方式：                                  │
│  ├── Attention 部分 → 动态执行              │  ← PIECEWISE 的动态段
│  └── FFN 等部分    → CUDA Graph 执行 ⚡    │  ← PIECEWISE 的图捕获段
└─────────────────────────────────────────────┘
```

> 💡 `mixed prefill-decode` 说明这个 step 同时在做输入处理和输出生成；`PIECEWISE` 说明用了分段式 CUDA Graph 来加速其中 shape 固定的部分。两者都是性能优化手段，知道是加速机制就够了。

---

## 五、PYTORCH_CUDA_ALLOC_CONF

拆开来看：

```
PYTORCH  _  CUDA  _  ALLOC  _  CONF
  ↓           ↓        ↓         ↓
PyTorch    CUDA     分配器     配置项

= "PyTorch 的 CUDA 内存分配器的配置"
```

本质上是一个**环境变量**，用来调整 PyTorch 内部显存分配器的行为，支持多个配置项用逗号分隔：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
#                               ↑ 配置项1                ↑ 配置项2
```

常用配置项：

| 配置项 | 含义 |
|--------|------|
| `expandable_segments:True` | 允许显存块按需扩展，减少碎片化 |
| `max_split_size_mb:N` | 限制单次分配的最大块大小（MB），减少显存碎片 |

---

## 六、实际案例：日志分析

```
INFO Available KV cache memory: 5.14 GiB
INFO GPU KV cache size: 21,056 tokens
INFO Maximum concurrency for 7,250 tokens per request: 2.90x
```

- 模型权重占 **62.46 GiB**
- 只剩 **5.14 GB** 给 KV Cache
- `max_model_len=7250` 时，最多同时处理 **2.9 个**并发请求

若设置 `workers=5`，vLLM 只能同时跑 2~3 个，剩余请求在等待队列中积压，叠加 Prefill 激活值峰值，容易 OOM。

**建议调整**：

```bash
--gpu-memory-utilization 0.88 \   # 从 0.94 降低，给激活值留更多空间
--max-num-seqs 3 \                # 限制最大并发序列数
--max-model-len 8192              # 够用就好，省 KV Cache 显存
```

同时将推理脚本的 `workers` 也调整为 `3`，与 `max_num_seqs` 对齐。

---

## 七、压测指标详解：吞吐量、TTFT、TPOT

对 vLLM 服务做压测时，通常会得到一张类似下面的结果表（不同并发数下的各项指标）：

| 并发 | 输出吞吐 (tok/s) | 总吞吐 (tok/s) | 平均 TTFT (ms) | P99 TTFT (ms) | 平均 TPOT (ms/token) |
|------|-----------------|---------------|---------------|---------------|---------------------|
| 1 | 41.2 | 121.8 | 104 | 121 | 24.0 |
| 2 | 81.2 | 240.2 | 72 | 75 | 24.1 |
| 4 | 158.6 | 469.1 | 75 | 93 | 24.4 |
| 8 | 257.0 | 770.4 | 601 | 7112 | 27.8 |
| 16 | 500.9 | 1494.2 | 250 | 1338 | 29.5 |
| 24 | 657.2 | 1978.4 | 302 | 1972 | 33.3 |
| 32 | 804.1 | 2412.5 | 365 | 2602 | 37.0 |
| 40 | 911.7 | 2730.1 | 416 | 3168 | 40.0 |
| 48 | 1014.9 | 3063.0 | 472 | 3567 | 43.6 |
| 64 | 1131.1 | 3404.4 | 587 | 4773 | 52.4 |

### 各项指标含义

**① 输出吞吐（Output Throughput, tok/s）**

指服务器**每秒能生成多少个输出 token**（不算输入的 prompt token）。这是衡量模型"生成速度"最直接的指标——数值越高，说明单位时间内能给用户吐出的字/词越多。比如并发=16 时是 500.9 tok/s，意味着这一秒钟，16 个并发请求加起来一共生成了约 500 个 token。

**② 总吞吐（Total Token Throughput, tok/s）**

= （输出 token 数 + 输入 prompt token 数）÷ 总耗时。因为处理输入 prompt（也就是 Prefill 阶段）也要消耗 GPU 算力，所以这个指标把输入和输出都算进去，更能反映 GPU 的整体算力利用情况。一般这个数值约等于输出吞吐的 3 倍左右（取决于 input_len 相对 output_len 的比例，input 越长这个倍数越大）。

**③ TTFT（Time To First Token，首 token 延迟）**

指从发送请求到**收到第一个返回的 token** 之间的耗时。这个指标反映的是"用户发出问题后，要等多久才能看到模型开始打字"，对交互式聊天体验的影响非常大——TTFT 越低，用户感觉响应越快。

- **平均 TTFT**：所有请求 TTFT 的平均值，反映整体水平。
- **P99 TTFT**：99% 的请求 TTFT 都低于这个值（也就是最慢的 1% 请求里最快的那个）。这个指标很关键，因为平均值容易被大多数"快请求"掩盖，而 P99 能暴露出排队等待导致的最坏情况。比如并发=8 时，平均 TTFT 只有 601ms，但 P99 高达 7112ms——说明大部分请求还好，但有一小部分请求要排队等 7 秒才开始收到回复，这就是明显的拥堵信号。

**④ TPOT（Time Per Output Token，平均单 token 生成耗时）**

指生成第一个 token 之后，**后续每个 token 之间平均间隔多久**（也就是 Decode 阶段的速度）。这个指标决定了"打字机效果"的流畅度——TPOT 越小，感觉模型"打字"越快越流畅。比如 TPOT=24ms，意味着模型平均每 24 毫秒吐一个字，换算成速度大约是 41.7 token/秒（这是单个请求的生成速度，不是吞吐量）。

随着并发数增大，TPOT 会变大（比如从 24ms 涨到 52ms），这是因为 GPU 要同时处理更多请求的解码，每个请求分到的算力和调度轮次变少了，所以单个请求的"打字"速度会变慢。

### 一个类比：把 vLLM 服务想象成一个餐厅厨房

| 指标 | 类比 |
|------|------|
| TTFT | 你点完菜后，等第一道菜端上桌的时间 |
| TPOT | 上完第一道菜后，后续每道菜之间的间隔 |
| 输出吞吐 | 整个厨房每分钟能出多少道菜（不管是哪桌的） |
| 总吞吐 | 厨房处理的总"工作量"（备菜 + 炒菜都算） |
| 并发数 | 同时有多少桌客人在点餐等菜 |

客人少（并发低）时，厨师专心伺候你一桌，上菜快（TTFT 低）、菜与菜间隔短（TPOT 低）；客人一多（并发高），厨师要来回切换做很多桌的菜，虽然厨房整体出菜总量（吞吐）上去了，但轮到你这桌，等待和上菜间隔都会变长。

> 💡 压测的本质，就是要找到一个并发数，既能让厨房（GPU）尽量不闲着、多出活（吞吐高），又不会让客人（用户）等太久（延迟可接受）。

### 结合实测数据看关键拐点

从上面的表格可以看出几个规律：

```
① 吞吐量：一直到并发 64 都还在稳步增长，没有出现明显下降拐点
   → 说明服务器还没被压垮，但增长曲线已经开始变缓（边际收益递减）

② 延迟的关键拐点在并发=8附近：
   并发 ≤4 时：TTFT 很低（<100ms），TPOT 稳定在 24ms 左右，几乎没有排队
   并发=8 开始：TTFT 暴增（P99 从 93ms 直接跳到 7112ms）
   → 请求开始在调度队列排队等待，说明此时已经出现 KV cache 抢占/排队现象
   → 排队瓶颈出现得比理论并发上限早，更可能是调度策略或
     max_num_batched_tokens 的 Chunked Prefill 限制导致

③ TPOT 随并发数增大而线性上升：从 24ms（并发1）涨到 52ms（并发64）
   → 解码阶段批处理变大后，单 token 生成变慢，这是正常现象
   → 因为 batch 内的请求在共享同一份 GPU 算力
```

### 根据场景选择并发数

| 场景 | 建议并发 | 理由 |
|------|---------|------|
| 低延迟优先（交互式对话） | ≤ 4 | TTFT < 100ms，TPOT 最低，体验最流畅 |
| 吞吐与延迟的平衡（多数生产场景） | 16 ~ 24 | TTFT 均值 250~300ms 可接受，吞吐已达单请求的 12~16 倍 |
| 最大吞吐优先（离线批处理，不在乎延迟） | 48 ~ 64+ | P99 TTFT 会达到 3.5~4.8 秒，只适合非交互式场景 |

> ⚠️ **注意**：以上数据基于纯文本、`input_len≈512`、`output_len≈256` 的短对话场景。如果实际业务是**图文混合**（VL 模型的核心用途）或更长上下文，结果会不同——图像 token 的视觉编码开销会显著拉长 TTFT，实际能承受的并发通常会更低。压测时应尽量使用贴近真实业务的 prompt 长度分布。
