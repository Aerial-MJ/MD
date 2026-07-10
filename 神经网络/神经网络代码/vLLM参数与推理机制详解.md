# vLLM 参数与推理机制详解

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

时间线：
step1: [请求A Prefill 前 2048 token ████] + [请求B Decode token1]
step2: [请求A Prefill 中 2048 token ████] + [请求B Decode token2]
step3: [请求A Prefill 后 1904 token ███ ] + [请求B Decode token3]
step4: [请求A Decode token1]              + [请求B Decode token4]
```

**好处**：
- ✅ 请求 B 不用等请求 A 的 Prefill 全部完成
- ✅ 每个 step 的 token 数不超过 2048，激活值峰值可控
- ✅ 整体吞吐量更高

---

## 四、mixed prefill-decode 与 PIECEWISE

### mixed prefill-decode

这是 Chunked Prefill 的直接体现：开启 Chunked Prefill 后，一个 step 里可以**同时包含** Prefill chunk 和 Decode token，即"混合执行"。

```
日志：Chunked prefill is enabled with max_num_batched_tokens=2048
                ↓
每个 step 最多 2048 token，里面可以混合 Prefill 和 Decode
                ↓
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)
```

### CUDA Graph 简介

**没有 CUDA Graph 时**：CPU 每个 step 都要反复向 GPU 发指令并等待响应，通信开销大。

**有 CUDA Graph 时**：第一次把整个 step 的 GPU 指令录制成一张"图"，之后 CPU 只需说"执行这张图"，GPU 一气呵成，大幅降低 CPU-GPU 通信开销。

### PIECEWISE（分段式 CUDA Graph）

**问题**：CUDA Graph 只能捕获**固定 shape** 的计算：
- Decode 阶段：每次只有 1 个新 token，shape 固定 → 好捕获 ✅
- Prefill 阶段：每次 token 数不同，shape 动态 → 难捕获 ❌

**PIECEWISE 解决方案**：把模型计算图切成两段分别处理：

```
┌─────────────────────────────────┐
│ Attention 部分（shape 动态）      │ → 不用 CUDA Graph，动态执行
└─────────────────────────────────┘
┌─────────────────────────────────┐
│ FFN / 其他部分（shape 相对固定） │ → 用 CUDA Graph 捕获，快速执行 ⚡
└─────────────────────────────────┘
```

所以叫 **PIECEWISE = 分段捕获**，不是整个 step 一张图。

### 日志解读

```
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 0/3 → 1/3 → 2/3 → 3/3
```

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
