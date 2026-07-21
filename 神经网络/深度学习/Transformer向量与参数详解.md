# Transformer 向量与参数详解

> 关联笔记：本文的参数命名可对照 [MiniMind从0到1构建大模型 · 3 模型结构](../神经网络代码/MiniMind从0到1构建大模型.md#3模型结构) 中的实际代码实现（GQA、RoPE、SwiGLU FFN）。

---

## 零、最顶层的分类：先分「参数」和「激活」，再谈其他

在细分"权重矩阵"和"嵌入矩阵"之前，有一个更上层、更重要的分类必须先分清楚，否则后面所有的名词都会绕在一起：**Transformer 里的所有"矩阵/向量"，先分成参数（parameters）和激活（activations）两大类，参数是"练出来存起来的"，激活是"每次 forward 临时算出来的"。**

### 判断标准：只问一句话

```
拿到一个东西（比如 Q vector、W_Q、hidden state、KV cache……），
只需要问自己一个问题：

  "换一个输入句子，这个东西的值会不会变？"

  会变     → 它是 activation（激活 / 中间计算结果）
  不会变   → 它是 parameter（参数 / 权重）
             （唯一能让它变的方式是"训练更新"，而不是"换个输入句子"）
```

拿几个具体例子过一遍这个判断标准：

```
W_Q（Query 权重矩阵）：
  换一句话输入，W_Q 的数值变不变？→ 不变（除非你在训练、做了一次 backward+step）
  → W_Q 是 parameter

Q（Query 向量，= x @ W_Q）：
  换一句话输入，Q 的数值变不变？→ 变！因为 x 变了，Q = x @ W_Q 自然跟着变
  → Q 是 activation

Token Embedding 矩阵 E（比如 [vocab_size, d_model] 那张大表）：
  换一句话输入，E 这张表本身变不变？→ 不变，E 是固定存好的一张表
  → E 是 parameter
```

### 完整分类图

```
Transformer 里的所有"东西"
        │
   ┌────┴────┐
   │         │
参数(parameters)              激活(activations)
训练时被更新                   每次 forward 临时产生
保存在 checkpoint 里            用完即可丢弃（除了 KV Cache 会被主动缓存）
   │                              │
   ├── Token Embedding 矩阵 E      ├── Embedding 向量（E[id] 查表结果）
   ├── Positional Embedding PE     ├── Hidden state（input_embedding/hidden_1/hidden_2……，见七）
   │   （若可学习；RoPE 无参数）      ├── Q / K / V 向量
   ├── W_Q / W_K / W_V / W_O       ├── Attention Score / 权重矩阵（softmax 后）
   ├── W1 / W2 / W_gate（FFN）      ├── Attention output
   ├── LM Head（lm_head.weight）    ├── FFN 中间结果 / FFN output
   ├── LayerNorm 的 γ / β          ├── KV Cache（被缓存的 K/V，本质仍是激活）
   └── 各类 bias（若存在）           └── Logits / Softmax 概率
```

**一张对照表，把"是不是参数"和"训练方式"串起来**：

| 名字 | 是参数还是激活 | 会不会出现在 checkpoint 里 |
|------|---------------|---------------------------|
| Token Embedding 矩阵 `E` | 参数 | ✅ 会 |
| `E[id]`（embedding 向量） | 激活 | ❌ 不会（每次要用重新查） |
| `W_Q / W_K / W_V / W_O` | 参数 | ✅ 会 |
| `Q / K / V` 向量 | 激活 | ❌ 不会 |
| Hidden state | 激活 | ❌ 不会 |
| KV Cache | 激活（但推理时会被主动缓存复用） | ❌ 不会（只在推理过程中临时存在显存里） |
| LM Head `lm_head.weight` | 参数 | ✅ 会（有时和 `E` 共享，见下文 Weight Tying） |
| Logits / probs | 激活 | ❌ 不会 |


**参数**

```
参数(parameters)
|
├── Embedding 参数
│      ├── Token embedding
│      ├── Position embedding（如果有）
│      └── Output embedding / LM head
│
├── Attention 参数
│      ├── Wq
│      ├── Wk
│      ├── Wv
│      └── Wo
│
├── FFN 参数
│      ├── W1
│      ├── W2
│      └── W3（SwiGLU）
│
├── Norm 参数
│      ├── LayerNorm weight
│      └── LayerNorm bias（部分模型）
│
└── Bias（如果存在）
```

```
中间向量(activation)
|
├── input embedding
├── hidden states
├── query vector
├── key vector
├── value vector
├── attention output
├── FFN output
└── KV cache
```

### 参数内部还能再细分：矩阵乘法型 vs 查表型

上面把"参数"和"激活"分清楚之后，"参数"这个大类内部还可以再按**用法**细分成两种，这就是下一节要讲的内容——这一层细分和"参数 vs 激活"是两个不同维度的分类，不冲突，是嵌套关系：

```
参数(parameters)              ← 第一层分类：是否训练
    │
    ├── 矩阵乘法型权重          ← 第二层分类：怎么用
    │     W_Q, W_K, W_V, W_O, W1, W2, lm_head
    │     用法：output = x @ W
    │
    └── 查表型权重（嵌入矩阵）
          Token Embedding E, 可学习式 Positional Embedding PE
          用法：output = E[id]
```

---

## 一、两大类划分（承接上面：参数内部的两种用法）

```
Transformer 里所有参数，本质上都是矩阵（或向量），
按"用法"可以分成两类，而不是按"是不是矩阵"来分：

┌─────────────────────────────────────────────────┐
│  变换型权重矩阵 (Weight Matrix)                  │
│  → 用于矩阵乘法做线性变换：output = x @ W        │
│  → 形状是 [d_in, d_out]                          │
│  → 比如 W_Q, W_K, W_V, W_O, W_FFN                │
├─────────────────────────────────────────────────┤
│  嵌入矩阵 / 查找表 (Embedding Matrix / Lookup)   │
│  → 本质也是一个可训练的参数矩阵                  │
│  → 但用法是"按 id 查表取行"：E[token_id]         │
│    而不是矩阵乘法（等价于和 one-hot 向量相乘）    │
│  → 形状是 [num_entries, d_model]                 │
│  → 比如 Token Embedding E, Pos Embedding PE      │
│  → 表中的某一行 E[i] 才是真正的"嵌入向量"        │
└─────────────────────────────────────────────────┘
```

> ⚠️ 注意：`E` 本身是矩阵，不是向量。之前笼统地把 Embedding 归为"向量"不够严谨，
> 更准确的说法是：`E` 是一个嵌入矩阵（查找表），从中查出的某一行才是嵌入向量。
> `E` 和 `W_Q` 本质上都是可训练的权重矩阵，区别只在于**用法**（查表 vs 矩阵乘法）。

---

## 二、完整参数列表（从输入到输出）

### 第一层：输入层

| 名字 | 英文全称 | 形状 | 作用 |
|------|---------|------|------|
| 词嵌入矩阵 | Token Embedding / Word Embedding `E` | `[vocab_size, d_model]` | 把 token id 变成向量 |
| 位置嵌入矩阵 | Positional Embedding `PE` | `[max_seq_len, d_model]` | 给每个位置加位置信息 |

```
输入：token id = 42
  → 查 Embedding 表第42行
  → 得到一个 d_model 维的向量
  → 加上位置编码
  → 得到 x ∈ R^d_model
```

---

### 第二层：Transformer Block（重复 N 次）

#### 2.1 Layer Norm

| 名字 | 英文 | 形状 | 作用 |
|------|------|------|------|
| 缩放参数 | `gamma` / `weight` | `[d_model]` | 归一化后的缩放 |
| 偏移参数 | `beta` / `bias` | `[d_model]` | 归一化后的偏移 |

#### 2.2 Self-Attention

| 名字 | 英文全称 | 形状 | 作用 |
|------|---------|------|------|
| Query 权重 | `W_Q` / `query.weight` | `[d_model, d_k]` | 把 x 变成 Query |
| Key 权重 | `W_K` / `key.weight` | `[d_model, d_k]` | 把 x 变成 Key |
| Value 权重 | `W_V` / `value.weight` | `[d_model, d_v]` | 把 x 变成 Value |
| Output 权重 | `W_O` / `out_proj.weight` | `[d_v × n_heads, d_model]` | 把多头结果投影回去 |
| Q/K/V bias | `q_bias`, `k_bias`, `v_bias` | `[d_k]` | 偏置（有些模型没有） |

```
计算过程：
  Q = x @ W_Q        [seq_len, d_k]
  K = x @ W_K        [seq_len, d_k]
  V = x @ W_V        [seq_len, d_v]

  scores = Q @ K^T / sqrt(d_k)   [seq_len, seq_len]
  attn   = softmax(scores)        [seq_len, seq_len]
  out    = attn @ V               [seq_len, d_v]

  output = out @ W_O              [seq_len, d_model]
```

#### 2.3 FFN（前馈网络）

| 名字 | 英文全称 | 形状 | 作用 |
|------|---------|------|------|
| 第一层权重 | `W1` / `fc1.weight` / `up_proj` | `[d_model, d_ffn]` | 升维 |
| 第二层权重 | `W2` / `fc2.weight` / `down_proj` | `[d_ffn, d_model]` | 降维 |
| Gate 权重（SwiGLU） | `W_gate` / `gate_proj` | `[d_model, d_ffn]` | 门控（LLaMA/Qwen 用） |
| 偏置 | `b1`, `b2` | `[d_ffn]`, `[d_model]` | 偏置（有些模型没有） |

```
标准 FFN：
  h = GELU(x @ W1 + b1)
  output = h @ W2 + b2

SwiGLU（LLaMA/Qwen）：
  h = SiLU(x @ W_gate) × (x @ W1)
  output = h @ W2
```

---

### 第三层：输出层

| 名字 | 英文全称 | 形状 | 作用 |
|------|---------|------|------|
| LM Head | `lm_head.weight` | `[d_model, vocab_size]` | 把向量变成词表概率 |

```
⚠️ 重要：LM Head 通常和 Token Embedding 共享权重！
  lm_head.weight = E.T   （转置）
  这叫 Weight Tying
```

---

## 三、完整前向传播流程（穿起来）

```
输入：["The", "answer", "is"]
  ↓
① Token ID
  "The"=1234, "answer"=5678, "is"=910

  ↓
② Embedding 查表
  x = E[token_ids]          shape: [3, d_model]
  x = x + PE[:3]            加位置编码

  ↓
③ Transformer Block × N层
  ┌─────────────────────────────┐
  │  x = LayerNorm(x)           │
  │  x = x + SelfAttention(x)  │  残差连接
  │  x = LayerNorm(x)           │
  │  x = x + FFN(x)            │  残差连接
  └─────────────────────────────┘

  ↓
④ 最后一层 LayerNorm
  x = LayerNorm(x)           shape: [3, d_model]

  ↓
⑤ LM Head
  logits = x @ lm_head.weight  shape: [3, vocab_size]

  比如 vocab_size = 50000
  → 每个位置得到 50000 个数字

  ↓
⑥ 取最后一个位置的 logits
  logits[-1]                 shape: [vocab_size]
  = [0.1, -2.3, 4.5, 1.2, ...]  ← 50000个原始分数

  ↓
⑦ Softmax → 概率分布
  probs = softmax(logits[-1])
  = [0.001, 0.0003, 0.32, ...]  ← 加起来=1

  ↓
⑧ 预测下一个 token = argmax 或 采样
  → "42"
```

---

## 四、Logits 是什么

```
logits = x @ lm_head.weight

x:              [d_model]        比如 [4096]
lm_head.weight: [d_model, vocab] 比如 [4096, 50000]
logits:         [vocab_size]     比如 [50000]

logits 就是：
  每个 token 的"原始得分"
  还没有经过 softmax
  可以是任意实数（负数也行）

经过 softmax 之后才变成概率：
  P(token_i) = exp(logits_i) / sum(exp(logits))
```

---

## 五、怎么算 Loss（SFT 为例）

```
真实下一个 token = "42"，对应 id = 8765

logits = [0.1, -2.3, 4.5, ..., 6.8, ...]
                                 ↑
                           id=8765 位置

Step 1：softmax
  probs = softmax(logits)
  P("42") = exp(6.8) / sum(exp(all)) = 0.72

Step 2：取真实 token 的概率
  P_correct = 0.72

Step 3：算 cross entropy loss
  L = -log(0.72) = 0.329

Step 4：对所有 token 位置取平均
  L_seq = mean([L_t1, L_t2, ..., L_tT])
```

---

## 六、哪些参数可以被训练

| 参数 | 全量微调 | LoRA | 冻结训练 |
|------|---------|------|---------|
| Token Embedding `E` | ✅ | 可选 | ❌ |
| Positional Embedding | ✅ | ❌ | ❌ |
| `W_Q`, `W_K`, `W_V`, `W_O` | ✅ | ✅ 加低秩矩阵 | ❌ |
| `W1`, `W2`, `W_gate` | ✅ | ✅ 可选 | ❌ |
| LayerNorm `γ`, `β` | ✅ | ✅ 通常训练 | ❌ |
| LM Head | ✅ | 可选 | ❌ |

### LoRA 怎么训练

```
原始权重 W（冻结不动）
加上低秩矩阵 ΔW = A @ B

  A: [d_model, r]   r 很小，比如 r=8
  B: [r, d_model]

实际计算：
  output = x @ (W + A@B)
         = x@W + x@A@B
           ↑冻结   ↑只训练这两个小矩阵

参数量：
  原始 W：4096×4096 = 16M 参数
  A+B：4096×8 + 8×4096 = 65K 参数  ← 只有原来的0.4%
```

---

## 七、一张图总结所有向量

> 图中每一步产生的激活都标了具体名字（不再统一叫 `x`），左边是**参数**（不随输入变化），右边箭头上标的是**激活**（每次 forward 随输入重新算出来的东西），对照着 [零、参数 vs 激活](#零最顶层的分类先分参数和激活再谈其他) 的判断标准看会更清楚。

```
token_id
  ↓  [查表：E[token_id]]
E ──────────────────────── Token Embedding 参数  [vocab, d_model]
PE ─────────────────────── Pos Embedding 参数    [seq_len, d_model]
  ↓  [两者相加]
  input_embedding ←────────────────────────────── 激活①：输入嵌入   [seq_len, d_model]
  ↓
  ┌──────────────────────── Transformer Block × N层 ─────────────────────────────────┐
  │                                                                                   │
  │  ① LayerNorm (γ, β)          ← Pre-LN：先 Norm 再 Attention                      │
  │       ↓                                                                           │
  │  normed_1 ←──────────────────────────────────────── 激活②：Norm 后的隐状态         │
  │       ↓                                                                           │
  │  W_Q, W_K, W_V 参数 ────── Attention权重   [d_model, d_k/v]                       │
  │       ↓  [矩阵乘法]                                                                │
  │  Q, K, V ←───────────────────────────────────────── 激活③④⑤：Query/Key/Value 向量 │
  │       ↓  [QK^T / √d_k → softmax]                                                  │
  │  attn_weights ←──────────────────────────────────── 激活⑥：Attention 权重矩阵      │
  │       ↓  [attn_weights @ V]                                                       │
  │  attn_output ←───────────────────────────────────── 激活⑦：多头拼接后的注意力输出   │
  │       ↓  [@ W_O]                                                                  │
  │  attn_out_proj ←─────────────────────────────────── 激活⑧：Output 投影后的结果     │
  │       ↓                                                                           │
  │  hidden_1 = input_embedding + attn_out_proj  ←────── 激活⑨：残差相加后的隐状态       │
  │       (残差：绕过了①的 LN，直接把最初的 input_embedding 加回来)                     │
  │       ↓                                                                           │
  │  ② LayerNorm (γ, β)          ← Pre-LN：先 Norm 再 FFN                            │
  │       ↓                                                                           │
  │  normed_2 ←──────────────────────────────────────── 激活⑩：Norm 后的隐状态         │
  │       ↓                                                                           │
  │  W1/gate_proj, W2/down_proj 参数 ── FFN权重  [d_model, d_ffn] / [d_ffn, d_model]   │
  │       ↓  [矩阵乘法 + 激活函数]                                                      │
  │  ffn_output ←────────────────────────────────────── 激活⑪：FFN 输出               │
  │       ↓                                                                           │
  │  hidden_2 = hidden_1 + ffn_output  ←──────────────── 激活⑫：本层最终输出的隐状态     │
  │       (残差：绕过了②的 LN，直接把 hidden_1 加回来)                                  │
  │                                                                                   │
  └───────────────────────────────────────────────────────────────────────────────────┘
  ↓  hidden_2 作为下一层的输入，重复 N 次上面整个 Block，
  ↓  每一层都会产生自己的一整套 normed_1/Q/K/V/attn_output/hidden_1/hidden_2……（激活不跨层复用）
  ↓
Final LayerNorm (γ, β)     ← 单独再 Norm 一次！因为最后的 hidden_2 没被 Norm 过
  ↓                           （Post-LN 没有这步，Pre-LN 必须有）
final_hidden ←────────────────────────────────────────── 激活⑬：最终隐状态  [seq_len, d_model]
  ↓
lm_head 参数 ────────────── LM Head            [d_model, vocab]
  ↓  [矩阵乘法]
logits ←──────────────────────────────────────────────── 激活⑭：未归一化得分  [vocab_size]
  ↓  [softmax]
probs ←───────────────────────────────────────────────── 激活⑮：概率分布     [vocab_size]
  ↓  [-log(P_correct)]
Loss（标量，也是激活，只是不再往下传）
  ↓  [backward]
所有参数（E, PE, W_Q...W_O, W1/W2/W_gate, γ/β, lm_head）的梯度
  ↓  [optimizer.step]
参数更新（只更新左边的参数，右边这些激活每次 forward 都会被重新算出来，不会被保存）
```

**一句话：** token → **input_embedding**（激活①）→ N 层反复产生 **normed → Q/K/V → attn_weights → attn_output → hidden_1 → normed → ffn_output → hidden_2**（激活②~⑫，每层都是全新的一套，不复用上一层的）→ **final_hidden**（激活⑬）→ **logits**（激活⑭）→ **probs**（激活⑮）→ -log → loss → backward → 更新左边那些参数矩阵（`E/PE/W_Q/W_K/W_V/W_O/W1/W2/γ/β/lm_head`）。

> **命名说明**：这里的 `hidden_1`、`hidden_2` 特指"本层内部第 1 次 / 第 2 次残差相加之后"的状态，是本文为了教学标注方便起的编号，不是学术界统一叫法（论文里更常见的是按层号写成 $h_l$，一层一个）。如果你在别的资料里看到 $h_l$ 这种写法，$l$ 指的是"第几层"，而这里的 `hidden_1/hidden_2` 指的是"层内部第几步"，两套编号维度不同，注意别混用。

---

## 八、Decoder 的三种变体（为什么你看到的结构不一样）

```
Transformer 有三种主流架构，Decoder 的结构各不相同：

┌─────────────────────┬──────────────────────────┬──────────────────────────────────┐
│ 架构                │ 代表模型                  │ Decoder Block 结构               │
├─────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ Encoder-Decoder     │ 原版Transformer、T5、BART │ Masked Self-Attn                 │
│ （原始论文）        │                           │ → Cross-Attn（多了这层！）       │
│                     │                           │ → FFN                            │
├─────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ Decoder-Only        │ GPT、LLaMA、Qwen          │ Masked Self-Attn                 │
│ （本笔记描述的）    │                           │ → FFN（无 Cross-Attn）           │
├─────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ Encoder-Only        │ BERT                      │ （双向）Self-Attn → FFN          │
└─────────────────────┴──────────────────────────┴──────────────────────────────────┘
```

### 原始论文 Decoder（3个子层，Post-LN）

```
输入（目标序列，如翻译的目标语言）
  ↓
① Masked Self-Attention   ← 只看自己和之前的 token（因果遮掩）
  x = LayerNorm(x + Attn(x))    ← Post-LN：残差之后才 Norm
  ↓
② Cross-Attention          ← Q 来自 decoder 当前层，K/V 来自 Encoder 的输出
  x = LayerNorm(x + CrossAttn(x, enc_output))
  ↓
③ FFN
  x = LayerNorm(x + FFN(x))
  ↓
输出

关键：Cross-Attention 是为了"看"Encoder 端的源语言信息
     比如翻译任务：Encoder 处理英文，Decoder 生成中文时通过 Cross-Attn 参考英文
```

### 现代 LLM Decoder（2个子层，Pre-LN，本笔记用的）

```
输入（前面已有的 token）
  ↓
① Masked Self-Attention   ← 因果遮掩，只看过去
  x = x + Attn(LayerNorm(x))    ← Pre-LN：先 Norm 再 Attention，残差绕过 LN
  ↓
② FFN
  x = x + FFN(LayerNorm(x))     ← 同上
  ↓
输出

关键：没有 Cross-Attention！因为根本没有 Encoder
     整个模型只有 Decoder，自己生成自己
```

### Post-LN vs Pre-LN 对比

```
Post-LN（原始论文）：
  x → Attention → 残差 → LayerNorm → FFN → 残差 → LayerNorm → 输出
                                ↑LN在后                ↑LN在后

Pre-LN（现代LLM）：
  x → LayerNorm → Attention → 残差 → LayerNorm → FFN → 残差 → 输出
      ↑LN在前                         ↑LN在前

区别：
  LN 总共出现两次，只是挪了位置，数量相同
  Pre-LN 训练更稳定，不容易梯度爆炸，所以现代 LLM 普遍采用
  Pre-LN 因为残差绕过了 LN，所以 N 层结束后需要额外一个 Final LayerNorm
```

---

## 九、Multi-Head Attention（MHA）详解

### 9.1 核心直觉

```
单头 Attention：一个人同时只能关注一件事
Multi-Head：分出 h 个"头"，每个头独立关注不同的信息，最后把结果拼起来

比如：
  头1 可能在关注"语法依存关系"
  头2 可能在关注"指代关系"（它/他/她 指的是谁）
  头3 可能在关注"局部相邻词"
  ...
每个头学到的模式不同，最终融合更丰富
```

### 9.2 完整计算流程

```
输入 X: [seq_len, d_model]     例如 [10, 512]
超参数：头数 h = 8，每头维度 d_k = d_model / h = 64

━━━ 概念版（最直观，h 个独立矩阵）━━━

对每一个头 i = 1, 2, ..., h：
  Q_i = X @ W_Q_i    W_Q_i: [d_model, d_k]  → Q_i: [seq_len, d_k]
  K_i = X @ W_K_i    W_K_i: [d_model, d_k]  → K_i: [seq_len, d_k]
  V_i = X @ W_V_i    W_V_i: [d_model, d_v]  → V_i: [seq_len, d_v]

  scores_i = Q_i @ K_i^T / sqrt(d_k)        [seq_len, seq_len]
  attn_i   = softmax(scores_i)               [seq_len, seq_len]
  head_i   = attn_i @ V_i                    [seq_len, d_v]

把所有头拼起来：
  concat = Concat(head_1, ..., head_h)       [seq_len, h*d_v]  = [seq_len, d_model]

最后投影（融合多头信息）：
  output = concat @ W_O                      [seq_len, d_model]
  W_O: [d_model, d_model]
```

### 9.3 为什么你看到"不同版本"——两种实现方式

**版本A（概念版，h 个独立矩阵，直观但低效）**

```python
head_outputs = []
for i in range(h):
    Q = X @ W_Q[i]   # W_Q[i]: [d_model, d_k]
    K = X @ W_K[i]
    V = X @ W_V[i]
    scores = Q @ K.T / sqrt(d_k)
    attn = softmax(scores)
    head_outputs.append(attn @ V)
output = concat(head_outputs) @ W_O
```

**版本B（工程实现版，合并大矩阵 + reshape，高效）**

```python
# 用一个大矩阵一次算出所有头的 Q/K/V
Q = X @ W_Q   # W_Q: [d_model, h*d_k]，等价于把 h 个 W_Q_i 横向拼接
K = X @ W_K
V = X @ W_V

# reshape 成多头形式（分拆出每个头）
Q = Q.view(seq_len, h, d_k).transpose(1, 2)  # [h, seq_len, d_k]
K = K.view(seq_len, h, d_k).transpose(1, 2)
V = V.view(seq_len, h, d_v).transpose(1, 2)

# 批量并行计算所有头的 attention
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)  # [h, seq_len, seq_len]
attn   = softmax(scores)                        # [h, seq_len, seq_len]
out    = attn @ V                               # [h, seq_len, d_v]

# 拼回来
out    = out.transpose(1, 2).contiguous().view(seq_len, h * d_v)  # [seq_len, d_model]
output = out @ W_O                              # [seq_len, d_model]
```

```
两个版本数学上完全等价！
版本B只是工程优化：一次大矩阵乘法 + reshape，比 h 次循环快得多
实际代码（PyTorch、HuggingFace）用的都是版本B
```

### 9.4 几个关键细节

```
问题                      答案
─────────────────────────────────────────────────────────────────
d_k 怎么定？              通常 d_k = d_model / h
                          这样 h 个头 concat 后刚好还是 d_model 维

为什么除以 sqrt(d_k)？    d_k 越大，点积结果的方差越大
                          除以 sqrt(d_k) 把方差压回1，防止 softmax 梯度消失

W_O 的作用？              把 concat 后的 [h*d_v] 维向量投影回 d_model
                          让各个头的信息混合融合，而不是简单拼接

Cross-Attention 怎么变？  Q 来自 decoder 当前层的输入
                          K、V 来自 encoder 的输出
                          其余计算完全一样
```

### 9.5 一张图理清 MHA

```
                ┌──── 头1: W_Q1,W_K1,W_V1 → Attn → head_1 ────┐
                │                                                │
X ──────────────┼──── 头2: W_Q2,W_K2,W_V2 → Attn → head_2 ────┼── Concat ── W_O ── 输出
                │                                                │
                └──── 头h: W_Qh,W_Kh,W_Vh → Attn → head_h ────┘

工程实现：
  W_Q = [W_Q1 | W_Q2 | ... | W_Qh]  拼成 [d_model, h*d_k] 的大矩阵
  一次矩阵乘法得到所有头的 Q，再 reshape 成 [h, seq_len, d_k]
  K、V 同理
```
