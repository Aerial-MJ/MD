# Transformer 向量与参数详解

> 关联笔记：本文的参数命名可对照 [MiniMind从0到1构建大模型 · 3 模型结构](../神经网络代码/MiniMind从0到1构建大模型.md#3模型结构) 中的实际代码实现（GQA、RoPE、SwiGLU FFN）。

---

## 一、两大类划分

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
| 位置嵌入 | Positional Embedding `PE` | `[max_seq_len, d_model]` | 给每个位置加位置信息 |

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

```
token_id
  ↓  [查表]
E ──────────────────────── Token Embedding    [vocab, d_model]
PE ─────────────────────── Pos Embedding      [seq_len, d_model]
  ↓  [相加]
  x                                           [seq_len, d_model]
  ↓
  ┌──────────────────────── Transformer Block × N层 ─────────────────────────┐
  │                                                                           │
  │  ① LayerNorm (γ, β)          ← Pre-LN：先 Norm 再 Attention              │
  │       ↓                                                                   │
  │  W_Q, W_K, W_V ────────── Attention权重   [d_model, d_k/v]               │
  │       ↓  [attention计算]                                                  │
  │  W_O ──────────────────── Output投影      [d_v×heads, d_model]            │
  │       ↓                                                                   │
  │  x = x + Attn_output        ← 残差（绕过了上面的 LN，直接加回原始 x）     │
  │       ↓                                                                   │
  │  ② LayerNorm (γ, β)          ← Pre-LN：先 Norm 再 FFN                    │
  │       ↓                                                                   │
  │  W1/gate_proj ─────────── FFN升维         [d_model, d_ffn]                │
  │  W2/down_proj ─────────── FFN降维         [d_ffn, d_model]                │
  │       ↓                                                                   │
  │  x = x + FFN_output         ← 残差（绕过了上面的 LN，直接加回）           │
  │                                                                           │
  └───────────────────────────────────────────────────────────────────────────┘
  ↓
Final LayerNorm (γ, β)     ← 单独再 Norm 一次！因为最后的 x 没被 Norm 过
  ↓                           （Post-LN 没有这步，Pre-LN 必须有）
lm_head ────────────────── LM Head            [d_model, vocab]
  ↓
logits                                        [vocab_size]
  ↓  [softmax]
probs                                         [vocab_size]
  ↓  [-log(P_correct)]
Loss（标量）
  ↓  [backward]
所有参数的梯度
  ↓  [optimizer.step]
参数更新
```

**一句话：** token → embedding → N层(attention+FFN) → lm_head → logits → softmax → 概率 → -log → loss → backward → 更新所有权重矩阵。

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
