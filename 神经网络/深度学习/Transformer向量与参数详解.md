# Transformer 向量与参数详解

---

## 一、两大类划分

```
Transformer 里所有参数，本质上就两类：

┌─────────────────────────────────────────┐
│  权重矩阵 (Weight Matrix)               │
│  → 做变换用的，形状是 [d_in, d_out]     │
│  → 比如 W_Q, W_K, W_V, W_O, W_FFN      │
├─────────────────────────────────────────┤
│  嵌入向量 (Embedding Vector)            │
│  → 把离散符号映射成连续向量             │
│  → 比如 Token Embedding, Pos Embedding  │
└─────────────────────────────────────────┘
```

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
LayerNorm (γ, β)
  ↓
W_Q, W_K, W_V ─────────── Attention权重      [d_model, d_k/v]
  ↓  [attention计算]
W_O ───────────────────── Output投影          [d_v×heads, d_model]
  ↓  [残差+LayerNorm]
W1/gate_proj ──────────── FFN升维             [d_model, d_ffn]
W2/down_proj ──────────── FFN降维             [d_ffn, d_model]
  ↓  [残差]
  × N层
  ↓
LayerNorm (γ, β)
  ↓
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
