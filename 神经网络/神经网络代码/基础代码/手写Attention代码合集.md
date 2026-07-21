# 手写 Attention 代码合集

> 从 SelfAttention → MultiHeadAttention → MultiQueryAttention（MQA）→ DeepSeek MLA + RoPE，一步步进阶的手写实现合集，附 Softmax 数值稳定实现。
>
> 关联笔记：
> - 面试向的四重境界版本（更详细的注释和演进思路）：[手写 Self-Attention 的四重境界](../../../论文/docs/Hand%20on%20Code/手写%20Self-Attention%20的四重境界，从%20self-attention%20到%20multi-head-self-attention.md)
> - 权重矩阵形状定义：[Transformer向量与参数详解](../../深度学习/Transformer向量与参数详解.md)
> - MLA 原理与公式角度解读：[MLA(2)：从代码和公式角度理解 DeepSeek MLA 的矩阵吸收](../../../论文/docs/大模型/MLA(2)：从代码和公式角度理解%20DeepSeek%20MLA%20的矩阵吸收%20(Projection%20Absorption).md)

---

## 一、SelfAttention（基础版）

最基础的单头 Self-Attention 实现，直接对着公式写：$SelfAttention(X) = softmax(\frac{Q K^T}{\sqrt{d}}) V$

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        # 定义生成 Q, K, V 的三个线性变换矩阵
        # 相当于公式里的 W_q, W_k, W_v
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # 输入 x 的形状通常是: (batch_size, seq_len, embed_dim)
        # 第一步: 通过线性层，生成 Q, K, V
        Q = self.W_q(x)  # 形状: (batch_size, seq_len, embed_dim)
        K = self.W_k(x)  # 形状: (batch_size, seq_len, embed_dim)
        V = self.W_v(x)  # 形状: (batch_size, seq_len, embed_dim)

        # 第二步: 计算注意力原始分数 (Q 乘以 K 的转置)
        # 注意: 只能转置最后两个维度 (seq_len 和 embed_dim), 不能动 batch 维度
        K_transposed = K.transpose(-2, -1)
        scores = torch.matmul(Q, K_transposed)  # 形状: (batch_size, seq_len, seq_len)

        # 第三步: 缩放 (除以根号下 d_k)
        d_k = self.embed_dim
        scores = scores / math.sqrt(d_k)

        # 第四步: 计算 Softmax，得到注意力权重
        # 在最后一维度 (seq_len) 上进行归一化，保证每个词对其他词的注意力之和为 1
        attn_weights = torch.softmax(scores, dim=-1)

        # 第五步: 将权重作用于 V，得到最终输出
        output = torch.matmul(attn_weights, V)  # 形状: (batch_size, seq_len, embed_dim)

        return output
```

**关键点回顾**：

- Q/K/V 都是同一个输入 `x` 经过三个不同的线性层得到的
- `transpose(-2, -1)` 只转置最后两个维度，保留 batch 维度不动
- 除以 `sqrt(d_k)` 是为了防止点积结果方差过大，导致 softmax 梯度消失

---

## 二、MultiHeadAttention（标准多头注意力）

在 SelfAttention 基础上引入多头机制：把 `embed_dim` 拆成 `num_heads` 份，每个头独立计算 attention，最后拼接再过一个输出投影层。

```python
# MultiHeadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        # B = batch size, L = 序列长度, D = embedding 维度

        # 1. 线性投影
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # 2. reshape 分头: (B, L, D) -> (B, num_heads, L, d_k)
        Q = Q.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        # 3. 计算 attention score
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 4. mask（可选，用于 padding mask 或 causal mask）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 5. softmax
        attn = F.softmax(scores, dim=-1)

        # 6. 加权求和
        out = torch.matmul(attn, V)

        # 7. 拼接 heads: (B, num_heads, L, d_k) -> (B, L, D)
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        # 8. 输出投影
        out = self.Wo(out)
        return out
```

**关键点回顾**：

- `d_k = d_model // num_heads`，保证多头拼接后维度还原为 `d_model`
- `.contiguous()` 是因为 `transpose` 之后内存不连续了，`view()` 之前必须调用
- `Wo` 是多头拼接之后额外的输出投影矩阵，用于融合各头的信息

---

## 三、MultiQueryAttention（MQA）

MQA 的核心区别：**Q 依然生成所有头的数据（维度还是 `embed_dim`），但 K 和 V 只生成 1 个头的数据（维度只有 `head_dim`）**，所有 Query 头共享同一份 K/V，通过广播机制完成计算。这样可以大幅减少推理时 KV Cache 的显存占用。

```python
import torch
import torch.nn as nn
import math

class MultiQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiQueryAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # 算一下每个头的维度
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.head_dim = embed_dim // num_heads

        # 🌟 MQA 的核心区别在这里！
        # Q 依然生成所有头的数据，所以输出维度是 embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim)
        # K 和 V 只生成 1 个头的数据！所以输出维度只有 head_dim
        self.W_k = nn.Linear(embed_dim, self.head_dim)
        self.W_v = nn.Linear(embed_dim, self.head_dim)
        # 最后的输出投影层
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x)  # (batch_size, seq_len, embed_dim)
        K = self.W_k(x)  # (batch_size, seq_len, head_dim)
        V = self.W_v(x)  # (batch_size, seq_len, head_dim)

        # Q 拆分成多个头: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # K 和 V 只有一个头，人为给它增加一个 "1" 的维度，方便后续利用广播机制 (Broadcasting)
        # K/V: (batch, seq_len, head_dim) -> (batch, 1, seq_len, head_dim)
        K = K.unsqueeze(1)
        V = V.unsqueeze(1)

        # 3. 计算注意力分数
        # Q: (batch, num_heads, seq_len, head_dim)
        # K 转置后: (batch, 1, head_dim, seq_len)
        # 🌟 魔法在这里：PyTorch 的 matmul 会自动把 K 的 "1" 广播 (broadcast) 成 "num_heads"
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores shape: (batch, num_heads, seq_len, seq_len)

        attn_weights = torch.softmax(scores, dim=-1)
        # attn_weights shape: (batch, num_heads, seq_len, seq_len)

        # V: (batch, 1, seq_len, head_dim) -> 同样会自动广播！
        output = torch.matmul(attn_weights, V)
        # output shape: (batch, num_heads, seq_len, head_dim)

        # transpose 换回来: (batch, seq_len, num_heads, head_dim)
        # contiguous() 是因为 transpose 后内存不连续了，view 之前必须调用
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # 最后的线性层
        return self.W_o(output)
```

**关键点回顾**：

- Q 仍然有 `num_heads` 份，K/V 只有 1 份（省显存的关键：推理时 KV Cache 只需存 1 份而不是 `num_heads` 份）
- `K.unsqueeze(1)` 给 K/V 增加一个大小为 1 的头维度，之后 `matmul` 时 PyTorch 会自动广播成 `num_heads`
- MQA 是 GQA（Grouped Query Attention，多个 Q 头共享一组 K/V）的极端情况（K/V 分组数 = 1）

---

## 四、DeepSeek MLA + RoPE（融合 RoPE 的 MLA 核心模块）

MLA（Multi-head Latent Attention）是 DeepSeek-V2/V3 提出的注意力变体，核心思路：**把 Q/K/V 先降维压缩，推理时只需要缓存压缩后的低维向量（`c_kv`），大幅减少 KV Cache 显存**；同时把 RoPE 位置编码拆到一个独立的小维度通道上，不参与压缩，保证旋转位置信息不失真。

```python
import torch
import torch.nn as nn
import math

# ==========================================
# 融合 RoPE 的 MLA 核心模块
# ==========================================
class DeepSeekMLAWithRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads, q_lora_rank=1536, kv_lora_rank=512, rope_dim=64, max_seq_len=2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads   # 内容维度（通常 128）
        self.rope_dim = rope_dim                 # RoPE 专属维度（通常 64）

        # ------------------------------------------------
        # 1. Query 通道（先降维压缩，再升维解压 + 生成 RoPE 向量）
        # ------------------------------------------------
        self.W_dq = nn.Linear(embed_dim, q_lora_rank, bias=False)                       # 降维
        self.W_uq = nn.Linear(q_lora_rank, num_heads * self.head_dim, bias=False)       # 解压出内容 Q
        self.W_qr = nn.Linear(q_lora_rank, num_heads * self.rope_dim, bias=False)       # 解压出 RoPE Q

        # ------------------------------------------------
        # 2. KV 通道（生成被极度压缩的 c_kv，这是推理时唯一需要 Cache 的东西！）
        # ------------------------------------------------
        self.W_dkv = nn.Linear(embed_dim, kv_lora_rank, bias=False)                     # 降维得到 c_kv
        self.W_uk = nn.Linear(kv_lora_rank, num_heads * self.head_dim, bias=False)      # 从 c_kv 解压出内容 K
        self.W_uv = nn.Linear(kv_lora_rank, num_heads * self.head_dim, bias=False)      # 从 c_kv 解压出内容 V

        # ------------------------------------------------
        # 3. K 的 RoPE 专属通道（全局共享一个 K_rope，不参与压缩）
        # ------------------------------------------------
        self.W_kr = nn.Linear(embed_dim, self.rope_dim, bias=False)

        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        # 🌟 注意：这里的 RoPE 维度是 rope_dim，而不是 head_dim！
        cos, sin = precompute_freqs_cis(self.rope_dim, max_seq_len)
        self.register_buffer("cos_cached", cos)
        self.register_buffer("sin_cached", sin)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # ==========================================
        # Step 1: Query 的生成与拆分
        # ==========================================
        c_q = self.W_dq(x)  # (batch, seq_len, q_lora_rank)

        # 内容 Q: (batch, heads, seq_len, head_dim)
        q_c = self.W_uq(c_q).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # RoPE Q: (batch, heads, seq_len, rope_dim)
        q_r = self.W_qr(c_q).view(batch_size, seq_len, self.num_heads, self.rope_dim).transpose(1, 2)

        # ==========================================
        # Step 2: KV 的生成与拆分（核心灵魂）
        # ==========================================
        c_kv = self.W_dkv(x)  # (batch, seq_len, kv_lora_rank) --> 推理时 Cache 里只存它！

        # 内容 K 和 V: (batch, heads, seq_len, head_dim)
        k_c = self.W_uk(c_kv).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_c = self.W_uv(c_kv).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 全局共享的 RoPE K：注意第 2 维是 1 (batch, 1, seq_len, rope_dim)
        k_r = self.W_kr(x).view(batch_size, seq_len, 1, self.rope_dim).transpose(1, 2)

        # ==========================================
        # Step 3: 对独立的 R 向量施加 RoPE 旋转
        # ==========================================
        cos = self.cos_cached[:seq_len, :]
        sin = self.sin_cached[:seq_len, :]

        # 这里 apply_rotary_emb 内部会自动把 k_r 的 1 广播 (Broadcast) 到 num_heads
        q_r, k_r = apply_rotary_emb(q_r, k_r, cos, sin)

        # ==========================================
        # Step 4: 把"内容"和"RoPE"拼接起来计算 Attention
        # ==========================================
        # 拼接后的 Q shape: (batch, heads, seq_len, head_dim + rope_dim)
        Q = torch.cat([q_c, q_r], dim=-1)

        # 把 k_r 从 1 物理复制成 num_heads，以便拼接
        k_r_expanded = k_r.expand(-1, self.num_heads, -1, -1)
        # 拼接后的 K shape: (batch, heads, seq_len, head_dim + rope_dim)
        K = torch.cat([k_c, k_r_expanded], dim=-1)

        # 算分：维度变了，除以 sqrt(head_dim + rope_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim + self.rope_dim)
        attn_weights = torch.softmax(scores, dim=-1)

        # 🌟 注意：乘的时候只乘内容 v_c！位置信息不需要加到 V 里
        output = torch.matmul(attn_weights, v_c)  # (batch, heads, seq_len, head_dim)

        # ==========================================
        # Step 5: 拼接回原维度并输出
        # ==========================================
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.W_o(output)


# ==========================================
# 测试代码：直接运行验证 Shape
# ==========================================
if __name__ == '__main__':
    # 模拟 DeepSeek-V2/V3 里的 MLA 配置
    embed_dim = 4096
    num_heads = 32
    seq_len = 128
    batch_size = 2

    # 默认 KV 被压缩到 512 维，Q 被压缩到 1536 维，外挂 RoPE 为 64 维
    model = DeepSeekMLAWithRoPE(embed_dim, num_heads)

    x = torch.randn(batch_size, seq_len, embed_dim)
    print("输入 x shape:", x.shape)
```

**关键点回顾**：

- **Query 通道**：`x → 降维(c_q, q_lora_rank) → 分别升维出「内容 Q」和「RoPE Q」`，两阶段投影本质上是低秩分解，减少参数量
- **KV 通道**：`x → 降维(c_kv, kv_lora_rank)`，**推理时 KV Cache 只需要缓存这一份低维的 `c_kv`**，而不是完整的多头 K/V，这是 MLA 节省显存的核心
- **RoPE 单独开一条通道**：`rope_dim` 维度不参与上面的压缩/解压，K 的 RoPE 部分（`k_r`）是**所有头共享的一份**（第 2 维是 1），只在计算 attention 前才广播到 `num_heads`
- **内容与位置信息拼接后再算 attention**：`Q = cat([内容Q, RoPE Q])`，`K = cat([内容K, RoPE K])`，拼接后维度是 `head_dim + rope_dim`，所以缩放要用 `sqrt(head_dim + rope_dim)`
- **V 不参与 RoPE**：位置信息只影响 Q/K 的相似度计算，最终加权求和用的是纯内容的 `v_c`

> `precompute_freqs_cis` 和 `apply_rotary_emb` 是 RoPE 的标准工具函数（预计算旋转角的 cos/sin，以及把旋转应用到 Q/K 上），完整实现可参考 [MiniMind从0到1构建大模型 · 模型结构](../MiniMind从0到1构建大模型.md#3模型结构) 中 RoPE 部分的代码。

---

## 五、Softmax（数值稳定实现）

手写 Softmax 时，必须减去每行最大值再做 `exp`，否则遇到较大数值容易发生数值溢出（`exp` 结果变成 `inf`）。

```python
# softmax
import numpy as np

def softmax_final(x):
    # 确保是 numpy 数组
    x = np.array(x)

    # 如果是二维矩阵 (batch_size, num_classes)
    if x.ndim == 2:
        # 沿每一行取最大值，keepdims=True 保证形状对齐，方便减法
        shift_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shift_x)
        # 沿行求和归一化
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # 如果是一维向量
    else:
        shift_x = x - np.max(x)
        exp_x = np.exp(shift_x)
        return exp_x / np.sum(exp_x)


# 测试代码
data = np.array([[1, 2, 3], [1000, 1001, 1002]])
print(softmax_final(data))
```

**关键点回顾**：

- `shift_x = x - max(x)`：减去最大值不改变 softmax 结果（数学上等价），但能把 `exp` 的输入范围压到 `(-∞, 0]`，避免 `exp` 上溢
- 二维输入（batch）要按 `axis=1`（每一行）分别取最大值和求和，`keepdims=True` 保证广播时形状正确
- 例子中 `[1000, 1001, 1002]` 如果不做数值稳定处理，`np.exp(1000)` 会直接溢出成 `inf`

---

## 六、三种 Attention 变体对比

| 变体 | Q 头数 | K/V 头数 | 推理 KV Cache 大小 | 典型代表 |
|------|--------|----------|---------------------|----------|
| MultiHeadAttention (MHA) | num_heads | num_heads | 最大（每头独立存） | 原版 Transformer、BERT |
| MultiQueryAttention (MQA) | num_heads | 1（所有头共享） | 最小（只存 1 份） | PaLM |
| GQA（分组，MQA 的推广） | num_heads | num_groups（1 < num_groups < num_heads） | 中等 | LLaMA2-70B、Qwen |
| MLA（多头潜在注意力） | num_heads | 压缩后的低秩向量 `c_kv` | 极小（远小于 MQA） | DeepSeek-V2/V3 |

> 一句话总结演进思路：MHA 效果最好但显存开销最大 → MQA 通过共享 K/V 大幅省显存但效果略降 → GQA 在两者之间做折中 → MLA 换了个思路，用低秩压缩同时保住效果和显存。
