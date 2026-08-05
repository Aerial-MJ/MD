# Llama Attention Mask 与 SFT Label 处理详解

> 代码参考：[`llama.py`](llama.py)（HuggingFace `transformers` 库中 Llama 系列模型的建模代码，可以认为是 Llama3 的主要建模逻辑，此文件在原版基础上做了少量定制字段适配，如 `bbox`/`pixel_values`，但 Attention Mask 的核心处理逻辑与官方一致）。

---

## 一、Attention Mask 在代码里的完整处理链路

### 1. 输入的 `attention_mask` 是什么

一开始传进模型的 `attention_mask` 是最原始的 **2D padding mask**，形状 `(batch_size, seq_len)`，值只有 0 和 1：

```
1 表示这个位置是真实 token，需要参与注意力计算
0 表示这个位置是 padding（补齐用的占位符），要被屏蔽
```

### 2. 核心转换函数：`_update_causal_mask`

```python
def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
    ...
    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask.bool(), diagonal=1).to(dtype=dtype)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    if attention_mask is not None:
        causal_mask = causal_mask.clone()
        if attention_mask.dim() == 2:
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
```

这个函数做的事情分三步：

**第一步：构造纯粹的因果 mask（下三角）**

```python
causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, ...)  # 全部填最小值
causal_mask = torch.triu(causal_mask.bool(), diagonal=1).to(dtype=dtype)  # 上三角部分保留最小值，下三角(含对角线)变0
```

```
用 min_dtype（比如 fp16 下的 -65504）表示"屏蔽"，0 表示"允许看"
triu(diagonal=1) 取严格上三角（不含对角线）

举例 seq_len=4：
       位置0  位置1  位置2  位置3
位置0 [  0,   -inf,  -inf,  -inf ]   ← 只能看自己
位置1 [  0,    0,   -inf,  -inf ]   ← 能看0,1
位置2 [  0,    0,    0,   -inf ]   ← 能看0,1,2
位置3 [  0,    0,    0,    0   ]   ← 能看全部

这就是标准的"下三角因果 mask"：每个位置只能看自己和之前的 token
```

**第二步：结合 KV Cache 的 `cache_position` 做偏移**（支持增量推理时的因果关系仍然正确）：

```python
causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
```

**第三步：叠加原始的 2D padding mask，变成 4D mask**

```python
if attention_mask.dim() == 2:
    mask_length = attention_mask.shape[-1]
    padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
    causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
```

```
把 2D 的 (batch, seq_len) padding mask 广播成 4D：(batch, 1, 1, seq_len)
和 causal_mask 做逐元素"与"操作：
  只要 causal_mask 允许看（=0） 且 attention_mask 也允许看（=1）
  才最终允许看，否则填充 min_dtype（屏蔽）

也就是：最终 mask = 因果限制 AND padding限制
```

最终输出的 `causal_mask` 形状是 **4D**：`(batch_size, 1, seq_len, target_length)`，可以直接广播到所有注意力头。

### 3. 针对不同 Attention 实现方式的分支处理

```python
if self.config._attn_implementation == "flash_attention_2" and attention_mask.dim() == 3:
    causal_mask = (attention_mask[:,:,0] == 0).int()
else:
    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)
    causal_mask = causal_mask * torch.finfo(causal_mask.dtype).min
```

三种实现方式对 mask 的处理完全不同：

| 实现方式 | Mask 处理方式 |
|---------|---------------|
| `flash_attention_2` | 不用显式的加性 4D mask，而是把 padding 信息转成变长序列索引（见下面的 `_upad_input`），因果关系靠 `causal=True` 参数隐式处理，效率最高 |
| `eager`（手写实现） | 用完整的 4D 加性 mask，直接加到 attention score 上 |
| `sdpa`（PyTorch 官方融合算子） | 同样用 4D 加性 mask，传给 `F.scaled_dot_product_attention` 的 `attn_mask` 参数 |

### 4. Mask 在 Attention 内部具体怎么用（以 eager 实现为例）

```python
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

if attention_mask is not None:  # no matter the length, we just slice it
    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = attn_weights + causal_mask
```

这里用的是**加性 mask（additive mask）**：直接把 mask 值加到 `attn_weights`（即 $QK^T/\sqrt{d}$）上。因为被屏蔽的位置是 `min_dtype`（一个极大的负数，比如 -65504），加上之后这个位置的分数变成极小值，经过 softmax 后概率约等于 0，等效于"看不到"这个位置。

```
attn_weights = QK^T/√d + mask

允许看的位置：+0，分数不变
屏蔽的位置：+(-65504)，softmax后≈0

这就是为什么用"加法"而不是直接置0——因为要在softmax之前操作，
让被屏蔽位置的指数值趋近于0，而不是让已经算好的概率变成0
```

### 5. Flash Attention 2 的特殊处理：不用加性 mask，而是"去 padding"

```python
def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    ...
    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
```

Flash Attention 不支持传统的加性 mask（会破坏其内存高效的融合 kernel 设计），所以做法完全不同：

```
把 batch 里所有序列的 padding token 直接"抽掉"，拼接成一个不含 padding 的变长序列
用 cu_seqlens（cumulative sequence lengths）记录每个样本的真实边界
调用 flash_attn_varlen_func，配合 causal=True 参数
计算完再用 pad_input 把结果重新填回原来的 (batch, seq_len) 形状
```

这样彻底避免了在 padding 位置上做无意义的计算，比"算了再屏蔽"的方式更高效。

### 6. 整个流程一张图总结

```
原始 2D attention_mask (batch, seq_len)：1=真实token，0=padding
              ↓
      _update_causal_mask()
              ↓
生成 4D 因果 mask（下三角，min_dtype表示屏蔽）
              ↓
      叠加 padding mask（AND操作）
              ↓
   最终 4D mask (batch, 1, seq_len, seq_len)
              ↓
     ┌────────┼────────┐
   eager     sdpa    flash_attention_2
     │         │            │
  加到QK^T   传给F.scaled_   转成变长序列索引
  上做加性   dot_product_    (不用显式mask矩阵，
  mask        attention      靠causal=True+去padding)
```

**一句话总结：** 这份代码里的 attention mask 本质是"因果下三角限制 AND padding限制"的合并结果，对 eager/sdpa 走的是加性 mask（把屏蔽位置加一个极大负数，softmax 后趋近于0），对 flash_attention_2 走的是完全不同的路线——直接把 padding token 从序列里物理移除再计算，效率更高但实现更复杂。

---

## 二、SFT 训练时，Question 部分的 attention_mask 和 labels 该怎么设置

### 结论先行

对于一条 SFT 样本 `[question_tokens, answer_tokens, padding_tokens]`：

| 部分 | `attention_mask` | `labels` |
|------|------------------|----------|
| Question（问题/指令部分） | **1**（参与计算） | **-100**（不算loss） |
| Answer（答案部分） | **1**（参与计算） | **真实 token id**（要算loss） |
| Padding（补齐部分） | **0**（不参与计算） | **-100**（不算loss） |

### 为什么 Question 部分 attention_mask 还是 1

`attention_mask` 控制的是"这个位置的 token 是不是真实存在的、要不要参与 attention 计算"，它和"要不要在这个位置算 loss"是两码事。

```
Question 部分虽然不需要预测（不算loss），但它是真实存在的上下文
模型必须能"看到"它，才能基于它去生成 answer

如果把 question 的 attention_mask 设成 0：
  → 相当于告诉模型"这部分不存在"
  → 模型在预测 answer 时也看不到 question 内容了
  → 完全不对，模型会失去上下文信息

所以 question 部分 attention_mask 必须是 1，只是不参与 loss 计算
```

### 为什么 Padding 部分 attention_mask 是 0

Padding 纯粹是为了把一个 batch 里长短不一的序列补齐到相同长度，方便做矩阵运算，它不是真实内容：

```
padding token 本身没有任何语义
attention_mask=0 告诉模型：这个位置是凑数的，别看它
（具体实现就是前面 causal_mask 里 padding_mask 那部分，加极大负数让 softmax 后≈0）
```

### `labels = -100` 的作用：告诉损失函数"这个位置不参与梯度计算"

`-100` 是 PyTorch `CrossEntropyLoss` 默认的 `ignore_index`：

```python
loss_fct = CrossEntropyLoss(ignore_index=-100)  # 默认就是-100
loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
```

```
CrossEntropyLoss 在计算的时候，会跳过 label=-100 的位置：
  这个位置不计入 loss 的平均
  这个位置也不会产生梯度

所以：
  question 位置：label=-100 → 不计入loss，模型不会因为"预测不出问题本身"被惩罚
  answer 位置：label=真实token id → 计入loss，模型要学着预测出正确答案
  padding 位置：label=-100 → 同样不计入loss，凑数的位置不该产生任何学习信号
```

### 一个完整例子直观感受一下

假设一条样本：`[你好, 吗, 我, 很, 好, PAD, PAD]`，其中 "你好吗" 是 question，"我很好" 是 answer：

```
token:           你好   吗    我      很      好     PAD    PAD
input_ids:       101   102   103    104     105     0      0
attention_mask:   1     1     1      1       1      0      0    ← padding才是0
labels:         -100  -100   103    104     105   -100   -100    ← question和padding都是-100
```

关键点对比一下：
- `attention_mask` 的 0 只出现在 **padding**
- `labels` 的 -100 出现在 **question + padding**（两种不同原因导致的"不算loss"）

### 为什么 labels 通常还要"错位一格"（顺带说明，容易和上面的问题搞混）

实际实现里通常还会做 shift（因为预测的是"下一个token"）：

```python
shift_logits = logits[..., :-1, :]   # 预测位置：0~n-2
shift_labels = labels[..., 1:]       # 真实答案：1~n-1（往后移一位）
```

```
位置 i 的 logits 是用来预测"位置 i+1 的 token"的
所以 labels 要整体左移一位，让 logits[i] 对应 labels 原来[i+1]位置的值
```

这个 shift 操作和 attention_mask、-100 是两个独立的机制，只是实现时经常一起出现，容易混在一起理解。

### 总结一句话

**`attention_mask` 管的是"模型能不能看见这个位置"（真实内容永远是1，只有padding是0）；`labels` 管的是"这个位置要不要算梯度学习"（question和padding都是-100，只有answer部分是真实token id）。两者互相独立，一个控制"看得见看不见"，一个控制"学不学"。**
