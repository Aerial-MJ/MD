# Llama Attention Mask 与 SFT Label 处理详解

> 代码参考：[`llama.py`](llama.py)（HuggingFace `transformers` 库中 Llama 系列模型的建模代码，可以认为是 Llama3 的主要建模逻辑，此文件在原版基础上做了少量定制字段适配，如 `bbox`/`pixel_values`，但 Attention Mask 的核心处理逻辑与官方一致）。

---

## 一、Attention Mask 在代码里的完整处理链路

### 0. 贯穿全文的统一例子

后面所有步骤都用**同一个例子**演进，方便对照：`batch_size=2`，`seq_len=4`，样本1无 padding，样本2后2位是 padding：

```
样本1 tokens: [我, 很, 好, 呀]        → 4个真实token
样本2 tokens: [你好, 吗, PAD, PAD]     → 只有2个真实token，补2个PAD到长度4

input_ids      = [[101, 102, 103, 104],
                  [201, 202,   0,   0]]   # 0是PAD的token_id

attention_mask = [[  1,   1,   1,   1],
                  [  1,   1,   0,   0]]   # 形状 (batch=2, seq_len=4)
```

`attention_mask` 只有 0/1 两种值：`1` 表示真实 token（要参与 attention 计算），`0` 表示 padding（要被屏蔽）。

### 1. 核心转换函数：`_update_causal_mask`

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

下面按代码**逐行**跟踪 `causal_mask` 的实际数值，先看单个 `(seq_len=4, seq_len=4)` 的 2D 矩阵（batch/head 维度是最后才广播上去的）。

> **记号约定（务必先看这条）**：全文表格里的 **`-inf` 是简化记号，实际代表两种不同的真实数值**——为了让你先建立"哪些位置该不该被屏蔽"这个直觉，本节表格暂时不区分它们，等下面第1.5节会把这两种值的区别和数值上的坑彻底摊开讲：
> 1. **`min_dtype`**：一个具体的、有限的极大负数（比如 fp16 下的 `-65504`），不是数学上的 `-∞`，也不是 `-1`。行1 `torch.full` 直接填的就是它，行5-8 `masked_fill` 也是直接写入它。
> 2. **`1.0`**：`triu(causal_mask.bool(), diagonal=1)` 这一步，`.bool()` 把非零值都变 `True`，`triu` 筛选后 `.to(dtype)` 转回浮点，`True→1.0`、`False→0.0`。也就是说**因果规则（triu）屏蔽的位置，函数内部实际数值是 `1.0`，不是 `min_dtype`**！

---

**行1：`causal_mask = torch.full((4, 4), fill_value=-inf)`**

生成一个全部填满 `-inf` 的 4×4 矩阵，和输入内容完全无关：

```
causal_mask（行1执行后）:
       col0  col1  col2  col3
row0 [-inf, -inf, -inf, -inf]
row1 [-inf, -inf, -inf, -inf]
row2 [-inf, -inf, -inf, -inf]
row3 [-inf, -inf, -inf, -inf]
```

---

**行2：`causal_mask = torch.triu(causal_mask.bool(), diagonal=1)`**

`triu(diagonal=1)` 只保留**严格上三角**（不含对角线，即 `col > row` 的位置）为 `-inf`，其余全部变成 `0`：

```
causal_mask（行2执行后）:
       col0  col1  col2  col3
row0 [  0,   -inf,  -inf,  -inf ]
row1 [  0,    0,   -inf,  -inf ]
row2 [  0,    0,    0,   -inf ]
row3 [  0,    0,    0,    0   ]
```

逐格验证：`row2,col3` 因为 `3>2` 属于严格上三角，是 `-inf`；`row2,col2` 因为 `2=2`（对角线，不满足 `col>row`），是 `0`；`row1,col0` 因为 `0<1`（下三角），是 `0`。

**怎么读**：每一**行** = "当前 query 位置"，每一**列** = "它能不能看这个 key 位置"。`row2` 这一行是 `[0, 0, 0, -inf]`，意思是位置2能看位置0、1、2（自己），不能看位置3（未来）——这就是"下三角因果 mask"，执行完这一行代码，**纯因果部分已经完全定型**，后面两步都是在这个基础上叠加别的限制。

---

**行3：`causal_mask *= torch.arange(4) > cache_position.reshape(-1, 1)`**

这一步只在 **KV Cache 增量推理**场景下才会改变结果，分两种情况看数值：

**情况A（训练/prefill，一次性算完整个序列）**：`cache_position = [0,1,2,3]`（每个位置的索引就是它自己）：

```
torch.arange(4)              = [0, 1, 2, 3]
cache_position.reshape(-1,1) = [[0],[1],[2],[3]]

逐行广播比较 arange(4) > cache_position：
row0: [0,1,2,3] > 0 → [False, True,  True,  True]
row1: [0,1,2,3] > 1 → [False, False, True,  True]
row2: [0,1,2,3] > 2 → [False, False, False, True]
row3: [0,1,2,3] > 3 → [False, False, False, False]
```

这个 bool 矩阵里 `True` 的位置，**正好和行2里 `-inf` 的位置完全重合**（`col>row` ⟺ `True`）。乘以 `1`（True）或 `0`（False）后，`causal_mask` 数值上**完全不变**：

```
causal_mask（行3执行后，情况A，与行2一模一样）:
       col0  col1  col2  col3
row0 [  0,   -inf,  -inf,  -inf ]
row1 [  0,    0,   -inf,  -inf ]
row2 [  0,    0,    0,   -inf ]
row3 [  0,    0,    0,    0   ]
```

**情况B（KV Cache 增量推理，已生成4个token，现在只算第5个）**：此时 `sequence_length=1`（只有1个新位置要算），代码里 `if sequence_length != 1` 不成立，行2的 `triu` 被跳过，`causal_mask` 还停留在行1的状态，形状是 `(1, target_length)`，这里 `target_length=5`（历史4个+当前1个）：

```
causal_mask（行1状态，情况B，形状(1,5)）: [[-inf, -inf, -inf, -inf, -inf]]

cache_position = [4]（新token的位置索引是4）
torch.arange(5) = [0,1,2,3,4]
arange(5) > 4 → [False, False, False, False, False]   ← 全部False

causal_mask *= [F,F,F,F,F] → 全部乘以0 → 全部变成0
causal_mask（行3执行后，情况B）: [[0, 0, 0, 0, 0]]
```

全部变 `0`（全部允许看）是对的：新生成的这个 token（索引4）本来就应该能看到历史的0~3全部位置和它自己，不需要屏蔽任何东西。**这就是行3真正的作用**：在 KV Cache 场景下，用 `cache_position` 重新生成一次因果关系（替代被跳过的 `triu`），而在最常见的"一次性算完整个序列"场景下，它对结果没有任何影响（只是又确认了一遍已经算对的因果关系）。

下面继续用**情况A**（`(4,4)`，训练/prefill 场景）往下走。

---

### 1.5 插播：把"简化的 `-inf`"还原成真实数值，一次讲清楚

在继续往下看"行4/行5-8"之前，先花一分钟把前面为了直觉简化掉的东西还原一遍，这是后面很多疑问（"是不是`-1`"、"到底谁把它变成负数"）的根源，一次讲透就不会再反复纠结。

**行1、行2 真实的数值应该是这样的**（不用 `-inf` 简化，用真实数值）：

```
行1 torch.full(fill_value=min_dtype) 之后:
       col0        col1        col2        col3
row0 [min_dtype, min_dtype, min_dtype, min_dtype]
row1 [min_dtype, min_dtype, min_dtype, min_dtype]
row2 [min_dtype, min_dtype, min_dtype, min_dtype]
row3 [min_dtype, min_dtype, min_dtype, min_dtype]

行2 triu(causal_mask.bool(), diagonal=1).to(dtype) 之后:
       col0  col1  col2  col3
row0 [ 0.0,  1.0,  1.0,  1.0 ]
row1 [ 0.0,  0.0,  1.0,  1.0 ]
row2 [ 0.0,  0.0,  0.0,  1.0 ]
row3 [ 0.0,  0.0,  0.0,  0.0 ]
```

**行2 是关键**：`.bool()` 把 `min_dtype`（非零）全部变成 `True`，`triu` 把非上三角部分改成 `False`，再转回浮点，`True→1.0`、`False→0.0`。所以**因果规则要屏蔽的位置，此刻的真实数值是 `1.0`，跟一开始的 `min_dtype` 已经没关系了**——本文前面表格图省事，直接把这个 `1.0` 标注成了 `-inf`，这是为了让你先抓住"哪里被屏蔽"，但数值上不对，这里做个订正。

**行3**（`*= arange>cache_position`）在训练/prefill 场景下数值不变，还是上面这个 `0.0`/`1.0` 矩阵。

**行5-8**（`masked_fill(padding_mask, min_dtype)`）只对"因果允许（`0.0`）但恰好是padding"的位置，直接写入 `min_dtype`，其余位置不受影响。所以走到这一步，`causal_mask` 变成了**三态并存**：

```
样本2（走完行5-8）:
       col0  col1  col2(PAD)      col3(PAD)
row0 [ 0.0,  1.0,   1.0,           1.0        ]  ← col1~3因果屏蔽，值是1.0；col2,3虽是PAD但已被1.0覆盖，masked_fill不生效（因为.eq(0.0)判断不满足）
row1 [ 0.0,  0.0,   1.0,           1.0        ]  ← 同上
row2 [ 0.0,  0.0,  min_dtype,      1.0        ]  ← col2从0.0(因果允许)被padding改成min_dtype；col3本来就是1.0(因果屏蔽)
row3 [ 0.0,  0.0,  min_dtype,     min_dtype   ]  ← col2,col3都从0.0被padding改成min_dtype
```

**函数 `return` 时，`causal_mask` 里同时存在 `0.0`（真允许）、`1.0`（因果屏蔽）、`min_dtype`（padding屏蔽）三种值**，本文前面表格把 `1.0` 和 `min_dtype` 都简化标注成了同一个符号 `-inf`，这是简化演示导致的，实际底层数值并不相同。

**外层紧接着的 `causal_mask = causal_mask * torch.finfo(causal_mask.dtype).min` 就是用来统一这件事的**：

```
0.0        × min_dtype = 0.0          ✅ 保持允许
1.0        × min_dtype = min_dtype    ✅ 因果屏蔽被正确修正为极大负数
min_dtype  × min_dtype = 巨大正数      ⚠️ padding屏蔽的位置被意外乘成了正数（数值上不严谨，下面第2节详细说）
```

**这也是为什么本文前面的表格可以放心地把"因果屏蔽"和"padding屏蔽"都统一画成 `-inf`**：从最终"该不该被屏蔽"这个二元结论上看，两者最终都应该走向"极大负数"，只是"因果屏蔽"这条路是函数内部`1.0`+外层乘法两步共同完成的，"padding屏蔽"这条路函数内部一步就到位了（但又被外层多乘了一次，产生了数值上的瑕疵）。**你只需要记住表格里 `-inf` 代表"这个位置最终应该/原本想要被屏蔽"，不用纠结它具体是不是同一个数值路径变过来的。**

---

**行4：`causal_mask = causal_mask[None, None, :, :].expand(2, 1, -1, -1)`**

把 `(4,4)` 广播成 `(batch=2, 1, 4, 4)`，此时两个样本的内容还完全一样（padding 还没叠加）：

```
causal_mask[0] = causal_mask[1] =
       col0  col1  col2  col3
row0 [  0,   -inf,  -inf,  -inf ]
row1 [  0,    0,   -inf,  -inf ]
row2 [  0,    0,    0,   -inf ]
row3 [  0,    0,    0,    0   ]
```

---

**行5-8：叠加 padding mask**

```python
mask_length = attention_mask.shape[-1]                                            # = 4
padding_mask = causal_mask[..., :4].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
causal_mask[..., :4] = causal_mask[..., :4].masked_fill(padding_mask, min_dtype)
```

**只看样本2**（`attention_mask = [1,1,0,0]`），逐步算：

`A = causal_mask.eq(0.0)`（因果上允许看的位置，True=允许）：

```
A[row0] = [True,  False, False, False]
A[row1] = [True,  True,  False, False]
A[row2] = [True,  True,  True,  False]
A[row3] = [True,  True,  True,  True ]
```

`B = attention_mask.eq(0.0)`（哪些列是 PAD，True=是PAD），样本2的 `[1,1,0,0]` 取反：

```
B(样本2) = [False, False, True, True]   ← 广播到每一行都一样
```

`padding_mask = A * B`（逐元素相乘，两者都为 True 才为 True）：

```
row0: A=[T,F,F,F] & B=[F,F,T,T] → [F,F,F,F]
row1: A=[T,T,F,F] & B=[F,F,T,T] → [F,F,F,F]
row2: A=[T,T,T,F] & B=[F,F,T,T] → [F,F,T,F]   ← col2 变 True！
row3: A=[T,T,T,T] & B=[F,F,T,T] → [F,F,T,T]   ← col2,col3 变 True！
```

注意 `row0,col2`：`A=False`（这个位置因果上本来就已经是 `-inf` 了）、`B=True`（col2是PAD），相乘还是 `False`——因为它已经被因果规则屏蔽了，不需要 `padding_mask` 再标记一次。真正需要 `padding_mask` 新增标记的，只有"因果上原本允许看（A=True），但实际是PAD（B=True）"的位置，也就是 `row2,col2` 和 `row3,col2`、`row3,col3`。

最后 `masked_fill(padding_mask, -inf)`，把 `padding_mask` 为 `True` 的位置强制改成 `-inf`：

```
样本2 最终 causal_mask（行8执行后）:
       col0  col1  col2(PAD)  col3(PAD)
row0 [  0,   -inf,  -inf,      -inf   ]   ← 未变
row1 [  0,    0,   -inf,      -inf   ]   ← 未变
row2 [  0,    0,   -inf,      -inf   ]   ← 变了！原本是0（因果允许），现在因PAD被屏蔽
row3 [  0,    0,   -inf,      -inf   ]   ← 变了！col2,col3都被新屏蔽

样本1 最终 causal_mask（attention_mask=[1,1,1,1]，B全False，padding_mask全False，与行4完全一样）:
       col0  col1  col2  col3
row0 [  0,   -inf,  -inf,  -inf ]
row1 [  0,    0,   -inf,  -inf ]
row2 [  0,    0,    0,   -inf ]
row3 [  0,    0,    0,    0   ]
```

**这就是"最终 mask = 因果限制 AND padding限制"在具体数字上的体现**：样本2 的 `row2,col2`、`row3,col2`、`row3,col3` 本来按纯因果规则是允许看的（`0`），但因为 col2、col3 是 PAD，被额外改成了 `-inf`；样本1 因为没有 padding，全程和纯因果 mask 完全一样。

最终 `causal_mask` 形状是 `(batch=2, 1, 4, 4)`，可以直接广播到所有注意力头。

### 2. 针对不同 Attention 实现方式的分支处理

```python
if self.config._attn_implementation == "flash_attention_2" and attention_mask.dim() == 3:
    causal_mask = (attention_mask[:,:,0] == 0).int()
else:
    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)
    causal_mask = causal_mask * torch.finfo(causal_mask.dtype).min
```

这一行是**必需的关键步骤**，不是冗余（我之前的说法有误，这里更正）。回到第1.5节的结论：`_update_causal_mask()` 函数 `return` 时，`causal_mask` 里实际上混杂着**三种值**：

| 位置类型 | 函数返回时的真实数值 | 乘以 `min_dtype` 之后 |
|---|---|---|
| 真正允许看（因果允许 且 非padding） | `0.0` | `0.0 × min_dtype = 0.0`　✅ 保持允许 |
| 因果规则屏蔽（`triu` 标记的"未来"位置） | `1.0` | `1.0 × min_dtype = min_dtype`　✅ 正确变成极大负数 |
| padding 屏蔽（`masked_fill` 标记的位置） | `min_dtype` | `min_dtype × min_dtype = 一个巨大的正数`　⚠️ **数值错误** |

也就是说：**外层这一乘，对"因果屏蔽"的位置是必须的、正确的**（把 `1.0` 修正为 `min_dtype`）；但**对"padding屏蔽"的位置，则是把已经正确的 `min_dtype` 意外乘成了一个巨大正数**，这一小部分才是真正的数值问题（大概率是这份代码在 padding 分支里提前写入 `min_dtype`、又没有和 `triu` 分支的 `1.0` 语义对齐导致的历史遗留问题）。

**结论**：这行乘法对因果屏蔽是必要的关键步骤；对 padding 屏蔽则是不严谨的，但由于 padding 通常只占极少数位置，且后续在 `eager`/`sdpa` 实现里这份 mask 是直接加到 attention score 上做 softmax（一个异常大的正数在 softmax 后也会带来明显异常，实际不同版本/精度下具体数值表现可能不同），这里不再深究其在训练中的最终数值影响，只需记住"这一乘对padding位置在数值上不严谨"这个事实即可。

三种实现方式对这份 mask 的处理完全不同：

| 实现方式 | Mask 处理方式 |
|---------|---------------|
| `flash_attention_2` | 不用显式的加性 4D mask，而是把 padding 信息转成变长序列索引（见下面第4节 `_upad_input`），因果关系靠 `causal=True` 参数隐式处理，效率最高 |
| `eager`（手写实现） | 用完整的 4D 加性 mask，直接加到 attention score 上 |
| `sdpa`（PyTorch 官方融合算子） | 同样用 4D 加性 mask，传给 `F.scaled_dot_product_attention` 的 `attn_mask` 参数 |

### 3. Mask 在 Attention 内部具体怎么用（以 eager 实现为例，接着算样本2的数值）

```python
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

if attention_mask is not None:  # no matter the length, we just slice it
    causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = attn_weights + causal_mask
```

假设 `QK^T/√d` 算出来样本2 `row2`（位置2）这一行的原始分数是 `[2.1, -0.5, 3.8, 1.2]`（纯数值示例），加上上面算出来的样本2 `row2` mask `[0, 0, -65504, -65504]`：

```
attn_weights[row2] = [2.1, -0.5, 3.8, 1.2] + [0, 0, -65504, -65504]
                    = [2.1, -0.5, 3.8-65504, 1.2-65504]
                    = [2.1, -0.5, -65500.2, -65502.8]
```

再经过 `softmax`：

```
softmax([2.1, -0.5, -65500.2, -65502.8])
  ≈ [e^2.1, e^-0.5, e^-65500.2, e^-65502.8] / 归一化
  ≈ [0.93,  0.07,   ≈0,        ≈0        ]   ← col2、col3 概率约等于0
```

原本 `col2` 位置分数是 `3.8`（如果不加 mask，softmax 后反而会占最大权重），加上 `-65504` 之后直接变成约等于 0 的概率——**这就是"加性 mask 在 softmax 之前操作"的意义**：不是等概率算完了再把某些位置的概率抹成0（那样会破坏归一化，其余位置的概率也要重新算），而是在算概率**之前**就把对应 logit 拉到负无穷，让 softmax 自动把这部分概率挤压到 0，同时其余位置的相对大小、归一化都还是正常进行的。

### 4. Flash Attention 2 的特殊处理：不用加性 mask，而是"去 padding"

```python
def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    ...
    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
```

继续用本文的例子（`attention_mask = [[1,1,1,1],[1,1,0,0]]`）具体看这一步做了什么：

**第一步：找出所有真实 token 的位置索引 `indices_k`**

把 `attention_mask` 展平成 1D（`batch*seq_len = 8` 个位置），找出值为 `1` 的下标：

```
attention_mask 展平: [1,1,1,1, 1,1,0,0]
                      样本1(idx0~3)  样本2(idx4~7)

indices_k = [0, 1, 2, 3, 4, 5]   ← 只保留值为1的下标，样本2的idx6,7(PAD)被排除
```

**第二步：按 `indices_k` 把 `key_layer` 从 `(batch*seq_len, ...)` 里抽取出对应行**

```
key_layer 原始形状: (batch*seq_len=8, num_kv_heads, head_dim)
                    第0~3行属于样本1，第4~7行属于样本2

index_first_axis(key_layer, indices_k=[0,1,2,3,4,5])
→ 抽取第0,1,2,3,4,5行，丢弃第6,7行（样本2的PAD）
→ 得到形状 (6, num_kv_heads, head_dim) 的"去padding"张量
```

**第三步：用 `cu_seqlens` 记录每个样本的边界，供变长 attention 计算使用**

```
样本1长度4，样本2真实长度2（去掉PAD后）
cu_seqlens_k = [0, 4, 6]   ← 累积长度：样本1占[0:4)，样本2占[4:6)
```

后面调用 `flash_attn_varlen_func` 时传入这个不含 padding 的变长张量 + `cu_seqlens` + `causal=True`，Flash Attention 内部会根据 `cu_seqlens` 知道"第4~6行是另一个独立样本，不能跟第0~4行的样本1互相看"，因果关系在每个样本内部单独维护。算完之后再用 `pad_input` 把结果按 `indices_k` 逆向填回 `(batch, seq_len, ...)` 形状（样本2 原来的 PAD 位置补回全0或任意占位值，反正后面不会被用到）。

这样**样本2的PAD行（idx6,7）自始至终没有参与任何矩阵乘法**，比"eager/sdpa 里全部乘完 QK^T 再拿 mask 屏蔽"更省算力，序列越长、padding 比例越高，省下来的计算量越可观。

### 5. 整个流程一张图总结

```
原始 2D attention_mask (batch, seq_len)：1=真实token，0=padding
              ↓
      _update_causal_mask()
        ├─ 行1: torch.full 全部填 -inf
        ├─ 行2: triu(diagonal=1) 定型下三角因果结构（0/-inf）
        ├─ 行3: 结合 cache_position（prefill时无实际影响，increment时重建因果关系）
        └─ 行5-8: 广播出batch维度，AND上padding_mask，PAD列被强制置-inf
              ↓
最终 4D mask (batch, 1, seq_len, seq_len)，实际混杂0.0(允许)/1.0(因果屏蔽)/min_dtype(padding屏蔽)三种值
↓
外层再乘一次 torch.finfo(dtype).min：0.0→0.0（不变）；1.0→min_dtype（因果屏蔽正确修正）；
min_dtype→巨大正数（padding屏蔽被意外乘反，数值不严谨，但占比很小）
     ┌────────┼────────┐
   eager     sdpa    flash_attention_2
     │         │            │
  加到QK^T   传给F.scaled_   转成indices_k+cu_seqlens
  上做加性   dot_product_    (物理抽掉PAD行，不参与
  mask,      attention,      任何矩阵乘法，
  softmax后  同样是加性mask   causal=True隐式维护因果)
  PAD列≈0
```

**一句话总结：** 这份代码里的 attention mask 本质是"因果下三角限制 AND padding限制"的合并结果，对 eager/sdpa 走的是加性 mask（把屏蔽位置加一个极大负数，softmax 后趋近于0），对 flash_attention_2 走的是完全不同的路线——直接把 padding token 从序列里物理移除再计算，效率更高但实现更复杂。

---

## 二、FlashAttention 的底层原理：为什么它既省显存又更快

上面第 4 节提到 `flash_attention_2` 不用显式的加性 4D mask、效率最高，这一节深入讲讲它底层到底做了什么。

### 1. FlashAttention 要解决的根本问题：标准 Attention 太"吃显存带宽"

标准 Attention 的三步：

```
S = QK^T / √d        ← (seq_len, seq_len) 的完整分数矩阵，要整个写入显存(HBM)
P = softmax(S)        ← 再从显存读回来，算softmax，结果再写回显存
O = P @ V             ← 再从显存读回来，做最后一次矩阵乘法
```

问题不在于**算力（FLOPs）不够**，而在于：GPU 的**显存带宽（HBM 读写速度）远比计算慢**（比如 A100 的 SRAM 带宽是 HBM 的近 10 倍），而标准实现要把 $(seq\_len \times seq\_len)$ 这个巨大的中间矩阵 $S$、$P$ 反复在慢速的 HBM（显存）里读写好几遍，序列一长（比如 8K、32K），这个矩阵是平方级增长的，**大量时间都花在等数据从显存搬进搬出，而不是真正在算矩阵乘法**——这就是"显存带宽瓶颈"（memory-bound），而不是"算力瓶颈"（compute-bound）。

### 2. 核心思路：Tiling（分块）+ Kernel 融合，避免把中间结果写回显存

**核心思想只有一句话：把 Q、K、V 切成小块（tile），每次只把一小块搬进 GPU 的高速片上缓存（SRAM）里，在 SRAM 里一口气把这块的 attention 算完，全程不把中间的 $S$、$P$ 矩阵写回显存，只把最终结果写回去。**

具体展开：

**① 分块（Tiling）**

把 $Q$（长度 $N$）按行分成若干块 $Q_1, Q_2, \dots$，把 $K, V$ 也按行分成若干块 $K_1,K_2,\dots$、$V_1,V_2,\dots$，每块大小根据 SRAM 容量设定（比如每块 128 行）。

**② 外层循环 $K/V$ 块，内层循环 $Q$ 块（或反过来）**，对每一对 $(Q_i, K_j)$：

```
把 Q_i, K_j, V_j 从显存(HBM) 加载进片上高速缓存(SRAM)
在 SRAM 里就地计算：
    S_ij = Q_i @ K_j^T / √d          ← 只是一个小块，不是整个 N×N
    局部 softmax 相关统计量（见下面第③点）
    累加到输出 O_i 上
计算完这一块，只保留累加结果，S_ij 这个中间小块用完即弃，不写回显存
```

这样从头到尾，显存里只需要读一次 $Q,K,V$（原始数据）、写一次最终输出 $O$，中间那些平方级大小的 $S$、$P$ 矩阵**从未真正完整地存在于显存里**，全部都是在 SRAM 里算完就丢，这就是"IO-Aware"（对显存 IO 敏感）的算法设计。

**③ 关键技术难点：怎么在"分块计算"的情况下，正确算出全局 softmax？——Online Softmax（在线 Softmax）**

这是 FlashAttention 最精妙的部分。Softmax 公式是：

$$
\text{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j
$$

问题是，如果 $K$ 被分成了好几块，算第一块的时候你还不知道全局的最大值 $m$ 和全局的分母 $\sum_j e^{x_j-m}$（因为后面的块还没算到）。FlashAttention 用的解法是**增量式地维护和修正这两个统计量**：

每处理完一个新的 $K_j$ 块，都用一个"重新缩放"（rescale）的技巧，把之前累积的结果根据新出现的最大值做修正：

```
维护三个运行时状态：m_i（目前见过的最大logit）、l_i（目前的softmax分母累计和）、O_i（目前的输出累计值）

处理新的一块 K_j, V_j 时：
    m_new = max(m_i, 这一块内部的最大值)
    correction = exp(m_i - m_new)          ← 之前的结果要按新的最大值重新缩放
    l_i = l_i * correction + 这一块新的分母贡献
    O_i = O_i * correction + 这一块新的 P@V 贡献
    m_i = m_new
```

**用具体数字走一遍**：假设 $K$ 被分成2块，第一块内部最大logit是 `3.0`，此时 `m_i=3.0`，累积的分母 `l_i = e^{3.0-3.0} + e^{1.0-3.0} = 1 + 0.135 = 1.135`（假设第一块只有2个数 `[3.0, 1.0]`）。第二块进来后发现内部最大值是 `5.0`：

```
m_new = max(3.0, 5.0) = 5.0
correction = exp(3.0 - 5.0) = exp(-2.0) ≈ 0.135   ← 之前的分母要按新最大值缩小

l_i = 1.135 * 0.135 + (第二块新的分母贡献，比如 e^{5.0-5.0}+e^{2.0-5.0}=1+0.05=1.05)
    ≈ 0.153 + 1.05 = 1.203

O_i 同理按 correction 缩放后再累加第二块的贡献
m_i = 5.0   ← 更新为最新的全局最大值
```

处理完所有 $K$ 块之后，再用最终的 $l_i$ 做一次归一化，就能得到和"一次性算完整个 softmax"完全数学等价的结果——**这就是"safe softmax"数值稳定技巧（减最大值防溢出）在分块场景下的推广版本**，保证了分块计算不会丢失精度，结果和标准 attention 完全一致，只是算的顺序和存储方式变了。

### 3. 反向传播：不存 $S$、$P$，用"重计算"（Recomputation）换显存

标准 attention 反向传播需要用到前向时的 $S$、$P$ 矩阵去算梯度，通常做法是前向时把它们存下来（占用 $O(N^2)$ 显存）。FlashAttention 反而选择**前向时完全不存这些矩阵**，反向传播时**用存下来的 $O$、$m$、$l$（只有 $O(N)$ 大小）加上 $Q,K,V$ 重新算一遍局部的 $S_{ij}$、$P_{ij}$**（这次是在 SRAM 里现算现用）。

看起来"重新算一遍"好像更慢，但因为省下来的是显存 IO 时间（而不是算力），重新计算的这点额外 FLOPs 远比从显存反复读写 $N^2$ 大小的矩阵要快，这是一次典型的"用算力换带宽"的权衡，整体反而更快。

### 4. FlashAttention-2 相比 v1 的主要改进

- 减少了非矩阵乘法运算的比例（比如减少 rescale 操作的次数），让 GPU 的 Tensor Core 利用率更高；
- 改进了并行策略：把并行粒度细化到序列长度维度和 attention head 维度上，而不只是 batch 维度，长序列、小 batch 场景下能更充分利用 GPU 的多个 SM。

### 一句话总结

FlashAttention 的底层不是发明了新的数学公式（算出来的结果和标准 attention 完全一样），而是通过**分块（Tiling）+ 在线 Softmax（Online Softmax，增量维护最大值和归一化分母）+ kernel 融合（QK^T、softmax、乘V 全部在 SRAM 里一次完成，不落盘到 HBM）**，把一个"显存带宽瓶颈"的问题，转化成了"算力换带宽"的问题，从而大幅降低了显存占用（从 $O(N^2)$ 降到 $O(N)$）并提升了实际运行速度，这也是本文第一部分提到 `flash_attention_2` "不用显式的加性 4D mask，效率最高"背后真正的原因——它从算法设计的根子上就不允许你去物化一个完整的 $N\times N$ mask/score 矩阵。

---

## 三、SFT 训练时，Question 部分的 attention_mask 和 labels 该怎么设置

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

---

## 四、`cache_position` 到底是什么

前面第一节已经在用 `cache_position` 算 `causal_mask`，这里单独解释一下它到底是个什么东西——理解它是理解下一节 `sequence_length`/`target_length` 的前提。

### 一句话定义

**`cache_position` 是一个形状为 `(sequence_length,)` 的 1D 张量，记录"这一批新输入的每个 token，在它所属的完整序列里的绝对位置下标"。** 官方源码注释写得很清楚：

```python
"""
cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
    Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
    this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
    the complete sequence length.
"""
```

关键词是 **"not affected by padding"**（不受 padding 影响）——这是它和 `position_ids` 的核心区别。

### 它是怎么算出来的

```python
past_seen_tokens = 0
if use_cache:
    past_seen_tokens = past_key_values.get_seq_length()   # KV Cache里已经存了多少个token

if cache_position is None:
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )
```

就是从"历史已经算过的 token 数"（`past_seen_tokens`）开始，连续数到"历史 + 这次新输入的 token 数"，本质上是一个**绝对位置计数器**，和 batch 里有没有 padding 完全无关。

### 具体数值演示（接上文 prefill → decode 的例子）

**Prefill 阶段**：`past_seen_tokens=0`，新输入4个token：

```
cache_position = torch.arange(0, 0+4) = [0, 1, 2, 3]
```

**Decode 第1步**：KV Cache 里已经存了4个token，`past_seen_tokens=4`，新输入1个token：

```
cache_position = torch.arange(4, 4+1) = [4]
```

**Decode 第2步**：`past_seen_tokens=5`：

```
cache_position = torch.arange(5, 5+1) = [5]
```

也就是说，`cache_position` 就是在给这一批新 token 打上"它们在整条序列里排第几号"的绝对编号，从 prefill 到 decode 一路数下去，永远递增，不会因为某个样本前面有 padding 就产生偏移。

### 为什么强调"不受 padding 影响"——和 `position_ids` 的区别

`position_ids` 是**每个样本内部自己数的相对位置**，如果某条样本左边有 padding，`position_ids` 通常需要跳过 padding 从 0 开始数（不然 RoPE 位置编码会算错）；而 `cache_position` 只关心"这一次 forward 里，这批新 token 排在整个序列的哪个绝对下标"，和单条样本内部 padding 多少无关。举例说明这个区别：

```
样本2: [PAD, PAD, 你好, 吗]   ← 左padding，真实内容从下标2开始
                              ← attention_mask = [0, 0, 1, 1]

position_ids（每个样本内部相对位置，通常要跳过左padding）:
  样本2: [0, 0, 0, 1]  或类似的处理方式，让"你好"对应位置0（具体实现依模型而定）

cache_position（batch级别统一的绝对下标，与padding无关）:
  这一批一共4个token位置，不管样本内部是不是padding，都是 [0, 1, 2, 3]
  它描述的是"这次forward处理的是序列的第0~3个位置"，而不是"每个样本内部token的相对位置"
```

### `cache_position` 具体被用在哪两个地方

**用途1：算 `causal_mask`**（本文第一节的行3）——通过 `torch.arange(target_length) > cache_position.reshape(-1,1)`，告诉每个新 token"哪些位置对它来说算'未来'，需要屏蔽"。

**用途2：告诉 KV Cache，新算出来的 K、V 应该写入 Cache 的哪个绝对位置**：

```python
cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
```

比如 decode 第1步算出了新 token（绝对位置4）的 K、V，`cache_position=[4]` 告诉 Cache："把这份新算的 K、V 写到 Cache 张量的第4个位置上"，而不是简单地"追加到末尾"（这两者在没有 padding、顺序生成时结果一样，但在 Static Cache 预分配好固定长度空间、或者需要覆盖写入某个中间位置时，显式的绝对下标是必需的）。同时 RoPE 计算 `sin`/`cos` 时也需要知道每个 token 的绝对位置（RoPE 的角度直接和绝对位置相关），这也是为什么 `cache_kwargs` 里 `sin`/`cos` 和 `cache_position` 会一起传给 `update`。

### 一句话总结

**`cache_position` 就是这一批新输入 token 各自在完整序列里的"绝对楼层号"，从 KV Cache 已有的历史长度开始往后连续编号，不受任何样本内部 padding 影响；它同时被用来（1）判断 `causal_mask` 里哪些位置算"未来"需要屏蔽，（2）告诉 KV Cache 把新算的 K/V 精确写入哪个绝对下标。**

---

## 五、`sequence_length` 与 `target_length` 什么时候不相等

第一节的例子里两者都等于 4（`(4,4)` 的方阵），这是最常见但**并不是唯一**的情况。这两个变量的来源和计算方式是：

```python
sequence_length = input_tensor.shape[1]
if hasattr(self.layers[0].self_attn, "past_key_value"):  # static cache
    target_length = self.config.max_position_embeddings
else:  # dynamic cache
    target_length = (
        attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else cache_position[-1] + 1
    )

causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, ...)
```

**`sequence_length`** 永远等于 `input_tensor`（也就是 `inputs_embeds`）的第二维，即**这一次 forward 真正要新算的 token 数量**。

**`target_length`** 是"这个新 token 需要看到的 key/value 一共有多长"，取决于有没有用 KV Cache、用的是哪种 Cache：

| Cache 类型 | `target_length` 取值 | 含义 |
|---|---|---|
| 无 Cache（`hasattr(...,"past_key_value")` 为 False 且没有历史） | `attention_mask.shape[-1]` | 等于当前这批 `input_ids` 的长度 |
| Dynamic Cache（生成中，已有历史） | `attention_mask.shape[-1]` | 等于**历史长度 + 当前新增长度**（`attention_mask` 每步都会被拼接增长） |
| Static Cache（预分配好固定长度的 KV Cache，常用于 `torch.compile`/静态图场景） | `self.config.max_position_embeddings` | 固定为模型支持的最大长度，不随生成步数变化 |

### 场景对照：prefill 阶段 vs decode 阶段

用一个完整的生成过程说明，假设 prompt 是 4 个 token，模型要继续生成，用的是最常见的 **Dynamic Cache**：

**① Prefill 阶段（第一次 forward，一次性把 prompt 4 个 token 都喂进去，还没有任何历史）**

```
input_ids 形状: (batch=1, 4)          ← 4个token一次性输入
inputs_embeds 形状: (1, 4, hidden)     ← input_tensor

past_seen_tokens = 0   （还没有KV Cache历史）
cache_position = torch.arange(0, 0+4) = [0, 1, 2, 3]

sequence_length = inputs_embeds.shape[1] = 4
attention_mask 形状 = (1, 4)  ← 和当前输入长度一致，还没有历史
target_length = attention_mask.shape[-1] = 4

→ sequence_length(4) == target_length(4)，causal_mask 形状 (4,4)，方阵
→ 这就是本文第一节演示的情况
```

**② Decode 阶段第1步（用 prompt 的 KV Cache，只算新生成的第5个 token）**

```
经过 prefill，KV Cache 里已经存了4个token的K,V（past_seen_tokens=4）
这一步只输入新生成的1个token
input_ids 形状: (1, 1)
inputs_embeds 形状: (1, 1, hidden)     ← input_tensor，只有1个token了

cache_position = torch.arange(4, 4+1) = [4]
sequence_length = inputs_embeds.shape[1] = 1        ← 只算1个新位置

attention_mask 会被拼接成历史+当前的长度: 形状 (1, 5)  ← 4个历史 + 1个当前
target_length = attention_mask.shape[-1] = 5         ← 这个新token要能看到全部5个位置（含自己）

→ sequence_length(1) != target_length(5)！causal_mask 形状变成 (1, 5)，不再是方阵
```

**③ Decode 阶段第2步（再生成第6个 token）**

```
past_seen_tokens = 5（KV Cache里已有5个token）
input_ids 形状: (1, 1)
sequence_length = 1
cache_position = [5]

attention_mask 形状 (1, 6)
target_length = 6

→ sequence_length(1) != target_length(6)，causal_mask 形状 (1, 6)
```

**规律很清晰**：Dynamic Cache 场景下，**只有 prefill 第一步（一次性算完整个输入序列、还没有任何 KV Cache 历史）时 `sequence_length == target_length`**；从 decode 第一步开始，每一步都是"只算1个新 token（`sequence_length=1`），但要看到越来越长的历史（`target_length` 随生成步数逐步变大）"，两者必然不相等，且 `target_length` 会逐步增长（`4→5→6→7→...`），`causal_mask` 的形状从方阵变成"瘦长的一行"（`(1, target_length)`）。

### `sequence_length != target_length` 时，causal_mask 具体长什么样

回到本文第一节"行1~行3"的逻辑，代入 decode 第1步的例子（`sequence_length=1`，`target_length=5`，`cache_position=[4]`）：

**行1**：`torch.full((1, 5), fill_value=-inf)`：

```
causal_mask（行1执行后，形状(1,5)）: [[-inf, -inf, -inf, -inf, -inf]]
```

**行2**：`if sequence_length != 1` 不成立（这里 `sequence_length` 恰好等于1，是最常见的自回归单步解码），**这一步被跳过**，`causal_mask` 保持行1的全 `-inf` 状态。

> 注意：`sequence_length != 1` 这个判断是**巧合般地**用 `1` 这个字面量做条件，正好对应"自回归解码每步只生成1个新token"这个最常见场景——因为只有1行的时候，不存在"同一批新token互相之间要不要看见"的问题（只有1个query位置），所以不需要 `triu` 构造上三角，直接靠下一步的 `cache_position` 比较就够了。如果是**投机采样（speculative decoding）等一次验证多个新token**的场景，`sequence_length` 可能大于1（比如一次验证3个候选token），这时 `triu` 仍然会执行，需要在这新的几个token内部维持因果关系。

**行3**：`causal_mask *= torch.arange(5) > cache_position.reshape(-1,1)`，`cache_position=[[4]]`：

```
torch.arange(5) = [0,1,2,3,4]
arange(5) > 4 → [False, False, False, False, False]   ← 全部False

causal_mask *= [F,F,F,F,F] → 全部乘以0 → 全部变成0
causal_mask（行3执行后）: [[0, 0, 0, 0, 0]]
```

结果全部是 `0`（全部允许看）：这完全符合直觉——新生成的这第5个 token（索引4），理应能看到历史的0~3全部位置，以及它自己（索引4），没有任何"未来"位置需要屏蔽（这一步的 `target_length=5` 里根本不包含任何比它更新的位置）。

**如果 padding 也不为空**（比如 batch 里有的样本 prompt 短、有的长，历史 padding 需要延续屏蔽），第5-8行的 `padding_mask` 叠加逻辑和本文第一节完全一样，只是 `attention_mask` 此时形状是 `(batch, 5)`（历史+当前），把其中为 `0` 的历史 padding 列继续标记为 `-inf` 即可。

### Static Cache 场景：`target_length` 为什么固定不变

Static Cache 是为了配合 `torch.compile`/CUDA Graph 这类需要**固定张量形状**才能编译加速的场景设计的——如果每一步 `target_length` 都在变（4、5、6、7...），每次形状变化都会触发重新编译，非常慢。Static Cache 提前把 KV Cache 分配成 `(batch, num_heads, max_position_embeddings, head_dim)` 这样固定的最大形状，不管当前实际生成到第几步，`target_length` 都直接取 `self.config.max_position_embeddings`（比如 Llama3 的 `8192`/`4096`）：

```
第1步decode: sequence_length=1, target_length=4096（固定值，不是5）
第2步decode: sequence_length=1, target_length=4096（还是固定值，不是6）
...
```

这样每一步 `causal_mask` 的形状都固定是 `(1, 4096)`，代码结构和张量形状完全不随生成步数变化，`torch.compile` 只需要编译一次就能复用，避免了 dynamic shape 反复触发重新编译的开销。代价是提前分配了一大块"用不满"的显存（很多位置其实还没被真正使用到）。

### 一句话总结

**`sequence_length` 是"这一步新算的 token 数"，`target_length` 是"这一步的 key/value 覆盖到多长"。两者只在"一次性把完整序列算完"（训练、或推理的 prefill 首步）时相等；一旦进入 KV Cache 增量解码阶段，`sequence_length` 通常固定为 1（一次生成一个新 token），而 `target_length` 要么随生成步数持续增长（Dynamic Cache），要么固定为模型支持的最大长度（Static Cache），两者不相等是增量推理的常态，此时 `causal_mask` 也从方阵退化为形状 `(sequence_length, target_length)` 的"瘦长矩形"。**

---

## 六、Static Cache / Dynamic Cache 与 Prefill / Decode 的完整调用链路

前面几节已经在用这几个概念，这里专门把它们在 `llama.py` 里**具体是怎么创建、怎么判断类型、`generate()` 循环里数据怎么一步步流转**这件事讲透，串成一条完整的链路。

### 6.0 先明确一件事：`Static Cache` / `Dynamic Cache` 就是 **KV Cache** 的两种具体实现

**这两个类，本质上就是"KV Cache"这个概念在代码里的落地**，不是什么新东西——之前一直在说的 `past_key_value`、`past_key_values`，指的都是它。

**为什么需要 KV Cache**：Decode 阶段每生成一个新 token，理论上需要用它去 attend 前面所有的 token，而 attention 计算需要每个 token 的 Key、Value 向量。如果不缓存，每生成一个新 token 就要把"历史所有 token + 新token"重新完整地过一遍 `k_proj`/`v_proj` 计算一次 K、V，非常浪费——因为历史 token 的 K、V 其实每次算出来都是完全一样的值。**KV Cache 做的事情就是：把每个 token 算过一次的 K、V 存起来，下次直接复用，新来的 token 只需要计算它自己的 K、V，再和缓存里的历史 K、V 拼在一起做 attention。**

对应到代码里，就是本文第一节反复出现的这一行：

```python
key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
```

`past_key_value` 就是 KV Cache 对象（一个 `DynamicCache` 或 `StaticCache` 的实例），`.update()` 就是"把这一层新算出来的 K、V 存进缓存，同时取出（缓存里的历史K/V + 新K/V）拼好之后的完整结果，供后面做 attention 用"。**每一层 decoder 都有自己独立的一份 KV Cache**（因为每一层的 K、V 参数矩阵不同，算出来的 K、V 也不同），这也是为什么 `_setup_cache` 里要用 `for layer in self.model.layers` 循环，给每一层都分别挂一个 Cache 实例。

而 `Static`/`Dynamic` 只是这个"存起来"的动作具体怎么实现的两种策略（拼接 vs 预分配），不影响"KV Cache 是干什么用的"这个本质。

### 6.1 两种 Cache 到底是什么

`llama.py` 开头有这行 import：

```python
from transformers.cache_utils import Cache, DynamicCache, StaticCache
```

`DynamicCache`、`StaticCache` 都是 `transformers` 库提供的通用缓存类（这份 `llama.py` 只是**使用**它们，具体内部实现在 `transformers/cache_utils.py` 里，不在本文件中）。两者的核心区别：

| | `DynamicCache` | `StaticCache` |
|---|---|---|
| 底层存储 | 用 Python list，每次 `update()` 时用 `torch.cat` 把新 K/V **拼接**到已有 tensor 后面，长度随生成步数增长 | 初始化时就用 `torch.zeros` **预分配**好 `(batch, num_heads, max_cache_len, head_dim)` 固定形状的大张量，`update()` 只是用 `index_copy_`/切片赋值把新 K/V **写入**指定位置，张量形状全程不变 |
| 长度信息 | `get_seq_length()` 返回目前已经存了多少个真实token | 长度含义不同，因为张量本身形状是固定的，需要结合 `cache_position` 才知道哪些位置是"已写入的有效数据" |
| 适用场景 | 常规 `generate()`，最通用、默认的选择 | 需要配合 `torch.compile`/CUDA Graph 做静态图编译加速的场景，因为张量形状全程不变，不会触发重新编译 |
| 创建方式 | `DynamicCache.from_legacy_cache(past_key_values)`，可以从 `None` 直接开始，随用随长 | 必须显式调用 `model._setup_cache(StaticCache, max_batch_size, max_cache_len)` 提前分配好空间 |

### 6.2 代码里怎么判断当前用的是哪种 Cache

这是解开你之前疑问的关键代码，在 `_update_causal_mask` 里：

```python
if hasattr(self.layers[0].self_attn, "past_key_value"):  # static cache
    target_length = self.config.max_position_embeddings
else:  # dynamic cache
    target_length = attention_mask.shape[-1] ...
```

**这里 `hasattr(..., "past_key_value")` 是判断依据**：只有调用过 `_setup_cache(StaticCache, ...)` 之后，才会给每一层的 `self_attn` 挂上 `past_key_value` 这个属性（见下面 `_setup_cache` 源码），所以这个 `hasattr` 判断本质是在问"模型是不是提前配置好了 Static Cache"：

```python
def _setup_cache(self, cache_cls, max_batch_size, max_cache_len: Optional[int] = None):
    for layer in self.model.layers:
        ...
        layer.self_attn.past_key_value = cache_cls(
            self.config, max_batch_size, max_cache_len, device=device, dtype=dtype
        )

def _reset_cache(self):
    for layer in self.model.layers:
        layer.self_attn.past_key_value = None
```

也就是说：**默认情况下（不调用 `_setup_cache`）走的是 `DynamicCache`**；只有显式调用过 `model._setup_cache(StaticCache, ...)`（通常是在用 `torch.compile` 编译生成过程时才会这么做）之后，才会切换到 `Static Cache` 分支。

另外在 `LlamaModel.forward` 里也有一处类型判断：

```python
past_seen_tokens = 0
if use_cache:
    if not isinstance(past_key_values, StaticCache):
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_seen_tokens = past_key_values.get_seq_length()

if cache_position is None:
    if isinstance(past_key_values, StaticCache):
        raise ValueError("cache_position is a required argument when using StaticCache.")
    cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], ...)
```

**这里也能看出一个重要区别**：`DynamicCache` 场景下，如果调用者没传 `cache_position`，模型可以自己用 `get_seq_length()` 推算出来（比较"智能"）；但 **`StaticCache` 场景下，`cache_position` 是必须由外部显式传入的**（`raise ValueError`），因为 `StaticCache` 内部张量形状不会变，模型自己无法从"张量当前长度"推断出"现在生成到第几个token了"，必须靠外部明确告知。

### 6.3 完整调用链路：`generate()` 循环里发生了什么

`transformers` 的 `model.generate()` 本质上是一个 **while 循环，每一轮循环调用一次 `forward()`**，产出一个新 token，直到遇到结束符或达到最大长度。每一轮循环开始前，都会先调用 `prepare_inputs_for_generation()` 准备好这一轮的输入。用一个完整例子走一遍（prompt 4个token，Dynamic Cache）：

**循环第0轮（Prefill）**：

```
输入: input_ids = [101,102,103,104]（完整prompt）, past_key_values=None, cache_position=None

prepare_inputs_for_generation():
  past_key_values 是 None → past_length = 0
  cache_position = torch.arange(0, 0+4) = [0,1,2,3]     ← 见第1451行
  model_inputs = {"input_ids": [101,102,103,104], "cache_position": [0,1,2,3], ...}

LlamaModel.forward():
  past_seen_tokens = 0（past_key_values是None）
  cache_position = [0,1,2,3]（外部已传入，直接用）
  sequence_length = 4, target_length = 4   ← 本文第五节讲的"prefill方阵"情况
  causal_mask 形状 (4,4)
  ... 4层decoder跑完，每层的 self_attn 内部 past_key_value.update() 把这4个token的K,V存入Cache
  返回 next_cache（此时DynamicCache里已经存了4个token的K,V）
  输出 logits，取最后一个位置采样出第5个token，比如 105
```

**循环第1轮（Decode 第1步）**：

```
输入: input_ids = [101,102,103,104,105]（generate内部会维护完整序列）, past_key_values=DynamicCache(已存4个), cache_position=[0,1,2,3]（上一轮的）

prepare_inputs_for_generation():
  past_key_values 不是 None → past_length = past_key_values.get_seq_length() = 4   ← 见第1398行
  input_ids.shape[1]=5 > past_length=4 → 走 elif 分支，只保留还没处理过的部分:
      input_ids = input_ids[:, 4:] = [105]              ← 只留最新生成的这1个token！← 见第1418-1419行
  cache_position = torch.arange(4, 4+1) = [4]           ← 见第1451行
  model_inputs = {"input_ids": [105], "cache_position": [4], "past_key_values": DynamicCache, ...}

LlamaModel.forward():
  past_seen_tokens = past_key_values.get_seq_length() = 4
  cache_position = [4]（外部已传入）
  sequence_length = 1（只有105这1个新token）, target_length = 5（attention_mask历史4+当前1）
  causal_mask 形状 (1,5)    ← 本文第五节讲的"decode瘦长矩形"情况
  ... self_attn 内部 past_key_value.update() 把新算的K,V"拼接"到已有的4个后面，Cache变成存5个
  输出 logits，采样出第6个token，比如 106
```

**循环第2轮（Decode 第2步）**：和第1轮完全同构，`input_ids` 只留最新1个 `[106]`，`cache_position=[5]`，`target_length=6`……如此循环，直到生成结束。

**这就是"Prefill"和"Decode"的本质区别**：

- **Prefill**：一次性把完整 prompt 喂进去，`sequence_length` 等于 prompt 长度，是唯一一次"整段计算"，同时把 prompt 所有 token 的 K/V 一次性写入 Cache。
- **Decode**：每一轮只喂**新生成的那1个 token**（`prepare_inputs_for_generation` 靠 `past_length` 精确截取出"还没被Cache处理过的部分"），`sequence_length` 恒为1，但 `target_length`（要看的历史长度）每轮 `+1`，直到生成结束。这正是 KV Cache 存在的意义——**避免每生成一个新token，就要把前面所有token重新算一遍 K/V**，只需要用 Cache 里存好的历史K/V，加上新token自己的K/V即可完成一次 attention 计算。

### 6.4 Static Cache 场景下这条链路的差异

如果提前调用了 `model._setup_cache(StaticCache, max_batch_size, max_cache_len)`：

- `prepare_inputs_for_generation` 里 `has_static_cache=True`，会强制 `past_key_values = None` 传给 `forward`（因为 Static Cache 是挂在每层 `self_attn.past_key_value` 属性上的，不需要再通过参数传递）。
- `cache_position` 的计算逻辑一样是"新token的绝对位置"，但因为张量形状全程固定为 `max_cache_len`，每轮 `forward` 输入输出的张量形状都完全一致，`torch.compile` 只需要编译一次计算图就能复用到后续所有 decode 步骤，避免了 Dynamic Cache 场景下"每步 `target_length` 变化都触发重新编译"的开销——这也是本文第五节里已经讲过的"为什么 `target_length` 要固定"的根本原因。

### 6.5 一句话总结

**Prefill 是"整段处理 prompt、同时把所有历史 token 的 K/V 一次性灌进 Cache"的唯一一步；Decode 是"每步只处理新生成的 1 个 token、复用 Cache 里的历史 K/V"的循环过程，两者靠 `prepare_inputs_for_generation` 里对 `input_ids` 的截取和 `cache_position` 的递增来区分。Dynamic Cache 用拼接的方式让存储长度随 Decode 步数自然增长，逻辑简单但形状每步都变；Static Cache 用预分配固定空间+定点写入的方式让张量形状全程不变，牺牲一些显存换取 `torch.compile` 静态图编译带来的推理加速，两者的选择直接决定了本文第五节 `causal_mask` 里 `target_length` 是"逐步增长"还是"恒定不变"。**
