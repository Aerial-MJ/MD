# 从 0 到 1 构建大模型（以 MiniMind 为例）

> 本文以 [MiniMind](https://github.com/jingyaogong/minimind) 为实践项目，完整走一遍大语言模型的构建流程：训练 Tokenizer → 设计模型结构 → 数据处理 → 预训练 → 有监督微调 → 强化学习（PPO/GRPO/DPO）→ 训练工程 → 推理部署，是一份覆盖原理与工程实践的完整笔记。
>
> 关联笔记（以下是总览，各小节标题下还有更精确的对应章节链接，建议从正文直接跳转）：
> - 模型结构中的参数命名与形状细节 → [Transformer向量与参数详解](../深度学习/Transformer向量与参数详解.md)
> - 注意力机制的直觉与公式基础 → [8、注意力机制和Transformer](../邱熙鹏深度学习/8、注意力机制和Transformer.md)
> - 激活函数（SiLU/GELU 等）的选择依据 → [激活函数和损失函数](../深度学习/激活函数和损失函数.md)
> - MoE 架构的详细原理 → [Moe](../深度学习/Moe.md)
> - PPO / GRPO 公式的另一套推导与对照表 → [GRPO与PPO算法详解](../深度学习/GRPO与PPO算法详解.md)
> - RLHF / 对齐（SFT → RM → DPO/RLHF）的整体流程 → [对齐](../深度学习/对齐.md)
> - 工业级 RL 训练框架 verl 与本文 4.3 节 Agentic RL、4.5 节训练工程的关系 → [verl](verl.md)

## 目录

1. [总览](#1-总览)
2. [分词器（Tokenizer）](#2-分词器tokenizer)
3. [模型结构](#3-模型结构)
4. [训练方式](#4-训练方式)
   - 4.1 [预训练](#41-预训练)　4.2 [有监督微调（SFT）](#42-有监督微调sft)　4.3 [强化学习](#43-强化学习)（含 4.3.3 [DPO](#433-dpo跳过显式奖励模型和在线-rollout)）
   - 4.4 [数据处理](#44-数据处理从原始语料到训练样本)　4.5 [训练工程细节](#45-训练工程细节)　4.6 [推理](#46-推理)

---

## 1、总览

本次实践以 MiniMind 为例，从零走完一个大语言模型的构建流程，并回顾大模型的模型结构、训练方式等知识。

**我们有什么？**

- 预训练数据集
- 有监督微调数据集
- 以训练 Agent 为核心的强化学习数据集

**我们要做什么？**

| 步骤 | 目标 |
|------|------|
| 训练 Tokenizer | 构建模型使用的词表，将文本转换为 Token ID |
| 设计模型结构 | 实现 Embedding、Attention、FFN、RMSNorm 和位置编码等核心模块 |
| 预训练 | 让模型从大量文本中学习语言规律和基础知识 |
| 有监督微调 | 让模型理解用户指令，并以对话形式回答问题 |
| 强化学习 | 优化模型的回答质量，增强模型的 Agent 能力 |

**我们会得到什么？**

一个从 Tokenizer、模型结构到训练流程都可以完整运行的大语言模型，并理解它如何从随机初始化逐步获得文本生成、指令遵循和工具调用能力。

---

## 2、分词器（Tokenizer）

### 2.1 Byte-level BPE

#### 2.1.1 基本算法

BPE 全称 Byte Pair Encoding，最早是一种数据压缩算法，后来被迁移到 NLP 领域。

**算法流程**：不断寻找语料中最常见的相邻 token 对，并把它们合并成新的 token，直到达到目标词表大小。

**一个简单的例子**

假设训练语料中有这些词：`play`、`playing`、`played`、`player`、`replay`。

先把每个词拆成字符（实际会拆成字节）：

```text
play    -> p l a y
playing -> p l a y i n g
played  -> p l a y e d
player  -> p l a y e r
replay  -> r e p l a y
```

**第一轮：统计相邻 token 对**

观察语料，`p l` 经常一起出现 → 学到第一条合并规则：`p + l -> pl`

```text
play    -> pl a y
playing -> pl a y i n g
played  -> pl a y e d
player  -> pl a y e r
replay  -> r e pl a y
```

词表新增 token：`pl`

**第二轮：继续统计并合并**

`pl a` 经常一起出现 → 学到第二条合并规则：`pl + a -> pla`

```text
play    -> pla y
playing -> pla y i n g
played  -> pla y e d
player  -> pla y e r
replay  -> r e pla y
```

词表新增 token：`pla`

**重复该过程**，直到达到规定的词表大小，或者没有值得继续合并的高频 token 对为止：

1. 统计当前语料中所有相邻 token 对的频率
2. 找到出现次数最高的 token 对
3. 把这个 token 对合并成一个新的 token
4. 把新的 token 加入词表
5. 更新整个语料

**最简 Python 实现**

```python
from collections import Counter

class ByteLevelBPE:
    # BPE 算法有两个基本的数据结构：merges 和 vocab
    # merges 负责记录合并的规则，vocab 负责记录 token_id 和 token 的映射
    def __init__(self):
        self.merges = []          # [(a, b), ...]
        self.vocab = {i: bytes([i]) for i in range(256)}   # 1. 初始化 vocab = 0...255

    def train(self, texts, vocab_size=300):
        # 2. 把所有文本转成 UTF-8 bytes
        corpus = [list(t.encode("utf-8")) for t in texts]

        next_id = 256
        # 重复迭代，直到达到目标 vocab size
        while next_id < vocab_size:
            stats = Counter()
            # 3. 统计全语料中所有相邻 token pair 的频率
            for tokens in corpus:
                stats.update(zip(tokens, tokens[1:]))
            if not stats:
                break
            # 4. 找到最高频的 pair
            pair, freq = stats.most_common(1)[0]
            if freq < 2:
                break
            # 5. 将该 pair 合并成一个新 token
            self.vocab[next_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            # 6. 把这条 merge rule 追加到 merges
            self.merges.append(pair)

            corpus = [self._merge(tokens, pair, next_id) for tokens in corpus]
            next_id += 1

    def _merge(self, tokens, pair, new_id):
        # 在一个 token 序列中，从左到右扫描，把所有出现的指定 pair 替换成新的 token id
        out = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        for new_id, pair in enumerate(self.merges, start=256):
            tokens = self._merge(tokens, pair, new_id)
        return tokens

    def decode(self, tokens):
        data = b"".join(self.vocab[t] for t in tokens)
        return data.decode("utf-8", errors="replace")
```

#### 2.1.2 额外的控制信息：Added Token 与 Chat Template

**Added Token**

人为加入 tokenizer 的 token，不是 BPE 从语料里自然学出来的，作用是保证某些字符串不被拆开。例如：

```text
<think>  </think>
<tool_call>  </tool_call>
<|im_start|>  <|im_end|>
```

如果不加入，`<think>` 可能被拆成 `<`、`think`、`>` 三个 token；加入后它就是一个完整 token。

**Special Token**

有特殊控制语义的 added token，常见例子：

| Token 类型 | 作用 | 示例 |
|-----------|------|------|
| BOS token | 序列开始 | `<\|im_start\|>` |
| EOS token | 序列结束 | `<\|im_end\|>` |
| PAD token | 批处理补齐 | `<\|endoftext\|>` |
| UNK token | 未知 token | `<unk>` |

**Chat Template**

Chat Template 本质上是一段 Jinja 模板。Jinja 是一种模板语言，常用来把结构化数据渲染成字符串；在 tokenizer 里，它的作用是把 `messages` / `tools` 这些数据结构渲染成模型真正看到的 prompt 文本。

例如：

```python
messages = [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "你好"}
]
```

经过 Chat Template 后，会变成：

```text
<|im_start|>system
你是一个助手<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
```

一个 Chat Template 示例（ChatML 风格，支持思考过程与工具调用）：

```jinja
{%- if messages[0].role == "system" %}
{{- "<|im_start|>system\n" + messages[0].content + "<|im_end|>\n" }}
{%- endif %}

{%- for message in messages %}
{%- if message.role == "user" %}
{{- "<|im_start|>user\n" + message.content + "<|im_end|>\n" }}
{%- elif message.role == "assistant" %}
{{- "<|im_start|>assistant\n" }}
{{- "<think>\n" }}
{{- message.reasoning_content | default("") }}
{{- "\n</think>\n\n" }}
{{- message.content }}
{{- "<|im_end|>\n" }}
{%- elif message.role == "tool" %}
{{- "<|im_start|>user\n" }}
{{- "<tool_response>\n" + message.content + "\n</tool_response>" }}
{{- "<|im_end|>\n" }}
{%- endif %}
{%- endfor %}

{%- if add_generation_prompt %}
{{- "<|im_start|>assistant\n<think>\n\n</think>\n\n" }}
{%- endif %}
```

**Jinja 常见语法**

| 语法 | 含义 |
|------|------|
| `{{ ... }}` | 输出变量或表达式 |
| `{% ... %}` | 控制逻辑，比如 if、for |
| `{%- ... -%}` | 控制逻辑，同时裁剪前后的空白和换行，让最终 prompt 更干净 |

**这段模板做了什么**

- **system 消息**：如果第一条是 system，渲染成 `<|im_start|>system\n系统内容<|im_end|>`
- **user 消息**：渲染成 `<|im_start|>user\n用户内容<|im_end|>`
- **assistant 消息**：渲染成 `<|im_start|>assistant\n<think>\n思考内容\n</think>\n\n最终回答<|im_end|>`
- **tools 工具定义**：如果传入 tools，会在 system 段插入：

  ```text
  # Tools
  <tools>
  工具定义 JSON
  </tools>
  ```

- **tool_call**：如果 assistant 有工具调用，渲染成：

  ```text
  <tool_call>
  {"name": "函数名", "arguments": {...}}
  </tool_call>
  ```

- **tool_response**：工具返回渲染成：

  ```text
  <tool_response>
  工具返回内容
  </tool_response>
  ```

- **add_generation_prompt**：如果要让模型开始生成 assistant 回复，模板最后会追加：

  ```text
  <|im_start|>assistant
  <think>

  </think>


  ```

### 2.2 训练 Tokenizer

使用 Hugging Face `tokenizers` 库训练一个 tokenizer，需要确定以下配置：

| 配置项 | 本次实践采用的方案 |
|--------|-------------------|
| 词表大小 | 6400 |
| 训练语料 | 完整的预训练数据，约 8.3GB，共 847 万条文本 |
| Special Token 与 Added Token | 包括对话、思考、工具调用及预留 Token 等 |
| Chat Template | ChatML 格式，使用 `<\|im_start\|>` 和 `<\|im_end\|>` 组织多轮对话，并支持思考过程与工具调用 |

```python
# 1. 创建一个空的 BPE tokenizer
tokenizer = Tokenizer(models.BPE())

# 2. 设置 ByteLevel 预分词器
# 它会先把文本转成 UTF-8 bytes，再映射成安全的 Unicode 字符表示。
# 例如：
#   " hello\n" -> "ĠhelloĊ"
#   "你"       -> "ä½ł"
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

# 3. 创建 BPE 训练器
# 它定义 tokenizer 要怎么训练：目标词表大小、初始字符表、特殊 token 等。
trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=all_special_tokens
)

# 4. 从文本迭代器训练 tokenizer
# texts 通常是一个 iterator/generator，每次产出一段训练文本。
# 训练过程中，BPE 会反复统计高频相邻 token pair，并将它们合并进词表。
tokenizer.train_from_iterator(texts, trainer=trainer)

# 5. 设置 ByteLevel 解码器
# 编码前用了 ByteLevel 表示，解码时也要用 ByteLevel 还原。
# 例如：
#   "Ġ"   -> " "
#   "Ċ"   -> "\n"
#   "ä½ł" -> "你"
tokenizer.decoder = decoders.ByteLevel()

# 6. 注册 special tokens
tokenizer.add_special_tokens(special_tokens_list)
```

```python
from transformers import PreTrainedTokenizerFast

# 7. 用 Transformers 的 FastTokenizer 包装底层 tokenizers.Tokenizer
# 这一步是为了生成 tokenizer_config.json、special_tokens_map.json 等 Transformers 需要的配置文件。
bos_token = "<|im_start|>"
eos_token = "<|im_end|>"
pad_token = "<|endoftext|>"
unk_token = "<|endoftext|>"

core_special_tokens = {bos_token, eos_token, pad_token, unk_token}

additional_special_tokens = [
    t for t in special_tokens_list
    if t not in core_special_tokens
]

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token=bos_token,
    eos_token=eos_token,
    pad_token=pad_token,
    unk_token=unk_token,
    additional_special_tokens=additional_special_tokens,
    model_max_length=131072,
    clean_up_tokenization_spaces=False,
)

# 8. 设置 chat template
# 它定义 messages 如何被渲染成模型训练/推理用的 prompt。
fast_tokenizer.chat_template = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' }}"
    "{{ message['content'] }}"
    "{{ '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)

# 9. 保存完整 tokenizer 目录
# 会生成 tokenizer.json、tokenizer_config.json、special_tokens_map.json 等文件。
fast_tokenizer.save_pretrained(tokenizer_dir)
```

最终 tokenizer 以两个文件的形式保存：

- **`tokenizer.json`**：记录一段文本如何被切成 token id，以及 token id 又如何被还原成文本。
- **`tokenizer_config.json`**：记录这个 tokenizer 在大模型训练和推理时应该如何使用。

---

## 3、模型结构

> 参见：[Transformer向量与参数详解 · 二、完整参数列表](../深度学习/Transformer向量与参数详解.md#二完整参数列表从输入到输出)（W_Q/W_K/W_V/W_O、gate_proj/up_proj/down_proj 等完整形状速查表）。

### 3.1 模型结构概览

一个典型的 Decoder-Only Transformer 由以下部分堆叠而成：

```text
输入 token id
    │
    ▼
Embedding（词嵌入）
    │
    ▼
┌─────────────────────────┐
│   N × Transformer Block  │   ← 每个 Block: PreNorm + Attention(GQA+RoPE) + 残差
│                          │      PreNorm + FFN(SwiGLU) + 残差
└─────────────────────────┘
    │
    ▼
Final RMSNorm
    │
    ▼
LM Head（与 Embedding 权重共享）
    │
    ▼
下一个 token 的概率分布
```

### 3.2 注意力机制

#### 3.2.1 注意力机制的公式

注意力机制的核心公式（Scaled Dot-Product Attention）：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

> 参见：[4、自注意力机制(Self-attention)](../李宏毅机器学习课程/4、自注意力机制(Self-attention).md)（注意力机制的直觉推导）、[8、注意力机制和Transformer · 自注意力机制](../邱熙鹏深度学习/8、注意力机制和Transformer.md#自注意力机制)（QKV 模式的图解）。

#### 3.2.2 多头注意力机制（MHA）

单个注意力头只能在一个表示子空间中计算 token 之间的关系。多头注意力机制（Multi-Head Attention）将输入映射到多个不同的表示子空间，并行计算多组注意力，从而捕获不同类型的上下文关系。

每个注意力头的维度为：

$$
d_{head} = \frac{d_{model}}{n_{heads}}
$$

对于第 $i$ 个注意力头：

$$
\text{head}_i = \text{Attention}(Q W_i^Q,\ K W_i^K,\ V W_i^V)
$$

多个注意力头的结果拼接后，再进行一次线性变换：

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

因此，多头注意力机制可以理解为：**从多个不同的角度读取上下文信息，再将各个头的结果融合起来**。

#### 3.2.3 GQA、MQA

> 参见：[Transformer向量与参数详解 · 2.2 Self-Attention](../深度学习/Transformer向量与参数详解.md#22-self-attention) 中标准 MHA 的 W_Q/W_K/W_V 形状，可对照下方 GQA 中 KV 头数被压缩的变化。

在自回归推理过程中，历史 token 的 Key 和 Value 不会发生变化，因此可以将它们保存下来，避免重复计算，这部分缓存称为 **KV Cache**。

- **MHA**：每个 Query 头都有一组独立的 Key 和 Value。
- **GQA**（Grouped-Query Attention）：多个 Query 头被划分为若干组，同一组内的 Query 头共享一组 Key 和 Value。
- **MQA**（Multi-Query Attention）：所有 Query 头共享同一组 Key 和 Value。

每组包含的 Query 头数为 $n_{heads} / n_{kv\_heads}$，因此：

| 注意力机制 | Query 头数 | KV 头数 | KV Cache 相对大小 |
|-----------|-----------|---------|-------------------|
| MHA | $h$ | $h$ | $1$ |
| GQA | $h$ | $g$（$1 < g < h$） | $g / h$ |
| MQA | $h$ | $1$ | $1 / h$ |

MHA 和 MQA 都可以看作 GQA 的特殊情况：$g = h$ 时是 MHA，$g = 1$ 时是 MQA。GQA 在多头注意力的表达能力与 MQA 的推理效率之间取得了折中。

#### 3.2.4 注意力机制的优化

| 优化方向 | 代表方法 | 主要收益 |
|---------|---------|---------|
| 优化 KV Cache | MQA、GQA、MLA | 通过共享 KV 头或压缩 K、V 表示，减少 KV Cache 占用和 Decode 阶段的显存带宽 |
| 稀疏注意力 | SWA、DSA / IndexShare、CSA / HCA | 只计算局部窗口或动态选出的重要 Token/Block，降低长上下文的注意力计算量 |
| 线性注意力 | DeltaNet / Gated DeltaNet、KDA、Lightning Attention | 用固定大小的递归状态压缩历史信息，使计算量随序列长度近似线性增长，并显著减少长文本推理缓存 |
| 工程优化 | FlashAttention、PagedAttention | 不改变注意力的核心数学定义，通过减少显存读写、优化 KV Cache 分配来降低延迟并提高吞吐 |

#### 3.2.5 代码实现（GQA）

MiniMind 使用 GQA 作为注意力的实现方式，超参数如下：

| 参数 | 取值 | 说明 |
|------|------|------|
| 隐藏维度 | 768 | 每个 token 的向量维度 |
| Query 头数 | 8 | 共有 8 个 Query 头 |
| 每头维度 | 96 | $768/8=96$ |
| KV 头数（组数） | 4 | 共有 4 组 Key 和 Value |
| 每组 Query 头数 | 2 | 每 2 个 Query 头共享一组 K、V |

```python
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MiniMindConfig:
    hidden_size: int = 768
    num_attention_heads: int = 8
    num_key_value_heads: int = 4

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将每个 KV 头复用 n_rep 次，使 KV 头数与 Query 头数一致。

    输入:  [B, T, H_KV, D]
    输出:  [B, T, H_KV * n_rep, D] = [B, T, H_Q, D]
    """
    batch_size, seq_len, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, num_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, num_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()

        assert config.hidden_size % config.num_attention_heads == 0
        assert config.num_attention_heads % config.num_key_value_heads == 0

        self.num_heads = config.num_attention_heads          # H_Q = 8
        self.num_kv_heads = config.num_key_value_heads       # H_KV = 4
        self.n_rep = self.num_heads // self.num_kv_heads     # 8 / 4 = 2
        self.head_dim = config.head_dim                      # 768 / 8 = 96
        self.dropout = config.dropout

        # Q 有 8 个头，K、V 各只有 4 个头。
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x, past_key_value=None, use_cache=False):
        batch_size, seq_len, _ = x.shape

        # 1. 生成 Q、K、V，此时 Q 头数是 8，KV 头数是 4。
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # 2. 在这里对 Q、K 应用 QK-RMSNorm 和 RoPE（本节先略过旋转位置编码，具体实现见 3.3 节）。

        # 3. 追加历史 KV Cache。Cache 保存的仍是 4 个 KV 头，
        #    不要把 repeat_kv 之后的 8 个头存进 Cache。
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        present_key_value = (k, v) if use_cache else None

        # 4. 在真正计算注意力前，让每组 K、V 服务于 2 个 Query 头。
        #    repeat_kv 只是共享/展开数据，不会引入新的可学习参数。
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # [B, T, H, D] -> [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 5. 与普通多头注意力相同的 Scaled Dot-Product Attention。
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Decoder-Only mask：每个 token 只能看到自己和它之前的 token。
        kv_len = k.size(-2)
        past_len = kv_len - seq_len
        causal_mask = torch.triu(
            torch.ones(seq_len, kv_len, dtype=torch.bool, device=x.device),
            diagonal=past_len + 1,
        )
        scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        attention_weights = F.softmax(scores.float(), dim=-1).type_as(q)

        output = attention_weights @ v

        # 6. 合并 8 个 Query 头，投影回隐藏维度 768。
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)
        return output, present_key_value
```

### 3.3 位置编码

#### 3.3.1 旋转位置编码（RoPE）

注意力机制本身没有体现 token 的顺序，需要引入位置编码。

**二维向量的旋转**

二维向量旋转角度 $\theta$ 可以写成矩阵形式：

$$
R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

旋转不会改变向量的长度，并且满足角度可加性：$R(\theta_1) R(\theta_2) = R(\theta_1 + \theta_2)$。

因此，分别旋转 $m\theta$ 和 $n\theta$ 的两个向量，它们的内积只与两个位置的差 $m-n$ 有关。这是 RoPE 能够表示相对位置的核心。

**RoPE 如何加入位置信息**

对于位置 $m$ 的 Query 和位置 $n$ 的 Key，RoPE 分别对它们进行旋转：

$$
q_m = R(m\theta) q, \qquad k_n = R(n\theta) k
$$

旋转后的注意力分数为：

$$
q_m^\top k_n = q^\top R(m\theta)^\top R(n\theta) k = q^\top R((n-m)\theta) k
$$

因此，旋转后的 Query 和 Key 各自包含绝对位置，二者的内积则自然包含相对位置。RoPE 可以概括为：

> 将"位置"表示为旋转角度，将"相对距离"表示为两个旋转角度之差。

**从二维推广到高维**

注意力头通常包含多个维度。RoPE 将这些维度两两分组，把每一组看作一个二维向量，不同的维度组使用不同的旋转频率：

$$
\theta_i = \text{base}^{-2i/d}, \qquad i = 0, 1, \dots, d/2-1
$$

其中 $d$ 是注意力头的维度。第 $i$ 组向量在位置 $m$ 处的旋转角度为 $m\theta_i$。

高频维度（$i$ 较小）对较短的位置变化更敏感，低频维度（$i$ 较大）变化更慢，可以表示更长的距离。多组频率结合起来，使模型能够区分不同位置和不同尺度的相对距离。

#### 3.3.2 长度外推的方式

长度外推是指：模型在训练时只见过较短的序列，但在推理时需要处理更长的上下文（例如训练长度 2048，推理时扩展到 32768）。

对于 RoPE，位置 $m$ 在第 $i$ 组维度上的旋转角度为 $m\theta_i$。当 $m$ 超出训练长度后，旋转角度也进入模型从未见过的范围，注意力分数可能变得不稳定。

长度外推的核心问题：**如何调整 RoPE 的旋转频率，让更长的位置尽量使用模型熟悉的角度范围，同时保留对短距离的分辨能力**。

**位置插值（Position Interpolation）**

设模型的原始训练长度为 $L$，目标长度为 $L' = sL$（$s$ 是扩展倍数）。位置插值将新位置 $m$ 缩放回原始位置范围：

$$
m' = m / s
$$

它等价于将所有维度的旋转频率都缩小为原来的 $1/s$。例如将 2048 扩展到 8192 时，新位置 8191 会被映射到接近原位置 2048 的区域。

位置插值避免了模型看到过大的旋转角度，但也会压缩相邻位置之间的角度差；如果对所有频率都使用相同的缩放比例，模型对局部位置的分辨能力可能下降。

**NTK-aware 缩放**

RoPE 的不同维度具有不同的频率：高频维度主要描述相邻 token 之间的局部关系，低频维度变化较慢，更适合描述较长的距离。

NTK-aware 缩放不再对所有频率做相同的缩放，而是通过增大 RoPE 的 `base` 改变频率分布。由于第 0 组维度的频率不受 base 影响，而后续维度受影响逐渐增大，这种方式可以较好地保留高频维度的局部分辨率，同时压缩低频维度的旋转角度，以表示更长的距离。

**YaRN**

YaRN 进一步按频率区间采用不同的缩放策略：

- 高频部分保持原频率，尽量保留模型的局部位置能力；
- 低频部分使用位置插值，将频率缩小为原来的 $1/s$，以覆盖更长的上下文；
- 中间频率使用平滑的线性过渡，避免频率突然变化。

设 $\gamma_i \in [0,1]$ 表示第 $i$ 组维度的插值程度，YaRN 中的新频率可写为：

$$
\theta_i' = \theta_i \cdot \left(1 - \gamma_i + \frac{\gamma_i}{s}\right)
$$

当 $\gamma_i = 0$ 时，频率保持不变；当 $\gamma_i = 1$ 时，频率缩小为原来的 $1/s$。`beta_fast` 和 `beta_slow` 用于确定哪些维度保留原频率、哪些维度进行插值，中间部分则通过 $\gamma_i$ 平滑过渡。

与简单的位置插值相比，YaRN 在扩展长度的同时，减少了对原有短文本能力的影响；它还可以使用 `attention_factor` 调整 Query 和 Key 的幅度，以补偿上下文长度变化对注意力分布的影响。

#### 3.3.3 旋转位置编码的代码实现

相关参数：

| 参数 | 取值 | 说明 |
|------|------|------|
| `rope_theta` | $10^6$ | RoPE 的 base，控制不同维度的旋转频率 |
| `factor` | 16 | YaRN 的位置扩展倍数 |
| `original_max_position_embeddings` | 2048 | 模型原始训练的上下文长度 |
| `beta_fast` | 32 | 高频区域的旋转圈数阈值，高于该值时保留原频率 |
| `beta_slow` | 1 | 低频区域的旋转圈数阈值，低于该值时完全插值 |
| `attention_factor` | 1.0 | 对 RoPE 生成的 cos 和 sin 进行幅度缩放 |

**预计算 cos 和 sin**

```python
import math
import torch

def precompute_freqs_cis(
    dim,
    end,
    rope_base=1e6,
    rope_scaling=None,
):
    # 1. 计算 dim/2 组 RoPE 频率
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
    attention_factor = 1.0

    # 2. 可选：使用 YaRN 调整不同维度的频率
    if rope_scaling is not None:
        original_length = rope_scaling["original_max_position_embeddings"]
        factor = rope_scaling["factor"]
        beta_fast = rope_scaling["beta_fast"]
        beta_slow = rope_scaling["beta_slow"]
        attention_factor = rope_scaling["attention_factor"]

        def dimension_from_rotations(beta):
            return (
                dim
                * math.log(original_length / (beta * 2 * math.pi))
                / (2 * math.log(rope_base))
            )

        low = max(math.floor(dimension_from_rotations(beta_fast)), 0)
        high = min(math.ceil(dimension_from_rotations(beta_slow)), dim // 2 - 1)

        gamma = torch.clamp(
            (torch.arange(dim // 2).float() - low) / max(high - low, 0.001),
            0, 1,
        )

        inv_freq = inv_freq * (1 - gamma + gamma / factor)

    # 3. 计算所有位置的旋转角度
    positions = torch.arange(end)
    angles = torch.outer(positions, inv_freq)

    # 4. 生成旋转需要的 cos 和 sin
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1) * attention_factor
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1) * attention_factor

    return cos, sin
```

`angles` 的形状为 `[max_position_embeddings, head_dim / 2]`；复制后，`cos` 和 `sin` 的形状为 `[max_position_embeddings, head_dim]`。

**对 Query 和 Key 进行旋转**

将向量分成前后两半，构造 `rotate_half` 并完成旋转：

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    # [seq_len, head_dim]
    # -> [1, seq_len, 1, head_dim]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin

    return q, k
```

例如，当 `head_dim=8` 时，配对方式是：`(x0, x4), (x1, x5), (x2, x6), (x3, x7)`。

**在注意力层中应用 RoPE**

RoPE 应用在线性投影和 QK-RMSNorm 之后、计算注意力分数之前，并且只作用于 Query 和 Key：

```python
def forward(self, x, position_embeddings):
    batch_size, seq_len, _ = x.shape

    q = self.q_proj(x).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
    k = self.k_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
    v = self.v_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

    q = self.q_norm(q)
    k = self.k_norm(k)

    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # 后续使用旋转后的 Q、K 计算注意力分数
```

### 3.4 FFN

注意力机制负责在不同 token 之间传递和聚合信息，FFN（Feed-Forward Network，前馈神经网络）则负责对每个 token 的表示进行非线性变换。

FFN 对序列中的每个 token 独立地使用同一组参数，因此不会直接混合不同位置的信息，只在最后一个隐藏维度上进行计算：

| 模块 | 功能 |
|------|------|
| Attention | token 与 token 之间交换信息 |
| FFN | 每个 token 独立地加工已经得到的信息 |

虽然 FFN 的结构比注意力简单，但它通常占据 Transformer 中很大一部分参数量和计算量。

> 参见：[前馈神经网络介绍](../深度学习/前馈神经网络介绍.md)（FFN 的基础结构与 `nn.Linear` 计算细节）。

#### 3.4.1 基本的 FFN 与 SwiGLU

> 参见：[激活函数和损失函数 · 现代大模型为什么不用 ReLU 了（ReLU → GELU → SiLU → SwiGLU）](../深度学习/激活函数和损失函数.md#现代大模型为什么不用-relu-了relu-gelu-silu-swiglu)。

最基础的 FFN 由两个线性层和一个非线性激活函数组成：

$$
\text{FFN}(x) = W_2\, \sigma(W_1 x)
$$

第一个线性层将隐藏维度从 $d_{model}$ 扩展到中间维度 $d_{ff}$，激活函数 $\sigma$ 引入非线性，第二个线性层再将它投影回 $d_{model}$。

如果只连续使用两个没有激活函数的线性层，它们仍然等价于一个线性变换 $W_2 W_1 x$，无论叠加多少层，都无法表达复杂的非线性关系。因此，激活函数是 FFN 不可缺少的一部分。

现代大模型通常会在基本 FFN 上加入门控机制。MiniMind 使用的是 **SwiGLU** 风格的 FFN：

$$
\text{FFN}_{SwiGLU}(x) = W_{down}\big(\text{SiLU}(W_{gate}x) \odot W_{up}x\big)
$$

其中 $\odot$ 表示逐元素相乘。输入同时经过两条分支：

- `gate_proj` 经过 SiLU 后生成门控值，决定各个中间特征应该通过多少；
- `up_proj` 生成要被传递的特征；
- 两条分支逐元素相乘后，由 `down_proj` 投影回隐藏维度。

与基本 FFN 相比，门控 FFN 可以根据当前 token 的内容动态调节中间特征，具有更强的表达能力。这里的 SwiGLU 指的是"Swish/SiLU 激活函数 + GLU 门控结构"。

#### 3.4.2 MoE

Dense FFN 对每个 token 都使用同一套参数。如果希望增加模型容量，最直接的方式是继续增大 $d_{ff}$，但所有参数都会参与每一次计算，计算成本也会随之增加。

**MoE**（Mixture of Experts，混合专家）将一个 Dense FFN 替换成多个独立的 FFN 专家，并使用一个 Router 为每个 token 选择少数几个专家。

假设共有 $N$ 个专家，Router 首先根据 token 的隐藏状态 $x$ 计算每个专家的分数，然后只保留分数最高的 $k$ 个专家。设选中的专家集合为 $\mathcal{T}$，输出为：

$$
y = \sum_{i \in \mathcal{T}} g_i \cdot \text{FFN}_i(x)
$$

其中 $g_i$ 是 Top-k 范围内归一化后的权重。

> 参见：[Moe · MoE架构模型](../深度学习/Moe.md#moe架构模型)（路由机制、训练与推理挑战的详细展开）。

#### 3.4.3 代码实现（Dense FFN）

MiniMind 使用 Dense FFN，对应配置如下：

| 参数 | 取值 | 说明 |
|------|------|------|
| 隐藏维度 | 768 | 每个 token 的向量维度 |
| 中间维度 | 2432 | FFN 中间层的维度，约为隐藏维度的 3.17 倍 |
| 激活函数 | SiLU | gate_proj 分支使用的激活函数 |

```python
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MiniMindConfig:
    hidden_size: int = 768
    intermediate_size: int = 2432
    hidden_act: str = "silu"


class FeedForward(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size

        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        # 两条分支的形状都是 [B, T, intermediate_size]
        gate = self.act_fn(self.gate_proj(x))
        value = self.up_proj(x)

        # 逐元素门控后投影回 [B, T, hidden_size]
        return self.down_proj(gate * value)
```

### 3.5 Transformer Block

前面几节分别介绍了注意力、位置编码和 FFN。一个 Transformer Block 把它们组装成一个可重复堆叠的基本单元：每个 Block 包含一个注意力子层和一个 FFN 子层，并围绕它们加上归一化和残差连接。

#### 3.5.1 归一化：LayerNorm vs RMSNorm

> 参见：[Transformer向量与参数详解 · 2.1 Layer Norm](../深度学习/Transformer向量与参数详解.md#21-layer-norm)（LayerNorm 的 gamma/beta 参数形状）。

归一化的作用是把每个 token 的隐藏向量拉回稳定的范围，避免随着层数加深，数值不断放大或缩小，从而导致训练不稳定。

- **LayerNorm**：对每个 token 的向量先减去均值、再除以标准差，最后用可学习的缩放和偏置还原。
- **RMSNorm**：去掉了减均值和偏置，只用均方根（Root Mean Square）做缩放，再乘一个可学习的权重：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma
$$

相比之下，RMSNorm 少了一次均值计算和偏置参数，计算更轻，而在大模型上经验效果与 LayerNorm 接近。因此 LLaMA 之后的主流模型（包括 MiniMind）几乎都采用 RMSNorm。

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # 可学习的缩放 γ

    def forward(self, x):
        # 用均方根归一化，在 float 上计算避免半精度下溢
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * norm.float()).type_as(x)
```

#### 3.5.2 残差连接

随着网络变深，梯度在反向传播中容易消失或爆炸，深层几乎学不到东西。残差连接（Residual Connection）把子层的输入直接加到输出上，为梯度提供了一条"直通"路径：

$$
y = x + f(x)
$$

即使 $f(x)$ 学到接近 0，信息也能无损地通过 $x$ 传到下一层。一个 Block 中有两条残差，分别绕过注意力和 FFN。残差连接是 Transformer 能够堆叠到几十上百层的关键。

#### 3.5.3 PreNorm vs PostNorm

归一化放在残差的哪个位置，决定了是 PreNorm 还是 PostNorm。

| 形式 | 计算顺序 | 残差路径 | 训练稳定性 |
|------|---------|---------|-----------|
| PostNorm（原始 Transformer、GPT-2） | $y = \text{Norm}(x + f(x))$ | 经过 Norm | 较差 |
| PreNorm（现代大模型） | $y = x + f(\text{Norm}(x))$ | 不经过 Norm | 较好 |

PostNorm 中残差路径上夹着一个 Norm，梯度必须穿过它才能回传，深层时容易出现训练不稳定；PreNorm 中从输出到输入存在一条完全不经过 Norm 的残差通路，梯度流动更顺畅，训练更稳定。主流大模型中的绝大多数都采用 PreNorm。

#### 3.5.4 代码实现

```python
class MiniMindBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 1. 注意力子层：先归一化，再计算注意力，最后残差相加
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),       # PreNorm
            position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states = hidden_states + residual

        # 2. FFN 子层：同样先归一化，再前馈，最后残差相加
        residual = hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value
```

#### 3.5.5 完整结构与参数量

把 Embedding、N 个 Block、Final Norm 和 LM Head 拼起来，就是一个完整的语言模型：

```python
class MiniMindModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)   # 输入端
        self.layers = nn.ModuleList([
            MiniMindBlock(l, config) for l in range(config.num_hidden_layers)     # 8 个 Block
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)          # Final Norm
        # RoPE 的 cos/sin 在这里预计算（见 3.3）


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)   # 输出端
        # tie_word_embeddings=True：输入 embedding 与输出 lm_head 共享同一份权重
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
```

注意 `tie_word_embeddings=True`：输入端 Embedding 和输出端 LM Head 共享同一份 `[V, d]` 权重，因此这份参数只算一次。

**参数量计算**

关键配置：

| 参数 | 取值 | 说明 |
|------|------|------|
| 词表大小 $V$ | 6400 | Embedding / LM Head 的行数 |
| 隐藏维度 $d$ | 768 | 每个 token 的向量维度 |
| 层数 $L$ | 8 | Transformer Block 的数量 |
| Query 头数 $h$ | 8 | 每头维度 $d/h=96$ |
| KV 头数 $g$ | 4 | GQA，每 2 个 Q 头共享一组 KV |
| FFN 中间维度 $d_{ff}$ | 2432 | 约为 $d$ 的 3.17 倍 |

逐模块计算参数量（无偏置、权重共享）：

| 模块 | 计算式 | 参数量 |
|------|--------|--------|
| Embedding | $V \times d$ | 4,915,200 |
| q_proj | $d \times d$ | 589,824 |
| k_proj | $d \times (d \cdot g/h)$ | 294,912 |
| v_proj | $d \times (d \cdot g/h)$ | 294,912 |
| o_proj | $d \times d$ | 589,824 |
| q_norm + k_norm | $2 \times d_{head}$ | 192 |
| gate / up / down | $3 \times d \times d_{ff}$ | 5,603,328 |
| 单 Block 小计 | 上述 Attention + FFN 之和 | 7,372,992 |
| 8 个 Block | $8 \times$ 单 Block | 58,996,224 |
| Final RMSNorm | $d$ | 768 |
| LM Head（与 Embedding 共享） | — | 0 |
| **总计** | | **63,912,192 ≈ 64M** |

把上述计算抽象成一般公式：

$$
\text{Params} \approx V d + L \left[4d^2 \cdot \frac{h+g}{2h} + 3 d\, d_{ff}\right]
$$

其中 $4d^2$ 来自 q_proj 与 o_proj，$k\_proj/v\_proj$（GQA 时缩小为 $g/h$ 倍）。代入 MiniMind 的配置，便得到约 64M。

**附：主流的模型结构对比**

| 序号 | 模型名 | 发布日期 | FFN架构 | 上下文 | 总参数 | 激活参数 | 位置编码 | Attention1 | Attention2 | 激活函数 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | GPT-2 XL | 2019/11/05 | Dense | 1K | 1.5B | 1.5B | PE | MHA | — | GELU |
| 2 | GPT-3 | 2020/05/28 | Dense | 2K | 175B | 175B | PE | MHA | — | GELU |
| 3 | LLaMA 7B | 2023/02/24 | Dense | 2K | 7B | 7B | RoPE | MHA | — | SiLU |
| 4 | Llama 2 70B | 2023/07/18 | Dense | 4K | 70B | 70B | RoPE | GQA | — | SiLU |
| 5 | Mistral 7B | 2023/10/10 | Dense | 8K | 7.3B | 7.3B | RoPE | GQA | SWA | SiLU |
| 6 | Llama 3 8B | 2024/04/18 | Dense | 8K | 8B | 8B | RoPE | GQA | — | SiLU |
| 7 | DeepSeek-V2 | 2024/05/07 | Sparse | 128K | 236B | 21B | RoPE | MLA | — | SiLU |
| 8 | Qwen2.5 72B | 2024/09/19 | Dense | 128K | 72.7B | 72.7B | RoPE | GQA | — | SiLU |
| 9 | DeepSeek-V3 | 2024/12/26 | Sparse | 128K | 671B | 37B | RoPE | MLA | — | SiLU |
| 10 | DeepSeek-R1 | 2025/01/20 | Sparse | 128K | 671B | 37B | RoPE | MLA | — | SiLU |
| 11 | Gemma 3 27B | 2025/03/12 | Dense | 128K | 27B | 27B | RoPE | GQA | SWA | GELU |
| 12 | Llama 4 Maverick | 2025/04/05 | Sparse | 1M | 400B | 17B | RoPE + NoPE | GQA | — | SiLU |
| 13 | Qwen3 235B-A22B | 2025/04/29 | Sparse | 128K | 235B | 22B | RoPE | GQA | — | SiLU |
| 14 | gpt-oss-120b | 2025/08/05 | Sparse | 128K | 117B | 5.1B | RoPE + YaRN | GQA | SWA / Full | SiLU |
| 15 | Kimi Linear 48B-A3B | 2025/10/30 | Sparse | 1M | 48B | 3B | RoPE | MLA | KDA | SiLU |
| 16 | DeepSeek-V3.2 | 2025/12/01 | Sparse | 128K | 671B | 37B | RoPE | MLA | DSA | SiLU |
| 17 | Kimi K2.5 | 2026/01/27 | Sparse | 256K | 1T | 32B | RoPE | MLA | — | SiLU |
| 18 | MiniMax-M2.5 | 2026/02/12 | Sparse | 197K | 230B | 10B | RoPE | GQA | — | SiLU |
| 19 | Nemotron 3 Super 120B-A12B | 2026/03/11 | Sparse | 1M | 120B | 12B | RoPE | GQA | Mamba-2 | SiLU |
| 20 | Mistral Small 4 119B-A6B | 2026/03/16 | Sparse | 256K | 119B | 6.63B | RoPE | MLA | — | SiLU |
| 21 | Gemma 4 31B | 2026/04/02 | Dense | 256K | 30.7B | 30.7B | p-RoPE | GQA | SWA / Global | GELU |
| 22 | Qwen3.6 35B-A3B | 2026/04/15 | Sparse | 262K | 35B | 3B | RoPE | GatedAttn | DeltaNet | SiLU |
| 23 | DeepSeek-V4-Pro | 2026/04/24 | Sparse | 1M | 1.6T | 49B | RoPE | GQA | CSA / HCA | SiLU |
| 24 | GLM-5.2 | 2026/06/16 | Sparse | 1M | 753B | ~40B | RoPE | MLA | DSA + IndexShare | SiLU |

> 说明：除表中列出的字段外，绝大多数模型均采用 RMSNorm + PreNorm + 残差连接（RC）的组合；Gemma 系列例外地同时使用了 Pre-Norm 和 Post-Norm。

---

## 4、训练方式

### 4.1 预训练

> 参见：[Transformer向量与参数详解 · 五、怎么算 Loss（SFT 为例）](../深度学习/Transformer向量与参数详解.md#五怎么算-losssft-为例)（logits 到 loss 的图解计算过程）。

#### 4.1.1 损失函数：预测下一个 token

预训练阶段，大模型要完成的任务非常朴素：**根据前面的 token，预测下一个 token**。

假设一段文本被 tokenizer 转换为 $x_1, x_2, \dots, x_T$。模型在看到前 $t-1$ 个 token 后，对词表中每个 token 输出一个概率分布 $P_\theta(\cdot \mid x_{<t})$。我们希望模型给真实下一个 token 的概率尽量大，用交叉熵来衡量预测分布与真实标签之间的差距：

$$
\mathcal{L} = -\frac{1}{T-1}\sum_{t=1}^{T-1} \log P_\theta(x_{t+1} \mid x_{\le t})
$$

即对序列中每一个预测位置求真实 token 的负对数概率，再取平均。

**通过 shift 从无监督文本中构造标签**

输入和标签来自同一段文本，但预测位置要错开一位：

```text
原始 token:  <bos>  我    爱    北京   <eos>
模型输入:     <bos>  我    爱    北京
预测目标:       我    爱   北京   <eos>
```

#### 4.1.2 实现代码

在模型内部统一完成 shift 和交叉熵计算：

```python
if labels is not None:
    # 第 t 个 logits 预测第 t+1 个 label
    x = logits[..., :-1, :].contiguous()
    y = labels[..., 1:].contiguous()

    loss = F.cross_entropy(
        x.view(-1, x.size(-1)),
        y.view(-1),
        ignore_index=-100,
    )
```

为了将不同长度的文本堆叠成形状统一的 `[batch_size, max_length]` 张量，并在 GPU 上进行批量并行计算，预训练数据集会在每条序列右侧补齐 padding。padding 只用于对齐长度，不是真实的预测目标，因此将这些位置的 label 设置为 `-100`；`ignore_index=-100` 会让交叉熵忽略这些位置，使 padding 不计入 loss。

```python
tokens = [bos_id] + tokenizer(text).input_ids + [eos_id]
input_ids = tokens + [pad_id] * (max_length - len(tokens))

labels = input_ids.clone()
labels[input_ids == pad_id] = -100
```

训练循环非常简洁：

```python
for input_ids, labels in loader:
    result = model(input_ids, labels=labels)
    loss = result.loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad()
```

#### 4.1.3 数据集

预训练文件是 JSONL，每一行是一条独立 JSON，核心字段只有 `text`：

```json
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助。"}
{"text": "清晨的阳光透过窗帘洒进房间，桌上的书页被风轻轻翻动。"}
{"text": "Transformer 通过自注意力机制建模上下文关系。"}
```

每条 `text` 单独作为一个独立样本。处理流程：一条 text 先 tokenize 成 token 序列，前后加上 BOS/EOS 表示文档边界，再 padding 补齐到固定长度，最终得到一个 `(input_ids, labels)` 样本。padding 位置的 label 会被设成 `-100`，训练时不计入 loss。

数据集参数：

| 参数 | 数值 |
|------|------|
| 数据条数 | 8,468,827（约 846 万条） |
| 总 token 数 | 2,200,383,770（约 2.2 B） |
| 文件大小 | 8.28 GB |

#### 4.1.4 Loss 曲线与 Case 分析

| 提示词 | 模型输出 |
|--------|---------|
| 为什么天空是蓝色的 | 天空之所以呈现蓝色，是因为大气中的氧气和氮气分子对蓝色光的散射。在空气中，蓝色光的波长较短，因此它们更容易被散射，而其他颜色的光波长较长，因此更容易被散射。当太阳光射向地球时，其中的蓝色光波长被散射得更多，而其他颜色的光波长则被散射得更少。因此，天空呈现出蓝色。 |

> 可以观察到：预训练阶段的模型已经学会了流畅的语言表达和基本事实知识，但还不会以"指令-回答"的对话形式组织内容——这正是下一阶段 SFT 要解决的问题。

### 4.2 有监督微调（SFT）

> 参见：[激活函数和损失函数 · SFT 的 Loss：从 token 到 Batch](../深度学习/激活函数和损失函数.md#sft-的-loss从-token-到-batch)（交叉熵从单 token 到整个 batch 的完整推导，及 GRPO Loss 对比）。

#### 4.2.1 损失函数：仍然是交叉熵

预训练与有监督微调（Supervised Fine-Tuning，SFT）的损失函数在形式上完全相同，都是 next-token cross-entropy。区别在于：

- **数据分布不同**：预训练使用通用文本，SFT 使用 system/user/assistant/tool 等结构化指令数据；
- **loss mask 不同**：预训练几乎学习全部非 padding token，MiniMind 的 SFT 只学习 assistant 生成的内容。

例如：

```text
序列:        <system> 你是助手 <user> 1+1=? <assistant> 2 <eos>
预训练 mask:     1       1       1       1        1     1
SFT mask:        0       0       0       0        1     1
```

SFT 的输入中仍然必须保留 system prompt 和用户问题，因为 assistant 的每个 token 都要以它们为上下文；只是这些上下文 token 本身不作为预测目标。

#### 4.2.2 实现代码

先用 chat template 把结构化消息转化为模型真正看到的 token 序列：

```python
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
    tools=tools,
)
input_ids = tokenizer(prompt).input_ids[:max_length]
```

然后构造 labels。初始时所有位置均为 `-100`，代码扫描每个 assistant 起始标记到 EOS 的区间，只有这些位置被恢复成真实 token id：

```python
labels = [-100] * len(input_ids)
i = 0
while i < len(input_ids):
    if input_ids[i:i + len(assistant_bos_ids)] == assistant_bos_ids:
        start = i + len(assistant_bos_ids)
        end = start
        while input_ids[end:end + len(eos_ids)] != eos_ids:
            end += 1
        for j in range(start, end + len(eos_ids)):
            labels[j] = input_ids[j]
        i = end + len(eos_ids)
    else:
        i += 1
```

因此模型仍然调用与预训练相同的 forward：

```python
result = model(input_ids, labels=labels)
loss = result.loss
```

#### 4.2.3 数据集

普通多轮对话：

```json
{
  "conversations": [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是让计算机从数据中学习规律的方法。"},
    {"role": "user", "content": "举一个例子。"},
    {"role": "assistant", "content": "垃圾邮件分类就是一个常见例子。"}
  ]
}
```

包含 Tool Use 的样本还可以在 system 消息中携带工具定义，并包含 assistant tool call 与 tool observation：

```json
{
  "conversations": [
    {"role": "system", "content": "你可以使用工具。", "tools": "[...]"},
    {"role": "user", "content": "把你好世界翻译成英文"},
    {"role": "assistant", "content": "", "tool_calls": "[{...}]"},
    {"role": "tool", "content": "{\"translated_text\": \"Hello World\"}"},
    {"role": "assistant", "content": "Hello World"}
  ]
}
```

数据集参数：

| 参数 | 数值 |
|------|------|
| 数据条数 | 5,109,432（约 511 万条） |
| 总 prompt token 数 | 3,871,139,234（约 3.8 B） |
| 文件大小 | 14.10 GB |

#### 4.2.4 评估：Case 分析

| 提示词 | 模型输出 |
|--------|---------|
| 为什么天空是蓝色的 | 天空是蓝色的原因是因为太阳光在穿过大气层时，其中的分子和颗粒会散射光线，而蓝光波长较短，所以在我们看向天空时，散射的蓝光会更多，所以我们看到的天空呈现蓝色。 |
| 帮我算一下 256 乘以 37 等于多少（可用工具: `calculate_math`） | `<tool_call>{"name": "calculate_math", "arguments": {"expression": "256 * 37"}}</tool_call>` → `[Tool Called]: {"result": "9472"}` → 256 乘以 37 等于 9472。 |

> SFT 之后模型已经能够正确调用工具并利用返回结果作答，说明 chat template + loss mask 的组合成功教会了模型"何时该说话、何时该调用工具"。

### 4.3 强化学习

> 参见：[对齐](../深度学习/对齐.md)（RLHF 在整体对齐流程中的位置）、[verl · 三、典型工作流程](verl.md#三典型工作流程)（工业级 RL 训练框架的 rollout-reward-update 整体流程）。

#### 4.3.1 最简单的 policy-gradient 算法

> 参见：[GRPO与PPO算法详解 · 一、三种算法总览](../深度学习/GRPO与PPO算法详解.md#一三种算法总览)（SFT / GRPO / PPO 的完整对比表，可与下方 REINFORCE 公式交叉阅读）。

**policy gradient 就是带权重的交叉熵**

policy gradient 最核心的公式：

$$
\mathcal{L}_{PG} = -\frac{1}{T}\sum_{t=1}^{T} G_t \log \pi_\theta(a_t \mid s_t)
$$

其中：

- $a_t$ 是模型生成的第 $t$ 个 token；
- $\pi_\theta(a_t \mid s_t)$ 是模型生成这个 token 时给它的概率；
- $G_t$ 是这个 token 得到的累计回报，也是交叉熵前面的权重；
- $T$ 是回答包含的 token 数。

这个公式与 SFT 的交叉熵几乎一样：如果令所有 token 的 $G_t = 1$，policy-gradient loss 就变成了 SFT loss。因此，可以先把 policy gradient 理解为：**给每个 token 的交叉熵乘上一个表示"这次生成有多好或多坏"的权重，再将它们加起来**。

不同的 $G_t$ 会产生不同的训练效果：

| 回报 $G_t$ | loss 符号 | 训练效果 |
|-----------|----------|---------|
| $G_t > 0$ | 负 | 像 SFT 一样，提高该 token 的概率 |
| $G_t = 0$ | 0 | 不更新该 token |
| $G_t < 0$ | 正 | 产生与交叉熵相反的梯度，降低该 token 的概率 |
| $G_t = 2$ | — | 提高概率，力度约为 $G_t=1$ 时的两倍 |

**强化学习与 SFT 的两个核心区别**

1. **强化学习的权重可以为正、为零，也可以为负。** SFT 只能告诉模型"请模仿这个答案"；强化学习还可以告诉模型"这个回答很好""这个回答很差"，并用 $G_t$ 的大小表示好坏程度。
2. **训练数据的来源不同。** SFT 使用的是数据集中提前准备好的标准答案；on-policy policy gradient 使用的是当前模型在线生成的回答。基本流程是：

   1. 给当前模型一批 prompt；
   2. 让模型自己生成回答（rollout）；
   3. 给回答打分；
   4. 用加权求和 $G_t$ 更新模型。

因此，policy gradient 仍然在优化模型生成 token 时的 $\pi_\theta$，只是训练样本由模型自己产生，而且每个样本多了一个可正可负的权重。

**奖励如何分配给每个 token**

需要区分两个概念：

- $r_t$ 是第 $t$ 步结束时，环境立即给出的**奖励**（reward）；
- $G_t$ 是从第 $t$ 步开始，把之后可能得到的奖励也计算在内的累计**回报**（return）。

模型回答一道题：prompt 是"2+3等于多少？"，模型生成"2 + 3 = 5"。规则验证器只有在完整回答生成后，才能判断答案是否正确。因此，一种最直接的奖励记录方式是：前面的 token 奖励都是 0，生成最后一个 token 后得到奖励 1。

| 模型生成的 token | 2 | + | 3 | = | 5 |
|---|---|---|---|---|---|
| 即时奖励 $r_t$ | 0 | 0 | 0 | 0 | 1 |

如果直接用即时奖励 $r_t$ 作为交叉熵的权重，那么只有最后的 "5" 会被提高概率，前面的 "2、+、3、=" 都不会更新。但最后得到正确答案，显然不只是最后一个 token 的功劳：前面每一步生成的内容都会影响后面的结果。

这就是强化学习中的 **credit assignment（贡献分配）**问题：最终得到的奖励，应该怎样分配给之前做出的动作？对于语言模型来说，每个生成的 token 都是一个动作。

一种最直接的解决办法，是让前面的 token 也能获得它在未来实际观察到的奖励，这就是**蒙特卡洛回报**。

**蒙特卡洛回报：把未来奖励向前传递**

模型先生成完整回答，等所有奖励都已经观察到以后，再从后向前计算每个 token 的累计回报。第 $t$ 个 token 的回报为：

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$

这个公式可以这样读：

- $r_t$ 是当前这一步立即得到的奖励，不衰减；
- $r_{t+1}$ 是下一步才得到的奖励，乘一次 $\gamma$；
- $r_{t+2}$ 距离当前两步，乘 $\gamma^2$；
- 越晚得到的奖励，距离当前 token 越远，乘的 $\gamma$ 次数就越多。

$\gamma$ 称为**衰减因子**，它控制未来奖励可以向前传播多远：

- $\gamma = 0$：只考虑当前即时奖励，未来奖励完全不会向前传递；
- $0 < \gamma < 1$：未来奖励会向前传递，但距离越远，影响越小；
- $\gamma = 1$：未来奖励不衰减，之前的所有 token 都能获得完整的最终奖励。

回到刚才的例子，即时奖励为 $[0,0,0,0,1]$，令 $\gamma = 0.9$，每个 token 的蒙特卡洛回报为：

| 模型生成的 token | 2 | + | 3 | = | 5 |
|---|---|---|---|---|---|
| 即时奖励 $r_t$ | 0 | 0 | 0 | 0 | 1 |
| 蒙特卡洛回报 $G_t$ | 0.6561 | 0.729 | 0.81 | 0.9 | 1 |

现在，不只是最后的 "5"，前面的 token 也会因为最终答案正确而得到正梯度。越靠近最终奖励的 token，得到的回报越大。

这个计算可以从序列末尾递推实现，因为 $G_t = r_t + \gamma G_{t+1}$：

```python
rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
gamma = 0.9

returns = torch.zeros_like(rewards)
future_return = 0.0

# G_t = r_t + gamma * G_{t+1}
for t in reversed(range(len(rewards))):
    future_return = rewards[t] + gamma * future_return
    returns[t] = future_return

# tensor([0.6561, 0.7290, 0.8100, 0.9000, 1.0000])
```

计算出 $G_t$ 后，再把它作为每个 token 交叉熵的权重：

```python
# log_probs[t] 是模型实际生成第 t 个 token 时的 log probability
loss = -(returns * log_probs).mean()
```

**拒绝采样 SFT 可以看作 $\gamma=1$ 的特殊情况**

语言模型训练中经常只在回答结束后得到一个最终奖励 $R$。此时中间奖励都是 0。如果再令 $\gamma = 1$，那么最终奖励会被完整地分配给回答中的所有 token：

- 正确回答得到 $R=1$：所有 token 的 $G_t=1$，loss 与普通 SFT 相同；
- 错误回答得到 $R=0$：所有 token 的 $G_t=0$，整条回答不产生梯度。

因此，当奖励只有 0 和 1，并采用相同的 loss 聚合方式时，这种 policy-gradient 训练在形式上就是**拒绝采样 SFT**：保留得分为 1 的模型回答作为 SFT 数据，丢弃得分为 0 的回答。

如果允许错误回答得到负奖励（例如 $R=-1$），policy gradient 还会主动降低错误回答中这些 token 的生成概率；普通拒绝采样 SFT 只能丢弃错误回答，不能产生这种反向梯度。

对应的批量 loss 实现：

```python
# log_probs: 当前模型生成的回答 token 的 log probability，[B, T]
# rewards: 每个完整回答的最终奖励 R，[B]
# response_mask: 回答 token 为 1，prompt 和 padding 为 0

# 只有最终奖励且 gamma=1，因此每个回答的所有 token 都有 G_t=R
returns = rewards[:, None]
loss_per_token = -returns * log_probs
loss = (loss_per_token * response_mask).sum() / response_mask.sum().clamp_min(1)
```

**实验示例**

本文将上述方法称为**基础 REINFORCE**。后续许多 policy-gradient 算法都保留了"用回报加权 $\log \pi_\theta$"这一核心形式，并在此基础上加入降低方差、限制更新幅度或提高数据利用率等机制。尽管基础 REINFORCE 很简单，但在奖励明确、可以自动验证的任务上，它本身的效果也很不错。

实际例子：使用 Qwen3.5-0.8B 在 GSM8K 上进行训练。模型在线生成解题过程，规则验证器根据最终答案是否正确给出 0 或 1 的奖励，令 $\gamma=1$，然后使用基础 REINFORCE 更新模型，训练集和测试集分数均稳步提升。

**policy gradient 的数学推导（扩展阅读）**

给定 prompt $x$，当前模型生成回答 $y$。目标是让模型生成回答的期望奖励最大：

$$
J(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}\big[R(y)\big]
$$

奖励 $R(y)$ 往往不可导（例如规则验证器只返回 0 或 1）。但使用 log-derivative trick，可以把梯度写成：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}\Big[R(y) \nabla_\theta \log \pi_\theta(y|x)\Big]
$$

当只有最终奖励且 $\gamma=1$ 时，每个 token 的蒙特卡洛回报都有 $G_t = R(y)$，因此：

$$
\nabla_\theta J(\theta) \approx \frac{1}{T}\sum_{t=1}^T R(y) \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

梯度上升最大化 $J(\theta)$，等价于梯度下降最小化 $\mathcal{L}_{PG} = -\frac{1}{T}\sum_t G_t \log \pi_\theta(a_t|s_t)$。

#### 4.3.2 PPO：在 policy gradient 上加入稳定训练机制

> 参见：[GRPO与PPO算法详解 · 二、各算法 Loss 完整公式](../深度学习/GRPO与PPO算法详解.md#二各算法-loss-完整公式)、[五、PPO 的 Advantage：V(s)、δ 和 GAE](../深度学习/GRPO与PPO算法详解.md#五ppo-的-advantagevsδ-和-gae)（另一套 PPO Loss 推导与 GAE 数值例子）。

最简单的 REINFORCE 虽然公式漂亮，但实际训练有三个主要问题：采样回报方差大、数据只能使用一次、一次更新可能让 policy 改变过猛。PPO 在它周围加入一组稳定化机制：

| PPO 组件 | 要解决的问题 |
|---------|-------------|
| learned critic | 为每个状态估计 baseline，降低梯度方差 |
| GAE | 在 advantage 的偏差与方差之间折中 |
| old policy + importance ratio | 允许 rollout 数据被有限复用 |
| clipped surrogate | 限制单次策略变化，避免更新过猛 |
| value loss | 训练 critic |
| entropy bonus | 防止策略过早失去探索能力 |
| reference KL（LLM 常用扩展） | 防止模型偏离初始 SFT policy 太远 |
| advantage normalization、gradient clipping | 改善数值尺度与优化稳定性 |

**Critic：学习一个 baseline**

PPO 额外训练价值模型 $V_\phi(s_t)$，估计"从当前状态继续生成，预期能获得多少回报"。一步 TD error 为：

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

如果实际结果比 critic 的预期更好，advantage 为正；比预期更差，advantage 为负。critic 通常最小化 value loss：

$$
\mathcal{L}_{value} = \mathbb{E}\left[(V_\phi(s_t) - G_t)^2\right]
$$

LLM PPO 因此通常至少要维护 actor、critic 和 frozen reference model。critic 减少了方差，也增加了显存、计算量和联合训练难度。

**GAE：计算 token/step advantage**

Generalized Advantage Estimation（GAE）把未来多个 TD error 加权累加：

$$
A_t^{GAE(\gamma,\lambda)} = \sum_{k=0}^{\infty} (\gamma\lambda)^k \delta_{t+k}
$$

- $\lambda$ 较小：更依赖 critic 的 bootstrap，方差较低、偏差可能较大；
- $\lambda$ 接近 1：更接近 Monte Carlo return，偏差较小、方差可能较大；
- 在只有终局奖励的 LLM 任务中，中间 $r_t$ 常为 0，最终 token 才得到任务 reward。

实践中还常对一个 batch 的 advantage 做标准化（这是 advantage normalization，不是回答长度归一化）：

$$
\hat{A}_t = \frac{A_t - \text{mean}(A)}{\text{std}(A) + \epsilon}
$$

**Old policy 与 importance ratio**

最纯粹的 REINFORCE 使用当前策略 $\pi_\theta$ 采样，并立即更新一次。为了提高数据利用率，PPO 系算法通常先冻结一份行为策略 $\pi_{\theta_{old}}$：用它生成 rollout，然后让新策略 $\pi_\theta$ 在这批数据上更新若干次。由于样本来自 $\pi_{\theta_{old}}$ 而不是正在变化的 $\pi_\theta$，需要 importance ratio 校正：

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}
$$

rollout 刚结束、参数还未更新时通常有 $r_t \approx 1$。这不代表梯度为 0，因为 $\pi_\theta$ 仍然通过 $r_t(\theta)$ 对参数求导。

当 rollout 太旧或同一批数据被复用太多轮时，$\pi_\theta$ 与 $\pi_{\theta_{old}}$ 差距会变大，importance sampling 方差也会快速增大。因此 PPO/GRPO 虽然允许有限的数据复用，本质上仍属于**近似 on-policy** 方法。

token-level 方法对每个 token 分别计算 $r_t$；GSPO 一类方法则先把整条回答压缩为 sequence-level ratio。前者更新粒度细，但同一回答中的不同 token 可能被不同方式裁剪；后者把回答作为一个整体进行一致的放大或抑制。

**Clip：限制一次策略更新**

如果直接最大化 $r_t(\theta) A_t$，一次更新可能把策略推得太远。PPO 使用 clipped surrogate：

$$
\mathcal{L}_{CLIP} = \mathbb{E}_t\Big[\min\big(r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\big)\Big]
$$

注意 min 与 advantage 的正负会共同决定裁剪方向：

- $A_t > 0$：不允许好动作的概率被一次提高太多；
- $A_t < 0$：不允许坏动作的概率被一次降低太多。

训练代码通常最小化负目标：

```python
ratio = torch.exp(new_logps - old_logps)
ratio_clip = ratio.clamp(1 - epsilon, 1 + epsilon)
surrogate_1 = ratio * advantages[:, None]
surrogate_2 = ratio_clip * advantages[:, None]
policy_loss = -torch.minimum(surrogate_1, surrogate_2)
```

clipping 不是简单地把 ratio 永远裁剪后再相乘，而是取 unclipped surrogate 和 clipped surrogate 中更保守的一个。它主要阻止"朝有利方向走得过远"，不会替代全部 trust-region 约束。

**KL、entropy 与 PPO 完整目标**

在 LLM 后训练中，经常再冻结一份 SFT reference policy $\pi_{ref}$，惩罚当前 policy 偏离它太远：

$$
\text{KL}(\pi_\theta \| \pi_{ref}) = \mathbb{E}\left[\log\frac{\pi_\theta(a_t|s_t)}{\pi_{ref}(a_t|s_t)}\right]
$$

熵奖励鼓励 policy 保留一定探索性：

$$
\mathcal{H}(\pi_\theta) = -\mathbb{E}\big[\log \pi_\theta(a_t|s_t)\big]
$$

把各项放在一起，一个常见的最小化目标为：

$$
\mathcal{L} = -\mathcal{L}_{CLIP} + c_1 \mathcal{L}_{value} - c_2 \mathcal{H}(\pi_\theta) + \beta\, \text{KL}(\pi_\theta \| \pi_{ref}) + \mathcal{L}_{aux}
$$

其中 $\mathcal{L}_{aux}$ 可以是 MoE load-balancing loss 等辅助项。不同 PPO 实现不一定同时使用 entropy 和 reference KL，KL 也可能先进入 reward/return 再计算 advantage，而不是作为显式 loss。

> 关于 PPO/GRPO 公式的另一套完整对照（含 SFT/GRPO/PPO 三者变量对比表）见 [GRPO与PPO算法详解](../深度学习/GRPO与PPO算法详解.md)。

一次 PPO 迭代可以概括为：

```python
# 1. 用 old policy 采样 rollout
rollouts = sample(policy_old, prompts)

# 2. 计算 reward、value、return 和 GAE advantage
rewards = reward_fn(rollouts)
advantages, returns = compute_gae(rewards, critic)

# 3. 在同一批 rollout 上做有限轮 minibatch 更新
for _ in range(ppo_epochs):
    ratio = (policy.logp(rollouts) - old_logp).exp()
    policy_loss = clipped_policy_loss(ratio, advantages)
    value_loss = mse(critic(rollouts.states), returns)
    loss = policy_loss + value_coef * value_loss + kl_coef * ref_kl
    update(loss)

# 4. 下一轮 rollout 前刷新 old policy
policy_old.load_state_dict(policy.state_dict())
```

#### 4.3.3 DPO：跳过显式奖励模型和在线 rollout

> 参见：[对齐 · LLM对齐](../深度学习/对齐.md#llm对齐)（SFT → 奖励模型 → RLHF 的传统三段式流程，DPO 正是对后两步的简化）。

前面的 REINFORCE / PPO / GRPO 都是 **on-policy** 方法：需要让当前模型在线生成回答（rollout），再用奖励函数或奖励模型打分。这带来两个工程成本：

1. 需要一个独立训练好的 Reward Model（或规则验证器）在训练循环中实时打分；
2. 需要维护 rollout 引擎，不断用最新策略生成新样本，训练与推理耦合。

**DPO**（Direct Preference Optimization，直接偏好优化）绕开了这两点：它不需要显式奖励模型，也不需要在线 rollout，直接在一份离线的**偏好数据**（pairwise preference data）上做**监督学习**，因此属于 **off-policy** 方法。

**数据形式**

DPO 的训练数据是三元组 $(x, y_w, y_l)$：给定 prompt $x$，$y_w$（winner）是标注为更优的回答，$y_l$（loser）是标注为更差的回答：

```json
{
  "prompt": "如何看待熬夜？",
  "chosen": "长期熬夜会打乱昼夜节律，建议尽量保证规律作息……",
  "rejected": "熬夜没什么大不了的，年轻人熬一下没关系。"
}
```

这类偏好对可以来自人工标注，也可以由更强模型（或规则/RM 打分）自动构造，不需要在训练过程中反复调用模型生成。

**从 RLHF 目标推导 DPO Loss**

标准 RLHF 的优化目标是在 KL 约束下最大化奖励模型给出的期望奖励：

$$
\max_{\pi_\theta} \ \mathbb{E}_{x,\,y\sim\pi_\theta}\big[r(x,y)\big] - \beta\, \text{KL}\big(\pi_\theta(y|x) \,\|\, \pi_{ref}(y|x)\big)
$$

这个目标存在闭式解，最优策略满足：

$$
\pi^*(y|x) = \frac{1}{Z(x)}\, \pi_{ref}(y|x)\, \exp\!\left(\frac{1}{\beta} r(x,y)\right)
$$

反解出隐式奖励 $r(x,y) = \beta \log \dfrac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$，代入 Bradley-Terry 偏好模型（$y_w$ 优于 $y_l$ 的概率由两者奖励差的 sigmoid 给出，配分函数 $Z(x)$ 在做差时相互抵消），得到只用策略模型本身表示的 DPO Loss：

$$
\mathcal{L}_{DPO}(\theta) = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)
$$

**这个公式在做什么**

- 两个 $\log$ 比值分别衡量策略相对参考模型，在 chosen 回答和 rejected 回答上"变得更喜欢还是更不喜欢"；
- 训练目标是拉大这两者的差距：让 $\pi_\theta$ 相对 $\pi_{ref}$ 更偏好 $y_w$、更不偏好 $y_l$；
- $\beta$ 控制约束强度，等价于 RLHF 目标里的 KL 系数：$\beta$ 越大，策略被约束得越接近参考模型。

**代码实现**

```python
def dpo_loss(policy, ref_model, chosen_ids, rejected_ids, beta=0.1):
    # 分别计算 policy 和 ref model 对 chosen / rejected 回答的 token 级 log-prob 之和
    policy_chosen_logps = sequence_logps(policy, chosen_ids)
    policy_rejected_logps = sequence_logps(policy, rejected_ids)
    with torch.no_grad():
        ref_chosen_logps = sequence_logps(ref_model, chosen_ids)
        ref_rejected_logps = sequence_logps(ref_model, rejected_ids)

    # 隐式奖励：策略相对参考模型的 log-ratio
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps)

    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    return loss
```

`sequence_logps` 与 SFT 中计算 loss 的前向过程几乎一致，区别只是取 `log_softmax` 后按 `assistant` 部分的 token 求和而不是取负平均；这里不需要 `ignore_index=-100` 之外的特殊处理，因为 DPO 只需要一个标量的序列级 log-prob。

**DPO 与 PPO/GRPO 的对比**

| 维度 | PPO / GRPO（on-policy） | DPO（off-policy） |
|------|--------------------------|---------------------|
| 数据来源 | 训练时用当前策略在线 rollout | 离线预先收集的偏好对 $(x, y_w, y_l)$ |
| 是否需要奖励模型 | 需要（或规则验证器） | 不需要，奖励被隐式吸收进 policy/ref 的 log-ratio |
| 需要维护的模型 | actor（+ PPO 的 critic）+ reference | policy + reference（无需 critic，无需单独 RM） |
| 训练稳定性 | 需要 clip、KL 等多种稳定化机制 | 本质是二分类交叉熵，训练更稳定、超参更少 |
| 能否用于多轮工具调用等 Agentic 任务 | 天然支持在线多轮 rollout（如 4.3.5 节） | 较难：偏好对通常针对单轮最终回答，缺少中间过程的 in-the-loop 反馈 |
| 样本利用率 | 每次 rollout 的样本通常只用有限几轮就丢弃 | 静态数据集可反复多 epoch 训练，类似普通 SFT |

一种常见的实践路径是：**SFT → DPO 快速对齐一版偏好 → 如果需要更强的可验证任务能力（数学、代码、工具调用），再叠加 PPO/GRPO 等 on-policy RL**。DPO 用更低的工程成本解决"整体风格与人类偏好对齐"的问题，PPO/GRPO 则更适合处理有明确 reward 信号、需要模型自己探索的任务。

#### 4.3.4 从统一的视角看 policy-gradient 算法

> 参见：[GRPO与PPO算法详解 · 六、GRPO 为什么能省掉 Critic 网络？](../深度学习/GRPO与PPO算法详解.md#六grpo-为什么能省掉-critic-网络)、[七、KL 散度惩罚项详解](../深度学习/GRPO与PPO算法详解.md#七kl-散度惩罚项详解)。

从 REINFORCE 到 PPO，再到 GRPO、DAPO、GSPO、CISPO，各算法都可以放回同一条处理链：

给定 prompt $x$，行为策略 $\pi_{\theta_{old}}$ 生成 rollout；算法先构造 advantage $\hat{A}$，再统一写成 clip/ratio 加权的 loss。可以拆解成六个可独立替换的组件：

| 组件 | 核心问题 | 可能的设计 |
|------|---------|-----------|
| 1. Rollout 组织与数据复用 | 用哪些样本构造一次梯度？ | 每个 prompt 采样数；独立/group/greedy/branch rollout；更新轮数；无效组过滤 |
| 2. Baseline、critic 与 advantage | 怎样判断回答或动作是否优于预期？ | baseline 来源 / temporal estimator / advantage 粒度 / normalization |
| 3. 概率比与梯度粒度 | 新旧策略概率怎样进入梯度？ | 无 ratio、token ratio、sequence ratio；response 还是 token action |
| 4. Proximal 更新机制 | 怎样限制一次策略更新的幅度？ | 无约束、PPO clip、非对称 clip、IS weight clip、sequence clip、soft gate、divergence penalty |
| 5. Loss 聚合与样本加权 | 每个 token、回答、prompt 占多大权重？ | per-sequence/per-token reduction；长度除数；mask/filter；正负 advantage 权重 |
| 6. 正则化与辅助目标 | 主策略梯度之外还优化什么？ | reference KL、entropy、value loss、self-imitation、positive LM loss、MoE aux loss |

这六个位置分别对应 rollout、advantage estimator、ratio、proximal mechanism、reduction/weight 和正则项。**以后看到新的"XXPO"，先判断它改了哪一个组件，通常比背算法名字更有效。**

**典型 on-policy policy-gradient 算法对比**

下表关注"奖励如何变成策略更新"。同一算法在不同代码库中可能有实现差异，表中记录论文的主要识别特征，而不是所有工程 recipe。

| 算法 | Rollout 组织 | Advantage estimator | Ratio / 梯度粒度 | Proximal 机制 | Loss 聚合 / 加权 | 正则与辅助目标 |
|------|-------------|---------------------|------------------|----------------|-------------------|-----------------|
| PPO | $G$ rollout；常多轮 minibatch | learned critic / GAE / token | token ratio | 对称 PPO min-clip | timestep/token mean | value loss、entropy；LLM 中常加 ref-KL |
| REINFORCE | 新鲜 rollout；通常一次更新 | MC return；可减 batch baseline | 无 ratio；response log-prob | 无 | 常按 response 聚合 | 可选 ref-KL |
| RLOO | 每个 prompt 生成 $k$ 个回答 | leave-one-out / MC / response | REINFORCE-style | 通常无 PPO clip | per-response | 可选 ref-KL |
| ReMax | sampled + greedy rollout | sampled reward − greedy reward | REINFORCE-style | 无 | per-response | 通常保留 ref-KL |
| GRPO | 每个 prompt 一组回答；可有限复用 | group mean / MC / response→token / group std | token ratio | 对称 PPO clip | 原始形式先对每条 response 内 token 求均值 | 显式 ref-KL |
| Dr. GRPO | 与 GRPO 相近 | group mean / MC / response→token / 无 group std | token ratio | PPO clip | 删除实际响应长度除数，使用固定长度归一化 | KL 不是核心改动 |
| REINFORCE++ | online rollout | 无 critic；batch/global advantage normalization | PPO-style token ratio | PPO-style clipping | token-level | KL 与多种稳定化技巧 |
| VC-PPO | PPO rollout | critic 预训练；actor/critic 解耦 GAE | token ratio | PPO clip | PPO-style | value loss |
| VinePPO | 从中间推理状态分支 rollout | 分支 MC state value / step advantage | token ratio | PPO clip | token-level | 可选 ref-KL；不依赖 learned critic |
| DAPO | group rollout；动态过滤全对/全错组 | group-relative / MC / response→token | token ratio | 非对称 Clip-Higher | 全局 token-level reduction | 删除 ref-KL |
| VAPO | group sampling | 预训练 critic；decoupled、length-adaptive GAE | token ratio | Clip-Higher | token-level | value loss、positive-example LM/self-imitation |
| GSPO | group rollout | group-relative response advantage | sequence likelihood ratio | sequence-level clipping | sequence-coherent | 与主改动正交 |
| CISPO | group rollout | 可继承 GRPO advantage | detached clipped ratio 作为 token 权重 | 裁剪 IS weight，而非 PPO surrogate | token-level | 可配置 |
| SAPO | group rollout | group-relative | sequence-coherent、token-adaptive weighting | 温度控制 soft gate | token-adaptive | 可配置 |
| CFPO | group rollout | group-relative | 以 divergence 构造更新 | quadratic penalty 替代 clipping | 依具体实现 | 可配置 |
| RGRA | group rollout | group-relative | 删除 importance ratio | 无 ratio / clipping | 依具体实现 | 可配置 |

#### 4.3.5 Agentic RL 实践

> 参见：[verl · 六、GRPO 训练核心指标详解](verl.md#六grpo-训练核心指标详解)（工业框架中同样的 rollout/advantage/ratio 指标在真实训练日志中长什么样）。

多轮工具调用、延迟奖励、group-relative advantage、GRPO loss 和 rollout engine——Agentic RL 优化的不是单轮字符串，而是**完整交互轨迹**。

当前实现可归纳为：

| 六组件 | `train_agent.py` 的选择 |
|--------|--------------------------|
| Rollout | 当前 policy 同步采样；每个 prompt 默认生成 4 条；每条最多 3 轮工具交互 |
| Advantage | group mean / Monte Carlo terminal reward / response→action-token broadcast / group std |
| Ratio | 保存 rollout 时的 old token log-prob，训练时计算 token ratio |
| Proximal | 默认 CISPO；参数可切换为 GRPO min-clip |
| Reduction | 每条轨迹先按有效 action token 求均值，再对有效轨迹求均值 |
| 正则/辅助 | frozen reference model 的 token KL |

**数据集格式**

Agentic RL 数据不是提供一条标准 assistant 回答让模型模仿，而是提供初始状态、可用工具和最终校验目标。最后一个 assistant 留空，等待 policy 在 rollout 阶段自己生成：

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "你可以使用数学计算工具。",
      "tools": "[{\"type\":\"function\",\"function\":{\"name\":\"calculate_math\",\"parameters\":{...}}}]"
    },
    {"role": "user", "content": "计算 94-35，并给出最终结果。"},
    {"role": "assistant", "content": ""}
  ],
  "gt": ["59"]
}
```

`AgentRLDataset` 会返回：

```python
{
    "messages": conversations[:-1],  # rollout 的初始上下文
    "tools": tools,                  # 当前样本允许使用的工具
    "gt": sample["gt"],              # 轨迹结束后的校验目标
}
```

这里的 `gt` 只用于环境验证和 reward 计算，不会像 SFT label 一样直接告诉模型下一步应该生成哪个 token。

**Rollout：模型 action 与环境 observation 分开**

`rollout_single` 的核心循环可以简化为：

```python
for turn in range(max_turns):
    # 1. 根据当前消息、工具定义构造上下文
    context = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True
    )
    inputs = tokenizer(
        context, return_tensors="pt", add_special_tokens=False
    ).to(device)

    # 2. policy 生成 action，并保存行为策略的逐 token log-prob
    result = rollout_engine.rollout(
        prompt_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        num_generations=1,
    )
    new_ids = result.completion_ids[0].tolist()
    new_logps = result.per_token_logps[0].tolist()
    response_ids.extend(new_ids)
    response_old_logps.extend(new_logps)
    response_mask.extend([1] * len(new_ids))

    # 3. 如果 action 中包含工具调用，执行工具
    calls = parse_tool_calls(result.completions[0])
    if not calls:
        break
    observation = execute_tool(calls[0]["name"], calls[0]["arguments"])

    # 4. observation 拼回状态，但它不是 policy action，不参与策略损失
    obs_ids = tokenizer(observation).input_ids
    response_ids.extend(obs_ids)
    response_old_logps.extend([0.0] * len(obs_ids))
    response_mask.extend([0] * len(obs_ids))
```

真实代码还处理了 chat template、多工具调用、EOS、截断和未完成轨迹。最值得讲解的是 `response_mask`：**policy 自己生成的 token 标 1，环境返回的 observation 标 0**。如果忘记屏蔽 observation，训练就会错误地要求模型"预测环境返回值"，策略梯度的语义也会被破坏。

**奖励**

在整条轨迹结束后计算一次 reward。当前代码综合了答案命中、工具调用合法性、格式、是否完成、重复惩罚和可选 Reward Model 分数，并将总分裁剪到 $[-1, 1]$。

为了保持本节边界，后续统一把这段实现看成：

```python
rewards = calculate_rewards(trajectories)  # [batch_size * num_generations]
```

即只使用它的输出 $R$，不把 reward shaping 与 policy-gradient 算法混在一起。

**GRPO advantage：同一道题内部做相对比较**

MiniMind 对每个 prompt 的 $G$ 条轨迹分组：

```python
grouped_rewards = rewards.view(-1, num_generations)
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(num_generations)
std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(num_generations)
advantages = (rewards - mean_r) / (std_r + 1e-4)
```

对应公式：

$$
\hat{A}_g = \frac{R_g - \text{mean}(R_{1..G})}{\text{std}(R_{1..G}) + \epsilon}
$$

这是 **response-level advantage**，随后广播给轨迹中所有 action token：$A_t = \hat{A}_g,\ \forall t \in \text{action tokens of trajectory } g$。

它不提供细粒度 credit assignment：同一轨迹中的正确工具选择、冗余推理和最终回答共享同一个 advantage。它的优点是无需 critic，缺点是当同组奖励相同（全对、全错或打分器分辨率不足）时，$\text{std}(R) \approx 0$，这一组几乎没有策略梯度。

**Token log-prob、importance ratio 与 KL**

模型前向后取出实际 token 的 log-prob：

```python
logits = model(input_ids).logits[:, :-1]
new_logps = F.log_softmax(logits, dim=-1).gather(
    2, input_ids[:, 1:].unsqueeze(-1)
).squeeze(-1)

ratio = torch.exp(new_logps - old_logps)
```

reference model 不参与训练，只用于限制 policy 偏离初始 SFT 模型。MiniMind 使用的逐 token 非负 KL estimator 为：

```python
log_ratio_ref = ref_logps - new_logps
per_token_kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1
```

即：

$$
\text{KL}_t = e^{\log\pi_{ref} - \log\pi_\theta} - (\log\pi_{ref} - \log\pi_\theta) - 1 \ge 0
$$

**聚合损失**

最终聚合为：

```python
token_counts = completion_mask.sum(dim=1)
per_response_loss = (
    (per_token_loss * completion_mask).sum(dim=1)
    / token_counts.clamp_min(1)
)
policy_loss = per_response_loss[valid_rows].mean()
```

对应：

$$
\mathcal{L} = \frac{1}{|\{\text{valid}\}|}\sum_{g \in \text{valid}} \frac{1}{T_g}\sum_{t=1}^{T_g} \ell_{g,t}
$$

因此每条有效轨迹总体权重近似相等，长回答中的单个 token 权重更小。这是 **per-sequence reduction**。

**完整训练 step**

```python
for prompts in dataloader:
    # rollout policy = old policy
    trajectories, old_logps, action_mask = rollout(prompts, num_generations=G)

    # 外部给定的 trajectory reward
    rewards = reward_fn(trajectories)

    # group-relative response advantage
    advantages = group_zscore(rewards, group_size=G)

    # evaluate the sampled actions under current policy and reference policy
    new_logps = policy.log_probs(trajectories)
    ref_logps = reference.log_probs(trajectories)
    ratio = (new_logps - old_logps).exp()
    kl = token_kl(new_logps, ref_logps)

    # choose one proximal surrogate
    token_loss = grpo_loss(ratio, advantages, kl)       # --loss_type grpo
    # token_loss = cispo_loss(ratio, new_logps, advantages, kl)  # default

    # only policy-generated action tokens contribute gradients
    loss = per_sequence_reduce(token_loss, action_mask)
    loss.backward()
    optimizer.step()

    # rollout engine 若与训练模型分离，需要定期同步新权重
    rollout_engine.update_policy(policy)
```

#### 4.3.6 Reward 曲线与 Case 分析

下表对比同一批提示词在 pretrain / SFT / RL 三个阶段之后的模型输出，可以直观看到强化学习阶段带来的变化：

| 提示词 | pretrain 后 | sft 后 | rl 后 |
|--------|------------|--------|-------|
| 为什么天空是蓝色的 | 天空之所以呈现蓝色，是因为大气中的氧气和氮气分子对蓝色光的散射……当太阳光射向地球时，其中的蓝色光波长被散射得更多……因此，天空呈现出蓝色。 | 天空是蓝色的原因是因为太阳光在穿过大气层时，其中的分子和颗粒会散射光线，而蓝光波长较短……所以我们看到的天空呈现蓝色。 | 天空呈现蓝色是由于光的散射效应。太阳光中包含了所有颜色的光，其中蓝光波长较短，因此更容易被大气中的气体和微粒散射……这种现象称为瑞利散射。 |
| 请用 Python 写一个计算斐波那契数列的函数 | 输出存在递归逻辑重复、代码块之间界限模糊等问题（生成质量较低，代码可读性差） | 能给出结构完整的迭代版本，并配有注释和调用示例，但内部实现细节仍有轻微冗余（如循环体内出现两次赋值） | 能给出结构清晰、带文档字符串（docstring）的函数实现，变量命名更规范，示例调用与预期输出一致 |
| 帮我算一下 256 乘以 37 等于多少（可用工具: `calculate_math`） | 直接文本猜测答案，且结果错误（如"等于 7731"），没有调用工具的意识 | 会输出 `<tool_call>` 调用工具，但可能重复调用同一工具两次才给出最终答案 | 一次调用 `calculate_math` 即得到正确结果 9472 并直接作答，减少了冗余的重复工具调用 |
| 北京今天天气怎么样？（可用工具: `get_current_weather`） | 声明"AI 无法提供实时信息"后仍尝试直接编造天气描述，逻辑前后矛盾 | 会调用 `get_current_weather`，但可能重复调用，且复述结果时出现细节错误（如编造风速数值） | 调用一次工具后即准确复述返回的城市、温度、湿度、天气状况 |
| 查一下美元兑人民币汇率是多少（可用工具: 多个） | 直接声明无法提供实时金融数据，不会调用工具 | 会调用相关工具，但存在不必要的重复调用（如两次调用 `get_exchange_rate` 或误调用 `get_current_time`） | 只调用必要的 `get_exchange_rate` 一次，并准确给出汇率数值 |
| 北京今天天气怎么样？另外查一下美元兑人民币汇率。（多工具组合任务） | 无法正确编排两个子任务，倾向于用文本臆测替代真实调用 | 能依次调用 `get_current_weather` 和 `get_exchange_rate`，但可能出现工具调用顺序混乱、重复调用、结果混杂（如把降雨信息和晴天描述放在一起）等问题 | 能按顺序正确调用两个工具，并将两次工具结果分别、准确地整合进最终回答中 |

> 总体上可以看到：**预训练**阶段模型具备语言流畅度和基础世界知识，但不会良好地组织对话结构、也不会主动使用工具；**SFT** 阶段模型学会了对话格式和"看到工具就尝试调用"的基本模式，但工具调用常有重复或冗余；**RL** 阶段（GRPO/CISPO 等 group-relative policy gradient）在明确的 reward 信号（答案是否正确、工具调用是否合法、格式是否规范等）引导下，进一步减少了重复调用、提升了多工具组合任务的编排能力和最终答案的准确率。

---

### 4.4 数据处理：从原始语料到训练样本

前面 4.1～4.3 节的"数据集"小节只展示了最终喂给模型的 JSONL 格式和统计数字，本节补齐中间被跳过的部分：数据从哪里来、怎么清洗、以及三个阶段的数据如何逐步演化。

#### 4.4.1 数据来源

MiniMind 三阶段数据的来源和用途差异很大：

| 阶段 | 主要来源 | 特点 |
|------|---------|------|
| 预训练 | 网页百科（如中文维基）、新闻资讯、开源书籍、代码仓库等通用语料聚合数据集 | 规模大（846 万条/2.2B token）、无标注、质量参差不齐 |
| SFT | 开源指令数据集（如 Alpaca、BelleGroup 系列的中文改写版）、人工/模型合成对话、工具调用轨迹 | 规模中等（511 万条）、需要结构化为 `role/content` 多轮格式 |
| RL（Agentic） | 数学/代码等可自动验证任务的题库（如 GSM8K 类型问题）、自定义工具调用场景 | 规模较小，但每条样本都需要一个可编程验证的 `gt`（ground truth） |

三个阶段的数据在"规模"和"标注精度"之间呈反向关系：预训练数据量最大但几乎不需要标注，RL 数据量最小但标注/验证要求最高。

#### 4.4.2 数据清洗

原始语料直接喂给 tokenizer 训练或模型训练之前，通常需要经过几类基础清洗步骤：

| 清洗步骤 | 目的 | 常见做法 |
|---------|------|---------|
| 去重 | 避免高频重复片段让模型死记硬背、浪费训练算力 | 精确哈希去重 + MinHash/SimHash 近似去重 |
| 长度过滤 | 剔除过短（信息量不足）或异常过长（可能是乱码/表格转储）的样本 | 按字符数或 token 数设置上下限阈值 |
| 质量过滤 | 剔除乱码、广告、机器翻译痕迹明显、格式错乱的文本 | 规则过滤（特殊符号占比、中英文混杂度）+ 简单分类器打分 |
| 敏感内容过滤 | 减少违法违规、隐私泄露等内容进入训练集 | 关键词/正则规则 + 分类模型联合过滤 |
| 格式规整 | 统一编码、换行符、全角半角等 | 归一化脚本批量处理 |

SFT 阶段还需要额外的**对话质量清洗**：过滤掉回答过短、拒绝作答、前后矛盾、工具调用参数不合法的样本；工具调用类样本还需要校验 `tool_calls` 的 JSON 格式和参数是否能被真实执行。

#### 4.4.3 从预训练数据到 SFT 数据：格式演化

预训练阶段每条样本只有裸文本 `text` 字段（见 4.1.3），SFT 阶段需要把无结构文本包装成结构化多轮对话（见 4.2.3）。这一步通常由以下方式构造：

1. **人工/众包标注**：标注员直接撰写高质量的问答对；
2. **模型蒸馏**：用更强的模型（如 GPT-4 级别）根据种子问题生成回答，再做质量筛选；
3. **规则改写**：把百科条目、FAQ 等结构化知识改写成"提问-回答"的形式；
4. **工具调用轨迹合成**：为常见工具（计算器、天气、汇率查询等）编写模板问题，程序化拼出 `tool_calls`/`tool_response` 的完整轨迹，再人工或规则校验正确性。

#### 4.4.4 RL 数据的特殊要求：可验证性

强化学习阶段的数据与前两阶段有本质区别：预训练/SFT 数据只需要"看起来正确"，而 RL 数据的 `gt` 字段必须能被程序化验证（见 4.3.5 的 `AgentRLDataset` 格式）。这意味着 RL 数据的构造需要额外满足：

- **确定性**：同一个问题只能有一个（或一组等价的）标准答案，避免规则验证器给出模糊或矛盾的奖励；
- **可执行性**：如果任务涉及工具调用，需要提前准备好可以真实执行（或可靠 mock）的工具环境；
- **难度分层**：过于简单的题目会让一组 $G$ 条 rollout 全部答对（reward 方差为 0，GRPO advantage 退化，见 4.3.3 节），过难的题目则可能全部答错，两种情况都不能提供有效的策略梯度；因此实践中通常会对题库先做一次"难度探测"（用当前模型采样几次，统计正确率），保留正确率处于中等区间的题目。

> 参见：[对齐 · LLM对齐](../深度学习/对齐.md#llm对齐)（数据标注在 SFT/RM/RLHF 全流程中的位置）。

### 4.5 训练工程细节

前面各节的训练循环都简化为"forward → backward → step"，本节补充让训练真正跑得动、跑得稳的工程手段：混合精度、学习率调度和分布式训练。梯度裁剪、梯度累积的公式与代码见 [GRPO与PPO算法详解 · 八、梯度下降与反向传播](../深度学习/GRPO与PPO算法详解.md#八梯度下降与反向传播)，此处不再重复，只补充分布式与调度部分。

#### 4.5.1 混合精度训练

现代 GPU（如 A100/H100）对 fp16/bf16 张量核心的吞吐量远高于 fp32，混合精度训练用低精度做大部分计算、用 fp32 保留关键的数值累加，从而在几乎不损失精度的前提下大幅提速、节省显存。

| 精度 | 数值范围 | 常见问题 | 适用场景 |
|------|---------|---------|---------|
| fp32 | 范围大、精度高 | 显存占用大、计算慢 | 优化器状态、部分归约操作 |
| fp16 | 范围窄（易溢出/下溢） | 大模型训练中梯度容易 underflow | 需要配合 loss scaling |
| bf16 | 范围与 fp32 相同、尾数精度更低 | 几乎不需要 loss scaling | 当前大模型训练的主流选择 |

MiniMind 训练脚本中典型的 `autocast` 用法：

```python
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

with torch.cuda.amp.autocast(dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16):
    result = model(input_ids, labels=labels)
    loss = result.loss / accumulation_steps

# fp16 需要 loss scaling 防止梯度下溢；bf16 可以跳过 scaler，直接 backward
scaler.scale(loss).backward()

if (step + 1) % accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

`RMSNorm` 等对数值稳定性敏感的模块（见 3.5.1）通常会强制在 `.float()` 上计算再转换回原精度，这正是为了避免混合精度下均方根计算的累积误差。

#### 4.5.2 学习率调度：Warmup + Cosine Decay

学习率过大会导致训练初期损失震荡甚至发散，过小则收敛缓慢；warmup + cosine decay 是当前大模型训练的标准调度方式：

$$
\eta(t) =
\begin{cases}
\eta_{max} \cdot \dfrac{t}{T_{warmup}}, & t < T_{warmup} \\[6pt]
\eta_{min} + \dfrac{1}{2}(\eta_{max}-\eta_{min})\left(1+\cos\left(\pi \cdot \dfrac{t-T_{warmup}}{T_{total}-T_{warmup}}\right)\right), & t \ge T_{warmup}
\end{cases}
$$

- **Warmup 阶段**：学习率从 0 线性增长到 $\eta_{max}$，让模型参数（尤其是随机初始化的部分）在训练早期不被过大的梯度步长破坏；
- **Cosine Decay 阶段**：学习率按余弦曲线平滑衰减到 $\eta_{min}$，相比阶梯衰减，末期的过渡更平滑，通常有更好的最终收敛效果。

```python
import math

def lr_lambda(step, warmup_steps, total_steps, min_ratio=0.1):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_ratio + (1 - min_ratio) * cosine

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda step: lr_lambda(step, warmup_steps, total_steps)
)
```

> 参见：[verl · 6.5 Actor/LR（学习率）](verl.md#65-actorlr学习率)——该节额外说明了 GRPO 训练中为什么通常倾向使用固定学习率而非 cosine 衰减：GRPO 自身的 clip、KL 约束、reward 归一化已提供足够的稳定性，末期学习率过早衰减到 0 反而会让训练提前停滞。这与预训练/SFT 阶段"必须用 warmup+cosine"形成有意思的对比：**阶段的稳定性来源不同，调度策略也应不同。**

#### 4.5.3 分布式训练：DDP 与 ZeRO

> 本节只覆盖 MiniMind 训练脚本实际用到的 DDP/ZeRO 基础用法；更系统的分布式训练全景（DP/DDP/FSDP/TP/PP 的层次关系、ZeRO 三阶段显存精确计算、AllReduce/Reduce-Scatter 等通信原语、Pipeline Parallel 的 Bubble 问题、以及 Megatron/DeepSpeed/Accelerate/TRL/verl 等框架之间的关系）见 [分布式训练与推理加速全解](分布式训练与推理加速全解.md)。

单卡显存和算力有限，训练更大的模型或使用更大 batch size 需要多卡甚至多机协同。

**DDP（Distributed Data Parallel）**：每张卡都保存一份完整的模型副本，各自处理不同的数据分片，前向/反向传播独立进行，只在梯度计算完成后做一次 all-reduce 同步：

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

for input_ids, labels in loader:
    result = model(input_ids, labels=labels)
    loss = result.loss
    loss.backward()          # backward 内部自动完成梯度 all-reduce
    optimizer.step()
    optimizer.zero_grad()
```

DDP 的问题在于每张卡都要保存完整的模型参数、梯度和优化器状态（对 Adam 类优化器而言，优化器状态通常是参数量的 2 倍），显存开销随模型增大而线性增长，很快就会成为瓶颈。

**ZeRO（Zero Redundancy Optimizer）**：由 DeepSpeed 提出，核心思想是把参数、梯度、优化器状态分片存储在不同的卡上，而不是每张卡都保存完整副本：

| ZeRO 阶段 | 分片内容 | 显存节省 | 通信开销 |
|-----------|---------|---------|---------|
| ZeRO-1 | 仅分片优化器状态 | 中等 | 较低 |
| ZeRO-2 | 分片优化器状态 + 梯度 | 较大 | 中等 |
| ZeRO-3 | 分片优化器状态 + 梯度 + 模型参数 | 最大（可训练远超单卡显存的模型） | 最高（需要在前向/反向时临时聚合参数分片） |

使用 DeepSpeed 时，训练脚本本身几乎不变，只需提供配置文件：

```json
{
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 4,
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "allgather_bucket_size": 2e8,
    "reduce_bucket_size": 2e8
  }
}
```

```python
import deepspeed

model_engine, optimizer, loader, scheduler = deepspeed.initialize(
    model=model, optimizer=optimizer, training_data=dataset, config="ds_config.json"
)

for batch in loader:
    loss = model_engine(batch).loss
    model_engine.backward(loss)   # 梯度分片同步在内部自动处理
    model_engine.step()
```

**如何选择**：MiniMind 这种百 M 级别的小模型单卡即可容纳，用 DDP（甚至单卡）即可完成预训练；当模型规模增长到显存放不下完整的参数+梯度+优化器状态时，才需要引入 ZeRO-2/3 或更进一步的张量并行/流水线并行，张量并行的核心原理见下一节。

#### 4.5.4 张量并行：MLP 层如何切分到多卡

ZeRO 解决的是"参数、梯度、优化器状态"的显存分片问题，但每次前向/反向仍然是完整的矩阵计算。当单个矩阵本身大到一张卡都放不下（或者算力不够）时，就需要**张量并行（Tensor Parallelism）**：把一个大矩阵直接切开，分给多张卡各自计算一部分。

**目标**：切分之后各卡尽量独立计算，通信（跨卡同步）越少越好，因为通信是分布式训练里最慢的瓶颈。

##### 1）MLP 的标准结构

FFN/MLP 层的计算可以简化为两层线性变换夹一个非线性激活函数：

```
MLP 数据流：
X（输入） → A（第一层线性） → GELU（非线性激活） → B（第二层线性） → Y（输出）

公式：Y = GELU(X @ A) @ B
```

要把 `A`、`B` 这两个大矩阵切到多卡上，无非两种切法：**列切**（按输出维度切）或**行切**（按输入维度切）。关键问题是：先切的第一层 `A` 到底该列切还是行切？

##### 2）第一层 A 必须列切：因为 GELU 是非线性的

`GELU` 满足**先加再激活 ≠ 先激活再加**，即：

```
GELU(a1 + a2) ≠ GELU(a1) + GELU(a2)      非线性，不能拆开算
```

这决定了 `A` 只能列切，不能行切：

| 切法 | 每卡计算出的中间结果 | 能否直接过 GELU |
|------|---------------------|-----------------|
| ❌ 行切 A（按输入维度切，每卡拿一部分 X 的列） | 每卡只能算出对最终结果的「部分和」 | 不能，必须先 AllReduce 把部分和加起来，再过 GELU（多一次通信） |
| ✅ 列切 A（按输出维度切，每卡拿完整的 X，算出输出的一部分列） | 每卡得到的是「完整的列」，即最终结果里独立的几列 | 可以，GELU 是逐元素运算，每一列互不影响，各卡直接独立算，0 通信 |

```
所以 A 必须列切，才能让 GELU 在各卡上独立计算，不需要任何通信
```

##### 3）第二层 B 只能行切：正好接住 A 列切后的输出

`A` 列切之后，每张卡上 `GELU(X @ A_i)` 输出的是结果矩阵按列分块的一部分；要接住这些列分块，`B` 必须按对应的行切分，这样每张卡才能维度对齐做矩阵乘法：

```
GPU1: X → A1（列切）→ GELU → B1（行切）→ 部分积
GPU2: X → A2（列切）→ GELU → B2（行切）→ 部分积
                                  ↓
                    AllReduce 求和 → Y（最终输出）
```

`B` 行切后，每张卡算出来的只是最终输出 `Y` 的「部分和」（因为每张卡只贡献了完整求和链路里的一部分维度），所以最后必须做**一次 AllReduce** 把各卡的部分积加起来，才能得到正确的 `Y`。

##### 4）为什么是「先列切、后行切」而不是反过来

| 切分方案 | 通信次数 | 原因 |
|---------|---------|------|
| ✅ 先列切 A、后行切 B（标准做法） | **1 次 AllReduce**（只在最后求和） | GELU 前各卡独立、互不依赖，只需在两层线性变换算完后同步一次 |
| ❌ 先行切 A | **2 次通信**（GELU 前一次 AllReduce，末尾还要一次） | 行切 A 得到的是部分和，必须先同步才能过 GELU，之后 B 还要再同步一次，通信量翻倍 |

一句话总结：**关键在于 GELU 非线性不能作用在“部分和”上，只能作用在“完整的列”上，所以第一层必须列切，让非线性激活各卡独立算；第二层顺势行切接住列分块，最后只需一次 AllReduce 收尾。**

##### 5）Self-Attention 的张量并行同理

Attention 层的切分遵循同样的“先列切、后行切”套路：

```
Q/K/V 投影矩阵：按注意力头（head）列切 → 每张卡负责若干个头，头与头之间计算完全独立，0 通信
输出投影 W_O：行切 → 接住多头拼接后的列分块，最后一次 AllReduce 收尾
```

标准套路：**先列切、后行切**，全程只需 1 次 AllReduce，这也是 Megatron-LM 等张量并行框架里 MLP 和 Self-Attention 层的通用切分方式。

> 关联笔记：GELU 等激活函数的原理见 [激活函数和损失函数](../深度学习/激活函数和损失函数.md)；`W_Q/W_K/W_V/W_O` 等权重矩阵的形状定义见 [Transformer向量与参数详解](../深度学习/Transformer向量与参数详解.md)；Attention 层（QKV/输出投影）如何做张量并行切分、以及张量并行之外的其他并行策略（流水线并行、ZeRO 等），见 [分布式训练与推理加速全解](分布式训练与推理加速全解.md)。

### 4.6 推理

前面几节讲的都是"如何训练"，本节补上训练完成后"如何使用"：KV Cache 在真实推理中如何生效、如何从 logits 采样出下一个 token，以及如何实现流式输出。

#### 4.6.1 自回归生成与 KV Cache 的实际使用

3.2.3 节介绍了 KV Cache 的原理（缓存历史 Key/Value，避免重复计算），这里给出它在自回归生成循环中的完整使用方式：

```python
@torch.no_grad()
def generate_step_by_step(model, input_ids, max_new_tokens, eos_id):
    past_key_values = None
    generated = input_ids

    for _ in range(max_new_tokens):
        if past_key_values is None:
            # 第一步：完整地把 prompt 过一遍模型，同时建立 KV Cache
            outputs = model(generated, use_cache=True)
        else:
            # 之后每一步：只输入新生成的最后一个 token，
            # 历史信息全部来自 past_key_values，不需要重新计算
            outputs = model(generated[:, -1:], past_key_values=past_key_values, use_cache=True)

        logits = outputs.logits[:, -1, :]          # 只关心最后一个位置的下一个 token 分布
        past_key_values = outputs.past_key_values   # 更新缓存（见 3.2.5 中 present_key_value 的构造）

        next_token = sample_next_token(logits)       # 见 4.6.2 采样策略
        generated = torch.cat([generated, next_token], dim=1)

        if (next_token == eos_id).all():
            break

    return generated
```

关键点：**第一步（prefill）必须处理完整 prompt，之后每一步（decode）只需要处理一个新 token**。没有 KV Cache 时，decode 阶段每一步都要把已生成的全部序列重新过一遍模型，计算量随生成长度呈平方级增长；有了 KV Cache，每一步只需要计算新 token 与历史 Cache 的注意力，计算量随生成长度线性增长。

> 这里的 `model(...)` 内部实际执行 Attention 计算时，底层可以选择 `eager`（朴素实现）、`sdpa`（PyTorch 统一接口，自动选择最优 kernel）或 `flash_attention_2`（显式使用 FlashAttention）等不同实现方式，三者数值结果一致但速度/显存差异很大；生产级推理服务（如 vLLM）还会用 PagedAttention 管理多请求的 KV Cache 显存。详见 [分布式训练与推理加速全解 · 八、推理侧 Attention 实现方式](分布式训练与推理加速全解.md#八推理侧attention-的几种实现方式eager--sdpa--flashattention--pagedattention)。

#### 4.6.2 采样策略：从 logits 到下一个 token

模型每一步输出的是词表大小的 logits，如何从中选出下一个 token 直接影响生成质量与多样性：

| 策略 | 做法 | 效果 |
|------|------|------|
| Greedy（贪心） | 直接取概率最大的 token | 确定性输出，容易重复、缺乏多样性 |
| Temperature（温度） | 采样前将 logits 除以温度 $\tau$ 再 softmax | $\tau<1$ 让分布更尖锐（更保守），$\tau>1$ 让分布更平滑（更随机） |
| Top-k | 只在概率最高的 $k$ 个 token 中采样 | 截断长尾的低概率、不合理 token |
| Top-p（nucleus） | 只在累计概率达到 $p$ 的最小 token 集合中采样 | 比固定 $k$ 更自适应：分布尖锐时集合小，分布平坦时集合大 |
| Repetition penalty | 对已经出现过的 token 的 logits 做惩罚 | 减少复读机式的重复生成 |

```python
def sample_next_token(logits, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.1, generated_ids=None):
    # 1. 重复惩罚：降低已经出现过的 token 的 logits
    if generated_ids is not None:
        for token_id in set(generated_ids[0].tolist()):
            logits[0, token_id] /= repetition_penalty

    # 2. 温度缩放
    logits = logits / max(temperature, 1e-5)

    # 3. Top-k：只保留概率最高的 k 个 token，其余置为 -inf
    if top_k is not None:
        topk_values, _ = torch.topk(logits, top_k)
        logits[logits < topk_values[:, [-1]]] = float("-inf")

    # 4. Top-p：按概率降序累加，超过 p 之后的 token 置为 -inf
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = cumulative_probs - sorted_probs > top_p
    sorted_probs[sorted_mask] = 0.0
    probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)

    # 5. 按最终分布采样
    probs = probs / probs.sum(dim=-1, keepdim=True)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
```

`temperature → 0` 时采样退化为 greedy；RL 训练中的 rollout（见 4.3.5 节）通常使用较高的 temperature 以保证 $G$ 条轨迹之间有足够差异，而最终面向用户的推理服务则常用较低的 temperature 以提升稳定性。

#### 4.6.3 流式生成

面向用户的对话产品通常希望文字逐字/逐词吐出，而不是等待完整回答生成完毕才一次性返回。基于 4.6.1 的逐步生成循环，只需要在每一步产出新 token 后立即解码并 yield：

```python
def stream_generate(model, tokenizer, input_ids, max_new_tokens, eos_id):
    past_key_values = None
    generated = input_ids

    for _ in range(max_new_tokens):
        if past_key_values is None:
            outputs = model(generated, use_cache=True)
        else:
            outputs = model(generated[:, -1:], past_key_values=past_key_values, use_cache=True)

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        next_token = sample_next_token(logits, generated_ids=generated)
        generated = torch.cat([generated, next_token], dim=1)

        if (next_token == eos_id).all():
            break

        # 立即把新 token 解码成文本片段并产出，前端可以逐字渲染
        yield tokenizer.decode(next_token[0], skip_special_tokens=True)
```

工程上，流式生成通常还要处理：多字节 UTF-8 字符被单个 token 切断导致的乱码（需要缓冲到完整字符再输出）、多个并发请求的 batch 化调度（不同请求的生成长度不同，需要动态 batching），以及与 4.3.5 节 Agentic RL 中 rollout 引擎的关系——两者本质上是同一套自回归生成循环，区别只在于推理服务面向单次用户请求返回文本，而 RL rollout 需要额外记录每一步的 log-prob 用于后续计算 ratio 与 advantage。

---

## 总结

从 Tokenizer 训练、模型结构设计到预训练 / SFT / RL 三阶段训练，MiniMind 完整复现了现代大语言模型从随机初始化到具备文本生成、指令遵循和 Agent 能力的全过程。几个关键设计选择贯穿全文：

- **Tokenizer**：Byte-level BPE + Added/Special Token + Chat Template，让任意文本都能被稳定编码，并让对话结构、工具调用可以被模型理解。
- **模型结构**：GQA 平衡了 KV Cache 开销与表达能力，RoPE（及其 YaRN 等长度外推变体）注入相对位置信息，SwiGLU FFN 提供门控非线性，RMSNorm + PreNorm + 残差连接保证深层训练稳定。
- **数据处理**：预训练、SFT、RL 三阶段数据在"规模"与"标注精度"上此消彼长，分别对应通用语料清洗、结构化对话构造、可程序验证的 `gt` 设计三套不同流程。
- **训练方式**：预训练与 SFT 共享同一个 next-token 交叉熵框架，只是数据分布和 loss mask 不同；对齐阶段既可以走 **on-policy** 的强化学习路线（REINFORCE → PPO → GRPO/DAPO/GSPO/CISPO，统一视角是"回报加权的交叉熵"，差异在 rollout 组织、advantage 估计、ratio 粒度、proximal 机制、loss 聚合、正则化六个可替换组件），也可以走 **off-policy** 的 DPO 路线（在离线偏好对上做监督学习，跳过显式奖励模型和在线 rollout）；两条路线在实践中常常先 DPO 再 RL 组合使用。
- **训练工程**：混合精度、warmup+cosine 学习率调度、DDP/ZeRO 分布式训练，是让上述算法在真实硬件上跑得动、跑得稳的必要条件，其重要性并不亚于算法本身的设计。
- **推理部署**：训练好的模型依赖 KV Cache 把自回归生成的计算量从平方级降到线性级，再通过 temperature/top-k/top-p/repetition penalty 等采样策略和流式输出，才能变成一个可用的对话服务；这套自回归生成循环也正是 Agentic RL 中 rollout 引擎的底层实现。

延伸阅读见文首「关联笔记」列表，建议按 **Attention/Transformer 基础 → 参数与形状细节 → MoE → PPO/GRPO 对照表 → 对齐（含 DPO 视角）→ verl 工程实践** 的顺序交叉阅读，形成从原理到工程的完整闭环。