### data_collator

`data_collator = DataCollatorForLanguageModeling(...)`

 它定义了**每个 batch 的数据如何被组织（collate）成模型输入**。
 在训练中，`Trainer` 会：

- 从 `Dataset` 里取出若干条文本样本；
- 调用 `data_collator` 对这些样本进行：
  1. **tokenize**
  2. **padding（自动对齐到相同长度）**
  3. **创建 labels（通常就是输入右移一位）**

`DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)`

| 参数        | 含义                              | 解释                                                         |
| ----------- | --------------------------------- | ------------------------------------------------------------ |
| `tokenizer` | 分词器                            | 用来把文本转成 token id                                      |
| `mlm=False` | 是否使用 Masked Language Modeling | 你设为 False，表示用的是 **Causal LM**（自回归语言模型，比如 GPT、Qwen） |

✅ 所以：

mlm=True → 用于 BERT 类任务（随机遮盖部分 token 预测）

mlm=False → 用于 GPT/Qwen 类任务（预测下一个 token）

输入两句：

```
["你好，今天股市怎么样？", "请给出AAPL的分析。"]
```

`DataCollatorForLanguageModeling` 会自动生成：

```
{
  "input_ids": [[101, 872, 520, ...], [101, 123, 456, ...]],
  "attention_mask": [[1, 1, 1, ...], [1, 1, 1, ...]],
  "labels": [[-100, -100, 872, 520, ...], ...]
}
```

（`labels` 是训练目标，预测下一个 token，`-100` 的位置会在 loss 计算时被忽略。）

### tokenizer.padding_side = "left"

**背景：为什么要 padding**

在 NLP 中，模型通常需要 **固定长度的输入序列**：

```
input_ids = tokenizer(["Hello", "Hi there"], padding=True)
```

- "Hello" → `[101, 7592, 102]`  长度 3
- "Hi there" → `[101, 7632, 2088, 102]` 长度 4

为了做 batch，需要把它们 pad 到相同长度，比如 4：

- `[101, 7592, 102, 0]`
- `[101, 7632, 2088, 102]`

> `0` 是 pad token id（取决于 tokenizer.pad_token_id）

---

**padding_side 控制填充位置**

- `padding_side="right"`（默认)
  - 在序列 **右侧填充**
  - `[101, 7592, 102, 0]`
- `padding_side="left"`
  - 在序列 **左侧填充**
  - `[0, 101, 7592, 102]`

---

**为什么有时需要左填充**

对于 **因果语言模型（Causal LM）**，比如 Llama：

- 模型只关注 **每个 token 之前的上下文**
- 如果 padding 在右侧，attention mask 可能更自然
- 但某些训练/生成策略（尤其是 **右对齐生成** 或 batch 对齐时）需要 **左侧 padding**，这样生成 token 时序列末尾对齐，方便模型自回归生成

例子：

```
tokenizer.padding_side = "left"

tokenizer(["Hello", "Hi there"], padding=True, return_tensors="pt")
# 生成的 input_ids 可能长这样：
# tensor([[   0, 101, 7592, 102],
#         [101, 7632, 2088, 102]])
```

同时注意，左 padding 也要配合 **attention_mask** 使用，否则模型会把 padding token 当成正常 token 计算。

### 模型保存

#### A. `trainer.save_model()`

- **作用**：保存 **整个训练器里的模型**（包括 LoRA adapter，如果有的话）到指定目录
- **使用场景**：推荐用在 **微调完成后**，尤其是用 LoRA 或 PEFT 时
- **效果**：在 `output_dir` 里会生成：

```
output_dir/
  ├── adapter_config.json   ← LoRA 配置
  ├── pytorch_model.bin     ← 权重（含 LoRA） 
  ├── config.json           ← 模型配置
  └── ...
```

- **注意**：如果不传参数，默认保存到 `training_args.output_dir`

```
trainer.save_model()   # ✅ 保存当前训练器里的模型
trainer.save_model("./my_lora_model")  # 指定保存目录
```

------

#### B. `trainer.model.save_pretrained()`

- **作用**：保存 **模型本身的权重和配置**，**不包含训练器其他信息**
- **使用场景**：
  - 你只想单独保存 LoRA 后的模型
  - 或者不使用 Trainer，只是想保存当前 model 对象

```
trainer.model.save_pretrained("./my_model_only")
```

- **区别**：
  - `trainer.save_model()` → 保存 **训练器 + LoRA 适配器 + tokenizer 可选**
  - `trainer.model.save_pretrained()` → 保存 **模型权重 + config**，不保证 LoRA adapter 配置完整

✅ 小结：微调完成，**一般用 `trainer.save_model()` 最保险**。

------

#### 分词器保存`tokenizer.save_pretrained(output_dir)`

- **作用**：保存分词器相关文件到 `output_dir`
- 生成的文件：

```
output_dir/
  ├── tokenizer.json
  ├── tokenizer_config.json
  ├── special_tokens_map.json
  └── ...
```

- **使用场景**：
  - 微调后，想连同模型一起使用
  - 或单独保存 tokenizer 方便推理

```
tokenizer.save_pretrained("./my_lora_model")  # 保存 tokenizer 到同一目录
```

------

####  `output_dir=output_dir`（TrainingArguments）

- **作用**：告诉 Trainer **训练输出的默认目录**
- Trainer 会在里面保存：
  - `checkpoint-xxxx`（训练过程的 checkpoint）
  - `trainer_state.json`（训练状态）
  - eval/log 文件

### dataset

####  加载单 JSON 文件

`split` 用来指定数据集的“划分”，常见值：

- `"train"`：训练集
- `"validation"` / `"dev"`：验证集
- `"test"`：测试集
- `"all"`：全部数据

对于单 JSON 文件：

```
dataset_train = load_dataset("json", data_files="train.json", split="train")
dataset_val = load_dataset("json", data_files="val.json", split="validation")
```

- **注意**：`split` 并不会自动把文件分成 train/val，它只是给 Dataset 的标识。

- 如果你有 **单个文件且只想加载所有内容**，可以用：

  ```
  dataset_train = load_dataset("json", data_files="train.json", split="all")
  ```

#### 加载多个文件

假设你有 train/val/test 三个 JSON：

```
data_files = {
    "train": "train.json",
    "validation": "val.json",
    "test": "test.json"
}

datasets = load_dataset("json", data_files=data_files)
```

- 返回的 `datasets` 是一个字典：

```
datasets["train"]      # train.json
datasets["validation"] # val.json
datasets["test"]       # test.json
```

- 不需要单独写 `split`，因为 `load_dataset` 已经根据字典 key 自动划分。

**使用 `split` 加载部分数据**

你可以只加载其中一部分：

```
train_ds = load_dataset("json", data_files=data_files, split="train")
val_ds = load_dataset("json", data_files=data_files, split="validation")
```

#### 正确示例

```
from datasets import load_dataset

data_files = {
    "train": "../data/train.json",
    "validation": "../data/val.json"
}

dataset_train = load_dataset("json", data_files=data_files, split="train")
dataset_val = load_dataset("json", data_files=data_files, split="validation")
```

这样就很清晰，train 用训练文件，validation 用验证文件。

#### 划分训练集和验证集

把原本的 **train.json** 数据做一个 **8:2 划分训练集和验证集**，可以用 `datasets` 自带的 `train_test_split` 方法，而不是依赖 `split="train"`。

下面是示例写法：

```python
from datasets import load_dataset

# 1️⃣ 加载原始 train.json
dataset = load_dataset("json", data_files="../data/train.json", split="all")  # 加载全部数据

# 2️⃣ 按 8:2 划分
split_datasets = dataset.train_test_split(test_size=0.2, seed=42)  # test_size=0.2 即 20% 做验证集

dataset_train = split_datasets["train"]  # 80% 训练集
dataset_val = split_datasets["test"]     # 20% 验证集

# 3️⃣ 查看大小
print(len(dataset_train), len(dataset_val))
```

✅ 解释：

- `split="all"` 表示加载整个 JSON 文件的数据，而不是默认的 "train"。
- `train_test_split(test_size=0.2)` 会随机抽取 20% 作为验证集。
- `seed=42` 保证每次划分结果一致。
- 分割后的 key 是 `"train"` 和 `"test"`，可自由命名。

### DeepSpeed

**DeepSpeed** 是微软开源的一个 **大规模分布式训练加速库**，用于让超大模型（如百亿参数的 LLM）能够在有限 GPU 上 **更快、更省显存地训练和推理**。

------

#### 主要功能

| 功能类别                                        | 说明                                                         |
| ----------------------------------------------- | ------------------------------------------------------------ |
| **显存优化 (Memory Optimization)**              | 通过 **ZeRO（Zero Redundancy Optimizer）** 技术，把模型参数、梯度、优化器状态分布到多个 GPU 上，大幅减少每块卡显存占用。 |
| **分布式训练 (Distributed Training)**           | 支持 **数据并行、模型并行、流水线并行、张量并行** 等多种并行方式。 |
| **混合精度训练 (Mixed Precision)**              | 支持 FP16、BF16 训练，速度更快，显存更省。                   |
| **高性能通信 (High Performance Communication)** | 内置 NCCL 通信优化、Overlap 技术，提升多卡通信效率。         |
| **推理加速 (Inference Optimization)**           | 包含 **DeepSpeed-Inference**，在 LLM 推理阶段显著加速并降低显存占用。 |
| **Offload 技术**                                | 可以把部分参数或优化器状态转移到 CPU 或 NVMe（SSD）上存储，实现“超出显存的模型训练”。 |

------

#### 安装方式

```bash
pip install deepspeed
```

或使用 GPU 优化版本（例如 CUDA 12.1）：

```bash
DS_BUILD_OPS=1 pip install deepspeed --global-option="build_ext" --global-option="-j8"
```

------

#### 使用场景举例

##### 普通 PyTorch 训练

```python
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"
)

for batch in dataloader:
    outputs = model_engine(batch)
    loss = outputs.loss
    model_engine.backward(loss)
    model_engine.step()
```

##### `ds_config.json` 示例

```json
{
  "train_batch_size": 32,
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2
  }
}
```

------

#### ZeRO 技术等级说明

| Stage        | 优化内容                | 显存节省       | 特点             |
| ------------ | ----------------------- | -------------- | ---------------- |
| ZeRO-1       | 优化器状态分片          | 减少一部分显存 | 适合中等规模模型 |
| ZeRO-2       | + 梯度分片              | 显存减少更多   | 常用方案         |
| ZeRO-3       | + 参数分片              | 显存极致节省   | 支持百亿级模型   |
| ZeRO-Offload | 把部分数据放到 CPU/NVMe | 支持更大模型   | 但速度略慢       |

------

#### 应用在大模型项目中

DeepSpeed 常用于：

- 🤖 LLaMA、GPT、OPT、Qwen 等模型的多卡训练；
- 🧩 Hugging Face `Trainer` 集成；
- 🧠 微调框架如 `PEFT`、`LoRA`、`TRL`、`Accelerate` 等中底层调用。

------

#### 小结

| 优点                       | 缺点                 |
| -------------------------- | -------------------- |
| 显存利用率极高             | 配置略复杂           |
| 支持多种并行方式           | 对小模型不必要       |
| 可配合 Transformers 等使用 | 安装需匹配 CUDA 环境 |

### nn.Linear

在 PyTorch 里：

```
nn.Linear(in_features, out_features, bias=True)
```

- `in_features` → 输入特征的维度
- `out_features` → 输出特征的维度
- `weight` → 张量形状 `[out_features, in_features]`
- `bias` → `[out_features]`（可选）

> 注意：PyTorch 里 **weight 的形状是 `[output_size, input_size]`**，这就是你问的重点。

**weight 是 `[out_features, in_features]`**

线性层的数学公式是：

$y = x W^T + b$

- $x$ : `[batch_size, in_features]`
- $W$ : `[out_features, in_features]`
- $b $: `[out_features]`
- $y$ : `[batch_size, out_features]`

---

#### nn.Linear（PyTorch 模块化接口）

```python
import torch.nn as nn
layer = nn.Linear(128, 64)
output = layer(x)
```

- **自动管理权重和偏置**
- **模块化**，可以直接放在 `nn.Sequential` 或自定义 `nn.Module` 中
- **适合常规神经网络搭建**

 ####  Dense（Keras / TensorFlow 模块化接口）

```python
from tensorflow.keras.layers import Dense
layer = Dense(64, input_shape=(128,))
output = layer(x)
```

- Keras 风格的全连接层
- 自动管理参数
- 可以直接组合在 `Sequential` 或函数式 API 中

#### 函数式（Functional）接口

##### PyTorch 版

```python
import torch.nn.functional as F

output = F.linear(x, weight, bias)  # 需要自己提供权重和偏置
```

- **不保存状态**（不自动管理权重）
- 可以在 `forward` 中直接调用
- 灵活，适合**手动控制计算顺序或共享权重**

##### Keras / TensorFlow 版

```python
import tensorflow as tf
output = tf.linalg.matmul(x, W) + b
```

- 纯函数式实现
- 需要手动管理权重
- 灵活，适合实验或自定义网络

------

#### 自定义函数

```python
def linear_f(x, W, b):
    return x @ W + b
```

- 手动实现线性层
- 完全控制计算过程
- 缺点：不自动管理权重，也不自动求梯度（除非用 PyTorch tensor）

------

#### 小结对比

| 方法                               | 框架             | 权重管理 | 灵活性 | 使用场景                           |
| ---------------------------------- | ---------------- | -------- | ------ | ---------------------------------- |
| `nn.Linear`                        | PyTorch          | 自动     | 中等   | 普通网络模块化搭建                 |
| `Dense`                            | Keras / TF       | 自动     | 中等   | 顺序/函数式网络模块化搭建          |
| `F.linear` / `torch.nn.functional` | PyTorch          | 手动     | 高     | 共享权重、复杂计算流               |
| 自定义函数                         | PyTorch/TF/NumPy | 手动     | 很高   | 学习实验、特殊结构、自定义反向传播 |

**核心区别**

1. **状态管理**：模块化层（Linear/Dense）自动管理权重，函数式和自定义函数需要手动管理
2. **组合性**：模块化层方便堆叠成网络，函数式和自定义函数更灵活
3. **梯度计算**：PyTorch 的 `F.linear` + Tensor 自动求导可用，NumPy 自定义函数需要自己实现反向传播

**记忆技巧**：

- **模块化层（Linear/Dense）** → “方便堆叠，自动管理权重”
- **函数式（F.linear / 自定义函数）** → “灵活自定义，但要自己管权重”

### 常用的库

明白了，你想问的是在使用 **TRL（Transformers Reinforcement Learning）** 做大模型微调或强化学习时，除了 TRL，还有哪些库是必须或常用的。我们可以梳理一下按功能分类：

#### 核心库

| 库                      | 作用                                                         |
| ----------------------- | ------------------------------------------------------------ |
| `transformers`          | Hugging Face 模型加载、微调基础，所有 LLM 都基于它。         |
| `torch` / `torchvision` | PyTorch 训练基础，张量操作、模型定义、优化器、GPU 支持。     |
| `datasets`              | Hugging Face 数据集管理，方便加载、预处理、划分训练/验证集。 |
| `accelerate`            | 多 GPU / 分布式训练管理，自动设备映射，支持 DeepSpeed、FSDP 等。 |
| `trl`                   | 强化学习微调（RLHF/RL from human feedback）的核心库。        |

#### 可选但常用库

| 库                      | 用途                                                    |
| ----------------------- | ------------------------------------------------------- |
| `peft`                  | LoRA、P-Tuning 等轻量化微调方法，TRL 可配合 PEFT 使用。 |
| `deepspeed`             | 大模型训练加速，尤其 ZeRO 优化。TRL 可选加速。          |
| `safetensors`           | 高效模型权重存储，避免 pickle 的安全和性能问题。        |
| `evaluate`              | 方便做评测，如 BLEU、ROUGE、METEOR 等指标。             |
| `sentence-transformers` | 文本嵌入，常用于 reward 模型或相似度计算。              |

#### 数据 / 爬取 / 环境工具

| 库                             | 作用                              |
| ------------------------------ | --------------------------------- |
| `pandas` / `numpy`             | 数据处理、特征处理                |
| `datasets` + `huggingface_hub` | 下载、上传数据集或模型 checkpoint |
| `wandb` / `tensorboard`        | 训练日志可视化和监控              |

#### 实战推荐组合（针对 RLHF / TRL）

如果你想做 RLHF 或本地微调，大部分项目组合是：

```text
torch + transformers + datasets + accelerate + trl + peft + deepspeed + evaluate + safetensors
```

- `trl` 是核心逻辑
- `peft`/`deepspeed` 用于加速和节省显存
- `accelerate` 管理设备分布
- `evaluate` 做指标验证
- `datasets` 负责数据加载
- `safetensors` 提高权重保存效率

#### Detectron2

**功能**

- Facebook AI 开发的 **目标检测、实例分割、关键点检测框架**
- 基于 **PyTorch**，是 Detectron 的升级版
- 高性能、高可扩展性，支持 Mask R-CNN、RetinaNet、Panoptic FPN 等

**核心特点**

| 特点   | 说明                                                      |
| ------ | --------------------------------------------------------- |
| 模块化 | 配置和组件非常灵活，可以快速替换 backbone、RPN、ROI heads |
| 高性能 | 对 GPU 训练和推理做了优化，适合大规模数据集               |
| 多任务 | 支持目标检测、实例分割、全景分割、人关键点检测            |

**使用场景**

- 视觉多智能体系统里的场景理解
- OCR 前处理（检测文本区域）
- 自主驾驶、安防监控等目标检测任务

#### FlashAttention

##### 功能

- **加速 Transformer 注意力机制**
- 使用 **块状 GPU 内存访问和高效内核**，减少显存占用并提升速度
- 主要面向大模型（GPT、LLaMA、Qwen 等）的训练和推理

##### 核心特点

| 特点     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| GPU 高效 | 避免传统 attention 的 O(N²) 内存瓶颈                         |
| 精度     | 支持 FP16、BF16 和混合精度训练                               |
| 兼容性   | 可和 Hugging Face Transformers 结合，用在 `xformers` 或自定义 attention 里 |

##### 使用场景

- 大语言模型训练（GPT、LLaMA、Qwen）
- 注意力矩阵很大（比如长序列、图片 patch Transformer）
- 高速推理和显存受限环境
