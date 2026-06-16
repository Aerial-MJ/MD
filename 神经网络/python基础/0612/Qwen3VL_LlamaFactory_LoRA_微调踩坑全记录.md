# Qwen3-VL-2B LoRA 微调踩坑全记录

**日期**：2026-06-12  
**模型**：`Qwen3-VL-2B-Instruct`  
**框架**：`LlamaFactory`  
**任务**：图片有效性审核的 SFT 微调（多模态 LoRA）  
**数据**：IG 业务图片 + 决策树 CoT 标注  

---

## 一、背景说明

### 目标
用 LlamaFactory 对 Qwen3-VL-2B-Instruct 进行 LoRA 微调，训练数据是内部业务图片审核场景，每条数据包含：
- 一张图片
- 长 system prompt（包含完整决策树说明，约 800+ tokens）
- 用户输入（指定场景）
- 模型输出（结构化 JSON CoT）

### 环境
- GPU：NVIDIA A100-SXM4-40GB
- Python 3.12
- transformers 4.57.1
- LlamaFactory（本地 clone 版本）
- 模型路径：`/home/hadoop-grocery-rc/.../model/Qwen3-VL-2B-Instruct`

---

## 二、LlamaFactory LoRA 微调原理简介

### 2.1 LoRA 是什么

LoRA（Low-Rank Adaptation）是一种参数高效的微调方法。核心思想：

```
原始权重矩阵 W（冻结，不更新）
                ↓
新增低秩矩阵 A × B（可训练，参数量很小）
                ↓
实际计算：output = W·x + (A·B)·x·(alpha/rank)
```

**冻结的部分**（不参与训练，占绝大多数参数）：
- 所有原始 Transformer 权重（`q_proj`, `k_proj`, `v_proj`, `o_proj` 等）
- 视觉编码器（Vision Encoder）的所有参数
- Embedding 层
- LayerNorm 层

**训练的部分**（参数量约 4.4%）：
- 每个被选中线性层上附加的 LoRA A、B 矩阵
- 本次训练日志显示：`trainable params: 98,729,984 || all params: 2,226,262,016 || trainable%: 4.4348`

### 2.2 Qwen3-VL 多模态 LoRA 的特殊性

Qwen3-VL 是视觉语言模型，图片经过视觉编码器处理后变成 image token，再与文本 token 拼接输入语言模型。LoRA 可以作用于：
- 语言模型中的线性层（`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` 等）
- 视觉编码器的线性层（`qkv`, `proj`, `linear_fc1`, `linear_fc2`）
- 本次配置 `lora_target: all` 表示所有线性层都加 LoRA

### 2.3 LlamaFactory 多模态数据处理的两阶段流程

这是理解所有 bug 的关键！LlamaFactory 对多模态数据的处理分为两个阶段：

**阶段一：数据集预处理（tokenize 阶段）**
```
图片路径 + 文本
    → mm_plugin.process_messages()
        → _get_mm_inputs()（第一次处理图片）
            → _regularize_images()（PIL resize，限制到 image_max_pixels）
            → image_processor(images)（smart_resize，对齐到 patch 倍数，计算 grid_thw）
        → 计算每张图产生的 image token 数量
        → 将 IMAGE_PLACEHOLDER 替换为 N 个 <|image_pad|> token
    → tokenizer.encode()（文本+image token 一起 tokenize）
    → 如果总长度 > cutoff_len：截断（！！危险区域！！）
    → 存储到数据集缓存
```

**阶段二：训练 collator（每个 batch 时）**
```
从缓存读取 input_ids（已经可能被截断）
    → mm_plugin.get_mm_inputs()（第二次处理图片）
        → _get_mm_inputs()（重新从原始路径加载图片，重新处理）
        → 得到 pixel_values, image_grid_thw
    → 模型前向传播时对比：
        input_ids 中 image_pad token 数量（来自阶段一）
        vs pixel features 数量（来自阶段二）
        → 不一致就报错！
```

---

## 三、错误排查全过程

### 3.1 初始配置（失败）

**配置文件**：`train_qwen3vl_lora.yaml`（初始版本）

```yaml
model_name_or_path: .../Qwen3-VL-2B-Instruct
image_max_pixels: 1003520   # ← 问题点1
...
cutoff_len: 4096             # ← 问题点2（后来发现）
per_device_train_batch_size: 2  # ← 问题点3
eval_strategy: steps         # ← 问题点4
```

---

### 3.2 错误一：`tokens: 914, features 972`

**完整错误信息**：
```
ValueError: Image features and image tokens do not match: tokens: 914, features 972
```

**原因分析**：

`image_max_pixels: 1003520` 设置过大（约 1002×1002 像素），导致两阶段图片处理结果不一致。

具体机制：
- `_preprocess_image`（PIL resize）：等比例缩放到 ≤ 1003520 像素，结果是任意浮点尺寸取整
- `image_processor.smart_resize`（内部逻辑）：对齐到 `patch_size * temporal_patch_size * merge_size = 16 * 2 * 2 = 64` 的倍数

PIL 的等比例缩放：
```python
resize_factor = sqrt(1003520 / (width * height))
new_w = int(width * resize_factor)   # 取整，可能不是64的倍数
new_h = int(height * resize_factor)  # 取整，可能不是64的倍数
```

`smart_resize` 需要：
```python
# 宽高必须是 patch_size * merge_size = 32 的倍数
# 而 Qwen3VL 实际要求对齐到 64
new_w = round(new_w / 64) * 64    # 会调整！
new_h = round(new_h / 64) * 64    # 会调整！
```

两次处理后得到的网格尺寸（`image_grid_thw`）不同，token 数自然不同。

`image_max_pixels` 越大，PIL resize 后的尺寸越大，`smart_resize` 的调整幅度越可能造成显著的 token 数差异。

**修复**：将 `image_max_pixels` 从 `1003520` 降低到 `262144`（= 512×512）：
```yaml
image_max_pixels: 262144
```

---

### 3.3 错误二：`tokens: 254, features: 504`

**完整错误信息**：
```
ValueError: Image features and image tokens do not match: tokens: 254, features: 504
```

**注意特征**：features = 2 × tokens，正好是 2 倍关系！

**原因分析**：

`per_device_train_batch_size: 2` 时，collator 阶段一次处理 2 个样本，两张图片一起传入 `_get_mm_inputs`。

问题出在两阶段处理的不对称性：
- **阶段一**（tokenize）：每个样本单独处理，图片 A → 254 tokens
- **阶段二**（collator）：2 个样本合并处理，图片 A + 图片 B 一起传入 `image_processor`

`image_processor` 在批量处理时，可能因为 padding/对齐逻辑，导致单张图片产生的 features 数量与单独处理时不同（特别是当两张图尺寸不同时，内部会统一 padding 到相同尺寸）。

features = 504 正好是 features_A + features_B 合并后的结果（252 + 252），而 tokens 只有 254 （单张图的 token 数），说明合并处理改变了每张图的 features 数量。

**修复**：
```yaml
per_device_train_batch_size: 1  # 每次只处理1张图，消除批量处理不一致
gradient_accumulation_steps: 16  # 提高梯度累积步数，保持 effective batch size = 16
```

---

### 3.4 错误三：`False is not a valid IntervalStrategy`

**完整错误信息**：
```
ValueError: False is not a valid IntervalStrategy, please select one of ['no', 'steps', 'epoch']
```

**原因分析**：

YAML 文件中的语法问题：
```yaml
eval_strategy: no    # ← YAML 会把裸的 no 解析为布尔值 False！
```

YAML 规范中，`no`、`false`、`off` 都是布尔值 `False` 的合法写法（不带引号时）。

**修复**：加引号强制为字符串：
```yaml
eval_strategy: "no"   # ← 加引号，确保是字符串
```

---

### 3.5 错误四：`tokens: 246, features 252`（最终错误）

**完整错误信息**：
```
ValueError: Image features and image tokens do not match: tokens: 246, features 252
```

**注意特征**：
- batch_size 已经是 1，所以不是批量问题
- 差值只有 6，是减法不是倍数关系
- features（252）> tokens（246）

**原因分析**：这是最根本的问题！

**数据流追踪**：

```
数据集预处理阶段（process_messages）：
  1. 图片处理：image_max_pixels=262144 → smart_resize → grid_thw → 252 个 image token
  2. 系统 prompt + 252 image token + 用户输入 + JSON 输出 → 总长度超过 4096
  3. cutoff_len=4096 触发截断！input_ids 被截断，只保留前 4096 个 token
  4. 截断后，input_ids 中实际保留的 image_pad token 数 = 246

训练 collator 阶段（get_mm_inputs）：
  1. 从原始图片路径重新加载图片
  2. 重新处理 → 252 个 image features
  3. 对比：246 tokens vs 252 features → 不匹配！报错！
```

**为什么序列会超过 4096？**

看一条完整样本的内容构成：
- 系统 prompt（完整决策树说明）：约 900+ tokens
- `<|vision_start|>` + 252 个 `<|image_pad|>` + `<|vision_end|>`：约 254 tokens
- 用户输入（场景标签）：约 20 tokens
- 模型输出（JSON CoT，很详细）：约 600+ tokens
- 特殊 token、格式 token：约 20 tokens

**总计：约 1800 tokens**，不超过 4096？

等等，再仔细看 eval 日志里 input_ids 的长度——原来数据中还有**更长**的样本！决策树 prompt 非常详细，加上较长的 JSON 输出，完全可能超过 4096。

**从错误本身反推**：
- 252 个 features = 完整图片的 token 数（阶段二结果）
- 246 个 tokens = 截断后残留的 image token 数（阶段一结果）
- 差了 6 个 → 说明截断点恰好落在 image token 区域内，截掉了后面 6 个 image token

**修复**：将 `cutoff_len` 从 4096 增大到 8192：
```yaml
cutoff_len: 8192   # 确保完整序列不被截断
```

---

## 四、最终成功配置

```yaml
### model
model_name_or_path: /home/hadoop-grocery-rc/.../model/Qwen3-VL-2B-Instruct
image_max_pixels: 262144    # 512*512，足够小以避免 resize 不一致
video_max_pixels: 50176
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target: all

### dataset
dataset: ig_sft_train
dataset_dir: /home/hadoop-grocery-rc/.../work/IG/SFT/LlamaFactory
template: qwen2_vl            # Qwen3-VL 使用 qwen2_vl 模板（兼容）
cutoff_len: 8192              # 足够大，避免截断 image token
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 4

### eval
eval_dataset: ig_sft_val
val_size: 0.0
eval_strategy: "no"           # 带引号！避免 YAML 解析为布尔值
eval_steps: 200
per_device_eval_batch_size: 1

### output
output_dir: .../output/qwen3vl_lora
logging_steps: 10
save_strategy: steps
save_steps: 200
save_total_limit: 3
plot_loss: true
overwrite_output_dir: true
report_to: none

### train
per_device_train_batch_size: 1    # 必须为1，避免批量处理不一致
gradient_accumulation_steps: 16   # effective batch = 16
learning_rate: 1.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
fp16: false
gradient_checkpointing: true
ddp_timeout: 180000000
load_best_model_at_end: false
```

---

## 五、关键知识点汇总

### 5.1 `image_max_pixels` 的作用和陷阱

LlamaFactory 中 `image_max_pixels` 控制的是**预处理阶段 PIL resize 的上限**，通过 `patcher.py` 注入到 `processor` 对象：

```python
# model/patcher.py
setattr(processor, "image_max_pixels", model_args.image_max_pixels)
```

然后在 `mm_plugin.py` 的 `_preprocess_image` 中使用：

```python
def _preprocess_image(self, image, image_max_pixels, image_min_pixels):
    if (image.width * image.height) > image_max_pixels:
        resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
        width = int(image.width * resize_factor)   # 浮点取整！
        height = int(image.height * resize_factor) # 浮点取整！
        image = image.resize((width, height))
    return image
```

**陷阱**：这个 resize 是简单的等比例缩放，取整后的尺寸不一定是 Qwen3VL 所需的对齐倍数（64）。后续 `image_processor.smart_resize` 会再次调整。如果 `image_max_pixels` 太大，两次调整的差异会很显著。

**建议**：将 `image_max_pixels` 设置为 `patch_size * merge_size * N` 的平方的合理值，例如：
- `262144 = 512 × 512`（当前配置，256 个有效 patch，约 64 个 token per 图）
- `589824 = 768 × 768`（更高质量，约 144 个 token per 图）

### 5.2 `cutoff_len` 的作用和陷阱

`cutoff_len` 控制 tokenize 后的序列最大长度。**关键陷阱**：

```
多模态数据的 tokenize 顺序：
1. process_messages：展开 image token（例如 252 个 <|image_pad|>）
2. tokenizer.encode：整体 tokenize
3. 截断到 cutoff_len

截断是"硬截断"，不感知 image token 的边界！
可能切到一半 image token 序列里面。

而 collator 的 get_mm_inputs 是重新处理完整图片，
得到的 features 数量是截断前的完整数量。

→ 导致 features > tokens（截断后的图片 token 数）
```

**建议**：`cutoff_len` 要大于你所有数据中最长序列的长度。可以用以下脚本估算：

```python
# 估算最大序列长度
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("your_model_path")
max_len = 0
for text in your_texts:
    tokens = tokenizer.encode(text)
    max_len = max(max_len, len(tokens))
# 加上图片 token 数（约 252）
print(f"建议 cutoff_len >= {max_len + 300}")
```

### 5.3 `per_device_train_batch_size` 与图片处理的关系

**不建议 batch_size > 1** 用于 Qwen3-VL 等 dynamic-resolution 模型（每张图片 token 数不同）：

- `batch_size=1`：每次只处理 1 张图，两阶段处理完全一致 ✅
- `batch_size=2`：collator 阶段合并 2 张图，`image_processor` 可能因内部 padding 改变每张图的 grid，导致 token 数与阶段一不一致 ❌

如果想要更大的 effective batch size，用梯度累积代替：
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16   # effective batch = 16
```

### 5.4 YAML 布尔值陷阱

YAML 规范中，以下值**不带引号**时会被解析为布尔值：
```yaml
# 都是 False：
eval_strategy: no
eval_strategy: No
eval_strategy: NO
eval_strategy: false
eval_strategy: False
eval_strategy: FALSE
eval_strategy: off
eval_strategy: Off

# 都是 True：
some_flag: yes
some_flag: Yes
some_flag: YES
some_flag: true
some_flag: on
```

**解决方案**：需要字符串时加引号：
```yaml
eval_strategy: "no"    # 字符串 "no"
eval_strategy: 'no'    # 同样是字符串
```

### 5.5 Qwen3-VL image token 数量计算公式

```
image_token_count = (H / patch_size / merge_size) × (W / patch_size / merge_size)

其中：
  patch_size = 16（每个 patch 的像素大小）
  merge_size = 2（spatial merge，合并相邻 patch）
  temporal_patch_size = 2（时间维度，图片默认=1帧）

实际计算（smart_resize 后的图片尺寸 H × W）：
  grid_h = H / (patch_size * merge_size) = H / 32
  grid_w = W / (patch_size * merge_size) = W / 32
  tokens = grid_h * grid_w

例：512×512 的图片
  grid_h = 512 / 32 = 16
  grid_w = 512 / 32 = 16
  tokens = 256

但 smart_resize 可能把图片调整为非 512×512，例如 504×504：
  grid_h = 504 / 32 ≈ 15.75 → 但 504 / 32 = 15.75，不是整数...
  
实际上 smart_resize 保证 H 和 W 都是 patch_size * temporal_patch_size * merge_size = 64 的倍数：
  图片先被 PIL resize 到接近 512×512，
  再由 smart_resize 调整到最近的 64 倍数尺寸
  可能是 448×512、512×448 等
```

---

## 六、故障排查总结表

| 错误信息 | 根本原因 | 修复方法 |
|---------|---------|---------|
| `tokens: 914, features 972` | `image_max_pixels` 过大，PIL resize 与 smart_resize 结果不一致 | 降低 `image_max_pixels` 到 `262144` |
| `tokens: 254, features: 504` | `batch_size=2` 时批量图片处理改变了单图 features 数 | `per_device_train_batch_size: 1` |
| `False is not a valid IntervalStrategy` | YAML 中 `no` 被解析为布尔值 `False` | 改为 `eval_strategy: "no"` |
| `tokens: 246, features 252` | `cutoff_len=4096` 截断了 image token 序列 | 增大 `cutoff_len` 到 `8192` |

---

## 七、训练成功后的日志关键信息

```
[INFO] trainable params: 98,729,984 || all params: 2,226,262,016 || trainable%: 4.4348
[INFO] ***** Running training *****
[INFO]   Num examples = 742
[INFO]   Num Epochs = 3
[INFO]   Instantaneous batch size per device = 1
[INFO]   Total train batch size = 16
[INFO]   Gradient Accumulation steps = 16
[INFO]   Total optimization steps = 141
```

这说明：
- LoRA 只训练了全部参数的 4.4%（约 9900 万参数）
- 训练样本 742 条，3 个 epoch，每 16 步更新一次权重（共 141 步）

---

## 八、后续建议

1. **验证训练效果**：收敛后用测试集评估准确率，对比微调前后的变化

2. **如果显存不够**：
   - 降低 `image_max_pixels`（如 `131072 = 362×362`）
   - 降低 `lora_rank`（如从 64 降到 32）
   - 增大 `gradient_accumulation_steps` 替代 batch size

3. **如果准确率不够**：
   - 增加训练数据
   - 增大 `lora_rank`（从 64 到 128）
   - 调整学习率（当前 `1e-4`，可以尝试 `5e-5`）
   - 增加训练 epoch

4. **关于 eval_strategy**：如果需要开启评估，记得：
   ```yaml
   eval_strategy: "steps"   # 加引号！
   val_size: 0.1            # 从训练集切出 10% 作为验证集
   ```

5. **清除缓存**：每次修改配置后，建议设置 `overwrite_cache: true`，避免使用旧的数据缓存

---

## 九、相关文件路径

| 文件 | 说明 |
|-----|-----|
| `work/IG/SFT/LlamaFactory/train_qwen3vl_lora.yaml` | 最终成功的训练配置 |
| `work/IG/SFT/LlamaFactory/dataset_info.json` | 数据集注册信息 |
| `github/LLaMA-Factory/src/llamafactory/data/mm_plugin.py` | 多模态处理核心逻辑（含两阶段处理） |
| `github/LLaMA-Factory/src/llamafactory/data/collator.py` | 训练 collator（触发第二阶段处理） |
| `github/LLaMA-Factory/src/llamafactory/model/patcher.py` | 将 yaml 参数注入 processor |
| `model/Qwen3-VL-2B-Instruct/preprocessor_config.json` | 图片处理器的默认参数（patch_size, merge_size 等） |
| `work/IG/SFT/LlamaFactory/output/qwen3vl_lora/` | 训练输出（checkpoint, loss 曲线等） |
