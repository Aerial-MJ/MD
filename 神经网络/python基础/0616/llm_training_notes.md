# LLM 训练知识笔记（LlamaFactory / Qwen3-VL）

---

## 一、训练日志指标解读

### 1. 四张 TensorBoard 图的含义

#### train/epoch（左上）
- 横轴是 step，纵轴是 epoch 进度
- 图形是线性增长，表示训练在稳定推进
- 例：step=200 时 epoch≈0.1，表示才走完第一个 epoch 的 10%

#### train/loss（左下）⭐ 最重要
- Loss 表示模型预测值与真实标签之间的误差
- Loss 持续下降说明模型在有效学习
- 例：`0.46 → 0.44 → 0.39 → 0.27 → 0.21 → 0.17 → 0.16`
- 前 100 步下降很快，之后趋于平缓属于正常现象

#### train/grad_norm（中上）
- **grad_norm（梯度范数）** = 所有参数梯度的"总体大小"，公式为：

  ```
  grad_norm = ||∇L||₂ = sqrt(∑ gᵢ²)
  ```

- 反映梯度的幅度，前期波动较大（模型刚开始学习，参数更新剧烈），后期趋稳说明进入平稳收敛阶段
- 如果 grad_norm > 100，说明出现了**梯度爆炸**，需要警惕
- 正常范围内（1~3）属于健康状态

#### train/learning_rate（右上）
- 体现学习率调度策略，见下文 Warmup 章节

---

## 二、Warmup（学习率预热）机制

### 为什么要 Warmup？

训练初期参数随机，梯度不稳定。如果一开始就用大 LR，容易震荡或破坏预训练权重。因此先用小 LR "热身"，让模型适应数据，再用大 LR 加速学习。

### Warmup 阶段划分

```
Warmup 阶段：LR 从 0 线性增长到最大值（如 5e-5）
      ↓
Cosine/Linear 衰减阶段：LR 从最大值逐渐降低
```

### Warmup 步数计算

```
Warmup steps = Total steps × warmup_ratio
             = 3,696 × 0.05
             = 184.8 ≈ 185 步
```

对照日志验证（warmup_ratio=0.05）：

| Step | LR | 状态 |
|------|----|------|
| ~170 | 4.567e-05 | 还在上升 |
| ~180 | 4.837e-05 | 还在上升 |
| ~185 | 4.9999e-05 | ✅ 到达峰值，Warmup 结束 |
| ~190 | 4.9998e-05 | 开始微微下降 |

---

## 三、Iteration（迭代） vs Step（步骤）的区别

| 概念 | 含义 | 单位 |
|------|------|------|
| **Iteration（迭代）** | 每处理一个 mini-batch 就是一次迭代，参数不一定更新 | 次 |
| **Step（优化步）** | 每调用一次 `optimizer.step()` 才是一个 step，参数实际更新 | 次 |
| **Epoch（轮）** | 整个训练集被完整遍历一遍 | 轮 |

### 关键区别：梯度累积

```
Gradient Accumulation steps = 16
```

- 每跑 16 个 iteration（mini-batch），才做一次参数更新（1 step）
- 所以：`step = iteration / grad_accum_steps`
- 效果等价于把 batch_size 放大了 16 倍，但不需要增加显存

### 总步数计算公式

```
每个 epoch 的步数 = 总数据量 / (batch_size_per_device × grad_accum × GPU数)
                 = 29,568 / (1 × 16 × 1)
                 = 1,848 步

Total steps = 1,848 × 2 epochs = 3,696 步
```

> 反推：若 Total steps = 3,696，则实际参与训练数据量 = 3,696 × 16 / 2 = **29,568 条**
> （与标称 29,000 条有偏差，是因为数据预处理过滤/补齐导致）

---

## 四、Evaluation（评估）进度条 vs 训练进度条

训练日志里出现两种进度条，**含义完全不同**：

### 训练进度条
```
5%|▌  | 200/3696 [1:10:15<20:34:34, 21.19s/it]
```
- 总数 3696 = 总优化步数
- 代表训练的整体进度

### 评估进度条
```
68%|██████▊  | 2237/3286 [18:23<08:23, 2.08it/s]
```
- 总数 3286 = **验证集样本数量**（不是训练步数！）
- 这是在跑 **Evaluation（验证集推理）**，不是训练

日志里也明确写了：
```
Num examples = 3286
Batch size = 1
```

### 为什么评估这么慢？

| 原因 | 说明 |
|------|------|
| Batch size = 1 | 每次只推理 1 条，效率很低 |
| 样本数 3286 | 数据量不小 |
| 推理有开销 | 比训练还慢 |

### 加速评估的建议

```yaml
per_device_eval_batch_size: 4   # 从1改到4，速度提升约4倍
eval_steps: 500                  # 减少评估频率（原来可能是200步一次）
```

---

## 五、总训练步数 3,696 的完整来源

```
已知：
  总数据量     = 29,568 条（实际参与训练）
  Epochs       = 2
  batch_size_per_device = 1
  Gradient Accumulation steps = 16
  GPU 数量     = 1

计算：
  每 epoch 步数 = 29,568 / (1 × 16 × 1) = 1,848
  Total steps   = 1,848 × 2 = 3,696
```

从日志中直接可见：
```
Total optimization steps = 3,696
Gradient Accumulation steps = 16
Instantaneous batch size per device = 1
Total train batch size (w. parallel, distributed & accumulation) = 16
```

---

## 六、显存占用分析：Qwen3-VL-2B LoRA 训练

### 显存占用概览

配置约占用 **23.6GB / 40GB** 显存（A100-40G），对 2B 模型来说偏大，原因如下：

### 显存占用来源拆解

#### 1. 模型本身（基础显存）
```
Qwen3-VL-2B 参数量 ≈ 2B（实际含 Vision Encoder，参数量 > 标称 2B）
BF16 精度：2B × 2 bytes = ~4GB
```

#### 2. LoRA 配置开销较大 ⚠️
```yaml
lora_rank: 64        # rank 较大
lora_alpha: 128
lora_target: all     # ← 关键！对所有模块都加 LoRA（attention + FFN + 视觉编码器）
```
`lora_target: all` 意味着全部模块都加适配器，rank=64 较大，显存开销显著。

#### 3. 优化器状态（最大头）
```
AdamW：每个可训练参数需要 2 个动量（FP32 精度）
即：LoRA 参数显存 × 3（param + m1 + m2）
```

#### 4. 激活值
```yaml
gradient_checkpointing: true   # ✅ 已开启，有所缓解
cutoff_len: 8192               # ⚠️ 序列长度较长，激活值仍占用一定显存
```

#### 5. 视觉输入分辨率
```yaml
image_max_pixels: 262144   # = 512×512
# 视觉 token 数量 ≈ 262144 / patch_size²，产生大量视觉 token 进入 LLM
```

### 显存估算汇总

| 来源 | 估算显存 |
|------|---------|
| 模型权重 (BF16) | ~5-6 GB |
| LoRA 参数 (all, rank=64) | ~1-2 GB |
| 优化器状态 (FP32 Adam) | ~4-6 GB |
| 激活值 (seq=8192, 含视觉token) | ~6-8 GB |
| 视觉编码器前向 | ~2-3 GB |
| CUDA 上下文 + 碎片 | ~1-2 GB |
| **合计** | **~20-27 GB** ✅ 符合观测 |

### 降低显存的建议

#### 方案一：减小 LoRA 范围（最有效）
```yaml
# 只对语言模型的 attention 层加 LoRA，不动视觉编码器
lora_target: q_proj,v_proj
# 或者
lora_target: q_proj,k_proj,v_proj,o_proj
```

#### 方案二：降低 LoRA rank
```yaml
lora_rank: 16    # 从 64 降到 16，LoRA 相关显存减少约 75%
lora_alpha: 32
```

#### 方案三：冻结视觉编码器
```yaml
freeze_vision_tower: true   # 冻结 ViT，只训练 LLM 部分
```

#### 方案四：减小序列长度
```yaml
cutoff_len: 4096   # 从 8192 减半，激活值显存明显减少
```

#### 方案五：使用 8bit 优化器
```yaml
optim: adamw_8bit   # 优化器状态从 FP32 降为 8bit
```

#### ✅ 推荐组合配置（可节省 8-12GB）
```yaml
lora_rank: 16
lora_alpha: 32
lora_target: q_proj,k_proj,v_proj,o_proj  # 不用 all
freeze_vision_tower: true
cutoff_len: 4096
optim: adamw_8bit
```

> 当前 23.6GB 在 A100-40GB 上可以正常训练，如果不 OOM 无需强制优化。
> 但如果想跑更大 batch 或更长序列，上述优化会很有帮助。

---

## 七、训练状态健康度速查

| 指标 | 当前状态 | 说明 |
|------|---------|------|
| Loss | ✅ 正常下降 | 模型在有效学习 |
| grad_norm | ✅ 趋于稳定（~0.66~1.9） | 无梯度爆炸风险 |
| LR | ✅ Warmup 已结束（step 185） | 进入 cosine 衰减阶段 |
| 显存 | ✅ 23.6GB / 40GB | 有余量，可正常训练 |
| 进度 | ⏳ 约 47%（step 1743/3696） | 继续等待 |



# 迭代次数不乘 dp_size
不需要，**迭代次数不乘 dp_size**。

---

## 为什么不乘？

因为 DP 的多张卡是**同时并行**处理不同数据的，不是串行的。

```
dp_size = 4，每卡 micro_batch = 4

第 1 次迭代（同时发生）：
  卡0 处理样本 [0~3]
  卡1 处理样本 [4~7]
  卡2 处理样本 [8~11]
  卡3 处理样本 [12~15]
  → 共消耗 16 条样本，但只算 1 次迭代
```

4 张卡走完算 **1 次迭代**，消耗的样本数是 `micro_batch × dp_size = 16`。

---

## 所以迭代次数怎么算

```
# 不考虑梯度累积
迭代次数 = 样本总数 / (micro_batch_size × dp_size)
         = 样本总数 / global_batch_size（梯度累积=1时）

# 考虑梯度累积
迭代次数（forward次数）= 样本总数 / (micro_batch_size × dp_size)
参数更新次数（steps）  = 迭代次数 / 梯度累积步数
                      = 样本总数 / global_batch_size
```

---

## 关键区分

| 概念 | 含义 |
|------|------|
| **迭代（iteration）** | 一次 forward，消耗 `micro_batch × dp_size` 条数据 |
| **step（参数更新）** | 累积 N 次迭代后做一次 backward + optimizer.step |
| **epoch** | 所有样本过一遍 |

**dp_size 扩大的是"每次迭代消耗的样本数"，而不是迭代次数。** 迭代次数反而因此变少了（同样的数据，dp 越大，迭代越少，训练越快）。


# Loss的计算
---

## 关键在于 loss 怎么算的

每次 forward，loss 通常是这个 micro_batch 内所有样本的**平均**：

```python
loss = mean(L1, L2)   # 2条数据，取平均
# 而不是 loss = sum(L1, L2)
```

所以 backward 出来的梯度，已经是"单样本规模"的了。

---

## 用数字验证

```
4条数据，梯度分别是 0.2, 0.4, 0.1, 0.3

# 直接平均（理想）
梯度 = (0.2 + 0.4 + 0.1 + 0.3) / 4 = 0.25

# 梯度累积（micro_batch=2，累积2次）
第1次 backward：loss = mean(0.2, 0.4) → 梯度 = 0.3
第2次 backward：loss = mean(0.1, 0.3) → 梯度 = 0.2

累积后 .grad = 0.3 + 0.2 = 0.5
step 时除以累积次数：0.5 / 2 = 0.25  ✅ 完全一样
```

---

## 所以你说的"最后还是单样本规模"

✅ 完全正确——

- 每次 backward 的梯度是 micro_batch 内的**平均**（单样本规模）
- 累积多次后还要再**除以累积步数**
- 最终结果等价于对整个 global_batch 取平均

**本质就是：把一个大平均，拆成几个小平均再平均，数学上完全等价。**