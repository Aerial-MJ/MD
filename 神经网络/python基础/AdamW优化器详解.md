# AdamW 优化器详解

## 1. 背景与动机

### 1.1 从 SGD 到 Adam

| 优化器 | 核心思想 | 缺点 |
|--------|----------|------|
| SGD | 沿梯度方向下降 | 学习率敏感，收敛慢 |
| SGD + Momentum | 引入动量加速收敛 | 仍需手动调学习率 |
| AdaGrad | 自适应学习率（累积梯度平方） | 学习率单调递减，后期学习停滞 |
| RMSProp | 改进 AdaGrad，使用指数移动平均 | 无偏差修正 |
| **Adam** | 结合 Momentum + RMSProp | 权重衰减与梯度耦合 |
| **AdamW** | Adam + 解耦权重衰减 | 当前主流选择 ✅ |

### 1.2 为什么需要 AdamW？

Adam 中的 L2 正则化（权重衰减）存在问题：L2 正则化加入梯度后，会被二阶矩 `v` 的自适应缩放"稀释"，导致权重衰减效果不均匀。

AdamW 的核心贡献（Loshchilov & Hutter, 2019）：**将权重衰减从梯度更新中解耦出来**，直接作用于参数。

---

## 2. 核心变量：m 和 v 完全针对梯度

### 2.1 一阶矩 m（First Moment Estimate）

**本质**：梯度的指数加权移动平均，相当于带"惯性"的梯度方向。

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

- $g_t$：当前步的梯度
- $\beta_1$：一阶矩衰减系数（默认 0.9）
- $m_0 = 0$（初始化为零）

**直觉理解**：
- $m_t$ 是过去所有梯度的加权平均，越近的梯度权重越大
- 类似物理中的"动量"，使优化轨迹更平滑，减少震荡
- 展开后：$m_t = (1-\beta_1)\sum_{i=1}^{t} \beta_1^{t-i} g_i$

### 2.2 二阶矩 v（Second Moment Estimate）

**本质**：梯度平方的指数加权移动平均，相当于梯度幅度的"估计方差"。

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

- $\beta_2$：二阶矩衰减系数（默认 0.999）
- $v_0 = 0$（初始化为零）

**直觉理解**：
- $v_t$ 反映了该参数梯度的历史波动程度
- 梯度波动大（不稳定）的参数 → $v_t$ 大 → 实际学习率被压缩（保守更新）
- 梯度波动小（稳定）的参数 → $v_t$ 小 → 实际学习率相对较大（积极更新）

### 2.3 自适应学习率：怎么理解？

**核心问题**：为什么不让所有参数用同一个学习率 `lr`？

考虑一个 Transformer 模型，里面有两类参数：

```
参数 A（常见词 embedding）：每个 batch 都有大量样本涉及，梯度每步都在变化 → 波动大
参数 B（稀有词 embedding）：大多数 batch 根本没有该词，梯度时大时零       → 波动大（方向随机）
参数 C（某 LayerNorm 权重）：梯度方向非常稳定，每步变化很小               → 波动小
```

如果用固定 `lr`：
- 对 C 来说 `lr` 合适，但对 A、B 来说会来回震荡
- 调小 `lr` 保证 A、B 稳定，但 C 又会收敛极慢

**AdamW 的解法**：给每个参数一个**自己的有效学习率**：

$$\text{有效学习率}_i = \frac{\alpha}{\sqrt{\hat{v}_{t,i}} + \epsilon}$$

- $\hat{v}_{t,i}$ 大（该参数历史波动大）→ 有效 lr 小 → 走小步，稳定收敛
- $\hat{v}_{t,i}$ 小（该参数历史波动小）→ 有效 lr 大 → 走大步，快速收敛

```
直觉类比：骑自行车走山路
  SGD    = 全程一档，陡坡和平路用同样力气 → 陡坡颠簸，平路太慢
  AdamW  = 自动变速，陡坡（波动大）降档走稳，平路（波动小）升档加速
```

本质上，$\sqrt{\hat{v}_t}$ 是在**根据每个参数的"地形难度"自动调节步长**，这就是"自适应"的含义。

> **关键结论**：`m` 和 `v` 都只与梯度 $g_t$ 有关，与参数 $\theta$ 本身无关。

---

## 3. 偏差修正（Bias Correction）

由于 $m_0 = v_0 = 0$，训练初期估计会偏向零，需要修正：

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**为什么需要修正？**

问题根源：$m_0 = 0$ 是人为初始化的**假零**，不代表真实梯度，会在训练初期把 $m_t$ 拖低。

以"真实梯度每步都是 1"为例（$\beta_1 = 0.9$）：

| 步骤 | 递推结果 | 修正前 $m_t$ | 修正系数 $\frac{1}{1-0.9^t}$ | 修正后 $\hat{m}_t$ |
|------|---------|------------|--------------------------|-----------------|
| $t=1$ | $0.9\times0 + 0.1\times1$ | **0.1** | $\frac{1}{0.1}=10$ | **1.0** ✅ |
| $t=2$ | $0.9\times0.1 + 0.1\times1$ | **0.19** | $\frac{1}{0.19}\approx5.26$ | **1.0** ✅ |
| $t=10$ | ... | **≈0.65** | $\frac{1}{0.65}\approx1.54$ | **≈1.0** ✅ |
| $t=100$ | ... | **≈0.99** | $\frac{1}{0.99}\approx1.01$ | **≈1.0** ✅ |

修正把"被假零拖低"的部分补回来，**初期补偿多，后期自动消失**。

随着 $t$ 增大，$\beta_1^t \to 0$，分母 $1-\beta_1^t \to 1$，修正系数趋近于 1，等于不修正。

---

**⚠️ 关键：递推始终用未修正的 $m_t$**

$\hat{m}_t$ 只是**临时计算结果**，用完即弃，不会存回去：

```
存储：始终保存未修正的 m_t（账本，持续累积）
使用：临时除以 (1-β₁ᵗ) 得到 m̂_t，更新参数后扔掉

m₀ = 0
  ↓ 递推（用原始 m₀）
m₁ → 临时修正 → m̂₁ → 更新参数 → 扔掉 m̂₁
  ↓ 递推（用原始 m₁，不是 m̂₁！）
m₂ → 临时修正 → m̂₂ → 更新参数 → 扔掉 m̂₂
```

如果存 $\hat{m}_{t-1}$ 再递推，下一步就会**修正两次**，越修越乱。

---

## 4. AdamW 完整更新流程

### 4.1 算法伪代码

```python
# 初始化
m = 0  # 一阶矩（梯度均值）
v = 0  # 二阶矩（梯度平方均值）
t = 0  # 时间步

# 每个训练步骤
for each batch:
    t += 1
    
    # Step 1: 计算当前梯度
    g = gradient(loss, params)
    
    # Step 2: 更新一阶矩（针对梯度）
    m = beta1 * m + (1 - beta1) * g
    
    # Step 3: 更新二阶矩（针对梯度平方）
    v = beta2 * v + (1 - beta2) * g ** 2
    
    # Step 4: 偏差修正
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    
    # Step 5: AdamW 参数更新（两部分解耦）
    params = params - lr * m_hat / (sqrt(v_hat) + eps)  # 梯度驱动更新
    params = params - lr * weight_decay * params         # 权重衰减（独立）
```

### 4.2 数学表达

$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha \lambda \theta_{t-1}$$

其中：
- $\alpha$：学习率（默认 1e-3）
- $\epsilon$：数值稳定项（默认 1e-8），防止除零
- $\lambda$：权重衰减系数（默认 0.01）

---

## 5. Adam vs AdamW：权重衰减的本质区别

### 5.1 Adam 中的 L2 正则化（有缺陷）

```python
# Adam + L2 正则化（错误做法）
g = gradient(loss, params) + weight_decay * params  # 权重衰减混入梯度！
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g ** 2
params = params - lr * m_hat / (sqrt(v_hat) + eps)
```

**问题**：权重衰减项 $\lambda \theta$ 被加进梯度后，同样受到 $\sqrt{\hat{v}_t}$ 的自适应缩放。

> 注意：$\hat{v}_t$ 反映的是梯度的**历史波动幅度**（梯度平方的指数移动平均），不是某一步梯度的大小。

- 对于梯度**波动大**（不稳定）的参数：$\hat{v}_t$ 大 → 权重衰减被大幅压缩 → **正则化不足**
- 对于梯度**波动小**（稳定）的参数：$\hat{v}_t$ 小 → 权重衰减被相对放大 → **正则化过度**

结果：不同参数的实际权重衰减力度**完全依赖其梯度波动情况**，而非统一的 $\lambda$，导致正则化效果不可控。

### 5.2 AdamW 中的解耦权重衰减（正确做法）

```python
# AdamW（正确做法）
g = gradient(loss, params)          # 梯度不包含权重衰减
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g ** 2
params = params - lr * m_hat / (sqrt(v_hat) + eps)  # 梯度更新
params = params - lr * weight_decay * params         # 权重衰减独立作用！
```

**效果**：每个参数都以统一的比例 $\lambda$ 进行衰减，不受梯度自适应缩放的影响。

### 5.3 对比总结

| 特性 | Adam + L2 | AdamW |
|------|-----------|-------|
| 权重衰减方式 | 混入梯度，被 $\hat{v}$ 缩放 | 直接作用参数，独立更新 |
| 正则化效果 | 不均匀，实际效果弱 | 均匀有效 |
| 泛化性能 | 较差 | 更好（实验验证）|
| 适用场景 | 较少使用 | 大模型训练主流选择 |

---

## 6. 超参数详解

| 超参数 | 符号 | 代码变量 | 默认值 | 作用 | 是否常调 |
|--------|------|---------|--------|------|--------|
| **学习率** | **$\alpha$** | **`lr`** | 1e-3 | **控制整体更新步长** | ✅ 最重要 |
| **权重衰减** | **$\lambda$** | **`weight_decay`** | 0.01 | **防过拟合，控制参数大小** | ✅ 常调 |
| 一阶矩衰减 | $\beta_1$ | `beta1` | 0.9 | 控制梯度均值的"记忆长度" | 🔒 基本不动 |
| 二阶矩衰减 | $\beta_2$ | `beta2` | 0.999 | 控制梯度波动的"记忆长度" | 🔒 基本不动 |
| 数值稳定项 | $\epsilon$ | `eps` | 1e-8 | 防止除零 | 🔒 基本不动 |

> 💡 **实践建议**：虽然超参数看起来很多，但真正需要调的只有 `lr` 和 `weight_decay`，其余三个保持默认值即可。

### $\beta_1$ 和 $\beta_2$ 的影响

- $\beta_1 = 0.9$：$m_t$ 的有效"记忆窗口"约为 $\frac{1}{1-0.9} = 10$ 步
- $\beta_2 = 0.999$：$v_t$ 的有效"记忆窗口"约为 $\frac{1}{1-0.999} = 1000$ 步
- $v_t$ 记忆更长，使得学习率适应更稳定

---

## 7. PyTorch 实践

### 7.1 基本用法

```python
import torch
import torch.nn as nn

model = nn.TransformerEncoder(...)

# AdamW 优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),     # (beta1, beta2)
    eps=1e-8,
    weight_decay=0.01       # 权重衰减（λ）
)

# 训练循环
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

### 7.2 分组参数（常见技巧）

不对 bias 和 LayerNorm 参数进行权重衰减：

```python
def get_param_groups(model, weight_decay=0.01):
    """将参数分为两组：需要/不需要权重衰减"""
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # bias 和 LayerNorm 的参数不做权重衰减
        if param.ndim <= 1 or name.endswith('.bias'):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {'params': decay_params,    'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

param_groups = get_param_groups(model, weight_decay=0.1)
optimizer = torch.optim.AdamW(param_groups, lr=3e-4)
```

### 7.3 查看优化器状态

```python
# 训练一步后查看内部状态
optimizer.step()

# 获取第一个参数的 m 和 v
state = optimizer.state[list(model.parameters())[0]]
print(state.keys())        # dict_keys(['step', 'exp_avg', 'exp_avg_sq'])
print(state['exp_avg'])    # m_t（一阶矩）
print(state['exp_avg_sq']) # v_t（二阶矩）
print(state['step'])       # 当前时间步 t
```

---

## 8. 内存占用与精度分析

### 8.1 m 和 v 为什么必须是 FP32？

`m` 和 `v` 需要持续累积很小的增量。如果用 FP16：
- FP16 数值范围小，累积时容易**下溢（underflow）**，更新量直接变 0
- 训练初期 `v` 值极小，FP16 精度粒度不足，会引起数值不稳定甚至发散

因此，**无论是否使用混合精度训练，m 和 v 始终保持 FP32**。

### 8.2 纯 FP32 训练的内存占用

AdamW 需要为每个参数维护 `m` 和 `v` 两个额外状态：

| 组件 | 精度 | 每参数占用 |
|------|------|-----------|
| 模型参数 $\theta$ | FP32 | 4 bytes |
| 一阶矩 $m$ | FP32 | 4 bytes |
| 二阶矩 $v$ | FP32 | 4 bytes |
| **合计** | | **12 bytes/参数** |

以 GPT-2（117M 参数）为例：
- 参数：117M × 4 bytes ≈ **468 MB**
- m + v：2 × 468 MB ≈ **936 MB**
- 总计约 **1.4 GB**（仅优化器状态，不含激活值）

### 8.3 混合精度训练（AMP）的内存布局

混合精度训练的核心思路：**前向/反向用 FP16 加速，参数更新用 FP32 保证精度**。

```
前向传播 / 反向传播：
  模型参数（FP16）  ← 2 bytes，计算快，节省显存
  梯度 g_t（FP16） ← 反向传播产生

优化器 step（全程 FP32）：
  主参数副本（FP32）← 4 bytes，高精度参数，用于实际更新
  一阶矩 m（FP32）  ← 4 bytes
  二阶矩 v（FP32）  ← 4 bytes

更新完成后：FP32 主参数 → 转回 FP16 → 供下一步前向使用
```

**主参数副本（Master Weights）是什么？**

混合精度中需要额外保留一份 FP32 参数的原因：
- FP16 的精度粒度约为 $10^{-3}$，学习率 × 梯度往往是 $10^{-5} \sim 10^{-7}$ 量级
- 直接用 FP16 做 `param -= lr * update` 时，更新量会被**直接舍零**，参数永远不变
- FP32 副本保证每步微小更新都能被正确累积

**需要手动管理吗？** —— 不需要。PyTorch AMP、Accelerate、DeepSpeed 都会自动处理，只需：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():                          # 前向用 FP16
        loss = model(batch)
    scaler.scale(loss).backward()             # 梯度缩放防下溢
    scaler.step(optimizer)                    # 内部自动用 FP32 更新
    scaler.update()
    optimizer.zero_grad()
```

混合精度每参数实际占用：

| 组件 | 精度 | 每参数占用 |
|------|------|-----------|
| FP16 参数（前向用） | FP16 | 2 bytes |
| FP32 主参数副本 | FP32 | 4 bytes |
| 一阶矩 $m$ | FP32 | 4 bytes |
| 二阶矩 $v$ | FP32 | 4 bytes |
| **合计** | | **14 bytes/参数** |

> 相比纯 FP32 的 12 bytes/参数，混合精度反而多了 2 bytes（FP16 参数副本）。  
> 节省的是**激活值**（前向过程的中间结果），大 batch 时效果显著。

### 8.4 进一步压缩：8-bit AdamW

如果显存极度紧张，可以用 bitsandbytes 把 m/v 量化到 INT8：

```python
import bitsandbytes as bnb

optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01
)
# m/v 只占 1 byte，优化器状态显存减少约 75%
# 实践中对最终精度影响极小
```

| 方案 | m/v 精度 | 优化器状态/参数 |
|------|----------|----------------|
| 纯 FP32 | FP32 | 8 bytes（m+v） |
| 混合精度 AMP | FP32 | 8 bytes（m+v） |
| 8-bit AdamW | INT8 | 2 bytes（m+v）|

其他替代方案：
- **Adafactor**：不显式存储 v，用低秩近似压缩，显存接近 SGD
- **CAME**：在 Adafactor 基础上加入置信度感知

---

## 9. AdamW 在大模型训练中的使用

### 典型超参数配置

| 模型 | lr | $\beta_1$ | $\beta_2$ | $\lambda$ |
|------|-----|-----------|-----------|-----------|
| BERT | 5e-5 ~ 3e-4 | 0.9 | 0.999 | 0.01 |
| GPT-3 | 6e-5 | 0.9 | 0.95 | 0.1 |
| LLaMA | 3e-4 | 0.9 | 0.95 | 0.1 |
| ViT | 1e-3 | 0.9 | 0.999 | 0.1 |

> 注意：大模型通常使用 $\beta_2 = 0.95$ 而非 0.999，以减少二阶矩估计的"记忆"，更快适应梯度变化。

### 配合 Warmup + Cosine Decay

```python
from transformers import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,       # 前 1000 步线性增大 lr
    num_training_steps=100000,   # 总训练步数
)

# 训练循环
for step, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

---

## 10. 总结

```
AdamW 更新步骤：

1. g_t  = ∇L(θ_{t-1})               ← 计算梯度

2. m_t  = β1·m_{t-1} + (1-β1)·g_t   ← 一阶矩（对梯度）
3. v_t  = β2·v_{t-1} + (1-β2)·g_t²  ← 二阶矩（对梯度²）

4. m̂ = m_t / (1-β1^t)               ← 偏差修正
   v̂ = v_t / (1-β2^t)

5. θ_t = θ_{t-1} - α·m̂/(√v̂+ε)     ← 梯度驱动更新
        - α·λ·θ_{t-1}               ← 权重衰减（独立，不通过 v 缩放）

关键点：
  ✅ m 和 v 完全是对梯度 g_t 的统计量，与参数 θ 无关
  ✅ 权重衰减直接作用于参数，与梯度路径解耦
  ✅ 自适应学习率：梯度不稳定 → v 大 → 步长小；梯度稳定 → v 小 → 步长大
```

---

## 11. 权重衰减 vs 学习率调度（常见混淆）

### 11.1 两个完全不同的概念

| | 权重衰减 λ | Warmup + Cosine 调度 |
|--|-----------|----------------------|
| **作用对象** | 参数 θ 本身 | 学习率 lr |
| **目的** | 正则化，防止过拟合 | 训练稳定性，帮助收敛 |
| **作用时机** | 每步都持续缩小参数 | 控制每步的步长大小 |
| **超参数性质** | 固定常数（如 0.01） | 随 step 变化的动态曲线 |

### 11.2 权重衰减 λ 是什么？

**本质是正则化**，目的是防止参数过大、防止过拟合。

每步更新时，参数会额外"收缩"一点：

$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \underbrace{\alpha \lambda \theta_{t-1}}_{\text{权重衰减}}$$

等价形式：

$$\theta_t = \theta_{t-1} \cdot \underbrace{(1 - \alpha\lambda)}_{\approx 0.999999} - \alpha \cdot \text{梯度项}$$

以 λ=0.01、lr=1e-4 为例：
- 每步参数额外缩小系数：$1 - 10^{-4} \times 0.01 = 0.999999$
- 训练 10000 步后参数整体约缩小到 $0.999999^{10000} \approx 0.99$

**直觉**：大权重意味着模型对某些特征过度依赖 → 强迫参数保持小值 → 泛化更好。

> 🪒 **奥卡姆剃刀原则**：如果两个模型都能解释训练数据，优先选择更简单的那个。权重衰减就是这个原则的数学实现——它持续施压让参数趋向零，迫使模型"能用小参数解决的就不用大参数"，从而避免对训练集的过度拟合，获得更好的泛化能力。

### 11.3 Warmup + Cosine 调度是什么？

**本质是学习率调度（LR Schedule）**，控制学习率 `lr` 随训练步数的变化曲线，与参数本身无关。

```
lr
▲
│      ___-----------___
│     /                  \
│    /                    \___
│   /
│  /  ← Warmup        Cosine Decay →
└──────────────────────────────────→ step
     0    N_warmup          N_total
```

- **Warmup**：前 N 步 lr 从 0 线性增大到目标值。
  - 原因：训练初期 m/v 还未建立，梯度方向不可靠，大 lr 会导致震荡
- **Cosine Decay**：之后 lr 按余弦曲线从峰值降到接近 0。
  - 原因：训练后期需要精细调整，大 lr 会在最优解附近来回跳

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=100000,
)
```

### 11.4 两者同时使用，互不干扰

```python
optimizer = torch.optim.AdamW(
    params,
    lr=3e-4,          # lr 的峰值（被 scheduler 动态调整）
    weight_decay=0.01  # λ，固定不变，每步都作用于参数
)
scheduler = get_cosine_schedule_with_warmup(optimizer, ...)

for step, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()
    optimizer.step()   # ← 梯度更新 + 权重衰减（λ 固定）
    scheduler.step()   # ← 调整 lr（按 warmup/cosine 曲线）
    optimizer.zero_grad()
```

每一步实际发生的事：

```
Step t：
  lr_t   = cosine_schedule(t)          ← scheduler 决定当前步长
  θ_t    = θ_{t-1} - lr_t * 梯度项     ← 梯度更新，步长由 lr_t 决定
           - lr_t * λ * θ_{t-1}        ← 权重衰减，比例固定为 λ
```

> **总结**：权重衰减控制"参数有多大"，学习率调度控制"每步走多快"，是完全正交的两个机制。

---

## 12. SGD 原始公式与 AdamW 的区别

### 12.1 原始 SGD 公式

**最基础的 SGD（随机梯度下降）**：

$$\theta_t = \theta_{t-1} - \alpha \cdot g_t$$

其中 $g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$ 是当前 batch 的梯度。

---

**带动量的 SGD（SGD with Momentum）**：

$$v_t = \beta \cdot v_{t-1} + g_t$$

$$\theta_t = \theta_{t-1} - \alpha \cdot v_t$$

---

**带权重衰减的 SGD**：

$$\theta_t = \theta_{t-1} - \alpha \cdot g_t - \alpha \lambda \theta_{t-1}$$

---

### 12.2 SGD vs AdamW 核心区别对比

| 特性 | SGD | AdamW |
|------|-----|-------|
| **梯度估计** | 直接用原始梯度 $g_t$ | 一阶矩 $\hat{m}_t$（梯度指数平均） |
| **学习率自适应** | ❌ 所有参数同一学习率 | ✅ 二阶矩 $\hat{v}_t$（梯度波动）自适应缩放 |
| **更新幅度** | 随梯度波动大，不稳定 | $\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$ 归一化，幅度趋于均匀 |
| **权重衰减** | 与梯度耦合（L2正则） | **解耦**，独立施加 |
| **超参数** | $\alpha, \beta$（简单） | $\alpha, \beta_1, \beta_2, \epsilon, \lambda$（复杂） |
| **收敛速度** | 慢，需精心调 lr | 快，对 lr 不敏感 |
| **泛化能力** | 通常更好（平坦极小值） | 略差，但解耦 wd 后有改善 |

---

### 12.3 关键本质区别

**SGD 更新**：

$$\theta_t = \theta_{t-1} - \alpha \cdot g_t$$

步长 = 学习率 × 梯度，**梯度大的参数走得远**。

**AdamW 更新**：

$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha\lambda\theta_{t-1}$$

步长被 $\sqrt{\hat{v}_t}$（历史梯度波动幅度的平方根）**归一化**，梯度波动大的参数步长被压小（走得稳），梯度波动小的参数步长相对大（走得快），**每个参数的有效步长趋于均匀**。

> 💡 **一句话总结**：SGD 是"走多远全看这步梯度"，AdamW 是"用历史波动归一化，走得更稳更均匀"。这就是为什么 Adam 系列在 LLM 训练中更流行——参数量巨大时，各参数梯度波动差异悬殊，自适应学习率至关重要。

---

## 参考文献

- Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization.* ICLR 2015.
- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR 2019.
