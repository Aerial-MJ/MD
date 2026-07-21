# Verl

> 关联笔记：verl 这类框架所实现的 PPO/GRPO rollout-reward-update 流程，可对照 [MiniMind从0到1构建大模型 · 4.3 强化学习](MiniMind从0到1构建大模型.md#43-强化学习) 中从零手写的简化版实现（含 Agentic RL 多轮工具调用 rollout）；混合精度、学习率调度、DDP/ZeRO 等工程化细节见 [4.5 训练工程细节](MiniMind从0到1构建大模型.md#45-训练工程细节)。verl 与 TRL/OpenRLHF/DeepSpeed/Megatron/Accelerate 等框架在整个技术栈中所处的层次，以及训练侧用的 Megatron + 推理侧用 vLLM 这种"训练/推理后端解耦"设计背后的并行策略原理，见 [分布式训练与推理加速全解](分布式训练与推理加速全解.md)。

## 一、什么是 **verl**

**verl（Volcano Engine Reinforcement Learning）** 是字节跳动 Seed 团队推出的一个 **强化学习训练框架**，主要用于大型语言模型（LLMs）的强化学习（尤其是 RLHF，即 “人类反馈强化学习”）和 agent 训练任务。它的目标是：

- 提供高性能、高吞吐量的 RLHF 训练流水线；
- 支持多种强化学习算法（如 PPO、GRPO、DAPO 等）；
- 与现有大模型生态紧密集成（如 vLLM、Hugging Face、Megatron、FSDP 等）。

> 简单来说：它不是一个单纯的算法库，而是一个端到端强化学习训练 **工程级框架**，适合大模型训练。 

## 二、核心设计理念（内部原理）

verl 的设计在学术和工程层面上都有亮点，下面用“箭头式结构”总结最重要的点：

### 1）**HybridFlow 编程范式**

传统 RLHF 工具链往往将整个流程串成一条线，但 verl 提出了 **HybridFlow**：

- **单控制器（Single Controller）**：用于定义高层训练逻辑，类似传统 Python 脚本风格，开发灵活；
- **多控制器（Multi Controller）**：在执行上分摊到多个工作单元，实现高并行度；
- **HybridFlow 结合两者优点**：既能灵活定义训练逻辑，又兼顾高效执行。

这种设计的本质是把 **复杂流程编排和高性能执行** 分离，从而同时提高可扩展性和吞吐量。

------

### 2）**异步流水线与资源利用最大化**

verl 不是一步一步等执行完成，而是：

- 利用 **异步架构** 把「rollout（采样生成）」、「奖励评估」和「模型更新」等阶段解耦；
- 利用多 GPU / 多进程并发执行这些任务；
- 推理和训练使用不同后端实例（如 vLLM/SGLang 用于生成，FSDP/Megatron-LM 用于训练）。

这样可以在大模型训练中大幅提高 GPU 利用率和训练速度。

------

### 3）**模块化 & 可扩展性**

verl 采用**模块化设计**：

- 不同模块负责不同职责：策略、环境、奖励、数据存储、优化器；
- 用户可以插入自定义奖励函数、切换算法、接入不同后端；
- 对多模态（如视觉语言模型）和多任务场景都支持。

## 三、典型工作流程

以一个典型的 RLHF 训练流程为例，verl 的执行可以分为如下步骤：

```
Train Loop（训练循环）
├─ Rollout（环境采样）
│   ├─ Agent 使用当前策略生成样本
│   └─ 数据输出到缓冲区
├─ Reward（奖励评估）
│   ├─ 对 Rollout 样本产生奖励
│   └─ 存储 Reward 数据
├─ Optimize（策略优化）
│   ├─ 计算优势（Advantage）和损失
│   └─ 更新策略网络参数
└─ Sync（权重同步）
    ├─ 把新参数广播到生成模块（Inference）
    └─ 准备下一次 Rollout
```

verl 在这个流程中有更细致的调度机制，比如分片、异步执行等，但整体逻辑即是这种“生成 ➜ 评估 ➜ 优化”的循环。

## 四、配合代码讲解（上手示例）

下面给出一个简化的示例，展示 **如何用 verl 训练一个 PPO 强化学习任务**。

------

### 1）安装和准备

```
# 克隆 verl 仓库并安装
git clone https://github.com/volcengine/verl.git
cd verl
pip3 install -e .
```

------

### 2）创建一个 PPO 训练脚本

假设我们正在做简单的语言模型 RLHF 任务：

```python
from verl import PPOTrainer, RLConfig
from verl.envs import LanguageModelEnv
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载预训练模型
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# RL 配置
config = RLConfig(
    algorithm="ppo",
    batch_size=32,
    learning_rate=5e-5,
    clip_epsilon=0.2
)

# 创建强化学习环境
env = LanguageModelEnv(tokenizer, model)

# 初始化 Trainer
trainer = PPOTrainer(config=config, env=env)

# 训练循环（简化）
for epoch in range(100):
    # rollout 交互
    trajectories = trainer.collect_rollouts()
    
    # 估计奖励
    rewards = trainer.compute_rewards(trajectories)
    
    # 优化策略
    trainer.optimize(trajectories, rewards)

    print(f"Epoch {epoch}: reward avg = {rewards.mean()}")
```

> 说明：
>
> - `LanguageModelEnv` 是一个负责处理语言生成和计算奖励的环境；
> - `PPOTrainer` 在背后负责并发调度生成、评估和优化；
> - 这段代码虽然简化，但反映了强化学习三大阶段（采样、奖励、优化）的主逻辑。

注意：官方还有更详尽的配置文件和 YAML/CLI 支持，可查阅官方文档。

------

## 五、怎么理解它与其他 RL 框架的差异？

| 框架                | 主要特点                                                     |
| ------------------- | ------------------------------------------------------------ |
| Verl                | 侧重 **大模型 RLHF/Agent 训练** 的高性能工程框架，支持多算法和后端扩展 |
| OpenRLHF            | 更偏重通用 RLHF，基于 Ray 分布式生态                         |
| TRL（Hugging Face） | 简洁易用，适合中小规模 RLHF 实验                             |

verl 的最大优势在于 **高吞吐量和对大模型的高效支持**，特别适合大规模 GPU 集群。

---

## 六、GRPO 训练核心指标详解

### 6.1 为什么 verl 里有 `critic/` 前缀？

标准 GRPO 确实**没有独立的 critic 网络**，但 verl 框架为了代码复用，把 GRPO 计算的统计量放在了 `critic/` 命名空间下：

```
critic/advantages → GRPO 计算的优势值（reward - baseline）
critic/returns    → 每个回答的 reward 值
```

这不是真正的 critic 模型，只是 reward 的统计信息。

---

### 6.2 Actor/Loss（策略损失）

**本质：** 策略梯度损失，衡量模型"学习方向的正确性"。

```
训练目标：让好回答（advantages>0）的概率↑，差回答（advantages<0）的概率↓

loss 下降 → 模型在按 advantages 指引的方向优化 ✅
loss 不降 → 模型没有有效学习
```

#### 为什么需要 Loss？

神经网络训练的本质：

```
定义一个 Loss（衡量"现在有多差"）
    ↓
对 Loss 求梯度（找改进方向）
    ↓
更新参数（往更好的方向走一步）
```

GRPO 的 Loss 不是"对错"，而是**"哪个回答更好"**。

#### 完整公式

$$\mathcal{L}_\text{GRPO} = \underbrace{-\mathbb{E}\left[\min\left(\frac{\pi_\theta}{\pi_{\theta_\text{old}}} \cdot A,\ \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_\text{old}}}, 1{-}\varepsilon, 1{+}\varepsilon\right) \cdot A\right)\right]}_{\text{pg\_loss（主项）}} + \underbrace{\beta \cdot \text{KL}(\pi_\theta \| \pi_\text{ref})}_{\text{kl\_loss}} \underbrace{-\ \alpha \cdot H(\pi_\theta)}_{\text{entropy\_loss}}$$

#### 逐项拆解

**① pg_loss（策略梯度损失，主项）**

```
ratio = π_new(a|s) / π_old(a|s)    ← 新旧策略的概率比

结合 advantages（A）：
  A > 0（好回答）→ 希望 ratio 尽量大 → 增大输出概率
  A < 0（差回答）→ 希望 ratio 尽量小 → 减小输出概率

为什么要 clip？
  没有 clip：模型会拼命增大某 token 的概率 → 单步跳变 → 训练崩溃
  clip 限制 ratio 在 [1-ε, 1+ε]（ε 通常=0.2）
  → 每步最多允许概率变化 ±20%，像安全带防止"翻车"
```

**② kl_loss（KL 散度惩罚）**

```
KL(π_new || π_ref)：当前策略与参考模型（SFT 基础模型）的分布距离

作用：防止模型偏离参考模型太远（奖励作弊）
```

**③ entropy_loss（熵正则）**

```
-α × H(π)：负号表示鼓励熵大

作用：防止模式坍塌，保持输出多样性
```

#### 三项汇总

```
总 Loss = pg_loss    ← 主项：让好回答概率↑，差回答概率↓
        + kl_loss    ← 约束：别偏离参考模型太远
        - entropy    ← 正则：保持多样性

优化目标：最小化这个 Loss
```

**正常的变化趋势：**

```
训练初期：loss 较高（~0.15）← 模型随机，好坏回答概率差不多
    ↓
训练中期：loss 逐步下降    ← 模型学会让好回答概率 > 差回答概率
    ↓
训练后期：loss 趋于平稳（~0.05）→ 收敛 ✅
```

**⚠️ 注意：** loss 下降 ≠ 模型变好，必须配合 `reward/mean` 一起看。loss 下降但 reward 不涨，说明模型学会了"捷径"（比如对所有输入输出同一高分格式）。

---

#### 深入理解：Loss 是什么函数，怎么更新参数

##### Loss 是关于参数 θ 的多元函数

神经网络有几亿个参数，拼成一个向量 $\theta = [\theta_1, \theta_2, \dots, \theta_n]$，Loss 就是关于 $\theta$ 的 n 元函数：

$$\mathcal{L}(\theta_1, \theta_2, \dots, \theta_n)$$

**训练目标：找到让 $\mathcal{L}(\theta)$ 最小的 $\theta$。**

梯度就是对每个参数求偏导，组成一个 n 维向量：

$$\nabla_\theta \mathcal{L} = \left[\frac{\partial \mathcal{L}}{\partial \theta_1},\ \frac{\partial \mathcal{L}}{\partial \theta_2},\ \dots,\ \frac{\partial \mathcal{L}}{\partial \theta_n}\right]$$

- 偏导 > 0：增大这个参数会让 Loss 变大 → 应该减小它
- 偏导 < 0：增大这个参数会让 Loss 变小 → 应该增大它

参数更新公式（梯度下降）：

$$\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}$$

```
把 Loss 想象成山地地形，θ 是你的位置：

∇L = 当前位置最陡的上坡方向
-∇L = 最陡的下坡方向

每步往下坡方向走 η 距离 → 最终走到山谷（Loss 最小）
```

##### 为什么能算出来：反向传播（链式法则）

Loss 经过很多层计算才和 $\theta$ 关联，用链式法则逐层反向传播：

$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial a_n} \cdot \frac{\partial a_n}{\partial a_{n-1}} \cdots \frac{\partial a_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial \theta}$$

这就是"反向传播（Backprop）"，PyTorch 的 `loss.backward()` 自动帮你算，`optimizer.step()` 就是执行 $\theta \leftarrow \theta - \eta \cdot \nabla L$。

##### SFT 的 Loss 怎么从 token 叠到 Batch

**单个 token 的 Loss：**

$$\mathcal{L}_t = -\log P_\theta(y_t \mid x, y_{<t})$$

```
模型说"有效图"概率是 90% → -log(0.9) ≈ 0.105  ← loss 小，预测准
模型说"有效图"概率是 10% → -log(0.1) ≈ 2.303  ← loss 大，预测错
```

**一个序列（T 个 token）：**

$$\mathcal{L}_\text{seq} = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(y_t \mid x, y_{<t})$$

**一个 Batch（N 条样本）：**

$$\mathcal{L}_\text{SFT} = -\frac{1}{N}\sum_{i=1}^{N} \frac{1}{T_i}\sum_{t=1}^{T_i} \log P_\theta(y_t^{(i)} \mid x^{(i)}, y_{<t}^{(i)})$$

```
层层展开：

Batch（N 条样本）
  └── 样本 i（T_i 个 token）
        └── 每个 token 算 -log P(y_t)

总 Loss = 所有 token 的 -log P 取平均
```

##### Batch 梯度的本质：B 个样本"投票"

每个样本定义一个关于 $\theta$ 的函数（$x_i, y_i$ 是常数，$\theta$ 是变量）：

$$\ell_i(\theta) = \ell(\theta;\, x_i, y_i)$$

Batch Loss 是这些函数的平均：

$$\mathcal{L}_\text{batch}(\theta) = \frac{1}{B}\sum_{i=1}^{B} \ell_i(\theta)$$

对 Batch Loss 求梯度，由求导的线性性质：

$$\nabla_\theta \mathcal{L}_\text{batch} = \frac{1}{B}\sum_{i=1}^{B} \nabla_\theta \ell_i(\theta)$$

**关键理解：不同样本的梯度都是在同一个参数点 $\theta_0$ 处求的，x_i 不同但求导变量是 $\theta$，所以完全可以平均。**

```
Batch 里 B 个样本，每个样本"建议"参数往某个方向走：

样本1（猫）：θ₁ 应该 +0.3，θ₂ 应该 -0.1
样本2（狗）：θ₁ 应该 +0.1，θ₂ 应该 +0.2
样本3（车）：θ₁ 应该 +0.2，θ₂ 应该 -0.1

平均：θ₁ 应该 +0.2，θ₂ 应该 ≈ 0（抵消）

θ₂ 方向各样本意见相反 → 平均后接近 0 → 说明 θ₂ 不需要动 ✅
θ₁ 方向大家都说增大 → 平均后还是正 → 说明 θ₁ 确实需要增大 ✅

不同样本在"有共识"的维度叠加，在"有争议"的维度互相抵消
→ 这就是我们想要的全局最优方向
```

**真正想优化的是期望损失（整个数据集）：**

$$\mathcal{L}(\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\ell(\theta; x, y)\right]$$

数据集太大算不完，用 Batch 近似：

$$\nabla_\theta \mathcal{L} \approx \frac{1}{B}\sum_{i=1}^{B} \nabla_\theta \ell(\theta; x_i, y_i)$$

Batch 梯度是对"真实期望梯度"的蒙特卡洛估计，B 越大越准，但计算越慢。

**最简单的例子（只有两个参数）：**

```python
# 两个样本定义两个 loss 函数
ℓ₁(θ) = (θ₁ - 1)² + (θ₂ - 2)²   # 样本1（猫）
ℓ₂(θ) = (θ₁ - 3)² + (θ₂ - 0)²   # 样本2（狗）

# Batch loss = 取平均，这是一个普通的二元函数
L(θ) = ½ [ℓ₁(θ) + ℓ₂(θ)]
     = ½ [(θ₁-1)² + (θ₂-2)² + (θ₁-3)² + θ₂²]

# 在 θ=(0,0) 处求梯度，就能算出该往哪个方向走
```

##### PyTorch 里实际发生的事

```python
for batch in dataloader:
    # Step 1: 前向传播，算平均 loss（标量）
    loss = model(batch).mean()      # L = (L1+L2+...+LB)/B

    # Step 2: 反向传播，链式法则算出所有参数的偏导数
    loss.backward()                 # ∂L/∂θ 存在 param.grad 里

    # Step 3: 更新参数  θ ← θ - η∇L
    optimizer.step()

    # Step 4: 清零梯度（否则下次 backward 会累加）
    optimizer.zero_grad()
```

##### SFT vs GRPO 对比

| | SFT | GRPO |
|--|-----|------|
| **监督信号** | 正确答案的 token | 回答之间的相对 reward |
| **每个 token 的目标** | 预测正确 → loss 小 | 好回答概率↑，差回答概率↓ |
| **Loss 核心** | $-\log P(y_t)$ | $-\min(r\cdot A,\ \text{clip}(r)\cdot A)$ |
| **参数更新方式** | `loss.backward()` + 梯度下降 | **完全相同** |

参数更新方式两者完全相同，区别只在于 Loss 怎么算出来。

---

### 6.3 Actor/KL Loss（KL 散度损失）

**本质：** 当前策略与参考模型（ref model，通常是 SFT 基础模型）之间的分布差距。

$$\text{KL}(π_\theta \| π_\text{ref}) = \sum p_\theta(x) \log \frac{p_\theta(x)}{p_\text{ref}(x)}$$

**直觉理解：**

```
KL=0   → 当前模型和参考模型输出完全一样（没有学习）
KL 小  → 模型在参考模型基础上做了小幅调整 ✅
KL 大  → 模型偏离参考模型很远（可能过拟合/奖励作弊）
```

**正常的变化趋势：**

```
训练初期：KL≈0（模型还没开始学）
    ↓
训练中期：KL 缓慢上升（模型在学习偏离参考策略）
    ↓
训练后期：KL 稳定在某个值 ✅

⚠️ 危险：KL 持续快速增大 → 模型在"奖励作弊"，远离正常分布
```

---

### 6.4 Actor/KL Coef（KL 系数）

**本质：** KL 散度惩罚项的权重系数，控制"模型允许偏离参考模型多远"。

```
total_loss = pg_loss + kl_coef × kl_loss + entropy_loss

kl_coef 大 → 强制模型贴近参考模型（保守）
kl_coef 小 → 允许模型大幅偏离参考模型（激进）
```

**两种模式：**

| 模式 | 说明 | 特点 |
|------|------|------|
| 固定 KL（`kl_coef=0.001` 不变） | 整个训练过程系数恒定 | 简单稳定，verl 默认 |
| 自适应 KL | 根据实际 KL 值动态调整系数 | 更精细，但复杂 |

---

### 6.5 Actor/LR（学习率）

**本质：** 每步参数更新的"步长大小"。

```
lr 大 → 每步更新幅度大，学得快但可能不稳定
lr 小 → 每步更新幅度小，稳定但学得慢
lr = 0 → 完全不更新参数，训练停滞
```

**常见 lr_scheduler 类型：**

| 类型 | 行为 | 适用场景 |
|------|------|---------|
| `constant` | 全程固定 lr | GRPO 推荐，简单稳定 |
| `cosine` | 从初始值余弦衰减到 0 | 容易提前归零，需注意 `total_training_steps` |
| `warmup + cosine` | 先升后降 | 最常见，但末期 lr→0 后训练无意义 |

**⚠️ 常见坑：** `total_training_steps` 设置过小 → lr 提前归零 → 后续训练无参数更新。

```yaml
# 检查配置
actor:
  optim:
    lr: 1e-6
    lr_scheduler_type: cosine
    total_training_steps: 1000   # ← 是否设置太小？
    warmup_steps: 100
```

> **GRPO 建议用固定 lr：** GRPO 内置了 KL 约束、Clip 机制、reward 归一化等自适应机制，已经起到调节作用，不需要 lr 动态衰减。

---

### 6.6 Advantages（优势值）

**直觉理解：** GRPO 对同一个问题生成 N 个回答，用相对排名而非绝对分数来决定优化方向。

```
同一问题生成 4 个回答：
  回答1: score = 0.8
  回答2: score = 0.3
  回答3: score = 0.6
  回答4: score = 0.1

baseline（基准）= 平均值 = (0.8+0.3+0.6+0.1)/4 = 0.45

advantages:
  回答1: 0.8 - 0.45 = +0.35  ← 比平均好，增强这个回答
  回答2: 0.3 - 0.45 = -0.15  ← 比平均差，抑制这个回答
  回答3: 0.6 - 0.45 = +0.15  ← 比平均好，增强这个回答
  回答4: 0.1 - 0.45 = -0.35  ← 比平均差，抑制这个回答
```

> 类比班级考试：不看绝对分数，看相对排名。高于平均 → 增强；低于平均 → 抑制。

**⚠️ `advantages ≈ 0` 意味着什么？**

```
所有回答得分都一样
  → 没有"好回答"和"差回答"的区别
  → 模型不知道该往哪个方向优化
  → 训练停滞！
```

---

### 6.7 Entropy（熵）

**熵 = 模型输出的"不确定程度"**

$$H = -\sum p(x) \log p(x)$$

| 状态 | 熵值 | 含义 |
|------|------|------|
| 模型完全随机 | 高（≈2.0） | 每个词概率相同，完全迷茫 |
| 模型正常学习 | 中（≈0.7） | 有倾向但保留不确定性 ✅ |
| 模式坍塌 | 低（≈0.0） | 对所有输入输出同一答案 |

**具体例子（词表3个词）：**

```python
# 完全均匀：熵最大
p = [1/3, 1/3, 1/3]
H = log(3) ≈ 1.099   # 模型完全不确定

# 完全确定：熵=0
p = [1.0, 0.0, 0.0]
H = 0                 # 模型100%确定，可能坍塌

# 有倾向但不绝对：正常
p = [0.7, 0.2, 0.1]
H ≈ 0.80              # ✅ 理想状态
```

```
训练过程中熵的变化：

训练初期：熵高（模型乱猜）
    ↓
训练中期：熵下降（模型学到规律）
    ↓
训练正常结束：熵稳定在合理值

⚠️ 危险1：熵降到接近0 → 模式坍塌，所有图片输出同一答案
⚠️ 危险2：熵一直不下降 → 模型没学到任何东西
```

---

### 6.8 Entropy Loss（熵损失）

**熵 vs 熵 Loss 的区别：**

| | entropy | entropy_loss |
|--|---------|-------------|
| **本质** | 描述性统计（体温计） | 优化目标（退烧药） |
| **作用** | 告诉你模型现在多不确定 | 主动干预，防止输出太单一 |

**公式：**

$$\text{entropy\_loss} = -\text{entropy\_coef} \times \text{entropy}$$

**为什么有负号？**

```
Loss 的目标是"最小化"
我们想要熵大一点（保持多样性）→ 熵越大越好
但 loss 是越小越好

所以加负号：
  最小化(-entropy) = 最大化(entropy)
  → 优化 loss 的同时，鼓励熵增大
```

**总 Loss 构成：**

```
total_loss = pg_loss        ← 策略梯度损失（让好回答概率↑）
           + kl_loss        ← KL散度（别偏离参考模型太远）
           + entropy_loss   ← 熵损失（防止输出单一）
```

**`entropy_coef` 怎么调：**

| 情况 | 现象 | 解决 |
|------|------|------|
| 熵快速下降接近0 | 模型对所有输入输出同一答案 | 调大：0.001 → 0.005 |
| 熵一直很高不下降 | 模型一直乱猜，学不到东西 | 调小：0.001 → 0.0001 |

---

### 6.9 Grad Norm（梯度范数）

**梯度范数 = 所有参数梯度的"总更新力度"**

```
把训练想象成下山（找最低点）：

grad_norm = 每一步迈出的步伐大小

步伐太大（grad_norm很大）→ 可能跨过最低点，loss剧烈震荡
步伐太小（grad_norm很小）→ 走得慢，可能陷入局部最优
步伐=0  （grad_norm=0）  → 站着不动，完全没有学习
```

**正常范围：** 大模型训练通常在 1.0 ~ 5.0

**梯度裁剪（Gradient Clipping）：**

```python
max_grad_norm = 1.0  # 设置上限

# 如果 grad_norm > 1.0，就按比例缩小所有梯度
# 保证每步更新不会太剧烈
# 无论你多着急，每步最多只能迈1米
```

---

### 6.10 PG Clipfrac（策略梯度裁剪比例）

**本质：** 每一步中，有多少比例的 token 的概率比（新策略/旧策略）超出了 clip 范围 `[1-ε, 1+ε]`。

```
GRPO/PPO 的 clip 机制：

  ratio = π_new(a|s) / π_old(a|s)   ← 新旧策略的概率比

  loss = -min(
    ratio × A,                       ← 原始策略梯度
    clip(ratio, 1-ε, 1+ε) × A        ← 被截断的版本
  )

pg_clipfrac = 被 clip 的 token 比例
```

**怎么看：**

| 值 | 含义 | 状态 |
|---|------|------|
| 0.1 ~ 0.3 | 适度更新，有部分被 clip | ✅ 正常 |
| ≈ 0 | 几乎没有更新，策略原地不动 | ⚠️ 训练停滞 |
| > 0.5 | 大量被 clip，更新太激进 | ⚠️ lr 可能过大 |

> `pg_clipfrac_lower`：概率比低于 `1-ε` 的比例，即模型输出概率相比上一步大幅下降的 token。偶尔出现尖峰是正常的，持续非零说明某些回答被强烈抑制。

---

### 6.11 内存指标（max_memory_allocated / max_memory_reserved）

**本质：** GPU 显存使用情况。

| 指标 | 含义 |
|------|------|
| `max_memory_allocated` | 实际分配给张量的显存峰值 |
| `max_memory_reserved` | PyTorch 向 CUDA 申请的总显存（含缓存池） |

```
reserved ≥ allocated

reserved - allocated = PyTorch 缓存池（预留但未被占用）
```

**⚠️ 异常信号：**
- 内存在某个 step 后突然下降 → 可能是 batch size 变化、序列截断或 OOM 导致的异常
- 内存持续上涨直到 OOM → 存在显存泄漏或 KV cache 过大

---

### 6.12 GRPO 一步训练完整流程

```
┌─────────────────────────────────────────────┐
│           GRPO 训练一步的流程                │
├─────────────────────────────────────────────┤
│                                             │
│  1. 对同一问题生成 N 个回答                  │
│           ↓                                 │
│  2. 计算每个回答的 reward                   │
│           ↓                                 │
│  3. advantages = reward - baseline          │
│     → critic/advantages                     │
│           ↓                                 │
│  4. 计算 policy gradient loss               │
│     → actor/loss                            │
│           ↓                                 │
│  5. 加入 entropy_loss（防止输出单一）        │
│     → actor/entropy_loss                    │
│           ↓                                 │
│  6. 反向传播，计算梯度                       │
│     → actor/grad_norm                       │
│           ↓                                 │
│  7. 梯度裁剪后更新参数                       │
│     → actor/pg_clipfrac（被clip的比例）      │
│                                             │
└─────────────────────────────────────────────┘
```

---

### 6.13 指标总览与记忆口诀

| 指标 | 前缀 | 正常范围 | ⚠️ 危险信号 |
|------|------|---------|------------|
| `loss` | actor/ | 持续下降，最终稳定 | 不降或剧烈震荡 |
| `kl_loss` | actor/ | 缓慢上升后稳定 | 持续快速增大 |
| `kl_coef` | actor/ | 固定值（如 0.001） | — |
| `lr` | actor/ | 固定或按预期衰减 | 提前归零 |
| `advantages/mean` | critic/ | 非零，有正有负 | ≈0 → 训练停滞 |
| `entropy` | actor/ | 0.5 ~ 1.5 | →0 坍塌 / 一直高不降 |
| `entropy_loss` | actor/ | 小负值 | 绝对值持续增大 |
| `grad_norm` | actor/ | 1.0 ~ 5.0 | 持续 >10 → 不稳定 |
| `pg_clipfrac` | actor/ | 0.1 ~ 0.3 | ≈0 → 参数不更新 |
| `pg_clipfrac_lower` | actor/ | 接近 0 | 频繁尖峰 → 回答被强烈抑制 |
| `max_memory_allocated` | — | 稳定 | 突降（异常）/ 持续上涨（泄漏） |

**记忆口诀：**

```
loss         = 成绩单（下降说明在学习，但要配合 reward 看）
kl_loss      = 离家距离（别跑太远，偏离参考模型）
kl_coef      = 皮带松紧（控制允许偏离多远）
lr           = 步伐速度（=0 则停走了）
advantages   = 相对排名（=0 则不知道往哪走）
entropy      = 体温计（量体温，观测多样性状态）
entropy_loss = 退烧药（主动干预，防止输出单一）
entropy_coef = 药量  （超参数，控制剂量）
grad_norm    = 步伐力度（太大会摔跤，太小走不动）
pg_clipfrac  = 刹车使用率（太低说明刹车失效/没在动）
```