# Verl

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

### 6.2 Advantages（优势值）

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

### 6.3 Entropy（熵）

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

### 6.4 Entropy Loss（熵损失）

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

### 6.5 Grad Norm（梯度范数）

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

### 6.6 GRPO 一步训练完整流程

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

### 6.7 指标总览与记忆口诀

| 指标 | 正常范围 | ⚠️ 危险信号 |
|------|---------|------------|
| `advantages/mean` | 非零，有正有负 | ≈0 → 训练停滞 |
| `entropy` | 0.5 ~ 1.5 | →0 坍塌 / 一直高不降 |
| `entropy_loss` | 小负值 | 绝对值持续增大 |
| `grad_norm` | 1.0 ~ 5.0 | 持续 >10 → 不稳定 |
| `pg_clipfrac` | 0.1 ~ 0.3 | ≈0 → 参数不更新 |

**记忆口诀：**

```
entropy      = 体温计（量体温，观测状态）
entropy_loss = 退烧药（主动干预训练方向）
entropy_coef = 药量  （超参数，控制剂量）
grad_norm    = 步伐大小（太大会摔跤，太小走不动）
advantages   = 相对排名（为0则不知道往哪走）
```