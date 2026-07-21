# 手写 DPO / PPO / GRPO 核心 Loss 代码合集

> LLM 对齐阶段三种主流强化学习/偏好优化算法的手写核心代码：DPO（跳过显式 Reward Model）、PPO（经典 Actor-Critic + GAE）、GRPO（用组内相对 reward 替代 Critic）。
>
> 关联笔记：
> - 公式推导与三者对比：[GRPO与PPO算法详解](../../深度学习/GRPO与PPO算法详解.md)
> - 对齐整体流程（SFT → RM → RL）：[对齐](../../深度学习/对齐.md)
> - MiniMind 中的完整实践代码：[MiniMind从0到1构建大模型 · 4.3 强化学习](../MiniMind从0到1构建大模型.md#43-强化学习)（含 [4.3.3 DPO](../MiniMind从0到1构建大模型.md#433-dpo跳过显式奖励模型和在线-rollout)）

---

## 一、DPO（Direct Preference Optimization）

DPO 的核心思路：**跳过显式的 Reward Model 和在线 rollout**，直接用"偏好对"（chosen/reject 两个回答）训练策略模型，本质上是把 RLHF 的目标函数化简后变成了一个类似分类的 loss。

```python
import torch
from torch import nn
from torch.nn import functional as F

class DPO:
    def __init__(self, beta):
        self.beta = beta

    def dpo_loss(self, policy_chosen_logps, policy_reject_logps, ref_chosen_logps, ref_reject_logps):
        # 策略模型相对参考模型，在 chosen/reject 回答上的 log 概率提升量
        chosen_r = policy_chosen_logps - ref_chosen_logps
        reject_r = policy_reject_logps - ref_reject_logps

        # chosen 相对 reject 的"隐式 reward 差"越大，loss 越小
        loss = -F.logsigmoid(self.beta * (chosen_r - reject_r))
        return loss.mean()
```

**关键点回顾**：

- `policy_chosen_logps` / `policy_reject_logps`：当前训练中的策略模型，在 chosen（人类偏好的回答）和 reject（人类不喜欢的回答）上的对数概率（一般是整个回答所有 token 的 log prob 求和）
- `ref_chosen_logps` / `ref_reject_logps`：冻结的参考模型（通常是 SFT 后的初始模型）在同样两个回答上的对数概率，用来防止策略模型跑偏太远（隐式起到类似 KL 约束的作用）
- `chosen_r - reject_r`：这个差值越大，说明策略模型比参考模型更倾向于"多提升 chosen、少提升 reject"，即学到了正确的偏好方向
- `-log(sigmoid(β·Δ))` 本质上是一个 Bradley-Terry 偏好模型的极大似然 loss，`β` 控制对偏好差异的敏感程度（类似温度系数）
- **没有 Reward Model，没有 Critic，没有在线采样**，只需要离线准备好偏好数据对，训练稳定、显存开销小，这是 DPO 相比 PPO/GRPO 最大的工程优势

---

## 二、PPO（Proximal Policy Optimization）

PPO 是经典的 Actor-Critic 强化学习算法，核心包含三部分：用 **GAE** 估计优势函数、**clip 截断**防止策略更新步子过大、以及配套的 **Value Loss** 训练 Critic。

```python
import torch
from torch import nn
from torch.nn import functional as F

class PPO:
    def __init__(self, clip=0.2, gamma=1, lam=0.95):
        self.clip = clip
        self.gamma = gamma
        self.lam = lam

    def mask_mean(self, loss, mask, dim=-1):
        # 只对有效 token（mask=1）取平均，padding 部分（mask=0）不计入
        return (loss * mask).sum(dim=dim) / mask.sum(dim=dim)

    def advantage_estimate(self, rewards, values):
        ''' GAE 广义优势估计 '''
        seq_len = values.shape[1]
        advantages = torch.zeros_like(rewards)
        gae = 0
        # 从序列末尾往前递推，符合 GAE 的时序依赖关系
        for i in range(seq_len - 1, -1, -1):
            next_value = values[:, i + 1] if i < seq_len - 1 else 0.0
            delta = rewards[:, i] + self.gamma * next_value - values[:, i]
            gae = delta + self.lam * self.gamma * gae
            advantages[:, i] = gae
        returns = advantages + values
        return advantages, returns

    def policy_loss(self, new_probs, old_probs, advantages, act_mask):
        # 新旧策略的概率比（对数域相减再 exp，等价于概率相除）
        ratio = torch.exp(new_probs - old_probs)
        surr1 = ratio * advantages
        # clip 截断：防止 ratio 偏离 1 太远，从而限制单次更新幅度
        surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
        # 取两者较小值（悲观下界），是 PPO-Clip 的核心
        loss = -torch.min(surr1, surr2)
        return self.mask_mean(loss, act_mask)

    def value_loss(self, new_values, returns, act_mask):
        # Critic 的均方误差 loss：让 value 预测尽量逼近实际 return
        loss = (new_values - returns) ** 2
        return self.mask_mean(loss, act_mask)
```

**关键点回顾**：

- **`mask_mean`**：LLM 场景下序列长度不一致，需要用 `act_mask` 屏蔽 padding 部分，只对真实生成的 token 计算均值
- **`advantage_estimate`（GAE）**：
  - `delta` 是单步 TD 误差：`reward + γ·V(下一步) - V(当前步)`
  - `gae` 用 `λ` 做指数加权递归累加，是"单步 TD"和"蒙特卡洛回报"之间的平滑折中（`λ=0` 退化为单步 TD，`λ=1` 退化为蒙特卡洛）
  - 循环必须**从后往前**算，因为 `gae` 的递推依赖后一步的结果
  - `returns = advantages + values`：还原出真实的目标回报，用于训练 Critic
- **`policy_loss`（PPO-Clip 核心）**：
  - `ratio = exp(new_logp - old_logp)`：新旧策略在同一个动作上的概率比
  - `surr1` 是原始的 policy gradient 目标；`surr2` 是裁剪后的版本
  - 取 `min(surr1, surr2)` 保证：当 `advantage > 0` 时（好动作），ratio 增大到一定程度后收益就不再增加（防止过度更新）；当 `advantage < 0` 时同理起到限制作用
- **`value_loss`**：Critic 是独立于策略网络的一个价值预测头，用最简单的 MSE 训练，让它更准确地估计每个状态的期望回报，从而给出更准的 Advantage

---

## 三、GRPO（Group Relative Policy Optimization）

GRPO 是 DeepSeek 提出的 PPO 变体，核心改进：**去掉 Critic 网络**，改用"对同一个问题采样一组（group）回答，用组内 reward 的均值/标准差归一化"来代替 Critic 估计的优势函数，同时把 KL 约束直接写进 loss（不需要额外的参考策略网络做单独的价值预测）。

```python
import torch

class GRPO:
    def __init__(self, eps, clip, beta):
        self.eps = eps      # 防止除零的小常数
        self.clip = clip    # PPO 式 clip 范围
        self.beta = beta    # KL 惩罚系数

    def mask_mean(self, loss, mask, dim=-1):
        return (loss * mask).sum(dim=dim) / mask.sum(dim=dim)

    def group_advantage(self, rewards):
        # 组内归一化：用同一 prompt 采样出的 G 个回答的 reward 均值/标准差做归一化
        mean = torch.mean(rewards)
        std = torch.std(rewards, unbiased=False)
        advantages = (rewards - mean) / (std + self.eps)
        # detach：advantage 只作为常数系数参与梯度计算，本身不需要梯度
        return advantages.detach()

    def grpo_loss(self, old_logps, new_logps, ref_logps, advantages):
        # 新旧策略概率比
        ratio = torch.exp(new_logps - old_logps)

        # k3 估计的 KL 散度（无偏、方差更小的 KL 近似估计量）
        ref_logr = new_logps - ref_logps
        kl_score = (torch.exp(ref_logr) - ref_logr - 1) * self.beta

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages

        # pg_loss 和 KL 惩罚项写在同一个求和里面，一起取平均
        loss = -torch.mean(torch.min(surr1, surr2) - kl_score)
        return loss
```

**关键点回顾**：

- **`group_advantage`**：GRPO 不需要 Critic，而是对同一个 prompt 采样出 G 个不同回答，用这一组回答的 reward 做标准化（z-score），效果好的回答得到正的 advantage，效果差的得到负的，**同一个回答内所有 token 共享一个 advantage**（这一点和 PPO 的逐 token advantage 不同，可对照 [GRPO与PPO算法详解 · 三、GRPO 每个 token 的变量](../../深度学习/GRPO与PPO算法详解.md#三grpo-每个-token-的变量)）
- **`.detach()`**：advantage 只是加权系数，不应该参与反向传播产生梯度，否则会导致梯度计算错误
- **KL 散度的估计方式（k3 estimator）**：`kl_score = exp(Δ) - Δ - 1`，其中 `Δ = new_logp - ref_logp`。这是一个无偏且低方差的 KL 近似估计（相比直接用 `Δ` 本身作为 KL 估计更稳定），保证策略更新时不会偏离参考模型（通常是 SFT 模型）太远
- **`policy_loss` 与 `kl_score` 合并计算**：和 PPO 把 KL 放在整体 loss 外面加权不同，GRPO 把 KL 放在 `Σ_t` 求和号里面，与 pg_loss 逐 token 相加后再统一取平均（这一点在 [GRPO与PPO算法详解 · PPO 与 GRPO 中括号的区别](../../深度学习/GRPO与PPO算法详解.md#ppo-与-grpo-中括号的区别) 中有详细的公式对照）
- **没有 Critic 网络**：省掉了一整个价值网络的训练和显存开销，这是 GRPO 相比 PPO 最大的工程简化，代价是需要对同一个 prompt 采样多个回答（group），依赖组内相对好坏来估计优势

---

## 四、三种算法对比速查

| 维度 | DPO | PPO | GRPO |
|------|-----|-----|------|
| 是否需要 Reward Model | ❌ 不需要（用偏好对直接训练） | ✅ 需要 | ✅ 需要 |
| 是否需要 Critic（价值网络） | ❌ | ✅ 必须有 | ❌（用组内统计量代替） |
| 是否需要在线 rollout（采样） | ❌ 离线偏好数据即可 | ✅ 需要在线采样 | ✅ 需要在线采样（且需组内多个样本） |
| Advantage 计算方式 | 无显式 advantage | GAE（Critic 估计，逐 token） | 组内 reward 归一化（同回答内共享） |
| KL 约束位置 | 隐式在 loss 公式中体现 | 通常在 loss 外面单独加权 | 在 loss 求和号内部逐 token 相加 |
| 训练稳定性/显存开销 | 最低 | 最高（需要同时维护 Actor + Critic + Ref + Reward 四个模型） | 中等（省了 Critic，但需要多次采样） |

> 一句话总结：DPO 用离线偏好数据绕开了整个 RL 采样流程，最简单最省资源；PPO 是最经典最稳健的在线 RL 方案，但要维护 4 个模型开销最大；GRPO 保留了 PPO 的 clip 机制，但用"组内相对排名"替代 Critic，是 DeepSeek-R1 等推理模型训练中广泛采用的折中方案。
