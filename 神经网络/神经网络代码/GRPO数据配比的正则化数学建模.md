# GRPO 数据配比的正则化数学建模

> **背景**：在一个图像有效性二分类任务（有效图/无效图）的 GRPO 训练实验中，发现"纯难样本训练→大盘指标不如 SFT baseline"的问题，通过引入大盘高置信度简单样本混合训练来修复，本文对这一直觉进行严格的数学建模，说明这本质上是一种正则化（regularization）。

---

## 数据集说明

| 类别 | 来源 | 数量 | 特点 |
|------|------|------|------|
| **B 类**（难样本） | 人工标注的高置信度 badcase | 1223 条 | 模型原本容易出错的边界/歧义 case |
| **E 类**（简单样本） | 大盘真实前 1 万条 + 双模型一致高置信度筛选 | 5139 条 | 模型本来就能稳定判对的 case |

> 最终混合比例：E 类约占总样本 **80%**（5139/6362）。

---

## 一、纯难样本训练：分布外优化陷阱

如果只用难样本（B 类）训练，目标函数（以 PPO/GRPO 的 surrogate objective 简化为期望 reward 形式）是：

$$\max_\theta \; \mathbb{E}_{x \sim D_\text{hard}}\left[\mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}\left[r(x, y)\right]\right]$$

其中 $D_\text{hard}$ 是难样本的经验分布（人工标注的、模型本来就容易出错的 badcase）。这是一个典型的**分布外优化陷阱**：$D_\text{hard}$ 天然是模型在大盘真实分布 $D_\text{real}$ 上的一个**有偏子集**（专挑模型犯错的困难 case），而不是 $D_\text{real}$ 的无偏采样。

用重要性采样的视角看，难样本相对真实分布的采样权重是：

$$w(x) = \frac{p_\text{hard}(x)}{p_\text{real}(x)} \gg 1 \quad \text{（当 } x \text{ 是"边界/易错"样本时）}$$

对易分/送分样本，$w(x) \approx 0$（因为难样本池根本不包含这些样本）。这意味着**纯难样本训练相当于对真实分布做了极端的重加权**，模型的梯度更新方向被这批"刁钻案例"主导，容易导致两个问题：

1. **决策边界被难样本的噪声/歧义过度"掰弯"**：过拟合到 badcase 的特异性模式，而非学习真正的判别规则
2. **灾难性遗忘**：在原本简单、模型本来判得很好的 case 上性能下降

> 这正是"350 step GRPO 大盘指标反而不如 SFT baseline"现象的数学解释。

---

## 二、混入简单样本后：带正则化约束的优化

现在数据源是 $D = D_\text{hard} \cup D_\text{easy}$，目标函数变为：

$$\max_\theta \; \underbrace{\mathbb{E}_{x \sim D_\text{hard}}\left[\mathbb{E}_y\left[r(x,y)\right]\right]}_{\text{任务损失（改错）}} \; + \; \lambda \cdot \underbrace{\mathbb{E}_{x \sim D_\text{easy}}\left[\mathbb{E}_y\left[r(x,y)\right]\right]}_{\text{正则化项（保持/巩固已有能力）}}$$

这里 $\lambda$ 可以用数据配比做**一阶近似**估计——E 类约占 80%，大致相当于给"保持大盘分布一致性"这一项赋予了很大的权重。但严格来说，真实的 $\lambda$ 还受到学习率、每个 mini-batch 内 E/B 样本的实际比例、以及两类样本 reward 量级是否对齐等因素共同影响，配比只是其中最直观的近似。这个形式和经典正则化的通用范式**结构上同构**：

$$\mathcal{L}(\theta) = \mathcal{L}_\text{task}(\theta) + \lambda \cdot \Omega(\theta)$$

只不过这里的正则化项 $\Omega(\theta)$ 不是直接约束参数范数（像 L2/权重衰减那样），而是一种**分布匹配式的数据正则化（data-driven regularization / distributional constraint）**。

---

## 三、三种视角下的正则化解读

### 3.1 类比持续学习中的"防遗忘"手段（EWC vs Experience Replay）

#### EWC 是什么

**EWC（Elastic Weight Consolidation，弹性权重巩固）** 是 2017 年 DeepMind 提出的持续学习算法，用来解决神经网络学新任务时"忘掉"旧任务的**灾难性遗忘**问题。核心思路是：不是所有参数对旧任务都同等重要，如果某个参数对旧任务的输出影响很大，学新任务时就应该约束它别变太多：

$$\mathcal{L}(\theta) = \mathcal{L}_\text{new}(\theta) + \frac{\lambda}{2} \sum_i F_i \left(\theta_i - \theta^*_{\text{old},i}\right)^2$$

其中 $F_i$ 是 Fisher 信息矩阵的对角元素，代表参数 $\theta_i$ 对旧任务的重要程度。名字里的"弹性（Elastic）"来自这里——像弹簧一样，越往远拉惩罚越大，且不同参数的弹性系数不同。

#### E 类样本的作用更接近 Experience Replay，而非 EWC

两者都在解决"学新任务时防止旧能力被遗忘"这个问题，但**机制有本质差异**：

| | EWC | E 类样本混入（本方案） |
|---|---|---|
| 约束施加在哪里 | **参数空间**：直接在 loss 里加项，约束 $\theta_i$ 不要偏离 $\theta^*_\text{old}$ | **数据空间**：通过让模型在简单样本上持续拿到 reward，间接地约束参数更新方向 |
| 需要显式计算什么 | 需要计算 Fisher 信息矩阵（每个参数的重要程度） | 不需要，直接往 dataloader 里混入数据 |
| 约束的精细程度 | 每个参数单独有权重 $F_i$，精细 | 整体平均的约束，无法区分哪些参数更重要，粗粒度 |
| 更准确的名字 | Elastic Weight Consolidation | **Experience Replay / Rehearsal（经验回放）** |

所以 E 类样本的作用，和 EWC 的**目标相似**（都防灾难性遗忘），但**机制不同**——它更接近持续学习里的 experience replay：把"旧任务的代表性样本"加入训练，通过直接的梯度信号把旧能力"保持住"，而不是通过显式的参数约束项来"锚定"参数。这是防止灾难性遗忘最朴素也最有效的手段之一。

---

### 3.2 类比信任域 / KL 约束（PPO 本身的机制 + 数据层面的二次约束）

PPO/GRPO 本身在 loss 里就有一项防止新策略离旧策略太远：

$$\mathcal{L}_\text{PPO} = \mathbb{E}\left[\min\left(\rho_t \hat{A}_t,\; \text{clip}(\rho_t, 1{-}\epsilon, 1{+}\epsilon)\hat{A}_t\right)\right] - \beta\, \text{KL}\left[\pi_\theta \| \pi_\text{ref}\right]$$

这是**参数/策略层面**的信任域约束（管"更新步子别太大"）。而简单样本混合是在**数据分布层面**加了一层信任域约束——用真实大盘分布的高置信度样本去校准/约束策略更新的方向（管"更新方向别跑偏出真实数据分布"）。两者互补：

```
PPO KL 约束          → "更新步子别太大"（策略层面）
简单样本混合正则化   → "更新方向别跑偏"（数据分布层面）
```

---

### 3.3 类比混合先验 / 数据增强式正则化

如果把"训练数据的经验分布"本身看作模型的一种隐式先验，那么：

$$p_\text{train}(x) = (1-\alpha) \cdot p_\text{hard}(x) + \alpha \cdot p_\text{easy}(x), \quad \alpha \approx 0.8$$

这就是一个**混合先验（mixture prior）**，用来对冲 $p_\text{hard}$ 的极端偏置。如果我们定义模型在 $D_\text{easy}$ 上的期望损失为衡量"是否偏离真实分布"的代理指标，那么：

$$\Omega(\theta) = \mathbb{E}_{x \sim D_\text{easy}}\left[-r(x, \pi_\theta)\right]$$

就是一个具体的、可计算的正则化惩罚项——**模型在简单样本上表现变差，本身就是"过拟合难样本、偏离真实分布"的信号**，这一项会通过训练中的梯度直接抑制这种偏离。

---

## 四、"E 类答错重罚 ×0.4"的数学角色

在简单样本答错时施加 $\kappa = 0.4$ 的重罚系数，相当于把正则化项非对称地放大了：

$$r'(x, y) = \begin{cases} r(x, y) & x \in D_\text{hard} \\ r(x, y) & x \in D_\text{easy},\; \text{judge}(y) = \text{correct} \\ \kappa \cdot r(x, y) & x \in D_\text{easy},\; \text{judge}(y) = \text{wrong} \end{cases} \quad (\kappa = 0.4 < 1)$$

这在效果上等价于把正则化项的惩罚方向做了**单边加权**——用更陡峭的梯度惩罚"模型在应该稳固的区域发生退化"，这在优化理论里接近于**非对称/加权风险最小化（asymmetric risk minimization）**。

类比 **Focal Loss**：
- Focal Loss：放大**难分类**样本的权重，让梯度更集中地修正"容易分错的 case"
- 本方案：放大**简单样本犯错**时的惩罚权重，让梯度更集中地修正"不该退化的地方退化"

两者的共同点：都是让 loss 函数对某类**特别不该发生的错误更敏感**，从而让梯度更集中地修正这类错误。

---

## 五、统一公式：完整的正则化目标函数

综合来看，整个训练目标可以写成一个统一的正则化风险表达式：

$$J(\theta) = \underbrace{\mathbb{E}_{x \sim D_\text{hard}}\left[\mathbb{E}_{y \sim \pi_\theta}\left[r(x, y)\right]\right]}_{\text{任务损失（改错，主目标）}} + \lambda \underbrace{\mathbb{E}_{x \sim D_\text{easy}}\left[\mathbb{E}_{y \sim \pi_\theta}\left[r(x, y) \cdot \left(1 - (1-\kappa) \cdot \mathbf{1}[\hat{y} \neq y^*]\right)\right]\right]}_{\text{正则化损失（防遗忘，带非对称重罚）}}$$

其中：

| 符号 | 含义 |
|------|------|
| 第一项 | 难样本任务损失，负责"改错"（训练的主目标） |
| 第二项 | 简单样本正则化损失，$\lambda$ 可以用数据配比（约 4:1 的 E:B 条数比）做一阶近似，实际上还受学习率、mini-batch 构成、两类样本 reward 量级等因素共同影响，负责"防遗忘/防止分布漂移" |
| $\kappa = 0.4$ | 非对称惩罚系数，注意这是在 **reward shaping 层面**做的修改（修改 $r'$ 而非直接改 GRPO loss），通过改变输入给调度器的 reward 数值来间接影响 advantage 和梯度方向；给正则化项加了"陡峭墙"——一旦模型在简单样本上开始退化，负反馈比在难样本上单纯没学会更强烈 |

> ⚠️ **注意**：第四章里的 $\kappa$ 重罚是在 **reward shaping** 层面（改 $r'$）实施的，不是在 GRPO loss 函数里直接加权项。GRPO 的 loss 接收的是（经过 reward shaping 后的）advantage，所以上面的统一公式把两者揉在一起展示时，应理解为"对第二项的 reward 做了非对称缩放后再输入 GRPO 的标准 loss 计算"，而不是在 loss 函数的结构上做了改动。

---

## 六、为什么不加 hint

> **结论**：这一轮先不加 hint，先跑一版看效果。原因如下：

**① hint 副作用已被证实**：之前"350step + hint 推理"实验已经证明，加 hint 会让指标变差（增加解析失败率、没有提升 F1）。虽然那次是"训练不加、推理加"的 mismatch 场景，但也说明 hint 文本本身对模型输出的干扰不是完全无害的"锦上添花"——一旦引入就要背负"训练/推理必须完全一致"的枷锁，且长期还得考虑"退火"（后期去掉 hint）这个额外的复杂度和风险点（退火时机把握不好同样会造成 train-inference mismatch）。

**② 本轮 advantage 塌陷的根源已不同**：Hint 是针对"纯难样本 → 模型输出高度一致 → 组内方差趋零"设计的。但本次数据里超过 80%（5139/6362）是 E 类简单样本——这些样本模型大概率能稳定判对（4 条 rollout 一致给对的答案），组内方差确实趋近于 0，但这是**"好"的趋零**（模型学会了正确答案，reward 稳定拿高分），而非"坏"的趋零。GRPO 在简单样本上 advantage ≈ 0 是正常且期望的现象，不需要用 hint 去人为制造方差。

```
简单样本 advantage ≈ 0  → 好的趋零 ✅ 不需要 hint
难样本   advantage ≈ 0  → 坏的趋零 ❌ 才需要 hint（但先观察是否真的发生）
```

**③ 和"防止过拟合"的目标冲突**：核心思路是用简单样本做正则化防止模型在难样本上过拟合。如果又在难样本上加 hint，相当于直接把部分答案线索喂给模型，是**信息泄漏式的"抄近道"**，会削弱难样本本该提供的"独立解决陌生/边界案例"的训练信号：

```
简单样本的正则化 = "从简单样本学到稳定先验，防止对难样本死记硬背"
难样本 hint     = "给难样本发提示条让模型抄"

两者同时用 → 容易让模型学到"看到提示就走某条路径，没提示就乱猜"的坏习惯
```

**后续判断节点**：如果训练日志里观察到难样本子集依然出现 advantage 塌陷（组内 4 条 rollout 输出高度一致，无论对错），再考虑对**难样本单独开启 hint**（而不是全量开），这样可以做一个更干净的消融实验，也避免"简单样本 + hint"这种没必要的组合引入新的不确定性。

---

## 七、一句话总结

> 混入简单样本的直觉是完全站得住脚的数学直觉——本质上是用真实分布的高置信度样本作为**经验回放（replay buffer）**，对"难样本驱动的策略更新"施加一个**分布一致性约束**；而新增的重罚机制（× 0.4）则是把这个约束从"普通惩罚"升级为"**非对称的陡峭惩罚**"，用来更强力地抑制"模型在不该退化的地方退化"这个具体失败模式。
>
> 这在机器学习里有一个更精确的名字：**Constrained Policy Optimization with Replay-based Regularization**（基于经验回放的约束策略优化），是持续学习（continual learning）与强化学习结合场景下的标准做法——RLHF 里常见的"防止 reward hacking 导致能力退化"也用类似套路（混入原始 SFT 数据 / KL 惩罚来防止策略跑偏）。
