# GRPO 与 PPO 算法详解

> 关联笔记：这些公式在 MiniMind 强化学习实践中的完整代码与统一视角，见 [MiniMind从0到1构建大模型 · 4.3 强化学习](../神经网络代码/MiniMind从0到1构建大模型.md#43-强化学习)。

---

## 一、三种算法总览

### SFT / GRPO / PPO 完整对比

| 变量 | SFT | GRPO | PPO |
|------|-----|------|-----|
| $r_t$（ratio） | ❌ 没有 | ✅ token 级别 | ✅ token 级别 |
| $A$（advantage） | ❌ 没有 | ✅ **回答级别**（同回答内共享） | ✅ **token 级别**（每个不同） |
| $\text{KL}$ | ❌ 没有 | ✅ token 级别（近似） | ✅ token/序列级别 |
| Critic 网络 | ❌ | ❌ | ✅ 必须有 |
| 监督信号来源 | 标注数据 | 多个回答的相对 reward | Critic 估计的状态价值 |
| 显存开销 | 低 | 中 | 高（两个大模型） |

---

## 二、各算法 Loss 完整公式

### SFT Loss

$$\mathcal{L}_\text{SFT} = -\frac{1}{N}\sum_{i=1}^{N} \frac{1}{T_i}\sum_{t=1}^{T_i} \log P_\theta(y_t^{(i)} \mid x^{(i)}, y_{<t}^{(i)})$$

每个 token 算一个 $-\log P$，先在序列内对 token 取平均，再在 batch 内对样本取平均。

---

### PPO Loss

$$\mathcal{L}_\text{PPO} = -\frac{1}{T}\sum_{t=1}^{T} \min\left(r_t \cdot A_t,\ \text{clip}(r_t, 1{-}\varepsilon, 1{+}\varepsilon) \cdot A_t\right) + \beta \cdot \text{KL}$$

其中 KL 是整个序列平均后的标量，加在求和**外面**。

---

### GRPO Loss（完整）

$$\mathcal{L}_\text{GRPO}(\theta) = -\frac{1}{G}\sum_{g=1}^{G} \frac{1}{T_g}\sum_{t=1}^{T_g} \left[\min\left(r_{g,t}\cdot\hat{A}_g,\ \text{clip}(r_{g,t}, 1{-}\varepsilon, 1{+}\varepsilon)\cdot\hat{A}_g\right) - \beta\cdot\text{KL}_{g,t}\right]$$

各项定义：

$$r_{g,t} = \frac{\pi_\theta(y_{g,t} \mid x,\ y_{g,<t})}{\pi_{\theta_\text{old}}(y_{g,t} \mid x,\ y_{g,<t})}, \quad \hat{A}_g = \frac{R_g - \text{mean}(R_{1..G})}{\text{std}(R_{1..G})}$$

$$\text{KL}_{g,t} = \frac{\pi_\theta(y_{g,t})}{\pi_\text{ref}(y_{g,t})} - \log\frac{\pi_\theta(y_{g,t})}{\pi_\text{ref}(y_{g,t})} - 1$$

GRPO 的 KL 是每个 token 单独算的，放在求和**里面**（中括号内），和 pg_loss 一起被 $\sum_t$ 求和。

---

### PPO 与 GRPO 中括号的区别

数学上两者等价，写法不同：

```
PPO 写法：
  先算 pg_loss 均值，再加 KL 均值×β
  = (1/T)Σ pg_loss_t  +  β × (1/T)Σ KL_t

GRPO 写法：
  两项合并进一个求和
  = (1/T)Σ [pg_loss_t - β × KL_t]

展开完全一样，GRPO 的中括号只是说"这两项都要一起被 Σ 求和"
```

---

## 三、GRPO 每个 token 的变量

### 每个 token 里什么一样、什么不一样

| 变量 | 每个回答不同？ | 每个 token 不同？ | 说明 |
|------|------------|----------------|------|
| $r_{g,t}$ | ✅ 是 | ✅ 是 | 每个位置预测的词不同，概率比不同 |
| $\hat{A}_g$ | ✅ 是 | ❌ **否** | 同一回答内所有 token **共享**一个 A |
| $\text{KL}_{g,t}$ | ✅ 是 | ✅ 是 | 每个位置的概率不同，KL 也不同 |

**展开看一个回答内部：**

```
回答 g，A_g = +1.2（整个回答共享）

token1: r=1.05  → min(1.05×1.2, clip×1.2) - β×KL1
token2: r=0.87  → min(0.87×1.2, clip×1.2) - β×KL2
token3: r=1.23  → min(1.23×1.2, clip×1.2) - β×KL3
token4: r=0.95  → min(0.95×1.2, clip×1.2) - β×KL4

A 全是 1.2，但 r 和 KL 每个 token 都不一样
```

---

### $r_{g,t}$ 和 KL 里的 $u$ 的区别

两个公式形式相似，但**参照物完全不同**：

| | $r_{g,t}$（ratio） | KL 里的 $u$ |
|--|--|--|
| 公式 | $\dfrac{\pi_\theta(y_{g,t})}{\pi_{\theta_\text{old}}(y_{g,t})}$ | $\dfrac{\pi_\theta(y_{g,t})}{\pi_\text{ref}(y_{g,t})}$ |
| 分子 | 当前训练中的模型 $\pi_\theta$ | 当前训练中的模型 $\pi_\theta$ |
| 分母 | **上一轮**的模型 $\pi_{\theta_\text{old}}$（会更新） | **原始基座**模型 $\pi_\text{ref}$（永远冻结） |
| 衡量什么 | 这一步更新走了多远 | 总共训练了多远（离出发点） |
| 目的 | 防止单步更新太大（clip） | 防止遗忘原始能力（KL 惩罚） |

**生活类比：**

```
你在学一门新技能：

r：今天的你 vs 昨天的你
  → 今天学了多少？别一天学太猛（clip 截断）

u（KL）：今天的你 vs 入学前的你
  → 有没有忘掉基础？别跑太偏（KL 惩罚）

两个是完全不同的参照物！
```

---

## 四、PPO：r 和 off-policy 重复使用

### 为什么要重复使用同一批数据？

```
生成一条回答 → 很贵（要跑完整的前向传播）
算一次梯度  → 相对便宜

如果生成一次只用一次：
  生成 : 更新 = 1 : 1  ← 浪费！

PPO 的做法：生成一次，用 K 次
  生成 : 更新 = 1 : K  （K 通常为 4~8）
```

### 具体例子（数值）

**第 0 步：用当前模型采样，固定为 π_old**

```
问题：2 + 2 = ?

生成了 G=3 个回答：
  回答1："4"      reward = 1.0  → A1 = +1.2
  回答2："five"   reward = 0.0  → A2 = -0.8
  回答3："4.0"    reward = 0.5  → A3 = -0.4

π_old("4" | 问题) = 0.60   ← 记住，后面重复使用时不动！
```

**第 1 次梯度更新（用上面同一批数据）**

```
r = π_θ / π_old = 0.60 / 0.60 = 1.00  ← 没变

做 backward，更新参数
π_θ("4") 变成 0.65
```

**第 2 次梯度更新（还是同一批数据！）**

```
π_old("4") 还是 0.60  ← 不动
π_θ("4")   现在 0.65

r = 0.65 / 0.60 = 1.083  ← 稍微偏离了，在 clip 范围内

继续更新，π_θ("4") 变成 0.70
```

**第 4 次梯度更新（危险了）**

```
π_old("4") 还是 0.60
π_θ("4")   现在 0.75

r = 0.75 / 0.60 = 1.25  ← 超过 clip 上限 1.2！

min(r·A, clip(r)·A) = min(1.25×1.2, 1.2×1.2) = min(1.50, 1.44) = 1.44

梯度被 clip 削弱 → "你已经更新够多了，别再猛推"
```

### 三阶段循环

```
┌──────────────────────────────────────────────────┐
│  Phase 1：采样（Rollout）                         │
│    用 π_old 生成 G 个回答                         │
│    算每个回答的 reward → advantage                │
│    π_old 的概率全部记录，固定不动                  │
├──────────────────────────────────────────────────┤
│  Phase 2：多次更新（Training，重复 K 次）          │
│    每次用同一批回答                               │
│    r = π_θ(当前) / π_old(固定)                   │
│    r 偏离 1 太多 → clip 截断，防止更新过猛          │
│    做 K 次梯度更新                                │
├──────────────────────────────────────────────────┤
│  Phase 3：更新 π_old                             │
│    π_old ← π_θ（当前模型快照）                   │
│    回到 Phase 1，重新采样                         │
└──────────────────────────────────────────────────┘
```

**r 的作用一句话：**

```
r = 1.0   → 没变，正常用 advantage 更新
r = 1.3   → 超了，clip 截断，梯度变小，"别再往这推了"
r = 0.7   → 往反方向跑，clip 保护

π_old 是"基准锚点"，r 衡量你偏离多远，
clip 保证每次不走太远 → 可以安全地把同一批数据用 K 次
```

---

## 五、PPO 的 Advantage：V(s)、δ 和 GAE

### 为什么需要 V(s) 和 Advantage？

```
模型生成了一句话：
"The answer is 42 ."
  t1    t2  t3  t4  t5

最后得到 reward R = 1.0（答对了）

问题：这个 reward 是谁的功劳？
  是 "The" 的功劳？是 "42" 的功劳？
  PPO 需要知道每个 token "值多少钱" → 这就是 V(s_t) 的作用
```

### 核心概念

**状态 $s_t$：**

```
t=1：s1 = [问题 x]
t=2：s2 = [问题 x, "The"]
t=3：s3 = [问题 x, "The", "answer"]
t=4：s4 = [问题 x, "The", "answer", "is"]

状态 = 当前已经生成的所有内容（前缀）
```

**状态价值 $V(s_t)$：**

$$V(s_t) = \text{从状态 } s_t \text{ 出发，未来能拿到的期望 reward}$$

```
V(s1) = 0.50，V(s2) = 0.55，V(s3) = 0.60，V(s4) = 0.80

V 由 Critic 网络预测，是一个独立的神经网络
```

**语言模型里 reward 怎么分配：**

```
"The  answer  is   42   ."
  r1=0  r2=0  r3=0 r4=0 r5=1.0

中间 token 即时 reward = 0，最后一个 token 才给 reward
```

### TD Error δ

$$\delta_t = r_t + \gamma \cdot V(s_{t+1}) - V(s_t)$$

```
直觉：
  你预测从 s3 出发能拿 0.60
  实际走了一步发现 s4 值 0.80
  → s3 比预期更好！

  δ3 = (r3 + V(s4)) - V(s3) = (0 + 0.80) - 0.60 = +0.20

δ > 0 → 比预期好
δ < 0 → 比预期差
```

### GAE 公式（广义优势估计）

$$A_t^{\text{GAE}} = \sum_{k=0}^{T-t} (\gamma\lambda)^k \cdot \delta_{t+k} = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$$

离当前越远的 δ，权重越小（指数衰减），λ 控制偏差-方差权衡。

**具体数值例子（γ=0.99，λ=0.95，γλ=0.9405）：**

```
δ5=+0.100, δ4=+0.091, δ3=+0.192, δ2=+0.044, δ1=+0.045

A5 = 0.100
A4 = 0.091 + 0.9405×0.100 = 0.185
A3 = 0.192 + 0.9405×0.185 = 0.366
A2 = 0.044 + 0.9405×0.366 = 0.388
A1 = 0.045 + 0.9405×0.388 = 0.410
```

| token | $r_t$ | $V(s_t)$ | $\delta_t$ | $A_t^{\text{GAE}}$ | 含义 |
|-------|---------|-----------|-------------|---------------------|------|
| The | 0 | 0.50 | +0.045 | **+0.410** | 开头就走对了方向 |
| answer | 0 | 0.55 | +0.044 | **+0.388** | 也不错 |
| is | 0 | 0.60 | +0.192 | **+0.366** | 明显往好的方向走 |
| 42 | 0 | 0.80 | +0.091 | **+0.185** | 关键 token，贡献大 |
| . | 1.0 | 0.90 | +0.100 | **+0.100** | 拿到 reward |

> $A_t > 0$：这个 token 的选择比预期好 → 增大它的概率  
> $A_t < 0$：这个 token 的选择比预期差 → 减小它的概率

---

## 六、GRPO 为什么能省掉 Critic 网络？

```
PPO：
  需要 Critic 网络逐 token 估 V(s_t)
    → 算 δ_t → 算 GAE → 得到每个 token 的 A_t
    → Critic 网络和 Actor 一样大 → 显存翻倍

GRPO：
  直接生成 G 个回答，比较它们的 reward
    → A_g = (R_g - mean(R)) / std(R)
    → 同一回答内所有 token 共享这个 A_g
    → 不需要 Critic 网络，省掉一半显存

代价：
  A 的粒度从 token 级别 → 回答级别
  没法区分同一回答里哪个 token 贡献大
```

**GRPO 的核心创新：** 用"组内相对排名"代替"逐 token 价值估计"，砍掉了 Critic 网络。

### PPO vs GRPO 流程对比

```
PPO 流程：
  生成回答
    → Critic 估计每个 token 的 V(s_t)
    → 算 δ_t = r_t + γV(s_{t+1}) - V(s_t)
    → 算 A_t（GAE，从后往前）
    → 每个 token 用自己独立的 A_t 更新

GRPO 流程：
  生成 G 个回答
    → 每个回答算一个 reward R_g
    → A_g = (R_g - mean) / std
    → 同回答内所有 token 共享 A_g 更新
```

> **核心差异：** PPO 的 Advantage 是 **token 级别的精细估计**，GRPO 的 Advantage 是**回答级别的粗粒度估计**，用简单换省钱。

---

## 七、KL 散度惩罚项详解

### 为什么要有 KL 惩罚？

```
训练前：模型 π_ref（参考模型，冻结不动）
训练中：模型 π_θ（正在被更新）

如果没有 KL 惩罚：
  π_θ 可能为了拿高 reward
  → 说一些奇怪的话（reward hacking）
  → 或者完全忘了原来的语言能力

KL 惩罚的作用：
  让 π_θ 不要跑得离 π_ref 太远
  → 保持语言能力，防止 reward hacking
```

### 标准 KL 散度

$$\text{KL}(P \| Q) = \sum_{v \in \mathcal{V}} P(v) \log \frac{P(v)}{Q(v)}$$

需要对**全词表**求和，vocab_size 通常 50000~150000：

```
每个 token 位置，标准 KL 需要：
  1. 跑完整 softmax → 得到 50000 个概率
  2. 对每个词计算 π_θ(v)·log(π_θ(v)/π_ref(v))
  3. 全部加起来

计算量 = seq_len × vocab_size × 2次 forward → 显存爆炸
```

### GRPO 用的近似 KL（k3 形式）

令 $u = \dfrac{\pi_\theta(y_{g,t})}{\pi_\text{ref}(y_{g,t})}$（只需要**实际生成的那个 token** 的概率）：

$$\text{KL}_{g,t} = u - \log u - 1$$

**为什么只看实际生成的词就够？**

这里用了**蒙特卡洛估计**：

```
标准 KL 本质上是一个期望：
  KL = E_{v~π_θ}[ log(π_θ(v)/π_ref(v)) ]
     = Σ π_θ(v) · log(π_θ(v)/π_ref(v))    ← 对所有词加权求和

近似做法：
  从 π_θ 采样一个词（= 模型实际生成的词 y_t）
  用这个词的 f(u) 估计期望

  单次：不精确（就像扔一次骰子估计期望）
  多次平均：趋近真实 KL（就像扔 1000 次骰子）

训练很多 step，遇到很多不同的词
→ 这些近似值的平均收敛到真实 KL
→ 这就是"无偏估计"的含义
```

**三种 KL 变体（k1/k2/k3）：**

| 名字 | 公式 | 说明 |
|------|------|------|
| k1 | $\log\dfrac{\pi_\text{ref}}{\pi_\theta}$ | 最简单，直接算 log ratio |
| k2 | $\dfrac{1}{2}\left(\log\dfrac{\pi_\theta}{\pi_\text{ref}}\right)^2$ | 对称，数值更稳定 |
| **k3**（GRPO 默认） | $\dfrac{\pi_\theta}{\pi_\text{ref}} - \log\dfrac{\pi_\theta}{\pi_\text{ref}} - 1$ | 无偏估计，非负，GRPO 使用 |

### 标准 KL vs 近似 KL 对比

| 对比项 | 标准 KL | 近似 $u - \log u - 1$ |
|--------|---------|------------------------|
| 需要什么概率 | 词表所有词的概率 | **只需要实际生成的那个词** |
| 计算量 | $O(\text{vocab\_size})$ | $O(1)$ |
| 显存 | 需要存 logits 全表 | 只需要两个标量 |
| 数学性质 | 精确值 | 无偏估计（多次平均等于真实 KL） |
| 值域 | ≥ 0 | ≥ 0（$u=1$ 时取最小值 0） |

### f(u) 的图形直觉

```
f(u) = u - ln(u) - 1

u=0.25: 0.25 - (-1.386) - 1 = 0.636  ← 变小极多，大惩罚
u=0.50: 0.50 - (-0.693) - 1 = 0.193  ← 变小很多，也有惩罚
u=1.00: 1.00 -   0     - 1 = 0.000  ← 最小值，没变化
u=1.25: 1.25 -   0.223 - 1 = 0.031  ← 稍微变大，小惩罚
u=1.50: 1.50 -   0.405 - 1 = 0.095  ← 变大一些，中等惩罚
u=2.00: 2.00 -   0.693 - 1 = 0.307  ← 变大很多，大惩罚

图形：
  ↑ KL
  |  \        /
  |   \      /
  |    \    /
  |     \  /
  |      \/
  +------1.0----→ u
         ↑
       最小值=0（两个模型一致，无惩罚）

u 偏离 1 越远（不管变大还是变小）→ KL 越大 → 惩罚越大
```

### 不同情况下 KL 的值（数值对照表）

| $P_\theta$ | $P_\text{ref}$ | $u = P_\theta / P_\text{ref}$ | $\text{KL} = u - \ln u - 1$ | 含义 |
|-------------|-----------------|--------------------------------|-------------------------------|------|
| 0.4 | 0.4 | 1.00 | 0.000 | 完全没变，无惩罚 |
| 0.5 | 0.4 | 1.25 | 0.031 | 稍微变大，小惩罚 |
| 0.6 | 0.4 | 1.50 | 0.095 | 变大一些，中等惩罚 |
| 0.8 | 0.4 | 2.00 | 0.307 | 变大很多，大惩罚 |
| 0.2 | 0.4 | 0.50 | 0.193 | 变小很多，也有惩罚 |
| 0.1 | 0.4 | 0.25 | 0.636 | 变小极多，大惩罚 |

> $u = 1$（没变化）→ KL = 0；u 偏离 1 越远 → KL 越大。

### 完整的一个 token 计算过程（step by step）

```
位置 t，生成的 token = "42"

Step 1：两个模型分别给这个词的概率
  π_θ("42")   = 0.6   （当前训练中的模型）
  π_ref("42") = 0.4   （参考模型，冻结不动）

Step 2：算 ratio u
  u = 0.6 / 0.4 = 1.5

Step 3：算近似 KL
  KL = 1.5 - ln(1.5) - 1 = 1.5 - 0.405 - 1 = 0.095

Step 4：代入 GRPO loss
  loss_t = min(r_t·A, clip(r_t)·A) - β × 0.095
                                      ↑
                               β 控制惩罚力度
                               比如 β=0.01 → 惩罚项 = 0.00095
```

### GRPO loss 完整计算：逐层结构

| 层级 | 操作 | 公式 |
|------|------|------|
| 单个 token | 算 u 和 KL | $u_t = \pi_\theta / \pi_\text{ref}$，$\text{KL}_t = u_t - \ln u_t - 1$ |
| 单个 token | 算 loss 项 | $\text{loss}_t = \min(r_t A, \text{clip} \cdot A) - \beta \text{KL}_t$ |
| 单个回答（$T_g$ 个 token） | token 维度取平均 | $L_g = \dfrac{1}{T_g}\sum_t \text{loss}_t$ |
| 单个问题（G 个回答） | 回答维度取平均 | $L = \dfrac{1}{G}\sum_g L_g$ |
| 一个 batch（N 个问题） | 问题维度取平均 | $L_\text{batch} = \dfrac{1}{N}\sum_i L^{(i)}$ |

**完整数值例子：**

```
回答 g，A_g = +1.2，β = 0.01，ε = 0.2

token   π_θ    π_ref    u      KL_t    r_t    min项       loss_t
─────────────────────────────────────────────────────────────────
The     0.7    0.6     1.167  0.024   1.05   1.05×1.2   1.260-0.00024=1.260
answer  0.4    0.4     1.000  0.000   0.98   0.98×1.2   1.176-0.000 =1.176
is      0.5    0.3     1.667  0.099   1.25   1.20×1.2   1.440-0.001 =1.439
42      0.6    0.4     1.500  0.095   1.10   1.10×1.2   1.320-0.001 =1.319
.       0.3    0.4     0.750  0.038   0.90   0.90×1.2   1.080-0.0004=1.080

L_g = (1.260 + 1.176 + 1.439 + 1.319 + 1.080) / 5 = 1.255
```

**整体直觉：**

```
GRPO 每个 token 的 loss：

min(r·A, clip·A)          β · KL
      ↑                      ↑
  reward 信号             惩罚项
  A>0 → 增大概率          跑太远 → 惩罚增大
  A<0 → 减小概率          没变化 → 惩罚=0

两者博弈：reward 说"加强这个token"，KL 说"别跑太远"，β 控制谁说了算
```

---

## 八、梯度下降与反向传播

### Loss 是关于参数 θ 的 n 元函数

$$\nabla_\theta \mathcal{L} = \left[\frac{\partial \mathcal{L}}{\partial \theta_1},\ \frac{\partial \mathcal{L}}{\partial \theta_2},\ \dots,\ \frac{\partial \mathcal{L}}{\partial \theta_n}\right], \quad \theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}$$

### forward 建图，backward 反向传播

```
θ₁, θ₂, θ₃, ...（叶子节点，参数）
    ↓
  矩阵乘法 → 激活函数 → ... → softmax → -log(P) → 取平均
    ↓
  L = 2.34（根节点，标量）
```

`loss.backward()` 从根节点出发，沿计算图反向，用链式法则逐层传播：

```
L = 2.34
  ↑ ∂L/∂(取平均) = 1

取平均
  ↑ ∂L/∂(-logP) = 1/T

-logP
  ↑ ∂L/∂(softmax输出) = -1/P

softmax
  ↑ ...

矩阵乘法
  ↑ ∂L/∂θ_i = 最终算出来的偏导数
              存到 θ_i.grad 里
```

每个参数都会得到一个 `.grad`，就是 `∂L/∂θ_i`，等待 `optimizer.step()` 使用。

### PyTorch 完整训练流程

```python
for batch in dataloader:
    # Step 1: 前向传播，建计算图，算平均 loss（标量）
    loss = model(batch).mean()

    # Step 2: 反向传播，链式法则算所有参数的偏导数
    loss.backward()              # ∂L/∂θ 存在 param.grad 里

    # Step 3: 更新参数  θ ← θ - η∇L
    optimizer.step()

    # Step 4: 清零梯度（否则下次 backward 会继续累加！）
    optimizer.zero_grad()
```

### 梯度累积（Gradient Accumulation）

显存不够时，分多步累积梯度，数学上等价于更大的 batch：

```python
accumulation_steps = 4  # 累积4步 = 等价于 batch×4

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps  # ⚠️ 注意缩放
    loss.backward()         # grad 累加（不清零）

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

```
为什么能累积？
  .grad 是累加的，每次 backward 都往里加：

  第1步：θ₁.grad = 0.3
  第2步：θ₁.grad = 0.3 + 0.2 = 0.5
  第3步：θ₁.grad = 0.5 + 0.1 = 0.6
  第4步：θ₁.grad = 0.6 + 0.3 = 0.9

  optimizer.step()  → 用累积的 0.9 更新
  optimizer.zero_grad() → 清零，开始下一轮

数学上等价（梯度是线性的，可以分批算再加）
```

**记忆口诀：**

```
forward   → 建图
backward  → 链式法则算梯度，存到 .grad（累加！）
step      → θ ← θ - η·grad
zero_grad → 清零 .grad
```

---

## 九、Batch 梯度的本质

每个样本定义一个关于 $\theta$ 的函数（$x_i, y_i$ 是常数，$\theta$ 是变量）：

$$\mathcal{L}_\text{batch}(\theta) = \frac{1}{B}\sum_{i=1}^{B} \ell(\theta;\, x_i, y_i), \quad \nabla_\theta \mathcal{L}_\text{batch} = \frac{1}{B}\sum_{i=1}^{B} \nabla_\theta \ell_i(\theta)$$

**关键：不同样本都是在同一个参数点 $\theta_0$ 处求梯度，$x_i$ 不同但求导变量是 $\theta$，所以可以合法地取平均。**

```
B 个样本，每个"建议"参数往某个方向走：

样本1（猫）：θ₁ 应该 +0.3，θ₂ 应该 -0.1
样本2（狗）：θ₁ 应该 +0.1，θ₂ 应该 +0.2
样本3（车）：θ₁ 应该 +0.2，θ₂ 应该 -0.1

平均：θ₁ → +0.2，θ₂ → ≈0（抵消）

"有共识"的维度叠加，"有争议"的维度抵消 → 全局最优方向
```

真正想优化的是期望损失，Batch 是它的蒙特卡洛估计：

$$\mathcal{L}(\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\ell(\theta; x, y)\right] \approx \frac{1}{B}\sum_{i=1}^{B} \ell(\theta; x_i, y_i)$$

这和 KL 近似用蒙特卡洛估计期望是**同一个道理**：单次采样不精确，但多次平均收敛到真实值。
