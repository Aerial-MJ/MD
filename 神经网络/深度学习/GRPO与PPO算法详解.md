# GRPO 与 PPO 算法详解

---

## 一、每个 token 的变量是什么？

### GRPO 里每个 token 的情况

$$\min(r_{g,t} \cdot A_g, \text{clip}(r_{g,t}) \cdot A_g) - \beta \cdot \text{KL}_{g,t}$$

| 变量 | 每个回答不同？ | 每个 token 不同？ | 说明 |
|------|------------|----------------|------|
| \(r_{g,t}\) | ✅ 是 | ✅ 是 | 每个位置预测的 token 不同，概率比不同 |
| \(\hat{A}_g\) | ✅ 是 | ❌ 否 | 一个回答只有一个 reward，同一回答内所有 token **共享**同一个 A |
| \(\text{KL}_{g,t}\) | ✅ 是 | ✅ 是 | 每个位置的概率分布不同，KL 也不同 |

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

### GRPO vs PPO：Advantage 的核心区别

| 对比项 | GRPO | PPO |
|--------|------|-----|
| **Advantage 粒度** | 回答级别（同回答内共享） | token 级别（每个 token 独立） |
| **A 怎么算** | \(\hat{A}_g = (R_g - \text{mean}(R)) / \text{std}(R)\) | 用 Critic 网络估计 \(V(s_t)\)，再算 GAE |
| **需要 Critic 网络** | ❌ 不需要 | ✅ 需要（一个和 Actor 一样大的网络） |
| **r 粒度** | token 级别 | token 级别 |
| **KL 粒度** | token 级别 | token 级别 |

---

### SFT / GRPO / PPO 完整对比表

| 变量 | SFT | GRPO | PPO |
|------|-----|------|-----|
| \(r_t\)（ratio） | ❌ 没有 | ✅ token 级别 | ✅ token 级别 |
| \(A\)（advantage） | ❌ 没有 | ✅ 回答级别（同回答内共享） | ✅ token 级别（每个不同） |
| \(\text{KL}\) | ❌ 没有 | ✅ token 级别 | ✅ token 级别 |
| Critic 网络 | ❌ | ❌ | ✅ 必须有 |
| 监督信号来源 | 标注数据 | 多个回答的相对 reward | Critic 估计的状态价值 |

---

## 二、PPO 的 Advantage 怎么算：V(s)、δ 和 GAE

### 为什么需要 V(s) 和 Advantage？

```
模型生成了一句话：
"The answer is 42 ."
  t1    t2  t3  t4  t5

最后得到 reward R = 1.0（答对了）

问题：这个 reward 是谁的功劳？
  是 "The" 的功劳？
  是 "42" 的功劳？
  还是大家都有功劳？

PPO 需要知道每个 token 位置"值多少钱" → 这就是 V(s_t) 的作用
```

---

### 核心概念定义

#### 状态 \(s_t\)（State）

```
t=1 时的状态：s1 = [问题 x]
t=2 时的状态：s2 = [问题 x, "The"]
t=3 时的状态：s3 = [问题 x, "The", "answer"]
t=4 时的状态：s4 = [问题 x, "The", "answer", "is"]

状态 = 当前已经生成的所有内容（前缀）
```

#### 状态价值 \(V(s_t)\)（Value）

$$V(s_t) = \text{从状态 } s_t \text{ 出发，未来能拿到的期望 reward}$$

```
V(s1) = 从一开始，预计能拿多少 reward？  比如 0.50
V(s2) = 已经生成"The"，预计能拿多少？   比如 0.55
V(s3) = 已经生成"The answer"，预计？    比如 0.60
V(s4) = 已经生成"The answer is"，预计？ 比如 0.80

V 由 Critic 网络来预测，它是一个独立的神经网络。
```

---

### TD Error δ（时序差分误差）

**直觉：**

```
你预测从 s3 出发能拿 0.60
实际走了一步到 s4，发现 s4 值 0.80
说明 s3 比你预期的更好！

误差 δ3 = 实际情况 - 你的预测
        = (r3 + V(s4)) - V(s3)
```

**公式：**

$$\delta_t = r_t + \gamma \cdot V(s_{t+1}) - V(s_t)$$

| 符号 | 含义 |
|------|------|
| \(r_t\) | t 时刻的即时 reward |
| \(\gamma\) | 折扣因子（0~1，未来的钱打折） |
| \(V(s_{t+1})\) | Critic 预测下一状态的价值 |
| \(V(s_t)\) | Critic 预测当前状态的价值 |

**语言模型里 reward 怎么分配？**

```
"The  answer  is   42   ."
  t1    t2    t3   t4   t5

r1 = 0    （中间 token 没有即时 reward）
r2 = 0
r3 = 0
r4 = 0
r5 = 1.0  （最后一个 token 才给 reward）
```

---

### 具体数值例子

**已知条件：**

```
生成序列：The  answer  is   42   .
reward：   0     0     0    0   1.0

Critic 预测的 V：
  V(s1) = 0.50
  V(s2) = 0.55
  V(s3) = 0.60
  V(s4) = 0.80
  V(s5) = 0.90
  V(s6) = 0.00  （终止状态，价值为 0）

γ = 0.99，λ = 0.95
```

**Step 1：算每个位置的 δ_t**

$$\delta_t = r_t + \gamma \cdot V(s_{t+1}) - V(s_t)$$

```
δ5 = 1.0 + 0.99×0.00 - 0.90 = +0.100
δ4 = 0.0 + 0.99×0.90 - 0.80 = +0.091
δ3 = 0.0 + 0.99×0.80 - 0.60 = +0.192
δ2 = 0.0 + 0.99×0.60 - 0.55 = +0.044
δ1 = 0.0 + 0.99×0.55 - 0.50 = +0.045

δ > 0 说明这个位置比预期好
δ < 0 说明比预期差
```

---

### GAE 公式（广义优势估计）

**为什么不直接用 δ 作为 Advantage？**

```
只用 δ_t（高偏差低方差）：
  只看一步，估计不准

用完整 return - V（低偏差高方差）：
  看所有步，但噪声大

GAE：在两者之间取平衡，用 λ 控制
```

**GAE 公式：**

$$A_t^{\text{GAE}} = \sum_{k=0}^{T-t} (\gamma\lambda)^k \cdot \delta_{t+k}$$

展开就是：

$$A_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$$

离当前越远的 δ，权重越小（指数衰减）。

**具体数值计算 GAE（从后往前）：**

```
γ=0.99，λ=0.95，γλ = 0.9405

A5 = δ5
   = 0.100

A4 = δ4 + γλ·A5
   = 0.091 + 0.9405×0.100
   = 0.091 + 0.094 = 0.185

A3 = δ3 + γλ·A4
   = 0.192 + 0.9405×0.185
   = 0.192 + 0.174 = 0.366

A2 = δ2 + γλ·A3
   = 0.044 + 0.9405×0.366
   = 0.044 + 0.344 = 0.388

A1 = δ1 + γλ·A2
   = 0.045 + 0.9405×0.388
   = 0.045 + 0.365 = 0.410
```

**结果汇总：**

| token | \(r_t\) | \(V(s_t)\) | \(\delta_t\) | \(A_t^{\text{GAE}}\) | 含义 |
|-------|---------|-----------|-------------|---------------------|------|
| The | 0 | 0.50 | +0.045 | **+0.410** | 开头就走对了方向 |
| answer | 0 | 0.55 | +0.044 | **+0.388** | 也不错 |
| is | 0 | 0.60 | +0.192 | **+0.366** | 明显往好的方向走 |
| 42 | 0 | 0.80 | +0.091 | **+0.185** | 关键 token，贡献大 |
| . | 1.0 | 0.90 | +0.100 | **+0.100** | 拿到 reward |

> \(A_t > 0\)：这个 token 的选择比预期好 → 增大它的概率  
> \(A_t < 0\)：这个 token 的选择比预期差 → 减小它的概率

---

### PPO 用 A_t 更新参数

$$\mathcal{L}_\text{PPO} = -\frac{1}{T}\sum_{t=1}^{T} \min\left(r_t \cdot A_t,\ \text{clip}(r_t, 1{-}\varepsilon, 1{+}\varepsilon) \cdot A_t\right)$$

```
每个 token 都有自己独立的 A_t
→ "42" 的 A 和 "The" 的 A 不一样
→ 能精细地告诉模型：哪个 token 选得好，哪个选得差
```

---

## 三、GRPO 为什么能省掉 Critic 网络？

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

---

## 四、完整流程对比

```
PPO 流程：
  生成回答
    → Critic 估计每个 token 的 V(s_t)
    → 算 δ_t = r_t + γV(s_{t+1}) - V(s_t)
    → 算 A_t = δ_t + γλ·δ_{t+1} + ...（GAE，从后往前）
    → 每个 token 用自己独立的 A_t 更新

GRPO 流程：
  生成 G 个回答
    → 每个回答算一个 reward R_g
    → A_g = (R_g - mean) / std
    → 同回答内所有 token 共享 A_g 更新
```

**核心差异一句话：**

> PPO 的 Advantage 是 **token 级别的精细估计**，GRPO 的 Advantage 是**回答级别的粗粒度估计**，用简单换省钱。

---

## 五、各算法 Loss 完整公式

### SFT Loss

$$\mathcal{L}_\text{SFT} = -\frac{1}{N}\sum_{i=1}^{N} \frac{1}{T_i}\sum_{t=1}^{T_i} \log P_\theta(y_t^{(i)} \mid x^{(i)}, y_{<t}^{(i)})$$

### PPO Loss

$$\mathcal{L}_\text{PPO} = -\frac{1}{T}\sum_{t=1}^{T} \min\left(r_t \cdot A_t,\ \text{clip}(r_t, 1{-}\varepsilon, 1{+}\varepsilon) \cdot A_t\right) + \beta \cdot \text{KL}$$

### GRPO Loss

$$\mathcal{L}_\text{GRPO} = -\frac{1}{G}\sum_{g=1}^{G} \frac{1}{T_g}\sum_{t=1}^{T_g} \left[\min\left(r_{g,t}\cdot\hat{A}_g,\ \text{clip}(r_{g,t}, 1{-}\varepsilon, 1{+}\varepsilon)\cdot\hat{A}_g\right) - \beta\cdot\text{KL}_{g,t}\right]$$

其中：

$$r_{g,t} = \frac{P_\theta(y_{g,t} \mid x, y_{g,<t})}{P_{\theta_\text{old}}(y_{g,t} \mid x, y_{g,<t})}, \quad \hat{A}_g = \frac{R_g - \text{mean}(R)}{\text{std}(R)}$$

---

## 六、梯度下降与反向传播

### Loss 是关于参数 θ 的 n 元函数

$$\theta = [\theta_1, \theta_2, \dots, \theta_n], \quad \mathcal{L}(\theta_1, \theta_2, \dots, \theta_n)$$

梯度 = 所有参数的偏导数组成的向量：

$$\nabla_\theta \mathcal{L} = \left[\frac{\partial \mathcal{L}}{\partial \theta_1},\ \frac{\partial \mathcal{L}}{\partial \theta_2},\ \dots,\ \frac{\partial \mathcal{L}}{\partial \theta_n}\right]$$

参数更新：

$$\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}$$

### forward 建图，backward 反向传播

```
θ₁, θ₂, θ₃, ...（叶子节点，参数）
    ↓
  矩阵乘法 → 激活函数 → ... → softmax → -log(P) → 取平均
    ↓
  L = 2.34（根节点，标量）
```

`loss.backward()` 从根节点出发，沿计算图反向，用链式法则：

$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial a_n} \cdot \frac{\partial a_n}{\partial a_{n-1}} \cdots \frac{\partial a_1}{\partial \theta}$$

每个参数得到 `θ.grad = ∂L/∂θ`，存起来等待 `optimizer.step()`。

### PyTorch 训练完整流程

```python
for batch in dataloader:
    # Step 1: 前向传播，建计算图，算平均 loss
    loss = model(batch).mean()

    # Step 2: 反向传播，链式法则算所有参数的偏导数
    loss.backward()             # ∂L/∂θ 存在 param.grad 里

    # Step 3: 更新参数  θ ← θ - η∇L
    optimizer.step()

    # Step 4: 清零梯度（否则下次 backward 会累加！）
    optimizer.zero_grad()
```

### 梯度累积（Gradient Accumulation）

显存不够放大 batch 时，分多步累积梯度，等价于更大的 batch：

```python
accumulation_steps = 4  # 累积4步再更新，等价于 batch×4

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps  # ⚠️ 注意要缩放
    loss.backward()         # 梯度累加到 .grad（不清零！）

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

```
为什么能累积？
  .grad 是累加的，每次 backward 都往里加，不会覆盖：

  第1步 backward：θ₁.grad = 0.3
  第2步 backward：θ₁.grad = 0.3 + 0.2 = 0.5
  第3步 backward：θ₁.grad = 0.5 + 0.1 = 0.6
  第4步 backward：θ₁.grad = 0.6 + 0.3 = 0.9

  optimizer.step()：用累积的 0.9 更新参数
  optimizer.zero_grad()：清零，开始下一轮

数学上完全等价于一次处理 4×batch 条样本（梯度是线性的）
```

### 记忆口诀

```
forward   → 建图（记录每一步计算）
backward  → 沿图反向，链式法则算梯度，存到 .grad
step      → 用 .grad 更新参数  θ ← θ - η·grad
zero_grad → 清零 .grad，否则下次累加
```

---

## 七、Batch 梯度的本质

每个样本定义一个关于 \(\theta\) 的函数（\(x_i, y_i\) 是常数，**\(\theta\) 是变量**）：

$$\ell_i(\theta) = \ell(\theta;\, x_i, y_i)$$

Batch Loss 是这些函数的平均：

$$\mathcal{L}_\text{batch}(\theta) = \frac{1}{B}\sum_{i=1}^{B} \ell_i(\theta)$$

由求导的线性性质：

$$\nabla_\theta \mathcal{L}_\text{batch} = \frac{1}{B}\sum_{i=1}^{B} \nabla_\theta \ell_i(\theta)$$

**关键：不同样本都是在同一个参数点 \(\theta_0\) 处求梯度，\(x_i\) 不同但求导变量是 \(\theta\)，所以完全可以平均——这就是普通的多元函数求偏导。**

```
Batch 里 B 个样本，每个样本"建议"参数往某个方向走：

样本1（猫）：θ₁ 应该 +0.3，θ₂ 应该 -0.1
样本2（狗）：θ₁ 应该 +0.1，θ₂ 应该 +0.2
样本3（车）：θ₁ 应该 +0.2，θ₂ 应该 -0.1

平均：θ₁ 应该 +0.2，θ₂ 应该 ≈ 0（抵消）

θ₂ 方向各样本意见相反 → 平均后接近 0 → 不需要动 ✅
θ₁ 方向大家都说增大 → 平均后还是正 → 确实需要增大 ✅

"有共识"的维度叠加，"有争议"的维度抵消
→ 这就是全局最优方向
```

真正想优化的是期望损失，Batch 是它的蒙特卡洛估计：

$$\mathcal{L}(\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\ell(\theta; x, y)\right] \approx \frac{1}{B}\sum_{i=1}^{B} \ell(\theta; x_i, y_i)$$

---

## 八、KL 散度惩罚项详解

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
  → 保持语言能力
  → 防止 reward hacking
```

---

### 标准 KL 散度是什么？

$$\text{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

含义：分布 P 和分布 Q 有多不一样。

```
KL = 0    → 两个分布完全一样
KL 越大   → 两个分布差异越大
KL ≥ 0    → 永远非负
```

---

### GRPO 用的是哪种 KL？

GRPO 用的是一种近似形式，也叫**反向 KL 的无偏估计**：

$$\text{KL}_{g,t} = \frac{P_\theta}{P_\text{ref}} - \log\frac{P_\theta}{P_\text{ref}} - 1$$

令 \(u = \dfrac{P_\theta(y_{g,t})}{P_\text{ref}(y_{g,t})}\)，则：

$$\text{KL}_{g,t} = u - \log u - 1$$

---

### 具体数值怎么算

每个 token 位置 t，两个模型各给一个概率：

```
位置 t，真实生成的 token 是 "42"

π_θ   预测 "42" 的概率 = 0.6   （当前训练中的模型）
π_ref 预测 "42" 的概率 = 0.4   （参考模型，冻结）

u = 0.6 / 0.4 = 1.5
```

代入公式：

$$\text{KL} = u - \log u - 1 = 1.5 - \log(1.5) - 1 = 1.5 - 0.405 - 1 = 0.095$$

---

### 不同情况下 KL 的值

| \(P_\theta\) | \(P_\text{ref}\) | \(u = P_\theta / P_\text{ref}\) | \(\text{KL} = u - \ln u - 1\) | 含义 |
|-------------|-----------------|--------------------------------|-------------------------------|------|
| 0.4 | 0.4 | 1.00 | 0.000 | 完全没变，无惩罚 |
| 0.5 | 0.4 | 1.25 | 0.031 | 稍微变大，小惩罚 |
| 0.6 | 0.4 | 1.50 | 0.095 | 变大一些，中等惩罚 |
| 0.8 | 0.4 | 2.00 | 0.307 | 变大很多，大惩罚 |
| 0.2 | 0.4 | 0.50 | 0.193 | 变小很多，也有惩罚 |
| 0.1 | 0.4 | 0.25 | 0.636 | 变小极多，大惩罚 |

> \(u = 1\)（没变化）→ KL = 0；u 偏离 1 越远 → KL 越大。

---

### 为什么用 \(u - \log u - 1\) 而不用标准 KL？

```
f(u) = u - ln(u) - 1

u=0.5:  0.5 - (-0.693) - 1 = 0.193
u=1.0:  1.0 - 0       - 1 = 0.000  ← 最小值
u=1.5:  1.5 - 0.405   - 1 = 0.095
u=2.0:  2.0 - 0.693   - 1 = 0.307

图形：
  ↑ KL
  |  \        /
  |   \      /
  |    \    /
  |     \  /
  |      \/
  +------1.0----→ u
         ↑
       最小值=0
```

| 对比项 | 标准 KL | \(u - \log u - 1\) |
|--------|---------|-------------------|
| 计算方式 | 需要对所有词表求和 | 只需要当前 token 的概率 |
| 计算量 | 大（词表 5万+） | 小（一个除法） |
| 数学性质 | 无偏 | 是标准 KL 的无偏估计 |
| 值域 | ≥ 0 | ≥ 0 |

---

### 完整的一个 token 计算过程

```
位置 t，生成的 token = "42"

Step 1：两个模型分别给概率
  π_θ("42")   = 0.6
  π_ref("42") = 0.4

Step 2：算 ratio
  u = 0.6 / 0.4 = 1.5

Step 3：算 KL
  KL = 1.5 - ln(1.5) - 1 = 0.095

Step 4：代入 GRPO loss
  loss_t = min(r_t·A, clip(r_t)·A) - β × 0.095
                                      ↑
                               β 控制惩罚力度
                               比如 β=0.01
                               → 惩罚项 = 0.00095
```

---

### KL 是每个 token 单独算，然后取平均

是的，KL 是每个 token 单独算的：

```
生成序列：The  answer  is   42   .
           t1    t2    t3   t4   t5

每个位置都算一个 KL：

t1: π_θ("The")    / π_ref("The")    = u1 → KL1 = u1 - ln(u1) - 1
t2: π_θ("answer") / π_ref("answer") = u2 → KL2 = u2 - ln(u2) - 1
t3: π_θ("is")     / π_ref("is")     = u3 → KL3 = u3 - ln(u3) - 1
t4: π_θ("42")     / π_ref("42")     = u4 → KL4 = u4 - ln(u4) - 1
t5: π_θ(".")      / π_ref(".")      = u5 → KL5 = u5 - ln(u5) - 1
```

然后在序列内取平均（就是期望）：

$$\overline{\text{KL}} = \frac{1}{T}\sum_{t=1}^{T} \text{KL}_t$$

---

### 完整数值例子

```
回答 g，A_g = +1.2，β = 0.01，ε = 0.2

token   π_θ    π_ref    u      KL_t    r_t    min项      loss_t
─────────────────────────────────────────────────────────────────
The     0.7    0.6     1.167  0.024   1.05   1.05×1.2   1.260-0.00024=1.260
answer  0.4    0.4     1.000  0.000   0.98   0.98×1.2   1.176-0.000 =1.176
is      0.5    0.3     1.667  0.099   1.25   1.20×1.2   1.440-0.001 =1.439
42      0.6    0.4     1.500  0.095   1.10   1.10×1.2   1.320-0.001 =1.319
.       0.3    0.4     0.750  0.038   0.90   0.90×1.2   1.080-0.0004=1.080

L_g = (1.260 + 1.176 + 1.439 + 1.319 + 1.080) / 5 = 1.255
```

---

### 整体结构一张表说清楚

| 层级 | 操作 | 公式 |
|------|------|------|
| 单个 token | 算 KL、ratio、advantage | \(\text{KL}_t = u_t - \ln u_t - 1\) |
| 单个 token | 算 loss 项 | \(\text{loss}_t = \min(r_t A, \text{clip} \cdot A) - \beta \text{KL}_t\) |
| 单个回答（T 个 token） | token 维度取平均 | \(L_g = \dfrac{1}{T_g}\sum_t \text{loss}_t\) |
| 单个问题（G 个回答） | 回答维度取平均 | \(L = \dfrac{1}{G}\sum_g L_g\) |
| 一个 batch（N 个问题） | 问题维度取平均 | \(L_\text{batch} = \dfrac{1}{N}\sum_i L^{(i)}\) |

---

### 整体直觉

```
GRPO 的每个 token 的 loss：

min(r·A, clip·A)          β · KL
      ↑                      ↑
  reward 信号             惩罚项
  A>0 → 增大概率          跑太远 → 惩罚增大
  A<0 → 减小概率          没变化 → 惩罚=0

两者博弈：
  reward 说"这个token要加强"
  KL 说"但别跑太远"
  β 控制谁说了算
```

**一句话：** KL 就是量化"当前模型和参考模型在这个 token 上差了多少"，差得越多惩罚越大，用来防止模型训歪。每个 token 单独算 KL，然后和 reward 项加在一起，沿着 token 维度取平均，再沿着回答维度取平均，最后得到一个标量 loss，backward 一次搞定。
