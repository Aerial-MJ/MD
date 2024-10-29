## forward 中直接定义 nn.Module

可以在 `forward` 中直接定义 `nn.Module` 层，但一般不推荐这样做，因为会导致一些维护性和效率问题。让我们深入探讨一下在 `__init__` 和 `forward` 中定义层的不同影响：

**定义在 `__init__` 中的好处**

   - **参数管理**：在 `__init__` 中声明层（例如卷积层、全连接层）会将它们注册为模型的参数。这意味着这些层的参数（例如权重、偏置）会自动被加入模型的 `parameters()` 列表，从而被优化器识别、更新。
   - **模型结构清晰**：将层结构放在 `__init__` 中定义，可以一目了然地查看模型的组成部分，提高代码的可读性。尤其是在大型模型中，提前声明层结构可以让代码更具可维护性。
   - **避免重复创建**：如果在 `forward` 中定义层，模型每次调用 `forward` 时都会重新创建新的层对象。这会浪费内存，并导致模型每次计算时都包含不同的层，训练无法收敛。

**示例：推荐的写法**

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 26 * 26, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
```

**在 `forward` 中定义 `nn.Module` 层的问题**

   - **无法优化参数**：如果在 `forward` 中定义层（例如 `nn.Conv2d` 或 `nn.Linear`），这些层的权重和偏置不会被自动识别为模型参数，导致优化器无法更新它们的参数。
   - **浪费内存和效率低下**：每次前向传播都会创建新的层对象，增加内存开销并降低计算效率。层对象的参数将被频繁创建和销毁，导致训练变慢。
   - **模型结构不可见**：在 `forward` 中定义所有层使模型结构变得不可见或难以追踪，也不利于调试。

**示例：不推荐的写法**

```python
class MyModel(nn.Module):
    def forward(self, x):
        conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 每次调用forward都会创建新的层
        fc1 = nn.Linear(32 * 26 * 26, 10)
        x = conv1(x)
        x = x.view(x.size(0), -1)
        x = fc1(x)
        return x
```

```python
from torch import nn


class MyModel(nn.Module):
    def __int__(self):
        super(MyModel, self).__init__()
    def forward(self, x):
        conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 每次调用forward都会创建新的层
        fc1 = nn.Linear(32 * 26 * 26, 10)
        x = conv1(x)
        x = x.view(x.size(0), -1)
        x = fc1(x)
        return x

model=MyModel()


print(model.parameters())

# 打印所有参数的 requires_grad 状态
for name, param in model.named_parameters():
    print(f"{name}: {param.requires_grad}")
    ###什么都不会打印
```

==怎么修改这个模型==

**直接对model模块修改就可以，可以打到模型里面去**

```python
# 替换最后的全连接层
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)  # 假设为二分类任务
```

