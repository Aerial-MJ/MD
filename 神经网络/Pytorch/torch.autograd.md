# torch.autograd

在 PyTorch 中，当你构建和训练神经网络模型时，会隐式构建一个**计算图**（computation graph）。计算图由许多节点和边组成，其中：

- **节点**代表张量（Tensor）以及对张量进行的操作（Operation）。
- **边**表示张量之间的数据流，也就是从输入到输出的计算依赖关系。

PyTorch 中的计算图是**动态的**，这意味着图是在每次前向传播时根据当前的计算定义的。每次运行模型时，都会重新创建计算图，这与 TensorFlow 等使用静态计算图的框架不同。

在 PyTorch 中，**计算图是在前向传播过程中动态构建的**，而不是在计算完成后再构建。具体来说，每当你执行一个操作（如矩阵乘法、激活函数等），PyTorch 会即时地将该操作加入到计算图中，同时进行计算。

也就是说，**计算图的构建和计算是同步进行的**：
- 每当一个新操作（如张量乘法、激活函数等）被应用，计算图就会更新并记录操作，同时计算出结果。

这种即时构建计算图的方式允许 PyTorch 实现**动态计算图**（Define-by-Run），这意味着每次前向传播时，计算图都是根据当前执行的操作而动态创建的

### 在计算图中的节点
主要有以下几类节点出现在计算图中：

1. **叶子节点（Leaf Nodes）：**
   - 这些节点通常是网络的**输入张量**和模型中的**可学习参数**（如权重和偏差）。这些张量的属性 `requires_grad=True`，表示它们需要进行梯度计算。
   - 叶子节点是计算图的起点，它们没有由其他操作计算得到。

   示例：
   ```python
   input_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
   weight = torch.randn(3, requires_grad=True)
   bias = torch.randn(1, requires_grad=True)
   ```

2. **中间节点（Intermediate Nodes）：**
   - 这些是通过操作（如矩阵乘法、非线性激活等）生成的**中间结果张量**。这些张量是由其他张量计算得来的，因此它们在计算图中的位置是位于叶子节点之后的。
   - PyTorch 在构建计算图时，会根据这些操作动态创建新的节点。

   示例：
   ```python
   output = torch.matmul(input_tensor, weight) + bias
   ```

   这里，`output` 是一个中间节点，依赖于 `input_tensor`、`weight` 和 `bias`。

3. **根节点（Root Nodes）：**
   - 根节点通常是损失函数的输出或最终的计算结果，它是计算图中的**终点**。
   - 通过损失函数反向传播梯度时，会从根节点开始向叶子节点传播。

   示例：
   ```python
   loss = torch.mean((output - target) ** 2)  # 根节点
   ```

### 如何查看计算图中的节点
你可以通过张量的 `grad_fn` 属性来检查它的计算历史。`grad_fn` 表示创建该张量的函数，只有当张量是通过某些操作得到的中间张量时，`grad_fn` 才不为 None。

```python
print(output.grad_fn)  # 查看 output 的生成方式
print(loss.grad_fn)    # 查看 loss 的生成方式
```

如果你希望**可视化计算图**，可以使用一些工具来查看图的结构，如 `torchviz`：

```bash
pip install torchviz
```

然后：

```python
from torchviz import make_dot

y = model(input_tensor)  # 前向传播
make_dot(y, params=dict(list(model.named_parameters()) + [('input', input_tensor)]))
```

这样你就可以看到计算图中所有参与的张量和操作。

总结来说，计算图中的节点包括输入张量（叶子节点）、中间计算结果和最终的输出（根节点）。每个张量都有一个 `grad_fn` 来追踪它的创建过程，这在反向传播过程中至关重要。

## equires_grad=False的节点在计算图内吗

在 PyTorch 中，如果一个张量的 `requires_grad=False`，那么它的操作不会被包含在计算图中，也不会记录用于反向传播的梯度信息。因此，**不需要计算梯度的节点**是不会被构建到计算图中的。

1. `requires_grad=True` 的张量：
   - 这些张量会参与计算图的构建。PyTorch 会跟踪这些张量的操作，构建计算图，用于后续的反向传播计算。
   - 通过反向传播，PyTorch 能够计算这些张量的梯度。

2. `requires_grad=False` 的张量：
   - 这些张量不会参与构建计算图，即使它们被用于前向传播的计算，PyTorch 也不会记录这些计算的梯度。
   - 这样可以节省内存和计算资源，尤其是在推理阶段不需要计算梯度时。

**例子：**

```python
import torch

# 定义两个张量，一个需要梯度，一个不需要
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=False)

# 执行一些计算
z = x * y  # y 不需要梯度，因此它不会被包含在计算图中
loss = z.sum()

# 反向传播
loss.backward()

# 检查梯度
print(f"x的梯度: {x.grad}")  # x 有梯度，因为 requires_grad=True
print(f"y的梯度: {y.grad}")  # y 没有梯度，因为 requires_grad=False
```

**输出：**

```
x的梯度: tensor([4., 5., 6.])
y的梯度: None
```

在这个例子中：
- `x` 的 `requires_grad=True`，所以它参与了计算图的构建，并且 PyTorch 在反向传播时计算了它的梯度。
- `y` 的 `requires_grad=False`，因此虽然它参与了计算操作，但不会记录在计算图中，也不会计算它的梯度。

**总结：**

如果一个张量的 `requires_grad=False`，则它不会在计算图中记录，也不会为它计算梯度。因此，这类张量的计算虽然会参与前向传播，但不会参与梯度计算和反向传播。

## 计算图的释放

在 PyTorch 中，当你调用 `backward()` 时，计算图会被释放以节省内存。因此，不能直接进行第二次反向传播。如果你想在同一图中进行多次反向传播，需要在第一次调用时使用 `retain_graph=True` 参数。这会告诉 PyTorch 保留计算图的中间结果，以便后续可以进行更多的反向传播。

**示例代码**

```python
import torch

# 创建一个需要计算梯度的张量
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 创建一些操作
y = x + 2  # y 是由 x 生成的
z = sum(y * 3)  # z 是由 y 生成的

# 第一次反向传播，保留计算图
z.backward(retain_graph=True)  # 计算梯度并累积到 x.grad 中

# 查看 x 的梯度
print("First backward, x.grad:", x.grad)

# 再次反向传播，仍然保留计算图
z.backward()  # 计算梯度并累积到 x.grad 中

# 查看 x 的梯度
print("Second backward, x.grad:", x.grad)
```

**解释**

1. **`retain_graph=True`**：
   - 这个参数允许你在第一次反向传播后保留计算图的中间值。这样，你就可以对同一图进行多次反向传播。
   - 这样会占用更多内存，因为计算图没有被释放。

2. **梯度累积**：
   - 每次调用 `backward()` 时，计算的梯度会被加到现有的 `x.grad` 中。

**输出结果**

在这个示例中，第一次和第二次调用 `backward()` 后的 `x.grad` 会分别输出第一次和第二次的累积梯度。这样你就可以正确地计算累计梯度。

## 关闭梯度的计算

**没有梯度的张量和推理过程**

如果在推理（inference）过程中，通常不会需要计算梯度。因此，你可以使用 torch.no_grad() 上下文管理器禁用梯度计算，这会加速计算并节省内存。

```python
with torch.no_grad():
    output = model(input_tensor)   
```

要重新切换回需要计算梯度的模式，只需**退出** `torch.no_grad()` 的上下文管理器即可。`torch.no_grad()` 只在其作用的上下文（代码块）内禁用梯度计算，一旦退出该上下文，PyTorch 会恢复默认的计算梯度模式。

你可以通过以下几种方式在代码中控制是否计算梯度：

### 1. 退出 torch.no_grad() 上下文
一旦代码块执行完 `with torch.no_grad()`，PyTorch 会自动恢复回正常的梯度计算模式。

```python
# 推理模式，不计算梯度
with torch.no_grad():
    output = model(input_tensor)  # 这里不计算梯度

# 恢复正常模式，计算梯度
output_with_grad = model(input_tensor)  # 这里会计算梯度
```

### 2. 使用 torch.set_grad_enabled() 动态控制梯度开关
如果你希望显式控制是否计算梯度，可以使用 `torch.set_grad_enabled()`。这个函数根据传入的布尔值决定是否启用梯度计算。

```python
# 禁用梯度计算
torch.set_grad_enabled(False)
output = model(input_tensor)  # 不计算梯度

# 恢复梯度计算
torch.set_grad_enabled(True)
output_with_grad = model(input_tensor)  # 计算梯度
```

**总结**

- 当使用 `torch.no_grad()` 时，代码块内的操作不会计算梯度，但代码块外会恢复正常的梯度计算模式。
- 你也可以使用 `torch.set_grad_enabled(True)` 来显式启用或禁用梯度计算。

## 反向传播

**反向传播**

利用链式法则:反向传播算法基于微积分中的链式法则，通过逐层计算梯度来求解神经网络中参数的偏导数。

从输出层向输入层传播：算法从输出层开始，根据损失函数计算输出层的误差，然后将误差信息反向传播到隐藏层，逐层计算每个神经元的误差梯度，

计算权重和偏置的梯度:利用计算得到的误差梯度，可以进一步计算每个权重和偏置参数对于损失函数的梯度。

参数更新:根据计算得到的梯度信息，使用梯度下降或其他优化算法来更新网络中的权重和偏置参数，以最小化损失函数。

## 梯度清零

每次训练模型时，清零梯度是一个重要的步骤。这是因为 PyTorch 在每次调用 `backward()` 方法时，会累积梯度，而不是替换之前的梯度。以下是具体原因和相关解释：

### 1. **梯度累积的行为**

- **默认行为**：在 PyTorch 中，调用 `loss.backward()` 会将当前计算得到的梯度加到每个参数的 `.grad` 属性中，而不是直接覆盖它。这种行为允许在某些情况下（例如，使用小批量数据进行多次前向传播）对梯度进行累积。
  
- **累积的影响**：如果不在每个训练步骤开始时清零梯度，梯度将会在多个反向传播步骤中累积，这可能导致参数更新不正确，因为你可能会在多次计算中叠加了不相关的梯度。

### 2. **每个训练步骤的独立性**

- **每个批次独立**：每次训练步骤的目标是计算该步骤的损失并基于此损失更新模型的参数。为了保持每个训练步骤的独立性，必须在每个步骤开始时清除上一步的梯度，以确保只计算当前步骤的梯度。

### 3. **代码示例**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义模型
model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
input_tensor = torch.tensor([[1.0]], requires_grad=True)
target = torch.tensor([[2.0]])

# 训练循环
for epoch in range(100):
    # 清零梯度
    optimizer.zero_grad()
    
    # 前向传播
    output = model(input_tensor)
    
    # 计算损失
    loss_function = nn.MSELoss()
    loss = loss_function(output, target)
    
    # 反向传播
    loss.backward()
    
    # 更新参数
    optimizer.step()
```

### 4. **示例解释**

- 在上述代码中，每次迭代的开始都会调用 `optimizer.zero_grad()`，这会清零模型参数的梯度。这样做可以确保当前批次的梯度是独立的，反映的是当前输入数据与目标之间的关系。
- 接着，通过前向传播计算输出，再通过损失函数计算损失，并调用 `loss.backward()` 计算当前损失的梯度。
- 最后，通过 `optimizer.step()` 更新模型参数。

**总结**

每次训练迭代时清零梯度的目的是确保每个步骤的梯度计算都是基于当前数据的独立结果。这能避免不必要的梯度累积，从而保证模型参数的正确更新。

## 查看节点的历史--使用grad_fn

在 PyTorch 中，要查看某个张量的前一个节点及其构成，你可以访问该张量的 `grad_fn` 属性，进而追踪它的计算历史。每个 `grad_fn` 记录了生成该张量的操作和输入张量的关系。通过这种方式，你可以逐步回溯到生成该张量的所有输入节点。

### 查看前一个节点的步骤

1. 获取当前张量的 `grad_fn`：这将指向生成该张量的操作。
2. **查看输入张量**：`grad_fn` 对象通常具有 `next_functions` 属性，这个属性包含了生成当前张量的所有输入张量。

**示例代码**

```python
import torch

# 创建一个需要计算梯度的张量
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 创建一些操作
y = x + 2  # y 是由 x 生成的
z = y * 3  # z 是由 y 生成的

# 查看 z 的 grad_fn
print(z.grad_fn)  # 输出 <MulBackward0>

# 查看 y 的 grad_fn
print(z.grad_fn.next_functions)  # 查看 z 的输入函数（即 y）

# 继续查看 y 的输入
print(y.grad_fn)  # 输出 <AddBackward0>
print(y.grad_fn.next_functions)  # 查看 y 的输入函数（即 x）
```

### 输出说明

- `z.grad_fn`：表示 `z` 是通过乘法操作生成的，具体是 `<MulBackward0>`。
- `next_functions`：可以看到生成 `z` 的操作，通常是一个包含输入张量的列表。要获取这些输入张量，你可以进一步访问它们的 `grad_fn`。

### 注意事项

- **递归查看**：如果你想要查看更深层次的输入，可以递归地使用 `next_functions`，直到找到最底层的输入张量（通常是最初的输入数据）。
- **图的结构**：这种方法帮助你理解计算图的结构，并确认某个张量是由哪些操作生成的。

通过这种方式，你可以详细了解计算图中每个节点的来源以及它们是如何相互关联的。

## grad_fn属性

在 PyTorch 中，`grad_fn` 是每个张量（`Tensor`）的一个属性，用于指示该张量的生成方式及其计算历史。具体来说，`grad_fn` 属性包含了一个指向该张量的生成操作的引用，帮助自动微分过程了解如何计算梯度。

==PyTorch 的每个张量都会记录它的 `grad_fn`，即生成该张量的反向传播函数（如果该张量是通过操作生成的非叶子张量）。==

**`Backward` 对象（反向传播函数）**：

- 这些是 `torch.autograd.Function` 的子类，用于表示每个操作的反向传播逻辑。
- 每个 `Backward` 对象对应于前向传播中的一个操作（如加法、乘法、卷积等），当反向传播时，PyTorch 会从这些 `Backward` 对象开始，按需计算每个张量的梯度。

### grad_fn 属性的详细说明

1. **计算图中的节点**：
   - 每当你对张量进行操作（例如加法、乘法等），PyTorch 会自动创建一个新的张量，并将其 `grad_fn` 属性设置为一个表示该操作的对象。这些对象通常是 `Backward` 类型的，例如 `AddBackward`、`MulBackward` 等。

2. **指向生成操作**：
   - `grad_fn` 指向生成该张量的操作，这使得在反向传播时可以回溯计算图，计算输入张量的梯度。

3. **只有可计算梯度的张量有 `grad_fn`**：
   - 只有当张量的 `requires_grad` 属性被设置为 `True` 时，它的 `grad_fn` 属性才会被设置。如果张量是直接创建的（例如通过常量或没有梯度的操作），则它的 `grad_fn` 将为 `None`。

### 示例代码

```python
import torch

# 创建一个需要计算梯度的张量
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 进行一些操作
y = x + 2  # y 是由 x 生成的
z = y * 3  # z 是由 y 生成的

# 查看 grad_fn 属性
print("y's grad_fn:", y.grad_fn)  # 输出 <AddBackward0>
print("z's grad_fn:", z.grad_fn)  # 输出 <MulBackward0>
```

**总结**

- `grad_fn` 属性用于记录张量的计算历史。
- 它指向生成该张量的操作，有助于在反向传播时计算梯度。
- 只有 `requires_grad=True` 的张量会有 `grad_fn` 属性。通过 `grad_fn`，你可以追踪计算图中的操作，并理解每个张量是如何产生的。

## MulBackward0类--->grad_fn属性

在 PyTorch 中，`MulBackward0` 是一个表示乘法操作反向传播的类。这里的命名方式有其特定的含义：

### 命名解释

1. **操作类型**：
   - `Mul` 表示这个节点对应的是乘法操作。在计算图中，每个操作都有一个对应的反向传播类，例如加法、乘法、线性变换等。

2. **Backward**：
   - `Backward` 表示这是一个反向传播的操作类。它定义了如何计算输入张量的梯度。每个操作在前向传播中会生成一个新的张量，并在反向传播中需要定义如何从输出张量计算输入张量的梯度。

3. **后缀数字**：
   - 数字 `0` 是用来区分不同操作的。PyTorch 支持多个相同类型的操作在同一计算图中，比如多个乘法操作。后缀数字可以帮助唯一标识这些操作。例如，如果有多个乘法操作，后缀数字会递增（`MulBackward0`, `MulBackward1`, `MulBackward2` 等），确保每个操作都可以被独立追踪。

**总结**

- `MulBackward0` 指的是“乘法操作的反向传播”。
- `0` 是一个索引，帮助区分同一类型的多个操作实例。

这样的命名约定有助于在复杂的计算图中跟踪和管理操作，确保每个操作都能正确地定义其反向传播过程。

## (<AccumulateGrad object at 0x00000164B638AE20>, 0)--->grad_fn属性

在 PyTorch 中，当你查看某个张量的 `grad_fn` 属性时，输出通常包含一个对象和一个索引值（如 `(<AccumulateGrad object at 0x00000164B638AE20>, 0)`）。这里的 `0` 表示该操作的索引，用于区分同一类型的多个操作实例。

### 具体解释

1. `<AccumulateGrad object>`：
   - 这是 PyTorch 在计算梯度时使用的一个内部对象，通常表示梯度的累积。`AccumulateGrad` 是一种特殊的反向传播机制，用于将梯度累加到已经存在的梯度中。

2. **`0` 的含义**：
   - 这个数字是用来唯一标识该操作的实例。它的目的是在计算图中提供区分，以防出现多个相同类型的操作。例如，如果你对同一个张量执行了多次相同的操作，PyTorch 可能会创建多个 `AccumulateGrad` 对象，后缀数字可以帮助区分这些对象。
