# 神经网络工具箱torch.nn

torch.autograd库虽然实现了自动求导与梯度反向传播，但如果我们要完成一个模型的训练，仍需要手写参数的自动更新、训练过程的控制等，还是不够便利。为此，PyTorch进一步提供了集成度更高的模块化接口torch.nn，该接口构建于Autograd之上，提供了网络模组、优化器和初始化策略等一系列功能。

## torch.autograd

`torch.autograd`是PyTorch中用于实现自动求导的核心包。它能够根据操作记录自动计算梯度，这使得我们可以轻松地进行反向传播（Backpropagation），进而优化模型。在此过程中，`torch.autograd`会构建一个计算图，图中的节点是张量，而边则表示它们之间的操作关系。

### 使用自动求导的基本步骤

在使用`torch.autograd`时，主要的步骤如下：

1. **创建张量**：我们需要先创建需要计算梯度的张量，并设置其属性`requires_grad=True`。
2. **构建计算图**：对创建的张量进行各种操作以生成新张量。
3. **反向传播**：通过调用`.backward()`方法自动计算梯度。
4. **访问梯度**：可以通过`.grad`属性来访问计算得到的梯度。

### 案例：简单的线性回归

为了更好地理解`torch.autograd`的实现，我们通过一个简单的线性回归案例来演示自动求导过程。

#### 数据准备

我们首先生成一些线性数据作为训练集。考虑一个简单的线性方程：$ y=2x+1 $

我们在此基础上加入一些随机噪声来模拟实际数据。以下是数据准备的代码：

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)

# 生成数据
x = torch.linspace(0, 1, 100).reshape(-1, 1)  # 100个数据点
y = 2 * x + 1 + torch.randn(x.size()) * 0.1  # y = 2x + 1 + 噪声

# 可视化数据
plt.scatter(x.numpy(), y.numpy(), color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data')
plt.show()
```

#### 构建模型

接下来，我们构建一个简单的线性回归模型。这里我们使用`torch.nn.Linear`来创建一个线性层。

```python
import torch.nn as nn

# 定义线性模型
model = nn.Linear(1, 1)  # 输入是1维，输出也是1维
```

#### 定义损失函数(学习准则)和优化器

我们使用均方误差（MSE）作为损失函数，并使用随机梯度下降（SGD）作为优化器：

```python
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

#### 训练模型

现在，我们将进行模型的训练。在每个训练周期中，我们需要执行以下步骤：

1. **正向传播**：计算预测值。
2. **计算损失**：使用损失函数计算损失。
3. **反向传播**：调用`.backward()`来计算梯度。
4. **更新参数**：使用优化器更新模型参数。

以下是训练过程的代码：

```python
# 训练模型
num_epochs = 200

for epoch in range(num_epochs):
    # 正向传播
    outputs = model(x)  # 计算预测
    loss = criterion(outputs, y)  # 计算损失

    # 反向传播
    optimizer.zero_grad()  # 清零之前的梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

#### 可视化结果

训练完成后，我们可以可视化模型的预测结果：

```python
# 可视化结果
with torch.no_grad():  # 在这个上下文中不需要计算梯度
    predicted = model(x)

plt.scatter(x.numpy(), y.numpy(), color='blue')
plt.plot(x.numpy(), predicted.numpy(), color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Result')
plt.legend(['Predicted', 'Original'])
plt.show()
```

## nn.Module类

nn.Module是PyTorch提供的神经网络类，并在类中实现了网络各层的定义及前向计算与反向传播机制。在实际使用时，如果想要实现某个神经网络，只需继承nn.Module，在初始化中定义模型结构与参数，在函数forward()中编写网络前向过程即可。

1. nn.Parameter函数
2. forward()函数与反向传播
3. 多个Module的嵌套
4. nn.Module与nn.functional库
5. nn.Sequential()模块

`nn.Module` 是 PyTorch 提供的一个基类，用于构建神经网络模型。所有自定义的神经网络模型都应该继承 `nn.Module` 类。在定义模型时，需要在 `__init__` 方法中初始化网络层和参数，并在 `forward()` 方法中定义前向计算的过程。

### 1. nn.Parameter 函数

`nn.Parameter` 是一种张量，当它**作为模型的一部分**被分配时，自动注册为模型的参数。一般用于将常规的 `Tensor` 转化为模型参数。

**代码示例：**

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 将一个普通的张量转换为模型参数
        self.weight = nn.Parameter(torch.randn(2, 3))

    def forward(self, x):
        return x @ self.weight.t()
		#@ 是 Python 3.5 引入的矩阵乘法运算符。
        
# 创建模型实例
model = MyModel()
print("模型参数：", model.weight)
```

> Python 3.5 引入了 `@` 运算符，用于矩阵乘法。这个运算符提供了一种更直观和简洁的方式来进行矩阵乘法运算，尤其是对线性代数和数据处理领域的开发者来说，变得更加方便。
> 在 NumPy 中，`@` 运算符等价于 `numpy.matmul()` 或 `numpy.dot()`，当用于二维数组（矩阵）时，它表示常规的矩阵乘法。例如：
>
> ```python
> import numpy as np
> 
> # 定义两个矩阵
> A = np.array([[1, 2], [3, 4]])
> B = np.array([[5, 6], [7, 8]])
> 
> # 使用 @ 进行矩阵乘法
> C = A @ B
> 
> print(C)
> ```
>
> 输出将是：
>
> ```
> [[19 22]
>  [43 50]]
> ```
>
> 在此例中，`A @ B` 就是 A 矩阵和 B 矩阵的乘积。


**输出结果：**

```python
模型参数： Parameter containing:
tensor([[ 0.5687,  0.2231, -1.3217],
        [ 0.9637, -0.6682,  0.7135]], requires_grad=True)
```

在上面的例子中，`self.weight` 被声明为模型参数，这意味着它在反向传播时会被自动更新。

---

### 2. forward() 函数与反向传播

`forward()` 函数用于定义模型的前向传播过程，即如何从输入生成输出。在 PyTorch 中，反向传播和梯度计算是自动处理的，通常通过调用 `loss.backward()` 实现。

**代码示例：**

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

# 创建模型实例
model = MyModel()

# 定义输入数据和目标值
input_data = torch.tensor([[1.0, 2.0, 3.0]])
target = torch.tensor([[1.0]])

# 计算输出
output = model(input_data)
print("前向传播结果：", output)

# 定义损失函数
loss_fn = nn.MSELoss()
loss = loss_fn(output, target)
print("损失值：", loss.item())

# 反向传播
loss.backward()
print("线性层的权重梯度：", model.linear.weight.grad)
```

**输出结果：**

```python
前向传播结果： tensor([[-0.1547]], grad_fn=<AddmmBackward0>)
损失值： 1.327077865600586
线性层的权重梯度： tensor([[-1.0582, -2.1164, -3.1747]])
```

在这里，`forward()` 函数定义了前向传播过程，`loss.backward()` 计算并存储了梯度。

---

### 3. 多个 Module 的嵌套

在 PyTorch 中，可以通过将多个 `nn.Module` 组合在一起构建复杂的网络。每个子模块在前向传播时都会自动调用它们的 `forward()` 方法。

**代码示例：**

```python
import torch
import torch.nn as nn

class SubModel(nn.Module):
    def __init__(self):
        super(SubModel, self).__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.submodel = SubModel()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        x = self.submodel(x)
        return self.linear(x)

# 创建模型实例
model = MyModel()

# 定义输入数据
input_data = torch.tensor([[1.0, 2.0, 3.0]])
output = model(input_data)
print("嵌套模块前向传播结果：", output)
```

**输出结果：**

```python
嵌套模块前向传播结果： tensor([[0.1732]], grad_fn=<AddmmBackward0>)
```

在此示例中，`MyModel` 包含一个 `SubModel` 子模块，前向传播时会先通过子模块处理数据，然后再通过主模块进行处理。

---

### 4. nn.Module 与 nn.functional 库

`nn.functional` 是 PyTorch 的另一个模块，提供了与 `nn.Module` 类似的功能，但以函数形式出现。`nn.functional` 中的函数一般用于前向计算，而 `nn.Module` 提供的是包含状态的层。

>`torch.nn.functional` 是 PyTorch 中的一个模块，包含了许多神经网络中的常用函数。与 `torch.nn.Module` 类相比，`nn.functional` 提供的函数通常是无状态的，也就是说，它们不包含可学习的参数。因此，它们更像是直接的操作，而不是具有内部权重的层。
>
>`nn.functional` 中的函数可以用来执行各种操作，包括激活函数、卷积、池化、损失计算等。相对于 `torch.nn.Module` 的层对象，它们更适合在自定义 `forward` 函数中使用，因为它们不需要事先定义。
>
>#### 常见的 nn.functional 函数
>
>1. **激活函数**：
>   
>   - `F.relu(input)`：ReLU（整流线性单元）激活函数。
>   - `F.sigmoid(input)`：Sigmoid 激活函数。
>   - `F.softmax(input, dim)`：Softmax 激活函数，常用于分类任务中的输出层。
>   
>   示例：
>   ```python
>   import torch.nn.functional as F
>   
>   x = torch.randn(2, 3)
>   relu_x = F.relu(x)  # 使用 F.relu
>   softmax_x = F.softmax(x, dim=1)  # 使用 F.softmax
>   ```
>   
>2. **卷积操作**：
>   - `F.conv2d(input, weight, bias=None, stride=1, padding=0)`：二维卷积运算。
>
>   示例：
>   ```python
>   x = torch.randn(1, 3, 32, 32)  # 输入张量（batch_size=1, 通道数=3, 高度和宽度为32）
>   weight = torch.randn(6, 3, 5, 5)  # 卷积核权重（输出通道数=6, 输入通道数=3, 核大小=5x5）
>   conv_out = F.conv2d(x, weight)
>   ```
>
>3. **池化操作**：
>   - `F.max_pool2d(input, kernel_size, stride=None, padding=0)`：二维最大池化操作。
>   - `F.avg_pool2d(input, kernel_size, stride=None, padding=0)`：二维平均池化操作。
>
>   示例：
>   ```python
>   x = torch.randn(1, 3, 32, 32)
>   pooled_out = F.max_pool2d(x, kernel_size=2)  # 使用最大池化
>   ```
>
>4. **损失函数**：
>   - `F.cross_entropy(input, target)`：用于多分类任务的交叉熵损失函数。
>   - `F.mse_loss(input, target)`：均方误差（MSE）损失函数，常用于回归任务。
>
>   示例：
>   ```python
>   output = torch.randn(10, 5)  # 假设有 10 个样本，5 个类别的分类任务
>   target = torch.randint(0, 5, (10,))  # 随机生成 10 个标签
>   loss = F.cross_entropy(output, target)  # 计算交叉熵损失
>   ```
>
>5. **归一化**：
>   - `F.batch_norm(input, running_mean, running_var)`：批量归一化操作。
>   - `F.layer_norm(input, normalized_shape)`：层归一化操作。
>
>   示例：
>   ```python
>   x = torch.randn(2, 5)
>   normed_x = F.layer_norm(x, normalized_shape=(5,))  # 对每个样本的特征进行归一化
>   ```
>
>#### nn.functional vs nn.Module
>
>- **nn.Module** 提供的是具有状态的模块（如层、模型），这些模块通常包含可学习的参数（如 `nn.Linear`、`nn.Conv2d` 等）。
>- **nn.functional** 提供的是无状态的函数，这些函数可以直接用于张量操作或自定义的前向传播逻辑。
>
>通常情况下，如果你需要定义一个层并且它有参数（如权重和偏置），你会使用 `nn.Module`。但在实现自定义的前向传播逻辑时，`nn.functional` 中的函数是非常有用的。

**代码示例：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(3, 3))

    def forward(self, x):
        # 使用 nn.functional 进行线性变换
        return F.linear(x, self.weight)

# 创建模型实例
model = MyModel()

# 定义输入数据
input_data = torch.tensor([[1.0, 2.0, 3.0]])
output = model(input_data)
print("使用 nn.functional 计算的前向传播结果：", output)
```

**输出结果：**

```python
使用 nn.functional 计算的前向传播结果： tensor([[ 2.2663, -2.6118,  0.5638]], grad_fn=<AddmmBackward0>)
```

在这个例子中，使用了 `F.linear` 函数来进行前向传播，与直接使用 `nn.Linear` 类似，但更加灵活。

---

### 5. nn.Sequential() 模块

`nn.Sequential` 是一个特殊的容器模块，它允许将多个子模块按顺序执行。通过 `nn.Sequential`，可以快速搭建网络结构而不必显式定义 `forward()` 函数。

`Sequential()` 是 PyTorch 中的一个容器模块，它用于将多个层按顺序组合在一起，构成一个简单的神经网络。通过 `Sequential()`，你可以将一系列的神经网络层（如线性层、激活函数、卷积层等）堆叠在一起，无需显式定义前向传播逻辑。

其主要作用是简化模型构建过程，特别是当你的模型结构是按顺序排列的层时，`Sequential()` 提供了一个简洁的方式来组合它们。

**代码示例：**

```python
import torch
import torch.nn as nn

# 使用 nn.Sequential 定义模型
model = nn.Sequential(
    nn.Linear(3, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# 定义输入数据
input_data = torch.tensor([[1.0, 2.0, 3.0]])
output = model(input_data)
print("Sequential 模型前向传播结果：", output)
```

**输出结果：**

```python
Sequential 模型前向传播结果： tensor([[0.2176]], grad_fn=<AddmmBackward0>)
```

在这个例子中，`nn.Sequential` 容器按顺序执行了线性层、ReLU 激活函数和另一线性层的前向传播。

## 搭建简易神经网络

下面我们用torch搭一个简易神经网络：

1. 我们设置输入节点为1000，隐藏层的节点为100，输出层的节点为10
2. 输入100个具有1000个特征的数据，经过隐藏层后变成100个具有10个分类结果的特征，然后将得到的结果后向传播

```python
import torch
batch_n = 100#一个批次输入数据的数量
hidden_layer = 100
input_data = 1000#每个数据的特征为1000
output_data = 10

x = torch.randn(batch_n,input_data)
y = torch.randn(batch_n,output_data)

w1 = torch.randn(input_data,hidden_layer)
w2 = torch.randn(hidden_layer,output_data)

epoch_n = 20
lr = 1e-6

for epoch in range(epoch_n):
    h1=x.mm(w1)#(100,1000)*(1000,100)-->100*100
    print(h1.shape)
    h1=h1.clamp(min=0)
    y_pred = h1.mm(w2)
    
    loss = (y_pred-y).pow(2).sum()
    print("epoch:{},loss:{:.4f}".format(epoch,loss))
    
    grad_y_pred = 2*(y_pred-y)
    grad_w2 = h1.t().mm(grad_y_pred)
    
    grad_h = grad_y_pred.clone()
    grad_h = grad_h.mm(w2.t())
    grad_h.clamp_(min=0)#将小于0的值全部赋值为0，相当于sigmoid
    grad_w1 = x.t().mm(grad_h)
    
    w1 = w1 -lr*grad_w1
    w2 = w2 -lr*grad_w2

```

```python
torch.Size([100, 100])
epoch:0,loss:112145.7578
torch.Size([100, 100])
epoch:1,loss:110014.8203
torch.Size([100, 100])
epoch:2,loss:107948.0156
torch.Size([100, 100])
epoch:3,loss:105938.6719
torch.Size([100, 100])
epoch:4,loss:103985.1406
torch.Size([100, 100])
epoch:5,loss:102084.9609
torch.Size([100, 100])
epoch:6,loss:100236.9844
torch.Size([100, 100])
epoch:7,loss:98443.3359
torch.Size([100, 100])
epoch:8,loss:96699.5938
torch.Size([100, 100])
epoch:9,loss:95002.5234
torch.Size([100, 100])
epoch:10,loss:93349.7969
torch.Size([100, 100])
epoch:11,loss:91739.8438
torch.Size([100, 100])
epoch:12,loss:90171.6875
torch.Size([100, 100])
epoch:13,loss:88643.1094
torch.Size([100, 100])
epoch:14,loss:87152.6406
torch.Size([100, 100])
epoch:15,loss:85699.4297
torch.Size([100, 100])
epoch:16,loss:84282.2500
torch.Size([100, 100])
epoch:17,loss:82899.9062
torch.Size([100, 100])
epoch:18,loss:81550.3984
torch.Size([100, 100])
epoch:19,loss:80231.1484
```
