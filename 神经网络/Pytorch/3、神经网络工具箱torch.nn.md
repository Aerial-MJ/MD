# 神经网络工具箱torch.nn

torch.autograd库虽然实现了自动求导与梯度反向传播，但如果我们要完成一个模型的训练，仍需要手写参数的自动更新、训练过程的控制等，还是不够便利。为此，PyTorch进一步提供了集成度更高的模块化接口torch.nn，该接口构建于Autograd之上，提供了网络模组、优化器和初始化策略等一系列功能。

## nn.Module类
nn.Module是PyTorch提供的神经网络类，并在类中实现了网络各层的定义及前向计算与反向传播机制。在实际使用时，如果想要实现某个神经网络，只需继承nn.Module，在初始化中定义模型结构与参数，在函数forward()中编写网络前向过程即可。

1. nn.Parameter函数
2. forward()函数与反向传播
3. 多个Module的嵌套
4. nn.Module与nn.functional库
5. nn.Sequential()模块

**`nn.Module` 是 PyTorch 提供的一个基类，用于构建神经网络模型。所有自定义的神经网络模型都应该继承 `nn.Module` 类。在定义模型时，需要在 `__init__` 方法中初始化网络层和参数，并在 `forward()` 方法中定义前向计算的过程。**

### 1. nn.Parameter 函数

`nn.Parameter` 是一种张量，当它作为模型的一部分被分配时，自动注册为模型的参数。一般用于将常规的 `Tensor` 转化为模型参数。

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
