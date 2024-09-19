# torch实现一个完整的神经网络

## 4.1 torch.autograd和Variable
torch.autograd包的主要功能就是完成神经网络后向传播中的链式求导，手动去写这些求导程序会导致重复造轮子的现象。

自动梯度的功能过程大致为：先通过输入的Tensor数据类型的变量在神经网络的前向传播过程中生成一张计算图。然后通过反向传播，根据这个计算图和输出结果精确计算出每一个参数需要更新的梯度，并完成对参数的梯度更新。

完成自动梯度需要用到的torch.autograd包中的Variable类对我们定义的Tensor数据类型变量进行封装，在封装后，计算图中的各个节点就是一个Variable对象，这样才能应用自动梯度的功能。

下面我们使用autograd实现一个二层结构的神经网络模型

==从 PyTorch 0.4 版本开始，Variable 类被弃用，张量类（Tensor）自身就包含了所有 Variable 的功能。现在，创建张量时，只需指定 requires_grad=True 即可。==

```python
import torch
from torch.autograd import Variable
batch_n = 100#一个批次输入数据的数量
hidden_layer = 100
input_data = 1000#每个数据的特征为1000
output_data = 10

x = Variable(torch.randn(batch_n,input_data),requires_grad=False)
y = Variable(torch.randn(batch_n,output_data),requires_grad=False)
#用Variable对Tensor数据类型变量进行封装的操作。requires_grad如果是False，表示该变量在进行自动梯度计算的过程中不会保留梯度值。
w1 = Variable(torch.randn(input_data,hidden_layer),requires_grad=True)
w2 = Variable(torch.randn(hidden_layer,output_data),requires_grad=True)

#学习率和迭代次数
epoch_n=50
lr=1e-6

for epoch in range(epoch_n):
    h1=x.mm(w1)#(100,1000)*(1000,100)-->100*100
    print(h1.shape)
    h1=h1.clamp(min=0)
    y_pred = h1.mm(w2)
    #y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred-y).pow(2).sum()
    print("epoch:{},loss:{:.4f}".format(epoch,loss.data))
    
#     grad_y_pred = 2*(y_pred-y)
#     grad_w2 = h1.t().mm(grad_y_pred)
    loss.backward()#后向传播
#     grad_h = grad_y_pred.clone()
#     grad_h = grad_h.mm(w2.t())
#     grad_h.clamp_(min=0)#将小于0的值全部赋值为0，相当于sigmoid
#     grad_w1 = x.t().mm(grad_h)
    w1.data -= lr*w1.grad.data
    w2.data -= lr*w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()
    
#     w1 = w1 -lr*grad_w1
#     w2 = w2 -lr*grad_w2

```

## 4.2 torch.nn.Module自定义传播函数
其实除了可以采用自动梯度方法，我们还可以通过构建一个继承了torch.nn.Module的新类，来完成对前向传播函数和后向传播函数的重写。在这个新类中，我们使用forward作为前向传播函数的关键字，使用backward作为后向传播函数的关键字。下面我们进行自定义传播函数：

```python
import torch
from torch.autograd import Variable
batch_n = 64#一个批次输入数据的数量
hidden_layer = 100
input_data = 1000#每个数据的特征为1000
output_data = 10
class Model(torch.nn.Module):#完成类继承的操作
    def __init__(self):
        super(Model,self).__init__()#类的初始化
        
    def forward(self,input,w1,w2):
        x = torch.mm(input,w1)
        x = torch.clamp(x,min = 0)
        x = torch.mm(x,w2)
        return x
    
    def backward(self):
        pass
model = Model()
x = Variable(torch.randn(batch_n,input_data),requires_grad=False)
y = Variable(torch.randn(batch_n,output_data),requires_grad=False)
#用Variable对Tensor数据类型变量进行封装的操作。requires_grad如果是F，表示该变量在进行自动梯度计算的过程中不会保留梯度值。
w1 = Variable(torch.randn(input_data,hidden_layer),requires_grad=True)
w2 = Variable(torch.randn(hidden_layer,output_data),requires_grad=True)

epoch_n=30

for epoch in range(epoch_n):
    y_pred = model(x,w1,w2)
    
    loss = (y_pred-y).pow(2).sum()
    print("epoch:{},loss:{:.4f}".format(epoch,loss.data))
    loss.backward()
    w1.data -= lr*w1.grad.data
    w2.data -= lr*w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()
    

```

## Variable 类

在 PyTorch 的早期版本中，`Variable` 类用于封装张量，并提供计算梯度的功能。它在 PyTorch 的现代版本中已经被简化，张量（`Tensor`）现在直接支持梯度计算，而不需要显式地使用 `Variable`。不过，理解 `Variable` 的历史背景和作用对理解 PyTorch 的演变仍然很重要。

### Variable 类的作用

1. **封装张量和梯度信息**：
   - `Variable` 类的主要作用是将一个张量（`Tensor`）封装起来，并提供自动微分（自动计算梯度）的功能。这使得用户能够在进行计算时跟踪梯度并进行反向传播。

2. **计算梯度**：
   - `Variable` 允许对其包裹的张量进行梯度计算。通过设置 `requires_grad=True`，`Variable` 可以跟踪其计算历史，并在进行反向传播时计算梯度。

3. **历史记录**：
   - `Variable` 保存了计算历史，记录了如何通过各种操作生成当前的张量。这使得它能够在进行反向传播时计算所有涉及的梯度。

### Variable 的现代替代

从 PyTorch 0.4.0 版本开始，`Variable` 类的功能已经被集成到 `Tensor` 类中。现代的 PyTorch 代码直接使用 `Tensor`，不再需要显式使用 `Variable`。在新的 API 中，`Tensor` 自身可以管理梯度计算，记录计算历史，并支持自动微分。

### 现代 PyTorch 使用示例

使用 `Tensor`（无需 `Variable`）：
```python
import torch
import torch.nn as nn

# 定义简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 线性层
    
    def forward(self, x):
        return self.linear(x)  # 前向传播

# 初始化模型
model = SimpleModel()

# 定义输入和目标
input_tensor = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
target = torch.tensor([[2.0], [4.0], [6.0]])

# 前向传播
output = model(input_tensor)

# 计算损失
loss_function = nn.MSELoss()
loss = loss_function(output, target)

# 反向传播
loss.backward()

# 打印梯度
print(f"Input tensor gradients: {input_tensor.grad}")
```

### 旧版 PyTorch 使用 `Variable`

如果你仍然使用旧版 PyTorch 或查看旧代码，`Variable` 用法如下：

```python
import torch
from torch.autograd import Variable

# 定义输入和目标
input_tensor = Variable(torch.tensor([[1.0], [2.0], [3.0]]), requires_grad=True)
target = Variable(torch.tensor([[2.0], [4.0], [6.0]]))

# 定义模型和损失函数
model = nn.Linear(1, 1)  # 线性层
output = model(input_tensor)
loss_function = nn.MSELoss()
loss = loss_function(output, target)

# 反向传播
loss.backward()

# 打印梯度
print(f"Input tensor gradients: {input_tensor.grad}")
```

### 总结

- **旧版 PyTorch**：`Variable` 用于封装张量并支持自动微分。
- **现代 PyTorch**：`Variable` 的功能已经集成到 `Tensor` 类中，你可以直接使用 `Tensor` 进行梯度计算和反向传播，而不需要单独使用 `Variable`。
