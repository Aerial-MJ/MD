# 神经网络工具箱torch.nn

## nn.Module类
nn.Module是PyTorch提供的神经网络类，并在类中实现了网络各层的定义及前向计算与反向传播机制。在实际使用时，如果想要实现某个神经网络，只需继承nn.Module，在初始化中定义模型结构与参数，在函数forward()中编写网络前向过程即可。

1. nn.Parameter函数
2. forward()函数与反向传播
3. 多个Module的嵌套
4. nn.Module与nn.functional库
5. nn.Sequential()模块

```python
#这里用torch.nn实现一个MLP
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim1, hid_dim2, out_dim):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
          nn.Linear(in_dim, hid_dim1),
          nn.ReLU(),
          nn.Linear(hid_dim1, hid_dim2),
          nn.ReLU(),
          nn.Linear(hid_dim2, out_dim),
          nn.ReLU()
       )
    def forward(self, x):
        x = self.layer(x)
        return x
```

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

