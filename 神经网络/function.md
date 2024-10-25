**查看类型**

```
x = 10
print(type(x))  # 输出: <class 'int'>

y = "Hello"
print(type(y))  # 输出: <class 'str'>

z = [1, 2, 3]
print(type(z))  # 输出: <class 'list'>
```

**三个类型的相互转换**

**List 转 Tensor**：使用 `torch.tensor()`

**Tensor 转 List**：使用 `tensor.tolist()`



**NumPy 转 Tensor**：`torch.from_numpy()`

**Tensor 转 NumPy**：`tensor.numpy()`



**List 转 NumPy 数组**：使用 `np.array()`

**NumPy 数组转 List**：使用 `ndarray.tolist()`

**(NumPy 的 N 维数组对象 ndarray，它是一系列同类型数据的集合)**

- **list->tensor**

```
import torch

# 创建一个列表
my_list = [1, 2, 3, 4]

# 将列表转换为 tensor
my_tensor = torch.tensor(my_list)

print(my_tensor)
# 输出: tensor([1, 2, 3, 4])
```

- **tensor->list**

```
# 创建一个 tensor
my_tensor = torch.tensor([1, 2, 3, 4])

# 将 tensor 转换为列表
my_list = my_tensor.tolist()

print(my_list)
# 输出: [1, 2, 3, 4]

```





- **numpy->tensor**

```
import torch
import numpy as np

# 创建一个 NumPy 数组
np_array = np.array([1, 2, 3, 4])

# 将 NumPy 数组转换为 Tensor
tensor_from_np = torch.from_numpy(np_array)

print(tensor_from_np)
# 输出: tensor([1, 2, 3, 4], dtype=torch.int32)（dtype 根据 NumPy 数组而定）

```

- **tensor->numpy**

```
# 创建一个 Tensor
tensor1 = torch.tensor([1, 2, 3, 4])

# 将 Tensor 转换为 NumPy 数组
np_from_tensor = tensor1.numpy()

print(np_from_tensor)
# 输出: array([1, 2, 3, 4])

```





-  **List 转 NumPy 数组**

```
import numpy as np

# 创建一个 Python 列表
my_list = [1, 2, 3, 4]

# 将列表转换为 NumPy 数组
np_array = np.array(my_list)

print(np_array)
# 输出: array([1, 2, 3, 4])

```

- **NumPy 数组转 List**

```
# 创建一个 NumPy 数组
np_array = np.array([1, 2, 3, 4])

# 将 NumPy 数组转换为列表
my_list = np_array.tolist()

print(my_list)
# 输出: [1, 2, 3, 4]

```

## numpy

- **np.shape**

```python
import numpy as np

# 创建一个一维数组
a = np.array([1, 2, 3])
print(a.shape) # 输出: (3,)

# 创建一个二维数组
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b.shape) # 输出: (2, 3)

# 创建一个三维数组
c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(c.shape) # 输出: (2, 2, 2)
```

- **np.clip()**

使用numpy.clip(…)根据指定的min和max值将数据限定在一定范围内截断

```python
import numpy as np
data = np.array([i for i in range(-5, 6)])
array([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5])
# 指定min=-3， max=2，将数据限制在-3~2(包括-3和2)
data = np.clip(dat, -3, 2)
array([-3, -3, -3, -2, -1,  0,  1,  2,  2,  2,  2])
```

- **np.hstack**

按水平方向（列顺序）堆叠数组构成一个新的数组。堆叠的数组需要具有相同的维度

```python
import numpy as np

# 创建两个1-D数组
a = np.array((1,2,3))
b = np.array((4,5,6))

# 水平堆叠这两个数组
result = np.hstack((a,b))
print(result) # 输出: [1 2 3 4 5 6]
```

- **np.vstack()**

按垂直方向（行顺序）堆叠数组构成一个新的数组。堆叠的数组需要具有相同的维度

- np.ndarray.astype()

```python
import numpy as np

# 创建一个整数数组
arr = np.array([1, 2, 3])
print(arr, arr.dtype)

# 使用astype转换数组数据类型
float_arr = arr.astype(np.float64)
print(float_arr, float_arr.dtype)
```

## list

- **Python中列表截取（Slice，即冒号 : ）**

```
a[start:stop]  # 从 index=start 开始（即包含start），到 index=stop-1（即不包含stop）
a[start:]      # 从 index=start 开始（包含 start ），到最后
a[:stop]       # 从头开始，到 stop-1
a[:]           # 取整个 List
```

**选择不同的维度的时候直接使用逗号(,)**

## torch

- transpose函数

**transpose**函数的基本操作是接收两个维度**dim1**和**dim2**，并将这两个维度的内容进行调换。无论**dim1**和**dim2**的顺序如何，结果都是相同的。例如，对于一个二维张量*a*，可以使用**a.transpose(0,1)**或**a.transpose(1,0)**来交换其两个维度的内容。这个函数也可以通过**torch.transpose(tensor, dim1, dim2)**的方式调用。

```python
import torch

a = torch.Tensor([[1,2,3],[4,5,6]])

print(a.transpose(0,1))

# 输出:

# tensor([[1, 4],

# [2, 5],

# [3, 6]])
```
- permute函数

**permute**函数的基本操作是重组张量的维度。它支持高维操作，通过**tensor.permute(dim0, dim1, ..., dimn)**的方式来指定新的维度顺序。在调用**permute**时，必须指定所有维度。例如，对于一个三维张量**b**，可以使用**b.permute(2,0,1)**来重新排列其维度。

```python
b = torch.rand(2,3,4)

print(b.permute(2,0,1))
# 输出的张量将具有新的维度顺序
```
不同点

**transpose**一次只能交换两个维度，而**permute**可以交换多个维度。在处理二维张量时，**transpose**和**permute**可以互相替换。但在处理三维或更高维度的张量时，**permute**的功能更强大。例如，如果需要将三维张量的第零维放到第一维，第一维放到第二维，第二维放回第零维，**permute**可以一次性完成这个操作，而**transpose**则需要进行两次转换。

![image-20241025210038388](../Image/image-20241025210038388.png)

## 模块的概念

`numpy` 并不是一个类，而是一个**模块**。在 Python 中，模块是一个包含函数、类、变量等定义的文件或一组文件。NumPy 库是一个模块化的库，其中包含多个模块（如 `numpy.linalg`、`numpy.random` 等），提供了大量用于科学计算的函数和类，但 `numpy` 本身并不是一个类。

### numpy 的结构

- **模块级别函数**：如 `np.add()`、`np.clip()` 等直接定义在 `numpy` 模块中的函数，它们提供了数学运算、逻辑运算、数组操作等功能。
- **类**：NumPy 中的核心数据结构是 `ndarray`，这是一个类，用于表示多维数组。除了 `ndarray`，NumPy 还提供了其他辅助类（如 `matrix`、`dtype`、`random.Generator` 等）。
- **子模块**：NumPy 包含许多子模块，如 `numpy.linalg`（线性代数）、`numpy.fft`（快速傅里叶变换）和 `numpy.random`（随机数生成器）等，它们各自包含不同的函数和类。

例如，当我们使用 `np.array()` 创建一个数组时，`np.array()` 其实是 `numpy` 模块中的一个函数，而 `ndarray` 是一个类，最终返回的是 `ndarray` 类的一个实例：

```python
import numpy as np

# np.array() 是 numpy 模块中的一个函数，用于创建 ndarray 对象
arr = np.array([1, 2, 3, 4])
print(type(arr))
# 输出: <class 'numpy.ndarray'>
```

在这里：
- `np.array()` 是 `numpy` 模块中的一个函数。
- `arr` 是 `ndarray` 类的一个实例。

**总结**

- `numpy` 是一个模块，而不是一个类。
- `numpy` 模块包含多个函数、类和子模块，用于科学计算。

## numpy的random模块

`numpy.random` 是 NumPy 中用于生成随机数的一个**子模块**，它包含了大量用于随机数生成、随机采样和概率分布相关的函数和工具。

在 `numpy.random` 中，常用的随机数生成功能包括：

1. **基本随机数生成**：生成均匀分布、正态分布等常见分布的随机数。
2. **随机采样**：从指定的范围或数组中随机抽取数据。
3. **概率分布**：生成符合特定概率分布的随机数，如二项分布、泊松分布等。

### 常用方法

#### 1. 生成随机数

- `numpy.random.rand(d0, d1, ...)`：生成 \([0, 1)\) 区间的均匀分布随机数，返回形状为 `(d0, d1, ...)` 的数组。
- `numpy.random.randn(d0, d1, ...)`：生成标准正态分布的随机数，返回形状为 `(d0, d1, ...)` 的数组。
- `numpy.random.randint(low, high=None, size=None, dtype=int)`：生成指定范围内的随机整数。

**示例**：

```python
import numpy as np

# 生成一个形状为 (2, 3) 的 [0, 1) 间的均匀分布随机数
random_uniform = np.random.rand(2, 3)

# 生成一个形状为 (2, 2) 的标准正态分布随机数
random_normal = np.random.randn(2, 2)

# 生成范围在 1 到 10 之间的随机整数
random_integers = np.random.randint(1, 10, size=5)

print("Uniform Random Array:", random_uniform)
print("Normal Random Array:", random_normal)
print("Random Integers:", random_integers)
```

#### 2. **随机采样**

- `numpy.random.choice(a, size=None, replace=True, p=None)`：从数组 `a` 中随机采样，返回指定大小的数组。
  

**示例**：

```python
arr = [10, 20, 30, 40]
sample = np.random.choice(arr, size=2, replace=False)
print("Sampled Elements:", sample)
```

#### 3. **概率分布**

`numpy.random` 提供多种概率分布的随机数生成函数，比如：

- `numpy.random.normal(loc=0.0, scale=1.0, size=None)`：生成均值为 `loc`、标准差为 `scale` 的正态分布随机数。
- `numpy.random.binomial(n, p, size=None)`：生成符合二项分布的随机数。

**示例**：

```python
# 正态分布随机数
normal_dist = np.random.normal(0, 1, size=5)

# 二项分布随机数
binomial_dist = np.random.binomial(10, 0.5, size=5)

print("Normal Distribution:", normal_dist)
print("Binomial Distribution:", binomial_dist)
```

## 引用模块函数

在普通的 Python 文件中，可以通过定义函数并在文件中调用它们，或者从其他文件中引用这些函数。下面是两种常见的方法：

### 1. 定义和调用函数

你可以在同一个 Python 文件中定义一个函数，然后直接调用它。

```python
# my_script.py

def my_function(param1, param2):
    return param1 + param2

# 调用函数
result = my_function(5, 10)
print(result)  # 输出 15
```

### 2. 从其他文件引用函数

如果你有多个 Python 文件，并希望在一个文件中引用另一个文件中的函数，可以使用 `import` 语句。假设你有两个文件：`module.py` 和 `main.py`。

a. 创建 `module.py`

```python
# module.py

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
```

b. 在 `main.py` 中引用 `module.py`

```python
# main.py

# 导入整个模块
import module

result_add = module.add(5, 10)
result_subtract = module.subtract(10, 5)

print(result_add)      # 输出 15
print(result_subtract) # 输出 5
```

c. 只导入特定函数

如果只想导入某个特定的函数，可以这样做：

```python
# main.py

from module import add  # 只导入 add 函数

result = add(5, 10)
print(result)  # 输出 15
```

### 3. 使用包和模块

如果你的项目结构较复杂，可以将多个模块组织成包。包是包含 `__init__.py` 文件的目录。假设你有以下结构：

```
my_project/
│
├── my_package/
│   ├── __init__.py
│   ├── module1.py
│   └── module2.py
│
└── main.py
```

你可以在 `module1.py` 和 `module2.py` 中定义函数，然后在 `main.py` 中引用它们：

`module1.py`

```python
def func1():
    return "Hello from module1"
```

`module2.py`

```python
def func2():
    return "Hello from module2"
```

`main.py`

```python
from my_package.module1 import func1
from my_package.module2 import func2

print(func1())  # 输出 "Hello from module1"
print(func2())  # 输出 "Hello from module2"
```

在 Python 中，`__init__.py` 文件的主要作用是将包含它的目录标记为一个包。在 Python 3.3 及以后的版本中，这个文件不是强制性的，允许创建**无初始化文件的包**（也称为“命名空间包”）。你可以在没有 `__init__.py` 的文件夹中使用模块和函数，但有需要注意：

**注意路径**： 在没有 `__init__.py` 文件的情况下，Python 仍然能够找到这些模块，只要你确保 Python 的搜索路径中包含了这些目录。默认情况下，当前工作目录和安装的包路径会被包含在 `sys.path` 中。

### 小结

- **在同一个文件中定义和调用函数**：直接定义后调用。
- **从其他文件引用函数**：使用 `import` 语句，可以导入整个模块或特定的函数。
- **使用包和模块**：将函数组织成包，方便管理和引用。

![image-20241025201740790](../Image/image-20241025201740790.png)

## 列表推导式

在 Python 中，`value_expression for item in iterable if condition` 是一种列表推导式（list comprehension）或生成器表达式（generator expression）的语法，用于从可迭代对象（如列表、元组、集合等）中筛选元素并生成新序列。

这段语法的含义是：

- `value_expression`：表示生成的元素的表达式，可以是对 `item` 的任何操作。
- `for item in iterable`：表示从 `iterable` 中逐个取出元素赋值给 `item`。
- `if condition`：是一个可选的条件，用于过滤 `iterable` 中的元素。只有满足该条件的元素才会被包括在结果中。

以下是一个简单的示例：

```python
# 从一个列表中筛选出偶数并将它们平方
numbers = [1, 2, 3, 4, 5, 6]
squared_evens = [x**2 for x in numbers if x % 2 == 0]
print(squared_evens)  # 输出: [4, 16, 36]
```

在这个示例中，`squared_evens` 列表只包含了 `numbers` 列表中的偶数的平方。
