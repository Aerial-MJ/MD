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

**Tensor 转 List**：使用 `tensor实例.tolist()`



**NumPy 转 Tensor**：`torch.from_numpy()`

**Tensor 转 NumPy**：`tensor示例.numpy()`



**List 转 NumPy 数组**：使用 `np.array()`

**NumPy 数组转 List**：使用 `ndarray实例.tolist()`

**(NumPy 的 N 维数组对象 ndarray，它是一系列同类型数据的集合)**



## Tensor<->List

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



## Numpy<->Tensor

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

**使用 GPU Tensor**

如果 `tensor` 在 GPU 上，`tensor.numpy()` 不能直接使用。需要先将它移到 CPU，再转换为 NumPy 数组：

```python
#创建一个 GPU 上的 Tensor
tensor_gpu = torch.tensor([1, 2, 3, 4], device='cuda')

# 移到 CPU 后再转换为 NumPy 数组
np_from_tensor_gpu = tensor_gpu.cpu().numpy()

print(np_from_tensor_gpu)
# 输出: array([1, 2, 3, 4])
```

在 PyTorch 中，`.cpu()` 和 `.detach()` 是两个常用的方法，分别用于将张量转移到 CPU 设备以及从计算图中分离张量。下面是它们的具体用途和差异：

1. `.cpu()`

`.cpu()` 方法用于将张量从当前设备转移到 CPU 内存中。这个方法在使用 GPU 加速的情况下特别常用，例如当我们在 GPU 上训练模型，但希望将结果转回 CPU，以便进行后续处理或显示。

- **用法**：
  
  ```python
  tensor = torch.randn(3, 3, device='cuda')  # 创建在 GPU 上的张量
  tensor_cpu = tensor.cpu()  # 转移到 CPU
  ```
  
- **作用**：
  - 将张量从 GPU 转移到 CPU（如果已经在 CPU 上，则没有效果）。
  - 在进行某些非 GPU 支持的操作（如与 NumPy 兼容的操作）时，通常需要将张量放在 CPU 上。

2. `.detach()`

`.detach()` 方法用于从计算图中分离张量，以防止其在反向传播中产生梯度。这对于只需要张量值，但不希望影响梯度计算的情况很有用，例如在模型推理或记录中间输出时。

- **用法**：
  
  ```python
  tensor = torch.randn(3, 3, requires_grad=True)  # 需要梯度的张量
  detached_tensor = tensor.detach()  # 分离张量，不再跟踪梯度
  ```
  
- **作用**：
  - 生成一个新的张量，与原始张量共享数据存储，但不会被纳入自动微分的计算图中。
  - 不会对原始张量的梯度计算产生影响，适用于只需要前向值、不希望记录梯度的场景。
  

**组合使用**

在模型推理或记录输出时，常会组合 `.cpu()` 和 `.detach()` 方法。例如：

```python
output = model(input)  # 假设 output 是 GPU 上的张量
output_cpu = output.detach().cpu()  # 分离计算图并转移到 CPU
```

这种组合能有效地将张量移至 CPU，且保证该张量不会在反向传播中被计算。



## List<->Numpy

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

在 `numpy` 中，`shape` 和 `size` 是两个用于描述数组（`ndarray`）结构的属性。它们有不同的用途：

### ndarray.shape
   - **作用**：返回数组每个维度的大小，表示数组的形状。
   - **返回值**：`shape` 返回一个包含各维度大小的元组。例如，二维数组的 `shape` 为 `(rows, columns)`。
   - **用途**：用于查看或更改数组的结构，例如用于遍历数组的各个维度或将数组输入到特定形状的网络模型中。

**示例**：

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])  # 创建一个2x3的二维数组
print(a.shape)  # 输出：(2, 3)，表示2行3列
```

### ndarray.size
   - **作用**：返回数组的总元素数，即数组中所有维度的大小之积。
   - **返回值**：`size` 返回一个整数，表示数组中包含的元素个数。
   - **用途**：用于确定数组的总大小，比如在数据处理中用来计算数据集的整体规模。

**示例**：

```python
print(a.size)  # 输出：6，表示总共6个元素
```

- `shape`：返回数组各维度的大小，形状信息；数据类型为元组。
- `size`：返回数组中元素的总个数；数据类型为整数。

### np.clip()

`np.clip()` 不是类函数，而是 **NumPy 模块中的一个通用函数**，即**模块函数**。它属于 `numpy` 模块的顶层函数，可以直接通过 `numpy` 调用，而不依赖于任何特定的类。

使用**numpy.clip(…)**根据指定的min和max值将数据限定在一定范围内截断

```python
import numpy as np
data = np.array([i for i in range(-5, 6)])
>>> array([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5])
# 指定min=-3， max=2，将数据限制在-3~2(包括-3和2)
data = np.clip(data, -3, 2)
>>> array([-3, -3, -3, -2, -1,  0,  1,  2,  2,  2,  2])
```

### np.hstack()

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

### np.vstack()

按垂直方向（行顺序）堆叠数组构成一个新的数组。堆叠的数组需要具有相同的维度

### ndarray.astype()

```python
import numpy as np

# 创建一个整数数组
arr = np.array([1, 2, 3])
print(arr, arr.dtype)

# 使用astype转换数组数据类型
float_arr = arr.astype(np.float64)
print(float_arr, float_arr.dtype)
```

在 `numpy` 中，`view()` 和 `reshape()` 也用于改变数组的形状，但它们的作用与 PyTorch 中有一些不同：

### ndarray.reshape()

   - **用途**：`reshape()` 用于改变数组的形状，返回一个新的视图或副本，具体取决于内存布局。
   - **内存布局要求**：如果数据在内存中是连续的，`reshape()` 返回的是一个视图（view），而不是新的数据副本；如果数据不连续，则会创建一个新数组。
   - **灵活性**：`reshape()` 不会改变数组的数据内容，只是重新组织元素以满足新的形状。

**示例**：

```python
import numpy as np

x = np.arange(6)  # 创建一个一维数组
y = x.reshape(2, 3)  # 更改形状为 (2, 3)
print(y)
```

   - 在此例中，`y` 通常是 `x` 的视图，不会占用额外的内存空间，但前提是 `x` 在内存中是连续的。
   - **Note**: `numpy.reshape()` 可以通过参数 `order`（如 `order='C'` 或 `order='F'`）来指定内存布局顺序，分别表示行优先和列优先。

### ndarray.view()

   - **用途**：`view()` 返回相同数据的不同视图，通常用于创建新的数据类型或改变数据解读方式。
   - **不同数据类型的视图**：`view()` 可以用于创建具有不同数据类型（dtype）的视图。不同于 `reshape()`，`view()` 不会更改形状，而是允许你以不同的字节解释数据。
   - **内存开销**：`view()` 不会创建新的数据副本，因此效率高，适合数据的“重解读”。

**示例**：

```python
x = np.array([1, 2, 3, 4], dtype=np.int32)
y = x.view(dtype=np.int16)  # 将 int32 类型的数组视为 int16
print(y)  # 输出：[1 0 2 0 3 0 4 0]
```

   - `reshape()`：用于改变数组形状，返回视图或副本，具体取决于数据在内存中的连续性。
   - `view()`：用于创建相同数据的不同视图，通常用于更改数据类型或重新解释数据内容，而不改变形状。

在 `numpy` 中，`reshape()` 用于改变数组的形状，而 `view()` 更适合数据类型转换或重解读。

### numpy的切片

在 NumPy 中，切片（slicing）操作非常强大，可以用于访问数组的子数组。切片不仅仅是对一维数组有效，在多维数组中也同样适用。切片的基本语法是：

```python
array[start:stop:step]
```

其中：
- `start`：切片的起始位置（包含），默认为 0。
- `stop`：切片的结束位置（不包含），默认为数组的长度。
- `step`：切片的步长，默认为 1。

对于多维数组，你可以在每个维度上使用切片操作。

**1. 一维数组切片**

```python
import numpy as np

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 基本切片
print(arr[2:7])  # 输出：[2 3 4 5 6]

# 切片步长
print(arr[1:8:2])  # 输出：[1 3 5 7]

# 从头开始切到索引5
print(arr[:5])  # 输出：[0 1 2 3 4]

# 从索引3开始切到末尾
print(arr[3:])  # 输出：[3 4 5 6 7 8 9]

# 反向切片
print(arr[::-1])  # 输出：[9 8 7 6 5 4 3 2 1 0]
```

**2. 多维数组切片**

对于多维数组，切片操作会根据每个维度进行处理。你可以在每个维度上使用切片语法。

```python
arr_2d = np.array([[0, 1, 2, 3],
                   [4, 5, 6, 7],
                   [8, 9, 10, 11]])

# 基本切片
print(arr_2d[1, 2])  # 输出：6，取第二行第三列的元素

# 取第1行到第2行，所有列
print(arr_2d[0:2, :])  # 输出：
# [[0 1 2 3]
#  [4 5 6 7]]

# 取所有行，索引1到3的列
print(arr_2d[:, 1:3])  # 输出：
# [[ 1  2]
#  [ 5  6]
#  [ 9 10]]

# 步长切片：取每行的偶数列
print(arr_2d[:, ::2])  # 输出：
# [[ 0  2]
#  [ 4  6]
#  [ 8 10]]

# 取第0行到第2行，每行的偶数列
print(arr_2d[:3, ::2])  # 输出：
# [[ 0  2]
#  [ 4  6]
#  [ 8 10]]
```

**3. 高维数组切片**

对于更高维的数组（如三维数组），切片依然适用，且同样可以按维度分别进行切片操作。

```python
arr_3d = np.array([[[0, 1], [2, 3]],
                   [[4, 5], [6, 7]],
                   [[8, 9], [10, 11]]])

# 选择所有二维数组的第一个元素（第一个维度是0）
print(arr_3d[0, :, :])  # 输出：
# [[0 1]
#  [2 3]]

# 选择所有二维数组的第二个元素（第二个维度是1）
print(arr_3d[:, 1, :])  # 输出：
# [[ 2  3]
#  [ 6  7]
#  [10 11]]

# 选择所有二维数组的第一行（第三个维度是0）
print(arr_3d[:, :, 0])  # 输出：
# [[ 0  2]
#  [ 4  6]
#  [ 8 10]]

# 获取数组的特定块
print(arr_3d[0:2, 0:1, :])  # 输出：
# [[[0 1]]
#  [[4 5]]]
```

**4. 反向切片（逆序）**

你可以通过步长 `-1` 来反向切片数组。

```python
arr = np.array([10, 20, 30, 40, 50])

# 反向切片
print(arr[::-1])  # 输出：[50 40 30 20 10]

# 对多维数组反向切片
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 反转行
print(arr_2d[::-1, :])  # 输出：
# [[7 8 9]
#  [4 5 6]
#  [1 2 3]]

# 反转列
print(arr_2d[:, ::-1])  # 输出：
# [[3 2 1]
#  [6 5 4]
#  [9 8 7]]
```

**使用切片保持二维结构**

你也可以通过切片操作直接保持二维结构，例如：

```python
# 直接切片，保持二维
second_row = arr_2d[1:2, :]
print(second_row)
```

```python
[[4 5 6 7]]
```

这里，`1:2` 的切片方式保持了二维结构，它表示从第 1 行到第 2 行（不包括第 2 行），但只取了第 1 行，从而保留了二维结构。**单独取一个数，会减少维度**

**选择不同的维度的时候直接使用逗号(,)**

**总结**

在 NumPy 中，切片操作非常灵活，可以根据不同需求对一维、二维及更高维的数组进行访问和修改。切片的关键是掌握如何操作 `start`、`stop` 和 `step` 参数，以及如何在不同维度上进行切片。

## list

### list的切片

Python 的 `list` 没有名为 `slice` 的方法，但可以使用**切片操作**来获取列表的子部分。切片是 `list[start:stop:step]` 的一种索引方式，可以灵活地从列表中提取一部分元素。**Python中列表截取（Slice，即冒号 : ）**

```
a[start:stop]  # 从 index=start 开始（即包含start），到 index=stop-1（即不包含stop）
a[start:]      # 从 index=start 开始（包含 start ），到最后
a[:stop]       # 从头开始，到 stop-1
a[:]           # 取整个 List
```

在 Python 中，列表切片（slicing）操作是非常灵活的，可以用于获取列表的子集或部分元素。与字符串切片操作类似，列表切片也使用 `start:stop:step` 的格式。切片不仅适用于一维列表，也适用于多维（嵌套）列表。

**一维列表切片**

基本语法：`list[start:stop:step]`

- `start`: 起始索引（包括该元素）。
- `stop`: 结束索引（不包括该元素）。
- `step`: 步长，默认是 1。

**嵌套列表切片**

对于多维（嵌套）列表，切片操作在**每个维度上独立工作**。每个维度都可以单独指定 `start`、`stop` 和 `step`。

```python
matrix = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 10, 11]
]

# 取前两行，所有列
print(matrix[:2])  # 输出: [[0, 1, 2], [3, 4, 5]]

# 取第一列
print([row[0] for row in matrix])  # 输出: [0, 3, 6, 9]

# 取第二行，所有列
print(matrix[1])  # 输出: [3, 4, 5]

# 取第二行，前两列
print(matrix[1][:2])  # 输出: [3, 4]

# 取前三行，列步长为2
print([row[::2] for row in matrix[:3]])  # 输出: [[0, 2], [3, 5], [6, 8]]
```

## torch

### torch的切片

在 PyTorch 中，张量（Tensor）的切片操作与 NumPy 的数组切片方式类似，但也有其独特之处。通过切片可以灵活地访问张量的某些部分，支持基本的切片、步长切片、索引切片以及高级索引。

1. **基本切片**

基本切片是通过索引和冒号（`:`）操作符进行的，可以访问张量的某一维或多个维度。

**示例：**

```python
import torch

# 创建一个 3x3 张量
tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 选择第一行
print(tensor[0])  # 输出: tensor([1, 2, 3])

# 选择第一列
print(tensor[:, 0])  # 输出: tensor([1, 4, 7])

# 选择张量的中心元素
print(tensor[1, 1])  # 输出: tensor(5)

# 选择子张量（例如，前两行的前两列）
print(tensor[:2, :2])  # 输出: tensor([[1, 2], [4, 5]])
```

**解析：**

- `tensor[0]`：获取第一行。
- `tensor[:, 0]`：使用冒号（`:`）表示获取所有行，指定列索引 `0` 来获取第一列。
- `tensor[1, 1]`：指定行索引和列索引，直接获取特定元素。
- `tensor[:2, :2]`：从前两行、前两列中获取子张量。

2. **步长切片**

可以通过在冒号切片中使用步长（`start:stop:step`）来跳过元素进行切片。

**示例：**

```python
# 使用步长切片
print(tensor[::2, ::2])  # 输出: tensor([[1, 3], [7, 9]])

# 获取每一行的倒数第二个元素
print(tensor[:, :-1])  # 输出: tensor([[1, 2], [4, 5], [7, 8]])
```

**解析：**

- `tensor[::2, ::2]`：第一个冒号 `::2` 表示每隔一行取一个元素，第二个 `::2` 表示每隔一列取一个元素。
- `tensor[:, :-1]`：省略 `stop`，从起始元素到倒数第二个元素，等价于 `tensor[:, 0:-1]`。

3. **使用 `...`（省略）操作符**

如果张量有较多的维度，可以用 `...` 代表多个维度以简化操作。

**示例：**

```python
tensor3d = torch.randn(2, 3, 4)

# 使用 ... 获取最后一个维度上的切片
print(tensor3d[..., 0])  # 输出: 获取最后一个维度的第一个元素

#tensor([[ 1.4675,  1.4023, -0.3413],
#        [-0.2521,  0.2897,  0.9018]])
```

```python
tensor3d = torch.randn(2, 3, 4)
print(tensor3d)
# 使用 ... 获取最后一个维度上的切片
print(tensor3d[..., 0])  # 输出: 获取最后一个维度的第一个元素
print(tensor3d[0, ...])  # 输出: 获取最后一个维度的第一个元素
print(tensor3d[0, 0])  # 输出: 获取最后一个维度的第一个元素
```

```
tensor([[[-0.7589, -0.7245,  1.4267, -1.4031],
         [-0.2000,  0.7929, -0.2183,  0.5908],
         [-0.5548,  0.4153, -0.5483, -0.1310]],

        [[-0.4204,  0.2522,  0.3542,  0.0784],
         [-1.3930, -1.0333,  0.6371,  1.0087],
         [-0.3352, -1.4252, -1.4356, -0.5923]]])
tensor([[-0.7589, -0.2000, -0.5548],
        [-0.4204, -1.3930, -0.3352]])
tensor([[-0.7589, -0.7245,  1.4267, -1.4031],
        [-0.2000,  0.7929, -0.2183,  0.5908],
        [-0.5548,  0.4153, -0.5483, -0.1310]])
tensor([-0.7589, -0.7245,  1.4267, -1.4031])
```

### torch.float64

`torch.float64` 是 PyTorch 中的一个**对象**，而不是一个类。它是 `torch.dtype` 类的一个实例，表示 PyTorch 的浮点数数据类型（双精度浮点数，即 64 位浮点数）。

在 PyTorch 中，数据类型（dtype）用于指定张量的元素类型，比如 `torch.float32`、`torch.float64`、`torch.int32` 等。这些数据类型本质上是 `torch.dtype` 类的具体实例，而不是类本身。

`numpy.int64` 不是一个对象，而是 **NumPy 的一种数据类型**，具体来说，它是 `numpy` 模块中定义的 **数据类型类**。在 NumPy 中，`numpy.int64` 是 `numpy.dtype` 的一个子类，表示 64 位的整数类型。与 PyTorch 的 `torch.int32` 不同，`numpy.int64` 本质上是一个类，而不是 `dtype` 类的一个对象。

---

在 `torch` 中，`tensor.size()` 和 `tensor.shape` 都用于获取张量的维度信息，但它们有一些细微的差异：

### tensor.size()

- `size()` 是 `torch.Tensor` 的一个方法，返回 `torch.Size` 对象。
- `torch.Size` 是一个类似于元组的对象，包含张量的每个维度大小。
- `size()` 是一种调用方式，适用于需要动态获取张量尺寸的情况。

**示例**：

```python
import torch

x = torch.randn(3, 4, 5)
print(x.size())  # 输出：torch.Size([3, 4, 5])
```

### tensor.shape

- `shape` 是张量的一个属性，直接返回 `torch.Size` 对象。
- 作为属性，`shape` 更简洁易读。

**示例**：

```python
print(x.shape)  # 输出：torch.Size([3, 4, 5])
```

**两者的区别**

- **用法**：`size()` 是一个方法，需要括号；`shape` 是属性，不需要括号。
- **功能**：在功能上，两者完全相同，均返回 `torch.Size` 对象，表示张量的维度。

在大多数情况下，`shape` 属性更简洁，使用更为广泛。

### tensor.to()

**torch.device("cuda:0")     =   "cuda:0"**

```python
>>> tensor = torch.randn(2, 2)  # Initially dtype=float32, device=cpu
>>> tensor.to(torch.float64)
tensor([[-0.5044,  0.0005],
        [ 0.3310, -0.0584]], dtype=torch.float64)

>>> cuda0 = torch.device('cuda:0')
>>> tensor.to(cuda0)
tensor([[-0.5044,  0.0005],
        [ 0.3310, -0.0584]], device='cuda:0')

>>> tensor.to(cuda0, dtype=torch.float64)
tensor([[-0.5044,  0.0005],
        [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
```

`.to()` 方法可以指定数据类型或设备（例如 `cuda`）：

```python
# 创建一个浮点数 Tensor
tensor = torch.tensor([1.2, 3.4, 5.6])

# 转换为整型
tensor_int = tensor.to(torch.int)
print(tensor_int)

tensor = torch.randn(3, 3)
tensor_gpu = tensor.to('cuda')  # 将张量转移到 GPU
tensor_cpu = tensor.to('cpu')   # 将张量转移回 CPU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = tensor.to(device)
```

`.cuda()` 方法

`.cuda()` 方法专门用于将张量或模型转移到 GPU。它默认使用设备 `cuda:0`（即第一个 GPU），但也可以通过传入索引来指定不同的 GPU。

- 用法：

  ```python
  tensor = torch.randn(3, 3)
  tensor_gpu = tensor.cuda()         # 将张量转移到默认 GPU（cuda:0）
  tensor_gpu1 = tensor.cuda(1)       # 将张量转移到第 1 个 GPU（cuda:1）
  ```

`.to()` vs `.cuda()`

- `.to()` 更通用，可以指定任意设备（`cpu` 或 `cuda`），而 `.cuda()` 只能用于 GPU。
- `.cuda()` 更适合于需要代码在 GPU 和 CPU 之间兼容的场景。

### tensor.type()

```python
# 创建一个浮点数 Tensor
tensor = torch.tensor([1.2, 3.4, 5.6])

# 转换为整型
tensor_int = tensor.to("cuda")

print(tensor_int.device)
print(tensor_int.dtype)
print(tensor_int.type())
>>>cuda:0
>>>torch.float32
>>>torch.cuda.FloatTensor
```

```python
>>> import torch
>>> a = torch.ones(2,3)
>>> a.type()
'torch.FloatTensor'
>>> a.dtype
torch.float32
>>> b = a.type(dtype='torch.cuda.DoubleTensor')
>>> b
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='cuda:0', dtype=torch.float64)
```

`.type()` 方法也可以指定 Tensor 的具体类型：

这里的dtype的上面的dtype不一样

```python
# 创建一个浮点数 Tensor
tensor = torch.tensor([1.2, 3.4, 5.6])
tensor_int = tensor.type(torch.IntTensor)
print(tensor_int)
```

### torch.argmax()

`torch.argmax` 是 PyTorch 中的一个函数，用于返回张量（tensor）中最大值的索引。该函数可以用于多维张量，允许指定一个维度（axis），从而查找该维度上最大值的索引。默认情况下，如果没有指定维度，`torch.argmax` 会返回张量中所有元素的最大值的索引。

`dim=1` 确实是指 **列维度**，而 `torch.argmax` 沿着指定维度查找最大值的索引时，`dim=1` 会查找每一行中最大值的索引。让我们详细解释一下：

在 PyTorch 中，张量的维度是从 0 开始的：

- `dim=0` 表示沿着 **行**（第一个维度）进行操作。
- `dim=1` 表示沿着 **列**（第二个维度）进行操作。

所以，当你使用 `dim=1` 时，实际上是指沿每一行来查找最大值的索引，即在每一行内部查找最大值。

**示例解释：**

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
max_indices = torch.argmax(x, dim=1)
print(max_indices)
```

- `dim=1` 表示在每一行查找最大值的索引。
- 第一行 `[1, 2, 3]` 中，最大值是 `3`，它的索引是 `2`。
- 第二行 `[4, 5, 6]` 中，最大值是 `6`，它的索引是 `2`。

因此，输出结果是 `[2, 2]`，表示每一行中最大值的索引。

**小结：**

- `dim=0`：沿着行方向查找最大值的索引。
- `dim=1`：沿着列方向查找最大值的索引（即对每一行查找最大值）。

### tensor.transpose()

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
### tensor.permute()

**permute**函数的基本操作是重组张量的维度。它支持高维操作，通过**tensor.permute(dim0, dim1, ..., dimn)**的方式来指定新的维度顺序。在调用**permute**时，必须指定所有维度。例如，对于一个三维张量**b**，可以使用**b.permute(2,0,1)**来重新排列其维度。

```python
b = torch.rand(2,3,4)

print(b.permute(2,0,1))
# 输出的张量将具有新的维度顺序
```
不同点

**transpose**一次只能交换两个维度，而**permute**可以交换多个维度。在处理二维张量时，**transpose**和**permute**可以互相替换。但在处理三维或更高维度的张量时，**permute**的功能更强大。例如，如果需要将三维张量的第零维放到第一维，第一维放到第二维，第二维放回第零维，**permute**可以一次性完成这个操作，而**transpose**则需要进行两次转换。

![image-20241025210038388](../Image/image-20241025210038388.png)

在 `torch` 中，`view()` 和 `reshape()` 都用于改变张量的形状，但它们在一些细节上有所不同。让我们详细比较它们：

### tensor.view()
   - **用途**：用于重新调整张量的形状（例如改变维度），但要求张量在内存中是连续的。
   - **内存连续性要求**：`view()` 需要张量是**连续的**（contiguous），否则会抛出错误。
   - **执行效率**：如果满足内存连续性条件，`view()` 会直接在原张量上调整形状，而不是创建新的副本，速度更快。

**示例**：

```python
import torch

x = torch.randn(2, 3, 4)
y = x.view(6, 4)  # 更改形状为 (6, 4)
print(y.shape)  # 输出：torch.Size([6, 4])
```

   - **注意**：如果你在某些操作后发现 `view()` 抛出错误，可以先调用 `x = x.contiguous()`，使张量变为连续。

`view(-1, 2)` 是一种重新调整张量形状的方法，它会将一个张量变成两列 (`2` 表示列的数目)，而行数则由 PyTorch 自动推断。

假设我们有一个形状为 `(4, 4)` 的张量，其中包含 `4 * 4 = 16` 个元素，我们可以使用 `view(-1, 2)` 将其调整为有两列的张量。

```python
import torch

# 创建一个 4x4 张量
tensor = torch.arange(16).view(4, 4)
print("Original tensor shape:", tensor.shape)

# 使用 view(-1, 2)
reshaped_tensor = tensor.view(-1, 2)
print("Reshaped tensor shape:", reshaped_tensor.shape)
print(reshaped_tensor)
```

### tensor.reshape()
   - **用途**：同样用于调整张量形状，但不要求张量在内存中是连续的。
   - **灵活性**：`reshape()` 会尝试返回一个具有相同数据但新形状的张量。它首先会检查原张量是否可以直接使用新的形状；如果不可以，它会生成新的张量副本。
   - **内存开销**：`reshape()` 的操作更灵活，但可能会创建新的副本，因此有时候内存开销略高于 `view()`。

**示例**：

```python
x = torch.randn(2, 3, 4)
z = x.reshape(6, 4)  # 形状调整为 (6, 4)，即使不连续也能成功
print(z.shape)  # 输出：torch.Size([6, 4])
```

**总结与选择**

   - `view()`：适用于内存连续的情况，速度更快，通常用于简单的形状变换。
   - `reshape()`：更灵活，不依赖内存连续性，更适合在不确定张量是否连续的情况下使用。

在不确定是否连续的情况下，建议使用 `reshape()`；否则，`view()` 会提供更高的性能。

### torch.flatten(x, 1)

`torch.flatten(x, 1)` 是 PyTorch 中用于将张量扁平化的操作。这个方法的作用是将输入张量 `x` 从指定的维度（在此例中是第一个维度 `1`）开始，展平成一个一维向量。下面是对这个操作的详细解释：

**语法**

```python
torch.flatten(input, start_dim, end_dim=-1)
```

- `input`：需要展平的张量。
- `start_dim`：从哪个维度开始展平。
- `end_dim`：展平到哪个维度，默认为 `-1`，即最后一个维度。

例子：`torch.flatten(x, 1)`

假设 `x` 是一个形状为 `(batch_size, channels, height, width)` 的 4D 张量，通常出现在图像数据中（例如批量的图片数据）。

1. **操作效果**：
   - 使用 `torch.flatten(x, 1)` 将会保持第 0 维的批量大小不变。
   - 从第 1 维到最后一个维度 (`channels`, `height`, `width`) 展平成一个一维向量。
2. **结果形状**：
   - 输入张量 `x` 的形状是 `(batch_size, channels, height, width)`。
   - 经过 `torch.flatten(x, 1)` 后，输出张量的形状变为 `(batch_size, channels * height * width)`。

### torch.clip()

在 PyTorch 中，`torch.clip` 函数用于将一个张量的值限制在给定的最小值和最大值之间，通常用于避免数值溢出或者是裁剪梯度。其用法类似于 `torch.clamp` 函数，`torch.clip` 是 `torch.clamp` 的一个别名。

```python
torch.clip(input, min=None, max=None, out=None)
```

**参数说明**

`input`：需要裁剪的张量。

`min`：裁剪的最小值（可以为 `None`，如果为 `None` 则不限制最小值）。

`max`：裁剪的最大值（可以为 `None`，如果为 `None` 则不限制最大值）。

`out`：可选参数，将输出结果存储到一个特定的张量中。

**返回值**

返回裁剪后的张量，每个值都在 `[min, max]` 范围内。

```python
import torch

# 创建一个张量
x = torch.tensor([1.5, 2.8, -1.2, 3.9, 0.5])

# 将值裁剪到 [0, 2] 的范围内
y = torch.clip(x, min=0, max=2)
print(y)
# 输出: tensor([1.5000, 2.0000, 0.0000, 2.0000, 0.5000])


torch.clip()
torch.clamp()#类似于
np.clip()
```

### torch.flip()

`torch.flip` 是 PyTorch 提供的一个用于翻转张量的函数。它能够在指定的维度上对张量进行反转操作，这在图像处理等场景中十分常见。翻转操作会返回一个新的张量，而不会改变原始张量的数据。

**语法**

```python
torch.flip(input, dims)
```

- **`input`**: 要翻转的输入张量。
- **`dims`**: 需要进行翻转的维度列表。可以同时指定多个维度。

**参数说明**

- `dims` 参数是一个列表，指定了需要反转的维度索引。例如：
  - 若 `dims=[0]`，表示只在第0维（行）上进行翻转。
  - 若 `dims=[1]`，表示只在第1维（列）上进行翻转。
  - 若 `dims=[0, 1]`，则会同时在行和列两个维度上翻转，类似对矩阵进行了 180° 旋转。

```python
import torch

# 创建一个 3x3 的张量
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# 在第 0 维上翻转
x_flip0 = torch.flip(x, [0])
print(x_flip0)
# 输出：
# tensor([[7, 8, 9],
#         [4, 5, 6],
#         [1, 2, 3]])

# 在第 1 维上翻转
x_flip1 = torch.flip(x, [1])
print(x_flip1)
# 输出：
# tensor([[3, 2, 1],
#         [6, 5, 4],
#         [9, 8, 7]])

# 同时在第 0 和第 1 维上翻转
x_flip01 = torch.flip(x, [0, 1])
print(x_flip01)
# 输出：
# tensor([[9, 8, 7],
#         [6, 5, 4],
#         [3, 2, 1]])
```

**注意事项**

- `torch.flip` 适用于任意维度的张量。它不会改变原始张量的形状，但会在指定维度上反转数据。
- 由于 `torch.flip` 返回的是新的张量，因此如果不将结果赋值给新的变量，原始张量不会发生变化。

**应用场景**

`torch.flip` 通常用于数据增强（如图像左右翻转）或其他需要对张量内容进行翻转操作的场景中。在深度学习中的数据预处理、数据增强、以及数据分析中非常常见。

### torch.rand()

- **功能**：生成在 `[0, 1)` 区间内均匀分布的随机数。

- **用法**：`torch.rand(*size)`，其中 `size` 是张量的维度。

- **示例**：

  ```python
  tensor = torch.rand(3, 3)  # 生成 3x3 的张量，每个元素在 [0, 1) 范围内
  print(tensor)
  ```

  **输出示例**（随机）：

  ```lua
  tensor([[0.5125, 0.3456, 0.8765],
          [0.1523, 0.9524, 0.3412],
          [0.4261, 0.0012, 0.7634]])
  ```

### torch.randn()

**功能**：生成符合 **标准正态分布**（均值为 0，标准差为 1）的随机数。

**用法**：`torch.randn(*size)`，其中 `size` 是张量的维度。

`torch.randn` 用于创建一个符合标准正态分布（均值为 0，标准差为 1）的 Tensor。

**代码示例：**

```python
import torch

# 创建一个 2x3 的符合标准正态分布的 Tensor
a = torch.randn(2, 3)
print(a)
```

**输出结果：**

```python
tensor([[-0.8067, -0.0707, -0.6682],
        [ 0.8141,  1.1436,  0.5963]])
```

**传入一个包含形状的元组**：如果需要动态地指定张量的形状，可以传入一个元组作为参数，例如：

```python
shape = (3, 4)
torch.randn(*shape)  # 等同于 torch.randn(3, 4)
```

在 `torch.randn(*shape)` 中，星号 (`*`) 是 Python 中的“解包”操作符。它的作用是将一个可迭代对象（如列表或元组）中的元素逐一提取出来，作为单独的参数传递给函数。

### torch.randn_like()

- **功能**：生成一个与给定张量形状相同的新张量，且新张量的每个元素也来自标准正态分布（均值为 0，标准差为 1）。
- **用法**：`torch.randn_like(input)`，其中 `input` 是一个张量，生成的随机张量将与 `input` 具有相同的形状。

```python
tensor1 = torch.randn(3, 3)  # 原始张量
tensor2 = torch.randn_like(tensor1)  # 与 tensor1 形状相同的随机张量
print(tensor2)
```

**输出示例**（每次运行结果会不同）：

```lua
tensor([[ 1.1234, -0.6547,  0.2983],
        [-1.3789,  0.4561, -0.9235],
        [ 0.0218,  0.2342, -0.8126]])
```

这里，`torch.randn_like(tensor1)` 会生成一个形状与 `tensor1` 相同的张量，并且这个新张量的元素也来自标准正态分布。

`torch.randn_like`：您只需要传递一个已有张量作为输入，它会返回一个与该张量相同形状的新张量，且元素是从标准正态分布中随机采样的。这对于需要生成与现有张量相同形状的随机数时特别有用。

**总结**

| 方法               | 功能                                                         | 需要传入的参数                              |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------- |
| `torch.randn`      | 根据指定形状生成随机数张量，元素符合标准正态分布。           | 张量的形状，如 `(3, 3)` 或 `(2, 2, 2)` 等。 |
| `torch.randn_like` | 根据已有张量的形状生成新的随机数张量，元素符合标准正态分布。 | 已有的张量 `input`。                        |

**使用场景**：

当知道需要的张量形状时，使用 `torch.randn`。

当希望生成一个与现有张量具有相同形状的新张量时，使用 `torch.randn_like`。

### torch.randint()

- **功能**：生成在指定区间 `[low, high)` 内的随机整数。

- **用法**：`torch.randint(low, high, size)`，其中 `low` 是最小值（包含），`high` 是最大值（不包含），`size` 是张量的维度。

- **示例**：

  ```python
  tensor = torch.randint(0, 10, (3, 3))  # 生成 3x3 的张量，每个元素是 0 到 9 之间的整数
  print(tensor)
  ```

  **输出示例**（随机）：

  ```lua
  tensor([[5, 1, 8],
          [7, 3, 0],
          [9, 4, 2]])
  ```

| 方法            | 分布类型     | 数值范围      | 示例用途               |
| --------------- | ------------ | ------------- | ---------------------- |
| `torch.rand`    | 均匀分布     | `[0, 1)`      | 初始化权重             |
| `torch.randn`   | 标准正态分布 | `(-∞, +∞)`    | 添加噪声、初始化权重   |
| `torch.randint` | 整数均匀分布 | `[low, high)` | 生成离散标签、随机索引 |

这三个方法根据需求生成不同类型的随机数，用于初始化、生成随机样本或标签等。



## tensor autograd

### tensor.grad

**tensor张量的梯度**

**冻结层参数**：
在迁移学习中，常常将预训练模型的部分层的 `requires_grad` 设置为 `False`，以避免更新这些层的参数。

### tensor.grad_fn

`tensor.grad_fn` 是 PyTorch 中一个与自动微分相关的重要属性。它存储了生成此 `tensor` 的**最后一个操作的梯度函数**，用于追踪和计算该张量的反向传播路径。

当你对一个张量进行一系列运算（如加法、乘法等）时，PyTorch 会自动建立一个计算图，其中每个节点代表一个操作（operation），每个边代表数据流动的方向。这张计算图被用于反向传播，以便计算每个张量对最终损失的梯度。

`grad_fn` 表示梯度函数：
`grad_fn` 属性是一个**函数节点**，它保存了生成该张量的最后一个操作的相关信息（例如，`AddBackward0`、`MulBackward0`）。这些 `Backward` 类别表示的是反向传播的类型，具体和该张量的生成操作相关。

**自动计算图**：
PyTorch 自动追踪每个操作以建立计算图。当你对一个带有 `requires_grad=True` 的张量进行运算时，计算图会自动扩展，并记录每个运算的 `grad_fn`。在反向传播时，PyTorch 会使用这个计算图，从输出节点沿反向路径计算梯度。

**只有通过运算生成的张量才有 `grad_fn`**：
原始的叶子节点张量（即直接创建、没有依赖任何其他张量的张量，如 `x = torch.tensor([1.0], requires_grad=True)`）的 `grad_fn` 为 `None`。这是因为叶子节点本身并不是由任何运算产生的。

### tensor.grad_fn.next_functions

### tensor.requires_grad

`tensor.requires_grad` 是 PyTorch 中的一个属性，表示该张量是否需要进行梯度计算。在深度学习中，如果我们需要对某个张量执行反向传播并计算其梯度，就需要将这个属性设置为 `True`。

`tensor.requires_grad` 是 PyTorch 中张量（`Tensor`）的一个属性，用于控制是否启用自动求导（autograd）功能。这个属性的值决定了 PyTorch 是否为该张量构建计算图，以便在进行反向传播时计算梯度。

- `requires_grad=True`: 如果设置为 `True`，则会启用对该张量的自动求导功能。这意味着，该张量将被包含在计算图中，任何通过这个张量进行的操作都会被追踪，最终可以通过 `.backward()` 方法计算梯度。

- `requires_grad=False`: 如果设置为 `False`，则该张量不会在计算图中进行追踪，也不会计算梯度。通常情况下，数据输入或中间结果的张量会将 `requires_grad` 设置为 `False`，只有当需要进行反向传播时，才会显式地设置为 `True`。

### tensor模型的计算图

**对应的是grad_fn和function**

在 PyTorch 中，**grad_fn** 属性是动态计算图的关键部分，也是 PyTorch 的关键功能之一。与 TensorFlow 等其他使用静态计算图的深度学习框架不同，PyTorch 根据实际代码执行情况动态构建计算图。这种动态特性允许在每次迭代或运行期间根据不同的输入数据调整和创建计算图

## torch.nn

**==怎么修改这个模型==**

**直接对model模块修改就可以，可以打到模型里面去**

```python
# 替换最后的全连接层
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)  # 假设为二分类任务
```

### parameter class

**继承自Torch.Tensor**

**model.parameters()**

```python
model = models.vgg16(pretrained=True)

print(model.parameters())
```

`model.parameters` 返回的是一个生成器（generator），而不是一个类。当你调用 `model.parameters` 时，它会返回一个迭代器，可以用来遍历模型中的所有可学习参数（例如权重和偏置）。具体来说，调用 `print(model.parameters)` 只会输出 `<generator object Module.parameters at ...>` 这样的信息，因为生成器本身并没有直接输出内容。

如果你想查看模型的参数，可以将其转换为列表或者遍历它，如下所示：

```python
import torch
from torchvision import models

model = models.vgg16(pretrained=True)

# 打印所有参数
for param in model.parameters():
    print(param.shape)

# 或者将参数转化为列表
params_list = list(model.parameters())
print(params_list)
```

在这个例子中，`for param in model.parameters():` 会遍历所有参数，并输出每个参数的形状。这样你可以更清楚地看到模型中参数的具体信息。

**model.features.parameters()和model.parameters()**

`model.parameters()`

- **作用**：返回模型中所有可训练的参数，包括每一层的权重和偏置。

- **使用场景**：通常用于定义优化器时，传入所有参数以便进行梯度更新。

- **适用范围**：当你想更新模型的全部参数时，可以使用 `model.parameters()`，例如在以下代码中：

  ```python
  for param in model.parameters():
      print("参数类型：", type(param), "参数大小：", param.size(), param.requires_grad)
  
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  ```

----

`model.features.parameters()`

**作用**：只返回模型中 `features` 模块的参数。

**使用场景**：如果只想更新特定部分的参数（如特征提取层 `features`），可以使用 `model.features.parameters()`。这种做法常见于微调预训练模型时，只希望更新特定的几层。

**适用范围**：例如，在经典的卷积神经网络如 VGG、ResNet 等中，特征提取层通常命名为 `features`，而全连接层或分类层可能命名为 `classifier` 或 `fc`。可以分别通过 `model.features.parameters()` 或 `model.classifier.parameters()` 获取这些层的参数。

----
**named_parameters()**

```python
for name, param in model.named_parameters():
    print(f"{name}: {param.requires_grad}")
```

**name的取名很讲究**=> 模块名，模块的层数，具体的参数

features.0.weight: False
features.0.bias: False
features.2.weight: False
features.2.bias: False
features.5.weight: False
features.5.bias: False
features.7.weight: False
features.7.bias: False

```python
for name, param in model.named_parameters():
    print(f"{name}: {param.requires_grad}")
```

## 模块的概念

`numpy` 并不是一个类，而是一个**模块**。在 Python 中，模块是一个包含函数、类、变量等定义的文件或一组文件。NumPy 库是一个模块化的库，其中包含多个模块（如 `numpy.linalg`、`numpy.random` 等），提供了大量用于科学计算的函数和类，但 `numpy` 本身并不是一个类。

**numpy 的结构**

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

在普通的 Python 文件中，可以通过定义函数并在文件中调用它们，或者从其他文件中引用这些函数。下面是三种常见的方法：

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

- **注意路径**： 在没有 `__init__.py` 文件的情况下，Python 仍然能够找到这些模块，只要你确保 Python 的搜索路径中包含了这些目录。默认情况下，当前工作目录和安装的包路径会被包含在 `sys.path` 中。

**总结**

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

## 整除除法//

在 `torch` 中，`//` 表示 **整数除法**（也称为**地板除法**），它会将除法的结果向下取整到最接近的整数。例如：

```python
import torch

a = torch.tensor([5])
b = torch.tensor([2])
result = a // b  # 结果为 2，而不是 2.5
```

在上面的例子中，`5 // 2` 的结果是 `2`，而不是浮点数 `2.5`，因为 `//` 会向下取整。

在 `torch` 中使用 `//` 操作符的场景通常是为了获取一个分片、分割或索引中的整数值，避免浮点数索引产生错误。

### 整数除法与常规除法
- **常规除法 `/`**：返回浮点数结果，如 `5 / 2` 会返回 `2.5`。
- **整数除法 `//`**：返回整数结果，向下取整
