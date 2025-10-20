# Function

**查看类型**

```python
x = 10
print(type(x))  # 输出: <class 'int'>

y = "Hello"
print(type(y))  # 输出: <class 'str'>

z = [1, 2, 3]
print(type(z))  # 输出: <class 'list'>
```

**三个类型的相互转换**

**List 转 Tensor**：使用 `torch.tensor()`

**Tensor 转 List**：使用 `tensor实例.tolist()`     `list()`



**NumPy 转 Tensor**：`torch.from_numpy()`

**Tensor 转 NumPy**：`tensor示例.numpy()`



**List 转 NumPy 数组**：使用 `np.array()`

**NumPy 数组转 List**：使用 `ndarray实例.tolist()`    `list()`

```python
array = np.array([[1, 2, 3], [2, 3, 4]])
# list(array) [array([1, 2, 3]), array([2, 3, 4])]
```

**(NumPy 的 N 维数组对象 ndarray，它是一系列同类型数据的集合)**



## Tensor<->List

- **list->tensor**

```python
import torch

# 创建一个列表
my_list = [1, 2, 3, 4]

# 将列表转换为 tensor
my_tensor = torch.tensor(my_list)

print(my_tensor)
# 输出: tensor([1, 2, 3, 4])
```

- **tensor->list**

```python
# 创建一个 tensor
my_tensor = torch.tensor([1, 2, 3, 4])

# 将 tensor 转换为列表
my_list = my_tensor.tolist()

print(my_list)
# 输出: [1, 2, 3, 4]

```



## Numpy<->Tensor

- **numpy->tensor**

```python
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

```python
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

```python
import numpy as np

# 创建一个 Python 列表
my_list = [1, 2, 3, 4]

# 将列表转换为 NumPy 数组
np_array = np.array(my_list)

print(np_array)
# 输出: array([1, 2, 3, 4])

```

- **NumPy 数组转 List**

```python
# 创建一个 NumPy 数组
np_array = np.array([1, 2, 3, 4])

# 将 NumPy 数组转换为列表
my_list = np_array.tolist()

print(my_list)
# 输出: [1, 2, 3, 4]

```

## numpy

在 `numpy` 中，`shape` 和 `size` 是两个用于描述数组（`ndarray`）结构的属性。它们有不同的用途：

### numpy.array

numpy.array：总是复制（除非你显式指定 copy=False）

```python
import numpy as np

a = [1, 2, 3]
b = np.asarray(a)
```

特点：

总是生成一个**新的数组副本**，即使你传进去的已经是 NumPy 数组。

更灵活，比如你可以传 `dtype=...、copy=True/False` 等参数。

### numpy.asarray

```python
import numpy as np

a = [1, 2, 3]
b = np.asarray(a)

```

**如果输入已经是 NumPy 数组**，它不会复制内存，只是返回原数组的“视图”。

如果输入是列表、元组等其他格式，它会**转成 NumPy 数组**。

通常用于**提高兼容性**或**避免重复复制数据**。

```python
a = np.array([1, 2, 3])
b = np.asarray(a)
print(a is b)  # True，不会复制
```

**举例对比**

```python
lst = [1, 2, 3]
arr1 = np.array(lst)
arr2 = np.asarray(lst)

arr1[0] = 100
print(lst)  # [1, 2, 3] —— 原始列表不变

arr2[0] = 200
print(lst)  # [1, 2, 3] —— 仍然不变，因为 asarray 也是拷贝了
```

但是如果传入的是 NumPy 数组：

```python
a = np.array([1, 2, 3])
b = np.asarray(a)
b[0] = 99
print(a)  # [99 2 3] —— 没有复制，是引用
```

### np.zeros, np.ones, np.empty, np.full

这些方法是用于**创建新数组**，不是用已有的数据转换为数组。

```python
np.zeros((3,))        # 创建一个全是0的一维数组
np.ones((2, 2))       # 创建一个2x2全是1的数组
np.empty((2,))        # 创建未初始化的数组（内容随机）
```

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

**horizontal**

按水平方向堆叠数组构成一个新的数组。堆叠的数组需要具有相同的维度

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

**vertical**

按垂直方向堆叠数组构成一个新的数组。堆叠的数组需要具有相同的维度

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

观察到的结果是因为 `view()` 不改变数据，只是用目标数据类型来解释原始字节。而这些整数的二进制格式在 `float32` 解释下，变成了非常小的浮点值。

要避免这种混淆：

- **使用 `astype()`**：当需要实际转换为新类型时。
- **使用 `view()`**：当你明确需要重新解释数据的字节时。

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

**torch的切片不行**

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

### Numpy的拼接

在 NumPy 中，拼接数组通常使用 `numpy.concatenate()`、`numpy.vstack()`、`numpy.hstack()` 和 `numpy.stack()` 等函数。下面是这些函数的详细说明和示例。

**1. 使用 `numpy.concatenate()`**

`numpy.concatenate()` 用于在指定轴上拼接两个或多个数组。你可以选择沿着任意轴拼接（`axis=0` 为行，`axis=1` 为列）。

**语法**:

```python
numpy.concatenate((a1, a2, ...), axis=0, out=None)
```
- `a1, a2, ...`：待拼接的数组。
- `axis`：拼接的轴，默认为 0，表示沿着第一个轴（行）拼接。

**示例**:
```python
import numpy as np

# 创建两个二维数组
array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[7, 8, 9], [10, 11, 12]])

# 在第0轴（行方向）拼接
result = np.concatenate((array1, array2), axis=0)
print(result)
# 输出:
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]

# 在第1轴（列方向）拼接
result = np.concatenate((array1, array2), axis=1)
print(result)
# 输出:
# [[ 1  2  3  7  8  9]
#  [ 4  5  6 10 11 12]]
```

**2. 使用 `numpy.vstack()`**

`numpy.vstack()` 是 `numpy.concatenate()` 的一种简化形式，专门用于沿着垂直（行）方向堆叠数组。

**语法**:
```python
numpy.vstack(tup)
```
- `tup`：一个数组序列，必须具有相同数量的列。

**示例**:
```python
import numpy as np

# 创建两个二维数组
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

# 垂直堆叠
result = np.vstack((array1, array2))
print(result)
# 输出:
# [[1 2 3]
#  [4 5 6]]
```

**3. 使用 `numpy.hstack()`**

`numpy.hstack()` 是 `numpy.concatenate()` 的另一种简化形式，专门用于沿着水平方向（列）堆叠数组。

**语法**:
```python
numpy.hstack(tup)
```
- `tup`：一个数组序列，必须具有相同数量的行。

**示例**:
```python
import numpy as np

# 创建两个二维数组
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

# 水平堆叠
result = np.hstack((array1, array2))
print(result)
# 输出:
# [1 2 3 4 5 6]
```

**4. 使用 `numpy.stack()`**

`numpy.stack()` 用于沿着新轴堆叠数组。它会在指定的轴位置插入一个新的维度。**会增加维度**

**语法**:
```python
numpy.stack(arrays, axis=0)
```
- `arrays`：一个数组序列，必须具有相同的形状。
- `axis`：沿着哪一维度插入新轴。

**示例**:
```python
import numpy as np

# 创建两个二维数组
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

# 沿着新轴（axis=0）堆叠
result = np.stack((array1, array2), axis=0)
print(result)
# 输出:
# [[1 2 3]
#  [4 5 6]]

# 沿着新轴（axis=1）堆叠
result = np.stack((array1, array2), axis=1)
print(result)
# 输出:
# [[1 4]
#  [2 5]
#  [3 6]]
```

**5. 使用 `numpy.dstack()`**

`numpy.dstack()` 是用于沿着第三个维度（深度）拼接数组。它会在最后一个轴上进行堆叠。

**语法**:
```python
numpy.dstack(tup)
```
- `tup`：一个数组序列，必须具有相同的形状。

**示例**:
```python
import numpy as np

# 创建两个二维数组
array1 = np.array([[1, 2], [3, 4]])
array2 = np.array([[5, 6], [7, 8]])

# 沿着深度方向拼接
result = np.dstack((array1, array2))
print(result)
# 输出:
# [[[1 5]
#   [2 6]]
#
#  [[3 7]
#   [4 8]]]
```

**总结**

- **`numpy.concatenate()`**：在指定轴上拼接数组。
- **`numpy.vstack()`**：垂直拼接（沿着第0轴堆叠）。
- **`numpy.hstack()`**：水平拼接（沿着第1轴堆叠）。
- **`numpy.stack()`**：在指定轴插入一个新的维度来拼接数组。
- **`numpy.dstack()`**：沿着第三个维度拼接数组。

选择合适的方法取决于你希望在什么轴上拼接或堆叠数组，`concatenate` 和 `stack` 更加通用，而 `vstack`、`hstack` 和 `dstack` 是针对特定方向的简化版本。

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

### list的拼接

在 Python 中，拼接列表有多种方法。以下是几种常见的列表拼接方法：

1. **使用 `+` 运算符**

`+` 运算符可以用来拼接两个或多个列表，返回一个新的列表。

**示例**:
```python
list1 = [1, 2, 3]
list2 = [4, 5, 6]

# 拼接两个列表
result = list1 + list2
print(result)  # 输出: [1, 2, 3, 4, 5, 6]
```

2. **使用 `extend()` 方法**

`extend()` 方法将一个列表中的元素逐个添加到另一个列表的末尾。该方法会修改原始列表，返回 `None`。

**示例**:
```python
list1 = [1, 2, 3]
list2 = [4, 5, 6]

# 将 list2 中的元素添加到 list1 中
list1.extend(list2)
print(list1)  # 输出: [1, 2, 3, 4, 5, 6]
```

3. **使用 `append()` 方法（逐个元素添加）**

`append()` 方法将单个元素添加到列表的末尾。如果需要逐个拼接多个元素（而不是整个列表），可以使用 `append()` 方法。

**示例**:
```python
list1 = [1, 2, 3]
list2 = [4, 5, 6]

# 使用 append() 逐个添加 list2 中的元素
for item in list2:
    list1.append(item)
print(list1)  # 输出: [1, 2, 3, 4, 5, 6]
```

4. **使用 `*` 运算符（用于重复列表）**

如果你想要重复一个列表多个次数，可以使用 `*` 运算符。

**示例**:
```python
list1 = [1, 2, 3]

# 重复列表 3 次
result = list1 * 3
print(result)  # 输出: [1, 2, 3, 1, 2, 3, 1, 2, 3]
```

5. **使用 `itertools.chain()`（适用于多个列表的拼接）**

`itertools.chain()` 方法可以高效地拼接多个列表，适用于大规模拼接时性能更优。

**示例**:
```python
import itertools

list1 = [1, 2, 3]
list2 = [4, 5, 6]
list3 = [7, 8, 9]

# 使用 chain() 拼接多个列表
result = list(itertools.chain(list1, list2, list3))
print(result)  # 输出: [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

6. **列表推导式（适用于条件拼接）**

如果你需要在拼接时进行筛选或其他操作，可以使用列表推导式。

**示例**:
```python
list1 = [1, 2, 3]
list2 = [4, 5, 6]

# 使用列表推导式拼接并筛选
result = [x for x in list1] + [x for x in list2 if x > 4]
print(result)  # 输出: [1, 2, 3, 5, 6]
```

```python
list1 = [1, 2, 3]
list2 = [4, 5, 6]

# 拼接两个列表
result = [*list1 , *list2]
print(result)  # 输出: [1, 2, 3, 4, 5, 6]

result = [list1 , list2]
print(result)  # 输出: [[1, 2, 3], [4, 5, 6]]
```

在 Python 中，`*list` 是一种解包操作符，用于将列表或其他可迭代对象的元素逐个取出，并在调用函数、构造列表、赋值等场景中按需展开。

**总结**

- **`+` 运算符**：直接拼接两个列表，返回新的列表。
- **`extend()`**：将一个列表的元素逐个添加到另一个列表中，修改原始列表。
- **`append()`**：逐个添加元素，适用于单独元素的添加。
- **`*` 运算符**：用于重复列表的元素。
- **`itertools.chain()`**：适用于多个列表的拼接，性能更优。
- **列表推导式**：适用于条件拼接或在拼接时需要做操作的情况。

## random模块

Python 的 `random` 模块提供了用于生成伪随机数的功能，以及其他与随机性相关的实用工具。它是 Python 标准库的一部分，基于 Mersenne Twister 算法实现。

**常用功能**

1. **生成随机数**

- **`random.random()`**
   生成一个范围在 `[0.0, 1.0)` 的浮点数。

  ```python
  import random
  print(random.random())  # 输出例如：0.6394267984578837
  ```

- **`random.uniform(a, b)`**
   生成一个范围在 `[a, b]` 的浮点数。

  ```python
  print(random.uniform(1.5, 10.5))  # 输出例如：6.871710157079894
  ```

2. **生成随机整数**

- **`random.randint(a, b)`**
   返回一个范围在 `[a, b]` 的整数，包含 `a` 和 `b`。

  ```python
  print(random.randint(1, 10))  # 输出例如：7
  ```

- **`random.randrange(start, stop[, step])`**
   返回在 `range(start, stop, step)` 中的一个随机整数。

  ```python
  print(random.randrange(0, 10, 2))  # 输出例如：4
  ```

3. **随机选择**

- **`random.choice(seq)`**
   从非空序列中随机选择一个元素。

  ```python
  items = ['apple', 'banana', 'cherry']
  print(random.choice(items))  # 输出例如：'banana'
  ```

- **`random.choices(population, weights=None, k=1)`**
   从 `population` 中随机选择 `k` 个元素（可以重复），可指定权重。

  ```python
  print(random.choices(['a', 'b', 'c'], weights=[1, 2, 3], k=5))  # 输出例如：['c', 'b', 'c', 'a', 'c']
  ```

- **`random.sample(population, k)`**
   从 `population` 中随机选择 `k` 个唯一元素（不重复）。

  ```python
  print(random.sample(range(10), 4))  # 输出例如：[1, 3, 7, 2]
  ```

4. **打乱顺序**

- `random.shuffle(seq)`

  就地打乱序列（会修改原序列）。

  ```python
  numbers = [1, 2, 3, 4, 5]
  random.shuffle(numbers)
  print(numbers)  # 输出例如：[3, 1, 5, 2, 4]
  ```

5. **控制随机性（设置种子）**

- `random.seed(a=None)`

  设置随机数生成器的种子。相同的种子会生成相同的随机序列。

  ```python
  random.seed(42)
  print(random.random())  # 输出 0.6394267984578837
  ```

6. **生成正态分布或其他分布的随机数**

- **`random.gauss(mu, sigma)`**
   返回一个正态分布的随机数，均值为 `mu`，标准差为 `sigma`。

  ```python
  print(random.gauss(0, 1))  # 输出例如：-0.34038332775286323
  ```

- **`random.expovariate(lambd)`**
   返回一个指数分布的随机数，`lambd` 是分布的参数。

  ```python
  print(random.expovariate(1.5))  # 输出例如：0.4389147982460136
  ```

- **`random.betavariate(alpha, beta)`**
   返回 Beta 分布的随机数。

  ```python
  print(random.betavariate(2, 5))  # 输出例如：0.2537089593875248
  ```

**适用场景**

- 数据模拟（如抽样、分布生成）。
- 游戏开发（如随机地图、AI行为）。
- 安全（推荐使用 `secrets` 模块）。

**注意事项**

- `random` 模块生成的随机数是伪随机的，不能用于安全场景。
- 如果需要安全随机数，可以使用 `secrets` 模块或 `os.urandom()`。

**示例：随机抽奖**

```python
import random

participants = ['Alice', 'Bob', 'Charlie', 'David']
winner = random.choice(participants)
print(f"The winner is: {winner}")
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

**ValueError: step must be greater than zero**

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

### torch的拼接

在 PyTorch 中，拼接张量（tensor）有几种常见的方式，主要通过 `torch.cat()` 和 `torch.stack()` 来完成。它们的差异在于拼接的方式和维度。

**1. 使用 `torch.cat()`**

`torch.cat()` 用于在给定的维度上拼接多个张量。所有张量必须在除拼接维度外的其它维度上具有相同的形状。

**语法**:
```python
torch.cat(tensors, dim=0, out=None)
```
- `tensors` 是一个张量序列（例如一个列表或元组），这些张量将被拼接。
- `dim` 是指定在哪个维度上拼接，默认为 0。
- `out` 是可选的输出张量。

**示例**:
```python
import torch

# 创建两个张量
tensor1 = torch.randn(2, 3)
tensor2 = torch.randn(2, 3)

# 在第0维（行方向）拼接
result = torch.cat((tensor1, tensor2), dim=0)
print(result.shape)  # 输出：torch.Size([4, 3])

# 在第1维（列方向）拼接
result = torch.cat((tensor1, tensor2), dim=1)
print(result.shape)  # 输出：torch.Size([2, 6])
```

**2. 使用 `torch.stack()`**

`torch.stack()` 用于将多个张量沿着**一个新维度拼接**。拼接后的结果会增加一个新的维度。所有张量必须具有相同的形状。

**语法**:
```python
torch.stack(tensors, dim=0)
```
- `tensors` 是一个张量序列（例如一个列表或元组）。
- `dim` 是指定在哪个位置插入新维度（新维度会在这个位置创建）。

**示例**:

```python
import torch

# 创建两个张量
tensor1 = torch.randn(2, 3)
tensor2 = torch.randn(2, 3)

# 沿着新维度（dim=0）堆叠
result = torch.stack((tensor1, tensor2), dim=0)
print(result.shape)  # 输出：torch.Size([2, 2, 3])

# 沿着新维度（dim=1）堆叠
result = torch.stack((tensor1, tensor2), dim=1)
print(result.shape)  # 输出：torch.Size([2, 2, 3])
```

**3. 使用 `torch.unsqueeze()` 和 `torch.cat()`**

如果你希望在拼接时改变维度，也可以使用 `torch.unsqueeze()` 来为张量添加一个新的维度，然后再进行拼接。

**示例**:
```python
import torch

# 创建两个张量
tensor1 = torch.randn(2, 3)
tensor2 = torch.randn(2, 3)

# 使用 unsqueeze 增加维度
tensor1_unsqueezed = tensor1.unsqueeze(0)  # shape: [1, 2, 3]
tensor2_unsqueezed = tensor2.unsqueeze(0)  # shape: [1, 2, 3]

# 在第0维拼接
result = torch.cat((tensor1_unsqueezed, tensor2_unsqueezed), dim=0)
print(result.shape)  # 输出：torch.Size([2, 2, 3])
```

**4. 使用 `torch.view()` 和 `torch.cat()`**

如果你需要拼接张量并且调整其形状，也可以使用 `torch.view()`。

**示例**:
```python
import torch

# 创建两个张量
tensor1 = torch.randn(2, 3)
tensor2 = torch.randn(2, 3)

# 调整形状后拼接
tensor1_reshaped = tensor1.view(1, 6)  # 转为 1x6 的张量
tensor2_reshaped = tensor2.view(1, 6)  # 转为 1x6 的张量

# 拼接
result = torch.cat((tensor1_reshaped, tensor2_reshaped), dim=0)
print(result.shape)  # 输出：torch.Size([2, 6])
```

**总结**

- **`torch.cat()`** 用于沿指定维度拼接已有张量，张量在拼接维度外的形状必须相同。
- **`torch.stack()`** 用于沿新维度堆叠张量，结果会增加一个新的维度。
- **`torch.unsqueeze()`** 和 **`torch.view()`** 可以用来修改张量的形状，再进行拼接。

选择哪种方法，取决于你希望如何操作张量的维度以及你需要拼接的方式。

### torch.split

`torch.split` 用来**沿某个维度，把一个张量拆分成若干个子张量**。

它有两种主要用法：

👉 按**固定大小**拆分

```
torch.split(tensor, split_size, dim)
```

意思是：

> 沿着维度 `dim`，每 `split_size` 个元素拆成一块，最后一块可能会小一点。

📘 例子：

```
import torch
x = torch.arange(10)  # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
torch.split(x, 4)
```

输出：

```
(tensor([0, 1, 2, 3]), tensor([4, 5, 6, 7]), tensor([8, 9]))
```

👉 拆成每组 4 个元素的块。

按**指定长度列表**拆分

```
torch.split(tensor, split_sizes, dim)
```

其中 `split_sizes` 是一个列表或元组，表示每块的长度。

📘 例子：

```
x = torch.arange(10)
torch.split(x, [2, 3, 5])
```

输出：

```
(tensor([0, 1]), tensor([2, 3, 4]), tensor([5, 6, 7, 8, 9]))
```

👉 按 [2, 3, 5] 分割，三块加起来正好等于 10。

### tensor.item()

**从标量张量中提取值**

```python
import torch

# 单值张量
scalar_tensor = torch.tensor(42.0)
value = scalar_tensor.item()  # 提取值
print(value, type(value))  # 输出: 42.0 <class 'float'>
```

```python
import torch

print(torch.tensor([2983]).item())  # 输出: 2983
```

**原因**

1. `torch.tensor([2983])` 是一个一维张量，但它只包含一个元素。
2. 对于只含一个元素的一维张量，可以安全地调用 `.item()` 提取该元素的 Python 标量值。

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

- `dim=0` 表示 **沿着列方向** 找最大值（即跨行比较），返回 **行索引**。
- `dim=1` 表示 **沿着行方向** 找最大值（即跨列比较），返回 **列索引**。

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

### ==转置==

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

---

在 `torch` 中，`view()` 和 `reshape()` 都用于改变张量的形状，但它们在一些细节上有所不同。让我们详细比较它们：

### ==形状操作==

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

**如何相互转换**

`torch.flatten` 转换为 `tensor.reshape`：

- 如果知道原张量的形状，可以手动计算出目标形状并使用 `reshape()`。

- 示例：

  ```python
  x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
  flattened = torch.flatten(x, start_dim=1)  # torch.Size([2, 4])
  
  # 转换为 reshape 等效
  reshaped = x.reshape(x.size(0), -1)  # 需要指定其他维度
  print(flattened.equal(reshaped))  # True
  ```

### torch.unsqueeze()
- **功能**: 在指定的维度上**插入一个大小为1的维度**。
- **使用场景**: 当需要改变张量形状以适配特定操作（例如将 1D 张量扩展为 2D 或更高维度）时使用。

**语法**

```python
torch.unsqueeze(input, dim)
```

- `input`: 输入的张量。
- `dim`: 指定插入维度的位置（支持负数索引，表示从最后一维开始倒数）。

**示例**

```python
import torch

# 原始张量
x = torch.tensor([1, 2, 3])  # 形状: [3]

# 在第0维添加新维度
x1 = torch.unsqueeze(x, 0)  # 形状: [1, 3]

# 在第1维添加新维度
x2 = torch.unsqueeze(x, 1)  # 形状: [3, 1]

print(x1)  # tensor([[1, 2, 3]])
print(x2)  # tensor([[1], [2], [3]])
```

###  torch.squeeze()
- **功能**: **移除**张量中**大小为1的维度**。
- **使用场景**: 简化张量形状，使其更符合直观需求（例如从 [1, 3, 1, 4] 变为 [3, 4]）。

**语法**

```python
torch.squeeze(input, dim=None)
```

- `input`: 输入的张量。
- `dim`（可选）: 指定移除的维度。如果该维度大小不为1，则不执行任何操作。
  - 若未指定 `dim`，则移除**所有大小为1的维度**。

**示例**

```python
import torch

# 原始张量
x = torch.tensor([[[1, 2, 3]]])  # 形状: [1, 1, 3]

# 移除所有大小为1的维度
x1 = torch.squeeze(x)  # 形状: [3]

# 仅移除第0维
x2 = torch.squeeze(x, 0)  # 形状: [1, 3]

# 第1维大小为1，可以移除
x3 = torch.squeeze(x, 1)  # 形状: [1, 3]

print(x1)  # tensor([1, 2, 3])
print(x2)  # tensor([[1, 2, 3]])
print(x3)  # tensor([[1, 2, 3]])
```

**`unsqueeze` 和 `squeeze` 的组合使用**

```python
import torch

x = torch.tensor([1, 2, 3])  # 形状: [3]

# 增加一个维度
x = torch.unsqueeze(x, 0)  # 形状: [1, 3]

# 再移除这个维度
x = torch.squeeze(x, 0)  # 形状: [3]

print(x)  # tensor([1, 2, 3])
```

**注意事项**

1. **`dim` 的范围**:
   - 对于 `unsqueeze`，`dim` 必须在 `[-input.dim()-1, input.dim()]` 之间。
   - 对于 `squeeze`，`dim` 的范围是 `[-input.dim(), input.dim()-1]`。
   
2. **`squeeze` 不会改变非大小为1的维度**:
   - 例如，若尝试 `torch.squeeze(x, 2)` 对形状 `[3, 4, 5]` 的张量移除大小为1的第2维，不会产生任何变化，因为第2维大小为5。

**总结**

- **`unsqueeze`** 用于**添加**一个大小为1的维度。
- **`squeeze`** 用于**移除**大小为1的维度。
- 它们经常配合使用来调整张量形状以适配网络或计算需求。

### ==其他操作==

### torch.clip()-截断张量

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

### torch.flip()-翻转张量

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

- `torch.flip` 适用于任意维度的张量。**它不会改变原始张量的形状，但会在指定维度上反转数据。**
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
