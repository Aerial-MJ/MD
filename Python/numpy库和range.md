## linspace

在 Python 中，`numpy.linspace()` 函数可以创建一个在指定范围内等间隔的一维数组，该函数的作用是在指定的范围内，生成一个等差数列的一维数组。

函数的语法如下：

```python
numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
```

其中：

- `start`：序列的起始值；
- `stop`：序列的结束值；
- `num`：生成的等间隔样例数量，默认为50；
- `endpoint`：序列中是否包含 `stop` 值，默认为 True，即包含 `stop` 值；
- `retstep`：是否返回样例之间的步长，默认为 False，不返回；
- `dtype`：输出数组的数据类型，默认为 None；
- `axis`：沿着哪个轴生成样例，默认为 0。

使用 `numpy.linspace()` 函数，我们可以创建一个指定长度和范围的等间隔一维数组，从而方便进行数值计算和可视化。例如，如果我们要生成一个从0到10之间，包含11个数的等间隔一维数组，可以使用以下代码：

```python
import numpy as np

a = np.linspace(0, 10, 11)

print(a)
```

输出：

```text
[0  1  2  3  4  5  6  7  8  9  10]
```

上述代码中，`start` 参数为0，`stop` 参数为10，`num` 参数为11，因此生成的一维数组包含了从0到10的11个等间隔的数值。

除了 `numpy.linspace()` 函数外，还有一个类似的函数是 `numpy.arange()`，它可以在指定范围内生成等差数列的一维数组，但是可以指定步长。相比之下，`numpy.linspace()` 函数生成的数组样例数量是固定的，而步长是根据范围和样例数量自动计算的。

## arange

`numpy.arange()` 是一个用于生成等间隔数值的一维数组的函数。它的用法类似于 Python 内置的 `range()` 函数，但可以生成浮点数序列。`numpy.arange()` 函数的语法如下：

```python
numpy.arange([start, ]stop, [step, ]dtype=None)
```

其中：

- `start`：可选参数，表示序列的起始值，默认为0；
- `stop`：必需参数，表示序列的结束值，不包含在序列中；
- `step`：可选参数，表示序列中相邻两个数之间的间隔，默认为1；
- `dtype`：可选参数，表示序列的数据类型。

`numpy.arange()` 函数返回一个一维数组，包含从 `start` 开始，以 `step` 为间隔，直到 `stop` 之前的一组数值。其中，序列中的最后一个元素为**小于** `stop` 的最大数值，也就是说，如果序列中的最后一个元素加上 `step` 之后**大于等于** `stop`，则该元素不包含在序列中。

例如，如果我们要生成一个从0到9之间，步长为2的一维数组，可以使用以下代码：

```python
import numpy as np

a = np.arange(0, 10, 2)

print(a)
```

输出：

```text
[0 2 4 6 8]
```

当 `start` 和 `step` 参数都不指定时，默认生成从0开始，步长为1的整数序列。

```python
import numpy as np

a = np.arange(5)

print(a)
```

输出：

```text
[0 1 2 3 4]
```

当 `dtype` 参数指定为浮点数时，可以生成等间隔的浮点数序列。

```python
import numpy as np

a = np.arange(0, 1, 0.1)

print(a)
```

输出：

```text
[0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
```

需要注意的是，由于浮点数的表示精度限制，`numpy.arange()` 生成的浮点数序列的最后一个元素可能会略微偏离指定的结束值。如果需要精确地控制序列的结束值，建议使用 `numpy.linspace()` 函数。

## range

单独的类型：`#<class 'range'>`

```python
for i in range(2, 6):
    print(i)
```

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(type(a))  # 输出：<class 'numpy.ndarray'>
print(len(a))  # 输出：5
print(a.size)  # 输出：5

a1=[1,2,3,4]
print(type(a1))  #<class 'list'>

print(type(range(5)))  #<class 'range'>
```



