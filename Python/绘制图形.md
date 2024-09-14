# matplotli

如何将 `sin(x)` 和 `cos(x)` 画成两个独立的图像：

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 第一个图像
plt.figure()  # 创建新的图形窗口
plt.plot(x, y1, 'r')  # 绘制红色曲线
plt.grid(True)  # 显示网格线
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('sin(x)')

# 第二个图像
plt.figure()  # 创建另一个新的图形窗口
plt.plot(x, y2, 'b--')  # 绘制蓝色虚线
plt.grid(True)  # 显示网格线
plt.xlabel('x')
plt.ylabel('cos(x)')
plt.title('cos(x)')

plt.show()  # 显示图形窗口
```

在这个示例中，我们使用了 `plt.figure()` 来创建新的图形窗口。第一个 `plt.figure()` 创建了一个新的图形窗口，并在其中绘制了 `'sin(x)'` 的曲线。然后，第二个 `plt.figure()` 创建了另一个新的图形窗口，并在其中绘制了 `'cos(x)'` 的曲线。

通过多次调用 `plt.figure()`，你可以创建多个独立的图形窗口，并在每个窗口中进行绘图和设置。每个图形窗口都是独立的，它们可以单独进行交互和操作。

请注意，以上示例中的代码假定你已经安装了 **NumPy 和 Matplotlib 库**。如果没有安装，你可以使用以下命令在 Python 环境中安装它们：

```shell
pip install numpy matplotlib
```

这样就可以使用上述代码来运行并绘制两个独立的图形窗口了。

## plt.legend

`plt.legend()` 函数用于在 Matplotlib 图形中**显示图例**。图例是用于标识不同曲线、图形或数据集的说明性元素。

当你在 Matplotlib 中绘制多个曲线或图形时，你可以使用 `label` 参数为每个曲线或图形指定一个标签。然后，通过调用 `plt.legend()` 函数，你可以在图形中自动创建一个图例，并将每个曲线或图形与其对应的标签关联起来。

如何使用 `plt.legend()` 函数显示图例：

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, 'r', label='sin(x)')  # 绘制红色曲线，并指定标签
plt.plot(x, y2, 'b--', label='cos(x)')  # 绘制蓝色虚线，并指定标签
plt.grid(True)  # 显示网格线
plt.xlabel('x')
plt.ylabel('y')
plt.title('sin(x) and cos(x)')
plt.legend()  # 显示图例

plt.show()  # 显示图形
```

在这个示例中，我们在绘制 `'sin(x)'` 和 `'cos(x)'` 的曲线时，分别使用 `label` 参数指定了标签。然后，通过调用 `plt.legend()` 函数，我们将这些标签与相应的曲线关联起来。当我们运行代码并显示图形时，将显示一个图例，其中包含每个曲线的标签和相应的颜色或线型信息。

图例可以帮助阐明图形中的数据和曲线之间的对应关系，使图形更具可读性和说明性。你可以自定义图例的位置、样式和其他属性，以满足特定需求。

## plt.hold

在最新版本的 Matplotlib 中，`hold` 功能已经被移除，并且默认情况下会自动进行图形叠加。因此，在 Matplotlib 中，无需使用 `hold` 或类似的命令来绘制多个图形。

在 Matplotlib 中绘制 `'sin(x)'` 和 `'cos(x)'` 两个图像：

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, 'r', label='sin(x)')  # 绘制红色曲线
plt.plot(x, y2, 'b--', label='cos(x)')  # 绘制蓝色虚线
plt.grid(True)  # 显示网格线
plt.xlabel('x')
plt.ylabel('y')
plt.title('sin(x) and cos(x)')
plt.legend()  # 显示图例

plt.show()  # 显示图形
```

在这个示例中，我们使用 `plt.plot()` 函数两次来分别绘制 `'sin(x)'` 和 `'cos(x)'` 的图像。每次调用 `plt.plot()` 都会在同一张图中添加一个新的曲线。我们还使用 `plt.grid(True)` 显示网格线，以及 `plt.xlabel()`、`plt.ylabel()` 和 `plt.title()` 来设置轴标签和图标题。最后，我们使用 `plt.legend()` 显示图例。

`plt.figure() ` 创建另一个新的图形窗口

## 拟合曲线

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成原始数据
x = np.linspace(-10, 10, 100)
y = np.sin(x)

# 随机生成一些噪声数据
noise = np.random.normal(0, 0.1, size=len(x))
y_data = y + noise

# 构建模型，进行拟合
p = np.polyfit(x, y_data, 3)
# 使用三阶函数进行拟合
y_pred = np.polyval(p, x)

# 绘制散点图和拟合曲线
plt.scatter(x, y_data, label='data')
# 将散点图连接成线
plt.plot(x,y_data)
plt.plot(x, y_pred, 'r-', label='prediction')

# 添加图例
plt.legend(loc='upper left')

# 显示图像
plt.show()
```



