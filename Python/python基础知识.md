# Python基础

## python的循环

Python 中的循环结构主要有两种：`for` 循环和 `while` 循环。这两种循环各有其适用场景和特点。下面是对这两种循环的基本介绍和一些例子。

### 1. for 循环

`for` 循环在Python中用于遍历任何序列的项，如列表、元组、字典、字符串等，或者任何可迭代对象。

**基本语法**

```python
for variable in sequence:
    # 执行的代码块
```

**示例**

- 遍历列表：

  ```python
  fruits = ["apple", "banana", "cherry"]
  for fruit in fruits:
      print(fruit)
  ```

- 遍历字符串：

  ```python
  for char in "hello":
      print(char)
  ```

- 使用 `range()` 函数：

  ```python
  for i in range(5):  # 从 0 到 4
      print(i)
  ```

- 遍历字典：

  ```python
  person = {'name': 'John', 'age': 30}
  for key in person:
      print(key, person[key])
  ```

### 2. while 循环

`while` 循环在Python中用于在满足某条件的情况下重复执行一个代码块。

**基本语法**

```python
while condition:
    # 执行的代码块
```

**示例**

- 基本 `while` 循环：

  ```python
  count = 0
  while count < 5:
      print(count)
      count += 1  # 重要：确保有一个使条件最终变为 False 的语句
  ```

### 控制语句

在这两种类型的循环中，你可能会用到一些控制语句来修改循环的执行行为：

- `break`：立即退出整个循环体。
- `continue`：跳过当前循环的剩余部分，并开始下一次循环迭代。
- `else`：只有在循环正常结束时（没有通过 `break` 语句退出循环时），才执行的代码块。

#### 使用 break和 continue

```python
for i in range(10):
    if i == 3:
        continue  # 跳过数字 3
    if i == 8:
        break     # 当 i 等于 8 时终止循环
    print(i)
```

#### 循环的 else 子句

```python
for i in range(5):
    print(i)
else:
    print("Done!")  # 在循环正常结束后执行（未通过 break 退出）
```

这些基本的循环结构和控制语句构成了Python编程中处理重复任务的核心。每种循环有其特定的应用场景，比如当你需要遍历一个集合的元素时，`for` 循环非常适用；而当你需要在某个条件不再为真时停止循环时，`while` 循环则更加合适。

## python的类



## os.path.join

`os.path.join()` 是 Python 中 `os.path` 模块提供的一个方法，用于连接路径字符串。

在 Python 中，路径字符串可能会因操作系统的不同而有所不同，比如在 Unix/Linux 系统中，路径分隔符是斜杠 `/`，而在 Windows 系统中是反斜杠 `\`。为了使代码在不同操作系统上具有跨平台的兼容性，我们需要使用 `os.path.join()` 方法来连接路径，它会根据当前操作系统的规则自动地将多个路径片段连接起来，并返回一个规范化后的路径字符串。

以下是 `os.path.join()` 方法的基本用法：

```python
import os

# 连接路径
path = os.path.join('/path', 'to', 'directory', 'file.txt')

print(path)
```

在这个示例中，`os.path.join()` 方法将会返回一个合并后的路径字符串，它会自动根据当前操作系统的规则来连接路径片段。例如，在 Unix/Linux 系统中，返回的路径字符串可能是 `'/path/to/directory/file.txt'`，而在 Windows 系统中可能是 `'C:\\path\\to\\directory\\file.txt'`。

使用 `os.path.join()` 方法可以确保我们在拼接路径时遵循正确的规范，从而避免一些潜在的问题。

## 路径前面加r

在 Python 中，当字符串的前面加上 `r`（或者 `R`）时，表示该字符串是一个**原始字符串（raw string）**。原始字符串中的反斜杠 `\` 将被视为普通字符而不是转义字符。

通常情况下，在字符串中使用反斜杠 `\` 会被解释为转义字符，例如 `\n` 表示换行符，`\t` 表示制表符等。但是在原始字符串中，反斜杠 `\` 不会被解释为转义字符，而会保留原始的字面值。

例如，考虑以下两个字符串：

```python
path1 = 'C:\\Users\\Username\\Documents'
path2 = r'C:\Users\Username\Documents'
```

在 `path1` 中，反斜杠 `\` 被用作转义字符，因此 `\\` 表示一个单独的反斜杠。而在 `path2` 中，字符串被标记为原始字符串，所以 `\` 不会被解释为转义字符。

因此，`path1` 和 `path2` 表示的路径是相同的，但是在写代码时，使用原始字符串可以使路径更加清晰、简洁，特别是当路径包含大量反斜杠时。

## python的IO操作

### 1.read ＆ write

`imread()` 和 `imwrite()` 是图像处理库（如OpenCV）中的函数，用于读取和写入图像文件。而 `read()` 和 `write()` 是一般文件操作的函数，用于读取和写入文本文件或二进制文件。

主要区别在于它们处理的数据类型和所操作的对象：

1. **imread() 和 imwrite():**
   - `imread()` 用于读取图像文件，并将其加载为图像对象，通常是像素数组。
   - `imwrite()` 用于将图像对象保存到文件中。
   - 这两个函数是专门针对图像文件的读取和写入的，通常用于图像处理任务。

```python
import cv2

# 读取图像文件
image = cv2.imread('image.jpg')

# 处理图像

# 写入图像文件
cv2.imwrite('output.jpg', image)
```

2. **read() 和 write():**
   - `read()` 通常用于从文本文件或二进制文件中读取数据。
   - `write()` 通常用于将数据写入文本文件或二进制文件。
   - 这两个函数可以用于处理各种文件类型，包括文本文件、二进制文件等。

```python
# 读取文本文件
with open('text.txt', 'r') as file:
    data = file.read()

# 处理数据

# 写入文本文件
with open('output.txt', 'w') as file:
    file.write(data)
```

因此，`imread()` 和 `imwrite()` 专门用于处理图像文件，而 `read()` 和 `write()` 则是通用的文件操作函数，可用于处理各种文件类型。

### 2.使用 Pillow 处理图像

使用`read()`和`show()`函数，这在Python中通常不是直接关联到OpenCV的。然而，这些函数名可能是你在使用其他库时遇到的，如`PIL`/`Pillow`，一个用于图像处理的库。在`Pillow`中，可以用`open()`方法来读取图像，并使用`show()`方法来显示图像。这里我将说明如何在`Pillow`中使用这些方法。

以下是如何使用`Pillow`（也就是 PIL，Python Imaging Library 的更新版本）来读取和显示图像的步骤：

使用 `open()` 和 `show()` 显示图像

这里是如何使用 Pillow 来读取和显示图像：

```python
from PIL import Image

# 使用 Image.open() 读取图像
image = Image.open('your_image.jpg')  # 确保提供你的图像文件的正确路径

# 使用 show() 方法显示图像
image.show()
```

### 3.Matplotlib imshow() 方法

- plt.imshow()： plt.imshow()用于显示图像数据或二维数组（也可以是三维数组，表示RGB图像）。当你有一个二维数组或图像数据时，你可以使用plt.imshow()将其可视化为图像。它将数组中的每个元素的值映射为一个颜色，并将这些颜色排列成图像的形式。plt.imshow()可以接受许多参数，用于控制图像的外观，例如颜色映射（colormap）、插值方法等。
- plt.imshow()用于显示图像数据或二维数组（也可以是三维数组，表示RGB图像）。
- 当你有一个二维数组或图像数据时，你可以使用plt.imshow()将其可视化为图像。
- 它将数组中的每个元素的值映射为一个颜色，并将这些颜色排列成图像的形式。
- plt.imshow()可以接受许多参数，用于控制图像的外观，例如颜色映射（colormap）、插值方法等。
- plt.show()： plt.show()用于显示所有已创建的图形。在使用Matplotlib绘制图形时，图形被存储在内存中，但不会自动显示在屏幕上。为了在屏幕上显示图形，你需要调用plt.show()函数。通常，在你创建完所有的图形之后，调用plt.show()一次，它会同时显示所有的图形窗口。

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
# plt.legend()  # 显示图例

plt.show()  # 显示图形
```

### matplotlib和pillow两个库的异同

`Matplotlib` 和 `Pillow` 是 Python 中两个常用的图像处理库，它们在功能上有所重叠，但也有明显的区别。下面是对这两个库的比较。

#### Matplotlib

**主要特点：**

1. **数据可视化工具：**
   - `Matplotlib` 是一个广泛用于创建静态、动态和交互式图表的绘图库，尤其在数据科学、机器学习和科学计算中非常流行。它能够生成各种类型的图表，如折线图、散点图、柱状图、饼图等。

2. **图形绘制功能：**
   - 除了数据可视化，`Matplotlib` 还可以用于绘制矢量图形、形状、文本等，允许在图像上叠加绘制不同元素。

3. **图像处理能力：**
   - `Matplotlib` 也具备一定的图像处理能力，可以加载、显示、修改和保存图像。尽管如此，它的图像处理功能并不是非常强大或全面，更专注于图表的创建和展示。

4. **绘图接口：**
   - 提供了面向对象的绘图接口 (`Figure`, `Axes`)，以及更加高层的 `pyplot` 接口，后者使得绘图更加简单和类似于 MATLAB。

**主要用途：**
   - 数据可视化：绘制各种类型的图表。
   - 科学计算中的图像展示：显示数据处理或分析中的结果。
   - 在数据图表上叠加文本、线条、形状等元素。

#### Pillow

**主要特点：**

1. **图像处理库：**
   - `Pillow` 是 Python Imaging Library (PIL) 的分支，是一个专门用于图像处理的库。它能够进行图像加载、显示、编辑、保存等各种操作。

2. **广泛的图像格式支持：**
   - 支持多种图像格式，如 JPEG、PNG、GIF、TIFF、BMP 等，可以读取和保存不同格式的图像文件。

3. **丰富的图像操作：**
   - 提供了大量图像处理功能，如图像缩放、裁剪、旋转、滤镜应用、颜色调整、图像合成等。
   - 支持对单个像素或像素块进行操作，使得用户可以轻松地处理图像的各个部分。

4. **绘图功能：**
   - `Pillow` 也提供了绘图功能，可以在图像上绘制文本、形状、线条等，但这些功能更多地用于图像的编辑而非数据可视化。

**主要用途：**
   - 图像的加载、处理、编辑和保存。
   - 图像格式的转换。
   - 图像处理和增强（例如应用滤镜、改变亮度/对比度）。
   - 在图像上添加文本、形状等。

#### 异同点总结

**相似之处：**
- **图像显示**：两者都可以加载和显示图像。
- **绘图功能**：两者都可以在图像或图表上绘制文本、形状和线条。
- **易用性**：两者都有相对简单的 API，使得图像处理或绘图变得容易。

**不同之处：**

- **主要用途**：
  - `Matplotlib` 更侧重于 **数据可视化**，用于绘制各类数据图表。
  - `Pillow` 专注于 **图像处理**，用于对图像进行各种操作和编辑。
  
- **功能范围**：
  - `Matplotlib` 强大于 **科学计算和数据展示**，有丰富的图表绘制选项和定制能力。
  - `Pillow` 强大于 **图像的编辑和格式转换**，支持对图像的各种底层操作。

- **使用场景**：
  - `Matplotlib` 在数据分析、科研、机器学习等领域广泛使用。
  - `Pillow` 在图像处理、数字媒体、图形编辑等领域更为常见。

#### 什么时候使用？
- 如果你需要绘制图表、展示数据或在图表上叠加绘图，选择 `Matplotlib`。
- 如果你需要加载图像、处理图像、保存图像或在图像上添加一些基本的绘制，选择 `Pillow`。

## cv2.waitKey功能详解：

在 OpenCV 中，`cv2.waitKey()` 函数是非常重要的一个函数，用于处理窗口中的事件，特别是键盘事件。该函数的目的是让程序暂停一定时间，等待用户进行键盘输入，其参数指定了等待时间的长度，单位为毫秒。

- **参数**：该函数接受一个整数值（毫秒）作为参数。如果参数是正数，函数会等待指定的毫秒数，期间如果有按键被按下，函数将立即结束，并返回按键的 ASCII 值。如果在指定时间内没有任何按键被按下，函数最终会返回 -1。
- **参数为0**：如果传入的参数为0，那么 `cv2.waitKey(0)` 会无限期地等待用户的按键事件。这在你需要在某个窗口中展示一个静态图像，直到用户做出响应（如按下任意键）时非常有用。
- **返回值**：该函数返回按下键的 ASCII 码值。这个特性允许程序根据用户的按键输入做出响应。例如，可以设置当用户按下“q”键（ASCII 值为 113）时，程序结束运行。

在图像或视频处理应用中，`cv2.waitKey()` 通常与 `cv2.imshow()` 配合使用，以创建一个简单的图像查看器，如下所示：

```python
import cv2

# 读取图像
image = cv2.imread('your_image.jpg')

# 显示图像
cv2.imshow('Image Window', image)

# 等待键盘输入，等待时间为无限长
key = cv2.waitKey(0)

# 根据按键决定后续行为
if key == 27:  # ASCII for ESC
    cv2.destroyAllWindows()
elif key == ord('s'):  # 按 's' 键保存并退出
    cv2.imwrite('saved_image.jpg', image)
    cv2.destroyAllWindows()
```

在这个示例中，`cv2.waitKey(0)` 使程序暂停，直到用户按下一个键。如果按下的是 Esc 键（其 ASCII 码为 27），程序将关闭所有窗口；如果按下的是 's' 键，则图像会被保存后程序关闭所有窗口。

由于 `cv2.waitKey()` 返回的是整数值，有时需要通过与 `0xFF` 进行位与操作来获取实际的 ASCII 码，特别是在某些环境下（如64位机器）：

```python
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
```

这种用法确保从函数返回的结果只取低8位，这对于兼容性是很有帮助的。

## 正则表达式

### 将反斜杠转化成正斜杠

```python
import os
import re

def find_and_replace_in_file(file_path):
    # 用于检测图片链接的正则表达式
    image_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
    changed = False
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(file_path, 'w', encoding='utf-8') as file:
        for line in lines:
            # 查找带有图片的行
            if image_pattern.search(line):
                original_line = line
                # 查找并替换路径中的反斜杠为正斜杠
                line = re.sub(r'\\', '/', line)
                if line != original_line:
                    changed = True
            file.write(line)
    return changed

def find_md_files_and_replace(directory):
    # 遍历目录中的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                if find_and_replace_in_file(file_path):
                    print(f"Updated image paths in {file_path}")

# 调用函数，需替换为目标文件夹路径
directory_path = r'C:\Users\19409\Desktop\MD'
find_md_files_and_replace(directory_path)

```

在Python中，使用正则表达式主要依赖于内置的 `re` 模块。这个模块提供了一系列功能，用于字符串搜索、匹配、替换等操作。下面我将向你展示如何创建和使用Python中的正则表达式，以及一些常用的功能。

### 1. 导入 re 模块

在使用正则表达式之前，需要首先导入Python的 `re` 模块：

```python
import re
```

### 2. 编译正则表达式

虽然直接使用正则表达式的方法（如 `re.search()` 或 `re.match()`）时可以直接提供模式字符串，但为了效率和重用，通常先将正则表达式编译成一个正则表达式对象。这可以通过 `re.compile()` 完成：

```python
pattern = re.compile(r'\d+')  # 匹配一个或多个数字
```

### 3. 使用正则表达式

编译后的正则表达式对象可以使用多种方法，比如：

- `match()`：从字符串的开始处进行匹配检查。
- `search()`：在整个字符串中搜索第一次出现的模式。
- `findall()`：查找字符串中所有匹配的子串，并返回它们作为一个列表。
- `finditer()`：查找字符串中所有匹配的子串，并返回一个迭代器。
- `sub()`：替换字符串中的匹配项。

**示例**

下面是一些基本用法的例子：

```python
import re

# 编译正则表达式
pattern = re.compile(r'\d+')

# 在字符串中搜索数字
match = pattern.search("Hello 1234 World")
if match:
    print("Found:", match.group())  # 输出第一个匹配的结果

# 查找所有匹配的数字
numbers = pattern.findall("Example with 123 numbers 456 and 789")
print("Numbers found:", numbers)  # 输出所有匹配的结果

# 替换字符串中的数字为 #
replaced = pattern.sub("#", "Example with 123 numbers 456 and 789")
print("Replaced string:", replaced)

# 使用迭代器找到所有匹配项
for m in pattern.finditer("Example with 123 numbers 456 and 789"):
    print("Match at position:", m.start(), m.group())
```

### 4. 常见模式

- `\d`：匹配任何十进制数字，等价于 [0-9]。
- `\D`：匹配任何非数字字符。
- `\w`：匹配任何字母数字字符，等价于 [a-zA-Z0-9_]。
- `\W`：匹配任何非字母数字字符。
- `\s`：匹配任何空白字符，包括空格、制表符、换页符等等。
- `\S`：匹配任何非空白字符。
- `.`：匹配除换行符以外的任何单个字符。

这些基础概念和方法是使用Python正则表达式的基石，可以用于处理各种复杂的文本处理任务。

## 删除不匹配文件

这个脚本将比较两个文件夹内的文件名（不包括扩展名），删除那些没有匹配的文件。

请确保在运行此脚本之前已经安装了Python，并具有操作文件和文件夹的权限。这个脚本使用了`os`模块来操作文件系统。

```python
import os

def sync_folders(images_folder, xmls_folder):
    # 获取两个文件夹中所有文件的名称（不包含扩展名）
    images = {os.path.splitext(file)[0]: file for file in os.listdir(images_folder) if file.endswith('.jpg')}
    xmls = {os.path.splitext(file)[0]: file for file in os.listdir(xmls_folder) if file.endswith('.xml')}

    # 找出所有不匹配的图片文件
    images_to_delete = set(images) - set(xmls)
    # 找出所有不匹配的XML文件
    xmls_to_delete = set(xmls) - set(images)

    # 删除不匹配的图片文件
    for img in images_to_delete:
        os.remove(os.path.join(images_folder, images[img]))
        print(f"Deleted image: {images[img]}")

    # 删除不匹配的XML文件
    for xml in xmls_to_delete:
        os.remove(os.path.join(xmls_folder, xmls[xml]))
        print(f"Deleted XML: {xmls[xml]}")

# 设置文件夹路径
images_folder = 'path/to/images_folder'  # 替换为你的图片文件夹路径
xmls_folder = 'path/to/xmls_folder'  # 替换为你的XML文件夹路径

# 调用函数
sync_folders(images_folder, xmls_folder)
```

1. 将 `images_folder` 和 `xmls_folder` 变量的值替换为你实际图片文件夹和XML文件夹的路径。
2. 确保图片文件的扩展名是 `.jpg`（你可以根据实际情况修改脚本中的文件扩展名）。
3. 运行脚本。

该脚本会删除任何没有在另一文件夹中找到匹配文件名（忽略扩展名）的文件。请在运行此脚本之前确保备份你的数据，以防意外删除重要文件。

```
{key_expression: value_expression for item in iterable if condition}
```

- `key_expression`：字典键的表达式。
- `value_expression`：字典值的表达式。
- `iterable`：要迭代的可迭代对象（如列表、元组、字符串等）。
- `condition`（可选）：一个条件表达式，用于过滤哪些元素将包含在字典中。

### 代码解释

#### First Question

`os.path.splitext(file)[0]: file for file in os.listdir(images_folder) if file.endswith('.jpg') `

这段代码使用了Python字典推导式来创建一个字典，其中包含了指定文件夹中特定文件类型（在这里是`.jpg`格式的图片）的文件名（不含扩展名）作为键，完整文件名作为值。下面是这段代码的详细解释：

1. `os.listdir(images_folder)`: 这个函数调用会列出给定文件夹（`images_folder`）中的所有文件和目录名。这里返回的是一个字符串列表。

2. `file for file in os.listdir(images_folder) if file.endswith('.jpg')`: 这部分是一个列表推导式，它遍历`os.listdir(images_folder)`生成的列表。对于每一个元素（即文件名），检查文件名是否以`.jpg`结尾。如果是，这个文件名会被包含在生成的列表中。这样可以过滤出所有`.jpg`格式的图片文件。

3. `os.path.splitext(file)[0]: file`: 这部分是字典推导式的核心。对于上一步过滤出的每一个`.jpg`文件名`file`，`os.path.splitext(file)`会将文件名分割成两部分：文件名（不含扩展名）和扩展名。`os.path.splitext(file)[0]`就是获取文件名不包含扩展名的部分。这部分作为字典的键，而完整的文件名`file`（包括扩展名）作为字典的值。

整个字典推导式构建了一个字典，其中键是文件名不包括扩展名的部分，值是包含扩展名的完整文件名。这样的结构便于之后的代码查找和比较文件名（不考虑扩展名），以决定是否需要删除文件。

```python
my_dict = {'a': 1, 'b': 2, 'c': 3}
print(my_dict)  # 输出 {'a': 1, 'b': 2, 'c': 3}

set(my_dict)
{'a', 'b', 'c'}
```

#### Second Question

在Python中，`file for file in os.listdir(images_folder) if file.endswith('.jpg')` 这段代码是一个列表推导式（list comprehension），用于从指定文件夹中筛选出所有以 `.jpg` 结尾的文件名。这里详细解释每个部分的意义：

1. `os.listdir(images_folder)`：这是 `os` 模块的一个函数，用于列出给定目录（`images_folder`）下的所有文件和子目录的名字，结果是一个包含字符串的列表。

2. `for file in os.listdir(images_folder)`：这是一个循环，它遍历 `os.listdir(images_folder)` 返回的每一个元素，每个元素代表文件夹中的一个文件或目录名，这里的每个元素被赋值给变量 `file`。

3. `if file.endswith('.jpg')`：这是一个条件语句，用于检查变量 `file`（代表一个文件名）是否以字符串 `.jpg` 结尾。这个方法返回布尔值（True 或 False），如果文件名以 `.jpg` 结尾则为 True，否则为 False。

4. `file`：这部分是列表推导式的输出部分，如果 `file.endswith('.jpg')` 的结果为 True，`file` 的值就会被加入到最终生成的列表中。

整体来说，这行代码的作用是生成一个新的列表，这个列表中包含了 `images_folder` 文件夹下所有以 `.jpg` 结尾的文件名。这是一个非常高效的方法来筛选特定类型的文件。

如果你有一个文件夹 `images_folder`，里面包含了多个文件，其中一些是 JPG 图片，你可以使用这样的代码来找出所有的 JPG 文件：

```python
import os

images_folder = '/path/to/your/folder'
jpg_files = [file for file in os.listdir(images_folder) if file.endswith('.jpg')]

print(jpg_files)
```

这段代码将打印出 `images_folder` 中所有以 `.jpg` 结尾的文件名的列表。如果文件夹路径或文件扩展名不同，你需要根据实际情况调整路径和扩展名。

## 将.xml中\<path\>标签去掉中文

要批量修改文件夹内所有 XML 文件中的 `<path>` 标签内容，并去除路径中的特定中文字符（例如 "副本"），你可以使用 Python 的 xml.etree.ElementTree 模块来解析和修改 XML 文件，然后使用 os 和 shutil 模块来遍历文件和保存修改后的文件。下面是一个简单的 Python 脚本示例，展示了如何完成这一任务：

```python
import os
import xml.etree.ElementTree as ET

def remove_chinese_from_path(xml_folder):
    for filename in os.listdir(xml_folder):
        if filename.endswith('.xml'):
            filepath = os.path.join(xml_folder, filename)
            tree = ET.parse(filepath)
            root = tree.getroot()

            # 遍历所有的 path 元素
            for path in root.iter('path'):
                original_path = path.text
                # 替换掉路径中的 '副本'
                modified_path = original_path.replace(' - 副本', '')
                path.text = modified_path

            # 保存修改后的 XML 文件
            tree.write(filepath, encoding='utf-8', xml_declaration=True)

# 调用函数，指定 XML 文件所在的文件夹路径
xml_folder_path = 'C:\\Your\\Folder\\Path'
remove_chinese_from_path(xml_folder_path)
```

注意事项：
1. 确保你的 Python 环境已经安装了 xml.etree.ElementTree 模块（通常 Python 自带此模块，无需额外安装）。
2. 脚本中的 `xml_folder_path` 需要替换成你存放 XML 文件的实际文件夹路径。
3. 本脚本仅针对每个 XML 文件中的 `<path>` 元素进行查找并替换指定的字符串（本例中是 "副本"）。如果你需要修改其他字符串，可以相应地修改 `replace` 函数的参数。

在运行此脚本之前，请确保备份原始的 XML 文件以防万一，避免不可恢复的数据丢失。如果你的 XML 文件结构比较复杂或者有特殊的需求，你可能需要根据实际情况调整代码。

## OCR-pytesseract识别

```python
import cv2
from PIL import Image

import pytesseract


def Corver_Gray(image_path):
    # 读取模板图像
    img = cv2.imread(image_path)

    # 转换为灰度图 也可读取时直接转换
    ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值图像
    ref = cv2.threshold(ref, 60, 255, cv2.THRESH_BINARY_INV)[1]

    return ref


def Read_Img(img_path):
    image = Corver_Gray(img_path)
    image = cv2.imwrite("test.png", image)
    return image


Read_Img(r"C:\Users\19409\Desktop\data\yolov5-5.0\runs\detect\exp7\plates\test_60_1_1.jpg")


text = pytesseract.image_to_string(Image.open("test.png"))
print(text)
```

