# Python基础

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

## 1、删除不匹配文件

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

### First Question

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

### Second Question

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

## 2、将.xml中\<path\>标签去掉中文

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

## 3、OCR-pytesseract识别

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

## 4、将反斜杠转化成正斜杠

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

## 5、将图片的路径修改成相对路径

```python
import os
import re

def replace_with_relative_image_path(file_path, image_root):
    image_pattern = re.compile(r'!\[(.*?)\]\((?:file:///)?(?:[A-Za-z]:)?[/\\]+(?:.*?[/\\])*Image[/\\]([^)]+)\)')
    # 匹配 Markdown 中类似这样的图片路径：
    # ![描述](C:/Users/xxx/Desktop/MD/Image/xxx.png)
    """
    
    """


    changed = False
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 打开一个 Markdown 文件，读取它的全部内容，存在 content 变量中。
    # encoding='utf-8' 确保读取中文不会乱码。


    # 计算当前md文件相对image文件夹的路径
    md_dir = os.path.dirname(file_path)
    relative_image_path = os.path.relpath(image_root, md_dir).replace('\\', '/')
    #md_dir 获取当前 Markdown 文件所在的目录路径。
    #os.path.relpath(image_root, md_dir)   计算从 md_dir 到 image_root 的相对路径。
    #.replace('\\', '/') 是为了统一使用正斜杠（Markdown 里更标准）。


    # 替换图片路径为相对路径
    def replacer(match):
        image_name = match.group(2).replace('\\', '/')
        return f'![{match.group(1)}]({relative_image_path}/{image_name})'

        #"""
        #match.group(1) 是图片描述，例如 image-xxx
        #match.group(2) 是图片路径名称，例如 img001.png
        #它会返回新的 Markdown 图片语法，例如：![image-xxx](../../Image/img001.png)
        #"""

    new_content, count = image_pattern.subn(replacer, content)

    if count > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {file_path} with {count} replacements.")
        changed = True

    return changed

def process_all_md_files(md_root, image_root):
    for root, _, files in os.walk(md_root):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                replace_with_relative_image_path(file_path, image_root)
                
#从 md_root 这个目录开始，递归地遍历它的所有子文件夹，每走到一个文件夹，就返回一组三个值。
#这三个值是：
#变量名	含义
#root	当前遍历到的这个文件夹的路径
#dirs（我们写成 _）	当前文件夹里的所有子文件夹的名字列表（我们这次不关心它，所以用 _ 忽略）
#files	当前文件夹里的所有文件的名字列表（注意：只是名字，不带路径）


# 修改成你的实际路径
md_root_dir = r'C:\Users\19409\Desktop\MD'
image_dir = os.path.join(md_root_dir, 'Image')

process_all_md_files(md_root_dir, image_dir)

```

### 正则对象

`image_pattern = re.compile( ... )

这行代码的返回值是一个**“正则表达式对象”**，也叫 re.Pattern 对象。这个对象能用来在字符串中查找、匹配和替换我们设定好的模式（**pattern**）。

### 正则的捕获组

正则中的 `()`：

- `()` 是捕获组：你可以用 `group(n)` 来取
- `(?:...)` 是非捕获组：不进入 `group(n)`，只用于结构分组或控制匹配

所以你这个正则：

- `group(1)` 是 `![描述]` 里的描述文字
- `group(2)` 是 `Image/xxx.png` 里的文件名（相对于 Image 文件夹）

### image_pattern

它是通过方法来发挥作用的，最常用的几个是：

| 方法名                               | 作用                         | 示例                                       |
| ------------------------------------ | ---------------------------- | ------------------------------------------ |
| `image_pattern.search(text)`         | 在 `text` 中查找第一个匹配项 | 返回一个 `match` 对象或 `None`             |
| `image_pattern.findall(text)`        | 查找所有匹配项，返回元组列表 | `[(描述1, 图片名1), (描述2, 图片名2)]`     |
| `image_pattern.sub(replacer, text)`  | 替换所有匹配项，返回新字符串 | 会把匹配的内容替换成 replacer 生成的新内容 |
| `image_pattern.subn(replacer, text)` | 同上，但额外返回替换的数量   | `(new_text, 替换次数)`                     |
