## python注解

你看到的是 **Python 类型注解 (type hinting)**，它用于为函数的参数和返回值提供类型信息。这是一种提高代码可读性、可维护性和减少错误的方法，特别是在大型项目或多人协作时很有用。尽管 Python 是动态类型语言，类型注解不会强制类型检查，但可以通过工具（如 `mypy`）进行静态类型检查。

### 语法：

- 参数类型注解：在函数参数后面使用冒号 `:` 指定参数的类型。
- 返回值类型注解：在函数名的参数列表后面使用箭头 `->` 指定返回值的类型。

### 示例：基本类型注解

```python
def add(a: int, b: int) -> int:
    return a + b
```

**解释**：

- `a: int` 和 `b: int` 表示参数 `a` 和 `b` 期望是 `int` 类型。
- `-> int` 表示该函数应该返回 `int` 类型的值。
- 这并不意味着你不能传递其他类型，但类型注解是一个提示，告诉开发者和静态检查工具预期的类型。

**类型注解的详细解释**：

- `a: int`：表示 `a` 期望是整数类型 `int`。
- `b: int`：表示 `b` 期望是整数类型 `int`。
- `-> int`：表示函数返回一个整数类型的值。

---

### 示例：使用不同的数据类型

类型注解也可以用于其他数据类型，如字符串、列表、字典等。

#### 返回字符串：

```python
def greet(name: str) -> str:
    return f"Hello, {name}"
```

**解释**：

- `name: str` 表示 `name` 参数应该是一个字符串。
- `-> str` 表示函数应该返回一个字符串。

#### 返回列表：

```python
def get_numbers() -> list[int]:
    return [1, 2, 3, 4, 5]
```

**解释**：

- `-> list[int]` 表示函数返回一个包含整数的列表。

#### 参数为字典：

```python
def process_data(data: dict[str, int]) -> None:
    for key, value in data.items():
        print(f"{key}: {value}")
```

**解释**：

- `data: dict[str, int]` 表示 `data` 参数应该是一个字典，键是字符串类型，值是整数类型。
- `-> None` 表示函数不返回任何值（返回类型是 `None`）。

---

### 示例：类型注解在类方法中

类型注解同样可以用于类的方法。

```python
class Person:
    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age

    def get_info(self) -> str:
        return f"{self.name} is {self.age} years old."
```

**解释**：

- `__init__(self, name: str, age: int) -> None`：构造函数的参数 `name` 和 `age` 分别期望是字符串和整数，而返回值是 `None`（因为构造函数没有显式返回值）。
- `get_info(self) -> str`：表示 `get_info` 方法返回一个字符串。

---

### 使用 Optional 类型：

如果参数可以是 `None`，可以使用 `Optional` 注解。

```python
from typing import Optional

def greet(name: Optional[str] = None) -> str:
    if name:
        return f"Hello, {name}"
    else:
        return "Hello, Guest"
```

**解释**：

- `name: Optional[str]` 表示 `name` 可以是 `str` 类型，也可以是 `None`。`Optional`中的类型只能是一个`only a single argument`

- 如果参数不传递或传递 `None`，函数仍然能处理。

  >在 Python 中，`name: Optional[str] = None` 是一种类型注解，它有两个部分需要解释：
  >
  >1. `Optional[str]`：表示这个参数 `name` 可以是 **`str` 类型的字符串**，或者也可以是 **`None`**。`Optional[T]` 的意思是参数可以是某个类型 `T`，也可以是 `None`。在这个例子中，`T` 是 `str`，因此 `name` 可以是字符串类型或者 `None`。
  >
  >2. `= None`：表示 `name` 的**默认值是 `None`**。如果在调用函数时没有提供这个参数的值，`name` 会默认赋值为 `None`。
  >
  >**具体例子**：
  >
  >```python
  >from typing import Optional
  >
  >def greet(name: Optional[str] = None) -> str:
  >   if name:
  >       return f"Hello, {name}"
  >   else:
  >       return "Hello, Guest"
  >```
  >
  >**解释**：
  >
  >- `name: Optional[str]`：这个参数 `name` 可以是一个字符串，也可以是 `None`。
  >- `= None`：如果调用函数时不传递 `name` 参数，`name` 的默认值就是 `None`。
  >
  >调用这个函数时，有两种情况：
  >
  >1. **传递字符串**：
  >
  >   ```python
  >   print(greet("Alice"))  # 输出: Hello, Alice
  >   ```
  >
  >     在这种情况下，`name` 是 `"Alice"`，函数返回 `"Hello, Alice"`。
  >
  >2. **不传递参数或传递 `None`**：
  >
  >   ```python
  >   print(greet())         # 输出: Hello, Guest
  >   print(greet(None))     # 输出: Hello, Guest
  >   ```
  >
  >     在这种情况下，`name` 是 `None`，因此函数会返回 `"Hello, Guest"`。
  >
  >### 总结：
  >
  >- `Optional[str]` 表示 `name` 可以是 `str` 类型或者 `None`。
  >- `= None` 表示 `name` 的默认值是 `None`，如果没有提供值，`name` 会自动赋值为 `None`。

---

### 常见类型注解：

- `int`：整数
- `float`：浮点数
- `str`：字符串
- `bool`：布尔类型（True 或 False）
- `list[T]`：列表，其中每个元素是类型 `T`
- `dict[K, V]`：字典，其中键是类型 `K`，值是类型 `V`
- `Optional[T]`：类型 `T` 或 `None`

### 静态类型检查工具：

尽管 Python 不会在运行时强制类型注解，但是你可以使用静态类型检查工具来检查代码是否符合类型注解，例如：

- `mypy`：可以静态检查 Python 代码，确保类型注解的正确性。

**示例：**

```bash
mypy my_script.py
```

`mypy` 会检查代码中的类型是否匹配注解。

## Generator

在 Python 中，`generator`（生成器）是一种用于生成一系列值的数据结构，与列表类似，但生成器在每次迭代时按需计算每个值，从而节省内存。生成器是一种 **惰性**（lazy）序列计算方式，即它不会在开始时就把所有的值计算出来，而是需要时才生成值。

### 1. 生成器的特点
- **惰性求值**：生成器仅在需要时才生成下一个值，而不是一次性生成所有值。
- **节省内存**：因为生成器并不把所有值存在内存中，而是逐步生成，非常适合处理大数据或无限序列。
- **迭代性**：生成器是可迭代的，可以用于 `for` 循环或其他需要迭代的地方。

### 2. 创建生成器的方式

#### 方法 1：使用生成器函数（`yield` 关键字）
生成器函数是带有 `yield` 关键字的普通函数，每次执行到 `yield` 时暂停，返回一个值，并在下一次调用时从暂停处继续执行。

**示例**：
```python
def my_generator():
    yield 1
    yield 2
    yield 3

gen = my_generator()
for value in gen:
    print(value)
```

**输出**：
```
1
2
3
```

**解释**：`my_generator` 函数在调用时不会立即执行，而是返回一个生成器对象。每次循环调用时，生成器都会执行到下一个 `yield` 处，并返回相应的值。之后继续执行到下一个 `yield`，直到没有值可返回为止。

#### 方法 2：使用生成器表达式
生成器表达式类似于列表推导式，但使用圆括号 `()` 而不是方括号 `[]`，从而返回一个生成器对象。

**示例**：

```python
gen = (x ** 2 for x in range(5))
for value in gen:
    print(value)
```

**输出**：
```
0
1
4
9
16
```

### 3. 与列表的区别

- **列表**：一次性计算并存储所有值，占用大量内存。
- **生成器**：按需生成值，不需要存储所有值，内存占用较小。

**示例**：

```python
# 使用列表
squares_list = [x ** 2 for x in range(1000000)]  # 大列表，占用较多内存

# 使用生成器
squares_gen = (x ** 2 for x in range(1000000))  # 生成器，按需生成，节省内存
```

在这种情况下，`squares_list` 会占用更多内存，而 `squares_gen` 几乎不占用内存，只有在每次迭代时才计算一个新的平方值。

```lua
# tensor1 = torch.randn(gen)
#TypeError: randn(): argument 'size' (position 1) must be tuple of ints, not generator
# print(tensor1)

print(type((row[::2] for row in matrix[:3]))) <class 'generator'>
print(type(row[::2] for row in matrix[:3]))   <class 'generator'>
print(type([row[::2] for row in matrix[:3]])) <class 'list'>
```

### 4. 使用生成器的场景
- **大数据处理**：需要处理大量数据时，用生成器可以避免一次性将数据加载到内存中。
- **无限序列**：生成器可以创建无限序列，因为它们在每次请求时生成新值。
- **流式数据处理**：对于流数据或按序列产生的数据（如从文件逐行读取），生成器可以逐步生成数据。

### 总结
生成器是一种高效的惰性计算方式，非常适合内存敏感或大数据处理任务。它们通过 `yield` 或生成器表达式 `(expression for item in iterable)` 生成值，是 Python 处理迭代任务的重要工具。

## 注解类型

`Optional[Sequence[Union[str, ellipsis, None]]]` 是一个 Python 类型注解，描述了一种复杂的数据类型。我们可以分解它来理解每个部分的意义：

### 逐步解析

1. `Optional[...]`：
   
   - `Optional[T]` 是 `Union[T, None]` 的一种简写。
   - 这里表示这个对象可以是指定类型 `Sequence[Union[str, ellipsis, None]]`，也可以是 `None`。
   
   所以：`Optional[Sequence[Union[str, ellipsis, None]]]` 表示这个对象可以是 `Sequence[Union[str, ellipsis, None]]` 或 `None`。
   
2. `Sequence[...]`：
   
   - `Sequence` 是一种通用的序列类型，例如 `list`、`tuple`，用于存放有序的元素。
   - 这里表示对象是一个序列（如列表或元组），其元素类型为 `Union[str, ellipsis, None]`。
   
3. `Union[str, ellipsis, None]`：
   
   - `Union` 表示元素可以是多种类型中的一种。
   - `Union[str, ellipsis, None]` 表示序列中的每个元素可以是 `str`（字符串类型）、`ellipsis`（省略符号，用作占位符的 `...`），或者 `None`。

### 组合在一起

`Optional[Sequence[Union[str, ellipsis, None]]]` 表示的类型可以是：

- `None`
- 或一个序列（如 `list`、`tuple`），其中每个元素可以是：
  - `str` 类型的字符串
  - `ellipsis`（即 `...`，占位符）
  - `None`

### 实例示例

```python
from typing import Optional, Sequence, Union

# 示例变量符合类型 Optional[Sequence[Union[str, ellipsis, None]]]
example1: Optional[Sequence[Union[str, ellipsis, None]]] = ["hello", "world", None, ...]
example2: Optional[Sequence[Union[str, ellipsis, None]]] = None
example3: Optional[Sequence[Union[str, ellipsis, None]]] = ["foo", ..., None]

print(example1)  # 输出: ['hello', 'world', None, Ellipsis]
print(example2)  # 输出: None
print(example3)  # 输出: ['foo', Ellipsis, None]
```

### 总结
`Optional[Sequence[Union[str, ellipsis, None]]]` 表示一个可以为 `None` 的序列，且序列中的每个元素可以是 `str` 类型的字符串、`ellipsis`（`...`），或 `None`。

在Python中，`Union`、`Sequence` 和 `Optional` 是`typing`模块中的工具，用于类型注解以提高代码的可读性、可靠性，并帮助工具（如IDE和静态类型检查器）进行更好的代码分析。

## 注解

### 1. **`Union`**
`Union`表示一个变量可以是多种类型中的一种。

#### 语法
```python
from typing import Union

def process_data(data: Union[int, str]) -> None:
    if isinstance(data, int):
        print(f"Processing integer: {data}")
    elif isinstance(data, str):
        print(f"Processing string: {data}")
```

#### 说明
- 在这个例子中，`data`可以是`int`或`str`类型。
- `Union[int, str]`的含义是：**类型要么是`int`，要么是`str`，不能是其他类型。**

---

### 2. **`Sequence`**
`Sequence`是泛型类型，表示任何支持序列操作的容器类型，例如`list`、`tuple`、`str`等。

#### 语法
```python
from typing import Sequence

def calculate_sum(numbers: Sequence[int]) -> int:
    return sum(numbers)

result = calculate_sum([1, 2, 3])  # list
result = calculate_sum((4, 5, 6))  # tuple
# calculate_sum("123")  # 类型检查器可能会提示错误
```

#### 说明
- `Sequence`泛型需要提供一个具体的元素类型，例如`Sequence[int]`表示包含`int`类型元素的序列。
- 常见序列类型有`list`、`tuple`等，不包括`set`或`dict`，因为它们不支持按索引访问。

>具体来说，Python 中的序列类型 (Sequence) 有以下特点：
>
>1. **有序性（Ordered）**
>    字符串中的每个字符都有一个固定的位置，可以通过索引访问，例如 `s[0]` 访问字符串 `s` 的第一个字符。
>2. **可迭代性（Iterable）**
>    字符串是可迭代的，可以在 `for` 循环中逐个访问其字符。
>3. **支持切片（Slicing）**
>    字符串支持切片操作，例如 `s[1:4]` 返回字符串从索引 1 到 3 的子字符串。
>4. **不可变性（Immutable）**
>    字符串是不可变的，一旦创建就无法更改其内容。例如，不能直接通过 `s[0] = 'a'` 修改字符串。
>5. **支持常见序列操作**
>    字符串支持许多常见的序列操作，如：
>  - **连接**：`"hello" + "world"` 生成 `"helloworld"`
>  - **重复**：`"ha" * 3` 生成 `"hahaha"`
>  - **成员关系**：`'a' in "abc"` 返回 `True`
>  - **长度**：`len("hello")` 返回 `5`
>6. **与其他序列的区别**
>    虽然 `str` 是序列的一种，但它专门用于表示文本数据，而不像列表可以存储任意类型的对象。

==**`Sequence` 的定位：**==

- 是 `collections.abc` 模块中的抽象基类，用于统一定义有序容器的接口。
- 常见序列类型如 `list`, `tuple`, `str` 等都直接或间接继承了 `Sequence`。

---

### 3. **`Optional`**
`Optional`表示一个变量**可以是某种类型，也可以是`None`**。

#### 语法
```python
from typing import Optional

def get_username(user_id: int) -> Optional[str]:
    if user_id == 1:
        return "Alice"
    return None
```

#### 等效写法
```python
Optional[str] == Union[str, None]
```

#### 说明
- `Optional[str]`的含义是：该变量可以是`str`类型，也可以是`None`。
- 它是`Union[T, None]`的简写，更加直观和常用。

---

### 示例整合
```python
from typing import Union, Sequence, Optional

def describe_item(item: Union[int, str], tags: Optional[Sequence[str]] = None) -> None:
    print(f"Item: {item}")
    if tags:
        print("Tags:")
        for tag in tags:
            print(f"- {tag}")
    else:
        print("No tags provided.")

# 用法示例
describe_item(42, ["important", "urgent"])
describe_item("book", None)
```

#### 输出：
```
Item: 42
Tags:
- important
- urgent

Item: book
No tags provided.
```

---

### 总结
- **`Union`**：多种可能的类型（例如`int`或`str`）。
- **`Sequence`**：任意支持顺序操作的容器类型，指定元素类型。
- **`Optional`**：某种类型或`None`。
