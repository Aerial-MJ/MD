# Python 包与模块

## 一、什么是包

Python 判断一个目录是不是包，就看一件事：**有没有 `__init__.py` 文件**。

```
LLaMA-Factory/
└── llamafactory/
    ├── __init__.py     ← 有这个文件，llamafactory 就是一个包
    ├── cli.py          ← llamafactory.cli 模块
    ├── train/
    │   ├── __init__.py
    │   └── ...
    └── ...
```

只要目录里有 `__init__.py`，Python 就认为这个目录是一个**包**，里面的 `.py` 文件都是它的**模块**。

没有 `__init__.py` 的目录只是普通文件夹，`python -m mydir.cli` 会报错找不到包。

---

## 二、`pip install` 的两种方式

### 普通安装：`pip install .`

把代码文件**复制**到 `site-packages/llamafactory/`，是真实的文件副本，修改源码不影响已安装版本。

### 开发模式：`pip install -e .`

在 `site-packages/` 里放一个 `.pth` 或 `.egg-link` 文件，**指向你的项目目录**，类似软链接。修改源码立刻生效，不需要重新安装。`-e` 即 `--editable`（可编辑模式）。

```bash
# 查看指向
cat $(python -c "import site; print(site.getsitepackages()[0])")/llamafactory.egg-link
# 输出类似：/home/user/LLaMA-Factory
```

---

## 三、`python -m` 执行模块

```bash
python -m llamafactory.cli train config.yaml
```

Python 去 `sys.path` 中查找 `llamafactory/cli.py`，然后执行。

**能不能在任意目录执行，取决于是否已安装：**

| 情况 | 能否任意目录执行 |
|------|---------------|
| `pip install` 安装过 | ✅ 任意目录可用，包在 site-packages 里 |
| 只是 clone 了代码，未安装 | ❌ 只能在项目根目录下执行 |

验证是否已安装：
```bash
pip show llamafactory
```

---

## 四、三种执行方式的对比

### 方式一：`python -m llamafactory.cli`

- Python 直接去 `sys.path` 找 `llamafactory/cli.py`
- 会正确设置 `__package__ = "llamafactory"`
- ✅ 相对 import 正常

### 方式二：`python /path/to/bin/llamafactory-cli`

`/path/to/bin/llamafactory-cli` 是 pip install 时自动生成的**入口脚本**，内容大概是：

```python
#!/usr/bin/env python
from llamafactory.cli import main

if __name__ == '__main__':
    main()
```

执行流程：先执行脚本文件（不查找模块），脚本里再 `import llamafactory`（从 site-packages 找）。同样要求已安装。

### 方式三：`python /path/to/llamafactory/cli.py`

直接按路径执行脚本文件，**可能出问题**：

- 绝对 import（`from llamafactory.train import xxx`）→ ✅ 已安装就能找到
- 相对 import（`from .utils import yyy`）→ ❌ 报错

| 执行方式 | 查找模块 | 相对 import |
|---------|---------|------------|
| `python /path/to/cli.py` | 顶层脚本，`__package__ = None` | ❌ 报错 |
| `python -m llamafactory.cli` | 包内模块，`__package__ = "llamafactory"` | ✅ 正常 |

---

## 五、`__package__` 的作用

`__package__` 是 Python 在执行文件时**自动设置**的变量，告诉当前文件"你属于哪个包"。

### 不同执行方式下的值

```bash
# 直接执行脚本
python /path/to/llamafactory/cli.py
```
```python
print(__package__)   # None
print(__name__)      # __main__
```

```bash
# -m 执行
python -m llamafactory.cli
```
```python
print(__package__)   # llamafactory
print(__name__)      # __main__
```

```python
# 被 import 时
import llamafactory.cli
print(__package__)   # llamafactory
print(__name__)      # llamafactory.cli
```

### 相对 import 怎么用 `__package__`

```python
from .utils import yyy
# 等价于：
from {__package__}.utils import yyy

# __package__ = "llamafactory" → from llamafactory.utils import yyy  ✅
# __package__ = None           → from None.utils import yyy          ❌
```

**绝对 import 靠 `sys.path`（全局查找），和怎么执行无关；相对 import 靠 `__package__`，直接执行脚本时为 `None`，所以报错。**

---

## 六、总结

| 问题 | 结论 |
|------|------|
| 目录怎么成为包？ | 有 `__init__.py` 文件 |
| `pip install` vs `pip install -e` | 前者复制文件，后者放指针（类似软链接） |
| `-m` 任意目录可用吗？ | 已安装才行，否则只能在项目根目录 |
| 直接 `python cli.py` 有问题吗？ | 相对 import 会报错，绝对 import 没问题 |
| 为什么相对 import 会断？ | `__package__` 为 `None`，不知道自己在哪个包里 |
