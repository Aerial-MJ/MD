# Python 命令行参数 & pip install 速查手册

---

## 一、Python 命令行参数（argparse）

### 为什么要用 argparse

不用 argparse：每次改参数都要进代码里改，麻烦且容易出错。  
用 argparse：直接在命令行传参，代码不用动。

```bash
# 不用 argparse（硬编码）
python train.py          # 参数写死在代码里

# 用 argparse（灵活传参）
python train.py --lr 1e-4 --epochs 10 --output ./output
```

---

### 基本写法模板

```python
import argparse

parser = argparse.ArgumentParser(description="训练脚本")

# 添加参数
parser.add_argument("--lr",      type=float, default=1e-4,    help="学习率")
parser.add_argument("--epochs",  type=int,   default=10,      help="训练轮数")
parser.add_argument("--output",  type=str,   default="output", help="输出目录")
parser.add_argument("--debug",   action="store_true",          help="开启调试模式")

# 解析参数
args = parser.parse_args()

# 使用参数
print(args.lr)       # 1e-4
print(args.epochs)   # 10
print(args.debug)    # True / False
```

---

### 参数类型详解：Store 型 vs Flag 型（官方标准分类）

`--` 开头的可选参数（optional arguments），按 **"是否需要接一个值"** 这个维度，官方分为两大类：

| 分类 | `action` | 是否接值 | 例子 |
|------|----------|---------|------|
| **Store 型**（存值参数） | `action='store'`（默认，通常省略不写） | ✅ 是，必须接一个值 | `--input xxx`、`--batch_size 1000` |
| **Flag 型**（标志/开关参数） | `action='store_true'` 或 `action='store_false'` | ❌ 否，单独出现即可 | `--resume`、`--debug` |

> 这两套分类维度是**正交**的，可以叠加组合：一个脚本里可以同时有"必填的 store 型"（`required=True`）、"有默认值的 store 型"（`default=xxx`）、"flag 型"（`action='store_true'`），彼此互不冲突。

#### 一、Store 型（存值参数）—— 参数后面必须跟一个值

这是最常见的形式，argparse 把这个值存起来，默认行为就是 `action='store'`（平时都省略不写）。

**1. 普通 store 参数（`--key value` 形式）**

```python
parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
```

```bash
python train.py --lr 3e-5
python train.py --lr 0.001
```

| 字段 | 含义 |
|------|------|
| `--lr` | 参数名，命令行用 `--lr 值` 传入 |
| `type=float` | 自动把字符串转成 float |
| `default=1e-4` | 不传时的默认值 |
| `help="..."` | `python train.py --help` 时显示的说明 |

**2. 必填 vs 有默认值 —— store 型内部的细分**

store 型参数内部还可以按"是否必填"再分一层，区别在于给 `required=True` 还是 `default=xxx`：

```python
parser.add_argument("--input",      required=True)              # 必填，不传会报错
parser.add_argument("--batch_size", type=int, default=1000)     # 选填，不传用默认值
```

> ⚠️ **`required=True` 和 `default=xxx` 不要同时写，两者语义矛盾**：`required=True` 意味着"不传就报错"，`default` 意味着"不传就用默认值兜底"，二者互斥。同时写的话，argparse 不会报语法错，但 `default` 是死代码永远用不上——不传参时 `required=True` 会先拦截并报错，根本走不到用默认值兜底那一步；传参时又用的是传入值，也用不上默认值。正确做法是二选一：
>
> ```python
> # 必填，不给默认值
> parser.add_argument("--input", required=True)
>
> # 选填，给默认值
> parser.add_argument("--input", default="data.json")
> ```

**`required` 和 `default` 都不写会怎样？**

```python
parser.add_argument("--output_dir", type=str, help="输出目录")
```

- **`required` 的隐藏默认值是 `False`**：不写 `required` 就相当于 `required=False`，这个参数是**选填**的，不传不会报错
- **`default` 的隐藏默认值是 `None`**：不写 `default` 就相当于 `default=None`，不传时 `args.output_dir` 的值会是 `None`，而不是空字符串或其他值

验证：

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, help="输出目录")
args = parser.parse_args([])   # 模拟不传任何参数
print(args.output_dir)          # None
print(type(args.output_dir))    # <class 'NoneType'>
```

三种情况完整对比：

| 写法 | 不传时的值 | 不传会报错吗 |
|------|-----------|------------|
| `add_argument("--x")`（什么都不写） | `None` | ❌ 不会 |
| `add_argument("--x", default="abc")` | `"abc"` | ❌ 不会 |
| `add_argument("--x", required=True)` | 不适用 | ✅ 会报错：`error: the following arguments are required: --x` |

**3. 位置参数（不带 `--`，按顺序传入）**

位置参数本质上也是 store 型的一种（同样必须接一个值），只是不用 `--` 前缀，靠顺序传入：

```python
parser.add_argument("input_file", type=str, help="输入文件路径")
```

```bash
python train.py data.json      # args.input_file = "data.json"
```

> 位置参数是必填的，不填会报错。`--` 开头的参数是否必填取决于有没有写 `required=True`。

**4. 多值参数（一次传多个值）**

```python
parser.add_argument("--gpus", nargs="+", type=int, help="GPU编号列表")
```

```bash
python train.py --gpus 0 1 2   # args.gpus = [0, 1, 2]
```

**5. 选项限定（只能从几个值里选）**

```python
parser.add_argument("--mode", choices=["train", "eval", "infer"], default="train")
```

```bash
python train.py --mode eval    # ✅
python train.py --mode abc     # ❌ 报错
```

#### 二、Flag 型（标志/开关参数）—— 不需要接值

参数本身出现与否就是信息，对应 `action='store_true'`（或反过来的 `action='store_false'`）：

```python
parser.add_argument("--debug", action="store_true", help="开启调试模式")
```

```bash
python train.py --debug        # args.debug = True
python train.py                # args.debug = False（默认）
```

用的时候只写 `--debug` 四个字，后面不接任何值；出现了就是 `True`，不出现就是 `False`（默认值）。

```python
parser.add_argument("--resume", action="store_true", default=False)
```

```bash
python train.py --resume       # args.resume = True
python train.py                # args.resume = False
```

> 如果想要"默认是 True，加了参数才关闭"的反向开关，用 `action='store_false'`：
> ```python
> parser.add_argument("--no_shuffle", action="store_false", dest="shuffle")
> # 不传 --no_shuffle 时 args.shuffle = True；传了 --no_shuffle 时 args.shuffle = False
> ```

#### 一句话区分

> 之前讲的"必填 vs 有默认值"，其实都属于 **store 型内部的细分**（区别在于给不给 `required=True` 还是 `default=xxx`）；而 **flag 型是跟它们平行的另一条分类线**（区别在于 `action`）。一个脚本里的多个 store 型参数（有必填也有默认值）和 flag 型参数可以同时出现，刚好把这几种组合都占全了。

---

### 短参数 vs 长参数

可以同时定义短参数（`-x`）和长参数（`--xxx`），效果相同：

```python
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
```

```bash
python train.py -lr 3e-5          # 短参数写法
python train.py --learning_rate 3e-5  # 长参数写法（两者等价）
```

---

### 完整示例

```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="LLM 微调脚本")
    parser.add_argument("--model_name",  type=str,   required=True,   help="模型名称，必填")
    parser.add_argument("--data_path",   type=str,   required=True,   help="数据路径，必填")
    parser.add_argument("--output_dir",  type=str,   default="output", help="输出目录")
    parser.add_argument("--lr",          type=float, default=1e-4,    help="学习率")
    parser.add_argument("--epochs",      type=int,   default=3,       help="训练轮数")
    parser.add_argument("--batch_size",  type=int,   default=4,       help="批大小")
    parser.add_argument("--fp16",        action="store_true",          help="开启半精度训练")
    return parser.parse_args()

args = parse_args()
print(f"训练模型：{args.model_name}，学习率：{args.lr}，fp16：{args.fp16}")
```

命令行调用：

```bash
python train.py \
    --model_name Qwen2-7B \
    --data_path ./data/train.json \
    --lr 2e-5 \
    --epochs 5 \
    --fp16
```

---

### 查看参数帮助

任何用了 argparse 的脚本都可以用 `--help` 查看所有参数：

```bash
python train.py --help
```

输出示例：
```
usage: train.py [-h] --model_name MODEL_NAME --data_path DATA_PATH ...

LLM 微调脚本

options:
  --model_name   模型名称，必填
  --lr           学习率 (default: 0.0001)
  --epochs       训练轮数 (default: 3)
  --fp16         开启半精度训练
```

---

## 二、pip install 模式详解

### `pip install -e .` vs `pip install -e ".[torch,metrics]"`

| | `pip install -e ".[torch,metrics]"` | `pip install -e .` |
|---|---|---|
| 安装模式 | 开发模式（editable） | 开发模式（editable） |
| 核心包 | ✅ 安装 | ✅ 安装 |
| torch 相关依赖 | ✅ 额外安装 | ❌ 不安装 |
| metrics 相关依赖 | ✅ 额外安装 | ❌ 不安装 |
| 适合场景 | 完整训练/推理环境 | 环境已有 torch，只需让模块可 import |

---

### 逐个拆解

#### `pip install -e .`

```
pip install  -e   .
     ↑        ↑   ↑
     │        │   └── 当前目录（安装这里的包）
     │        └── editable 模式：代码改动即时生效，不用重新 install
     └── pip 安装命令
```

- **editable 模式**（`-e`）：不把代码复制到 site-packages，而是建一个软链接，改代码后立刻生效，无需重新安装
- 只安装 `setup.py` / `pyproject.toml` 里最基础的依赖
- torch、metrics 等"可选依赖"不会安装

#### `pip install -e ".[torch,metrics]"`

```
pip install  -e   ".[torch,metrics]"
                    ↑  ↑
                    │  └── 额外安装 torch 和 metrics 这两组可选依赖
                    └── 当前目录
```

- `[torch,metrics]` 是"extras"，定义在 `setup.py` 的 `extras_require` 里
- torch：PyTorch 相关依赖（训练模型必须要有）
- metrics：评估指标相关依赖（如 `evaluate`、`rouge-score` 等）

---

### 为什么需要引号

```bash
pip install -e ".[torch,metrics]"   # ✅ 正确：引号防止 [] 被终端特殊解析
pip install -e .[torch,metrics]     # ⚠️  在某些 shell 里可能报错（[] 是 glob 通配符）
```

> 引号是保险写法，加上没有坏处。

---

### 什么时候用哪个

```bash
# 场景一：从零搭建环境，需要完整依赖（推荐）
pip install -e ".[torch,metrics]"

# 场景二：环境里已经装好了 torch，只是 llamafactory 模块缺失
pip install -e .

# 场景三：普通安装（不开发，不需要改代码后即时生效）
pip install .
```

> IG-Py312 环境里已经有 torch，通常用 `pip install -e .` 即可。

---

### extras_require 在 setup.py 里长什么样

了解背后原理（不需要自己写，只是看懂）：

```python
# setup.py 示例
setup(
    name="llamafactory",
    install_requires=[          # 基础依赖，pip install . 会安装这些
        "transformers>=4.40.0",
        "datasets>=2.16.0",
        "peft>=0.11.1",
    ],
    extras_require={            # 可选依赖组
        "torch": [
            "torch>=2.0.0",
            "torchvision",
        ],
        "metrics": [
            "evaluate>=0.4.0",
            "rouge-score",
            "nltk",
        ],
    },
)
```

`pip install -e ".[torch,metrics]"` 就是在基础依赖之上，再把 `torch` 和 `metrics` 两组都安装上。
