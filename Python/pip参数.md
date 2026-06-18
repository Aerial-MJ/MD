# pip 常用参数总结

## 一、命令行结构说明

pip 的命令结构是：**主命令 + 子命令 + 操作对象 + 参数**

```
pip   install    verl      -i    https://xxx
 ↑       ↑         ↑        ↑        ↑
主命令  子命令   操作对象   参数     参数的值
```

| 概念 | 说明 | 有无前缀 |
|------|------|---------|
| **子命令** | 做什么动作（install / index / list ...） | 无 `-` |
| **操作对象** | 位置参数，按顺序填写 | 无 `-` |
| **参数/选项** | 修饰命令的行为 | 有 `-` 或 `--` |

### argparse 对应写法

```python
import argparse
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command")

install_parser = subparsers.add_parser("install")
install_parser.add_argument("package")              # 操作对象：位置参数，没有 --
install_parser.add_argument("-i", "--index-url")    # 选项：有 -
install_parser.add_argument("-e", "--editable", action="store_true")
```

```bash
python cli.py install verl -i https://xxx
# → Namespace(command='install', package='verl', index_url='https://xxx', editable=False)
```

---

## 二、安装相关参数

### 基础安装

```bash
pip install <包名>
pip install <包名>==1.2.3        # 指定版本
pip install "<包名>>=1.0,<2.0"   # 版本范围
pip install -r requirements.txt  # 从文件批量安装
```

---

### `-e` / `--editable`：开发模式安装

```bash
pip install -e .
```

- 不复制文件到 site-packages，而是放一个"指针"（`.egg-link` 或 `.pth` 文件）指向当前目录
- 修改源码后**立即生效**，不需要重新安装
- 适合开发阶段使用

```
普通安装：源码 → 复制到 site-packages/
开发安装：site-packages/ 里放一个指针 → 指回源码目录（类似软链接）
```

---

### `--no-build-isolation`：关闭构建隔离

```bash
pip install --no-build-isolation <包名>
```

**正常行为（有隔离）：**

```
pip 创建临时干净虚拟环境 → 在隔离环境里编译 → 安装到你的环境
```

**加上此参数（无隔离）：**

```
直接在当前环境里编译 → 安装到你的环境
```

**什么时候用：** 编译时需要依赖当前环境中已安装的包（如 `torch`），隔离环境里没有这些包会导致编译失败。

```bash
# 典型场景：flash-attn 编译依赖已安装的 torch
pip install --no-build-isolation flash-attn
```

---

### `--no-user`：不安装到用户目录

```bash
pip install --no-user <包名>
```

- 强制安装到当前虚拟环境（venv/conda），而不是用户目录 `~/.local/lib/`
- 在 conda/venv 环境中通常是默认行为，显式加上更保险

---

### `-U` / `--upgrade`：升级包

```bash
pip install -U <包名>
```

---

### `-i` / `--index-url`：指定镜像源

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <包名>
```

和 `--extra-index-url` 的区别：

| 参数 | 行为 |
|------|------|
| `-i` / `--index-url` | **替换**默认源 |
| `--extra-index-url` | **追加**到默认源，两个源都会查 |

```bash
pip install --extra-index-url https://xxx <包名>
```

---

### `--no-cache-dir`：不使用缓存

```bash
pip install --no-cache-dir <包名>
```

强制重新下载，不用本地缓存。磁盘空间不足或缓存损坏时使用。

---

### `--no-deps`：不安装依赖

```bash
pip install --no-deps <包名>
```

只装指定包，不自动安装其依赖。

---

## 三、子命令：pip index

`index` 是 pip 的子命令（不是参数），用于查询包的版本信息：

```bash
# 查看某个包的所有可用版本
pip index versions <包名>

# 指定源查询
pip index versions <包名> -i https://your-internal-source/simple/

# 技巧：写一个不存在的版本，pip 报错时会列出所有可用版本
pip install <包名>==
```

---

## 四、组合用法示例

```bash
# 安装需要编译的包（如 flash-attn），用当前环境编译，不装用户目录
pip install --no-build-isolation --no-user flash-attn

# 开发模式安装当前项目，用当前环境编译
pip install --no-build-isolation --no-user -e .

# 指定镜像源升级
pip install -U -i https://pypi.tuna.tsinghua.edu.cn/simple torch

# 查询内部源里某个包的版本
pip index versions verl -i http://your-internal-pypi-url/simple/
```

---

## 五、其他常用命令

```bash
pip list                       # 列出已安装的包
pip show <包名>                # 查看某个包的详细信息（版本、路径等）
pip freeze > requirements.txt  # 导出当前环境依赖
pip uninstall <包名>           # 卸载
pip cache purge                # 清空 pip 缓存
pip config list                # 查看当前 pip 源配置
pip config debug               # 查看配置文件路径
```
