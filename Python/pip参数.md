# pip 常用参数总结

## 安装相关

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

### `--index-url` / `-i`：指定镜像源

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <包名>
```

常用国内镜像：

| 镜像 | 地址 |
|------|------|
| 清华 | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| 阿里 | `https://mirrors.aliyun.com/pypi/simple` |
| 中科大 | `https://pypi.mirrors.ustc.edu.cn/simple` |

---

### `--extra-index-url`：追加镜像源

```bash
pip install --extra-index-url https://xxx <包名>
```

和 `-i` 的区别：`-i` 替换默认源，`--extra-index-url` 是在默认源基础上追加。

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

## 组合用法示例

```bash
# 安装需要编译的包（如 flash-attn），用当前环境编译，不装用户目录
pip install --no-build-isolation --no-user flash-attn

# 开发模式安装当前项目，用当前环境编译
pip install --no-build-isolation --no-user -e .

# 指定镜像源升级
pip install -U -i https://pypi.tuna.tsinghua.edu.cn/simple torch
```

---

## 其他常用命令

```bash
pip list                    # 列出已安装的包
pip show <包名>             # 查看某个包的详细信息（版本、路径等）
pip freeze > requirements.txt  # 导出当前环境依赖
pip uninstall <包名>        # 卸载
pip cache purge             # 清空 pip 缓存
```
