# pip 与 Conda 源管理

## 一、查看当前源配置

### 查看 Conda 源

```bash
# 查看当前配置的所有 conda 源
conda config --show channels

# 查看完整 conda 配置
conda config --show

# 查看配置文件位置（.condarc）
conda config --show-sources
```

输出示例：

```
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.aliyun.com/anaconda/pkgs/main
  - defaults
```

### 查看 pip 源

```bash
# 查看当前 pip 源配置
pip config list

# 查看详细配置及配置文件路径
pip config debug

# 直接查看配置文件
cat ~/.pip/pip.conf           # Linux/Mac 常用路径
cat ~/.config/pip/pip.conf    # Linux 备用路径
```

输出示例：

```
global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'
global.trusted-host='pypi.tuna.tsinghua.edu.cn'
```

---

## 二、配置文件位置汇总

| 工具 | 配置文件路径 |
|------|------------|
| Conda | `~/.condarc` |
| pip (Linux/Mac) | `~/.pip/pip.conf` 或 `~/.config/pip/pip.conf` |
| pip (Windows) | `%APPDATA%\pip\pip.ini` |

---

## 三、永久换源

### pip 永久换源

**方法一：命令行设置（推荐）**

```bash
# 设置清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 设置阿里源
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
pip config set global.trusted-host mirrors.aliyun.com
```

**方法二：直接编辑配置文件**

```bash
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
```

**恢复默认源：**

```bash
pip config unset global.index-url
pip config unset global.trusted-host
```

---

### Conda 永久换源

**方法一：命令行设置**

```bash
# 添加清华源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge

# 设置显示源 URL（安装时能看到从哪里下载）
conda config --set show_channel_urls yes

# 移除 defaults（可选，避免混用）
conda config --remove channels defaults
```

**方法二：直接编辑 `~/.condarc`**

```yaml
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
show_channel_urls: true
```

**恢复默认源：**

```bash
conda config --remove-key channels
# 或直接删除配置文件
rm ~/.condarc
```

---

## 四、临时指定源（不修改配置）

```bash
# pip 临时指定源
pip install <包名> -i https://pypi.tuna.tsinghua.edu.cn/simple

# conda 临时指定源
conda install <包名> -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
```

---

## 五、常用镜像源地址

### pip 镜像

| 镜像 | 地址 |
|------|------|
| 清华 | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| 阿里 | `https://mirrors.aliyun.com/pypi/simple` |
| 中科大 | `https://pypi.mirrors.ustc.edu.cn/simple` |
| 豆瓣 | `https://pypi.douban.com/simple` |

### Conda 镜像

| 镜像 | 地址前缀 |
|------|---------|
| 清华 | `https://mirrors.tuna.tsinghua.edu.cn/anaconda/` |
| 阿里 | `https://mirrors.aliyun.com/anaconda/` |
| 中科大 | `https://mirrors.ustc.edu.cn/anaconda/` |

---

## 六、查看包的可用版本

```bash
# 查看 pip 源中某个包的所有可用版本
pip index versions <包名>

# 指定源查询
pip index versions <包名> -i https://your-internal-source/simple/

# 技巧：故意写一个不存在的版本，pip 会报错并列出所有可用版本
pip install <包名>==

# conda 查看可用版本
conda search <包名>
```

---

## 七、查看 Python 版本

```bash
python --version        # 最常用
python3 --version
python -V

# 详细信息
python -c "import sys; print(sys.version)"

# 查看完整路径 + 版本
python -c "import sys; print(sys.executable, sys.version)"
```
