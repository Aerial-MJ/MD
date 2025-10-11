# Hugging Face

## Hugging Face

### Hugging Face 的操作概览

Hugging Face 主要提供了两个东西：

1. **模型仓库（Model Hub）**
   - 类似 GitHub，但专门放大模型和 NLP 模型（也有 CV、音频模型）。
   - 模型可以直接下载，也可以在线使用 API 调用。
2. **库/工具**
   - `transformers`：最核心的库，提供加载预训练模型、tokenizer、推理和训练的功能。
   - `datasets`：处理各种 NLP/ML 数据集。
   - `accelerate`：帮助模型分布式训练和加速。

###  下载 `transformers`

- `transformers` 是 Hugging Face 的官方库，主要功能：

  1. **快速加载模型**

     ```python
     from transformers import AutoModelForCausalLM, AutoTokenizer
     tokenizer = AutoTokenizer.from_pretrained("gpt2")
     model = AutoModelForCausalLM.from_pretrained("gpt2")
     ```

     上面可以直接下载模型权重，免去自己训练。

  2. **统一接口**
      不同模型（GPT、BERT、LLaMA、Qwen……）可以用统一方式加载和调用。

  3. **推理/微调**

     - 支持文本生成、分类、问答、特征提取。
     - 支持 GPU 加速、半精度 FP16。

  4. **兼容 Hugging Face Hub**
      可以直接从 Hugging Face 仓库下载 `.bin` 或 `.safetensors` 模型文件。

### **Hugging Face 的常用操作**

1. **安装库**

```
pip install transformers
pip install datasets accelerate
```

2. **登录 Hugging Face（可选，用于访问私有模型）**

```
huggingface-cli login
```

3. **下载模型**

- 直接在 Python 中：

```
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
model = AutoModel.from_pretrained("facebook/opt-1.3b")
```

- 或者 CLI 下载到本地：

```
huggingface-cli repo clone facebook/opt-1.3b
```

4. **使用模型进行推理**

```
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
```

### huggingFace-cli

#### **安装依赖库**

最核心的库是 `transformers` 和 `huggingface_hub`：

```bash
# 安装 transformers（包含模型加载和推理功能）
pip install transformers

# 安装 huggingface_hub（包含 huggingface-cli）
pip install huggingface_hub --upgrade
```

> ⚠️ 一般直接安装 `transformers` 就会带上 `huggingface_hub`，但是建议单独升级到最新版以保证 CLI 功能完整。

------

#### **检查 huggingface-cli 是否可用**

```bash
huggingface-cli --help
```

如果看到类似下面的输出，说明安装成功：

```bash
Usage: huggingface-cli [OPTIONS] COMMAND [ARGS]...
Commands:
  login       Login using your Hugging Face account token
  logout      Logout from Hugging Face
  whoami      Display information about your user
  repo        Manage repositories on the Hub
  ...
```

------

#### **常用命令**

1. **登录 Hugging Face**（用于下载私有模型或上传模型）：

```bash
huggingface-cli login
```

会要求你输入 Hugging Face 的 **Access Token**，可以在 Hugging Face 账户设置 获取。

1. **查看当前账户信息**：

```bash
huggingface-cli whoami
```

1. **克隆或下载模型仓库**：

```bash
huggingface-cli repo clone <MODEL_ID> <LOCAL_DIR>
# 示例
huggingface-cli repo clone facebook/opt-1.3b ./opt-1.3b
```

1. **上传模型或文件到 Hugging Face Hub**：

```bash
huggingface-cli repo create my-model
huggingface-cli upload ./local_file.txt --repo-id my-model
```

------

#### **总结**

- `huggingface-cli` 属于 `huggingface_hub` 提供的命令行工具。
- 先确保安装了 `transformers` 或单独安装 `huggingface_hub`。
- CLI 功能包括 **登录、管理仓库、下载模型、上传模型** 等。

### Github Vs Hugging Face

| 方面          | GitHub                           | Hugging Face                                                 |
| ------------- | -------------------------------- | ------------------------------------------------------------ |
| **定位**      | 代码托管                         | 模型/数据托管                                                |
| **核心对象**  | `.py`、`.c`、`.ipynb` 等源码文件 | 预训练模型权重（`.bin`, `.safetensors`）、tokenizer 配置、数据集 |
| **版本控制**  | Git                              | Git + 大文件存储（支持 Git LFS）                             |
| **用途**      | 协作开发、代码管理               | 下载/分享/发布 AI 模型、微调模型、推理                       |
| **API/工具**  | git CLI、GitHub API              | huggingface-cli、transformers、datasets                      |
| **私有/公开** | 可选公开或私有 repo              | 模型和数据集也可以公开或私有（需登录 Access Token）          |

------

#### **核心区别**

1. **大文件支持**
   - GitHub 有 Git LFS 也能存大文件，但通常不推荐存非常大的 AI 模型（几十 GB）。
   - Hugging Face 天然支持大模型权重存储和分发。
2. **面向 AI 社区**
   - Hugging Face 提供 **统一加载接口**（`transformers`、`datasets`），用户可以直接在 Python 中下载模型和数据集。
   - GitHub 更多是托管源码，需要自己写下载或加载逻辑。
3. **直接推理/训练兼容**
   - Hugging Face 的模型仓库里的文件结构都是标准化的（`pytorch_model.bin`、`config.json`、`tokenizer.json` 等），可以直接用 `transformers` 加载。
   - GitHub 上的文件通常不保证可以直接用作模型推理。

------

**总结**：

- GitHub = 代码仓库
- Hugging Face = AI 模型/数据仓库 + API/工具生态
- 如果你把模型当作“大文件代码”，可以理解 Hugging Face 是专门为 AI 模型优化的 GitHub。

### hf-mirror （huggingface 的国内镜像）

网站域名 [hf-mirror.com](https://hf-mirror.com/)，用于镜像 [huggingface.co](https://huggingface.co/) 域名。作为一个公益项目，致力于帮助国内AI开发者快速、稳定的下载模型、数据集

#### 方法一：网页下载

在https://hf-mirror.com/搜索，并在模型主页的Files and Version中下载文件

#### 方法二：huggingface-cli

**安装依赖**

```python
pip install -U huggingface_hub
```

注意：huggingface_hub 依赖于 Python>=3.8，此外需要安装 0.17.0 及以上的版本，推荐0.19.0+。

##### 1. 设置 Hugging Face 镜像环境变量

**Linux**

```bash
export HF_ENDPOINT=https://hf-mirror.com
# 建议写入 ~/.bashrc
echo 'export HF_ENDPOINT="https://hf-mirror.com"' >> ~/.bashrc
source ~/.bashrc
```

**Windows Powershell**

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

**Python**

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

------

##### 2. 使用 huggingface-cli 下载模型和数据集

###### 下载模型（例：gpt2）

```bash
huggingface-cli download --resume-download gpt2 --local-dir gpt2
```

###### 下载数据集（例：wikitext）

```bash
huggingface-cli download --repo-type dataset --resume-download wikitext --local-dir wikitext
```

> ⚠️ 值得注意的是，有个--local-dir-use-symlinks False 参数可选，因为huggingface的工具链默认会使用符号链接来存储下载的文件，导致--local-dir指定的目录中都是一些“链接文件”，真实模型则存储在~/.cache/huggingface下，如果不喜欢这个可以用 --local-dir-use-symlinks False取消这个逻辑。

------

##### 3. 使用 hfd（推荐，稳定下载）

hfd 是https://hf-mirror.com/开发的 huggingface 专用下载工具，基于成熟工具 git+aria2，可以做到`稳定下载不断线`。

###### 下载 hfd

```bash
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
```

###### 下载模型（例：gpt2，使用 aria2c 多线程）

```bash
./hfd.sh gpt2 --tool aria2c -x 4
```

###### 如果没有 aria2，可直接用 wget

```bash
./hfd.sh gpt2
```

###### 下载数据集（例：wikitext）

```bash
./hfd.sh wikitext --dataset --tool aria2c -x 4
```

## aria2c

**Aria2c** 是一个强大的命令行下载工具，支持多种协议，包括 **HTTP(S)**、**FTP**、**SFTP**、**BitTorrent** 和 **Metalink**。它以轻量级、多线程和高效著称，能够从多个来源同时下载文件，最大限度地利用带宽资源。

### 核心功能

- **多线程下载**：支持通过 *-x* 参数设置每个服务器的最大连接数（最多16个线程），并通过 *-s* 参数将文件分片并发下载。
- **多协议支持**：可以同时从 HTTP(S)、FTP 和 BitTorrent 网络下载文件。
- **断点续传**：通过 *-c* 参数实现下载中断后的续传功能。
- **批量下载**：支持通过 *-i* 参数读取包含多个下载链接的文件，进行批量任务处理。
- **数据完整性校验**：借助 Metalink 的分块校验功能，确保下载数据的完整性。

### 安装与配置

在不同系统上，安装 Aria2c 的方法略有不同：

- **Debian/Ubuntu**：使用 `sudo apt install aria2`
- **CentOS**：启用 EPEL 仓库后，运行 `sudo yum install aria2`
- **MacOS**：通过 Homebrew 安装，运行 `brew install aria2`

配置文件通常位于 *~/.aria2/aria2.conf*，可以设置下载目录、最大连接数、RPC 服务等。例如：

```
dir=/path/to/downloads
max-concurrent-downloads=5
enable-rpc=true
rpc-secret=your_password
split=16
max-connection-per-server=16
```

### 常用命令示例

- **单文件下载**：

```
aria2c -s 8 http://example.com/large_file.iso
```

- **批量下载**：创建包含多个链接的文本文件 *links.txt*，然后运行：

```
aria2c -i links.txt
```

- **启动 RPC 服务**：

```
aria2c --conf-path=/path/to/aria2.conf -D
```

### 高级功能

- **限速**：通过 *--max-download-limit* 和 *--max-upload-limit* 控制带宽。

```
aria2c --max-download-limit=1M http://example.com/file.iso
```

- **远程控制**：结合 WebUI（如 AriaNg）或命令行工具（如 aria2p），通过 JSON-RPC 管理下载任务。

### 使用场景

Aria2c 适用于需要高效下载大文件、多文件或需要断点续传的场景。它还可以集成到自动化脚本中，成为开发者和数据分析人员的得力工具。

## aria2 和 aria2c

其实 **aria2** 和 **aria2c** 是同一个软件的不同调用方式，区别主要如下：

------

### 1. 名称来源

- **aria2**：软件的官方名称，本身是指整个下载工具包。
- **aria2c**：aria2 的 **命令行客户端（client）**，是最常用的启动方式。

在大多数系统中，你在命令行中运行 `aria2` 或 `aria2c`，本质上调用的都是同一个程序，只是官方推荐使用 `aria2c` 来明确表示“命令行客户端”。

------

### 2. 用途

- **aria2c**：直接在终端执行命令下载文件、批量下载、RPC 控制等，是默认使用方式。
- **aria2**：有时指整个工具包或包含 GUI / RPC 功能的服务端组件，但在命令行中一般会自动映射到 `aria2c`。





