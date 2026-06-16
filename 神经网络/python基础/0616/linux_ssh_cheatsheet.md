# Linux 命令符号 & SSH 端口转发 速查手册

---

## 一、Linux 命令控制符号

### 1. `&` —— 后台运行

```bash
python train.py &
```

- 把命令放到**后台**运行，当前终端可以继续输入其他命令
- 但如果你关掉终端，进程会被杀掉（用 `nohup` 配合解决）

---

### 2. `&&` —— 串行执行（前一个成功才执行下一个）

```bash
mkdir output && python train.py
```

- 只有 `mkdir output` **成功**（exit code=0），才会执行 `python train.py`
- 相当于"如果成功，则继续"

---

### 3. `||` —— 或者执行（前一个失败才执行下一个）

```bash
mkdir output || echo "目录已存在"
```

- 只有 `mkdir output` **失败**，才会执行后面的命令
- 相当于"如果失败，则执行备用方案"

---

### 4. `;` —— 顺序执行（不管成功失败都执行）

```bash
cd /tmp ; ls ; pwd
```

- 不管前面的命令成不成功，后面的命令都会执行
- 和 `&&` 的区别：`&&` 会因失败中断，`;` 不会

---

### 5. `>` —— 输出重定向（覆盖写入）

```bash
echo "hello" > output.txt
```

- 把命令的输出写入文件，**会覆盖**原有内容
- 文件不存在会自动创建

---

### 6. `>>` —— 输出重定向（追加写入）

```bash
echo "hello" >> output.txt
```

- 把命令的输出**追加**到文件末尾，不覆盖原有内容
- 常用于持续写日志

---

### 7. `2>&1` —— 合并错误输出到标准输出

```bash
python train.py > train.log 2>&1
```

- `1` = 标准输出（stdout），正常打印的内容
- `2` = 标准错误（stderr），报错信息
- `2>&1` = 把 stderr 也重定向到 stdout 的地方（即同一个文件）
- 这样报错和正常日志都写到 `train.log`，不会漏掉任何信息

---

### 8. `|` —— 管道（把前一个命令的输出传给下一个命令）

```bash
ps aux | grep python
cat train.log | tail -100
```

- 把左边命令的**输出**作为右边命令的**输入**
- 常用组合：
  - `| grep xxx` → 过滤包含 xxx 的行
  - `| head -n 20` → 只看前 20 行
  - `| tail -n 100` → 只看最后 100 行
  - `| wc -l` → 统计行数

---

### 9. `nohup` —— 断开 SSH 后进程继续跑

```bash
nohup python train.py > train.log 2>&1 &
```

- 没有 `nohup`：关掉终端/SSH 断连，进程会收到 SIGHUP 信号被杀掉
- 有 `nohup`：进程忽略该信号，继续在后台运行
- 通常和 `&` 搭配使用，日志重定向到文件

---

### 综合示例（训练启动命令解析）

```bash
nohup env CUDA_VISIBLE_DEVICES=0 llamafactory-cli train train_qwen3vl_2b_lora.yaml \
    > output/qwen3vl_2b_v4_lora/train.log 2>&1 &
```

| 部分 | 含义 |
|------|------|
| `nohup` | SSH 断开后进程不中断 |
| `env CUDA_VISIBLE_DEVICES=0` | 临时设置环境变量，指定用第 0 块 GPU |
| `llamafactory-cli train xxx.yaml` | 实际的训练命令 |
| `\` | 命令太长换行续写，只是格式美观，实际是同一行 |
| `> output/.../train.log` | 标准输出写到日志文件（覆盖模式） |
| `2>&1` | 错误输出也写到同一个日志文件 |
| `&` | 整个命令放后台运行 |

---

## 二、SSH 端口转发

### 背景

你的环境是 **Kubernetes Pod（容器）**，不是直接的物理机：

```
本机 (Mac) → 10.164.10.177:8416 → K8s Pod（容器）
```

容器内服务（如 TensorBoard 6006 端口）无法直接从外部访问，需要 SSH 端口转发"打通隧道"。

---

### SSH 本地端口转发

```bash
ssh -L 本地端口:目标地址:目标端口 -N SSH别名
```

**参数说明：**

| 参数 | 含义 |
|------|------|
| `-L` | 本地端口转发（Local forwarding） |
| `本地端口` | 你本机上监听的端口 |
| `目标地址` | 从远端服务器看过去的目标地址（localhost = 服务器自己） |
| `目标端口` | 远端服务器上要转发到的端口 |
| `-N` | 不执行远程命令，只做转发，连接会一直挂着（无任何输出，正常现象） |

**实际例子：**

```bash
ssh -L 6006:localhost:6006 -N codelab-617710744
```

效果：本机访问 `http://localhost:6006` → 实际访问服务器容器内的 6006 端口（TensorBoard）

---

### ~/.ssh/config 配置文件解析

```
Host codelab-617710744          # SSH 别名，执行 ssh codelab-617710744 时使用
    HostName 10.164.10.177      # 服务器真实 IP
    Port 8416                   # SSH 端口（不是默认的 22）
    User hadoop-grocery-rc      # 登录用户名
    IdentityFile ~/.ssh/codelab_prod   # 私钥文件路径
    IdentitiesOnly yes          # 只用这个私钥，不尝试其他
    ForwardAgent yes            # 转发本地 SSH-agent，可以在远端使用本地的 key
```

配置好之后，`ssh codelab-617710744` 等价于：

```bash
ssh -p 8416 -i ~/.ssh/codelab_prod hadoop-grocery-rc@10.164.10.177
```

---

### TensorBoard 访问完整流程

**服务器上**（已在跑，无需重复启动）：
```bash
tensorboard --logdir output/qwen3vl_2b_v4_lora/runs --host 0.0.0.0 --port 6006 &
```

**本机新开一个终端**，执行端口转发（挂着不要关）：
```bash
ssh -L 6006:localhost:6006 -N codelab-617710744
```

**本机浏览器访问：**
```
http://localhost:6006
```

---

## 三、常用进程管理

### 查看后台任务

```bash
jobs          # 查看当前 shell 的后台任务
ps aux | grep python   # 查看所有 python 进程
```

### 杀掉进程

```bash
kill 12345           # 发送 SIGTERM，优雅退出
kill -9 12345        # 强制杀掉（SIGKILL）
pkill -f train.py    # 按进程名杀
```

### 实时查看日志

```bash
tail -f train.log          # 实时追踪日志最新内容
tail -n 100 train.log      # 查看最后 100 行
grep "loss" train.log      # 过滤包含 loss 的行
```

---

## 四、文件描述符与重定向进阶

### 三个标准通道

Linux/Mac 终端里，每个进程默认有三个 I/O 通道：

| 编号 | 名称 | 含义 |
|------|------|------|
| `0` | stdin  | 标准输入 |
| `1` | stdout | 标准输出（正常结果） |
| `2` | stderr | 标准错误输出（报错信息） |

---

### `2>/dev/null` —— 把报错信息丢进黑洞

```
2  >  /dev/null
↑  ↑      ↑
│  │      └── /dev/null：黑洞文件，写入的内容全部丢弃，不占空间
│  └── 重定向符号
└── 标准错误（stderr，编号 2）
```

**作用：** 把错误信息静默掉，只显示正常输出，让终端更干净。

```bash
# 不加 2>/dev/null —— 会输出大量 "Permission denied" 报错
find ~/Library -name "settings.json"

# 加了 2>/dev/null —— 报错被丢弃，只显示正常结果
find ~/Library -name "settings.json" 2>/dev/null
```

> `/dev/null` 可以理解为操作系统的"垃圾桶"，任何写入它的内容都直接消失，不占磁盘空间。

---

### 常见重定向组合对比

| 写法 | 含义 |
|------|------|
| `> file` | stdout 写入文件（覆盖） |
| `>> file` | stdout 追加到文件 |
| `2>/dev/null` | stderr 丢弃 |
| `2>&1` | stderr 合并到 stdout（同一个目标） |
| `> file 2>&1` | stdout 和 stderr 都写入同一文件 |
| `>/dev/null 2>&1` | 完全静默，所有输出全部丢弃 |

---

## 五、find 命令

### 基本语法

```bash
find  <搜索目录>  <搜索条件>
```

### 拆解示例

```bash
find ~/Library/Application\ Support -name "settings.json" 2>/dev/null
```

| 部分 | 含义 |
|------|------|
| `find` | 搜索文件的命令 |
| `~` | 当前用户的主目录（如 `/Users/yourname`） |
| `~/Library/Application\ Support` | 要搜索的目标文件夹 |
| `Application\ Support` | `\` 是转义符，因为文件夹名含空格，需告诉终端"这个空格是名字的一部分" |
| `-name "settings.json"` | 搜索条件：找名字为 `settings.json` 的文件 |
| `2>/dev/null` | 把无权限访问产生的报错静默掉 |

### 空格的两种处理方式

```bash
# 方式一：反斜杠转义空格
find ~/Library/Application\ Support -name "settings.json"

# 方式二：引号包裹整个路径（效果相同）
find "~/Library/Application Support" -name "settings.json"
```

### 常用搜索条件

```bash
find . -name "*.py"            # 当前目录下所有 .py 文件
find . -name "*.log" -mtime -1 # 最近 1 天修改过的 .log 文件
find . -type d -name "output"  # 找名为 output 的目录（-type d）
find . -type f -size +100M     # 找大于 100MB 的文件
find . -name "*.py" | xargs grep "import torch"  # 找所有引用 torch 的 py 文件
```

---

## 六、反斜杠 `\` 的两种用法

`\` 在终端里有两种完全不同的作用，很容易混淆：

### 用法一：转义字符（让特殊字符变成普通字符）

```bash
Application\ Support   # \ 紧跟空格，告诉终端"这个空格是路径名的一部分"
```

**原理：** 空格在终端里默认是"参数分隔符"，加 `\` 后变成普通字符。

| 场景 | 示例 | 效果 |
|------|------|------|
| 路径含空格 | `cd My\ Documents` | 进入 `My Documents` 文件夹 |
| 特殊符号 | `echo \$HOME` | 打印字面量 `$HOME`，不展开变量 |
| 单引号 | `echo it\'s` | 打印 `it's` |

### 用法二：命令换行续写（行尾 `\`）

```bash
nohup env CUDA_VISIBLE_DEVICES=0 llamafactory-cli train train.yaml \
    > output/train.log 2>&1 &
```

**原理：** `\` 放在行尾（后面直接回车），告诉终端"这行没写完，下一行是续行"。  
终端看到的其实是一整行命令，`\` + 换行 完全等价于一个空格。

```bash
# 这两种写法完全等价：

# 写法一（换行续写，更易读）
find /some/very/long/path \
    -name "*.log" \
    -mtime -1

# 写法二（一行写完）
find /some/very/long/path -name "*.log" -mtime -1
```

### 两种用法对比

| | 位置 | 作用 |
|---|------|------|
| `\` + 普通字符/空格 | 字符前 | **转义**：让后面的字符失去特殊含义 |
| `\` + 换行 | 行尾 | **续行**：告诉终端命令还没写完，下一行继续 |

> 记忆口诀：`\` 放中间是转义，`\` 放行尾是续行。

---

## 七、命令参数（flag）详解

### 什么是 flag / 参数

终端命令的参数通常有两种形式：

```bash
命令  -短参数  --长参数  值  位置参数
```

| 形式 | 示例 | 含义 |
|------|------|------|
| `-字母` | `-n`、`-f`、`-L` | 短参数，单个字母，前面一个横线 |
| `--单词` | `--name`、`--port` | 长参数，完整单词，前面两个横线 |
| 无横线 | `train.py`、`/tmp` | 位置参数，按顺序传入 |

### `-d` 这类 flag 能当参数吗？

**能。** `-d` 就是一个标准的短参数（short flag），和其他 `-x` 格式完全一样。

常见例子：

```bash
mkdir -p output/logs     # -p：自动创建父目录，不报错
find . -type d           # -type d：只搜索目录（d = directory）
find . -type f           # -type f：只搜索文件（f = file）
ls -l -a                 # -l：详细列表；-a：显示隐藏文件
ls -la                   # 短参数可合并写：等价于 -l -a
ssh -L 6006:localhost:6006 -N host  # -L、-N 都是 flag
tail -n 100 train.log    # -n 后面跟数字值
```

### flag 带值 vs 不带值

```bash
# 不带值的 flag（开关型，有它就生效）
find . -type d     # -type 后面跟 d，表示"只找目录类型"
ls -a              # -a 就是开关，加了就显示隐藏文件

# 带值的 flag（参数型，后面必须跟一个值）
tail -n 100        # -n 后面跟数字
find . -name "*.py"  # -name 后面跟文件名模式
ssh -p 8416        # -p 后面跟端口号
```

### 带值 flag 的两种写法：空格 vs 紧贴

带值的短参数，**值可以紧贴在 flag 后面，也可以用空格隔开**，两种写法完全等价：

```bash
# 空格隔开（更直观）
cut -d : -f 1 /etc/passwd
tail -n 100 train.log

# 值紧贴 flag（更常见，省去一个空格）
cut -d: -f1 /etc/passwd
tail -n100 train.log
```

**`cut -d: -f1 /etc/passwd` 拆解：**

```
cut  -d:  -f1  /etc/passwd
 ↑    ↑    ↑       ↑
 │    │    │       └── 要处理的文件
 │    │    └── -f1：取第 1 个字段（field）
 │    └── -d:：以冒号 : 作为分隔符（delimiter），: 紧贴 -d 后面
 └── cut：按列切割文本的命令
```

> **为什么冒号紧贴在 `-d` 后面？**  
> 因为 `-d` 是"带值的短参数"，它的值（分隔符 `:`）可以直接跟在后面，不需要空格。  
> 这是 Unix 命令行的传统习惯，`-d:` 和 `-d :` 对大多数命令来说完全一样。

**`/etc/passwd` 文件示例：**

```
root:x:0:0:root:/root:/bin/bash
daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
hadoop:x:1000:1000::/home/hadoop:/bin/bash
```

`cut -d: -f1 /etc/passwd` 的效果：以 `:` 分割每行，取第 1 个字段（用户名）：

```
root
daemon
hadoop
```

**cut 常用组合：**

```bash
cut -d: -f1 /etc/passwd          # 取第 1 列（用户名）
cut -d: -f1,6 /etc/passwd        # 取第 1 和第 6 列（用户名、家目录）
cut -d, -f2 data.csv             # CSV 文件取第 2 列（分隔符改为逗号）
cut -c1-10 file.txt              # 取每行第 1~10 个字符（按字符位置切）
echo "a:b:c" | cut -d: -f2      # 配合管道使用，输出 b
```

---

### 短参数合并写法

多个不带值的短参数可以合并：

```bash
ls -l -a -h   # 等价于：
ls -lah       # 合并写，顺序无所谓
```

> 注意：带值的参数（如 `-n 100`）不能合并到中间，只能放最后或单独写。
