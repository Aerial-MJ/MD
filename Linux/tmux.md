# Tmux

希望在 Linux 服务器后台自动运行，不用一直盯着终端。

------

## 1. 安装 tmux

大多数 Linux 服务器没自带，需要先安装：

```bash
# Debian/Ubuntu
sudo apt update && sudo apt install tmux -y

# CentOS/RHEL
sudo yum install tmux -y
```

------

## 2. 创建会话

启动一个新会话，命名为 `download`：

```bash
tmux new -s download
```

你会进入一个新的 tmux 界面，看起来和普通终端差不多。

------

## 3. 在会话里执行下载

进入项目目录并执行下载命令：

```bash
cd movie_agent/weight
git lfs install
git clone https://huggingface.co/weijiawu/MovieAgent-ROICtrl-Frozen
```

这时下载会跑起来。

------

## 4. 分离会话（后台运行）

下载过程中，你可以按下组合键 **`Ctrl+b` 然后 `d`**
 （意思是先按住 `Ctrl`，点 `b`，松开，再按 `d`）。

这样你就“离开”了 tmux 会话，但下载任务还在后台继续跑。

------

## 5. 恢复会话

如果你想再回来查看下载进度，输入：

```bash
tmux attach -t download
```

如果你有多个会话，可以先列出来：

```bash
tmux ls
```

再 attach 对应的会话。

------

## 6. 结束会话

下载完成后，可以在 tmux 窗口里输入 `exit`，会话就会关闭。
 或者在外部直接 kill：

```bash
tmux kill-session -t download
```

------

## 7. 小技巧

- **后台直接启动会话并执行命令**（不进入界面）：

  ```bash
  tmux new -d -s download "cd movie_agent/weight && git lfs install && git clone https://huggingface.co/weijiawu/MovieAgent-ROICtrl-Frozen"
  ```

- **查看日志输出**：
   如果你想把下载输出保存，命令写成：

  ```bash
  tmux new -d -s download "cd movie_agent/weight && git lfs install && git clone https://huggingface.co/weijiawu/MovieAgent-ROICtrl-Frozen > download.log 2>&1"
  ```

  然后随时用 `tail -f download.log` 看进度。

------

✅ 总结：
 tmux 的核心流程就是 **new → 跑任务 → detach → attach → exit**。
 这样就算你断开 ssh，任务也不会中断。

# Tmux window（窗口）

是 tmux 的一个核心概念。
很多人第一次用 tmux 只会 `new` / `attach` / `detach`，但其实 tmux 比 screen 强大很多，其中 **window** 就是它的特色之一。

------

## 🔹 tmux 的三层结构

可以把 tmux 想象成一个「多层工作台」：

1. **Session（会话）**
   - 最大单位，一个 session 可以理解为「一个工作区」。
   - 你常用的 `tmux new -s download` 就是开一个会话。
2. **Window（窗口）**
   - 相当于 session 里的「标签页」。
   - 一个 session 可以有多个 window，每个 window 就像一个全新的终端界面。
   - 默认第一个 window 叫 `0`，第二个叫 `1`，依次类推。
3. **Pane（面板）**
   - 每个 window 可以再分成多个 pane（分屏）。
   - 就像在同一个窗口里分成上下/左右的小格子，每个格子都是一个独立终端。

👉 类比一下：

- **Session = 浏览器**
- **Window = 标签页**
- **Pane = 标签页里的分屏**

------

## 🔹 关于 Window（窗口）

### 1. 新建窗口

在 tmux 会话里按：

```
Ctrl+b c
```

就会新建一个窗口，进入一个新的 shell，编号会从 `1` 开始。

### 2. 切换窗口

- `Ctrl+b n` → 下一个窗口 (next)
- `Ctrl+b p` → 上一个窗口 (previous)
- `Ctrl+b <数字>` → 直接跳到某个窗口，比如 `Ctrl+b 0` 回到第 0 个窗口。

### 3. 查看窗口列表

```
Ctrl+b w
```

会弹出一个窗口列表，选择你要去的。

### 4. 重命名窗口

```
Ctrl+b ,
```

可以给窗口改名字，比如 `download`、`train`，方便区分。

------

## 🔹 使用场景举例

假设你开了一个 `tmux` 会话，叫 `project`，里面你可以：

- **window 0**：专门下载权重（git clone）
- **window 1**：跑训练脚本（python train.py）
- **window 2**：实时看日志（tail -f log.txt）

你可以在不同的窗口间切换，就像浏览器开了多个标签页一样。
这样就不需要开多个 ssh 连接了。

------

✅ 总结：

- **Session**：整个工作区。
- **Window**：会话里的一个「标签页」。
- **Pane**：窗口里的分屏小终端。

所以 `tmux` 能让你在一个 SSH 连接里同时管理「多个标签页 + 多个分屏」，特别适合服务器开发。

# 安装

## tmux是什么

tmux是一个 terminal multiplexer（终端复用器），它可以启动一系列终端会话。

简单来说，安装tmux之前，一旦**与服务器断开连接**或者**关闭[xhell](https://zhida.zhihu.com/search?content_id=122724103&content_type=Article&match_order=1&q=xhell&zhida_source=entity)或其他shell终端**，我们的服务器上运行的程序就会终止，而且输入的历史消息全部消失。因此如果我们希望整晚在服务器上跑代码，我们的电脑也要整晚一直连接着服务器。而安装了tmux之后，即使我们关闭了shell终端或者不幸与服务器断开连接，我们在服务器上的程序**依然在运行**。

## 安装

root用户安装仅需一行

```text
sodu apt-get install tmux
```

非root用户太难了，需要下载源码安装，网上教程众多，不知道该用哪一个。下面是我今天刚刚安装的步骤，**安装时间2020/7/4，亲测有效。**

**1、下载**

下载tmux及其依赖软件。

```text
wget -c https://github.com/tmux/tmux/releases/download/3.0a/tmux-3.0a.tar.gz
wget -c https://github.com/libevent/libevent/releases/download/release-2.1.11-stable/libevent-2.1.11-stable.tar.gz
wget -c https://ftp.gnu.org/gnu/ncurses/ncurses-6.2.tar.gz
```

**2、解压安装包**

```text
tar -xzvf tmux-3.0a.tar.gz
tar -xzvf libevent-2.1.11-stable.tar.gz
tar -xzvf ncurses-6.2.tar.gz
```

**3、分别源码安装，先安装两个依赖包**

```text
cd  libevent-2.1.11-stable

# $HOME/tmux_depend是我的安装路径，大家可以修改

./configure --prefix=$HOME/tmux_depend --disable-shared
make && make install
```

libevent会安在 /tmux_depend / lib

```text
cd  ncurses-6.2
./configure --prefix=$HOME/tmux_depend
make && make install
```

ncurses会安在 /tmux_depend / include

**4、安装tmux**

```text
cd  tmux-3.0a
./configure CFLAGS="-I$HOME/tmux_depend/include -I/$HOME/tmux_depend/include/ncurses" LDFLAGS="-L/$HOME/tmux_depend/lib -L/$HOME/tmux_depend/include/ncurses -L/$HOME/tmux_depend/include"
make
cp tmux  $HOME/tmux_depend/bin
```

## 设置环境变量（此步骤建议手动添加到bashrc文件中）

```text
export PATH=$HOME/tmux_depend/bin:$PATH
source ~/.bashrc
```
