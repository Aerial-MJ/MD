## scp命令

使用 `scp` 命令进行文件和文件夹的传输，具体格式和使用说明如下：

### 1. 从本地将文件传输到服务器
```bash
scp 【本地文件的路径】 【服务器用户名】@【服务器地址】：【服务器上存放文件的路径】
```
**示例**：
```bash
scp /Users/mac_pc/Desktop/test.png root@192.168.1.1:/root
```

### 2. 从本地将文件夹传输到服务器
```bash
scp -r 【本地文件夹的路径】 【服务器用户名】@【服务器地址】：【服务器上存放文件夹的路径】
```
**示例**：
```bash
scp -r /Users/mac_pc/Desktop/test root@192.168.1.1:/root
```

### 3. 将服务器上的文件传输到本地
```bash
scp 【服务器用户名】@【服务器地址】：【服务器上存放文件的路径】 【本地文件的路径】
```
**示例**：
```bash
scp root@192.168.1.1:/data/wwwroot/default/111.png /Users/mac_pc/Desktop
```

### 4. 将服务器上的文件夹传输到本地
```bash
scp -r 【服务器用户名】@【服务器地址】：【服务器上存放文件夹的路径】 【本地文件夹的路径】
```
**示例**：
```bash
scp -r root@192.168.1.1:/data/wwwroot/default/test /Users/mac_pc/Desktop
```

### 注意事项：
- 确保本地路径和目标服务器路径是正确的，并且你有足够的权限进行文件传输。
- 使用 `-r` 选项时，用于递归复制整个目录。
- 传输过程中可能会要求输入服务器的密码，确保提供正确的凭据。
- 如果目标服务器的 SSH 端口不是默认的 22，可以使用 `-P` 选项指定端口。例如：
  ```bash
  scp -P <port_number> -r /path/to/local/directory username@hostname:/path/to/remote/directory
  ```

## source

在 Linux 里，`source` 是一个 **内置命令**，主要作用是：

👉 **在当前 shell 里执行一个脚本文件的内容**，而不是新开一个子 shell。

------

### 举个例子

假设你在 `~/.bashrc` 里加了这一行：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

如果你直接运行：

```bash
bash ~/.bashrc
```

它会开一个新的子 shell 去执行，执行完环境变量只存在于那个子 shell 里，对你当前终端没效果。

而如果你用：

```bash
source ~/.bashrc
```

就会在 **当前终端的 shell 环境** 里执行 `.bashrc` 的内容，所以变量会立刻生效。

------

### 常见用途

1. **刷新配置文件**（不用重启终端）

   ```bash
   source ~/.bashrc
   source ~/.zshrc
   ```

2. **加载环境变量**

   ```bash
   source env.sh
   ```

3. **运行脚本函数**（让脚本里的函数和变量在当前 shell 可用）

------

⚡可以理解成：
`source` 就是 **“把这个文件的命令拿过来在我当前的终端里直接执行”**。

------

## `source` 和 `.`（点命令）

#### 1. 基本作用

两者其实是**等价的命令**：

- `source filename`
- `. filename`

都表示：
 👉 **在当前 shell 里执行指定文件的内容**（不会新开子 shell）。

------

#### 2. 使用场景

- 加载环境变量、函数：

  ```bash
  source ~/.bashrc
  . ~/.bashrc
  ```

  效果一模一样。

- 加载配置脚本：

  ```bash
  source env.sh
  . env.sh
  ```

------

#### 3. 区别

- **语法层面**：
  - `source` 是 **bash 内置命令**（更直观）。
  - `.` 是 **POSIX 标准写法**（兼容性更好，在 sh、dash 等 shell 里也能用）。
- **可读性**：
  - `source` → 一眼就能看出是“加载文件”。
  - `.` → 简短，但不直观，尤其对新手不友好。

------

#### 4. 推荐习惯

- 如果只在 Linux 下用 **bash/zsh**：随意，两者都行。
- 如果写的是 **通用脚本（需要跨平台/不同 shell）**：最好用 `.`，因为它是 POSIX 标准，所有符合 POSIX 的 shell 都支持。

------

📌 **一句话总结**：
 `source` 和 `.` 功能一样，区别在于 **可读性 vs 兼容性**
