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
