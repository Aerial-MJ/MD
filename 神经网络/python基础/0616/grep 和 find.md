## find 和 grep 简要用法

---

### find —— 找**文件**

```bash
find 从哪找  条件
```

```bash
find . -name "*.py"          # 当前目录找所有 .py 文件
find . -name "config.yaml"   # 找指定文件名
find . -type f               # 只找文件
find . -type d               # 只找目录
find /home -name "*.log"     # 在 /home 下找 .log 文件
```

---

### grep —— 找**文件内容**

```bash
grep 关键词  文件
```

```bash
grep "save_freq" config.yaml        # 在某文件里找关键词
grep -r "save_freq" .               # 递归搜索当前目录所有文件
grep -r "save_freq" . --include="*.py"  # 只搜 .py 文件
grep -n "save_freq" config.yaml     # 显示行号
grep -i "save_freq" config.yaml     # 忽略大小写
```

---

### 一句话区别

| 命令 | 用来找 | 例子 |
|------|--------|------|
| `find` | **文件本身**（按名字/类型） | 找哪些文件叫 config.yaml |
| `grep` | **文件内容**（按关键词） | 找哪些文件里有 save_freq |

---

### 组合使用

```bash
# 先用 find 找文件，再用 grep 搜内容
grep -r "max_model_len" . --include="*.py"
```

> 💡 记忆方法：**find 找文件，grep 找内容** 😊


## lsof 是什么

**lsof = List Open Files**，列出所有**被打开的文件**

---

### 为什么查端口要用它？

> Linux 里**一切皆文件**，网络连接、端口也被当作文件处理，所以 lsof 也能查端口

---

### 常用用法

```bash
# 查看某个端口被谁占用
lsof -i :8080
lsof -i :6379    # 查 redis 端口

# 查看某个进程打开了哪些文件
lsof -p 1234     # 1234 是进程PID

# 查看某个文件被哪些进程占用
lsof /var/log/nginx.log

# 查看某个用户打开的文件
lsof -u root
```

---

### 输出示例

```bash
$ lsof -i :8080
COMMAND   PID  USER   TYPE  NODE NAME
python   1234  root   IPv4  TCP  *:8080 (LISTEN)
#  ↑进程名  ↑PID  ↑用户        ↑端口状态
```

---

### 常用场景

| 场景 | 命令 |
|------|------|
| 端口被占用，找是谁 | `lsof -i :端口号` |
| 删不掉文件，找谁在用 | `lsof 文件路径` |
| 查某进程打开了什么 | `lsof -p PID` |

> 💡 最常用的场景就是：**启动服务报端口占用，用 lsof 找到占用的进程，再 kill 掉** 😊