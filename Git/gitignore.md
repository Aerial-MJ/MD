# gitignore

### 简短回答：

> **如果文件已经被 Git 跟踪（tracked）了，再在 `.gitignore` 里写它，是没用的。**

`.gitignore` 只对**还没有被 Git 追踪**的文件有效。

------

### 解释一下原理：

Git 的工作流程大致是：

1. 你第一次执行

   ```
   git add <文件>
   ```

   这时这个文件被 **Git 追踪（tracked）**。

2. 从那之后，Git 就会继续监控它的变化，无论 `.gitignore` 写不写它。

所以：

- 如果某个文件 **已经提交（commit）过一次**，
- 然后你再把它加进 `.gitignore`，

它仍然会继续出现在 `git status` 中。

------

### 想让它真正被忽略（停止追踪）怎么办？

你需要**告诉 Git 停止追踪它**，然后再 `.gitignore` 生效。

操作步骤 👇：

#### 在 `.gitignore` 中写入要忽略的路径

例如：

```bash
data/
*.log
.env
```

#### 停止追踪已经被提交的文件

```bash
git rm --cached <文件或文件夹>
```

例如：

```bash
git rm --cached .env
git rm --cached -r data/
```

> ⚠️ `--cached` 表示 **只从 Git 索引中移除，不删除本地文件**。

#### 提交更改

```bash
git commit -m "Remove tracked files and update .gitignore"
```

之后，这些文件就不会再被追踪了，也不会再上传。

------

### 举个具体例子：

你之前提交了：

```bash
config.json
```

然后你才在 `.gitignore` 里加了：

```bash
config.json
```

要让它生效：

```bash
git rm --cached config.json
git commit -m "Stop tracking config.json"
```

之后就 OK 了。