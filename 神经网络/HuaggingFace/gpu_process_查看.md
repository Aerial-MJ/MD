# GPU 显存进程查看

删除显存的程序
```bash
lsof /dev/nvidia* 2>/dev/null | awk 'NR>1 {print $2}' | sort -u | xargs -r kill -9; echo "done"
```


## 背景

`nvidia-smi` 显示的 PID 是真实的，但有时是子进程 PID，看不出是哪个"任务"（尤其是 ray 多进程训练场景）。

---

## 方法一：nvidia-smi 加完整命令行（推荐）

```bash
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv
```

---

## 方法二：实时监控（watch）

```bash
watch -n 2 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
  | while IFS="," read pid mem; do
      pid=$(echo $pid | tr -d " ")
      printf "GPU %-8s | %s\n" "$mem" "$(ps -p $pid -o cmd= 2>/dev/null | cut -c1-100)"
    done'
```

---

## 方法三：fuser 查设备文件

```bash
fuser /dev/nvidia*
```

列出所有持有 `/dev/nvidia*` 的 PID。

---

## 方法四：通过 /proc 查持有 nvidia 设备的进程

```bash
ls -la /proc/*/fd 2>/dev/null | grep nvidia | grep '\->' \
  | awk -F'/' '{print $3}' | sort -u \
  | xargs -I{} ps -p {} -o pid,cmd --no-headers 2>/dev/null
```

> 必须加 `grep '\->'` 过滤，只保留软链接行（`ls -la` 输出中还有 `total xx` 和目录头行，混进去会导致 `awk $3` 取错）。

或者用更简洁的 `lsof` 替代（推荐）：

```bash
lsof /dev/nvidia* 2>/dev/null | awk 'NR>1 {print $1, $2}' | sort -u
```
```bash
lsof /dev/nvidia* 2>/dev/null | awk 'NR>1 {print $2}' | sort -u | xargs -r kill -9; echo "done"
```

---

## 方法五：ray 场景（多进程训练）

ray worker 的 PID 和父进程是分开的，用 grep 最直观：

```bash
ps aux | grep -E "ray|python|verl" | grep -v grep
```

结合 pstree 看进程树：

```bash
pstree -p $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -1)
```

---

## 方法六：nvidia-smi 显存 + 命令行一行输出

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
  | awk -F',' '{print $2, "PID:"$1}' \
  | while read mem pid_label pid; do
      echo "$mem - $pid_label$pid - $(ps -p $pid -o args= 2>/dev/null | cut -c1-80)"
    done
```

---

## 常见问题

### OOM 根因：model_dtype 默认 fp32

FSDP 模型默认 fp32，占用约 25GB 显存，需要显式改为 bfloat16：

```yaml
# 训练配置中加入
model_dtype: bfloat16
```

或命令行参数：

```bash
--model_dtype bfloat16
```

### 显存已空但还有残留进程

```bash
# 查找并 kill 残留的 python 训练进程
ps aux | grep -E "train|verl|llamafactory" | grep -v grep
kill -9 <PID>
```
