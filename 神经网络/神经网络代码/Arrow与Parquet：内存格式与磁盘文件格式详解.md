# Arrow：内存格式 vs 磁盘文件

这个问题问得很精准，很多人（包括很多用过 pandas/pyarrow 的人）都搞混过这一点。

## 1. Arrow 的本质：一种内存布局规范

Apache Arrow 首先是一个**跨语言的列式内存数据结构标准**。它规定了数据在 RAM 里应该怎么摆放：

- 同一列的数据连续存储（列式，而不是像普通对象数组那样按行存储）；
- 定长类型（`int32`/`float64`...）直接是一块连续 buffer；
- 变长类型（`string`）用一个 offsets buffer + 一个 data buffer；
- 有单独的 validity bitmap 表示 null；
- 数据按 8 字节/64 字节对齐，方便 SIMD 向量化运算。

这套东西本身只描述"内存里的字节怎么排列"，跟磁盘、跟文件格式没有必然关系。它的设计目标是：不同语言（Python/Java/C++/Rust/Go...）、不同进程，只要遵守这个内存布局，就可以"零拷贝"地共享同一块数据，不用序列化/反序列化。这才是 Arrow 项目最初起家的杀手锏（Pandas 和 Spark 之间传数据不用来回转换）。

## 2. 那 `.arrow` 文件是什么？

Arrow 确实也定义了磁盘文件格式，这是它内存格式的一个"序列化落盘"版本，主要有两种：

| 格式 | 特点 | 常见后缀 |
|------|------|---------|
| Arrow IPC Streaming Format | 流式，一批一批（RecordBatch）追加写，不能随机访问，适合网络传输/管道 | `.arrows` 或无固定后缀 |
| Arrow IPC File Format（Feather V2） | 在 streaming 格式基础上加了文件头/尾的元数据索引，支持随机访问某个 RecordBatch | `.arrow` / `.feather` |

关键点是：这两种磁盘格式在设计上几乎就是把内存里的 buffer 原样"拓印"到磁盘（加一点 schema 元信息和长度标记），所以从磁盘 mmap 读回来时，理论上可以不需要解析、不需要拷贝，直接把文件映射成内存里那套 Arrow 结构（这就是 `pyarrow.memory_map` + zero-copy 读取的原理）。

所以准确的说法是：

> Arrow 的内存格式和它的磁盘文件格式是"同构"的——磁盘格式基本就是内存格式加了个壳。这是 Arrow 和 Parquet 最大的区别。

## 3. Arrow vs Parquet vs Pandas，一张图理清

```text
                内存中                          磁盘上
Pandas:      行对象/NumPy 列块      <-序列化/反序列化->   CSV / pickle
Arrow:       列式内存 buffer        <==几乎零成本拓印==>   .arrow/.feather (IPC格式)
Parquet:     (需要先解压/解码)      <--压缩+编码存储-->    .parquet
```

Parquet 是为了磁盘存储效率设计的：它用行组（row group）+ 列压缩（snappy/zstd）+ 字典编码等手段尽量把文件压小，但代价是读的时候必须经过解压和解码才能变成可用的内存结构（往往就是解码成 Arrow 格式）。所以 Parquet 更像"压缩包"，Arrow 更像"解压后直接能用的内存镜像"。

CSV/JSON 每次读都要重新 parse 字符串。

训练脚本里用的 `.parquet`（例如 verl GRPO 数据）是为了省磁盘空间和加载稳定性；而 `pyarrow` 库在读 parquet 的过程中，中间态用的正是 Arrow 内存格式作为"解压目的地"，读完再转给 pandas/verl dataset。

## 4. 代码实操：三者之间到底怎么互相转换

前面都是概念，这里落到 `pandas` / `pyarrow` 的具体 API，把"箭头"画成真正能跑的代码。核心库只有一个：`pyarrow`，它同时提供了 Arrow 内存对象、Arrow 文件读写、Parquet 文件读写的接口。

```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as feather
```

### 4.1 Pandas ⇄ Arrow（内存 ⇄ 内存，几乎零拷贝，仅做布局转换）

```python
df = pd.DataFrame({"id": [1, 2, 3], "text": ["a", "b", "c"]})

# Pandas(行对象/NumPy列块) → Arrow(列式内存 Table)
table = pa.Table.from_pandas(df)        # 得到一个 pyarrow.Table
print(type(table))                       # <class 'pyarrow.lib.Table'>

# Arrow(列式内存 Table) → Pandas
df2 = table.to_pandas()
```

- `from_pandas` / `to_pandas` 这一步不是"跨磁盘"，纯粹是**内存里两种数据结构之间的转换**：Pandas 的 NumPy 列块本身跟 Arrow 的 buffer 布局很接近（尤其数值列），所以这一步开销很小，但**不是严格零拷贝**（字符串列、有 null 的列通常还是要重新构造一遍）。
- 这一步对应笔记图里 Pandas 那一行的"内存中"格子和 Arrow 那一行的"内存中"格子——是**同一层（内存层）内部的转换**，还没有涉及磁盘。

### 4.2 Arrow(内存) ⇄ `.arrow`/`.feather`（磁盘，真正接近"零成本拓印"）

```python
# Arrow Table(内存) → .arrow 文件(磁盘)
feather.write_feather(table, "data.feather")
# 或者用 IPC File Format 的原生写法
with pa.OSFile("data.arrow", "wb") as sink:
    with pa.RecordBatchFileWriter(sink, table.schema) as writer:
        writer.write_table(table)

# .arrow/.feather 文件(磁盘) → Arrow Table(内存)，可以 mmap 零拷贝读
table_back = feather.read_table("data.feather", memory_map=True)
```

- 关键参数是 `memory_map=True`：这时候读取不是"把文件内容拷贝进一块新内存"，而是直接把磁盘文件 `mmap` 映射到进程地址空间，Arrow 的 buffer 指针直接指向这块映射内存——**这才是"零成本拓印"的真正含义**：磁盘上的字节排布和内存里要求的字节排布完全一样，不需要解析/解码，只需要"指过去"。
- 这一步对应笔记图里 Arrow 那一行"内存中 ⇔ 磁盘上"的双向箭头。

### 4.3 Parquet(磁盘) ⇄ Arrow(内存)：必须经过编解码，不能 mmap 直接用

```python
# Arrow Table(内存) → .parquet 文件(磁盘)：写入时做压缩+编码
pq.write_table(table, "data.parquet", compression="zstd")

# .parquet 文件(磁盘) → Arrow Table(内存)：读取时先解压+解码，重建 Arrow buffer
table_from_pq = pq.read_table("data.parquet")
```

- `pq.write_table` 内部会做**列压缩**（`snappy`/`zstd`/`gzip`）、**字典编码**（重复字符串只存一份+索引）、**按 row group 分块**——这些手段让文件变小，但都是"有损于直接可用性"的变换。
- `pq.read_table` 拿到的 `table_from_pq` 类型也是 `pyarrow.Table`（和 4.1 里的 `table` 类型一样！），但它是**读的时候现造出来的**，不是像 4.2 那样直接映射磁盘文件；这一步天然就有解压/解码的 CPU 开销，磁盘文件字节和最终内存 Arrow buffer 的字节排布是不同的。
- 这一步对应笔记图里 Parquet 那一行"（需要先解压/解码）⇔ 磁盘上"的箭头，箭头两端的字节内容本质不同，中间必须做真实的计算工作。

### 4.4 串起来：Parquet → Arrow → Pandas 的完整链路

实际训练脚本（比如读 verl 的 GRPO 数据）常见的调用链是：

```python
# 一行代码看似简单，但背后经历了完整的三层转换
df = pd.read_parquet("train.parquet")

# 等价于分解成以下三步：
table = pq.read_table("train.parquet")   # 磁盘 Parquet → 内存 Arrow Table（解压+解码）
df    = table.to_pandas()                 # 内存 Arrow Table → 内存 Pandas DataFrame（布局转换）
```

反过来，把训练结果重新落盘存成 parquet：

```python
df.to_parquet("output.parquet")
# 等价于：
table = pa.Table.from_pandas(df)          # Pandas → Arrow（内存布局转换）
pq.write_table(table, "output.parquet")   # Arrow → Parquet（压缩编码落盘）
```

**这就是三者关系的本质**：Arrow 是中间的"通用内存货币"，Pandas 和 Parquet 都要经过它来互相转换——Pandas 那边转换成本低（都在内存里，布局相近），Parquet 那边转换成本高（要过压缩/解压这一关），但没有 Arrow 作为中转层，Pandas 和 Parquet 之间也没法直接对话。

### 4.5 一张表总结转换成本

| 转换方向 | 是否跨内存/磁盘 | 主要开销 | 典型 API |
|---------|----------------|---------|---------|
| Pandas ⇄ Arrow(内存) | 否，纯内存 | 低（列布局重排，字符串/null列稍贵） | `pa.Table.from_pandas` / `.to_pandas()` |
| Arrow(内存) ⇄ `.arrow`/`.feather`(磁盘) | 是，但近似"拓印" | 极低（可 mmap 零拷贝） | `feather.write_feather` / `feather.read_table(memory_map=True)` |
| Arrow(内存) ⇄ `.parquet`(磁盘) | 是，且必须编解码 | 高（压缩/解压、字典编解码） | `pq.write_table` / `pq.read_table` |
| Pandas ⇄ `.parquet`(磁盘) | 是（内部自动走 Arrow 中转） | 高（内含上面两步） | `df.to_parquet()` / `pd.read_parquet()` |

## 一句话总结

Arrow 不是"只能是内存格式"，它有对应的磁盘文件格式，但这个磁盘格式的特殊之处在于它跟内存格式几乎一模一样（几乎零拷贝互转），这是它区别于 Parquet 这种"压缩型"磁盘格式的核心特征。
