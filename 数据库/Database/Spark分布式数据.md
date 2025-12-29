# 分布式数据库

### 1. Hadoop 是什么？

👉 **Hadoop 是一个大数据处理平台/生态系统**，用于大规模数据存储和计算。
它的核心组件主要有：

- **HDFS（分布式文件系统）** — 存储数据，把数据分布到很多机器上。
- **MapReduce** — 处理数据的计算模型/引擎（较早期设计）。
- **YARN** — 资源管理器和任务调度器。
  📌 Hadoop 最早就是用 HDFS 存数据 + 用 MapReduce 做计算的组合。 ([IBM](https://www.ibm.com/think/insights/hadoop-vs-spark?utm_source=chatgpt.com))

------

### 2. HDFS 是什么？

👉 **HDFS（Hadoop Distributed File System）** 是 Hadoop 里面的 **分布式文件系统**，负责把大量数据分片并放在集群中的不同机器上，以便大数据可靠存储和访问。 

------

### 3. MapReduce 是什么？

👉 **MapReduce 是一个分布式计算模型/框架**，设计来在 Hadoop 集群上并行处理数据。
它的特点：

- 把任务切成 **Map** 和 **Reduce** 两个阶段
- 每个阶段在很多机器上并行执行
- 适合大规模批处理任务（但速度较慢，因为每步都写磁盘）
  👉 在 Hadoop 系统中最早是用 MapReduce 做计算的。 ([IBM](https://www.ibm.com/think/insights/hadoop-vs-spark?utm_source=chatgpt.com))

------

### 4. Hive 是什么？

👉 **Hive 是一个数据仓库工具**，它提供 **类似 SQL 的查询语言（HiveQL）** 让用户编写 SQL 来分析 Hadoop 上的数据。
但实际执行时：

- 早期 Hive 的 SQL 会被转换成 **MapReduce 任务** 去执行。
- Hive **依赖 HDFS 存数据**。 ([T-Systems 文档](https://docs.otc.t-systems.com/mapreduce-service/umn/faq/mrs_overview/what_is_the_relationship_between_hive_and_other_components.html?utm_source=chatgpt.com))

📌 这让不会写 MapReduce 代码的人也能用类 SQL 的方式分析大数据。

------

### 5. Spark 是什么？

👉 **Spark 是一个更现代、更快的分布式计算引擎**，它比 MapReduce 更快、更灵活。
关键特点：

- **内存计算**（不每一步写磁盘，速度快）
- 支持 **批处理、实时流、机器学习、图计算** 等多种任务
- 能直接读取 HDFS 数据，也能处理 Kafka、数据库、云存储等数据源
  👉 Spark 的速度通常比 MapReduce 快很多。 ([IBM](https://www.ibm.com/think/insights/hadoop-vs-spark?utm_source=chatgpt.com))

------

## 它们之间的关系（一句话版）

📌 **Hadoop = HDFS（存数据） + 计算引擎（最早是 MapReduce） + 管理 YARN**。 ([IBM](https://www.ibm.com/think/insights/hadoop-vs-spark?utm_source=chatgpt.com))
📌 **Hive 是 SQL 查询层**，让你用 SQL 去分析 Hadoop 上的数据（最早内部转 MapReduce 执行）。 ([T-Systems 文档](https://docs.otc.t-systems.com/mapreduce-service/umn/faq/mrs_overview/what_is_the_relationship_between_hive_and_other_components.html?utm_source=chatgpt.com))
📌 **Spark 是更快的计算引擎**，可以替代 MapReduce 运行任务（包括 Hive SQL 查询）。 ([袋鼠云](https://www.dtstack.com/bbs/article/33408?utm_source=chatgpt.com))

------

## 一张最简单的“概念图”

```
  数据存储层：HDFS （整个集群的数据都放这里）
              ↓
  计算引擎：
     MapReduce — Hadoop 原有引擎
     Spark — 新一代引擎（更快、支持多种处理）
              ↓
  用户接口：
     Hive（写 SQL 查询，然后底层让 MapReduce 或 Spark 去执行）
```

------

## 核心记忆点（最重要）

✔ **HDFS：存数据**
✔ **MapReduce：旧的计算方式**
✔ **Hive：用 SQL 查询大数据（靠其他引擎执行）**
✔ **Spark：现今更常用、更强大的计算引擎**

------

“**HDFS 是仓库、MapReduce/Spark 是工人、Hive 是会 SQL 的指令官**”。