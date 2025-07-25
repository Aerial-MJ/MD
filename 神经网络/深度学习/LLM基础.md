# 传统LLM

## 总体处理顺序（从底层到高层）

```
1. 分词（Tokenization）
    ↓
2. 词性标注（POS Tagging）
    ↓
3. 命名实体识别（NER）
    ↓
4. 句法分析（Syntactic Parsing）
    ↓
5. 依存句法分析（Dependency Parsing）
    ↓
6. 语义角色标注（Semantic Role Labeling, SRL）
    ↓
7. 语义分析（Semantic Parsing / Understanding）
```

------

## 各模块之间的依赖关系解释：

### 分词（Tokenization）

- 所有任务的基础。
- 不管你做词性、NER 还是依存句法，必须先把文本分成“词”或“子词”。

### 词性标注（POS Tagging）

- 命名实体识别、句法分析通常依赖词性信息。
- 举例：`Apple/NOUN` vs `apple/NOUN` vs `Apple/ORG`，词性不同影响后续任务。

### 命名实体识别（NER）

- 通常会用词性信息辅助判断。
- NER 属于浅层语义任务，是识别「专名」的一步，如“北京”是地点、“乔布斯”是人名。

### 句法分析 / 依存句法分析

- 理解句子结构：“谁做了什么”，比如主谓宾关系。
- 依赖词性信息，有时也用到实体信息。
- 分为：
  - **成分句法分析**：分块（NP, VP 等）
  - **依存句法分析**：找出词与词的依赖关系

目标是分析词与词之间的**语法依赖关系**，例如主谓宾结构、状语、宾语等。

我们来看这句话的依存结构（部分简化）：

```markdown
       读（谓词）
      /   |     \
  小明   书     在
 (主语) (宾语) (状语)
                 \
                图书馆（状语补充）
```

也可以用依存三元组表示：

```markdown
1. 读 ← 主语 ← 小明
2. 读 ← 宾语 ← 书
3. 读 ← 状语 ← 在
4. 在 ← 宾语 ← 图书馆
```

🔹 **重点**：谁依赖谁（主谓宾/状语等结构），**句法结构**清晰呈现。

### 语义角色标注（SRL）

- SRL 是句法分析后的高级任务，要理解谁是施事者、受事者、动作是什么。
- 依赖于句法结构，尤其是谓词动词及其依赖成分。

### 语义分析（Semantic Parsing）

- 最终理解句子意思，可以包括构建逻辑表达式、SQL 查询等。
- 是最抽象的高级语义理解，综合了前面所有信息。

------

## 在 LLM 中的情况：

- 这些任务**没有明确模块分开**，而是通过大规模预训练「统一建模」。
- 但你**依然可以通过 probing 或 fine-tuning** 提取这些信息：
  - 提一个 prompt，让模型输出词性/依存结构/SRL。
  - 也可以把它们作为下游任务微调。

------

## 总结（一图胜千言）：

```
Tokenization
   ↓
POS Tagging
   ↓
NER        ← POS辅助NER
   ↓
Syntactic Parsing
   ↓
Dependency Parsing
   ↓
Semantic Role Labeling (SRL)
   ↓
Semantic Parsing / Understanding
```

# ViT

















