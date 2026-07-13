# 无效图检测：Token 计算与自进化闭环分析

## 一、Token 是怎么算出来的

### 1.1 Token 数从哪来

在 batch_infer.py 的 call_api 函数中，prompt_tokens 和 completion_tokens 都是 vLLM 服务端自动计算并返回的，代码只是将其取出来存入 CSV。

对应代码逻辑：

```python
resp_json = resp.json()
raw = resp_json["choices"][0]["message"]["content"]
usage = resp_json.get("usage", {})
return {
    ...
    "prompt_tokens":      usage.get("prompt_tokens", ""),
    "completion_tokens":  usage.get("completion_tokens", ""),
    ...
}
```

vLLM API 返回的完整响应结构如下：

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "{\"label\": \"无效图\", ...}"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 3542,
    "completion_tokens": 287,
    "total_tokens": 3829
  }
}
```

### 1.2 Token 具体怎么算的

**prompt_tokens（输入 token）**

包含发送给模型的所有文本 + 图片的 token 总数：

| 组成部分 | 对应代码变量 | 说明 |
|---------|------------|------|
| system prompt | sys_p | 场景 + 版本对应的完整判定规则文本 |
| user prompt | user_t | 用户提问文本 |
| 图片 | img_content（base64）| 图片被模型视觉编码器转为一系列 vision token |

图片也会占 token，这是多模态模型的特点。Qwen3-VL 处理一张图片通常消耗几百到上千个 token，取决于图片分辨率和内容复杂度。

**completion_tokens（输出 token）**

就是模型生成的回复文本的 token 数，即 model_raw 那段 JSON（含 decision_path、label、reason 等）的 token 数量。

### 1.3 典型数值估算

```
输入：system_prompt（几百~一千多 token）+ 图片（几百 vision token）+ user_text（几十 token）
     → prompt_tokens ≈ 1500~4000

输出：完整 JSON 含 3~5 步 decision_path
     → completion_tokens ≈ 200~600
```

### 1.4 验证 Token 数的方法

可单独调 API 查看返回值：

```bash
curl http://10.164.28.208:8000/v1/chat/completions \
  -H "Authorization: Bearer 1839490748392423506" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "32B",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 10
  }' | python -m json.tool
```

### 1.5 小结

| 问题 | 答案 |
|------|------|
| Token 是代码算的吗？ | ❌ 是 vLLM 服务端算好返回的，代码只是取出来 |
| prompt_tokens 包含什么？ | system prompt + user prompt + 图片（vision token）|
| completion_tokens 是什么？ | 模型生成的回复（model_raw 那段 JSON）的 token 数 |
| 图片占 token 吗？ | ✅ 占，多模态模型会把图片编码成几百~上千个 token |

如需省 token，主要两个方向：① 缩短 system prompt（减少输入），② 用 max_tokens 限制输出长度（当前已设 2048）。

---

## 二、图片预处理与传入 vLLM 的流程

### 2.1 代码端：图片预处理

**读取与缩放 — encode_local 函数**

```python
def encode_local(path: str, max_side: int = 1120) -> tuple:
```

处理步骤：

| 步骤 | 代码做了什么 |
|------|------------|
| ① 读取文件 | `with open(path, "rb") as f: data = f.read()` 以二进制读取图片 |
| ② 打开为 PIL Image | `img = Image.open(io.BytesIO(data))` 解码为图像对象 |
| ③ 缩放 | `if max(w, h) > max_side: ratio = max_side / max(w, h); img = img.resize(...)` 长边超过 1120 就等比缩放 |
| ④ 色彩模式统一 | `if img.mode not in ("RGB", "L"): img = img.convert("RGB")` 非 RGB/灰度图转 RGB |
| ⑤ 重新编码为 JPEG | `img.save(buf, format="JPEG", quality=90)` 压缩为 JPEG，质量 90% |
| ⑥ 转 base64 | `base64.b64encode(buf.getvalue()).decode("utf-8")` 编码为 base64 字符串 |

**优先本地、兜底下载 — call_api 函数**

```python
if local_path and Path(local_path).exists() and Path(local_path).stat().st_size > 0:
    b64, mime = encode_local(local_path)        # ✅ 优先本地
else:
    b64, mime = download_and_encode(image_url)  # 兜底：下载 URL
```

download_and_encode 和 encode_local 做的事一样，只是多了一步 `requests.get(url)` 下载。

**组装成 API 请求格式**

```python
img_content = {
    "type": "image_url",
    "image_url": {"url": f"data:{mime};base64,{b64}"}  # base64 data URI
}
user_content = [img_content, {"type": "text", "text": user_text}]
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": user_content},  # 图片 + 文本放一起
]
```

关键结构：

```python
messages[1]["content"] = [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}},
    {"type": "text", "text": "请判断这张图片..."}
]
```

图片以 base64 data URI 的形式嵌入 JSON 请求体，通过 HTTP POST 发给 vLLM。

### 2.2 vLLM 服务端：接手后的处理

代码到 HTTP 请求发出去就结束了，后面是 vLLM 服务端的事：

```
你的代码                          vLLM 服务端
──────                          ──────────
base64 字符串  ──HTTP POST──→  ① 解码 base64 → 原始图片字节
                                ② Qwen3-VL 的 Vision Transformer 处理
                                   - 图片 → patch embedding → vision tokens
                                ③ vision tokens + 文本 tokens 拼接
                                ④ 送入 LLM 生成回复
```

vLLM 内部对图片的处理（Qwen3-VL 模型特有）：

| 步骤 | 说明 |
|------|------|
| ① 解码 | 从 base64 恢复为原始像素 |
| ② 切 patch | 图片被切成若干个小方块（patch），每个 patch 编码为一个 vision token |
| ③ 自适应分辨率 | Qwen3-VL 支持动态分辨率，会根据图片尺寸自动调整 patch 数量 |
| ④ 视觉编码 | ViT (Vision Transformer) 把每个 patch 编码成固定维度的向量 |
| ⑤ 投影对齐 | 通过一个投影层（Projector）把视觉向量映射到语言模型的词表空间 |
| ⑥ 拼接输入 | vision tokens 和文本 tokens 拼成一整条序列，送入 LLM |

### 2.3 完整数据流

```
原始图片 (jpg/png, 任意尺寸)
    │
    ▼  [代码 - encode_local]
缩放 (长边≤1120) + 转RGB + JPEG压缩(quality=90) + base64编码
    │
    ▼  [代码 - call_api]
组装成 OpenAI API 格式的 JSON (data:image/jpeg;base64,...)
    │
    ▼  [HTTP POST]
vLLM 服务端接收
    │
    ▼  [vLLM - 解码]
base64 → 原始像素
    │
    ▼  [vLLM - Qwen3-VL Vision Encoder]
图片 → patch → vision tokens (几百~上千个)
    │
    ▼  [vLLM - Projector]
vision tokens → 语言空间向量
    │
    ▼  [vLLM - LLM]
[vision tokens] + [system prompt tokens] + [user text tokens] → 生成回复
```

### 2.4 常见问题

**Q1: 缩放到 1120 和 vLLM 的处理有什么关系？**

你的缩放是网络传输层的优化——减少 base64 体积，加快请求速度。vLLM 收到图片后还会再做一次自己的预处理（切 patch、动态分辨率），和你的缩放是独立的。但你的缩放确实会影响 vLLM 内部生成的 vision token 数量——图片越小，patch 越少，token 越少。

**Q2: 能不缩放直接传原图吗？**

能，把 max_side 设很大就行（如 `max_side=4096`）。但后果是：

- base64 体积大 → HTTP 请求慢
- vLLM 生成更多 vision token → 推理慢、显存占用高

1120 是个合理的平衡点，既能看清图片内容，又不会太浪费资源。

**Q3: 为什么用 base64 而不是直接传 URL？**

代码里其实有 URL 的逻辑（download_and_encode），但最终都转成了 base64。因为 vLLM 的 OpenAI 兼容 API 格式要求图片以 base64 data URI 传入，这是 OpenAI API 标准的约定。

---

## 三、错误类型是怎么知道的 — 自进化闭环

### 3.1 核心原理：比较真实标签 vs 模型预测

错误类型不是模型告诉你的，是拿**真实标签（label）和模型预测标签（model_label）**做对比得出的。

**export_wrong.py 的逻辑**

```python
def error_type(row):
    if row['label'] == '无效图' and row['model_label'] == '有效图':
        return 'FP_漏检（无效图→有效图）'     # 应该抓出来但没抓出来
    elif row['label'] == '有效图' and row['model_label'] == '无效图':
        return 'FN_误判（有效图→无效图）'     # 不该抓但抓了
    else:
        return f'无法判断（真实={row["label"]}）'
```

二维对照表：

| | 模型预测=有效图 | 模型预测=无效图 |
|--|--------------|--------------|
| 真实=无效图 | ❌ FP 漏检（该抓没抓）| ✅ 正确 |
| 真实=有效图 | ✅ 正确 | ❌ FN 误判（不该抓抓了）|

**analyze_results.py 的逻辑**

```python
tp = ((valid['label']=='无效图') & (valid['model_label']=='无效图')).sum()       # 真无效，判无效
fp_model = ((valid['label']=='有效图') & (valid['model_label']=='无效图']).sum()  # 真有效，判无效
fn_model = ((valid['label']=='无效图') & (valid['model_label']=='有效图']).sum()  # 真无效，判有效
tn = ((valid['label']=='有效图') & (valid['model_label']=='有效图']).sum()        # 真有效，判有效
```

完全一样的思路，只是换成了 TP/FP/FN/TN 的标准术语。

### 3.2 FP 漏检的原因怎么来的

知道了哪些是 FP 之后，analyze_results.py 进一步分析模型为什么判错了。它用的是关键词匹配模型的推理文本：

```python
FP_PATTERNS = {
    "地面/楼道/电梯场景误判": ["地面", "楼道", "电梯", "楼梯"],
    "门牌号/送达地点":       ["门牌号", "送达地点", "门头", "楼道场景", "送达"],
    "仓库商品文字/标签":     ["文字清单", "价格标签", "商品名称", "文字"],
    "疑似商品/包装":         ["疑似商品", "疑似", "包装物", "包装"],
    "包含室外/室内环境元素": ["室外地面", "室内地面", "室外", "室内"],
    "其他货物/配送相关":     ["货物", "配送", "宣传布", "商品残渣", "车辆", "轮胎"],
    "模糊但认为可识别":      ["模糊但可识别", "虽模糊", "模糊但"],
}

def classify_reason(reason):
    reason = str(reason)
    matched = []
    for cat, keywords in FP_PATTERNS.items():
        if any(kw in reason for kw in keywords):
            matched.append(cat)
    return matched if matched else ["其他/未分类"]
```

流程示意：

```
model_reason 文本（模型自己输出的推理过程）
        │
        ▼  关键词匹配
"图片中可见地面、门框及脚部"  →  命中 "地面"  →  归类为 "地面/楼道/电梯场景误判"
"图片中有疑似商品包装"        →  命中 "疑似商品" →  归类为 "疑似商品/包装"
```

### 3.3 完整的自进化闭环

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  ① batch_infer.py                                   │
│     用当前 prompt 跑全量数据                          │
│     → 输出 results.csv（含 model_label, model_reason）│
│                          │                          │
│                          ▼                          │
│  ② analyze_results.py                               │
│     比较 label vs model_label → 找出 FP/FN           │
│     分析 model_reason → 按关键词归类错误原因            │
│     → "地面场景误判 42 条" "疑似商品误判 28 条" ...    │
│                          │                          │
│                          ▼                          │
│  ③ 人工分析 + 改 prompt                              │
│     发现问题：模型把"地面+脚部"误判为有效               │
│     → 在 prompt 里加一条规则：                        │
│       "仅显示地面和脚部，无其他物品 → 无效图"          │
│     保存为新版本（如 V2）                             │
│                          │                          │
│                          ▼                          │
│  ④ 回到 ①，用新 prompt 重跑                           │
│     对比 V1 vs V2 的 FP/FN 变化                      │
│     看地面场景误判是否减少                              │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 3.4 关键问题总结

| 问题 | 答案 |
|------|------|
| 错误类型谁定义的？ | 自己定义的。FP_PATTERNS 里的关键词是根据观察手动写的 |
| 模型知道自己的错误类型吗？ | 不知道。模型只输出 label + reason，错误分类是事后分析做的 |
| 为什么要分析 model_reason？ | 因为单纯知道"判错了"不够，得知道错在哪才能改 prompt |
| 关键词列表是固定的吗？ | 不是。每轮分析后发现新的错误模式，就往 FP_PATTERNS 里加新关键词 |

**一句话总结：** 错误类型不是模型告诉你的，是拿真实标签和预测标签做对比得出 FP/FN，再用关键词匹配模型输出的 reason 文本来归类错误原因，最后根据归因结果去改 prompt，形成自进化闭环。


# 两种模式的对比

不是走的 OpenAI API 形式，这个 [eval_ig_vllm.py]
---

## 两种模式的对比

| | batch_infer.py（远程 API） | eval_ig_vllm.py（本地离线） |
|---|---|---|
| 调用方式 | HTTP POST 请求远端 vLLM 服务 | 本地加载模型，直接推理 |
| API 形式 | ✅ OpenAI 兼容 API（`/v1/chat/completions`） | ❌ 不走 API，用 vLLM Python SDK |
| 图片传入 | base64 data URI 嵌入 JSON | PIL Image 对象直接传入 |
| Prompt 格式 | OpenAI messages 格式 | [apply_chat_template] 生成文本 |
| 速度 | 受网络 + 远端队列影响 | 本地 GPU 直接跑，批量并行 |

---

## eval_ig_vllm.py 的 Prompt 流程

### 第一步：构建 messages

[build_messages] 函数把训练数据转成 messages 格式：

```python
conv.append({"role": role, "content": content})
# content 可能是：
#   - 纯文本字符串（system 轮）
#   - [{"type": "image"}, {"type": "text", "text": "..."}]（user 轮，含图片）
```

这步和 OpenAI API 的 messages 格式**看起来类似**，但图片的表示方式不同：
- OpenAI API：`{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}`
- 这里：[{"type": "image"}]（只是一个占位符，实际图片数据另外传入）

### 第二步：apply_chat_template 生成文本 prompt

```python
text_prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

这步把 messages 转成模型能理解的**纯文本 prompt**，类似：

```
<|im_start|>system
你是一个图片判定助手...<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>
请判断这张图片是否有效...<|im_end|>
<|im_start|>assistant
```

注意：[<|image_pad|>] 只是占位符，实际图片数据从 [multi_modal_data]传入。

### 第三步：图片通过 multi_modal_data 单独传入

```python
prompts.append({
    "prompt": text_prompt,
    "multi_modal_data": {"image": img},  # PIL Image 对象，不是 base64
})
```

### 第四步：vLLM 本地推理

```python
llm = LLM(model=model_path, ...)
outputs = llm.generate(batch, sampling_params)
```

**直接在本地 GPU 上跑**，不走任何 HTTP 请求。

---

## 完整流程对比图

```
batch_infer.py（远程 API 模式）
═════════════════════════
messages (OpenAI格式，base64图片)
    │
    ▼  requests.post()
HTTP JSON 请求
    │
    ▼  远端 vLLM 服务
生成回复 → 从 response JSON 取结果


eval_ig_vllm.py（本地离线模式）
═════════════════════════
messages → apply_chat_template → 文本 prompt
                                         │
图片 PIL Image → multi_modal_data ←──────┘
                    │
                    ▼  llm.generate()
              本地 GPU 推理
                    │
                    ▼
              输出对象（直接取 text）
```

---

## 为什么评测用离线模式

| 原因 | 说明 |
|------|------|
| **速度** | 本地批量推理，vLLM 内部做 continuous batching，比逐条 HTTP 快 10~30x |
| **一致性** | 和训练用同一个 `apply_chat_template`（定义于 `eval_ig_vllm.py`），确保 prompt 格式完全一致 |
| **不需要服务** | 不用先启动 vLLM server，加载模型就能跑 |
| **批量处理** | 一批 64 条并行推理，API 模式要自己管并发 |

---

**一句话总结**：eval_ig_vllm.py 走的是 **vLLM 本地 SDK 调用**（[LLM] + [llm.generate]，不走 OpenAI API。Prompt 先用 [apply_chat_template] 生成文本，图片通过 [multi_modal_data] 以 PIL Image 对象直接传入，和 batch_infer.py 的 base64 + HTTP 模式完全不同。