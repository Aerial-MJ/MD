# vLLM 调用方式与 OpenAI 协议生态

> 本文关注 vLLM 本地部署后，除了 HTTP API 还有哪些调用方式；以及为什么几乎所有开源模型（包括 Qwen）都选择兼容 OpenAI 的 API 协议；顺带梳理 LangChain 的 tool calling 为什么在很多模型上用不了。
>
> 关联笔记：
> - vLLM 推理机制基础概念（Prefill/Decode/PIECEWISE/CUDA Graph） → [vLLM参数与推理机制详解](vLLM参数与推理机制详解.md)
> - 并发估算、显存/算力综合分析 → [vLLM并发估算与显存-算力综合分析](vLLM并发估算与显存-算力综合分析.md)

## 目录

1. [vLLM 的几种调用方式](#一vllm-的几种调用方式)
2. [为什么大家都兼容 OpenAI API 协议](#二为什么大家都兼容-openai-api-协议)
3. [为什么小模型用不了 LangChain 的 tool calling](#三为什么小模型用不了-langchain-的-tool-calling)

---

## 一、vLLM 的几种调用方式

vLLM 提供的调用方式主要分两大类：**离线批量推理（Offline Inference）** 和 **在线服务（Online Serving）**。

### 1. 离线推理 —— `LLM` 类（Python API，不经过网络）

不需要 `vllm serve` 起服务，直接在 Python 进程内加载模型、跑推理，没有 HTTP 开销，适合一次性批量跑数据：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/Qwen3-VL-32B-Instruct",
    tensor_parallel_size=2,   # 双卡 tp
    trust_remote_code=True,
    max_model_len=7500,
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=1500)

# 多模态输入（图片+文本）
outputs = llm.chat(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
            {"type": "text", "text": user_text},
        ]},
    ],
    sampling_params=sampling_params,
)
print(outputs[0].outputs[0].text)
```

**优点**：省掉 HTTP/JSON 序列化开销，进程内直接调用，适合脚本化批量跑；能更精细地控制批处理（把多条 prompt 一次性传入 `llm.generate`/`llm.chat` 的 list，vLLM 自动做 continuous batching）。

**缺点**：模型常驻在这一个 Python 进程里，不能像 API 服务那样被多个客户端/多个脚本同时复用；每次跑都要重新加载模型（除非把进程做成常驻服务）。

### 2. 在线服务 —— `vllm serve`

`vllm serve` 启动一个**兼容 OpenAI API 规范**的 HTTP 服务，暴露 `/v1/chat/completions`、`/v1/completions`、`/v1/models` 等接口，和 OpenAI 官方 API 的请求/响应格式几乎一致。

除了用 `requests` 手搓 HTTP 请求，也可以直接用**官方 `openai` Python SDK**，只需把 `base_url` 指向本地 vLLM 服务：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="随便填，vLLM 默认不校验，除非启动时加了 --api-key",
)

resp = client.chat.completions.create(
    model="qwen3vl-32b",   # 对应 --served-model-name
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": user_text},
        ]},
    ],
    temperature=0.0,
    max_tokens=1500,
)
print(resp.choices[0].message.content)
```

**好处**：用官方 SDK 自带重试、流式（`stream=True`）、异步（`AsyncOpenAI`）支持，比手写 `requests` 更省心，尤其是想做 `async/await` 并发（比 `ThreadPoolExecutor` 更轻量）时更方便。批量推理脚本如果想优化并发效率，可以考虑把 `requests + ThreadPoolExecutor` 换成 `AsyncOpenAI + asyncio.gather`。

### 3. `vllm serve` 里也支持原生 Completions 接口

除了 chat 接口 `/v1/chat/completions`，还有更底层的 `/v1/completions`（纯文本续写，不走 chat template），一般用不上，除非要自己拼 prompt 模板而不想让 vLLM 自动套用聊天模板。

### 4. gRPC / 其他生产级方案

vLLM 本身没有官方 gRPC 接口；如果需要更高吞吐的服务化部署，通常是在 vLLM OpenAI 兼容服务前面再套一层网关（如 Triton Inference Server 的 vLLM backend，或用 Ray Serve 包装），这些属于生产级部署方案，日常跑实验用不到。

### 如何选择

```
只是本地批量跑一次性任务（不需要多客户端复用）
  → 优先用离线 LLM 类，省掉 HTTP 序列化和网络往返，
    且 vLLM 的 continuous batching 在离线模式下调度效率更高

需要"边跑边看日志/中途能重启客户端脚本而不用重新加载模型"的灵活性，
或需要多个脚本/客户端同时访问同一个模型服务
  → vllm serve + HTTP API（requests 或 openai SDK 均可）
```

---

## 二、为什么大家都兼容 OpenAI API 协议

本地部署的是 Qwen，为什么调用协议长得跟 OpenAI 一模一样？这和模型是谁训练的没有关系，而是**行业生态选择了 OpenAI 的接口格式作为事实标准（de facto standard）**。

### 1. OpenAI 先发制人，定义了这套"语言"

2022~2023 年 ChatGPT/GPT-4 API 爆火时，`messages=[{"role": "system"/"user"/"assistant", "content": ...}]` 这套 chat completion 格式是第一个被大规模采用的标准接口。当时几乎所有上层应用（LangChain、LlamaIndex、各种 Agent 框架、IDE 插件、聊天机器人前端）都是照着这套协议写的。

### 2. 生态锁定效应（Ecosystem Lock-in）

一旦成千上万个开源项目、企业内部系统都是基于 `client.chat.completions.create(...)` 这套写法开发的，后来者（无论是 vLLM、Ollama、TGI，还是国产的 Qwen/GLM/DeepSeek 官方 API）如果想让用户"零成本迁移"，最省事的做法就是让自己的服务兼容这套协议，而不是发明一套新协议让所有人重写代码。

这就是为什么：
- vLLM/SGLang/TGI 起服务时都提供 `/v1/chat/completions` 接口
- 阿里云百炼、通义千问官方 API、DeepSeek 官方 API、月之暗面 Kimi API，全部都是"OpenAI 兼容"模式，只需要换 `base_url` 和 `api_key` 就能无缝切换模型
- 国内几乎所有大模型厂商发布 API 时都会强调一句"完全兼容 OpenAI SDK"

### 3. 协议本身足够通用，和"谁训练模型"无关

OpenAI 的 chat completion 协议本质上只是定义了：
- **输入**：一个角色化的消息列表（system/user/assistant/tool），支持多模态 content（文本+图片）
- **输出**：一个统一的 JSON 结构（`choices[0].message.content`、`usage` token 统计等）

这套抽象和底层是 GPT 还是 Qwen、Llama、GLM 完全无关——协议只是"打包/解包数据的格式"，跟运输的"货物"（也就是具体模型的能力）是两回事。就像 HTTP 协议不关心服务器上跑的是 Nginx 还是 Apache 一样。

### 4. 对实际使用的好处

正因为 vLLM 遵循这套协议：
- 今天用 Qwen3-VL-32B，明天想换成 InternVL 或者别的多模态模型，只需要改 `--model` 参数，代码完全不用动
- 甚至可以直接把 `base_url` 从本地 vLLM 换成阿里云百炼的 Qwen 官方 API 地址，代码原封不动
- 想对比"本地 Qwen3-VL vs GPT-4V 谁判断得更准"，同一套调用代码改两个参数就能测，不用为每个厂商写一套适配代码

### 一句话总结

```
不是"Qwen 在用 OpenAI 的协议"，
而是"vLLM（作为推理引擎）选择了实现 OpenAI 协议的服务端接口，
     而 Qwen 只是被 vLLM 加载的模型权重"

协议层和模型层是解耦的。
Qwen 官方自己发布 API 服务时同样也是照抄这套协议，是整个行业约定俗成的结果。
```

---

## 三、为什么小模型用不了 LangChain 的 tool calling

很多模型定义了 tool 也调用不了，这大概率不是调用姿势的问题，而是**模型本身能力 + 服务端实现**两层原因共同导致的。

### 1. Tool Calling 本质上依赖模型的"原生训练能力"

LangChain 的 `bind_tools()` / `.invoke()` 背后做的事情很简单：
- 把定义的 tool schema（函数名、参数、描述）转换成 OpenAI 协议里的 `tools` 字段，塞进请求
- 把模型返回的 `tool_calls` 字段解析出来，变成 LangChain 的 `AIMessage.tool_calls`

**关键点**：模型必须在预训练/微调阶段专门学过怎么按照 `{"name": "xxx", "arguments": {...}}` 这种结构化格式输出，才能稳定触发 tool call。这不是"调用姿势"的问题，而是模型压根没被训练成"看到 `tools` 参数就应该怎么反应"。

很多小模型（尤其是没做过专门的 function-calling SFT 的通用小模型，比如很多 <7B 的基础对话模型、纯 VL 多模态模型）根本不具备这个能力，即使把 `tools` 塞进去了，它也只会当成普通文本提示，用自然语言"聊"回来，而不会输出规范的 `tool_calls` 结构。多模态模型（VL 系列）通常更专注于视觉理解，很多版本没有针对 tool calling 做专门优化，尤其某些早期/轻量版本。

### 2. 服务端（推理引擎）要不要"翻译"这套协议

这里还有第二层坑，很容易被忽略：**vLLM/Ollama 等推理引擎本身要不要解析模型输出并转成 OpenAI 的 `tool_calls` 格式**。

即使模型本身"内心"是按照它自己训练时用的特殊 token/格式（比如 Qwen 系列常用 `<tool_call>...</tool_call>` 这种 XML 风格标签）输出了调用意图，如果推理引擎没有配置对应的 "tool-call parser"，它只会把这段文本原样塞进 `content` 字段返回，而不会填充到 OpenAI 协议要求的 `tool_calls` 字段里。LangChain 依赖的正是 `tool_calls` 这个结构化字段，看不到它就会认为"这次没有调用工具"。

vLLM 的解决方案是需要在 `vllm serve` 启动参数里显式开启：

```bash
vllm serve /path/to/model \
  --enable-auto-tool-choice \
  --tool-call-parser hermes   # 或 qwen / llama3_json 等，取决于模型家族
```

如果没加这两个参数，哪怕模型本身支持 tool calling，vLLM 服务端也不会帮你解析出来，LangChain 自然拿到的就是空的 `tool_calls`。

### 3. 哪些模型确实能兼容 LangChain 的 tool calling

按能力从强到弱大致分层：

| 层级 | 代表模型 | 说明 |
|---|---|---|
| 原生强支持 | GPT-4o/4.1、Claude 3.5+、Qwen2.5-Instruct 系列（非VL）、DeepSeek-V3、GLM-4 | 专门做过 function-calling 微调，指令遵循稳定 |
| 部分支持 | Qwen2.5-VL、Llama 3.1/3.3-Instruct | 支持，但多模态/小尺寸版本稳定性打折扣 |
| 基本不支持 | 大多数 <7B 基础模型、纯 Base（非 Instruct/Chat）模型、专精 VL 早期版本 | 没做过专项训练，塞 tools 也没用 |

多模态视觉模型（如 Qwen-VL 系列）的训练重心在图文理解，tool calling 能力通常弱于同尺寸的纯文本 Instruct 模型（比如 Qwen2.5-Instruct）。

### 4. 排查建议

1. **确认模型本身支持**：查一下具体模型版本的官方文档/模型卡（如 Qwen 官方 HuggingFace 页面），看有没有明确写"support function calling"。
2. **确认 vLLM 启动参数**：检查 `vllm serve` 时有没有加 `--enable-auto-tool-choice --tool-call-parser <对应parser>`。
3. **换纯文本模型验证**：如果只是想验证 LangChain 链路本身没问题，可以先临时换 Qwen2.5-Instruct（非VL版本）这种明确支持 tool calling 的模型测一遍，排除"是不是代码写错了"。
4. **看原始返回**：不要只看 LangChain 封装后的结果，直接用 `requests` 打一下 `/v1/chat/completions` 看原始 JSON 里到底有没有 `tool_calls` 字段，还是模型把调用意图写在了 `content` 里的纯文本（比如带着 `<tool_call>` 标签），这样能一眼区分是"模型没输出"还是"引擎没解析"。
