## OpenAI Chat Completions 与 Responses API对比，内置工具使用总结

### Chat Completions

**优势：** - 通用接口，方便切换不同账户 - 会持续支持和更新，是构建通用 AI 应用的稳定选择

**不足：** - 需自己管理全部对话历史 - 不支持内建工具的使用

**代码示例：**

```python
from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-4.1",
  messages=[
    {"role": "user", "content": "简要说一下量子力学"}
  ]
)

print(completion.choices[0].message.content)
```

### Responses API

**优势：**

- 使用 `input` 字段，不强制 `messages`。
- 原生支持上下文记忆与状态追踪（无需重复发送历史）
- 内置工具如网页搜索、文件检索、计算机操作等
- 构建“能思考、能行动”的智能体（Agent）更容易

**代码示例：**

```python
response = client.responses.create(
  model="gpt-4.1",
  input=[
    {"role": "user", "content": "简要说一下量子力学"}
  ]
)

print(response.output_text)
```

**持续对话：**

```python
response2 = client.responses.create(
  model="gpt-4.1",
  previous_response_id=response.id,
  input=[{"role": "user", "content": "它和相对论有什么区别"}]
)
```

**工具调用：**

```python
response = client.responses.create(
  model="gpt-4.1",
  tool_choice="auto",
  tools=[{"type": "web_search"}],
  input=[{"role": "user", "content": "查看 GPT-5 最新时间"}]
)
```

- 目前MCP只在response api中支持
- 音频相关功能目前只有chat completions api支持，response api将会支持

### 使用方式对比

| 特性         | Chat Completions API                                         | Responses API                                        |
| ------------ | ------------------------------------------------------------ | ---------------------------------------------------- |
| 输入字段     | messages                                                     | input                                                |
| 输出字段     | choices[0].message.content                                   | output_text                                          |
| 对话历史管理 | 需自己管理                                                   | previous_response_id + store自动管理                 |
| 结构化输出   | client.chat.completions.parse(..., response_format=xxx)      | client.responses.parse(..., text_format=xxx）        |
| 函数调用     | 函数name, description, parameters, strict字段需要放在function字段内 | 函数相关字段不需要放在function字段内）               |
| 内置工具     | 支持web search，不支持MCP、file search、computer use、code interpreter | 支持MCP、file search、computer use、code interpreter |
| 应用场景     | 简单对话、客服等简易场景                                     | Agent、检索助手、任务执行型 AI                       |