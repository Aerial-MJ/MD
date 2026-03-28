# VLLM

安装VLLM

pip install vllm==0.6.4.post1 --extra-index-url https://download.pytorch.org/whl/cu121

## VLLM



```python
llm = LLM(
    model=model_path,
    tensor_parallel_size=4,
    max_model_len=4096,   # 等于模型真实 context window
)

sampling_params = SamplingParams(
    max_tokens=256,       # 你真实需要的输出长度
    temperature=0.0,
)

```

### 参数

- `n`：要生成的序列的数量，默认为 1。
- `best_of`：从多少个序列中选择最佳序列，需要大于 n，默认等于 n。
- `temperature`：用于控制生成结果的随机性，较低的温度会使生成结果更确定性，较高的温度会使生成结果更随机。
- `top_p`：用于过滤掉生成词汇表中概率低于给定阈值的词汇，控制随机性。
- `top_k`：选择前 k 个候选 token，控制多样性。
- `presence_penalty`：用于控制生成结果中特定词汇的出现频率。
- `frequency_penalty`：用于控制生成结果中词汇的频率分布。
- `repetition_penalty`：用于控制生成结果中的词汇重复程度。
- `use_beam_search`：是否使用束搜索来生成序列。
- `length_penalty`：用于控制生成结果的长度分布。
- `early_stopping`：是否在生成过程中提前停止。
- `stop`：要停止生成的词汇列表。
- `stop_token_ids`：要停止生成的词汇的ID列表。
- `include_stop_str_in_output`：是否在输出结果中包含停止字符串。
- `ignore_eos`：在生成过程中是否忽略结束符号。
- `max_tokens`：生成序列的最大长度。
- `logprobs`：用于记录生成过程的概率信息。
- `prompt_logprobs`：用于记录生成过程的概率信息，用于特定提示。
- `skip_special_tokens`：是否跳过特殊符号。
- `spaces_between_special_tokens`：是否在特殊符号之间添加空格。

这些参数的设置通常取决于具体需求和模型性能。以下是一些常见的设置指导方法：

- `temperature`：较低的温度（如0.2）会产生更确定性的结果，而较高的温度（如0.8）会产生更随机的结果。您可以根据您的需求进行调整。
- `presence_penalty、frequency_penalty 和 repetition_penalty`：这些参数可以用于控制生成结果中的词汇分布和重复程度。您可以根据您的需求进行调整。
- `use_beam_search`：束搜索通常用于生成更高质量的结果，但可能会降低生成速度。您可以根据您的需求进行调整。
- `length_penalty`：这个参数可以用于控制生成结果的长度。较高的值会产生更长的结果，而较低的值会产生更短的结果。您可以根据您的需求进行调整。
- `early_stopping`：如果您不希望生成过长的结果，可以设置此参数为True。
- `stop 和 stop_token_ids`：您可以使用这些参数来指定生成结果的结束条件。

```python

```

## Paged Attention

