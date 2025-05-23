## main.py

整个脚本是为了驱动一个**基于 LLM 的多智能体问答系统**运行在一个指定的数据集（如 `medqa`）上，并支持难度自适应的推理流程。

### 1. 引入依赖和工具函数

```python
import os
import json
import random
import argparse
from tqdm import tqdm
from termcolor import cprint
from pptree import print_tree
from prettytable import PrettyTable
```

- 导入了很多工具库，包括进度条（`tqdm`）、彩色输出（`termcolor`）、树状结构可视化（`pptree`）、表格美化（`PrettyTable`），这些用于**增强交互性和调试体验**。

```python
from utils import (
    Agent, Group, parse_hierarchy, parse_group_info, setup_model,
    load_data, create_question, determine_difficulty,
    process_basic_query, process_intermediate_query, process_advanced_query
)
```

- 从 `utils.py` 引入了**核心组件函数和类**，比如：
  - `Agent`, `Group`: 多智能体系统中的基本单位。
  - `setup_model`: 初始化 LLM（如 OpenAI 模型）。
  - `process_*_query`: 不同难度等级的推理过程函数。
  - `create_question`, `determine_difficulty`: 用于处理输入数据和自适应调度难度。

### 2. 解析命令行参数

```python
parser = argparse.ArgumentParser()
"""
这一句是创建一个 ArgumentParser 对象，它是 Python 标准库 argparse 提供的，用于处理命令行参数的。
比如你运行脚本时可以这样写：
python main.py --dataset medqa --model gpt-4o-mini --difficulty advanced --num_samples 20
这个 parser 就会帮你解析这些参数，并把它们变成程序里的变量。
"""

parser.add_argument('--dataset', type=str, default='medqa')
parser.add_argument('--model', type=str, default='gpt-4o-mini')
parser.add_argument('--difficulty', type=str, default='adaptive')
parser.add_argument('--num_samples', type=int, default=100)
args = parser.parse_args()


"""
这句会去实际读取命令行中传入的参数，并把它们打包成一个对象 args，你就可以这样用：
args.dataset        # ==> 'medqa'
args.model          # ==> 'gpt-4o-mini'
args.difficulty     # ==> 'advanced'
args.num_samples    # ==> 20
简单来说：args 是一个可以通过点（.）语法访问的配置对象，读取的是你运行程序时通过命令行传进来的内容。
"""
```

- 这里允许用户指定：
  - 数据集（如 `medqa`）
  - 模型名称（如 `gpt-4o-mini`）
  - 难度等级（可以设为 `'basic'`, `'intermediate'`, `'advanced'`, 或 `'adaptive'`）
  - 执行样本数量（例如最多运行100个问题）

### 3. 初始化模型和数据

```python
model, client = setup_model(args.model)
test_qa, examplers = load_data(args.dataset)
```

- `setup_model`：根据输入的模型名，初始化语言模型和 client（大概率是 OpenAI 或其他 API 封装）。
- `load_data`：加载测试数据和“示例题”（**即 few-shot prompting 所需的 examples**）。

###  **4. 初始化 agent 表情符号**

```python
agent_emoji = ['🧑‍⚕️', ...]

random.shuffle(agent_emoji)
# 作用是：将 agent_emoji 列表中的元素打乱顺序（就地乱序处理）。
"""
因为这个列表是为多智能体准备的“emoji 头像”，用于展示每个 Agent。打乱顺序可以：
保证每次运行时分配给智能体的 emoji 是随机的；
增加可视化多样性，避免总是第一个 Agent 是固定的头像。
"""
```

- 这些是用于每个 agent 的视觉标识，虽然只是 **cosmetic**，但在多智能体输出结果中帮助直观区分谁是谁。

### **5. 主循环处理每个样本**

```python
results = []
for no, sample in enumerate(tqdm(test_qa)):
    if no == args.num_samples:
        break
```

- 遍历测试数据中的每个问题，限制最多运行 `args.num_samples` 条。

### **6. 为每道题创建 question & 判断难度**

```python
	question, img_path = create_question(sample, args.dataset)
    difficulty = determine_difficulty(question, args.difficulty)
```

- `create_question`: 提取文字问题，有些问题可能附带图片。
- `determine_difficulty`: 根据题目的内容和设置判断难度（如果设置为 `adaptive`，可能会自动评估）。

### **7. 调用不同的 agent 推理流程**

```python
    if difficulty == 'basic':
        final_decision = process_basic_query(question, examplers, args.model, args)
    elif difficulty == 'intermediate':
        final_decision = process_intermediate_query(question, examplers, args.model, args)
    elif difficulty == 'advanced':
        final_decision = process_advanced_query(question, args.model, args)
```

- **关键逻辑分支**：根据难度选择不同的推理方法。
  - `basic`：可能是单智能体简单回答。
  - `intermediate`：可能有小规模 agent 协作或 chain-of-thought。
  - `advanced`：可能是 full multi-agent cooperation，甚至有 agent hierarchy。

### **8. 保存结果（结构化）**

```python
    if args.dataset == 'medqa':
        results.append({
            'question': question,
            'label': sample['answer_idx'],
            'answer': sample['answer'],
            'options': sample['options'],
            'response': final_decision,
            'difficulty': difficulty
        })
```

- 把每道题目的信息、真实答案、模型预测结果、难度一起保存进 `results`。

## utils.py

### Agent

`Agent` 类就是你系统里的「智能体」。每个 Agent 有：

- 一个角色（`role`）
- 一个行为指令（`instruction`）
- 一个 LLM 模型（GPT or Gemini）
- 一段会话历史（messages）
- 可以进行问答（`chat()`）或控制温度的测试（`temp_responses()`）

```python
def __init__(self, instruction, role, examplers=None, model_info='gpt-4o-mini', img_path=None):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path
	"""
		这个构造器用来初始化一个 Agent，包含：
		instruction：系统 prompt，设定 Agent 的行为准则，比如「你是一名心脏病专家」
		role：Agent 的角色名（只是记录，辅助分析）
		examplers：可选的 few-shot 样例（加入对话上下文）
		model_info：使用的模型，如 "gpt-4o-mini" 或 "gemini-pro"
		img_path：可选，图片路径（目前未启用）
	"""

        if self.model_info == 'gemini-pro':
            self.model = genai.GenerativeModel('gemini-pro')
            self._chat = self.model.start_chat(history=[])
            """
            Gemini 使用 Google 的 SDK，构建一个会话对象 _chat，可以多轮交互。
            """
            
        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.client = OpenAI(api_key=os.environ['openai_api_key'])
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            """
            OpenAI 模型（GPT）使用 openai 包，构建 messages 列表作为对话上下文。
            """
            
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})
                    #并加入 few-shot 示例：
                    #这是标准 few-shot learning 构造方式，帮 LLM 理解格式和风格。
                    
    def chat(self, message, img_path=None, chat_mode=True):
        if self.model_info == 'gemini-pro':
            for _ in range(10):
                try:
                    response = self._chat.send_message(message, stream=True)
                    responses = ""
                    for chunk in response:
                        responses += chunk.text + "\n"
                    return responses
                except:
                    continue
            return "Error: Failed to get response from Gemini."
        """
        尝试最多 10 次获取回复（有些时候 Gemini API 不稳定）
        使用 stream=True 实时获取响应流
		"""

        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.messages.append({"role": "user", "content": message})
            
            if self.model_info == 'gpt-3.5':
                model_name = "gpt-3.5-turbo"
            else:
                model_name = "gpt-4o-mini"

            response = self.client.chat.completions.create(
                model=model_name,
                messages=self.messages
            )

            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            return response.choices[0].message.content
        """
        将用户输入加入上下文
		用最新上下文请求 LLM 响应
		将 LLM 响应也加入上下文，形成完整历史
		"""

    def temp_responses(self, message, img_path=None):
        if self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:      
            self.messages.append({"role": "user", "content": message})
            
            temperatures = [0.0]
            
            responses = {}
            for temperature in temperatures:
                if self.model_info == 'gpt-3.5':
                    model_info = 'gpt-3.5-turbo'
                else:
                    model_info = 'gpt-4o-mini'
                response = self.client.chat.completions.create(
                    model=model_info,
                    messages=self.messages,
                    temperature=temperature,
                )
                
                responses[temperature] = response.choices[0].message.content
                
            return responses
        
        elif self.model_info == 'gemini-pro':
            response = self._chat.send_message(message, stream=True)
            responses = ""
            for chunk in response:
                responses += chunk.text + "\n"
            return responses
        """
        这个函数用来以不同的 temperature（随机性）生成多种回答，可以用于：
		检查模型稳定性
		做多数投票（majority voting）
		看多样性（diversity）
		"""
```

### Group

它定义了一个多智能体系统中“一个团队”的行为方式，尤其是如何**以合作的方式回答医学问题**。

```python
class Group:
    def __init__(self, goal, members, question, examplers=None):
        """
        goal	小组的整体目标，比如“解答医学问题”
		members	每个成员的信息（通常是一个 dict 的列表）
		question	要求解的问题（医学题）
		examplers	示例数据，用于支持 few-shot prompting
        """
        self.goal = goal
        self.members = []
        
        """
        self.goal = goal
		self.members = []
		先保存目标，准备成员列表。
		"""
        
        for member_info in members:
            _agent = Agent('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()), role=member_info['role'], model_info='gpt-4o-mini')
            """
            用每个成员的角色和专业描述，初始化一个 Agent
            指定模型是 gpt-4o-mini
            """
            
            _agent.chat('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()))
            self.members.append(_agent)
            #给 Agent 发一条系统消息，相当于“让他进入角色”
			#例如：“你是一个心脏病专家，你擅长心电图分析和临床诊断。”
        self.question = question
        self.examplers = examplers


    def interact(self, comm_type, message=None, img_path=None):
        """
        def interact(self, comm_type, message=None, img_path=None):
		目前只实现了 comm_type == 'internal' 的逻辑，也就是 组内沟通模式。
		"""
        if comm_type == 'internal':
            lead_member = None
            assist_members = []
            for member in self.members:
                member_role = member.role

                if 'lead' in member_role.lower():
                    lead_member = member
                else:
                    assist_members.append(member)

            if lead_member is None:
                lead_member = assist_members[0]
            
            delivery_prompt = f'''You are the lead of the medical group which aims to {self.goal}. You have the following assistant clinicians who work for you:'''
            for a_mem in assist_members:
                delivery_prompt += "\n{}".format(a_mem.role)
            
            delivery_prompt += "\n\nNow, given the medical query, provide a short answer to what kind investigations are needed from each assistant clinicians.\nQuestion: {}".format(self.question)
            try:
                delivery = lead_member.chat(delivery_prompt)
            except:
                delivery = assist_members[0].chat(delivery_prompt)

            investigations = []
            for a_mem in assist_members:
                investigation = a_mem.chat("You are in a medical group where the goal is to {}. Your group lead is asking for the following investigations:\n{}\n\nPlease remind your expertise and return your investigation summary that contains the core information.".format(self.goal, delivery))
                investigations.append([a_mem.role, investigation])
            
            gathered_investigation = ""
            for investigation in investigations:
                gathered_investigation += "[{}]\n{}\n".format(investigation[0], investigation[1])

            if self.examplers is not None:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, after reviewing the following example cases, return your answer to the medical query among the option provided:\n\n{self.examplers}\nQuestion: {self.question}"""
            else:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, return your answer to the medical query among the option provided.\n\nQuestion: {self.question}"""

            response = lead_member.chat(investigation_prompt)

            return response

        elif comm_type == 'external':
            return
```

### parse_hierarchy

```python
def parse_hierarchy(info, emojis):
    """
    info: 一个包含 (expert, hierarchy) 元组的列表，描述专家的名称和他们的上下级关系。
	emojis: 每个 Agent 对应的 emoji，用来让树状图更有趣、可视化更清晰。
	"""
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]
    #这表示团队最顶层有一个协调人（moderator）。

    count = 0
    for expert, hierarchy in info:
        #info 每一项是一个元组，例如：("Dr. Smith - Cardiologist", "Lead > Assistant")
        try:
            expert = expert.split('-')[0].split('.')[1].strip()
        except:
            expert = expert.split('-')[0].strip()
            """
            这段代码非常“防御式”，确保能正确提取专家名字。例子：
			输入："Dr. Smith - Cardiologist"
			结果："Smith"
			如果失败了，就：
			"""
        
        if hierarchy is None:
            hierarchy = 'Independent'
            #判断是否是独立专家
        	#默认没有给层级的就当作“独立专家”。
        
        if 'independent' not in hierarchy.lower():
            #表示该专家有上下级关系。
            #如果有层级关系：添加为某个“parent”的子节点
            parent = hierarchy.split(">")[0].strip()
            child = hierarchy.split(">")[1].strip()

            for agent in agents:
                if agent.name.split("(")[0].strip().lower() == parent.strip().lower():
                    child_agent = Node("{} ({})".format(child, emojis[count]), agent)
                    agents.append(child_agent)
            
                    """
                    找到父节点的名字，比如 "Lead"
                    在已有 agents 中查找名字为 "Lead" 的节点
                    找到之后，把 child 加到它的下面
                    """
        else:
            agent = Node("{} ({})".format(expert, emojis[count]), moderator)
            agents.append(agent)
            #如果是独立专家：直接挂在 moderator 下面
            #这说明这个专家不属于任何上下级，直接挂到 moderator（协调人）下面。

        count += 1

    return agents
	#注意这不是返回一个完整的树结构，而是一个“节点列表”，可以拿去用 pptree.print_tree() 来打印展示。
```

**举个例子**

```python
info = [
    ("Dr. Smith - Cardiologist", "Independent"),
    ("Dr. Lee - Neurologist", "Lead > Assistant"),
    ("Dr. Kim - Assistant", "Lead > Assistant"),
]
```

配合 emojis：

```scss
['🧑‍', '👩‍', '👨‍']
```

就会构造出一棵如下结构的树：

```scss
moderator (👨‍)
├── Smith (🧑‍)
└── Lead (👩‍)
    └── Assistant (👨‍)
```

### parse_group_info

```python
def parse_group_info(group_info):
    lines = group_info.split('\n')
    
    parsed_info = {
        'group_goal': '',
        'members': []
    }

    parsed_info['group_goal'] = "".join(lines[0].split('-')[1:])
    
    
    for line in lines[1:]:
        if line.startswith('Member'):
            member_info = line.split(':')
            member_role_description = member_info[1].split('-')
            
            member_role = member_role_description[0].strip()
            member_expertise = member_role_description[1].strip() if len(member_role_description) > 1 else ''
            
            parsed_info['members'].append({
                'role': member_role,
                'expertise_description': member_expertise
            })
    
    return parsed_info
```

>说明：`group_info` 是一个字符串，表示一组信息，第一行是“小组目标”，后面几行是成员信息，例如：
>
>```python
>group_info = """Group Goal - Diagnose lung cancer
>Member 1: Radiologist - Expert in CT scan interpretation
>Member 2: Oncologist - Specialized in cancer treatment"""
>```
>
>`lines[0]`
>
>取出第一行，也就是：
>
>```python
>"Group Goal - Diagnose lung cancer"
>```
>
>`.split('-')`
>
>按 `-` 拆分成列表：
>
>```python
>['Group Goal ', ' Diagnose lung cancer']
>```
>
> `[1:]`
>
>取出从第1个（索引为1）开始的所有元素（不包括第0个）：
>
>```python
>[' Diagnose lung cancer']
>```
>
> `"".join(...)`
>
>把这个列表里的字符串全部拼接起来（这里只有一个元素，拼接后还是一样）：
>
>```python
>' Diagnose lung cancer'
>```
>
>最终结果就是：
>
>```python
>parsed_info['group_goal'] = ' Diagnose lung cancer'
>```
>
>你也可以加 `.strip()` 去除空格，更干净：
>
>```python
>parsed_info['group_goal'] = "".join(lines[0].split('-')[1:]).strip()
>```

**把一个字符串形式的 group 信息解析成结构化的 Python 字典，方便后续使用。**

```python
#输入：
group_info = """Goal - Diagnose cardiovascular disease
Member1: Cardiologist - Expert in heart and blood vessels
Member2: Radiologist - Expert in imaging"""
#输出:

{
  'group_goal': 'Diagnose cardiovascular disease',
  'members': [
    {'role': 'Cardiologist', 'expertise_description': 'Expert in heart and blood vessels'},
    {'role': 'Radiologist', 'expertise_description': 'Expert in imaging'}
  ]
}
```

按行拆分字符串 → `lines = group_info.split('\n')`

第一行取出任务目标（`Goal - ...`） → 存入 `group_goal`

其余行判断是否为成员信息，按冒号和横线切分角色与专业。

### setup_model

```python
def setup_model(model_name):
    if 'gemini' in model_name:
        genai.configure(api_key=os.environ['genai_api_key'])
        return genai, None
    elif 'gpt' in model_name:
        client = OpenAI(api_key=os.environ['openai_api_key'])
        return None, client
    else:
        raise ValueError(f"Unsupported model: {model_name}")

```

根据模型名（如 `gpt-4o-mini`、`gemini-pro`）配置相应的 API 客户端，并返回。

输入：模型名称（字符串）

输出：

* 如果是 `gemini`，返回 `genai`（Gemini 的模块）、`None`
* 如果是 `gpt` 系列，返回 `None`、`OpenAI client`
* 否则抛出异常

这是一个**模型工厂函数**，根据模型名字来做 API 初始化。后续调用时只需用返回的对象进行调用即可。

### load_data

```python
def load_data(dataset):
    test_qa = []
    examplers = []

    test_path = f'../data/{dataset}/test.jsonl'
    with open(test_path, 'r') as file:
        for line in file:
            test_qa.append(json.loads(line))

    train_path = f'../data/{dataset}/train.jsonl'
    with open(train_path, 'r') as file:
        for line in file:
            examplers.append(json.loads(line))

    return test_qa, examplers
```

读取指定数据集的训练样本和测试样本（`.jsonl` 格式），并解析成 Python 对象。

**输入：**

数据集名称（例如 `"medqa"`）

**输出：**

- `test_qa`: 测试样本列表
- `examplers`: 训练样本列表（作为示例题）

按行读取 `.jsonl`（每行一个 JSON 对象），用 `json.loads` 转换为 Python 字典，便于使用。

### create_question

```python
def create_question(sample, dataset):
    if dataset == 'medqa':
        #根据不同数据集（主要是 medqa）的格式，把题目和选项拼接成一条完整的问题字符串，供模型输入。
        """
        {
    		"question": "What is the first-line treatment for hypertension?",
    		"options": {
        		"A": "ACE inhibitors",
        		"B": "Beta blockers",
        		"C": "Calcium channel blockers",
        		"D": "Diuretics"
    		}
		}"""
        question = sample['question'] + " Options: "
        options = []
        for k, v in sample['options'].items():
            options.append("({}) {}".format(k, v))
        random.shuffle(options)
        
        question += " ".join(options)
        #将处理好的选项用空格拼接，加到题干后面。
        #"What is the first-line treatment for hypertension? Options: (C) Calcium channel blockers (A) ACE inhibitors (B) Beta blockers (D) Diuretics"

        
        return question, None
    	"""
    	返回构造好的字符串；
		第二个返回值 None 是为未来扩展图像输入预留的（比如图文题目时加图像路径）。
		"""
    
    return sample['question'], None 
	#对于不是 medqa 的数据集，不处理选项，只返回题干（原始 question）。
```

将原始样本转成一个拼接了选项的标准问题字符串。

**输入：**

- `sample`: 单个样本（字典）
- `dataset`: 数据集名（目前只处理 `'medqa'`）

**输出：**

- `question`: 拼接后的问题字符串
- `img_path`: 始终为 `None`（为将来支持图像预留）

**逻辑解析：**

把 `"question"` + `"options"` 转成一个自然语言形式的问题，格式像这样：

```
What is the likely diagnosis? Options: (A) X (B) Y (C) Z ...
```

### determine_difficulty

```python
def determine_difficulty(question, difficulty):
    if difficulty != 'adaptive':
        return difficulty
    
    difficulty_prompt = f"""Now, given the medical query as below, you need to decide the difficulty/complexity of it:\n{question}.\n\nPlease indicate the difficulty/complexity of the medical query among below options:\n1) basic: a single medical agent can output an answer.\n2) intermediate: number of medical experts with different expertise should dicuss and make final decision.\n3) advanced: multiple teams of clinicians from different departments need to collaborate with each other to make final decision."""
    
    medical_agent = Agent(instruction='You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.', role='medical expert', model_info='gpt-3.5')
    medical_agent.chat('You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.')
    response = medical_agent.chat(difficulty_prompt)

    if 'basic' in response.lower() or '1)' in response.lower():
        return 'basic'
    elif 'intermediate' in response.lower() or '2)' in response.lower():
        return 'intermediate'
    elif 'advanced' in response.lower() or '3)' in response.lower():
        return 'advanced'
```

根据题目内容和用户输入的 `--difficulty` 参数，**自动或手动判断题目难度等级**。

**输入：**

- `question`: 问题字符串
- `difficulty`: 指定难度或 `"adaptive"`（自适应）

**输出：**

难度字符串：`'basic'`, `'intermediate'`, `'advanced'`

**逻辑解析：**

如果不是 `"adaptive"`，直接返回；

如果是 `"adaptive"`：

- 构造提示 prompt，问一个 LLM 代理（GPT-3.5）来判断难度；
- 根据返回文本中是否包含 `basic`/`intermediate`/`advanced` 等关键词来判断等级。

























































































































