# VLM
| 名称                            | 简写     | 解释                                                         |
| ------------------------------- | -------- | ------------------------------------------------------------ |
| Large Language Model            | LLM      | 单模态语言模型，如 GPT、LLaMA                                |
| Vision-Language Model           | VLM      | 图文模型，如 CLIP、BLIP                                      |
| Multimodal Large Language Model | **MLLM** | 能处理图像、文本（甚至音频、视频）的语言模型，如 GPT-4V、Gemini、MiniGPT-4 |

**LLM：Large Language Model（大语言模型）**

- **定义**：仅处理**文本**的语言模型，通常基于 Transformer 架构，通过大规模文本数据预训练而成。
- **能力**：文本生成、问答、摘要、翻译、推理、代码生成等。
- **特点**：
  - 输入输出都是**文本**
  - 单模态（language only）
- **代表模型**：
  - GPT 系列（GPT-3.5, GPT-4）
  - LLaMA、Claude、PaLM、Qwen 等
- **应用场景**：
  - ChatGPT、智能客服、自动写作、编程助手等

**VLM：Vision-Language Model（视觉语言模型）**

- **定义**：能同时处理**图像和文本**的多模态模型，理解或生成图文内容。
- **能力**： 
  - 图文检索（Text→Image / Image→Text）
  - 图像描述生成（Image Captioning）
  - VQA（Visual Question Answering）
  - 多模态推理（如 ScienceQA）
- **结构组成**：
  - 图像编码器（如 ViT、ResNet）
  - 文本编码器（如 BERT、LLM）
  - 融合机制（如 cross-attention、Q-Former）
- **代表模型**：
  - CLIP、BLIP/BLIP-2、GIT、Flamingo、MiniGPT-4、Kosmos-1、GPT-4V
- **应用场景**：
  - 多模态助手（例如 GPT-4V、Gemini）
  - 图像搜索、无障碍图像描述
  - 医学图像辅助分析（图文匹配）

## ViT

**ViT（Vision Transformer）** 属于：

**单模态视觉模型（视觉方向）**

| 名称    | 类型                     | 描述                                                     |
| ------- | ------------------------ | -------------------------------------------------------- |
| **ViT** | Vision Model（视觉模型） | 只处理图像的模型，基于 Transformer 架构替代传统 CNN 结构 |

### 核心思想：

- 把图像切成固定大小的 **patches（如 16x16）**
- 每个 patch 被当作一个“词”，与 NLP 中 token 类似
- 然后喂入标准的 **Transformer Encoder** 结构进行处理
- 不再使用卷积（CNN）

### ViT 属于哪一类？

| 分类                                      | ViT 属于                          |
| ----------------------------------------- | --------------------------------- |
| 👁️ Vision Model（视觉模型）                | ✅ 是                              |
| 🗣️ Language Model（语言模型）              | ❌ 否                              |
| 🤖 Multi-modal Model（多模态模型，如 VLM） | ❌ 否（但常作为 VLM 的视觉编码器） |

### ViT 与 VLM 的关系：

虽然 ViT 自身是单模态的，但它**经常被作为多模态模型（VLM）中的图像编码器**。

例如：

- **CLIP** = ViT + Text Encoder + 对比对齐（VLA）
- **BLIP-2**：用 ViT 提取视觉特征后与语言模型对接
- **MiniGPT-4**：ViT → Q-Former → LLM

## VLM

### 引言

- **背景：**预训练大语言模型（LLM）虽强大，但面临高质量文本数据有限和单模态无法完全理解现实世界（需要多模态信息）的挑战。
- **VLM 的价值：** VLM 通过结合视觉（图像、视频）和文本输入，能更全面地理解视觉空间关系、物体、场景和抽象概念，扩展了模型的表示能力（例如，视觉问答 VQA、自动驾驶）。
- **VLM 的新挑战：** 区别于单模态模型，VLM 存在特有挑战，如**视觉幻觉**（Visual Hallucination），即模型在没有真正理解视觉内容的情况下，仅依赖 LLM 组件的参数知识生成响应。
- **本文目标：** 对 VLM 的主要架构、评估与基准、面临的挑战进行系统性回顾。

### 架构演变 (核心趋势)：

- - **早期：** 从零开始训练模型（如 CLIP）。
  - **近期 (主流)：** 利用**预训练的 LLM 作为骨干 (Backbone)**，将视觉信息对齐到 LLM 的表示空间，以充分利用 LLM 强大的语言理解和生成能力。**Table 1**清晰展示了这一转变，列出了从 2021 年到 2025 年（预测）的代表性 VLM，如 CLIP (从零训练) vs LLaVA, GPT-4V, Gemini, LLaMA 3.2-vision 等 (使用 LLM Backbone)。

### VLM 训练与对齐方法
