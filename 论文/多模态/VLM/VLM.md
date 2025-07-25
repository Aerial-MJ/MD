# VLM
| 名称                            | 简写     | 解释                                                         |
| ------------------------------- | -------- | ------------------------------------------------------------ |
| Large Language Model            | LLM      | 单模态语言模型，如 GPT、LLaMA                                |
| Vision-Language Model           | VLM      | 图文模型，如 CLIP、BLIP                                      |
| Multimodal Large Language Model | **MLLM** | 能处理图像、文本（甚至音频、视频）的语言模型，如 GPT-4V、Gemini、MiniGPT-4 |

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

