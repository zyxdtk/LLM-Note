# 1. Code LLM

## 1.1. 学习资料

- [BigCode Project Documentation](https://www.bigcode-project.org/) BigCode 项目官方文档
- [Meta Code Llama Documentation](https://github.com/facebookresearch/codellama) Meta 官方 Code Llama 文档
- [DeepSeek Coder Documentation](https://github.com/deepseek-ai/DeepSeek-Coder) 深度求索官方文档
- [WizardCoder Documentation](https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder) WizardCoder 官方文档
- [Salesforce CodeT5 Documentation](https://github.com/salesforce/CodeT5) Salesforce 官方文档

## 1.2. 开源模型

### 1.2.1. 主流模型

- [2026.01] [Stable-DiffCoder: Pushing the Frontier of Code Diffusion Large Language Model](https://arxiv.org/abs/2601.15892) 代码扩散模型
- [2024.01] [DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence](https://arxiv.org/abs/2401.14196) deepseek
  - [DeepSeek Coder](https://github.com/deepseek-ai/DeepSeek-Coder) 
- [2023.12] [Magicoder: Empowering Code Generation with OSS-Instruct](https://arxiv.org/abs/2312.02120) 从codebase中提取代码片段，用llm生成code任务，用llm生成solution。生成的prompt因为包含了代码片段，所以更多样。本质还是蒸馏strong2weak
- [2023.08] [Code Llama: Open Foundation Models for Code](https://arxiv.org/abs/2308.12950) meta
  - [codellama](https://github.com/meta-llama/codellama)
- [2022.07] [Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/abs/2207.14255) openai

- [BigCode Project](https://github.com/bigcode-project/starcoder) Hugging Face 和 ServiceNow 联合发起的开源项目，在 80+ 种编程语言上训练
  - **代表作**: StarCoder, StarCoder2

- [2022] [WizardCoder](https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder) 微软研究院支持的项目，使用 Evol-Instruct 方法提升代码生成能力
  - **代表作**: WizardCoder-Python

- [2022] [Salesforce CodeT5](https://github.com/salesforce/CodeT5) Salesforce 开源的代码理解和生成模型系列。已经不更新了
  - **代表作**: CodeT5, CodeT5+

- [2022] [AlphaCode](https://github.com/deepmind/code_contests) DeepMind 开发的编程竞赛级代码生成模型。已经不更新了
  - **机构**: Google DeepMind

## 1.3. 研究机构

### 1.3.1. 国际机构

- [BigCode Project](https://www.bigcode-project.org/) Hugging Face 和 ServiceNow 联合发起的开源项目
- [Google DeepMind](https://deepmind.google/) 开发 AlphaCode 等代码模型
- [Meta AI](https://ai.meta.com/) 开发 Code Llama 系列模型
- [Microsoft Research](https://www.microsoft.com/en-us/research/) 支持 WizardCoder 等项目
- [Salesforce Research](https://blog.salesforceairesearch.com/) 开发 CodeT5 系列模型

### 1.3.2. 国内机构

- [深度求索 (DeepSeek)](https://www.deepseek.com/) 开发 DeepSeek Coder 系列模型
- [智谱 AI](https://www.zhipuai.cn/) 开发代码相关模型
- [阿里云通义](https://tongyi.aliyun.com/) 提供代码生成能力

## 1.4. 论文合集

### 1.4.1. Awesome 系列

- [Awesome Code LLM](https://github.com/codefuse-ai/Awesome-Code-LLM) 蚂蚁集团整理的代码大模型资源
- [LLM Survey](https://github.com/RUCAIBox/LLMSurvey) 人大整理的大模型综述

## 1.5. 评测基准

### 1.5.1. 代码生成评测

- [HumanEval](https://github.com/openai/human-eval) OpenAI 提出的代码生成经典评测集
- [SWE-bench](https://github.com/swe-bench/swe-bench) 真实 GitHub issue 修复任务评测

## 1.6. 核心技术

### 1.6.1. 预训练技术

- 代码语料收集和过滤
- 多语言代码训练
- 代码结构理解

### 1.6.2. 微调技术

- 指令微调
- 偏好对齐
- 多任务学习

### 1.6.3. 推理优化

- 代码补全
- 代码修复
- 代码理解
## 1.7. 🎓 训练指南专题

**完整教程**:
1. [如何训练 SOTA 代码大模型](how-to-train-sota-code-llm.md) - 完整训练流程和技术细节
2. [Code LLM 训练快速开始](code-llm-quickstart.md) - 30 分钟理解核心概念
3. [Code LLM 关键论文阅读清单](code-llm-reading-list.md) - 精选 30 篇核心论文

**核心论文**:
- DeepSeek-Coder (2024) - 预训练最佳实践
- WizardCoder (2023) - Evol-Instruct 指令进化
- Magicoder (2023) - OSS-Instruct 开源指令生成
- StarCoder2 (2024) - 多语言代码预训练
- OpenCoder (2024) - 开源训练手册
