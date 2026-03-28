# 1. Code LLM

## 1.1. 学习资料

- [BigCode Project Documentation](https://www.bigcode-project.org/) BigCode 项目官方文档
- [Meta Code Llama Documentation](https://github.com/facebookresearch/codellama) Meta 官方 Code Llama 文档
- [DeepSeek Coder Documentation](https://github.com/deepseek-ai/DeepSeek-Coder) 深度求索官方文档
- [WizardCoder Documentation](https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder) WizardCoder 官方文档
- [Salesforce CodeT5 Documentation](https://github.com/salesforce/CodeT5) Salesforce 官方文档

## 1.2. 开源模型

### 1.2.1. 主流模型

- [2026.02] [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763) 智谱 AI，旨在将编程范式从 Vibe Coding 过渡到智能体工程
- [2026.02] [CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation](https://arxiv.org/abs/2602.24286) 字节跳动 + 清华，自动生成高性能 CUDA 内核，在 KernelBench 基准测试上取得 SOTA 成绩
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
