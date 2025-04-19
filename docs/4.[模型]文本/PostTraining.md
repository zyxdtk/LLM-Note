# 大模型后训练技术

## 学习资料

- [huggingface-llm-course](https://huggingface.co/learn/llm-course/chapter11/1?fw=pt) HuggingFace的LLM课程，主要看了11章对齐和12章推理模型。
- [huggingface-smol-course](https://github.com/huggingface/smol-course) HuggingFace的SMOL课程，用小模型学习对齐技术

## 开源工具

- [TRL](https://github.com/huggingface/trl) HuggingFace RLHF工具库
- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples) 微软RLHF实现
- [verl](https://github.com/volcengine/verl) 火山引擎RLHF实现
- [open-r1](https://github.com/huggingface/open-r1)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)


## 核心模块

### 对齐算法

- SFT
- PEFT
    - LORA
    - QLORA
    - Adapter
    - Prefix Tuning
    - Prompt Tuning
    - BitFit
- RL
    - DPO
        - [2023.05] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) / [【笔记】](https://www.cnblogs.com/lemonzhang/p/17910358.html)
    - PPO
    - GRPO

### Reward Model

- [2025.04] [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495) / [【论文解读】](https://zhuanlan.zhihu.com/p/1892290985284855414)  SPCT 让模型自己生成原则，然后生成打分。

## 细分方向

### 形式化证明

- [2025.04] [Kimina-Prover Preview: Towards Large Formal Reasoning Models with Reinforcement Learning](https://github.com/MoonshotAI/Kimina-Prover-Preview)  MiniF2F-test 80+%
- [2025.04] [Leanabell-Prover: Posttraining Scaling in Formal Reasoning](https://arxiv.org/pdf/2504.06122) MiniF2F-test 59.8%
- [2024.08] [DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search](https://arxiv.org/abs/2408.08152)  miniF2F-test达到63.5%
    - [github:DeepSeek-Prover-V1.5](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5)


### 角色扮演

- [2023.03] [Rewarding Chatbots for Real-World Engagement with Millions of Users](https://arxiv.org/pdf/2303.06135) Chai的论文，用RLHF优化Chatbot



## 理解对齐

- [LLM 能从单个例子中学习吗？](https://www.fast.ai/posts/2023-09-04-learning-jumps/)


