# 1. 大模型后训练技术

## 1.1. 学习资料

- [huggingface-llm-course](https://huggingface.co/learn/llm-course/chapter11/1?fw=pt) HuggingFace的LLM课程，主要看了11章对齐和12章推理模型。
- [huggingface-smol-course](https://github.com/huggingface/smol-course) HuggingFace的SMOL课程，用小模型学习对齐技术

## 1.2. 开源工具

- [TRL](https://github.com/huggingface/trl) HuggingFace RLHF工具库
- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples) 微软RLHF实现
- [verl](https://github.com/volcengine/verl) 火山引擎RLHF实现
- [open-r1](https://github.com/huggingface/open-r1)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)


## 1.3. 核心模块

### 1.3.1. 对齐算法

#### 1.3.1.1. SFT



#### 1.3.1.2. PEFT
相关资料：

- [huggingface:peft](https://huggingface.co/docs/peft/index)
- [大模型参数高效微调(PEFT)](https://zhuanlan.zhihu.com/p/621700272)

算法：

- LORA
- QLORA
- Adapter
- Prefix Tuning
- Prompt Tuning
- BitFit

#### 1.3.1.3. DPO

- [2024.05] [Exploratory Preference Optimization: Harnessing Implicit Q*-Approximation for Sample-Efficient RLHF](https://arxiv.org/abs/2405.21046) XPO在Online DPO基础上在loss上加了鼓励探索的正则项。
- [2024.04] [Binary Classifier Optimization for Large Language Model Alignment](https://arxiv.org/abs/2404.04656) BCO用BCE。奖励转移、底层分布匹配。
- [2024.03] [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) DPO基础上去掉reference model。
- [2024.02] [Direct Language Model Alignment from Online AI Feedback](https://arxiv.org/abs/2402.04792) Online DPO​ 结合在线数据更新，动态调整偏好数据集，缓解分布偏移。用LLM+Prompt实时对样本打标得到对比对。
- [2024.02] [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) 基于前景理论，直接优化人类感知效用，替代传统偏好对数似然
    - [ContextualAI/HALOs](https://github.com/ContextualAI/HALOs)
- [2024.01] [Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation](https://arxiv.org/abs/2401.08417)  CPO是负对数似然损失+偏好损失。
- [2023.12] [Nash Learning from Human Feedback](https://arxiv.org/abs/2312.00886) 在act和ref的模型上分别得到logit然后加权求和得到额外策略。
- [2023.10] [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036) IPO相当于 在DPO的损失函数上添加了一个正则项
- [2023.05] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) / [【笔记】](https://www.cnblogs.com/lemonzhang/p/17910358.html) 通过偏好数据直接优化策略，绕过显式奖励建模
    - [DPO为什么会让大语言模型输出变长](https://zhuanlan.zhihu.com/p/5830338806)
    - [大模型对齐方法笔记一：DPO及其变种IPO、KTO、CPO](https://blog.csdn.net/beingstrong/article/details/138973997)
    - [一文看尽LLM对齐技术：RLHF、RLAIF、PPO、DPO……](https://zhuanlan.zhihu.com/p/712819706)

#### 1.3.1.4. RL

- [2025.04] [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/html/2504.05118v1) 加上value function。
- [2025.03] [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) 调高clip上界，动态采样去掉reward为1的prompt，soft超长惩罚，去掉kl
- [2024.04] [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300)
- [2024.02] [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/pdf/2402.14740) 避免使用value model和GAE，减少显存占用
- [2023.05] [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [2022.12] [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [2022.11] [Solving math word problems with process- and outcome-based feedback](https://arxiv.org/abs/2211.14275)
- [2022.04] [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)
- [2017.07] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)


#### 1.3.1.5. 蒸馏

- [2023.06] [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649) GKD解决训推不一致问题。


### 1.3.2. Reward Model

- [2025.04] [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495) / [【论文解读】](https://zhuanlan.zhihu.com/p/1892290985284855414)  SPCT 让模型自己生成原则，然后生成打分。

## 1.4. 细分方向

### 1.4.1. 形式化证明

- [2025.04] [Kimina-Prover Preview: Towards Large Formal Reasoning Models with Reinforcement Learning](https://github.com/MoonshotAI/Kimina-Prover-Preview)  MiniF2F-test 80+%
- [2025.04] [Leanabell-Prover: Posttraining Scaling in Formal Reasoning](https://arxiv.org/pdf/2504.06122) MiniF2F-test 59.8%
- [2024.08] [DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search](https://arxiv.org/abs/2408.08152)  miniF2F-test达到63.5%
    - [github:DeepSeek-Prover-V1.5](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5)


### 1.4.2. 角色扮演

- [2023.03] [Rewarding Chatbots for Real-World Engagement with Millions of Users](https://arxiv.org/pdf/2303.06135) Chai的论文，用RLHF优化Chatbot



## 1.5. 理解对齐

- [LLM 能从单个例子中学习吗？](https://www.fast.ai/posts/2023-09-04-learning-jumps/)


