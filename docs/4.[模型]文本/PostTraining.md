# 1. 大模型后训练技术

## 1.1. 学习资料

- [huggingface-llm-course](https://huggingface.co/learn/llm-course/chapter11/1?fw=pt) HuggingFace的LLM课程，主要看了11章对齐和12章推理模型。
- [huggingface-smol-course](https://github.com/huggingface/smol-course) HuggingFace的SMOL课程，用小模型学习对齐技术
- [工业界主流大语言模型后训练(Post-Training)技术总结](https://zhuanlan.zhihu.com/p/987052830)
- [从零开始训练大模型](https://agijuejin.feishu.cn/wiki/VhqZwf34riSekcksULFcx6K3nDg)


## 1.2. 开源工具

- [TRL](https://github.com/huggingface/trl) HuggingFace RLHF工具库
- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples) 微软RLHF实现
- [verl](https://github.com/volcengine/verl) 火山引擎RLHF实现
- [open-r1](https://github.com/huggingface/open-r1)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)

## 1.3. 研究机构


- [Sea AI Lab](https://sail.sea.com/) sea的ai实验室，新加坡
    - [github:sail-sg](https://github.com/sail-sg)

## 1.4. 核心模块

### 1.4.1. 对齐算法

#### 1.4.1.1. SFT

- [2023.08] [Aligning Language Models with Offline Learning from Human Feedback](https://arxiv.org/abs/2308.12050) conditional-sft 不同的样本有不同的权重


#### 1.4.1.2. PEFT
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

#### 1.4.1.3. DPO

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

#### 1.4.1.4. RL

- [2025.04] [Seed1.5-Thinking: Advancing Superb Reasoning Models with Reinforcement Learning](https://arxiv.org/abs/2504.13914)
- [2025.04] [A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce](https://arxiv.org/abs/2504.11343)  RAFT++，
    - [GRPO=高级版拒绝采样？强化学习祛魅时刻：负样本“去芜存菁”才是关键！](https://zhuanlan.zhihu.com/p/1909203956380460977) 
- [2025.04] [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/html/2504.05118v1) VAPO,seed,加上value function。
- [2025.03] [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) DAPO,seed,调高clip上界，动态采样去掉reward为1的prompt，soft超长惩罚，去掉kl
- [2025.03] [What's Behind PPO's Collapse in Long-CoT? Value Optimization Holds the Secret](https://arxiv.org/abs/2503.01491) VC-PPO, 字节seed。long-cot的问题在于value估计不准，靠后的token的V大，A小。V做预训练，用lamada=1。Policy学习的时候对应的A用lamada=0.95减少方差，因为A不会因为V的引入bias。
- [2025.03] [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783) Dr.GRPO 
    - [sail-sg/understand-r1-zero](https://github.com/sail-sg/understand-r1-zero)
- [2025.02] [Process Reinforcement through Implicit Rewards](https://arxiv.org/abs/2502.01456) PRIME
    - [【论文解读】PRIME：通过「隐式过程奖励」来提升LLM的推理能力](https://zhuanlan.zhihu.com/p/18181925892)
- [2025.01] [REINFORCE++: An Efficient RLHF Algorithm with Robustness to Both Prompt and Reward Models](https://arxiv.org/abs/2501.03262) 在batch内归一化，用kl散度作为正则项。
    - [2025.02] [Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning](https://arxiv.org/abs/2502.14768) 
    - [Logic-RL：基于规则强化学习的大语言模型推理能力突破](https://zhuanlan.zhihu.com/p/26480918542)
- [2024.07] [ReaL: Efficient RLHF Training of Large Language Models with Parameter Reallocation](https://arxiv.org/abs/2406.14088) 
- [2024.04] [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300)
- [2024.02] [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/pdf/2402.14740) RLOO,避免使用value model和GAE，减少显存占用，留一法做归一化
- [2024.01] [ReFT: Reasoning with Reinforced Fine-Tuning](https://arxiv.org/abs/2401.08967) REFT。使用跟SFT一样的数据，只是会采样更多cot，然后用PPO优化。
- [2023.10] [ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models](https://arxiv.org/abs/2310.10505) Remax, 用贪婪采样作为base。
- [2023.07] [Secrets of RLHF in Large Language Models Part I: PPO](https://arxiv.org/abs/2307.04964) PPO-max
    - [PPO探索（如何实现稳定训练）](https://zhuanlan.zhihu.com/p/687850058)
- [2023.05] [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [2023.03] [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) 用few-shot得到feedback，然后优化回答。
- [2022.12] [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [2022.11] [Solving math word problems with process- and outcome-based feedback](https://arxiv.org/abs/2211.14275)

- [2022.04] [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)
- [2018.06] [Self-Imitation Learning](https://arxiv.org/abs/1806.05635)
- [2017.07] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

#### 1.4.1.5. 推理和工具

- [2022.11] [PAL: Program-aided Language Models]([2211.10435](https://arxiv.org/abs/2211.10435)) PAL
- [2022.10] [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)  ReAct，Google，query、think、action、result。

#### 1.4.1.6. 蒸馏

- [2023.06] [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649) GKD解决训推不一致问题。


### 1.4.2. Reward Model

- [2025.04] [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495) / [【论文解读】](https://zhuanlan.zhihu.com/p/1892290985284855414)  SPCT 让模型自己生成原则，然后生成打分。
- [2024.10] [Generative Reward Models](https://arxiv.org/abs/2410.12832) 斯坦福，合成实验室，CoT-GenRM，先通过prompt让模型给Q生成推理和A标记出正确的，或给定Q和A生成推理链。用得到的推理数据做sft和dpo。在难得推理任务上表现更好。
- [2024.08] [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/abs/2408.15240) 谷歌，deepmind，Cot-GenRM，只做了sft
- [2024.06] [HelpSteer2: Open-source dataset for training top-performing reward models](https://arxiv.org/abs/2406.08673) 
    - [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2)
    - [NVIDIA/NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner)
- [2024.03] [RewardBench: Evaluating Reward Models for Language Modeling](https://arxiv.org/abs/2403.13787)
    - [allenai/reward-bench](https://github.com/allenai/reward-bench)
- [2024.01] [Secrets of RLHF in Large Language Models Part II: Reward Modeling](https://arxiv.org/abs/2401.06080) 
    - [复旦大学邱锡鹏老师文章解读：Secrets of RLHF in Large Language Models Part II: Reward Modeling](https://zhuanlan.zhihu.com/p/705168755)
- [2023.06] [PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization](https://arxiv.org/abs/2306.05087)
    - [WeOpenML/PandaLM](https://github.com/WeOpenML/PandaLM)


## 1.5. 细分方向

### 1.5.1. 形式化证明

- [2025.04] [Kimina-Prover Preview: Towards Large Formal Reasoning Models with Reinforcement Learning](https://github.com/MoonshotAI/Kimina-Prover-Preview)  MiniF2F-test 80+%
- [2025.04] [Leanabell-Prover: Posttraining Scaling in Formal Reasoning](https://arxiv.org/pdf/2504.06122) MiniF2F-test 59.8%
- [2024.08] [DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search](https://arxiv.org/abs/2408.08152)  miniF2F-test达到63.5%
    - [github:DeepSeek-Prover-V1.5](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5)


### 1.5.2. 角色扮演

- [2023.03] [Rewarding Chatbots for Real-World Engagement with Millions of Users](https://arxiv.org/pdf/2303.06135) Chai的论文，用RLHF优化Chatbot



## 1.6. 理解对齐

- [LLM 能从单个例子中学习吗？](https://www.fast.ai/posts/2023-09-04-learning-jumps/)


