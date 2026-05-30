# Context Engineering

- [2025年中] Andrej Karpathy提出Context Engineering比Prompt Engineering更重要。核心问题从"该怎么问"变为"该让模型看到什么"
- [2025.06] Shopify CEO Tobias Lütke推特首发，Karpathy "+1"支持
- [2025.10] [Context Engineering 2.0: The Context of Context Engineering](https://arxiv.org/abs/2510.26493) 复旦大学等，调研1400+篇论文，系统化定义上下文工程
- [2025.09] [Context Engineering for Trustworthiness: Rescorla Wagner Steering Under Mixed and Inappropriate Contexts](https://arxiv.org/abs/2509.04500)
- IBM苏黎世实验：向GPT-4.1注入结构化认知工具后，AIME2024数学竞赛准确率从26.7%跃升至43.3%，提升61.4%

## 概念演进

### Prompt → Context → Harness Engineering

- [2026.02] 继Context Engineering之后的新范式。核心问题从"该让模型看到什么"到"系统该阻止什么、度量什么、修复什么"，关注Agent在生产环境中的约束与治理
  - Prompt Engineering → Context Engineering → Harness Engineering 三层递进
  - 参见 [CSDN博文](https://blog.csdn.net/qhvssonic/article/details/159475751)
- [2026.04] [SemaClaw: A Step Towards General-Purpose Personal AI Agents through Harness Engineering](https://arxiv.org/abs/2604.11548) 开源多agent框架，正式提出从prompt/context engineering到harness engineering的范式迁移。DAG两阶段混合编排 + PermissionBridge行为安全系统 + 三层上下文管理架构
- [2026.03] [Herding CATs: ALARA for Agent Harness Engineering in Portable Composable Multi-Agent Teams](https://arxiv.org/abs/2603.20380) 提出ALARA原则用于agent harness工程，解决便携可组合多agent团队的控制问题

### Umwelt Engineering

- [2026.03] [Umwelt Engineering: Designing the Cognitive Worlds of Linguistic Agents](https://arxiv.org/abs/2603.27626) 在prompt和context engineering之上的第三层——设计语言智能体的"认知世界"。通过改变推理所用的语言媒介（如消除"to be"动词的E-Prime约束），从语言层面重塑认知结构。实验表明No-Have约束改善伦理推理19.1pp

### Coordination Engineering

- [2026.05] [Swarm Skills: A Portable, Self-Evolving Multi-Agent System Specification for Coordination Engineering](https://arxiv.org/abs/2605.10052) 从单agent的Prompt/Context Engineering走向多agent的Coordination Engineering，关注如何系统化改进多agent间的协作编码与协同

## 理论

- [2026.03] [The Root Theorem of Context Engineering](https://arxiv.org/abs/2604.20874) 将CE形式化为信息论学科：最大化有界有损信道内的信噪比(maximize signal-to-token ratio within bounded, lossy channels)。推导5个无需额外假设的推论，证明append-only系统必然在有限时间内超出有效窗口。60+session持久架构工程验证。Shannon解决点对点传输，CE解决连续性
- [2025.12] [Monadic Context Engineering](https://arxiv.org/abs/2512.22431) 姚期智等。用Functor/Applicative/Monad代数结构解决CE中状态管理、错误处理、并发的脆弱性问题，将CE从命令式ad-hoc模式提升为声明式架构
- [2026.01] [Entropic Context Shaping: Information-Theoretic Filtering for Context-Aware LLM Agents](https://arxiv.org/abs/2601.11585) 信息论框架，通过答案分布偏移衡量上下文效用，区分有用信息与误导性干扰

## 框架

- [2025.10] [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://www.arxiv.org/abs/2510.04618) ACE框架，将上下文视为动态演化空间，管理离散结构化元素(策略、代码片段、错误处理器)，超越GEPA
- [2025.12] [PAACE: Plan-Aware Automated Context Engineering](https://arxiv.org/abs/2512.16970) 针对多步plan-aware推理的自动CE框架，通过next-k-task相关性建模和plan结构分析优化agent上下文
- [2026.01] [Meta Context Engineering via Agentic Skill Evolution](https://arxiv.org/abs/2601.21557) 将CE本身视为可演化的meta-level过程，通过agentic skill evolution自动改进上下文。46页长文
- [2026.04] [Context Engineering: A Practitioner Methodology for Structured Human-AI Collaboration](https://arxiv.org/abs/2604.04258) 39页实践方法论，提出context completeness比prompt技巧更关联输出质量，含开放数据集
- [2026.03] [Context Engineering: From Prompts to Corporate Multi-Agent Architecture](https://arxiv.org/abs/2603.09619) 从单体chatbot到多步agent，提出CE作为独立学科，设计/结构化/管理整个信息环境
- [2026.05] [A Language for Describing Agentic LLM Contexts (ACD)](https://arxiv.org/abs/2605.01920) 描述LLM agent上下文组成的标准语言ACD，解决CE缺乏标准化描述的问题。CAIS'26，项目页 www.acdlang.org
- [2026.05] [Contexting as Recommendation: Evolutionary Collaborative Filtering for Context Engineering](https://arxiv.org/abs/2605.15721) 将CE建模为推荐系统问题，用进化协同过滤自动选择最优上下文
- [2026.04] [HYVE: Hybrid Views for LLM Context Engineering over Machine Data](https://arxiv.org/abs/2604.05400) 数据库管理原理启发的CE框架，处理机器数据负载的长/深/重复结构输入
- [2026.04] [CLEAR: Context Augmentation from Contrastive Learning of Experience via Agentic Reflection](https://arxiv.org/abs/2604.07487) 对比学习+反思的上下文增强，超越纯检索历史经验的CE方法
- [2026.02] [Structured Context Engineering for File-Native Agentic Systems](https://arxiv.org/abs/2602.05447) 9649次实验系统比较YAML/Markdown/JSON/TOON四种格式的CE效果
- [2026.03] [PRECEPT: Planning Resilience via Experience, Context Engineering & Probing Trajectories](https://arxiv.org/abs/2603.09641) 统一框架：确定性规则检索 + 组合规则学习 + Pareto引导prompt进化
- [2026.01] [CEDAR: Context Engineering for Agentic Data Science](https://arxiv.org/abs/2601.06606) 数据科学场景CE，DS-specific输入字段结构化 + 上下文工程策略减少幻觉。ECIR 2026
- [2026.02] [CL4SE: Benchmarking Context Learning on Software Engineering](https://arxiv.org/abs/2602.23047) 软件工程CE基准，系统分类SE-specific上下文类型
- [2026.03] [A Context Engineering Framework for Improving Enterprise AI Agents based on Digital-Twin MDP](https://arxiv.org/abs/2603.22083) 轻量级模型无关框架，通过离线RL改进企业级LLM agent

## RAG

- [2025.11] [Principled Context Engineering for RAG: Statistical Guarantees via Conformal Prediction](https://arxiv.org/abs/2511.17908) 用Conformal Prediction为RAG中的CE提供统计保证。ECIR 2026
- [2026.04] [ROZA Graphs: Self-Improving Near-Deterministic RAG through Evidence-Centric Feedback](https://arxiv.org/abs/2604.07595) 冻结基础模型，所有增益来自图遍历的上下文工程
- [2025.06] [Refract ICL: Rethinking Example Selection in the Era of Million-Token Models](https://arxiv.org/abs/2506.12346)

## 安全

- [2025.12] [Invasive Context Engineering to Control Large Language Models](https://arxiv.org/abs/2512.03001) 通过向LLM上下文注入"控制句子"控制长上下文场景下的行为
- [2026.01] [Lost in the Noise: How Reasoning Models Fail with Contextual Distractors](https://arxiv.org/abs/2601.07226) 推理模型面对上下文干扰时的失效分析，prompt/CE/SFT/RL均无法保证鲁棒性，提出Rationale-Aware Reward

## 编码场景

- [2026.05] [Mise en Place for Agentic Coding: Deliberate Preparation as Context Engineering Methodology](https://arxiv.org/abs/2605.05400) 对"vibe coding"的反思，提出有准备的CE方法论。VibeX 2026
- [2026.04] [Tokalator: A Context Engineering Toolkit for AI Coding Assistants](https://arxiv.org/abs/2604.08290) CE工具包，VS Code扩展+实时token预算监控+11个slash命令
- [2026.03] [Building Effective AI Coding Agents for the Terminal](https://arxiv.org/abs/2603.05344) OPENDEV开源CLI编码agent，含Scaffolding/Harness/CE实践总结
- [2025.12] [Everything is Context: Agentic File System Abstraction for Context Engineering](https://arxiv.org/abs/2512.05470) 将文件系统抽象为CE的载体

## 开源

- [74.8k] [dair-ai/Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide) Prompt/Context Engineering + RAG + AI Agents综合指南
- [63.2k] [gsd-build/get-shit-done](https://github.com/gsd-build/get-shit-done) Claude Code的meta-prompting + context engineering + spec驱动开发系统
- [16k] [muratcankoylan/Agent-Skills-for-Context-Engineering](https://github.com/muratcankoylan/Agent-Skills-for-Context-Engineering) CE agent skills集合，含多agent架构和生产级agent系统
- [13.4k] [coleam00/context-engineering-intro](https://github.com/coleam00/context-engineering-intro) CE入门+Claude Code实践
- [9k] [davidkimai/Context-Engineering](https://github.com/davidkimai/Context-Engineering) Karpathy引用的CE指南
- [5.3k] [MineContext](https://github.com/MineContext/MineContext) Context-Engineering + ChatGPT Pulse
- [3.1k] [Meirtz/Awesome-Context-Engineering](https://github.com/Meirtz/Awesome-Context-Engineering) CE领域awesome list，含数百篇论文和框架
- [2.4k] [Windy3f3f3f3f/how-claude-code-works](https://github.com/Windy3f3f3f3f/how-claude-code-works) Claude Code源码深度解析（架构、agent循环、上下文工程）

---

# Prompt Engineering

## 开源

- [27.2k] [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) 优化prompt和模型权重
- [12.8k] [linshenkx/prompt-optimizer](https://github.com/linshenkx/prompt-optimizer) 工程助手
    - [2025-04-06｜Prompt-Optimizer: AI 提示词优化神器全攻略](https://zhuanlan.zhihu.com/p/1892351710292332883)
- [2.7k] [Eladlev/AutoPrompt](https://github.com/Eladlev/AutoPrompt)
- [458] [AIDotNet/auto-prompt](https://github.com/AIDotNet/auto-prompt)
- [250] [auto-openai-prompter](https://github.com/hwchase17/auto-openai-prompter)
    - [Langchain创始人新项目Auto-Prompt Builder一键优化你的Prompt，再也不担心写不好Prompt了](https://developer.volcengine.com/articles/7382253806995636233)
- [42] [doganarif/promptpilot](https://github.com/doganarif/promptpilot) 主要是工程助手，没有prompt优化算法。支持prompt版本控制，ab测试。用各种model+prompt的组合来提供更新prompt。
- [Airmomo/SPO](https://github.com/Airmomo/SPO) Self-Supervised Prompt Optimization，不依赖人工标注，通过对比不同提示生成结果进行自监督优化

## 产品

- [autoprompt](https://www.autoprompt.cc/)
- [阿里云百炼-prompt自动优化](https://help.aliyun.com/zh/model-studio/prompt-feedback-optimization)

## 方法

- [anthropic:prompt-improver](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prompt-improver)
- [Automatic Prompt Engineer (APE)](https://www.promptingguide.ai/techniques/ape)
- [Datawhale：Prompt工程](https://datawhaler.feishu.cn/wiki/MgWgw5Zpfie8h9kfx4OcUUiYnJc)

## 论文

### 综述

- [2025.10] [Context Engineering 2.0: The Context of Context Engineering](https://arxiv.org/abs/2510.26493) 复旦大学等，调研1400+篇论文
- [2025.02] [A Survey of Automatic Prompt Engineering: An Optimization Perspective](https://arxiv.org/abs/2502.11560) 系统梳理FM优化、进化方法、梯度优化、RL四大类方法
- [2025.12] [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564) 大规模综述，明确区分agent memory与RAG、CE的概念边界，从forms/functions/dynamics三维度分析

### Prompt优化

- [2025.07] [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457) UC伯克利&斯坦福。遗传式提示词进化+自然语言反思+帕累托候选选择。比GRPO高10%-20%，rollout减少35倍。比MIPROv2(DSPy优化器)高10%+
- [2025.03] [DLPO: Towards a Robust, Efficient, and Generalizable Prompt Optimization Framework from a Deep-Learning Perspective](https://arxiv.org/abs/2503.13413) 从深度学习角度提出鲁棒、高效、可泛化的Prompt优化框架
- [2024.10] [StablePrompt: Automatic Prompt Tuning using Reinforcement Learning for Large Language Models](https://arxiv.org/abs/2410.07652) 韩国科技院KAIST。achor model更新更慢，且只有在agent model提升超过阈值的时候才会更新。
- [2024.04] [Automatic Prompt Selection for Large Language Models](https://arxiv.org/abs/2404.02717) Deakin澳大利亚迪肯大学。对样本聚类每类一个prompt，新问题用prompt评估器挑选最适合的prompt。
