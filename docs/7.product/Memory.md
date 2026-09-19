# Agent 记忆与知识自进化

> 记忆是 Agent 的核心模块：没有记忆的 agent 是无状态的函数调用器，有记忆的 agent 才是持续进化的智能体。知识自进化本质上是"记忆的主动写入+组织+检索+更新"，类似于店铺在日常经营中沉淀 know-how。

---

## 一、综述与概念框架

- [2025.12] [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564) **最新最全综述**，46 位作者。三维框架：Forms（记忆形态：Token级/参数化/潜在）、Functions（记忆功能：事实型/经验型/工作型）、Dynamics（记忆动态：形成/演化/检索）。明确区分 agent memory 与 RAG、Context Engineering 的概念边界。开源 [论文列表](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)
- [2024.04] [大模型智能体记忆机制综述](https://arxiv.org/abs/2404.13501) 经典综述。参数化记忆 vs 上下文记忆。六大操作：巩固/更新/索引/遗忘/检索/压缩
- [2025.05] [From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms](https://arxiv.org/abs/2605.06716) 记忆机制演进综述。从简单存储到经验学习
- [2025.08] [A Survey of Self-Evolving Agents: On Path to Artificial Super Intelligence](https://arxiv.org/abs/2507.21046) 自进化 agent 综述
- [2026.08] [The Horizon Gap: Planning, Memory, Execution, Training, and Evaluation for Long-Horizon LLM Agents](https://arxiv.org/abs/2608.06663) 长程 agent 的规划-记忆-执行-训练-评估全景
- [2026.07] [Dynamic Agent Skills: A Lifecycle Survey and Taxonomy of Evolving Skill Libraries](https://arxiv.org/abs/2607.10113) 技能库生命周期综述：技能的创建、检索、演化、废弃

### 两大流派

| 维度 | 模型驱动派 | 应用驱动派 |
|------|-----------|-----------|
| 代表 | Memorizing Transformers, MemoryLLM, Titans, WISE | MemGPT, Mem0, Zep, MemOS, MIRIX |
| 核心思路 | 改造模型内部结构 | 应用层记忆管理 |
| 优势 | 性能上限高、读取效率高 | 落地快、易扩展、模型无关 |
| 劣势 | 研发成本高、需重新训练 | 依赖底层模型、可能有延迟 |
| 适合 | 算法创新 | 工程落地 |

### 记忆的分类（认知科学映射）

| 记忆类型 | 定义 | Agent 对应 | 店铺类比 |
|---------|------|-----------|---------|
| 短期/工作记忆 | 当前任务的临时上下文 | context window / scratchpad | 正在服务的这桌客人 |
| 情景记忆 (Episodic) | 具体事件的经验 | trajectory 存储、过往对话记录 | 昨天某个客人的特殊要求 |
| 语义记忆 (Semantic) | 抽象知识和事实 | 知识库、用户画像、规则 | "这家店的主打菜是什么" |
| 程序记忆 (Procedural) | 如何做某事的技能 | skill library、代码片段、工作流 | 后厨的标准操作手册 |

---

## 二、记忆架构与开源框架

### 应用驱动：基础工作

- [2023.04] [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) Stanford。记忆流 (Memory Stream) + 反思 (Reflection) + 规划。检索三标准：Recency（新近性）、Importance（重要性）、Relevance（相关性）。**情景记忆+反思机制的奠基工作**
- [2023.05] [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291) Minecraft 终身学习 agent。三大组件：自动课程、技能库、迭代提示。**技能库是程序记忆的典范**：agent 将成功行为序列编码为可复用代码，存入技能库供后续调用
- [2023.10] [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) UC Berkeley。将 LLM 视为操作系统，虚拟内存管理：主上下文（Main Context）+ 外部存储（External Context），LLM 自主决定 swap in/out。分层：System Instructions / Working Context / FIFO。**开源为 [Letta](https://github.com/letta-ai/letta)**
- [2023.05] [MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://arxiv.org/abs/2305.10250) **遗忘曲线先驱**。三层存储：In-Depth Memory（详细对话+时间戳）、Hierarchical Event Summary（层次化摘要）、Dynamic Personality Understanding（用户个性动态评估）。借鉴艾宾浩斯遗忘曲线实现记忆遗忘
- [2024.05] [HippoRAG: Neurobiologically Inspired Long-Term Memory](https://arxiv.org/abs/2405.14831) NeurIPS 2024。模拟人脑海马体：海马索引（Personalized PageRank）+ 新皮层存储（知识图谱）+ 模式分离。多跳问答超越传统 RAG 20%+
- [2024.09] [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2409.09908) **卡片笔记法（Zettelkasten）**。每个记忆是知识卡片：原始内容+时间戳+关键词+标签+上下文+链接集合。通过链接建立记忆网络，LOCOMO 上优于 MemGPT 和 MemoryBank
- [2025.01] [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956) 时序知识图谱记忆层。三层图结构：情境子图（Episodic）→ 语义子图（Semantic）→ 社区子图（Community）。记忆去重+时间边失效机制。DMR 基准上超越 MemGPT
- [2025.04] [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413) **生产级首选**。两种实现：Mem0（向量检索）和 Mem0-G（图增强）。记忆生成=上下文感知（当前问答+最近 M 条+会话 Summary），记忆更新=检索相似 N 条→LLM 判断增/改/删。冲突检测标记无效而非物理删除。LOCOMO 综合 SOTA
- [2025.05] [MemOS: A Memory OS for AI System](https://arxiv.org/abs/2505.22101) 记忆张素（亿元天使轮）。**记忆操作系统**。三类记忆：纯文本（显式长期）、激活（隐式短期 KV-Cache）、参数（隐式长期）。动态转换：纯文本→激活（预生成 KV-Cache 降低 TTFT）、纯文本/激活→参数（蒸馏/LoRA 固化）、参数→纯文本（卸载过时知识）。统一抽象 MemCube，LOCOMO SOTA
- [2025.07] [MIRIX: Multi-Agent Memory System for LLM-Based Agents](https://arxiv.org/abs/2507.07957) **多模块记忆**。借鉴人类记忆分工，6 个记忆组件+Multi-Agent 架构管理。元记忆管理器路由→对应记忆管理器→两阶段检索（粗检索摘要→详细搜索）
- [2024.08] [MemLong: Memory-Augmented Retrieval for Long Context Modeling](https://arxiv.org/abs/2408.16967) 检索记忆模块+可控注意力，超长文本处理

### 模型驱动：改造模型内部结构

- [2022.03] [Memorizing Transformers](https://arxiv.org/abs/2203.08913) Google。**模型驱动开山之作**。首次在 Transformer 中引入外部记忆：每个 token 存入外部记忆库，推理时 KNN 检索相关历史，融合外部记忆和内部注意力
- [2024.02] [MemoryLLM: Towards Self-Updatable Large Language Models](https://arxiv.org/abs/2402.04624) ICML 2024。每层 Transformer 插入可更新的 Memory Tokens：固定参数（预训练）+ Memory Tokens（可读写）。Read/Write Gates 控制更新，遗忘控制对抗灾难性遗忘
- [2024.07] [Memory³: Language Modeling with Explicit Memory](https://arxiv.org/abs/2407.01178) 三层显式记忆：感知记忆（原始输入）→ 短期记忆（工作上下文）→ 长期记忆（知识存储），每层不同编码和检索策略
- [2024.05] [WISE: Rethinking the Knowledge Memory for Lifelong Model Editing](https://arxiv.org/abs/2405.14768) 浙江大学。双参数体系：主记忆（预训练知识冻结）+ 侧记忆（后续编辑可更新）。避免灾难性遗忘。开源 [EasyEdit](https://github.com/zjunlp/EasyEdit)
- [2024.12] [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) Google。**学习遗忘**。三个门控：存储门控（什么值得记）、遗忘门控（何时清理）、检索门控（调用哪些）。测试时学习（Test-Time Learning），Surprise 指标（梯度误差）决定新颖性。处理 2M+ 上下文

### 认知架构与理论框架

- [2023.09] [CoALA: Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427) 将 agent 记忆正式结构化为四类：Working Memory（工作记忆）、Episodic Memory（情景记忆）、Semantic Memory（语义记忆）、Procedural Memory（程序记忆）。**统一了所有论文的术语，理解 Agent 架构的通用蓝图**
- [2024.04] [大模型智能体记忆机制综述](https://arxiv.org/abs/2404.13501) 经典综述。记忆分类：参数化记忆 vs 上下文记忆。六大操作：巩固（Consolidation）、更新（Update）、索引（Indexing）、遗忘（Forgetting）、检索（Retrieval）、压缩（Compression）。开源 [论文列表](https://github.com/nuster1128/LLM_Agent_Memory_Survey)
- [多模态记忆综述](https://github.com/patrick-tssn/Awesome-Multimodal-Memory) 400+ 篇多模态记忆论文：视觉记忆、机器人记忆、多模态上下文建模

### 记忆架构创新

- [2026.07] [Filesystem-Based Memory for LLM Agents](https://arxiv.org/abs/2607.26637) 将记忆组织为文件系统：目录树+markdown 文件，agent 自己读写重组。研究了文件系统作为记忆媒介的组织、演化和可持续性
- [2026.07] [MemChain: Learning Interpretable Memory Traces](https://arxiv.org/abs/2607.24097) 记忆链：不只检索记忆作为证据，还学习可解释的记忆追踪路径
- [2026.07] [Experience Memory Graph: One-Shot Error Correction for Agents](https://arxiv.org/abs/2607.13884) 经验记忆图：将过往错误和修正编码为图结构，实现一次性错误纠正
- [2026.07] [Bridging Inference-Time Scaling and Episodic Memory with Action-Centric Graphs](https://arxiv.org/abs/2607.27415) 以动作为中心的记忆图，连接推理时扩展和情景记忆，避免无状态重复搜索
- [2026.08] [PGMem: Tightly Coupled Persona-Memory Graph for Lifelong Personalized Agents](https://arxiv.org/abs/2608.01708) 人格-记忆紧密耦合图：不把人格存为扁平 profile，而是与事件关联，解决人格与事件脱节问题
- [2026.06] [Neural Procedural Memory: Empowering LLM Agents with Implicit Activation Steering](https://arxiv.org/abs/2606.29824) 神经程序记忆：不依赖文本检索，通过隐式激活引导来调用程序性知识
- [2026.06] [DYNA: Dynamic Episodic Memory Networks for Temporal Knowledge Graphs](https://arxiv.org/abs/2606.15778) 动态情景记忆网络，用 时序知识图谱增强冻结的 LLM

### 开源记忆框架

| 框架 | 核心隐喻 | 记忆类型 | 仓库 |
|------|---------|---------|------|
| [mem0](https://github.com/mem0ai/mem0) | 知识图谱 | 图增强记忆，增/改/删 pipeline | 29k+ stars |
| [Letta (MemGPT)](https://github.com/letta-ai/letta) | 操作系统 | 分层（RAM/Disk），虚拟内存分页 | 22k+ stars |
| [Zep](https://github.com/getzep/zep) | 时间图谱 | 时序知识图谱，自研 Graphiti 引擎 | 3k+ stars |
| [MemOS](https://github.com/MemTensor/MemOS) | 记忆OS | 三类记忆动态转换，MemCube 统一抽象 | - |
| [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) | 海马体 | 神经生物学启发，PPR+知识图谱 | 1k+ stars |
| [LangMem (LangChain)](https://github.com/langchain-ai/langmem) | - | 写入/修订/删除，与 LangGraph 集成 | - |

---

## 三、知识自进化 = 店铺记忆

> 核心观点：知识自进化本质上就是 agent 的"店铺记忆"——在日常运行中主动积累经验、组织知识、检测冲突、更新认知。

### 知识自进化的四个环节（对应店铺经营）

1. **经验积累**（写入记忆）：agent 执行任务 → 记录 trajectory → 提取可复用知识
   - 店铺类比：每天接待客人，记录什么做法有效
2. **知识组织**（记忆管理）：将碎片化经验结构化为可检索的知识
   - 店铺类比：把零散经验整理成操作手册
3. **知识复用**（记忆检索）：新任务来时，检索相关经验指导决策
   - 店铺类比：遇到类似客人，参考过去的成功经验
4. **知识更新**（记忆演化）：发现旧知识过时或错误时，更新或遗忘
   - 店铺类比：菜单变了，更新操作手册，废弃旧做法

### 自进化记忆 (Self-Evolving Memory)

- [2025.11] [Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory](https://arxiv.org/abs/2511.20857) 自进化记忆基准。评估 agent 在测试时通过记忆管理实现持续学习的能力
- [2026.07] [CrystalMem: Elastic Memory for Self-Evolving LLM Agents via Knowledge Crystallization](https://arxiv.org/abs/2608.00303) 知识结晶化：将松散经验结晶为紧凑知识。发现记忆配额压缩后能力不会自动恢复，提出弹性记忆架构
- [2026.04] [Adaptive Memory Crystallization (AMC)](https://arxiv.org/abs/2604.13085) 自适应记忆结晶：渐进式经验合并的持续 RL 记忆架构
- [2026.08] [RoMeRL: Balancing Feedback Coverage and the Memory-Reward Trap](https://arxiv.org/abs/2608.02508) 自进化记忆的两大挑战：(1) 轨迹索引效用随历史增长而分散；(2) 记忆本身成为 reward hack 目标。提出降阶效用状态
- [2026.07] [Causal-AgentIR: Self-Evolving Causal Memory for Adaptive Image Restoration Agents](https://arxiv.org/abs/2607.21125) 因果记忆：不只是存什么，还存因果关系，实现自适应记忆

### 知识中心式自改进 (Knowledge-Centric Self-Improvement)

- [2026.07] [Knowledge-Centric Self-Improvement](https://arxiv.org/abs/2607.19592) **关键论文**。提出从 agent-centric 到 knowledge-centric 的范式转变：不是改进 agent 本身（prompt/workflow/harness），而是改进 agent 使用的知识。知识可迁移、可审计、可组合
- [2026.08] [Self-Improving LLMs via Progressive Experience Evolution](https://arxiv.org/abs/2608.02139) 渐进式经验进化：将短暂交互经验转化为持久模型能力
- [2026.07] [Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops](https://arxiv.org/abs/2607.07663) 递归自改进全景：从有界自精炼到自主研究循环

### 技能自进化 (Skill Self-Evolution)

- [2026.07] [MetaSkill-Evolve: Recursive Self-Improvement via Two-Timescale Meta-Skill Evolution](https://arxiv.org/abs/2607.05297) 元技能进化：双层时间尺度的递归自改进，技能本身也在演化
- [2026.08] [SkillProx: Self-Evolving Agent Skills via Proximal Textual Gradient Descent](https://arxiv.org/abs/2608.07449) 近端文本梯度下降优化技能，轻量级可复用
- [2026.07] [SkillRise: Agentic RL for Cross-Task Skill Evolution](https://arxiv.org/abs/2607.26784) 跨任务技能进化的 agentic RL
- [2026.08] [SkillZip: Contract-Preserving Graph Compression for Scalable Skill Libraries](https://arxiv.org/abs/2608.05604) 技能库压缩：保持执行契约的前提下压缩技能上下文
- [2026.08] [Comparative Approaches to Agent Retrieval over Large Skill Libraries](https://arxiv.org/abs/2608.06196) 大规模技能库的检索策略
- [2026.05] [Skill-as-Pseudocode: Refactoring Skill Libraries to Pseudocode](https://arxiv.org/abs/2605.27955) 技能从自由文本重构为伪代码，减少每次检索后的推理开销

### 经验学习与反思

- [2023.03] [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) 语言强化学习：agent 执行→反思失败原因→将教训存入记忆→下次避免。**最经典的"从失败中学习并记忆"方案**
- [2023.08] [ExpeL: LLM Agents Are Experiential Learners](https://arxiv.org/abs/2308.10144) 经验学习者：从 trajectory 中提取 insights 和 rules，形成可复用的经验知识
- [2023.10] [AgentTuning: Enabling Generalized Agent Abilities for LLMs](https://arxiv.org/abs/2310.12823) 通过 agent 任务微调赋予 LLM 通用 agent 能力
- [2026.06] [Escaping the Self-Confirmation Trap: An Execute-Distill-Verify Paradigm](https://arxiv.org/abs/2606.24428) 自确认陷阱：同一 agent 执行+总结会导致自我确认偏差。提出执行-蒸馏-验证分离范式
- [2026.07] [MemoHarness: Agent Harnesses That Learn from Experience](https://arxiv.org/abs/2607.14159) harness 本身从经验中学习：不只是 agent 改进，管理 agent 的框架也在改进
- [2026.07] [Sample-Efficient Learning from Agent Experience](https://arxiv.org/abs/2607.21051) 从 agent 自身经验中高效学习，减少环境交互成本

### 联邦知识进化

- [2026.07] [FedAgentKE: Federated Semantic Knowledge Evolution for Heterogeneous Agents](https://arxiv.org/abs/2607.21361) 多个异构 agent 的联邦知识进化：不只是单 agent 进化，多个 agent 共享和进化知识
- [2026.07] [AgentBrew: Lifelong Knowledge Brewing from Strong Teachers to Weak LLM Agents](https://arxiv.org/abs/2607.16851) 知识酿造：将 teacher agent 的交互经验蒸馏为 student 的持久外部记忆

---

## 四、记忆的写入、检索与遗忘

### 记忆写入与检索

- [2026.07] [LeanMem: Simple and Efficient Long-Term Memory for LLM Agents](https://arxiv.org/abs/2608.03463) 简洁高效的长时记忆：针对异构对话内容设计差异化摘要和检索
- [2026.07] [Exploratory and Assimilating Reflection: Reflective Recall Cycle](https://arxiv.org/abs/2607.17879) 探索-同化反思循环：不只是检索记忆，还通过反思改进检索
- [2026.07] [CMI-Mem: Generalizable Long-Term Memory Management via CMI-Augmented RL](https://arxiv.org/abs/2607.20553) 用 RL 管理记忆：不依赖 LLM 判断，用强化学习训练记忆管理策略
- [2026.07] [TRUSTMEM: Learning Trustworthy Memory Consolidation](https://arxiv.org/abs/2606.25161) 可信记忆合并：学习何时写入、修订、删除记忆

### 记忆遗忘与冲突

- [2026.07] [What to Keep, What to Forget: A Rate-Distortion View of Memory Compaction](https://arxiv.org/abs/2607.08032) **信息论视角**：用率失真理论分析记忆压缩，什么该留什么该忘
- [2026.06] [Learning What Not to Forget: Long-Horizon Agent Memory from a Few Kilobytes](https://arxiv.org/abs/2606.20954) 只用几 KB 学习参数就能决定长程记忆中什么不能忘
- [2026.06] [Control-Plane Placement Shapes Forgetting](https://arxiv.org/abs/2606.15903) 13 种系统配置的架构研究：记忆管道中 LLM 的位置决定遗忘模式
- [2026.06] [Manufactured Confidence: How Memory Consolidation Turns Hearsay into Confident Facts](https://arxiv.org/abs/2606.29279) 记忆合并的风险：将随口一说变成"确信事实"。mem0/LangMem 等都有此问题
- [2026.08] [When Memory Becomes Authority: Benchmarking Authority Collapse](https://arxiv.org/abs/2608.01679) 记忆合并边界处的"权威坍缩"：合并后的记忆被过度信任

### 记忆安全

- [2026.07] [Do Agents Dream of False Memories? Black-box Visual Attacks on Long-term Memory](https://arxiv.org/abs/2607.15657) 多模态 agent 的虚假记忆注入攻击
- [2026.07] [Insecure Coding Preferences in Long-Term Memory](https://arxiv.org/abs/2607.17619) 不安全编码偏好被存入记忆后的安全风险
- [2026.08] [When Experience Becomes Instruction: Trajectory Poisoning in Self-Evolving Agent Skill Systems](https://arxiv.org/abs/2608.05563) 技能自进化系统的轨迹投毒攻击

---

## 五、记忆评估基准

### 主流基准

- **LOCOMO** 长对话记忆评测。Mem0/MemGPT/Zep/A-MEM 等均在 LOCOMO 上对比。Mem0 综合 SOTA，MemOS SOTA
- **LongMemEval** 企业级长期记忆评测。Zep 准确度提升 18.5%
- **DMR Benchmark** 动态记忆检索评测。Zep 94.8% vs MemGPT 93.4%
- **MemoryBench** 多任务记忆能力评测

### 最新基准

- [2025.11] [Evo-Memory](https://arxiv.org/abs/2511.20857) 自进化记忆基准：评估测试时学习
- [2026.07] [Ground Truth First](https://arxiv.org/abs/2607.21962) 纵向评估工具：先有 ground truth 再生成对话，避免标签错误
- [2026.07] [MemOps: Benchmarking Lifecycle Memory Operations](https://arxiv.org/abs/2607.12893) 记忆生命周期操作基准：不只测问答，测写入/修订/删除全流程
- [2026.07] [RECON: Benchmarking Agent Memory for Compositional Reasoning](https://arxiv.org/abs/2607.16716) 组合推理记忆基准
- [2026.07] [Keep It InMind: Benchmarking the Implicit-Association Blind Spot](https://arxiv.org/abs/2607.24368) 隐式关联盲区：需要的记忆不一定和查询相似
- [2026.08] [FinPerMA: Personalized-Memory Benchmark for Financial Agents](https://arxiv.org/abs/2608.04095) 金融场景个性化记忆基准
- [2026.08] [PATH-Bench: Path-Dependent Evaluation of Lifelong Agents](https://arxiv.org/abs/2608.01149) 路径依赖评估：经验积累的路径影响 agent 最终能力
- [2026.06] [AFTER: Managing Procedural Memory in LLM Agents](https://arxiv.org/abs/2606.23127) 程序记忆基准：382 个企业任务，6 种角色，22 项程序技能

---

## 六、持续学习与终身记忆

- [2026.07] [When Does Continual Learning Require Learning](https://arxiv.org/abs/2607.07847) 质问持续学习的本质：LLM 时代，多少"持续学习"其实是上下文管理而非真正学习
- [2026.07] [Latent-LoRA: Compact Latent-Space Adapters with Gradient-Free Routing for Continual Learning](https://arxiv.org/abs/2607.23837) 潜空间适配器的持续学习：无梯度路由分配不同任务
- [2026.06] [LLM Evolution as an Industry-Scale Ecosystem](https://arxiv.org/abs/2606.24901) 工业级 LLM 持续学习的生命周期视角
- [2026.05] [From Fact Overwriting to Knowledge Evolution: Causal Editing via On-Policy Self-Distillation](https://arxiv.org/abs/2605.28303) 从事实覆写到知识进化：因果编辑替代暴力注入

---

## 七、业界产品级方案

| 产品 | 记忆机制 | 特点 |
|------|---------|------|
| OpenAI ChatGPT Memory | 跨会话用户偏好记忆 | 用户可查看/删除，分 saved memories 和 chat history 两类 |
| Claude (Anthropic) | 项目级记忆 | Projects 中的知识库+上下文，不跨项目 |
| Google Gemini Memory | 用户偏好+历史 | 跨对话记忆，可管理 |
| mem0 | 开源记忆层 | 提取-存储-检索 pipeline，支持多种后端 |
| Letta (MemGPT) | 虚拟内存分页 | LLM as OS，主上下文+外部存储 |
| Zep | 时序知识图谱 | 对话→知识图谱三元组，时间推理 |
| LangMem | LangChain 记忆工具 | 写入/修订/删除操作，与 LangGraph 集成 |

---

## 八、知识自进化可借鉴方案总结

> 针对你正在做的"知识自进化"，以下是最直接可借鉴的方案路线：

### 路线 A：技能库自进化（程序记忆路线）

参考 Voyager → MetaSkill-Evolve → SkillProx 路线：
1. agent 执行任务，记录成功 trajectory
2. 从 trajectory 中提取可复用技能（代码/自然语言指令）
3. 技能存入库，新任务时检索相关技能
4. 技能使用后根据反馈迭代优化
5. 定期压缩技能库（SkillZip），废弃低效技能

### 路线 B：经验蒸馏自进化（情景→语义记忆路线）

参考 ExpeL → Reflexion → Knowledge-Centric Self-Improvement 路线：
1. agent 执行任务，记录成功/失败 trajectory
2. 反思失败原因，提取教训（Reflexion 式）
3. 从多次经验中归纳通用 insights 和 rules（ExpeL 式）
4. 将 insights 组织为可检索的知识库（知识中心式）
5. 新任务时检索相关 knowledge 指导决策
6. 检测知识冲突，更新过时知识

### 路线 C：知识结晶化（记忆压缩路线）

参考 CrystalMem → AMC → Rate-Distortion 路线：
1. 原始经验以情景记忆形式存储（详细 trajectory）
2. 定期"结晶化"：将松散经验压缩为紧凑知识
3. 用率失真理论指导压缩（什么该留什么该忘）
4. 结晶后的知识可跨任务复用
5. 处理记忆配额波动（弹性记忆）

### 关键设计决策

1. **记忆存什么**：trajectory 原文 vs 提炼后的 insights？前者保真但冗余，后者紧凑但可能丢失上下文
2. **记忆怎么检索**：向量相似度 vs 图遍历 vs LLM 判断？各有利弊
3. **记忆何时更新**：实时 vs 批量？实时开销大，批量可能用过时知识
4. **记忆冲突怎么解决**：覆盖 vs 保留多版本 vs 时间戳优先？
5. **记忆安全**：如何防止错误经验/投毒轨迹污染知识库？
