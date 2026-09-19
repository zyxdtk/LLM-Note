# Physical AI Agent

> Physical AI：AI 从屏幕走向物理世界。agent 感知环境、理解指令、规划动作、通过物理身体执行。核心挑战：实时安全约束、物理定律遵循、sim-to-real 鸿沟、长程任务。

---

## 一、综述与概念

- [2025.01] [Generative Physical AI in Vision: A Survey](https://arxiv.org/abs/2501.10928) 生成式 Physical AI 综述，覆盖图像/视频/3D/4D 生成在物理智能中的应用
- [2025.08] [Towards Embodied Agentic AI: Review and Classification of LLM- and VLM-Driven Robot Autonomy and Interaction](https://arxiv.org/abs/2508.05294) LLM/VLM 驱动的机器人自主性和交互分类综述
- [2026.05] [World Action Models: The Next Frontier in Embodied AI](https://arxiv.org/abs/2605.12090) World Action Model 作为 Embodied AI 下一前沿。VLA 学语义泛化但不建模物理世界演化，WAM 显式建模干预下的世界动态
- [2026.07] [Physical AI Governance: From Theory to Practice Across Life Cycle](https://arxiv.org/abs/2607.22877) Physical AI 全生命周期治理：实时安全约束、动态环境交互
- [2026.03] [Safety in Embodied AI: A Survey of Risks, Attacks, and Defenses](https://arxiv.org/abs/2605.02900) Embodied AI 安全：风险、攻击、防御
- [2026.06] [Safe Embodied AI for Long-horizon Tasks: A Cross-layer Analysis of Robotic Manipulation](https://arxiv.org/abs/2606.05660) 长程任务安全的跨层分析
- [2026.07] [Security of World-Model-Based Embodied AI: A Lifecycle of Threats, Defenses, and Evaluation](https://arxiv.org/abs/2607.28226) 世界模型驱动的 Embodied AI 安全生命周期
- [2026.08] [What Language Does and What the Evidence Supports: A Functional Role Taxonomy of Language Grounding in Embodied Agents](https://arxiv.org/abs/2608.03099) 语言在 Embodied Agent 中的功能角色分类

---

## 二、世界模型 (World Model)

Physical AI 的核心基础设施：物理世界模拟器，用于训练和验证 agent。

### 世界基础模型

- [2025.01] [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575) NVIDIA。Physical AI 需要"数字优先训练"：policy model（自身的数字孪生）+ world model（世界的数字孪生）。Cosmos 提供可定制的世界基础模型平台
- [2026.06] [Cosmos 3: Omnimodal World Models for Physical AI](https://arxiv.org/abs/2606.02800) NVIDIA。全模态世界模型：统一处理语言、图像、视频、音频、动作序列。混合 Transformer 架构，支持灵活输入输出配置
- [2025.05] [DreamGen: Unlocking Generalization in Robot Learning through Video World Models](https://arxiv.org/abs/2505.12705) 4 阶段 pipeline：用视频世界模型生成神经轨迹（合成机器人数据），实现跨行为和环境泛化

### World Action Model (WAM)

VLA 的进化方向：不只观察→动作，还建模"干预下世界如何演化"。

- [2026.07] [From World Action Models to Embodied Brains: A Roadmap for Open-World Physical Intelligence](https://arxiv.org/abs/2607.11689) WAM 路线图：连接候选干预与预测后果，是通向开放世界 Physical Intelligence 的路径
- [2026.07] [DSWAM: A Dual-System World Action Foundation Model for Fine-Grained Robot Manipulation](https://arxiv.org/abs/2607.04927) 双系统 WAM：视频世界建模提供密集监督 + 语言级规划接口
- [2026.07] [DynaWM: A Base-VLA-Guided World Foundation Model for Moving-Object Manipulation](https://arxiv.org/abs/2607.02604) 动态移动物体的操作：VLA+世界模型联合
- [2026.08] [Foresight Without Seeing: Latent Futures for World Action Models](https://arxiv.org/abs/2608.11605) 隐式未来预测：不需要显式视频 rollout，用潜在表示预测未来指导动作
- [2026.08] [Keep the Future, Drop the Rollout: RIFT for World Action Models](https://arxiv.org/abs/2608.11521) 只需未来表示，不需要迭代视频 rollout。4 个 WAM × 40 个 LIBERO 任务验证
- [2026.08] [Flex-π: A Multi-Stream World-Action Model with Compute Flexibility](https://arxiv.org/abs/2608.10860) 多流 WAM：同一视频生成骨干同时预测 RGB+3D几何+物体语义
- [2026.08] [JEPA-WAM: Stage-Level Joint-Embedding Prediction for World-Action Models](https://arxiv.org/abs/2608.10780) 阶段级联合嵌入预测：不只预测短期局部未来，还预测阶段级全局未来
- [2026.08] [Surgical WAM: A World-Action Model for Data-Efficient Surgical Robot Learning](https://arxiv.org/abs/2608.11204) 手术机器人 WAM：解决动作标注数据稀缺问题

### Agent 世界模型

- [2026.07] [DSWorld: A Data Science World Model for Efficient Autonomous Agents](https://arxiv.org/abs/2607.15901) 数据科学世界模型：预测数据操作的效果，减少试错计算
- [2026.08] [FedWorld: Scope-Aware Federation of Agent World Models](https://arxiv.org/abs/2608.01561) 联邦世界模型：多个 agent 共享世界知识，scope-aware 合并
- [2026.06] [Active Inference as the Test-Time Scaling Law for Physical AI Agents](https://arxiv.org/abs/2606.22813) 主动推理作为 Physical AI 的测试时 scaling law：agent 用世界模型推理应对未见场景

---

## 三、Embodied 基础模型

### 通用 Embodied 基础模型

- [2026.07] [ACE-Brain-0.5: A Unified Embodied Foundational Model for Physical Agentic AI](https://arxiv.org/abs/2607.04426) 统一 Embodied 基础模型：理解→规划→执行→监控→改进的完整循环。现有系统只解决局部，ACE-Brain 做完整闭环
- [2026.06] [Embodied-R1.5: Evolving Physical Intelligence via Embodied Foundation Models](https://arxiv.org/abs/2606.11324) 统一 EFM：embodied 认知+任务规划+纠错+指向，单一架构。三种自动化数据 pipeline
- [2026.07] [RynnBrain 1.1: Towards More Capable and Generalizable Embodied Foundation Model](https://arxiv.org/abs/2607.17977) 2B/9B/122B-A10B 三规模，统一时空+物理接地框架，支持感知+空间推理+定位+规划
- [2026.04] [HY-Embodied-0.5: Embodied Foundation Models for Real-World Agents](https://arxiv.org/abs/2604.07430) 面向真实世界 embodied agent 的基础模型
- [2026.06] [ROBOSHACKLES: A Safety Dataset for Human-Injury Prevention in Embodied Foundation Models](https://arxiv.org/abs/2606.18632) Embodied 基础模型的人身伤害预防安全数据集

### 机器人基础模型

- [2025.03] [GR00T N1: An Open Foundation Model for Generalist Humanoid Robots](https://arxiv.org/abs/2503.14734) NVIDIA。通用人形机器人基础模型。需要 versatile body + intelligent mind
- [2025.07] [RoboBrain 2.0 Technical Report](https://arxiv.org/abs/2507.02029) Embodied 视觉语言基础模型，7B/32B 双版本，统一感知+推理+规划
- [2026.01] [RoboBrain 2.5: Depth in Sight, Time in Mind](https://arxiv.org/abs/2601.14352) 引入深度感知+时序建模
- [2025.02] [RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete](https://arxiv.org/abs/2502.21257) 统一脑模型：从抽象语义到具体操作的层次化处理
- [2024.05] [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213) 开源通用机器人策略：预训练+少量数据微调即可适配新场景

### VLA 模型

详见 [VLA.md](./VLA.md)。此处补充最新进展：

- [2024.06] [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246) 开源 VLA，大规模预训练+少量微调
- [2026.08] [StellaVLA: In-Context Structured Demonstration for Generalizable VLA Models](https://arxiv.org/abs/2608.11671) 上下文结构化示例，解决 OOD 性能坍缩
- [2026.08] [XCoT-VLA: Executable Chain-of-Thought for VLA Driving](https://arxiv.org/abs/2608.10976) 可执行思维链：自然语言 CoT 不适合实时控制，改为可执行格式
- [2026.08] [Neural Introspection Gating for Adaptive KV-Cache Reuse in VLA Models](https://arxiv.org/abs/2608.10824) VLA 实时控制的 KV-Cache 复用：视觉 token 跨步变化小，门控决定是否重算
- [2026.08] [Lost in Reconstruction: Aligning Action Representations with Language in VLA Models](https://arxiv.org/abs/2608.10484) 动作表示对齐：L1/L2 重建损失的数值接近不等于语义接近
- [2026.08] [VANE: Reliable Test-Time Training for VLA Models](https://arxiv.org/abs/2608.09448) VLA 测试时训练：从无标注部署流自适应
- [2026.08] [WA-SpecDec: World-Aware Speculative Decoding for VLA Models](https://arxiv.org/abs/2608.08725) VLA 的投机解码：用世界模型做 draft 验证，降低闭环延迟
- [2026.07] [Towards Human-like Physical Intelligence: Lifelong VLA Learning for Robotic Manipulation](https://arxiv.org/abs/2607.14852) VLA 终身学习：像人一样顺序学习新任务
- [2026.08] [Hidden in Plain Sight: Diffusion-Based Unrestricted Robotic Attacks on VLA Models](https://arxiv.org/abs/2608.10393) VLA 对抗攻击：扩散模型生成物理世界攻击

---

## 四、Embodied Agent 架构

### 规划与执行

- [2026.08] [When Replanning Becomes the Bottleneck: Budgeted Replanning for Embodied Agents](https://arxiv.org/abs/2608.01428) 重规划瓶颈：LLM 重规划调用累积上下文导致延迟，提出预算化重规划
- [2026.07] [Mosaic: Runtime-Efficient Multi-Agent Embodied Planning](https://arxiv.org/abs/2607.09603) 多 agent embodied 规划：失败动作是主要瓶颈，来自不准确状态追踪+低效协调
- [2026.07] [TypeGo: An OS Runtime for Embodied Agents](https://arxiv.org/abs/2607.05482) Embodied agent 的操作系统式运行时：LLM 不在关键路径上，支持实时控制和并发目标
- [2026.07] [EFLUX: Elastic Multi-Robot Formation Navigation with Agentic LLMs](https://arxiv.org/abs/2607.12050) LLM 驱动的多机器人编队导航：自适应编队几何+拓扑
- [2026.05] [Hierarchical Prompting with Dual LLM Modules for Robotic Task and Motion Planning](https://arxiv.org/abs/2605.08330) 双 LLM 层次化框架：高层规划 agent + 低层空间推理
- [2024.04] [Long-horizon Locomotion and Manipulation on a Quadrupedal Robot with LLMs](https://arxiv.org/abs/2404.05291) 四足机器人长程运动+操作：LLM 做高层语义理解+低层运动控制

### 分布式与多 Agent

- [2026.07] [Distributed Agent System: Fault-Tolerant Collaboration Among Embodied Agents](https://arxiv.org/abs/2607.10811) Embodied Agent 容错协作：长程任务+资源约束+环境不确定性
- [2026.07] [Communication-Efficient Digital-Twin Coordination for Heterogeneous LLM Embodied Agents](https://arxiv.org/abs/2607.09330) 异构 LLM Embodied Agent 的数字孪生协调：智能工厂/仓库/服务机器人
- [2026.08] [When Coordination Becomes a Threat: Communication Attacks in LLM-Controlled Multi-Robot Systems](https://arxiv.org/abs/2608.06830) LLM 多机器人系统的通信攻击安全风险

### Embodied 记忆

- [2026.08] [Mimir: A Neuro-Symbolic Memory System with Dynamic Grounding for Embodied Agents](https://arxiv.org/abs/2608.04933) 长程 embodied 任务的记忆：场景信念+执行进度，不是扁平历史
- [2026.08] [Long-Horizon Embodied Decision-Making via Multimodal Memory Compression](https://arxiv.org/abs/2608.01456) 多模态记忆压缩：长程决策需要跨时间积累证据
- [2026.07] [Beyond Episodic Evaluation: Memory Architectural Bottlenecks in Sequential Embodied QA](https://arxiv.org/abs/2607.21571) Embodied QA 的记忆架构瓶颈：连续操作中积累、保留、选择性复用信息
- [2026.06] [Analytic Concept-Centric Memory for Agentic Embodied Manipulation](https://arxiv.org/abs/2606.29774) 概念中心记忆：记住持久物体+追踪场景状态+复用交互知识
- [2026.07] [GEM-Occ: From Visual Geometry Evidence to Embodied Semantic Occupancy Memory](https://arxiv.org/abs/2607.05543) 语义占据记忆：结构化空间记忆，联合表示占据区域+自由空间+未知区域+物体语义
- [2026.07] [ABot-AgentOS: A General Robotic Agent OS with Lifelong Multi-modal Memory](https://arxiv.org/abs/2607.10350) 通用机器人 Agent OS：终身多模态记忆，支持推理/记忆/工具/验证/跨身体执行

---

## 五、操作与交互

### 触觉与灵巧操作

- [2026.07] [TouchWorld: A Predictive and Reactive Tactile Foundation Model for Dexterous Manipulation](https://arxiv.org/abs/2607.07287) 触觉基础模型：预测接触演化+快速纠正局部错误（滑动/错位/不稳定抓取/力失配）
- [2026.06] [Tactile Genesis: Exploring Tactile Sensors at Scale for Learning Dexterous Tasks](https://arxiv.org/abs/2606.22332) 大规模触觉传感器探索：哪种触觉抽象是策略需要的
- [2026.05] [DexSim2Real: Foundation Model-Guided Sim-to-Real Transfer for Generalizable Dexterous Manipulation](https://arxiv.org/abs/2605.05241) 基础模型引导的 sim-to-real 迁移
- [2025.07] [HumanoidGen: Data Generation for Bimanual Dexterous Manipulation via LLM Reasoning](https://arxiv.org/abs/2507.00833) LLM 推理生成人形机器人双手灵巧操作数据

### Affordance 与场景理解

- [2026.05] [StateVLM: A State-Aware Vision-Language Model for Robotic Affordance Reasoning](https://arxiv.org/abs/2605.03927) 状态感知 VLM 做 affordance 推理
- [2025.11] [OVAL-Grasp: Open-Vocabulary Affordance Localization for Task Oriented Grasping](https://arxiv.org/abs/2511.20841) 开放词汇 affordance 定位：零样本任务导向抓取
- [2025.09] [O³Afford: One-Shot 3D Object-to-Object Affordance Grounding](https://arxiv.org/abs/2509.06233) 一次性 3D 物体间 affordance 定位

### 人机协作

- [2026.08] [Structured LLM Reasoning for Zero-Shot Human-Robot Coordination Under Hidden Goals](https://arxiv.org/abs/2608.04309) 零样本人机协作：Dec-POMDP 建模，动作条件化 ToM 推理
- [2026.05] [Assistance Without Interruption: Benchmark for Non-Intrusive Human-Robot Assistance](https://arxiv.org/abs/2605.01368) 非侵入式辅助：机器人主动支持人类多步活动但严格不打断
- [2026.08] [Diminishing Returns of Intelligence: LLM Scale vs User Perception in Social HRI](https://arxiv.org/abs/2608.08320) 社交人机交互中 LLM 规模的边际递减效应

---

## 六、数据与仿真

### 数据集

- [2024.03] [DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset](https://arxiv.org/abs/2403.12945) 大规模真实世界机器人操作数据集
- [2023.10] [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/abs/2310.08864) 多机构联合机器人学习数据集
- [2023.10] [GenSim: Generating Robotic Simulation Tasks via Large Language Models](https://arxiv.org/abs/2310.01361) LLM 生成机器人仿真任务

### Sim-to-Real

- [2025.12] [AnyTask: Automated Task and Data Generation Framework for Advancing Sim-to-Real Policy Learning](https://arxiv.org/abs/2512.17853) 自动化任务+数据生成框架，推进 sim-to-real 策略学习
- [2026.04] [3D Generation for Embodied AI and Robotic Simulation: A Survey](https://arxiv.org/abs/2604.26509) Embodied AI 的 3D 生成：超越视觉真实，需要物理接地的可交互 3D 内容

### 部署与推理

- [2026.08] [PhyAI: Real-Time Physical AI at the Edge, Scalable Rollouts in the Cloud](https://arxiv.org/abs/2608.03682) Physical AI 全生命周期推理：模型评估→云端 RL rollout→边缘 GPU serving→机载部署，统一 checkpoint
- [2026.07] [Edge Physical AI Deployment of Vision Transformers on Heterogeneous Edge GPU](https://arxiv.org/abs/2607.10942) 自动驾驶边缘部署：ViT 在异构边缘 GPU 上的延迟和能耗优化

---

## 七、空间智能

- [2025.04] [A Survey of LLM-Powered Spatial Intelligence Across Scales](https://arxiv.org/abs/2504.09848) 跨尺度空间智能：导航→城市规划→遥感
- [2025.08] [From reactive to cognitive: brain-inspired spatial intelligence for embodied agents](https://arxiv.org/abs/2508.17198) 脑启发空间智能：landmarks→route knowledge→survey knowledge 三层
- [2025.09] [Embodied Spatial Intelligence: from Implicit Scene Modeling to Spatial Reasoning](https://arxiv.org/abs/2509.00465) Embodied 空间智能：隐式场景建模到空间推理
- [2026.06] [Keep It in Mind: User Centric Continual Spatial Intelligence Reasoning in Egocentric Video Streams](https://arxiv.org/abs/2606.15200) 第一人称视频流中的持续空间智能推理，170+ 小时数据

---

## 八、安全与治理

- [2026.07] [Physical AI Governance: From Theory to Practice Across Life Cycle](https://arxiv.org/abs/2607.22877) Physical AI 全生命周期治理
- [2026.05] [Silent Failures in Physical AI: Runtime Action Authorization for Autonomous Systems](https://arxiv.org/abs/2606.00090) Physical AI 的静默失败：运行时动作授权
- [2026.07] [Initiation Safety: A Missing Dimension in Generalist-Robot Safety](https://arxiv.org/abs/2607.07420) 通用机器人安全的缺失维度：是否应该执行第一个不可逆社交动作
- [2025.09] [SafeMind: Benchmarking and Mitigating Safety Risks in Embodied LLM Agents](https://arxiv.org/abs/2509.25885) Embodied LLM Agent 安全基准：任务理解→环境感知→规划→执行四阶段风险
- [2026.07] [Clinical Pathways as Safety Specifications for Physical AI in Hospital Wards](https://arxiv.org/abs/2607.19827) 医院场景 Physical AI 安全：临床路径作为运行时安全规范
- [2026.02] [What Breaks Embodied AI Security: LLM Vulnerabilities, CPS Flaws, or Something Else?](https://arxiv.org/abs/2602.17345) Embodied AI 安全破因分析：LLM 漏洞 vs 信息物理系统缺陷

---

## 九、评估基准

- [2026.05] [The Yes-Man Syndrome: Benchmarking Abstention in Embodied Robotic Agents](https://arxiv.org/abs/2605.20544) Embodied Agent 的"应声虫综合征"：VLM 做规划时不懂得拒绝
- [2026.04] [SocialGrid: A Benchmark for Planning and Social Reasoning in Embodied Multi-Agent Systems](https://arxiv.org/abs/2604.16022) Embodied 多 Agent 社交推理基准
- [2025.11] [SWITCH: Benchmarking Tangible Interfaces in Long-horizon Embodied Scenarios](https://arxiv.org/abs/2511.17649) 有形控制接口（家电面板/遥控器/电梯）的长程 embodied 基准
- [2025.12] [VLNVerse: A Benchmark for Vision-Language Navigation](https://arxiv.org/abs/2512.19021) 视觉语言导航基准：sim-to-real 泛化
- [2026.02] [RADAR: Benchmarking VLA Generalization via Real-World Dynamics](https://arxiv.org/abs/2602.10980) VLA 泛化基准：真实世界动力学+空间物理智能+自主评估
- [2026.07] [Active Real-World Factor-Based Evaluation for Generalist Robot Policies](https://arxiv.org/abs/2607.14439) 通用机器人策略的真实世界因子评估
- [2026.07] [Apple-π: Benchmarking Thinking with Video Towards Law-Grounded Physical Intelligence](https://arxiv.org/abs/2607.16401) 视频生成模型是否真正理解物理定律的基准

---

## 十、代表厂商/方案

| 方案 | 机构 | 特点 |
|------|------|------|
| [Cosmos](https://arxiv.org/abs/2501.03575) | NVIDIA | 世界基础模型平台，Physical AI 数字孪生 |
| [GR00T N1](https://arxiv.org/abs/2503.14734) | NVIDIA | 人形机器人基础模型 |
| [RT-1/RT-2](https://arxiv.org/abs/2307.15818) | Google | Robotics Transformer，VLA 开山 |
| [OpenVLA](https://arxiv.org/abs/2406.09246) | Stanford 等 | 开源 VLA |
| [Octo](https://arxiv.org/abs/2405.12213) | Berkeley 等 | 开源通用机器人策略 |
| [RoboBrain](https://arxiv.org/abs/2502.21257) | - | 统一脑模型，从抽象到具体 |
| [Embodied-R1.5](https://arxiv.org/abs/2606.11324) | - | 统一 EFM，认知+规划+纠错 |
| [ACE-Brain](https://arxiv.org/abs/2607.04426) | - | Physical Agentic AI 完整闭环 |
| Figure AI | Figure | 人形机器人（Helix VLA，未公开论文） |
| Physical Intelligence (π) | PI | 通用机器人策略（π0，未在 arXiv） |
| Tesla Optimus | Tesla | 人形机器人（未公开论文） |
