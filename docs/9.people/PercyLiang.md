# Percy Liang 与 Stanford CRFM

> 起因：用户对 Percy Liang 的推文和他在做的工作感兴趣。本文整理他的学术轨迹、核心项目、研究方向和影响力。
> 来源：arXiv 论文搜索（57 篇）、Stanford 主页、CRFM 官网、CS336 课程页面、X/Twitter 推文（via r.jina.ai 镜像）、Simile 官网、Marin 官网。

---

## 1. Percy Liang 是谁

- **职位**：Stanford 大学计算机科学教授（兼统计学），Gates 366
- **创建**：Stanford CRFM（Center for Research on Foundation Models，基础模型研究中心）
- **教学**：CS336 "Language Models from Scratch"（与 Tatsunori Hashimoto 共同授课）
- **创建**：Marin（开放模型开发项目，marin.community）、CodaLab Worksheets（实验管理平台）
- **关联**：Together AI（X 资料页标注）、Simile（创始人 Joon Sung Park 是他的博士生）
- **X/Twitter**：@percyliang，1,327 posts，2009 年 10 月加入，Stanford CA
- **主页**：https://cs.stanford.edu/~pliang/

### 自述研究方向

他的 Stanford 主页上写了几句话，很能反映他的哲学：

> "I am drawn to simple things, want to understand things deeply, and like to build useful systems. I am interested in fundamental questions around learning and intelligence. How well can we learn in the data-limited, infinite compute regime? How do we model interestingness and curiosity? How can we push the frontier of human knowledge?"

翻译：我偏爱简洁的东西，想深入理解事物，喜欢构建有用的系统。我对学习和智能的根本问题感兴趣：在数据有限、算力无限的 regime 下我们能学得多好？如何建模有趣性和好奇心？如何推动人类知识前沿？

---

## 2. 核心贡献：五条主线

Percy Liang 的工作横跨技术、评估、安全、政策和教育五个维度。下面按主线梳理。

### 主线一：命名了"Foundation Model"（基础模型）

**论文**：On the Opportunities and Risks of Foundation Models
**作者**：Rishi Bommasani, Drew Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, ..., Percy Liang (100+ 作者)
**时间**：2021.08
**arXiv**：https://arxiv.org/abs/2108.07258

这篇论文创造了"Foundation Model"这个术语，定义了"在广域数据上大规模训练、可适配多种下游任务的模型"这一范式。论文不是技术论文，而是一篇 200+ 页的立场文件，涵盖了技术基础、应用、社会影响三个维度。

意义：在 GPT-3 之后、ChatGPT 之前，这篇论文给了整个领域一个概念框架。后来所有人讨论 GPT/Claude/Gemini 都在用他定义的术语。

**后续衍生工作**：
- Ecosystem Graphs (arXiv:2303.15772)：追踪基础模型的社会足迹——谁训练了模型、谁部署、谁依赖
- Foundation Models and Fair Use (arXiv:2303.15715)：基础模型与版权合理使用
- Picking on the Same Person (arXiv:2211.13972)：算法单一文化导致结果同质化

### 主线二：HELM — 最系统的大模型评估体系

这是 Percy Liang 影响力最大的工程贡献。

**原始 HELM**：Holistic Evaluation of Language Models
**时间**：2022.11
**arXiv**：https://arxiv.org/abs/2211.09110
**作者**：Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, et al.

HELM 的核心理念：不只看模型在某个基准上的分数，而是"整体评估"——准确性、鲁棒性、公平性、偏见、毒性、效率等多维度。HELM 成为 2023-2024 年最权威的 LLM 排行榜之一。

**HELM 扩展为全模态评估体系**：

| 评估 | 全称 | 对象 | arXiv |
|------|------|------|-------|
| HELM | Holistic Evaluation of Language Models | 文本 LLM | 2211.09110 |
| VHELM | Holistic Evaluation of Vision Language Models | 视觉语言模型 | 2410.07112 |
| AHELM | Holistic Evaluation of Audio-Language Models | 音频语言模型 | 2508.21376 |
| HELM-T2I | Holistic Evaluation of Text-To-Image Models | 文生图模型 | 2311.04287 |
| Image2Struct | Benchmarking Structure Extraction for VLMs | VLM 结构提取 | 2410.22456 |
| MedHELM | Holistic Evaluation of LLMs for Medical Tasks | 医疗 LLM | 2505.23802 |
| EconEvals | Economic Evaluations of Language Models | 经济价值评估 | 2607.19375 |

**最新评估方向（2025-2026）**：

- **UQ (Unsolved Questions)** (arXiv:2508.17580)：用"未解决的问题"做基准——题目必须对前沿模型有挑战性，同时反映真实世界用途
- **HAL (Holistic Agent Leaderboard)** (arXiv:2510.11977)：Agent 评估缺少基础设施，HAL 填补这个空白
- **BountyBench** (arXiv:2505.15216)：用真实网络安全系统评估 AI 攻防能力，以美元影响量化
- **Best Practices for Agentic Benchmarks** (arXiv:2507.02825)：建立 Agent 基准的最佳实践

### 主线三：Foundation Model Transparency Index (FMTI) — 透明度运动

**论文**：The Foundation Model Transparency Index
**时间**：2023.10（首发），2024.07（年度更新），2025.12（最新）
**arXiv**：https://arxiv.org/abs/2310.12941

FMTI 是一个年度指数，给 OpenAI、Google、Anthropic、Meta 等基础模型开发商打"透明度分数"。评估维度包括：训练数据来源、模型架构、训练计算、评估方法、下游使用等。

**系列论文**：
- The Foundation Model Transparency Index (2023, arXiv:2310.12941)
- Foundation Model Transparency Reports (2024.02, arXiv:2402.16268)
- The 2024 Foundation Model Transparency Index (2024.07, arXiv:2407.12929)
- The 2025 Foundation Model Transparency Index (2025.12, arXiv:2512.10169)
- On the Societal Impact of Open Foundation Models (2024.02, arXiv:2403.07918)
- Language model developers should report train-test overlap (2024.10, arXiv:2410.08385)
- The Responsible Foundation Model Development Cheatsheet (2024.06, arXiv:2406.16746)

意义：这个指数直接影响了 AI 政策讨论。OpenAI、Google 等公司开始公开更多信息，部分原因是 FMTI 把不透明量化了。

### 主线四：记忆化、版权与安全

这是 Percy Liang 近年最活跃的研究方向，直接关联当前 AI 法律争议。

**记忆化提取系列**：
- Extracting memorized pieces of (copyrighted) books from open-weight LLMs (2025.05, arXiv:2505.12546) — 从开源权重模型中提取受版权保护的书籍片段
- Extracting books from production language models (2026.01, arXiv:2601.02671) — 从生产级模型中提取书籍
- Extractable Memorization From First Principles (2026.07, arXiv:2607.12649) — 从第一性原理定义"可提取记忆化"
- Estimating near-verbatim extraction risk (2026.03, arXiv:2603.24917) — 用解码约束束搜索估计近似逐字提取风险
- Blackbox Model Provenance via Palimpsestic Membership Inference (2025.10, arXiv:2510.19796) — 通过"羊皮纸式"成员推断证明模型血缘

**安全与缺陷报告**：
- FLARE-AI: Flaw Reporting for AI (2026.06, arXiv:2606.31567) — 部署 AI 系统的缺陷报告框架
- DelusionEval (2026.08, arXiv:2608.05004) — 测量 AI 聊天机器人的妄想关联行为
- SpecEval (2025.09, arXiv:2509.02464) — 评估模型是否遵守自己发布的行为规范
- AIR-Bench 2024 (2024.07, arXiv:2407.17436) — 基于法规和政策的安全基准
- DTap (2026.05, arXiv:2605.04808) — Agent 红队测试平台
- In-House Evaluation Is Not Enough (2025.03, arXiv:2503.16861) — 呼吁第三方缺陷披露
- Auditing Prompt Caching (2025.02, arXiv:2502.07776) — Prompt 缓存引入时序侧信道风险
- AILuminate v1.0 (2025.02, arXiv:2503.05731) — MLCommons 的 AI 风险与可靠性基准
- AI Audits Under Constraints (2025.09, arXiv:2509.05627) — 资源约束下的 AI 审计 scaling law

### 主线五：机器人 VLA 与 Agent

Percy Liang 近两年在 Physical AI 方向也很活跃，主要与 Chelsea Finn 合作。

**机器人/VLA**：
- Fine-Tuning Vision-Language-Action Models (2025.02, arXiv:2502.19645) — VLA 微调的速度与成功率优化
- RoboArena (2025.06, arXiv:2506.18123) — 分布式真实世界机器人策略评估
- WorldGym (2025.06, arXiv:2506.00613) — 用世界模型做策略评估环境
- RoboReward (2026.01, arXiv:2601.00675) — 通用视觉语言奖励模型
- Cosmos Policy (2026.01, arXiv:2601.16163) — 微调视频模型做视觉运动控制
- VLAW (2026.02, arXiv:2602.12063) — VLA 策略与世界模型迭代共提升

**Agent（非物理）**：
- Reinforcement Learning for MLE Agents (2025.09, arXiv:2509.01684) — 用 RL 让 ML 工程 Agent 从经验中学习
- MLE-Dojo (2025.05, arXiv:2505.07782) — ML 工程 Agent 的 Gym 式训练环境
- MLE-Smith (2025.10, arXiv:2510.07307) — 自动化多 Agent 管线扩展 MLE 任务
- Comparing AI Agents to Cybersecurity Professionals (2025.12, arXiv:2512.09882) — AI Agent vs 人类安全专家的真实渗透测试对比

---

## 3. Marin：激进开放的模型开发

Percy Liang 创建了 Marin 项目（marin.community），践行他称为"激进开放"（radical openness）的模型开发理念。

### 核心理念

Marin 超越了 open-weight（开放权重）和 open-source（开放源码）：

> "What would truly open-source AI look like? Not just open weights, open code/data, but *open development*, where the entire research and development process is public *and* anyone can contribute. We built Marin, an open lab, to fulfill this vision."（2025.05.19 置顶推文）

所有实验——无论成功还是失败——都预注册并实时公开。任何人都能看到和参与。

### 开放生态四要素

2026.07.25 推文，Percy Liang 提出了开放生态的四层框架：

> "The most important word here is *ecosystem*. It's not just about having an open-weight model. Open-weight models are a means to an end. To have a truly strong, open ecosystem, we need four critical frontier-level ingredients: open-weight models, open training datasets, open software stacks, and open process knowledge."

| 要素 | 含义 | 现状 |
|------|------|------|
| Open-weight models | 开放模型权重 | Llama/Qwen/GLM 等已做到 |
| Open training datasets | 开放训练数据集 | 少数做到（如 NVIDIA Nemotron） |
| Open software stacks | 开放软件栈 | 部分做到（vLLM/Levanter 等） |
| Open process knowledge | 开放过程知识 | **几乎没人做，Marin 填补这个空白** |

他特别称赞了 NVIDIA Nemotron 超越了开放权重，还发布了代码和数据集。Marin 进一步开放"过程知识"——不只是怎么训练一个模型，而是如何迭代地改进和塑造模型：如何设计 scaling law 和评估来指导架构和数据消融。

他把这四个要素称为"可再生资源"（renewable resources），而算力是"可消耗资源"（consumable resources）。开放生态的价值是让每个人能最高效地把算力转化为符合自己需求和价值观的模型。

### Marin 的工作流程

Marin 把 GitHub 改造为 AI 实验平台：

1. 在 GitHub 上创建 issue 预注册实验（假设、目标）
2. 提交 PR，包含可复现实验的代码
3. 社区专家审查 PR
4. 代码定义的 provenance graph 被执行，结果汇总到 WandB 报告
5. 所有实验（成功和失败的）都公开

### Marin 的成果

**Marin-8B**（2025.05 发布）：
- Marin-8B-Base：在 19 个基准中的 14 个上击败 Llama 3.1 8B
- Marin-8B-Instruct：在 Together AI 上可调用
- 从零训练，全流程公开

**Marin 535B-A23B**（2026.08.21 启动训练，最新）：

这是 Marin 迄今为止最大的训练运行。推文原文：

> "🚢 Marin 535B-A23B started training this week! As usual, the whole process is open. Voyage plan: pretraining (80%) + midtraining (20%) on 18.75T tokens on 11 x GB200 NVL72 for ~3 months (2.7e24 FLOPs). Post-training will follow."

关键参数：
- **模型**：535B 总参数，23B 激活参数（MoE 架构）
- **数据**：18.75T tokens
- **硬件**：11 x GB200 NVL72
- **时长**：约 3 个月
- **计算量**：2.7e24 FLOPs
- **阶段**：预训练 80% + 中期训练 20%，后训练随后

scaling ladder（用于调试和预测）：
1. 1.6B-A61M（48B tokens）
2. 27.7B-A1.2B（926B tokens）
3. 535B-A23B（18.75T tokens）← hero run

全部实时公开：
- 数据构成：storage.googleapis.com/marin-public/...
- 实时训练：wandb.ai/marin-community/marin_moe
- 完整细节：github.com/marin-community/marin/issues/8435

### 为什么重要

当前大模型开发的黑箱问题是：OpenAI/Anthropic 只发布最终结果，不公开失败实验和试错过程。Marin 试图证明：完全透明的开发流程也能产出高质量模型。

Marin 还提供分析工具：
- Perplexity Gap Dashboard（困惑度差距看板）
- PPL Circuit Coverage Dashboard
- Scaling Laws Explorer（Scaling Law 探索器）
- Speedrun Leaderboard（训练算法速度排行榜：谁能最快降低 validation loss）

### CS336：Language Models from Scratch

Percy Liang 和 Tatsunori Hashimoto 共同教授 CS336，这是 Stanford 最火的课之一。

课程理念：类比操作系统课让学生从零写一个 OS，CS336 让学生从零构建一个语言模型——数据收集清洗、Transformer 构建、训练、评估、部署全流程。

- 2024 年首次开设，已开 3 届（2024/2025/2026 Spring）
- YouTube 有完整课程录像
- 成为全球学习 LLM 工程的事实标准课程
- 课程产出工具如 Gigatoken（500-1000x 快于 HuggingFace，100x 快于 tiktoken 的分词器）

---

## 4. Simile：模拟人类行为的基础模型

Percy Liang 的博士生 Joon Sung Park（2025 年毕业，共同导师 Michael Bernstein）创立了 Simile，Percy Liang 深度参与。

### 使命

> "Our mission is to simulate all eight billion people on earth, accurately and honestly."
>
> "Our ambition is to build a foundation model that can predict what anyone will do in any situation."

Simile 要建一个能预测"任何人在任何情况下会做什么"的基础模型——本质是用 LLM 模拟人类行为。

### Percy Liang 的定位

2026.07.30 推文：

> "It's time to scale that Simile. Our ambition is to build a foundation model that can predict what anyone will do in any situation. Research-wise, Simile in 2026 for simulation is like OpenAI in 2019 for AI assistants - signs of life, a path to scaling, yet still many open [questions]."

他把 Simile 在 2026 年的位置类比为 OpenAI 在 2019 年——已看到希望，有 scaling 路径，但还有很多开放问题。

### 融资与商业化

- **Series B**：$200M，$2B 估值（2026.07.30）
- **领投**：Greenoaks、Index Ventures
- **跟投**：Hanabi、Bain Capital Ventures、A*、Factory、CVS Health Ventures、Definition
- **成立仅 5 个月**，收入增长 5 倍，50+ 员工
- **客户**：CVS Health、Gallup、Itaú、Deloitte、Wealthfront、Suntory、Garnett Station Partners
- **产品**：人类行为基础模型 + 置信度模型（预测每次模拟的准确度）+ 预测未来的产品

### 技术路线

从 Simile 官网和推文推断：
- 基于角色/人口属性生成模拟人类行为
- 置信度模型：预测每次模拟的准确性（可验证的预测）
- 已运行数千万次模拟
- 面向企业：产品发布策略、客户体验优化、新市场进入

### 与 Generative Agents 的关系

Simile 的技术源头是 Joon Sung Park 在 Stanford 的工作——Generative Agents（生成式智能体，用 LLM 模拟人类行为的小镇实验）。Percy Liang 是这个方向的推动者之一。Simile 把学术研究变成了商业产品。

---

## 5. CRFM：Stanford 基础模型研究中心

Percy Liang 创建并领导的 CRFM（Center for Research on Foundation Models）是 Stanford HAI 下的跨学科中心。

### 四大支柱

1. **Technical foundations（技术基础）**：数据、系统、架构、训练、适配、推理、可解释性、评估、新范式
2. **Beneficial applications（有益应用）**：法律、音乐、机器人、生物医学等垂直领域
3. **Societal impact（社会影响）**：透明度、供应链、开放性、版权、隐私、系统性风险
4. **AI policy（AI 政策）**：支持多辖区政府制定循证政策

### CRFM 的项目

- **HELM**：评估体系
- **Levanter**：开源模型训练框架
- **FMTI**：透明度指数
- **Ecosystem Graphs**：模型生态图谱
- **Openness**：开放性研究
- **Policy**：政策支持

---

## 6. Percy Liang 的学术谱系

他的学生和博后遍布 AI 领域核心位置，影响力辐射极广。

### 当前学生/博后（2026）

| 姓名 | 共同导师 | 研究方向 |
|------|---------|---------|
| Rohith Kuditipudi | John Duchi | 模型溯源 |
| Steven Cao | Gregory Valiant | 理论 |
| Ahmed Ahmed | Sanmi Koyejo | 记忆化/安全 |
| Moo Jin Kim | Chelsea Finn | 机器人/VLA |
| Ken Liu | Sanmi Koyejo | 评估 |
| Suhas Kotha | Tatsunori Hashimoto | 预训练/微调 |
| Anne Ouyang | Azalia Mirhoseini | 系统 |
| Kaiyue Wen | Tengyu Ma | 优化器 |
| Marina Zhang | Carlos Guestrin | |
| Tanishq Kumar | Tatsunori Hashimoto | Scaling Laws for Precision |
| Tony Lee | | HELM/多模态评估 |
| Michael Ryan | Diyi Yang | |
| A. Feder Cooper (博后) | Dan Ho | 记忆化/版权 |
| Morris Yau (博后) | Chris Ré | |

### 知名校友

| 姓名 | 毕业时间 | 现在位置 |
|------|---------|---------|
| Tatsunori Hashimoto | 博后 2019 | Stanford 教授（共同授课 CS336） |
| Jacob Steinhardt | 博士 2018 | UC Berkeley 教授，Transluce 创始人 |
| Rishi Bommasani | 博士 2025 | Stanford HAI 高级研究学者 |
| Joon Sung Park | 博士 2025 | Simile 创始人/CEO（Multi-Agent 仿真） |
| Sherry Yang | 博后 2025 | NYU 教授 |
| Xiang Lisa Li | 博士 2025 | UW 教授 |
| Nelson Liu | 博士 2025 | OpenAI |
| Yann Dubois | 博士 2025 | OpenAI |
| Sidd Karamcheti | 博士 2025 | Georgia Tech 教授 |
| Shiori Sagawa | 博士 2024 | OpenAI |
| John Hewitt | 博士 2024 | Columbia 教授 |
| Mina Lee | 博士 2023 | U Chicago 教授 |
| Aditi Raghunathan | 博士 2021 | CMU 教授 |
| Pang Wei Koh | 博士 2022 | UW 教授 |
| Robin Jia | 博士 2020 | USC 教授 |
| Pranav Rajpurkar | 博士 2021 | Harvard 教授 |
| Chris Donahue | 博后 2022 | CMU 教授 |
| Sarah Cen | 博后 2025 | CMU 教授 |
| Roy Frostig | 博士 2016 | Google，JAX 共同创建者 |
| Yuhuai (Tony) Wu | 博后 2023 | xAI 联合创始人 |

---

## 7. 近期研究动态（2025-2026 最新）

从 arXiv 最新论文看，Percy Liang 当前最活跃的方向：

### 训练技术

- **Hyperball Optimizer** (2026.06, arXiv:2606.16899)：Muon 等矩阵优化器在大规模时收益缩小的分析与改进
- **Synthetic Megadocs** (2026.03, arXiv:2603.18534)：用合成大文档做数据高效的预训练
- **Replaying Pre-training Data** (2026.03, arXiv:2603.04964)：微调时回放预训练数据提升效果
- **Self-Verified Distillation** (2026.05, arXiv:2605.26132)：模型用未标注 prompt 自我蒸馏，不需要外部教师

### 经济与社会

- **Economic Evaluations** (2026.06, arXiv:2607.19375)：评估语言模型执行经济价值任务的能力
- **Algorithmic Monocultures in Hiring** (2026.05, arXiv:2605.27371)：招聘算法单一文化的影响
- **FLARE-AI** (2026.06, arXiv:2606.31567)：AI 缺陷报告生态系统

---

## 8. Percy Liang 的独特定位

在整个 AI 学术界，Percy Liang 的定位很独特：

1. **他既是技术研究者也是政策推动者**：不只是发论文，FMTI 直接影响政策，AI 审计/版权研究直接介入法律争议
2. **他建立了评估基础设施**：HELM 系列成为行业标准，不是单篇论文而是持续运营的平台
3. **他践行"激进开放"**：Marin 不只是开放权重，是开放整个开发过程。Marin 535B-A23B 训练实时在 wandb 上公开，这在业界是独一无二的
4. **他通过教育扩大影响**：CS336 培养了大量懂 LLM 工程的学生
5. **他的学术谱系覆盖了半个 AI 学术界**：学生分布在 Stanford/Berkeley/CMU/UW/USC/OpenAI/Google/xAI 等核心位置
6. **他通过创业扩大影响**：Simile 把 Generative Agents 研究变成商业产品，5 个月做到 $2B 估值；Marin-8B 在 Together AI 上部署

与唐杰的对比：
- 唐杰：从学术到工业，GLM 系列是产品，Scaling Law 是技术主张，核心是"参数到阈值后转 post-training 旋钮"
- Percy Liang：从学术到基础设施+政策+开放生态，HELM/FMTI/Marin 是公共产品，不追求自己的模型超越 GPT，而是让整个生态更透明更可评估。但他也在通过 Marin 535B 证明：开放流程也能训练前沿模型

与 Ilya Sutskever 的对比：
- Ilya：从 Scale 转向持续学习（TTT），走神秘路线，SSI 不公开技术细节
- Percy Liang：走完全相反的路线——最大化透明，Marin 把所有失败实验都公开，用开放对抗黑箱

---

## 9. 与已有文档的交叉引用

- **[ScalingLaws.md](../4.text-model/ScalingLaws.md)**：Percy Liang 的学生 Tanishq Kumar 是 "Scaling Laws for Precision"（arXiv:2411.04330）的第一作者，那篇论文正是我们 ScalingLaws.md 第 6 章的核心论文
- **[AGI-Feynman.md](../8.tutorial/AGI-Feynman.md)**：Percy Liang 的"Foundation Model"定义和 FMTI 透明度运动，是 AGI 治理层面的重要工作
- **[PhysicalAI-Feynman.md](../8.tutorial/PhysicalAI-Feynman.md)**：Percy Liang 近期在 VLA 方向的工作（VLAW/RoboReward/Cosmos Policy）与 Physical AI 直接相关
- **[SelfEvolving.md](../7.product/SelfEvolving.md)**：Self-Verified Distillation（arXiv:2605.26132）是自进化方向的最新工作——模型只用未标注 prompt 自我提升

---

## 10. 论文索引（精选 40 篇）

| # | 论文 | arXiv | 时间 | 方向 |
|---|------|-------|------|------|
| 1 | On the Opportunities and Risks of Foundation Models | 2108.07258 | 2021.08 | Foundation Model 定义 |
| 2 | Holistic Evaluation of Language Models (HELM) | 2211.09110 | 2022.11 | 评估 |
| 3 | Picking on the Same Person (Algorithmic Monoculture) | 2211.13972 | 2022.11 | 社会影响 |
| 4 | Deep Bidirectional Language-Knowledge Graph Pretraining | 2210.09338 | 2022.10 | 技术 |
| 5 | Ecosystem Graphs: The Social Footprint of Foundation Models | 2303.15772 | 2023.03 | 生态 |
| 6 | Foundation Models and Fair Use | 2303.15715 | 2023.03 | 版权 |
| 7 | FlexGen: High-Throughput LLM Inference | 2303.06865 | 2023.03 | 推理 |
| 8 | Holistic Evaluation of Text-To-Image Models | 2311.04287 | 2023.11 | 评估 |
| 9 | The Foundation Model Transparency Index (FMTI) | 2310.12941 | 2023.10 | 透明度 |
| 10 | Designing and Interpreting Probes with Control Tasks | 1909.03368 | 2019.09 | 可解释性 |
| 11 | Why Overparameterization Exacerbates Spurious Correlations | 2005.04345 | 2020.05 | 公平性 |
| 12 | The 2024 Foundation Model Transparency Index | 2407.12929 | 2024.07 | 透明度 |
| 13 | Foundation Model Transparency Reports | 2402.16268 | 2024.02 | 透明度 |
| 14 | On the Societal Impact of Open Foundation Models | 2403.07918 | 2024.02 | 社会影响 |
| 15 | The Responsible Foundation Model Development Cheatsheet | 2406.16746 | 2024.06 | 开发实践 |
| 16 | AIR-Bench 2024: Safety Benchmark | 2407.17436 | 2024.07 | 安全 |
| 17 | Language model developers should report train-test overlap | 2410.08385 | 2024.10 | 透明度 |
| 18 | VHELM: Holistic Evaluation of Vision Language Models | 2410.07112 | 2024.10 | 评估 |
| 19 | Image2Struct: Benchmarking Structure Extraction for VLMs | 2410.22456 | 2024.10 | 评估 |
| 20 | Decentralized Training of Foundation Models | 2206.01288 | 2022.06 | 训练 |
| 21 | Fine-Tuning Vision-Language-Action Models | 2502.19645 | 2025.02 | 机器人 |
| 22 | Auditing Prompt Caching in Language Model APIs | 2502.07776 | 2025.02 | 安全 |
| 23 | AILuminate: AI Risk and Reliability Benchmark | 2503.05731 | 2025.02 | 安全 |
| 24 | In-House Evaluation Is Not Enough | 2503.16861 | 2025.03 | 安全 |
| 25 | Reliable and Efficient Amortized Model-based Evaluation | 2503.13335 | 2025.03 | 评估 |
| 26 | Extracting memorized pieces of (copyrighted) books | 2505.12546 | 2025.05 | 记忆化/版权 |
| 27 | MLE-Dojo: Interactive Environments for LLM Agents | 2505.07782 | 2025.05 | Agent |
| 28 | BountyBench: AI Agent Attackers and Defenders | 2505.15216 | 2025.05 | 安全 |
| 29 | MedHELM: Holistic Evaluation for Medical Tasks | 2505.23802 | 2025.05 | 评估 |
| 30 | RoboArena: Distributed Real-World Robot Evaluation | 2506.18123 | 2025.06 | 机器人 |
| 31 | WorldGym: World Model as Environment for Policy Evaluation | 2506.00613 | 2025.06 | 机器人 |
| 32 | Best Practices for Agentic Benchmarks | 2507.02825 | 2025.07 | 评估 |
| 33 | UQ: Assessing Language Models on Unsolved Questions | 2508.17580 | 2025.08 | 评估 |
| 34 | AHELM: Holistic Evaluation of Audio-Language Models | 2508.21376 | 2025.08 | 评估 |
| 35 | SpecEval: Model Adherence to Behavior Specifications | 2509.02464 | 2025.09 | 安全 |
| 36 | RL for Machine Learning Engineering Agents | 2509.01684 | 2025.09 | Agent |
| 37 | AI Audits Under Resource Constraints | 2509.05627 | 2025.09 | 安全 |
| 38 | HAL: Holistic Agent Leaderboard | 2510.11977 | 2025.10 | 评估 |
| 39 | MLE-Smith: Scaling MLE Tasks | 2510.07307 | 2025.10 | Agent |
| 40 | Blackbox Model Provenance | 2510.19796 | 2025.10 | 记忆化 |
| 41 | Comparing AI Agents to Cybersecurity Professionals | 2512.09882 | 2025.12 | 安全 |
| 42 | The 2025 Foundation Model Transparency Index | 2512.10169 | 2025.12 | 透明度 |
| 43 | Self-Verified Distillation | 2605.26132 | 2026.05 | 训练 |
| 44 | Algorithmic Monocultures in Hiring | 2605.27371 | 2026.05 | 社会影响 |
| 45 | DTap: Red-Teaming Platform for AI Agents | 2605.04808 | 2026.05 | 安全 |
| 46 | FLARE-AI: Flaw Reporting for AI | 2606.31567 | 2026.06 | 安全 |
| 47 | Hyperball Optimizer | 2606.16899 | 2026.06 | 训练 |
| 48 | Economic Evaluations of Language Models | 2607.19375 | 2026.06 | 评估 |
| 49 | Extractable Memorization From First Principles | 2607.12649 | 2026.07 | 记忆化 |
| 50 | DelusionEval: Measuring Delusion-Linked Behaviors | 2608.05004 | 2026.08 | 安全 |
| 51 | Extracting books from production language models | 2601.02671 | 2026.01 | 记忆化/版权 |
| 52 | Estimating near-verbatim extraction risk | 2603.24917 | 2026.03 | 记忆化 |
| 53 | Synthetic Megadocs for Pre-training | 2603.18534 | 2026.03 | 训练 |
| 54 | Replaying pre-training data improves fine-tuning | 2603.04964 | 2026.03 | 训练 |
| 55 | VLAW: VLA Policy and World Model Co-Improvement | 2602.12063 | 2026.02 | 机器人 |
| 56 | Cosmos Policy: Fine-Tuning Video Models for Robotics | 2601.16163 | 2026.01 | 机器人 |
| 57 | RoboReward: Vision-Language Reward Models for Robotics | 2601.00675 | 2026.01 | 机器人 |
