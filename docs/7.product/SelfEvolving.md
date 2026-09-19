# 自进化 Agent (Self-Evolving Agents)

> 核心命题：AI 利用自身能力、经验、产物、评价或研究结果，修改产生这些能力的系统，使后续系统不仅在目标任务上更强，而且具备更强的进一步改进自身的能力。
>
> 关键词：self-evolving、recursive self-improvement (RSI)、skill evolution、harness evolution、experience-driven learning
>
> 参考：[知乎 RSI 研究报告](https://zhuanlan.zhihu.com/p/2057943837826348906)（2026.08，证据截止 2026-08-11）

---

## 一、RSI 分级体系

> 不是所有"self-improvement"都是 RSI。核心区分：improvement mechanism 本身是否进入优化闭环。

| 级别 | 优化对象 | 典型机制 | RSI 程度 |
|------|---------|---------|---------|
| L0 | Output（输出） | Self-Refine / Reflection | 非 RSI |
| L1 | Memory / Skill / Context | Reflexion / ACE / Skill Evolution | 弱 RSI |
| L2 | Harness / Workflow | Harness Evolution / DGM | RSI-like |
| L3 | Training / Policy / Weights | Self-training / Self-play / TTT | 强 RSI |
| L4 | AI Research Process | AutoResearch / AI Scientist | 更强 RSI |
| L5 | Improvement Mechanism | Meta-improvement / Hyperagents | 真正递归 |
| L6 | Successor Model | AI designs/trains/evaluates successor | Strict RSI |
| L7 | Open-ended Intelligence Explosion | 持续、开放式、自加速 | 尚未证明 |

### Broad RSI vs. Strict RSI

- **Broad RSI**：AI 利用自身产生的信息/经验/代码/评价/训练数据改善下一轮 AI 系统。已广泛存在。
- **Strict RSI**：AI 能产生改进 → 改进被客观验证 → 改进进入下一代 → 下一代拥有更强 improvement capability → 循环持续不依赖人工。尚未公开证明。

### RSI 的五条路线

```
                    Recursive Self-Improvement
                              │
       ┌──────────────┬───────┼────────┬──────────────┐
       │              │       │        │              │
   Model Loop     Agent Loop  Research  Evolution   Experience
       │              │       │        │              │
  Self-training   Harness    AI Sci.   DGM        RL / Self-play
  Self-reward     Memory     AutoRes.   AlphaEvolve World Models
  Test-time RL    Skills     AutoResearch DGM-H     Continual RL
       │              │       │        │              │
       └──────────────┴───────┴────────┴──────────────┘
                              │
                       Meta-Improvement
                              │
                    Improve the Improver
```

---

## 二、经典研究谱系

### 2.1 Gödel Machine（2003）— 理论祖先

- 论文：[Schmidhuber, Gödel Machines](https://people.idsia.ch/~juergen/gmweb3/gmweb3.html)（2003）
- 核心思想：self-referential universal problem solver，可修改自己的程序，但只有当内部 proof searcher 能证明修改会提高未来 utility 时才执行
- 与现代 DGM 的区别：Gödel Machine = proof-driven；DGM = empirical-evaluation-driven

### 2.2 STaR（2022）— 自生成推理训练

- 论文：[arXiv:2203.14465](https://arxiv.org/abs/2203.14465)（Zelikman et al., 2022）
- 机制：模型生成 reasoning → 检查答案 → 对错误答案用正确答案重新生成 reasoning → 加入训练集 → fine-tune → 重复
- 意义：模型第一次比较明确地参与了自己的 training-data generation

### 2.3 Self-Refine（2023）— 推理时迭代

- 论文：[arXiv:2303.17651](https://arxiv.org/abs/2303.17651)（Madaan et al., 2023）
- 机制：Generate → Feedback → Refine 循环，同一 LLM 同时负责生成、评价和修改
- 结果：七类任务平均约 20pp 绝对提升
- 局限：模型权重不变，不是严格 RSI

### 2.4 Reflexion（2023）— 语言强化学习

- 论文：[arXiv:2303.11366](https://arxiv.org/abs/2303.11366)（Shinn et al., 2023）
- 机制：Act → Environment feedback → Verbal reflection → Episodic memory → Next trial
- 结果：HumanEval 91% pass@1
- 意义：Agent memory learning 的祖先之一

### 2.5 Voyager（2023）— 终身技能积累

- 论文：[arXiv:2305.16291](https://arxiv.org/abs/2305.16291)（Wang et al., 2023）
- 机制：自动 curriculum + 自动探索 + 自动产生 executable skills + skill library + 环境反馈修改程序
- 意义：Agent 可以不改变 model weights 也能产生真正持久的能力增长
- 代码：[github.com/MineDojo/Voyager](https://github.com/MineDojo/Voyager)

### 2.6 Eureka（2023）— 自动奖励设计

- 论文：[arXiv:2310.12931](https://arxiv.org/abs/2310.12931)（Ma et al., 2023）
- 机制：LLM 自动编写 reward function → RL 训练 → 评估 → 反思 → 更好的 reward code
- 结果：29 个 RL 环境中 83% 超过 human-engineered reward，平均 52% normalized improvement
- 意义：AI 开始优化"训练 AI 的程序"

### 2.7 ExpeL（2023）— 经验学习

- 论文：[arXiv:2308.10379](https://arxiv.org/abs/2308.10379)（Zhao et al., 2023）
- 核心思想：LLM Agent 从成功和失败轨迹中抽取经验，形成可复用的 insights

### 2.8 Self-Rewarding Language Models（2024）

- 论文：[arXiv:2401.10020](https://arxiv.org/abs/2401.10020)（Yuan et al., 2024）
- 机制：模型自己作为 judge 评估自己的输出，用偏好对做 DPO 训练
- 意义：self-reward 闭环的早期工作

### 2.9 Absolute Zero（2025）— 零数据自博弈

- 论文：[arXiv:2505.03335](https://arxiv.org/abs/2505.03335)（Zhao et al., 2025）
- 核心思想：一个模型同时扮演 proposer 和 solver，零外部数据实现 self-play reasoning

### 2.10 Gödel Agent（2024）— 自指框架

- 论文：[arXiv:2410.04444](https://arxiv.org/abs/2410.04444)（Shi et al., 2024）
- 核心思想：受 Gödel machine 启发，Agent 动态修改自己的逻辑和行为，仅受高层目标引导
- 结果：数学推理和复杂 Agent 任务上持续自我改进，超越人工设计 Agent

---

## 三、现代 RSI 核心工作（2024-2026）

### 3.1 The AI Scientist（2024→2026）— 自动化科研

- 论文：[arXiv:2408.06292](https://arxiv.org/abs/2408.06292)（Sakana AI, 2024）
- v2：[arXiv:2504.08066](https://arxiv.org/abs/2504.08066)（2025，agentic tree search + AI-generated paper 通过 ICLR workshop peer review）
- 2026.03：正式发表于 Nature
- 流程：Idea → Code → Experiment → Analysis → Paper → Review

### 3.2 Darwin Gödel Machine（2025）— 进化式自改进

- 论文：[arXiv:2505.22954](https://arxiv.org/abs/2505.22954)（Zhang et al., 2025）
- 核心思想：Agent 不仅解决任务，还可修改自己的代码；维护进化 archive，保留优秀变体
- 结果：SWE-bench 20%→50%，Polyglot 14.2%→30.7%
- 核心论点："DGM improves not only the solution, but the agent that produces solutions."

### 3.3 AlphaEvolve（2025）— Google DeepMind

- 博客：[Google DeepMind](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
- 机制：Gemini 生成程序 → 自动评估器打分 → 进化数据库 → 选择优秀程序 → 生成下一代
- 应用：数据中心优化、芯片设计、AI 训练优化、矩阵乘法、数学问题
- 关键：官方明确表示 AlphaEvolve 优化过用于训练 Gemini 模型本身的 processes

### 3.4 Agentic Context Engineering（2025-2026）

- ACE：[HuggingFace](https://huggingface.co/papers/2510.04618)（Zhang et al., 2025）
  - Context 作为可进化 playbook：Generator → Reflector → Curator
- Meta Context Engineering (MCE)：[arXiv:2601.21557](https://arxiv.org/abs/2601.21557)（Ye et al., 2026）
  - 双层优化：meta-level skill evolution + base-level context optimization
  - 结果：多个 domain 上 5.6%-53.8% relative improvement

### 3.5 Agentic Harness Engineering（2026）

- 论文：[arXiv:2604.25850](https://arxiv.org/abs/2604.25850)（Lin et al., 2026）
- 三大可观察性：Component / Experience / Decision
- 流程：Edit → Prediction → Execution → Measurement → Verification
- 结果：Terminal-Bench 2 pass@1 十轮迭代从 69.7% 提升到 77.0%
- 同一冻结 harness 迁移到 SWE-bench-verified 仍有效

### 3.6 Meta-Harness（2026）

- 论文：[arXiv:2603.28052](https://arxiv.org/abs/2603.28052)（Lee et al., 2026）
- 优化对象是"决定如何优化 harness 的代码"
- Proposer 本身是 coding agent，输出 Pareto frontier 上的 harness 候选

### 3.7 MetaSkill-Evolve（2026）— 递归元技能进化

- 论文：[arXiv:2607.05297](https://arxiv.org/abs/2607.05297)（2026.07）
- 核心突破：同时优化 Task Skill 和"如何改进 Skill"的 Meta-Skill
- 结果：OfficeQA +23.54pp，SealQA +16.09pp，ALFWorld +1.92pp
- 意义：从 self-improvement 走向 recursive self-improvement 的结构性变化

### 3.8 Hyperagents（2026）— 元 Agent 自改进

- 论文：[arXiv:2603.19461](https://arxiv.org/abs/2603.19461)（Zhang et al., 2026）
- 核心思想：解耦 task agent 和 meta agent，meta-level modification procedure 本身可编辑
- 路线：Task Solver ← Meta Solver ← Meta Meta Solver

### 3.9 Continual Harness（2026）— 在线持续适应

- 论文：[arXiv:2605.09998](https://arxiv.org/abs/2605.09998)（Karten et al., 2026）
- 场景：长周期游戏中的 harness 更新 + 策略模型协同学习
- 特点：online、reset-free

### 3.10 Self-Harness（2026）— 自我弱项挖掘

- 论文：[arXiv:2606.09498](https://arxiv.org/abs/2606.09498)（Zhang et al., 2026）
- 三阶段：Weakness mining → Bounded harness proposal → Validation
- 在 MiniMax M2.5、Qwen3.5-35B-A3B、GLM-5 上学到模型特定的 harness 指令

---

## 四、RSI 综述与理论

### 4.1 RSI 综述：从有界自精炼到自主研究循环

- 论文：[arXiv:2607.07663](https://arxiv.org/abs/2607.07663)（2026.07）
- 统计 2024-2026 年约 1,250 篇 arXiv 论文
- 两个分类轴：系统改进什么（behavior/policy/evaluator/research process）× 闭环程度（human-in-loop → fully closed）
- 核心贡献：验证层级体系（formal verifier 最强 → intrinsic self-assessment 最弱）
- 发现：自改进强度跟验证层级正相关，失败模式（self-confirming loop、model collapse、diversity collapse）源于违反验证层级

### 4.2 自进化 Agent 综述：What, When, How, Where

- 论文：[arXiv:2507.21046](https://arxiv.org/abs/2507.21046)（2025.07）
- 三个维度：what to evolve（model/memory/tools/architecture）、when to evolve（intra/inter-test-time）、how to evolve（rewards/feedback/single/multi-agent）

### 4.3 自进化 AI Agent 综合综述

- 论文：[arXiv:2508.07407](https://arxiv.org/abs/2508.07407)（2025.08）
- 统一框架：System Inputs → Agent System → Environment → Optimisers
- 覆盖领域专用进化策略（生物医学、编程、金融）

### 4.4 自进化编程 Agent 综述

- 论文：[arXiv:2608.03392](https://arxiv.org/abs/2608.03392)（2026.08）
- 对象中心分类法：what evolves（framework/memory/skills/tools/models/collaboration）
- 论文合集：[github.com/zhouhao1024/Awesome-Self-Evolving-Coding-Agents](https://github.com/zhouhao1024/Awesome-Self-Evolving-Coding-Agents)

### 4.5 Agent Memory 综述：第二半场

- 论文：[arXiv:2602.06052](https://arxiv.org/abs/2602.06052)（2026.01）
- 记忆作为自进化的 substrate：short-term memory gates perception，long-term memory consolidates into reusable knowledge

### 4.6 内省阈值理论

- 论文：[arXiv:2607.04277](https://arxiv.org/abs/2607.04277)（2026.07）
- 类比 von Neumann 自复制自动机复杂度阈值，论证 LLM 需要功能内省才能实现可持续 RSI
- 当前 LLM 只有 quasi-introspection，缺少 complete self-access 和 fixed-point iteration

---

## 五、技能进化 (Skill Evolution)

| 论文 | 年份 | 关键贡献 | arXiv |
|------|------|---------|-------|
| MetaSkill-Evolve | 2026.07 | 双时间尺度元技能递归进化 | [2607.05297](https://arxiv.org/abs/2607.05297) |
| SkillZip | 2026.08 | 无评估技能压缩，发现可复用结构 | [2608.11079](https://arxiv.org/abs/2608.11079) |
| SkillHEX | 2026.08 | 假设驱动自主探索与利用 | [2608.05628](https://arxiv.org/abs/2608.05628) |
| SkillAudit | 2026.06 | 无 ground-truth 的配对轨迹审计 | [2606.14239](https://arxiv.org/abs/2606.14239) |
| SkillProx | 2026.08 | 近端文本梯度下降的技能进化 | [2608.07449](https://arxiv.org/abs/2608.07449) |
| SoK: Agentic Skills | 2026.02 | 技能形式化分类，超越 tool use | [2602.20867](https://arxiv.org/abs/2602.20867) |
| Learning Globally Reusable Skills | 2026.08 | 编程 Agent 全局可复用技能 | [2608.06153](https://arxiv.org/abs/2608.06153) |
| Training Skills Like Parameters | 2026.07 | 自监督语义扩散训练技能 | [2607.27557](https://arxiv.org/abs/2607.27557) |
| Search2Skill | 2026.08 | Rubric RL 的技能蒸馏 | [2608.05245](https://arxiv.org/abs/2608.05245) |
| Emotion2Skill | 2026.08 | 模型内部情绪信号驱动技能选择 | [2608.09248](https://arxiv.org/abs/2608.09248) |

---

## 六、记忆驱动进化 (Memory-Driven Evolution)

| 论文 | 年份 | 关键贡献 | arXiv |
|------|------|---------|-------|
| CrystalMem | 2026.07 | 知识结晶化的弹性记忆 | [2608.00303](https://arxiv.org/abs/2608.00303) |
| Tree-of-Experience | 2026.08 | 层次化经验管理 | [2608.09044](https://arxiv.org/abs/2608.09044) |
| RoMeRL | 2026.08 | 记忆-奖励陷阱平衡，降阶效用状态 | [2608.02508](https://arxiv.org/abs/2608.02508) |
| Authority Collapse | 2026.08 | 记忆合并边界处的权威崩溃 | [2608.01679](https://arxiv.org/abs/2608.01679) |
| Memory→Skills Co-Evolution | 2026.07 | 证据驱动的记忆-技能协同进化治理 | [2607.16621](https://arxiv.org/abs/2607.16621) |
| Memory Reward Inflation | 2026.06 | 自改进中的记忆奖励膨胀 | [2608.00017](https://arxiv.org/abs/2608.00017) |
| Critic Experience Bank | 2026.07 | 步级置信度估计的自进化 | [2607.12397](https://arxiv.org/abs/2607.12397) |

---

## 七、Harness 与框架进化 (Harness & Framework Evolution)

| 论文 | 年份 | 关键贡献 | arXiv |
|------|------|---------|-------|
| Agentic Harness Engineering | 2026.04 | 三大可观察性驱动的 harness 自动进化 | [2604.25850](https://arxiv.org/abs/2604.25850) |
| Meta-Harness | 2026.03 | harness 优化器的优化 | [2603.28052](https://arxiv.org/abs/2603.28052) |
| Self-Harness | 2026.06 | 自我弱项挖掘的 propose-validate 循环 | [2606.09498](https://arxiv.org/abs/2606.09498) |
| Continual Harness | 2026.05 | 在线 reset-free harness 适应 | [2605.09998](https://arxiv.org/abs/2605.09998) |
| EvoHarness-RL | 2026.08 | RL 学习自进化运行时 harness | [2608.05446](https://arxiv.org/abs/2608.05446) |
| Living-Harness | 2026.07 | 交互式 Agent 进化器 | [2607.26598](https://arxiv.org/abs/2607.26598) |
| HarnessBank | 2026.07 | 语义基因库搜索 + 门控验证 | [2607.13683](https://arxiv.org/abs/2607.13683) |
| MEGA | 2026.08 | 智慧图驱动的 Agent 优化基础设施 | [2608.10504](https://arxiv.org/abs/2608.10504) |
| Phantom Guardrails | 2026.07 | 修复从未发生的失败（假阳性修复） | [2607.13083](https://arxiv.org/abs/2607.13083) |

---

## 八、安全与攻击 (Safety & Security)

| 论文 | 年份 | 关键贡献 | arXiv |
|------|------|---------|-------|
| SkillJack | 2026.08 | 自进化 Agent 中的持久技能后门 | [2608.03509](https://arxiv.org/abs/2608.03509) |
| Trajectory Poisoning | 2026.08 | 经验变指令的轨迹投毒攻击 | [2608.05563](https://arxiv.org/abs/2608.05563) |
| Query-Only Backdoor | 2026.08 | 仅查询的后门攻击 | [2608.08303](https://arxiv.org/abs/2608.08303) |
| Benign Alone, Harmful Together | 2026.08 | 经验组合利用 | [2608.01759](https://arxiv.org/abs/2608.01759) |
| Self-Evolution Backfires | 2026.08 | 技能污染的 pre-commit 门控 | [2608.05810](https://arxiv.org/abs/2608.05810) |
| ActBench | 2026.08 | Cowork Agent 行为安全自进化基准 | [2608.09476](https://arxiv.org/abs/2608.09476) |
| Agent Skill Security | 2026.07 | 技能安全威胁模型综合 | [2607.13987](https://arxiv.org/abs/2607.13987) |
| Yesterday's Shield | 2026.08 | 自进化安全护栏的生产实践 | [2608.08471](https://arxiv.org/abs/2608.08471) |
| Recursive Self-Training Collapse | 2026.06 | AI 审查自己代码时的 rubber-stamp 退化 | [2606.28438](https://arxiv.org/abs/2606.28438) |

---

## 九、评估基准 (Benchmarks)

| 论文 | 年份 | 评估维度 | arXiv |
|------|------|---------|-------|
| PAST-Bench | 2026.08 | 个人 Agent 的 RSI 基础能力 | [2608.04003](https://arxiv.org/abs/2608.04003) |
| RSIBench-Data | 2026.07 | 数据中心的 RSI 研究 | [2607.25886](https://arxiv.org/abs/2607.25886) |
| AgentStream | 2026.07 | 流式任务下的自进化表现 | [2608.00155](https://arxiv.org/abs/2608.00155) |
| FinEvo-Bench | 2026.08 | 金融工作流纵向基准 | [2608.06144](https://arxiv.org/abs/2608.06144) |
| Who Grades the Grader? | 2026.07 | 评估指标与技能协同进化 | [2607.12790](https://arxiv.org/abs/2607.12790) |

---

## 十、前沿实验室 RSI 实践

> 证据分级：A = 官方公开 + 可复现论文/实验；B = 公司公开 thesis/early result

| 机构 | 公开 RSI 证据 | 主要方向 | 证据等级 |
|------|-------------|---------|---------|
| Anthropic | 明确（"When AI builds itself"） | AI 开发 AI / coding agents | A |
| OpenAI | 明确但有限 | 自改进 tax agents / GPT-Red self-play | A |
| Google DeepMind | 强间接 | AlphaEvolve / 算法进化 | A |
| Sakana AI | 明确 | AI Scientist / DGM / RSI Lab | A |
| Recursive | 明确 | 自动 AI 研究循环 | B+ |
| Weco AI | 明确 | AIDE²（autoresearch on autoresearch） | B |
| Inherent | 明确 thesis | 递归研究组织 | B |
| Ricursive Intelligence | 明确 thesis | AI ↔ 硬件协同进化 | B |
| Ineffable Intelligence | 明确 thesis | 经验驱动持续 RL（David Silver） | B |

### 关键事实

- **Anthropic**：明确表示 "We are not there yet"；AI 完成任务时间跨度约每四个月翻倍；Claude Opus 4.6 达到约 12 小时级 software tasks
- **OpenAI GPT-Red**：self-play 自动 red-team，84% attack success（人类 13%），产生的攻击数据用于训练 GPT-5.6
- **Google DeepMind AlphaEvolve**：优化过用于训练 Gemini 模型本身的 processes — AI improving infrastructure used to build AI
- **Weco AI AIDE²**：8 天内发现比人工构建两年版本更好的 autoresearch harness，宣称 RSI Level-1
- **Sakana AI**：AI Scientist + DGM + Digital Red Queen 组合放入 RSI Lab；AI Scientist 已发表于 Nature

---

## 十一、RSI 失败模式

### 11.1 Model Collapse
- 模型不断使用自身生成数据训练，缺少外部信息 → 分布坍缩
- AI 自己作为 reviewer 时可能逐渐进入"rubber-stamp"状态（接受率上升但正确率下降）

### 11.2 Self-Confirming Loop
- Evaluator ≈ Generator 时：生成坏想法 → 自评"不错" → 训练坏数据 → 生成更差想法
- 核心问题：self-evaluation ≠ reliable evaluation

### 11.3 Reward Hacking
- Agent 操纵 evaluator → Score↑ 但 Capability↔
- 所有严肃 RSI 系统都需 automated validation、variance checking、reward-hack detection

### 11.4 Benchmark Overfitting
- Agent 提升 benchmark ≠ 提升通用智能
- 严格 RSI benchmark 应测试：In-Distribution / OOD / Cross-Model / Cross-Task / Cross-Evaluator

### 11.5 Diversity Collapse
- 进化和 RL 循环倾向于利用已知高奖励模式
- 需要机制防止种群坍缩为同一方案的变体

---

## 十二、RSI 的核心工程原则

### 12.1 Evaluator > Generator

RSI 真正的瓶颈不是"AI 能否产生更好的想法"，而是"AI 能否可靠地知道想法是否更好"。

验证层级（从强到弱）：
1. Formal verifier（最强）
2. Programmatic verifier
3. External judge
4. Process reward model
5. Intrinsic self-assessment（最弱）

### 12.2 固定资源计量

RSI 必须报告以下指标以区分 Scaling 与 Self-improvement：
- Capability / FLOPs（固定算力）
- Capability / WallClock（固定时间）
- Capability / Dollar（固定成本）
- Capability / HumanHour（固定人工）
- Capability / GPU（固定硬件）
- Capability / ExternalData（固定数据）

### 12.3 元生产率

$$\text{MetaProductivity} = \frac{\text{ImprovementRate}_{t+1}}{\text{ImprovementRate}_t}$$

- > 1：改进过程正在加速（intelligence explosion 的工程指标）
- = 1：线性迭代
- < 1：递减回报

### 12.4 Coding Agent 是 RSI 的最佳实验场

软件工程同时拥有：可执行性、可验证性、可修改性、可组合性、可持久化、可递归（Agent 可修改运行 Agent 的代码）。

### 12.5 开放性 vs. 可验证性

RSI 的核心矛盾：越 open-ended 越难验证。固定 workflow 稳定但 improvement space 被人类定义；open-ended workflow 真正自主但难以评估。

---

## 十三、应用领域

| 论文 | 年份 | 应用领域 | arXiv |
|------|------|---------|-------|
| LinkedIn Customer Support | 2026.08 | 客服系统自进化 | [2608.10224](https://arxiv.org/abs/2608.10224) |
| Skill-Harness Evolution (Embodied) | 2026.08 | 具身 Agent 技能进化 | [2608.11350](https://arxiv.org/abs/2608.11350) |
| GUI Visual Grounding | 2026.08 | 测试时自进化 GUI 定位 | [2608.11191](https://arxiv.org/abs/2608.11191) |
| GeoForge | 2026.08 | 地球观测推理 | [2608.10494](https://arxiv.org/abs/2608.10494) |
| A-SR (Symbolic Regression) | 2026.08 | 符号回归 | [2608.04872](https://arxiv.org/abs/2608.04872) |
| A/B Agent | 2026.08 | 工业 A/B 测试策略迭代 | [2608.04625](https://arxiv.org/abs/2608.04625) |
| SciToolAgent-Evo | 2026.07 | 科学工具获取 | [2607.28692](https://arxiv.org/abs/2607.28692) |
| Self-Evolving Coding Agents | 2026.08 | 编程 Agent 综述 | [2608.03392](https://arxiv.org/abs/2608.03392) |
| Clinical Systems | 2026.07 | 临床医学自进化 | [2607.11175](https://arxiv.org/abs/2607.11175) |
| WebGrader | 2026.08 | Web 开发自进化评分器 | [2608.06474](https://arxiv.org/abs/2608.06474) |
| VASO | 2026.06 | Physical AI 可验证技能 | [2606.05395](https://arxiv.org/abs/2606.05395) |

---

## 十四、技术演化路径

```
Prompt Engineering → Context Engineering → Harness Engineering → Meta-Harness Engineering

2022: STaR
2023: Self-Refine / Reflexion / Voyager / Eureka
2024: AI Scientist / SWE Agents / Gödel Agent
2025: AI Scientist-v2 / AlphaEvolve / DGM / Absolute Zero / ACE
2026: AutoResearch / Harness Evolution / MCE / MetaSkill-Evolve / Hyperagents / AIDE²
Future: Self-improving Researcher → AI-designed Successor → Open-ended RSI
```

优化对象的演化：
```
Answer → Agent → Agent Generator → AI Research Process → AI Improvement Process
```

---

## 十五、论文总索引

### 综述（6 篇）

| 论文 | 年份 | arXiv |
|------|------|-------|
| A Survey of Self-Evolving Agents: What, When, How, Where | 2025.07 | [2507.21046](https://arxiv.org/abs/2507.21046) |
| A Comprehensive Survey of Self-Evolving AI Agents | 2025.08 | [2508.07407](https://arxiv.org/abs/2508.07407) |
| Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops | 2026.07 | [2607.07663](https://arxiv.org/abs/2607.07663) |
| A Survey of Agent Memory: Towards Self-Evolving and Long-Horizon Agents | 2026.01 | [2602.06052](https://arxiv.org/abs/2602.06052) |
| Self-Evolving Coding Agents | 2026.08 | [2608.03392](https://arxiv.org/abs/2608.03392) |
| The Path to Self-Evolving Clinical Systems | 2026.07 | [2607.11175](https://arxiv.org/abs/2607.11175) |

### 基础工作（8 篇）

| 论文 | 年份 | arXiv |
|------|------|-------|
| Voyager | 2023.05 | [2305.16291](https://arxiv.org/abs/2305.16291) |
| Reflexion | 2023.03 | [2303.11366](https://arxiv.org/abs/2303.11366) |
| Self-Refine | 2023.03 | [2303.17651](https://arxiv.org/abs/2303.17651) |
| STaR | 2022.03 | [2203.14465](https://arxiv.org/abs/2203.14465) |
| Self-Rewarding Language Models | 2024.01 | [2401.10020](https://arxiv.org/abs/2401.10020) |
| Gödel Agent | 2024.10 | [2410.04444](https://arxiv.org/abs/2410.04444) |
| ExpeL | 2023.08 | [2308.10379](https://arxiv.org/abs/2308.10379) |
| Absolute Zero | 2025.05 | [2505.03335](https://arxiv.org/abs/2505.03335) |

### 现代 RSI 核心（10 篇）

| 论文 | 年份 | arXiv |
|------|------|-------|
| The AI Scientist | 2024.08 | [2408.06292](https://arxiv.org/abs/2408.06292) |
| AI Scientist-v2 | 2025.04 | [2504.08066](https://arxiv.org/abs/2504.08066) |
| Darwin Gödel Machine | 2025.05 | [2505.22954](https://arxiv.org/abs/2505.22954) |
| Agentic Context Engineering | 2025.10 | [2510.04618](https://huggingface.co/papers/2510.04618) |
| Meta Context Engineering | 2026.01 | [2601.21557](https://arxiv.org/abs/2601.21557) |
| Agentic Harness Engineering | 2026.04 | [2604.25850](https://arxiv.org/abs/2604.25850) |
| Meta-Harness | 2026.03 | [2603.28052](https://arxiv.org/abs/2603.28052) |
| MetaSkill-Evolve | 2026.07 | [2607.05297](https://arxiv.org/abs/2607.05297) |
| Hyperagents | 2026.03 | [2603.19461](https://arxiv.org/abs/2603.19461) |
| Continual Harness | 2026.05 | [2605.09998](https://arxiv.org/abs/2605.09998) |

### 技能进化（10 篇）

| 论文 | 年份 | arXiv |
|------|------|-------|
| MetaSkill-Evolve | 2026.07 | [2607.05297](https://arxiv.org/abs/2607.05297) |
| SkillZip | 2026.08 | [2608.11079](https://arxiv.org/abs/2608.11079) |
| SkillHEX | 2026.08 | [2608.05628](https://arxiv.org/abs/2608.05628) |
| SkillAudit | 2026.06 | [2606.14239](https://arxiv.org/abs/2606.14239) |
| SkillProx | 2026.08 | [2608.07449](https://arxiv.org/abs/2608.07449) |
| SoK: Agentic Skills | 2026.02 | [2602.20867](https://arxiv.org/abs/2602.20867) |
| Learning Globally Reusable Skills | 2026.08 | [2608.06153](https://arxiv.org/abs/2608.06153) |
| Training Skills Like Parameters | 2026.07 | [2607.27557](https://arxiv.org/abs/2607.27557) |
| Search2Skill | 2026.08 | [2608.05245](https://arxiv.org/abs/2608.05245) |
| Emotion2Skill | 2026.08 | [2608.09248](https://arxiv.org/abs/2608.09248) |

### 记忆驱动（7 篇）

| 论文 | 年份 | arXiv |
|------|------|-------|
| CrystalMem | 2026.07 | [2608.00303](https://arxiv.org/abs/2608.00303) |
| Tree-of-Experience | 2026.08 | [2608.09044](https://arxiv.org/abs/2608.09044) |
| RoMeRL | 2026.08 | [2608.02508](https://arxiv.org/abs/2608.02508) |
| Authority Collapse | 2026.08 | [2608.01679](https://arxiv.org/abs/2608.01679) |
| Memory→Skills Co-Evolution | 2026.07 | [2607.16621](https://arxiv.org/abs/2607.16621) |
| Memory Reward Inflation | 2026.06 | [2608.00017](https://arxiv.org/abs/2608.00017) |
| Critic Experience Bank | 2026.07 | [2607.12397](https://arxiv.org/abs/2607.12397) |

### 安全与攻击（9 篇）

| 论文 | 年份 | arXiv |
|------|------|-------|
| SkillJack | 2026.08 | [2608.03509](https://arxiv.org/abs/2608.03509) |
| Trajectory Poisoning | 2026.08 | [2608.05563](https://arxiv.org/abs/2608.05563) |
| Query-Only Backdoor | 2026.08 | [2608.08303](https://arxiv.org/abs/2608.08303) |
| Benign Alone, Harmful Together | 2026.08 | [2608.01759](https://arxiv.org/abs/2608.01759) |
| Self-Evolution Backfires | 2026.08 | [2608.05810](https://arxiv.org/abs/2608.05810) |
| ActBench | 2026.08 | [2608.09476](https://arxiv.org/abs/2608.09476) |
| Agent Skill Security | 2026.07 | [2607.13987](https://arxiv.org/abs/2607.13987) |
| Yesterday's Shield | 2026.08 | [2608.08471](https://arxiv.org/abs/2608.08471) |
| Recursive Self-Training Collapse | 2026.06 | [2606.28438](https://arxiv.org/abs/2606.28438) |

### 理论与极限（5 篇）

| 论文 | 年份 | arXiv |
|------|------|-------|
| Self-Reference in LLMs: Introspection Threshold | 2026.07 | [2607.04277](https://arxiv.org/abs/2607.04277) |
| SARSI: Self-Aware Recursively Self-Improving Agents | 2026.07 | [2607.12254](https://arxiv.org/abs/2607.12254) |
| Beyond Capability Boundary: Zeroth-Order Optimization | 2026.08 | [2608.09292](https://arxiv.org/abs/2608.09292) |
| Rethinking Self-Evolving Agents | 2026.08 | [2608.09629](https://arxiv.org/abs/2608.09629) |
| Frontis-MA1: AI4AI Model for RSI in MLE | 2026.07 | [2607.28568](https://arxiv.org/abs/2607.28568) |

### 其他重要工作

| 论文 | 年份 | 关键贡献 | arXiv |
|------|------|---------|-------|
| Symbolic Learning Enables Self-Evolving Agents | 2024.06 | 符号学习自进化 | [2406.18532](https://arxiv.org/abs/2406.18532) |
| OPD-Evolver | 2026.06 | 策略蒸馏驱动的整体 Agent 进化 | [2606.17628](https://arxiv.org/abs/2606.17628) |
| Self-evolving LLM agents with in-distribution Optimization | 2026.06 | 分布内优化 | [2606.07367](https://arxiv.org/abs/2606.07367) |
| DataEvolver | 2026.06 | 多级自进化数据准备 | [2606.07001](https://arxiv.org/abs/2606.07001) |
| Reward-Free Evolving Agents | 2026.07 | 配对验证器实现无奖励进化 | [2607.14408](https://arxiv.org/abs/2607.14408) |
| SERPO | 2026.07 | 自进化 rubric 策略优化 | [2607.26873](https://arxiv.org/abs/2607.26873) |
| Teaching LLMs to Self-Evolve | 2026.07 | RL 培养核心元技能 | [2607.21971](https://arxiv.org/abs/2607.21971) |
| Knowledge-Centric Self-Improvement | 2026.07 | 知识中心的自我改进 | [2607.19592](https://arxiv.org/abs/2607.19592) |
| SWE-RL | 2025.12 | Self-play 训练软件 Agent | [2512.18552](https://arxiv.org/abs/2512.18552) |
| One Recipe, Many Harnesses | 2026.08 | 自进化跨语言/模型编码 | [2608.10178](https://arxiv.org/abs/2608.10178) |

---

## 十六、延伸阅读

- [Lilian Weng: Harness Engineering for Self-Improvement](https://lilianweng.github.io/posts/2026-07-04-harness/)（2026.07）
- [Artifocial: From Self-Play to Self-Research (RSI Workshop 解读)](https://artifocial.com/blog/rsi-workshop-2026-mar-18)（2026.03）
- [Awesome Recursive Self-Improvement](https://github.com/natnew/awesome-recursive-self-improvement)
- [Awesome Self-Evolving Coding Agents](https://github.com/zhouhao1024/Awesome-Self-Evolving-Coding-Agents)
- [ICLR 2026 RSI Workshop](https://recursive-workshop.github.io/)
- → 相关文档：[测试时训练 (TTT)](./TTT.md)、[Agent记忆与知识自进化](../7.product/Memory.md)、[长上下文](./LongContext.md)
