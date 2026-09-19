# Scaling Laws 全景：从 Kaplan 到 Post-Training Scaling

> 起因：唐杰发布 GLM-5.3 的推文（https://x.com/jietang/status/2089941544581403107），系统回顾了 Scaling Law 的演进脉络。本文整理推文中涉及的每一篇关键论文、每一个关键概念的技术背景。

---

## 0. 唐杰推文核心论点

GLM-5.3 与 GLM-5.2 同 base、同架构、同参数量，唯一区别是一个月的 long-horizon RL post-training。这验证了一个论点：参数量到一定阈值后，额外能力来自其他"旋钮"——post-training 是其中剩余空间最大的一个。

推文勾勒了 Scaling Law 的五个阶段：
1. Kaplan 2020：参数增长快于数据（2.7:1）
2. Chinchilla 2022：纠正为等速增长（~20 TPP）
3. 推理成本进入目标函数：小模型长训练（过训练）
4. MoE 改变游戏规则：总参数 vs 激活参数分离
5. Post-training 成为新旋钮：RL + long-horizon environments

---

## 1. Kaplan 2020：Scaling Laws 的开端

**论文**：Scaling Laws for Neural Language Models
**作者**：Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (OpenAI)
**时间**：2020.01
**arXiv**：https://arxiv.org/abs/2001.08361

### 核心发现

Transformer 语言模型的 loss 与三个量呈幂律关系：模型参数量 N、数据集大小 D、训练计算量 C。关键结论：

- Loss L(N, D, C) ∝ N^{-α} + D^{-β} + C^{-γ}，其中 α, β, γ 是正的幂指数
- 在固定计算预算下，最优分配是参数增长快于数据，比例约 2.7:1（每增加 1 倍参数只需增加 0.37 倍数据）
- 这个幂律在跨越 7 个数量级时仍然成立

### 对行业的影响

整个行业照着 Kaplan 的配方做：GPT-3（175B 参数，300B tokens）、Google Gopher（280B 参数，300B tokens）、Microsoft MT-NLG（530B 参数，270B tokens）。这些模型都是参数巨大但训练数据相对不足。

### 唐杰的评价

"Kaplan et al. (2020) fit an exponent that told everyone to grow parameters faster than data — roughly 2.7:1 — and the industry complied."

---

## 2. Chinchilla 2022：纠正 Kaplan 的错误

**论文**：Training Compute-Optimal Large Language Models
**作者**：Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan N. Gomez, et al. (DeepMind)
**时间**：2022.03
**arXiv**：https://arxiv.org/abs/2203.15556

### 核心发现

DeepMind 训练了 400 多个模型（参数从 70M 到 10B，计算量从 1e18 到 5e21 FLOPs），发现：

- Kaplan 的实验方法有系统性偏差：他们只在很短的训练步数上拟合，外推到长训练时不准
- 真正的 compute-optimal 分配是 ~20 tokens per parameter（TPP），而非 Kaplan 的 ~3.6 TPP
- 最优策略是参数和数据等速增长，而非参数快于数据
- 现有的大模型（GPT-3、Gopher）都显著"训练不足"（undertrained）

### Chinchilla 模型的验证

DeepMind 用 Chinchilla（70B 参数，1.4T tokens = 20 TPP）对比 Gopher（280B 参数，300B tokens = ~1.1 TPP），计算量相同，但 Chinchilla 全面超越 Gopher。

### 对行业的影响

- 证明了万亿参数那轮是弯路：用同样算力，小模型多数据更好
- 行业转向"Chinchilla optimal"策略：Llama 1（65B, 1.4T tokens ≈ 21.5 TPP）、MassivelyMultilingual 等都遵循了这个比例
- "Chinchilla optimal" 成为默认的预训练配方

### 唐杰的评价

"Hoffmann et al. (2022) redid the experiment across four hundred models and found the compute-optimal split is closer to 20 tokens per parameter... The trillion-parameter round was, in retrospect, a detour the whole field took together and then reversed."

---

## 3. 过训练（Over-Training）：推理成本改变最优解

Chinchilla 只优化训练成本，假设模型训完就评估。但现实中模型每天被调用数十亿次，推理成本主导全生命周期。

### 3.1 核心论文：Language Models Scale Reliably with Over-Training

**作者**：Samir Yitzhak Gadre, Georgios Smyrnis, Vaishaal Shankar, Suchin Gururangan, Mitchell Wortsman, Rulin Shao, Jean Mercat, Alex Fang, et al.
**时间**：2024.03
**arXiv**：https://arxiv.org/abs/2403.08540

#### 核心发现

- 训练了 104 个模型（11M 到 6.9B 参数），首次系统研究过训练区域的 scaling law
- Scaling law 可以在过训练区域可靠外推：用 300 倍少的算力预测 1.4B 参数、900B tokens（32 倍过训练）的 loss
- 过训练区域的 loss 遵循与 compute-optimal 区域相同的幂律，只是系数不同
- 提出了 perplexity 到 downstream task 性能的幂律映射

#### 关键意义

为行业"故意过训练小模型"提供了理论依据——不是拍脑袋，而是 scaling law 在过训练区域仍然可预测。

### 3.2 More Compute Is What You Need

**作者**：Zhen Guo
**时间**：2024.04
**arXiv**：https://arxiv.org/abs/2404.19484

#### 核心发现

- 提出统一 scaling law：模型性能主要取决于总计算量，与具体的"参数 vs 数据"分配关系不大
- 推论1：为了推理效率，应该优先小模型 + 大数据（过训练）
- 推论2：当网络数据耗尽后，增大参数可能是唯一提升途径

### 3.3 实践案例：Llama 2 和 Gemma 2

**Llama 2**
- 论文：Llama 2: Open Foundation and Fine-Tuned Chat Models
- 作者：Hugo Touvron, Louis Martin, Kevin Stone, et al. (Meta)
- 时间：2023.07
- arXiv：https://arxiv.org/abs/2307.09288
- Llama-2-7B：7B 参数，2T tokens → ~290 TPP（Chinchilla optimal 的 ~14.5 倍）
- 明确为推理效率而过训练

**Gemma 2**
- 论文：Gemma 2: Improving Open Language Models at a Practical Size
- 作者：Gemma Team, Morgane Riviere, Shreya Pathak, et al. (Google)
- 时间：2024.07
- arXiv：https://arxiv.org/abs/2408.00118
- Gemma-2-9B：9B 参数，8T tokens → ~889 TPP（Chinchilla optimal 的 ~44 倍）
- 极端过训练，用数据量弥补参数量

### 唐杰的评价

"Put inference into the objective and the optimum moves toward smaller models trained far longer — deliberate over-training, which is what Llama-2-7B and Gemma-2-9B were doing at roughly 290 and 889 tokens per parameter."

---

## 4. 推理感知的 Scaling Law

### 4.1 Scaling Inference-Efficient Language Models

**作者**：Song Bian, Minghao Yan, Shivaram Venkataraman
**时间**：2025.01
**arXiv**：https://arxiv.org/abs/2501.18107

#### 核心发现

- 同样大小的模型因架构不同，推理延迟可差 3.5 倍
- 修改 Chinchilla scaling law，联合优化参数量、训练 token 数和模型架构
- 训练了 63 个模型（80M 到 1B），发布 Morph-1B：推理延迟降低 1.8 倍，精度不降
- 发现"宽而浅"的模型在保持精度的同时提升推理效率

### 4.2 Inference Scaling Laws（Test-Time Scaling）

**论文**：Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models
**作者**：Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, Yiming Yang (CMU)
**时间**：2024.08
**arXiv**：https://arxiv.org/abs/2408.00724

#### 核心发现

- 首次系统研究推理（test-time）的计算最优配置
- 在推理时增加计算（多数投票、best-of-n、树搜索）可以比增大模型参数更高效
- 小模型 + 高级推理算法在成本-性能帕累托前沿上优于大模型
- 例如：Llemma-7B + 树搜索在 MATH 基准上超越 Llemma-34B

#### 与唐杰论点的关系

这篇论文从"推理端"论证了同样的观点：能力不只在参数量里，推理时的计算分配也很重要。这是 test-time scaling 的理论基础，也是 GLM-5.3 "long-horizon environments" 的背景之一。

### 4.3 Test-Time Scaling Makes Overtraining Compute-Optimal

**作者**：Nicholas Roberts, Sungjun Cho, Zhiqi Gao, Tzu-Heng Huang, Albert Wu, Gabriel Orlanski
**时间**：2026.04
**arXiv**：https://arxiv.org/abs/2604.01411

#### 核心发现

- 提出了 Train-to-Test (T²) scaling law：联合优化模型大小、训练 token 数和推理采样数
- 当考虑 test-time scaling（如 repeated sampling）时，过训练变得 compute-optimal
- 因为 test-time scaling 的成本随模型大小和采样数增长，小模型 + 过训练 + 多采样在端到端预算下更优
- 这为 Llama/Gemma 的过训练策略提供了事后理论解释：它们不是"浪费"，而是在 test-time scaling 场景下最优

#### 与唐杰论点的关系

这篇论文可能是唐杰所说的"Roberts et al. (2025)"（注：arXiv 版本标注 2026.04，但可能在 2025 年已在会议上发表或流传）。它直接支持了"推理成本改变最优解"的论点。

---

## 5. MoE 的 Scaling Law：总参数 vs 激活参数

MoE（Mixture of Experts）架构把 scaling 拆成了两个维度：总参数量（知识容量）和激活参数量（计算量/推理能力）。这彻底改变了 scaling law 的图景。

### 5.1 Mixture of Parrots：Experts 提升记忆而非推理

**论文**：Mixture of Parrots: Experts improve memorization more than reasoning
**作者**：Samy Jelassi, Clara Mohri, David Brandfonbrener, Alex Gu, Nikhil Vyas, Nikhil Anand, David Alvarez-Melis, Yuanzhi Li (Meta/Princeton)
**时间**：2024.10
**arXiv**：https://arxiv.org/abs/2410.19034

#### 核心发现

- 固定激活参数量，增加 expert 数量：记忆任务持续提升，推理任务饱和
- 理论证明：存在图问题无法被任何数量的 expert（给定宽度）解决，但略宽的 dense 模型可以轻松解决
- MoE 在知识密集型任务（闭书检索）上有效利用大量 expert
- 但在数学/推理任务上，增加 expert 没有同等收益

#### 关键意义

这是唐杰所说"memorization favoring more parameters and reasoning favoring more data"的核心证据之一。MoE 的总参数主要装知识，不直接提升推理。

### 5.2 Parameters vs FLOPs：MoE 最优稀疏度

**论文**：Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models
**作者**：Samira Abnar, Harshay Shah, Dan Busbridge, Alaaeldin Mohamed Elnouby Ali, Josh Susskind, Vimal Thilak (Apple)
**时间**：2025.01
**arXiv**：https://arxiv.org/abs/2501.12370

#### 核心发现

- 在不同约束（参数量、总训练计算量）下，存在最优稀疏度（活跃参数比例）
- 最优稀疏度同时提升训练效率和模型性能
- 为 MoE 架构设计提供了 scaling law 指导

### 5.3 Optimal Sparsity of MoE for Reasoning Tasks

**论文**：Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks
**作者**：Taishi Nakamura, Satoki Ishikawa, Masaki Kawamura, Takumi Okamoto, Daisuke Nohara, Jun Suzuki, Rio Yokota
**时间**：2025.08
**arXiv**：https://arxiv.org/abs/2508.18672

#### 核心发现

这是与唐杰描述最匹配的论文，包含他引用的两个核心结论：

**结论1 — Active FLOPs 决定推理能力**：训练 loss 相同的模型，激活参数更多的推理准确率更高。

**结论2 — TPP 任务相关**：记忆任务随总参数增加而改善；推理任务需要最优 TPP（即数据量重要），说明推理是"数据饥渴"的。

**结论3 — RL 后训练不改变这些趋势**：即使经过 GRPO 强化学习后训练或增加 test-time compute，上述规律不变。

**结论4 — 固定 TPP 下加大总参数损害推理**：在固定 tokens-per-parameter 下，增加总参数反而降低推理性能，而增加激活 expert 数量可靠地提升推理。

#### 与唐杰论点的精确对应

唐杰原话："Roberts et al. (2025) find the optimal tokens-per-parameter is task-dependent, with memorization favoring more parameters and reasoning favoring more data. Follow-up work on MoE observes that at fixed TPP, pushing total parameters higher actually degrades reasoning, while activating more experts reliably helps it."

这篇论文（Nakamura et al.）包含了上述全部发现。唐杰可能将多篇论文归因为"Roberts et al."，或引用了该团队的一个尚未在 arXiv 上找到的版本。Roberts et al. (arXiv:2604.01411) 是相关但侧重 test-time scaling 的论文。

### 5.4 其他 MoE Scaling 论文

**Joint MoE Scaling Laws** (Ludziejewski et al., 2025.02)
- arXiv：https://arxiv.org/abs/2502.05172
- 联合 scaling law 包含激活参数、数据集大小和 expert 数量
- 惊人发现：MoE 模型可以比 dense 模型更省内存
- 280+ 实验，最大 2.7B 激活参数 / 5B 总参数

**Towards Greater Leverage** (Tian et al., 2025.07)
- arXiv：https://arxiv.org/abs/2507.17702
- 提出 Efficiency Leverage (EL) 指标量化 MoE 相对 dense 的计算优势
- 300+ 模型实验，最大 28B 参数
- Ling-mini-beta：0.85B 激活参数匹配 6.1B dense 模型，省 7 倍算力

**Holistic Scaling Laws for MoE** (Wan et al., 2026.03)
- arXiv：https://arxiv.org/abs/2603.21862
- 提出 FLOPs/token + 激活参数 + 总参数的三元约束
- 16 维搜索空间降维为两阶段优化
- 关键发现：随规模增大，近最优配置带变宽（给实践者更多灵活性）

**Generalizing Scaling Laws for Dense and Sparse LLMs** (Hossain et al., 2025.08)
- arXiv：https://arxiv.org/abs/2508.06617
- 架构无关的统一 scaling law

### 唐杰的评价

"In a MoE model two quantities have to be kept apart: total parameters govern roughly how much the model can hold — knowledge, facts, the long tail — while activated parameters and effective depth govern roughly how far it can think, how many steps of a causal chain it can carry before it comes apart."

---

## 6. Scaling Laws for Precision：精度也是一个旋钮

**论文**：Scaling Laws for Precision
**作者**：Tanishq Kumar, Zachary Ankner, Benjamin F. Spector, Blake Bordelon, Niklas Muennighoff, Mansheej Paul, Cengiz Pehlevan, Christopher Ré (Stanford/Harvard)
**时间**：2024.11
**arXiv**：https://arxiv.org/abs/2411.04330

### 核心发现

- 低精度训练降低模型的"有效参数量"
- 可以预测不同精度组合下的 loss（465+ 预训练实验拟合）
- 推理量化：模型训练数据越多，量化降质越严重——最终额外预训练数据会"主动有害"
- 训练端：低精度训练更大模型可能是 compute-optimal 的

### 与唐杰论点的关系

这篇论文引入了"精度"作为 scaling 的另一个维度。唐杰虽然没直接提到精度，但这属于"Scaling, but not only of parameters"的范畴——精度、参数、数据、推理计算都是旋钮。

---

## 7. Post-Training：新的 Scaling 旋钮

### 7.1 DeepSeek-R1：纯 RL 激励推理能力

**论文**：DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
**作者**：DeepSeek-AI, Daya Guo, Dejian Yang, et al.
**时间**：2025.01
**arXiv**：https://arxiv.org/abs/2501.12948

#### 核心发现

- 证明了 LLM 的推理能力可以纯靠 RL 激发，不需要人类标注的推理轨迹
- RL 框架促进了高级推理模式的涌现：自我反思、验证、动态策略切换
- 在数学、编程竞赛、STEM 领域超越传统 SFT 方法
- 涌现的推理模式可以系统性地蒸馏到更小的模型

#### 与唐杰论点的关系

DeepSeek-R1 是"post-training scaling"的里程碑。它证明了不增加参数，纯靠 RL post-training 就能获得巨大推理提升——这正是 GLM-5.3 实验的逻辑前提。

### 7.2 LLM Post-Training 综述

**论文**：LLM Post-Training: A Deep Dive into Reasoning Large Language Models
**作者**：Komal Kumar, Tajamul Ashraf, Omkar Thawakar, Rao Muhammad Anwer, Hisham Cholakkal, Mubarak Shah
**时间**：2025.02
**arXiv**：https://arxiv.org/abs/2502.21321

系统综述了 post-training 技术全貌：SFT → RLHF → RLVR → 推理模型，覆盖知识精炼、推理增强、事实准确性提升等方向。

### 7.3 Post-Training Reasoning Data Primer

**论文**：A Primer in Post-Training Reasoning Data: What We Know About How It Works
**作者**：Yaoming Li, Guangxiang Zhao, Qilong Shi, Lin Sun, Xiangzheng Zhang, Tong Yang
**时间**：2026.06
**arXiv**：https://arxiv.org/abs/2606.02113

综合 150+ 篇研究，系统梳理 post-training 推理数据的关键变量：数据集设计、RL 配方、奖励模型、基准测试、前沿系统报告。

### 7.4 Select and Improve：Post-Training 的机制理解

**论文**：Select and Improve: Understanding the Mechanics of Post-Training for Reasoning
**时间**：2026.06
**arXiv**：https://arxiv.org/abs/2606.13125

从机制角度研究 RL post-training 如何获得能力：是发现了新行为，还是 sharpen 了已有行为？这对理解 post-training scaling 的"天花板"至关重要。

### 7.5 Slow Thinking 综述

**论文**：A Survey of Slow Thinking-based Reasoning LLMs using Reinforced Learning and Inference-time Scaling Law
**作者**：Qianjun Pan, Wenkai Ji, Yuyang Ding, Junsong Li, Shilian Chen
**时间**：2025.05
**arXiv**：https://arxiv.org/abs/2505.02665

从 Kahneman"快与慢思考"视角综述推理 LLM，覆盖 RL 后训练和推理时 scaling 两条路线。

---

## 8. GLM-5.3 的实验意义

### 唐杰的关键声明

> "GLM-5.3 is our controlled experiment on that claim. Same base, same architecture, same total and activated parameters as GLM-5.2. One month of scaling long-horizon environments and RL. The gains are not marginal."

### 实验设计分析

这是一个精心设计的受控实验：
- **控制变量**：base model、架构、总参数、激活参数全部相同
- **自变量**：post-training（long-horizon environments + RL）的强度
- **结论**：post-training scaling 的收益"not marginal"（不是边际的）

这直接验证了推文的核心论点：参数量到一定阈值后（"enough to hold the world"），额外能力来自其他旋钮。

### "Long-horizon environments" 的含义

结合 7.1（DeepSeek-R1）和搜索到的相关论文，long-horizon environments 指的是：
- 需要多步推理的复杂任务环境（数学证明、代码调试、安全分析等）
- RL agent 在环境中探索长推理链，获得 delayed reward
- 相关论文：
  - Context-Folding (arXiv:2510.11967)：agent 主动管理长程上下文
  - Rewarding Beliefs (arXiv:2605.20061)：长程任务的信用分配
  - HiMAC (arXiv:2603.00977)：长程 LLM agent 的层次化学习

### 安全场景的例子

唐杰特别提到："Finding a vulnerability is not a retrieval problem. It doesn't come from having memorized more CVEs; it comes from carrying a twenty-step chain of inference to the end without losing the thread."

这呼应了 Mixture of Parrots 的发现：MoE 的总参数提升记忆但不提升推理。20 步推理链的稳定性来自激活参数和有效深度，而非总参数量。

---

## 9. 唐杰提到的"旋钮"总览

| 旋钮 | 代表论文 | 核心发现 | 当前状态 |
|------|---------|---------|---------|
| 参数量 N | Kaplan 2020 | 参数越大 loss 越低（幂律） | 到阈值后边际递减 |
| 数据量 D | Chinchilla 2022 | ~20 TPP 最优 | 数据稀缺是新瓶颈 |
| 训练计算 C | Chinchilla 2022 | N 和 D 等速增长 | 仍是主旋钮 |
| 过训练 TPP | Gadre 2024 | 推理成本主导→小模型长训练 | Llama/Gemma 已践行 |
| MoE 稀疏度 | Abnar 2025, Nakamura 2025 | 总参数=知识，激活=推理 | 任务相关最优 |
| 精度 | Kumar 2024 | 低精度降低有效参数 | 量化有上限 |
| 推理计算 | Wu 2024 | 小模型+搜索可超大模型 | Test-time scaling 兴起 |
| Test-time + 过训练 | Roberts 2026 | 联合优化后过训练最优 | 理论框架刚建立 |
| Post-training RL | DeepSeek-R1 2025 | 纯 RL 激发推理 | **GLM-5.3 验证的旋钮** |
| Long-horizon environments | 唐杰 2026 | 长程任务 RL 提升 | GLM-5.3 实验 |

---

## 10. 未来方向：还没转的旋钮

唐杰在推文最后明确列了三个还没动的旋钮，并暗示了更多方向。原话：

> "Base model size, pretraining data, compute spent per forward pass: all of them are still on the table, and we will come back to each. What this experiment taught us is that the dials do not have to be turned together, and that the one worth turning next is rarely the one that was worth turning last. We are not done scaling. Next time, maybe mid-training, pre-training, and even more."

### 10.1 明确点名的三个旋钮

#### (1) Base model size（基础模型尺寸）

GLM-5.3 故意没动 base model，是为了做受控实验——只验证 post-training 的效果。但唐杰说"会回来转这个旋钮"。

怎么做？不是简单地把参数从 X 扩到 2X。根据前面搜集的论文，关键约束变了：

- Chinchilla 告诉你参数和数据要等速增长，但现在高质量网络数据快耗尽了。"More Compute Is What You Need"（arXiv:2404.19484）指出：数据耗尽后加参数是唯一途径
- MoE 架构下，加大总参数主要增加知识容量（Mixture of Parrots 证明的），但推理能力靠激活参数和有效深度
- 所以"加 base size"的正确做法可能是：加大总参数（装更多知识）同时加大激活参数（提升推理深度），而不是只加总参数

唐杰原话有个关键前提："Total parameters appear to matter up to a threshold — enough to hold the world"。意思是 GLM-5.2 的参数量可能已经接近"装下世界知识"的阈值了，再加大总参数的边际收益递减。但如果激活参数还有空间，那加大激活参数就还有用。

#### (2) Pretraining data（预训练数据）

数据是 Chinchilla 之后的核心瓶颈。几个方向：

- **数据质量**：从"爬多少 token"转向"筛多少高质量 token"。Common Crawl 里大量是垃圾，去重+质量过滤后的有效数据远少于原始 token 数
- **合成数据**：用强模型生成训练数据。DeepSeek-R1 证明了 RL 涌现的推理模式可以蒸馏给小模型——这本质上就是用强模型生成高质量推理数据
- **数据重复**："Data Repetition Beats Data Scaling in Long-CoT SFT"（arXiv:2602.11149）发现，在 long-CoT 微调中，重复数据比增加新数据更有效——这跟经典 ML 直觉相反
- **神经压缩文本训练**：arXiv:2404.03622（Adam Roberts 作者）探索直接在神经压缩的文本上训练，压缩率远高于 subword tokenizer，等于在同样 token 数下塞进更多信息

#### (3) Compute spent per forward pass（每次前向传播的计算量）

这个旋钮最微妙，也最前沿。它不是"训练时花更多算力"，而是"推理时每次 forward 花更多算力"。具体含义：

- **有效深度（effective depth）**：模型更深的 forward pass 能走更长的因果链。唐杰说的"carrying a twenty-step chain of inference"就靠这个。加深网络、或者用 looped transformer（同一个层循环多次）都能增加有效深度。相关论文："Reasoning with Latent Thoughts: On the Power of Looped Transformers"（arXiv:2502.17416）提出许多推理问题需要的不是更多参数而是更多深度
- **MoE 激活更多 expert**：Nakamura 2025 证明"activating more experts reliably helps reasoning"。当前 GLM-5.2 可能只激活了很少的 expert，增加激活比例 = 每次 forward 算更多
- **Test-time compute**：推理时多采样、树搜索、self-consistency。Wu 2024 证明小模型+搜索可超大模型

这个旋钮跟前面 MoE 的研究直接关联：总参数不动（知识容量不变），但每次 forward 激活更多参数（推理更深）。

### 10.2 暗示的更多方向

原话："Next time, maybe mid-training, pre-training, and even more."

#### (4) Mid-training（中期训练）

这是一个比较新的概念，介于 pre-training 和 post-training 之间。目前行业没有统一定义，但常见的 mid-training 做法：

- **长上下文扩展**：在预训练后、RL 前，用长序列数据继续训练，扩展上下文窗口。GLM-5.2 可能已经做了，但还可以进一步推
- **领域注入**：在通用预训练后，用大量代码、数学、科学论文做 continual pre-training，给模型注入特定领域知识
- **推理轨迹训练**：在 RL 之前，先用 SFT 在大量 CoT（chain-of-thought）数据上训练，建立基础推理模式。DeepSeek-R1 的流程就是 SFT → RL，中间这一步可以算 mid-training
- **数据混合优化**：调整不同数据类型的比例（代码 vs 文本 vs 数学 vs 多语言），找到最优混合配比

#### (5) Pre-training（重回预训练）

这里"pre-training"不是指第一次预训练，而是指用新的认知重新做预训练。意思是：现在知道了 post-training 的效果，可以反过来改进预训练。比如：

- 如果 post-training 发现模型在某些推理模式上学不好，那就在预训练阶段就加入相关数据
- 如果过训练策略有效，可以在预训练阶段就用更激进的 TPP
- 如果精度感知 scaling law（Kumar 2024）的发现成立，预训练时就应该考虑量化友好的训练策略

### 10.3 核心洞察：旋钮不需要一起转

唐杰最重要的一句话：

> "What this experiment taught us is that the dials do not have to be turned together, and that the one worth turning next is rarely the one that was worth turning last."

这是对整个 Scaling Law 演进史的总结：

- 2020 年 Kaplan 说转参数旋钮 → 行业照做
- 2022 年 Chinchilla 说不对，该转数据旋钮 → 行业转向
- 2023 年 Llama 2 说该转过训练旋钮 → 行业跟随
- 2024 年发现该转 MoE 稀疏度旋钮
- 2025 年 DeepSeek-R1 证明该转 post-training 旋钮
- 2026 年 GLM-5.3 验证 post-training 旋钮有效

每次"最值得转的旋钮"都跟上次不同。原因很简单：每个旋钮都有边际递减效应，转到底之后剩余空间就小了，下一个最值得转的永远是剩余空间最大的那个。

当前 GLM-5.2 的状态推测：
- 参数量：接近"装下世界"的阈值，边际递减
- 预训练数据：可能已经用了很多，但质量还有提升空间
- 过训练：可能已经做了（作为 MoE 模型，TPP 应该不低）
- MoE 稀疏度：总参数够大了，但激活参数可能还有空间
- Post-training：GLM-5.3 刚验证了，这个旋钮还有油水

所以唐杰的逻辑是：GLM-5.3 先把 post-training 旋钮转到"剩余空间最大"的位置，下次回来转别的——可能是 mid-training，可能是预训练数据质量，可能是激活参数比例。

### 10.4 旋钮之间的依赖关系

这些旋钮不是完全独立的，存在依赖和反哺：

```
Pre-training data ──→ Base model size ──→ Compute/forward pass
       │                    │                      │
       ↓                    ↓                      ↓
   数据质量提升         参数+激活共扩         有效深度增加
       │                    │                      │
       └──── Mid-training ←─┘                      │
                  │                                │
                  ↓                                │
            推理轨迹训练 ←──────────────────────────┘
                  │
                  ↓
           Post-training RL  ← GLM-5.3 转的就是这个
                  │
                  ↓
            反哺预训练策略（用 post-training 的发现改进 pre-training）
```

关键的反哺路径：post-training 的结果可以反过来指导预训练。如果 RL 发现模型在某种推理模式上学不好，那就在预训练阶段就加入相关数据。唐杰说"Next time, maybe pre-training"就是这个意思——不是从头重做，而是用 post-training 的认知重新优化预训练配方。

---

## 11. Scaling Law 演进时间线

```
2020.01  Kaplan    参数:数据 = 2.7:1（参数优先）
           ↓ GPT-3, Gopher, MT-NLG 跟随
2022.03  Chinchilla  参数:数据 = 1:20（等速，数据优先）
           ↓ Llama 1 等跟随
2023.07  Llama 2    过训练 290 TPP（推理优先）
2024.03  Gadre      过训练 scaling law 可靠外推
2024.07  Gemma 2    极端过训练 889 TPP
2024.08  Wu         推理 scaling law（test-time compute）
2024.10  Jelassi    MoE: experts 提升记忆不提升推理
2024.11  Kumar      精度感知 scaling law
2025.01  DeepSeek-R1  纯 RL 激发推理（post-training scaling 里程碑）
2025.01  Abnar      MoE 最优稀疏度 scaling law
2025.01  Bian       推理高效 scaling law
2025.08  Nakamura   MoE: TPP 任务相关，推理数据饥渴
2026.04  Roberts    Test-time scaling 使过训练最优（T² scaling law）
2026.08  唐杰/GLM-5.3  Post-training scaling 受控实验
```

---

## 12. 与已有文档的交叉引用

- **AGI-Feynman.md**：唐杰的论点与 AGI 五条路线中的"经验驱动 RL"直接对应。Post-training scaling 本质是用 RL 让模型在环境中积累经验，不靠加参数。
- **TTT-Feynman.md**：Test-time scaling（推理时多采样/搜索）与 TTT（测试时训练）不同但相关。前者不改参数，后者改参数。Roberts 的 T² scaling law 连接了两者。
- **[SSI-Ilya.md](../9.people/SSI-Ilya.md)**：Ilya 从 Scale 转向持续学习。唐杰的推文从另一个角度验证了这个转向：Scale 的参数维度到阈值了，新维度是 post-training。
- **PostTraining.md**（4.text-model/）：本文可与已有的 post-training 文档互补，前者聚焦技术方法，本文聚焦 scaling law 演进。

---

## 13. 论文索引

| # | 论文 | arXiv | 时间 |
|---|------|-------|------|
| 1 | Scaling Laws for Neural Language Models (Kaplan) | 2001.08361 | 2020.01 |
| 2 | Training Compute-Optimal Large Language Models (Chinchilla) | 2203.15556 | 2022.03 |
| 3 | Llama 2: Open Foundation and Fine-Tuned Chat Models | 2307.09288 | 2023.07 |
| 4 | Language models scale reliably with over-training | 2403.08540 | 2024.03 |
| 5 | More Compute Is What You Need | 2404.19484 | 2024.04 |
| 6 | Gemma 2: Improving Open Language Models at a Practical Size | 2408.00118 | 2024.07 |
| 7 | Inference Scaling Laws (Compute-Optimal Inference) | 2408.00724 | 2024.08 |
| 8 | Mixture of Parrots: Experts improve memorization more than reasoning | 2410.19034 | 2024.10 |
| 9 | Scaling Laws for Precision | 2411.04330 | 2024.11 |
| 10 | DeepSeek-R1: Incentivizing Reasoning via RL | 2501.12948 | 2025.01 |
| 11 | Parameters vs FLOPs: MoE Optimal Sparsity | 2501.12370 | 2025.01 |
| 12 | Scaling Inference-Efficient Language Models | 2501.18107 | 2025.01 |
| 13 | LLM Post-Training: A Deep Dive into Reasoning LLMs | 2502.21321 | 2025.02 |
| 14 | Joint MoE Scaling Laws | 2502.05172 | 2025.02 |
| 15 | Slow Thinking-based Reasoning LLMs Survey | 2505.02665 | 2025.05 |
| 16 | Towards Greater Leverage: MoE Scaling Laws | 2507.17702 | 2025.07 |
| 17 | Optimal Sparsity of MoE for Reasoning Tasks | 2508.18672 | 2025.08 |
| 18 | Generalizing Scaling Laws for Dense and Sparse LLMs | 2508.06617 | 2025.08 |
| 19 | Holistic Scaling Laws for MoE Architecture | 2603.21862 | 2026.03 |
| 20 | Test-Time Scaling Makes Overtraining Compute-Optimal (Roberts) | 2604.01411 | 2026.04 |
| 21 | A Primer in Post-Training Reasoning Data | 2606.02113 | 2026.06 |
| 22 | Select and Improve: Mechanics of Post-Training | 2606.13125 | 2026.06 |
| 23 | Data Repetition Beats Data Scaling in Long-CoT SFT | 2602.11149 | 2026.02 |
| 24 | Reasoning with Latent Thoughts: On the Power of Looped Transformers | 2502.17416 | 2025.02 |
| 25 | Training LLMs over Neurally Compressed Text | 2404.03626 | 2024.04 |
