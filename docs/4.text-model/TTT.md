# 测试时训练 (Test-Time Training, TTT)

> 核心思想：模型在推理（测试）阶段对自身参数或隐藏状态进行梯度更新，从而适应新输入、新分布。不同于"训练完就冻结"的传统范式，TTT 让模型具备在线学习/持续学习的能力。
>
> 关键词：fast weights、expressive hidden state、self-supervised inner loop、continual learning、RSI

---

## 技术脉络

TTT 的发展可以梳理为三条主线：

1. **TTT 作为域适应方法**（2019-2024）：推理时用自监督任务更新部分参数，应对分布偏移
2. **TTT 作为序列建模层**（2024-2025）：把 RNN 的 hidden state 本身变成一个小模型，推理时训练它 → TTT Layers / Titans
3. **TTT 作为持续学习/自我改进范式**（2025-2026）：在 LLM 和 Agent 场景中实现推理时持续适应 → In-Place TTT / aTTT / RSI Workshop

---

## 一、TTT 原始方法：测试时域适应

### 1.1 Test-Time Training with Self-Supervision for Generalization under Distribution Shifts

- 论文：[arXiv:1909.13231](https://arxiv.org/abs/1909.13231)（Sun et al., ICML 2020，原始 TTT 论文）
- 核心思想：模型同时有一个主任务头（分类/回归）和一个自监督头（如旋转预测）。测试时，冻结主任务头，只更新自监督头的参数（即共享的 backbone 部分参数），使模型适应测试分布
- 机制：
  - 训练时：主任务 loss + 自监督 loss 联合优化
  - 测试时：对每个测试样本，先做几步自监督梯度更新（更新 backbone 的 fast weights），再用更新后的模型做主任务预测
- 意义：首次系统化提出"测试时训练"概念，后续所有 TTT 工作都源于此

### 1.2 TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?

- 论文：[arXiv:2007.10964](https://arxiv.org/abs/2007.10964)（Liu et al., ICLR 2021）
- 贡献：分析了 TTT 在何种条件下失效（如自监督任务与主任务不对齐时），提出用特征对齐改进

---

## 二、TTT Layers：把 Hidden State 变成模型

### 2.1 Learning to (Learn at Test Time): RNNs with Expressive Hidden States

- 论文：[arXiv:2407.04620](https://arxiv.org/abs/2407.04620)（Yu Sun et al., 2024）
- 这是 TTT 从"域适应技巧"升级为"架构创新"的里程碑论文

**核心问题**：传统 RNN（含 Mamba）的 hidden state 是固定大小向量，表达能力受限；Attention 虽强但 O(n²) 不可扩展。

**关键洞察**：让 hidden state 本身就是一个机器学习模型，更新规则是一步自监督学习。因为 hidden state 在测试序列上被"训练"，所以叫 Test-Time Training layers。

**两种实例化**：

| 实例化 | Hidden State | 等价于 | 特点 |
|--------|-------------|--------|------|
| TTT-Linear | 线性模型 | 线性注意力 | 最快，等价于 linear attention |
| TTT-MLP | 两层 MLP | 自注意力的推广 | 更强表达力，但内存 I/O 是瓶颈 |

**理论统一**：
- 内循环学习器 = 线性模型 → 等价于线性注意力
- 内循环学习器 = 核估计 → 等价于自注意力
- 内循环学习器 = 神经网络 → 超越两者

**实验结果**：
- 125M~1.3B 参数规模，TTT-Linear 和 TTT-MLP 能像 Transformer 一样在长上下文持续降低 perplexity
- Mamba 在 16k 上下文后 perplexity 不再下降
- TTT-MLP 在 ImageNet 224×224 原始像素上大幅超越 linear attention Transformer

**为什么小模型能强泛化**：模型的有效容量来自迭代深度（递归次数）而非参数量。同一个 k 层 block 被反复应用 N 次，hidden state 在每次应用中被"训练"一步。

### 2.2 TTT Layers 论文配套解读

- 项目主页：[yueatsprograms.github.io/ttt](https://yueatsprograms.github.io/ttt/home.html)
- 核心代码已开源，包含 TTT-Linear 和 TTT-MLP 的实现

---

## 三、Titans：神经长期记忆

### 3.1 Titans: Learning to Memorize at Test Time

- 论文：[arXiv:2501.00663](https://arxiv.org/abs/2501.00663)（Google, 2025.01）
- 在 TTT Layers 基础上引入"神经长期记忆模块"

**核心思想**：
- Attention = 短期记忆（精确建模依赖关系，但受限于上下文窗口长度）
- Neural Memory = 长期记忆（在测试时学习记忆历史上下文，持久且可并行训练）

**三个变体**：

| 变体 | 记忆融合方式 | 适用场景 |
|------|-------------|---------|
| Titans Mac | 记忆作为上下文拼接 | 通用 |
| Titans Nagi | 记忆作为门控分支 | 需要选择性记忆 |
| Titans Alyiah | 记忆与注意力交错混合 | 长序列 |

**关键结果**：
- 语言建模、常识推理、基因组学、时间序列任务上超越 Transformer 和现代线性 RNN
- 可扩展到 2M+ 上下文窗口
- Needle-in-haystack 准确率高于基线

**记忆更新机制**：
- 使用 surprise 度量（梯度幅度）决定何时写入记忆
- 使用 delta rule（类似在线梯度下降）更新记忆权重
- 记忆可以遗忘（decay 机制），避免无限膨胀

---

## 四、LLM 中的 TTT：持续学习范式

### 4.1 In-Place Test-Time Training

- 论文：[arXiv:2604.06169](https://arxiv.org/abs/2604.06169)（ICLR 2026）
- 目标：作为 drop-in 增强给现有 LLM 加上 TTT 能力，不需要从头训练

**技术方案**：
1. 把 MLP 块的最终投影矩阵作为 fast weights，推理时更新
2. 用与 Next-Token Prediction 对齐的目标函数替代 TTT 的通用重建目标
3. 分块（chunk-wise）更新机制，兼容上下文并行（context parallelism）

**关键改进**：目标函数对齐
- 原始 TTT 用通用自监督重建目标 → 快权重更新方向可能与语言建模目标不一致
- In-Place TTT 用理论上对齐的目标 → 快权重的更新直接服务于 next-token prediction

**实验结果**：
- 4B 参数模型，128k 上下文任务超越基线
- 从头预训练时持续优于竞争性 TTT 方法
- 论文原话："establish In-Place TTT as a promising step towards a paradigm of continual learning in LLMs"

### 4.2 Agentic Test-Time Training (aTTT)

- 论文：[arXiv:2607.03441](https://arxiv.org/abs/2607.03441)（ICLR 2026 Workshop RSI）
- 场景：多轮 Agent 交互中的持续 TTT

**核心问题**：LLM Agent 在长轨迹中退化——重复探索已知状态、重复失败动作、丢失有效策略。

**技术方案**：
- 在多轮 episode 中持续做 TTT，每次更新改变后续生成策略（self-training loop）
- 发现"更新文本重复"是退化的关键信号
- aTTT：token 级重加权——降低在先前更新中出现过的重复 n-gram token 的 loss 权重，保留新 token 的完整权重
- 工程实现：基于 vLLM runtime LoRA API 的并发服务系统，开销仅为无 TTT 的 1.9 倍

**结果**：ALFWorld +5.0 分，SWE-bench Lite +4.9 分。增益集中在模型已有能力但长轨迹漂移的场景。

### 4.3 TEMPO: Scaling Test-time Training for Large Reasoning Models

- 论文：[arXiv:2604.19295](https://arxiv.org/abs/2604.19295)（2026.04）
- 方向：将 TTT 扩展到大型推理模型（如 o1/R1 类模型）

### 4.4 VDS-TTT: Verifier-Driven Sample Selection for Test-Time Training

- 论文：[arXiv:2505.19475](https://arxiv.org/abs/2505.19475)（2025.05）
- 核心思想：用验证器从模型生成的 N 个候选答案中选出高置信度的伪标签，用于测试时 LoRA 微调
- 结果：相对基线提升 32.29%，比无 TTT 的验证器方法提升 6.66%

---

## 五、RSI Workshop 中的 TTT 相关工作

> ICLR 2026 Workshop on Recursive Self-Improvement（RSI），2026年4月，巴西里约热内卢，收录 110 篇论文。
>
> 参考综述：Lilian Weng, ["Harness Engineering for Self-Improvement"](https://lilianweng.github.io/posts/2026-07-04-harness/)（2026.07）

### 5.1 Test-Time Self-Distillation（Spotlight #20）

- OpenReview: [iEmRSwdzyw](https://openreview.net/forum?id=iEmRSwdzyw)
- 作者：Hübotter et al.
- 方法：Self-Distillation Policy Optimization (SDPO)——将环境对失败轨迹的反馈转化为 dense learning 信号，把反馈条件化的模型自身作为 self-teacher
- 结果：LiveCodeBench 竞赛编程上，用 3x 更少的尝试达到与 best-of-k sampling 相同的发现概率

### 5.2 Adaptive Meta-Curriculum for Test-Time Self-Improvement（Spotlight #25）

- OpenReview: [GjoUJTfXiW](https://openreview.net/forum?id=GjoUJTfXiW)
- 作者：Bukkapatnam et al.
- 方法：元学习课程调度器 + 自适应改进算子（在 revision / search / reflection 策略间选择）
- 结果：2.3x 计算效率提升，数学推理准确率 +18.7%

### 5.3 ALMA: Learning to Continually Learn via Meta-learning Agentic Memory Designs（Oral）

- 论文：[arXiv:2602.07755](https://arxiv.org/abs/2602.07755)
- OpenReview: [sOq52KnJmR](https://openreview.net/forum?id=sOq52KnJmR)
- 作者：Xiong, Hu, Clune
- 核心问题：基础模型是无状态的（stateless），限制了 Agent 的持续学习能力
- 方法：Meta Agent 搜索代码形式的记忆设计（数据库 schema + 检索/更新机制），替代人工设计
- 结果：在四个序列决策领域，学到的记忆设计优于 SOTA 人工设计

### 5.4 Federated Nested Learning (FedNL)

- 论文：[arXiv:2605.16350](https://arxiv.org/abs/2605.16350)（2026.05）
- 将 Titans 的线性注意力嵌入联邦学习，实现零样本测试时适应
- 用 delta rule 作为在线梯度步，保持常数推理内存

---

## 六、Tiny Recursive Models (TRM)：小模型的另一种路径

> TRM 不是 TTT 的直接变体，但同样展示了"小模型+递归=强能力"的范式，且与 TTT 在"迭代精化隐藏状态"的思路上高度相关。

### 6.1 Tiny Recursive Models on ARC-AGI

- 论文：[arXiv:2511.02886](https://arxiv.org/abs/2511.02886)（2025.11，原始 TRM 论文）
- 核心：7M 参数的递归神经网络，通过两级潜在层次结构（reasoning state z + answer y）的迭代精化，在 ARC-AGI 上与大模型竞争
- 机制：一个紧凑的推理模块被反复应用，通过迭代深度而非参数量获得有效容量

### 6.2 Probabilistic TRM (PTRM)

- 论文：[arXiv:2605.19943](https://arxiv.org/abs/2605.19943)（2026.05）
- 问题：TRM 的确定性递归会收敛到次优解，无逃逸机制
- 方案：在每个递归步注入高斯噪声，并行探索多样解空间，用 Q head 选择最优
- 结果：Sudoku-Extreme 87.4% → 98.75%；Pencil Puzzle Bench 62.6% → 91.2%（前沿 LLM 仅 55.1%），仅用 7M 参数

### 6.3 Test-time Adaptation of Tiny Recursive Models

- 论文：[arXiv:2511.02886](https://arxiv.org/abs/2511.02886)（2025.11）
- 在竞赛计算限制内，对预训练 TRM 做全参数微调（非 LoRA）
- 预训练：1280 个公开任务，4×H100 训练 48 小时
- 竞赛中：12,500 步梯度更新达到 6.67% 准确率

### 6.4 CGAR: 加速 TRM 训练

- 论文：[arXiv:2511.08653](https://arxiv.org/abs/2511.08653)（2025.11）
- 课程学习应用到架构深度：从浅层 (2,1) 逐步到满深度 (6,3)
- 结果：训练加速 1.71x，精度仅降 0.63%

### 6.5 Vision TRM (ViTRM)

- 论文：[arXiv:2603.19503](https://arxiv.org/abs/2603.19503)（2026.03）
- 将 TRM 的递归状态精化范式迁移到视觉：用 3 层 block 递归替代 L 层 ViT
- 参数量减少 6~84 倍，CIFAR-10/100 上保持竞争力

---

## 七、核心技术拆解

### 7.1 Fast Weights（快权重）

- 不是所有参数都需要在训练时固定
- 把一部分参数设为"快权重"，在推理时根据输入做梯度更新
- 本质上是在模型内部嵌入一个在线学习器
- 常见选择：MLP 投影矩阵（In-Place TTT）、hidden state 本身（TTT Layers）、LoRA adapter（VDS-TTT）

### 7.2 Self-Supervised Inner Loop（自监督内循环）

- TTT 的"训练"不需要外部标签
- 内循环用一个自监督任务更新快权重
- 外循环（正常预训练）学习的是"如何做自监督更新"本身
- 这就是 "Learning to Learn at Test Time" 的含义：外循环学元学习器，内循环在测试时执行

### 7.3 Expressive Hidden State（表达性隐藏状态）

| 架构 | Hidden State | 表达能力 | 复杂度 |
|------|-------------|---------|--------|
| 标准 RNN | 固定大小向量 | 弱 | O(1) |
| Mamba | 固定大小向量 + 选择性 | 中 | O(1) |
| TTT-Linear | 线性模型参数 | 中强 | O(1) 更新 |
| TTT-MLP | 两层 MLP 参数 | 强 | O(1) 更新 |
| Attention | 完整 KV cache | 最强 | O(n²) |

### 7.4 线性注意力等价性

TTT 的理论统一意义：

```
内循环学习器 = 线性模型  →  等价于线性注意力
内循环学习器 = 核估计    →  等价于自注意力
内循环学习器 = 神经网络  →  超越两者的新架构
```

这意味着 TTT 不是另起炉灶，而是统一了 RNN 和 Attention 的更一般框架。

### 7.5 目标函数对齐

- 原始 TTT：通用自监督重建目标 → 快权重更新方向可能与下游任务不一致
- In-Place TTT：与 Next-Token Prediction 对齐的目标 → 更新直接服务语言建模
- aTTT：token 级重加权 → 降低重复 n-gram 的 loss 权重，防止自训练漂移

### 7.6 分块更新与并行

- 将序列分成 chunk，每个 chunk 内做一次快权重更新
- 与上下文并行（context parallelism）兼容
- In-Place TTT 验证了该方案可扩展到 128k+ 上下文
- Titans 的神经记忆支持并行化训练

---

## 八、TTT 与相关领域的关系

### 8.1 TTT vs 持续学习 (Continual Learning)

| 维度 | 传统持续学习 | TTT |
|------|------------|-----|
| 更新时机 | 训练阶段 | 推理阶段 |
| 更新对象 | 全部或大部分参数 | 快权重子集 |
| 灾难性遗忘 | 核心问题 | 天然缓解（主参数不变） |
| 数据需求 | 需要任务流数据 | 自监督，无需外部标签 |
| 更新频率 | 每个任务/epoch | 每个样本/chunk |

### 8.2 TTT vs In-Context Learning (ICL)

- ICL：通过上下文窗口中的示例引导模型行为，不更新参数
- TTT：通过梯度更新改变模型参数/状态，效果更持久
- 关系：TTT 的 hidden state 更新可以看作"压缩版的 ICL"——把上下文信息压缩到参数中

### 8.3 TTT vs 检索增强 (RAG)

- RAG：从外部数据库检索相关信息，拼接到上下文
- TTT：将信息"学习"到模型参数中，不需要外部存储
- 互补性：RAG 适合精确事实，TTT 适合分布适应和技能学习

### 8.4 TTT 与 RSI（递归自我改进）

RSI 的系统栈（来自 ICLR 2026 RSI Workshop）：

| 层级 | 功能 | TTT 的角色 |
|------|------|-----------|
| 课程生成 | 决定下一个训练任务 | TTT 的反馈可指导课程 |
| 执行与求解 | 完成任务 | TTT 在执行中持续适应 |
| 验证与奖励 | 评估方案质量 | — |
| 诊断与失败分析 | 理解失败原因 | — |
| 记忆与持久化 | 跨迭代保留知识 | TTT 的快权重是一种参数化记忆 |
| 元学习与效率 | 学会高效学习 | TTT 本质上是 meta-learning |
| 研究自动化 | AI 改进 AI 研究 | — |

---

## 九、论文索引

### 核心论文

| 论文 | 年份 | 关键贡献 | arXiv |
|------|------|---------|-------|
| Test-Time Training with Self-Supervision | 2020 | 首次提出 TTT 概念 | [1909.13231](https://arxiv.org/abs/1909.13231) |
| Learning to (Learn at Test Time) | 2024 | TTT Layers，统一 RNN 与 Attention | [2407.04620](https://arxiv.org/abs/2407.04620) |
| Titans: Learning to Memorize at Test Time | 2025 | 神经长期记忆模块 | [2501.00663](https://arxiv.org/abs/2501.00663) |
| In-Place Test-Time Training | 2026 | Drop-in LLM 增强，目标对齐 | [2604.06169](https://arxiv.org/abs/2604.06169) |
| Agentic Test-Time Training (aTTT) | 2026 | 多轮 Agent 持续 TTT | [2607.03441](https://arxiv.org/abs/2607.03441) |

### 拓展论文

| 论文 | 年份 | 方向 | arXiv |
|------|------|------|-------|
| TTT++: When Does TTT Fail or Thrive? | 2021 | TTT 失效条件分析 | [2007.10964](https://arxiv.org/abs/2007.10964) |
| VDS-TTT: Verifier-Driven Sample Selection | 2025 | 验证器驱动选样 + LoRA TTT | [2505.19475](https://arxiv.org/abs/2505.19475) |
| TEMPO: Scaling TTT for Reasoning Models | 2026 | TTT 扩展到推理模型 | [2604.19295](https://arxiv.org/abs/2604.19295) |
| Specialization after Generalization | 2025 | TTT 理论理解 | [2509.24510](https://arxiv.org/abs/2509.24510) |
| Federated Nested Learning (FedNL) | 2026 | 联邦学习 + Titans | [2605.16350](https://arxiv.org/abs/2605.16350) |

### Tiny Recursive Models 系列

| 论文 | 年份 | 关键贡献 | arXiv |
|------|------|---------|-------|
| Tiny Recursive Models on ARC-AGI | 2025 | 原始 TRM，7M 参数 | [2511.02886](https://arxiv.org/abs/2511.02886) |
| Test-time Adaptation of TRM | 2025 | TRM 测试时微调 | [2511.02886](https://arxiv.org/abs/2511.02886) |
| CGAR: 加速 TRM 训练 | 2025 | 课程引导自适应递归 | [2511.08653](https://arxiv.org/abs/2511.08653) |
| Probabilistic TRM (PTRM) | 2026 | 随机探索 + Q head 选择 | [2605.19943](https://arxiv.org/abs/2605.19943) |
| Vision TRM (ViTRM) | 2026 | TRM 迁移到视觉 | [2603.19503](https://arxiv.org/abs/2603.19503) |
| Tiny Autoregressive Recursive Models | 2026 | TRM 迁移到自回归 | [2603.08082](https://arxiv.org/abs/2603.08082) |
| TRM on ARC-AGI-1: 分析报告 | 2025 | TRM 性能归因分析 | [2512.11847](https://arxiv.org/abs/2512.11847) |

### RSI Workshop 相关

| 论文 | 类型 | OpenReview |
|------|------|-----------|
| ALMA: Meta-learning Memory Designs | Oral | [sOq52KnJmR](https://openreview.net/forum?id=sOq52KnJmR) |
| Test-Time Self-Distillation (SDPO) | Spotlight | [iEmRSwdzyw](https://openreview.net/forum?id=iEmRSwdzyw) |
| Adaptive Meta-Curriculum for TTSI | Spotlight | [GjoUJTfXiW](https://openreview.net/forum?id=GjoUJTfXiW) |
| RSI Workshop 主页 | — | [recursive-workshop.github.io](https://recursive-workshop.github.io/) |

---

## 十、延伸阅读

- [Lilian Weng: Harness Engineering for Self-Improvement](https://lilianweng.github.io/posts/2026-07-04-harness/)（2026.07）— RSI 全景综述，将 TTT 定位为 RSI 核心组件
- [Artifocial: From Self-Play to Self-Research](https://artifocial.com/blog/rsi-workshop-2026-mar-18)（2026.03）— ICLR 2026 RSI Workshop 论文解读
- [Awesome Recursive Self-Improvement](https://github.com/natnew/awesome-recursive-self-improvement) — RSI 论文合集
- → 相关文档：[Agent记忆与知识自进化](../7.product/Memory.md)、[长上下文](./LongContext.md)
