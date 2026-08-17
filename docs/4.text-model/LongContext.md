# 长上下文 (Long Context)

> 10M token 上下文 ≈ 700万汉字 / 32MB 文本。研究方向：架构创新、记忆机制、评估基准。

## 综述

- [2025.03] [A Survey on Transformer Context Extension: Approaches and Evaluation](https://arxiv.org/abs/2503.13299) 长上下文扩展方法与评估综述

---

## 厂商注意力优化方案（1M+ 上下文）

> 聚焦各厂商在百万 token 上下文场景下对注意力机制的具体优化。核心问题：标准注意力 O(n²) 复杂度在 1M 长度下不可行，需要架构级创新。

### DeepSeek：MLA → DSA → CSA+HCA

DeepSeek 在注意力优化上经历了三代演进，是目前最系统的长上下文注意力优化路线。

**第一代：MLA (Multi-head Latent Attention)** — DeepSeek-V2/V3

- 论文：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)（2024.12）
- 原理：将 Key 和 Value 压缩到低秩潜在空间（latent space），推理时只需缓存压缩后的潜在向量，而非完整的 KV pair。解码时通过上投影矩阵重建 KV
- 效果：KV cache 减少约 93%（相比标准 MHA），大幅降低长上下文的显存压力
- 上下文：128K（V3）

**第二代：DSA (DeepSeek Sparse Attention)** — DeepSeek-V3.2

- 论文：[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)（2025.12）
- 原理：稀疏注意力机制，在保持模型性能的前提下大幅降低长上下文场景的计算复杂度
- 意义：V3.2 是 V3 到 V4 的过渡，DSA 为 V4 的 CSA+HCA 混合注意力奠定了基础

**第三代：CSA + HCA 混合注意力** — DeepSeek-V4

- 论文：[DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348)（2026.04）
- 上下文：1M（标准），Pro 1.6T 参数 / 49B 激活，Flash 284B / 13B 激活
- CSA (Compressed Sparse Attention)：沿序列维度压缩 KV cache，然后在压缩后的 KV 上执行 DSA。两层压缩降低计算量
- HCA (Heavily Compressed Attention)：比 CSA 更激进的 KV 压缩，但保持 dense attention（不做稀疏选择）。适合需要全局信息的层
- 混合策略：不同层使用不同注意力类型，CSA 和 HCA 交替部署，平衡效率和表达力
- 效果：1M token 设置下，单 token 推理 FLOPs 仅为 V3.2 的 27%，KV cache 仅为 10%
- 其他优化：FP4 量化感知训练（MoE 权重 + CSA 索引器 QK 路径），Muon 优化器

### Kimi (Moonshot AI)：MoBA → Kimi Linear → KDA (K3)

三代演进：块级稀疏注意力 → 线性注意力 → Delta Attention。

**MoBA (Mixture of Block Attention)** — K2 时代

- 论文：[MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189)（2025.02）
- 原理：将上下文划分为块（block），借鉴 MoE 思想，通过门控机制动态选择每个 query 需要关注哪些块。无参数的块级稀疏注意力，可根据序列长度自适应调整稀疏度
- 特点：可在全注意力和稀疏注意力之间无缝切换；已在 Kimi 产品中验证
- 上下文：Kimi 支持 1M+（MoBA 使其可行）
- 后续优化：[Optimizing Mixture of Block Attention](https://arxiv.org/abs/2511.11571)（MIT Song Han 团队, 2025.11）

**Kimi Linear**

- 论文：[Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)（2025.10）
- 原理：新的线性注意力架构，兼顾表达力和效率

**Kimi Delta Attention (KDA)** — Kimi K3

- 论文：[Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)（2026.07）
- 上下文：1M token，2.8T 参数 MoE / 104B 激活，原生多模态
- KDA 原理：基于 [Delta Attention Residuals](https://arxiv.org/abs/2605.18855)（2026.05）。标准 Attention Residuals 对前层累积 hidden states 做 softmax 路由，但深层累积状态高度冗余，导致路由坍缩（max weight ≈0.2）。KDA 改为对"delta"——每层引入的变化量 (v_i = h_{i+1} - h_i)——做注意力路由。Delta 表示结构性更多样，注意力分布对比度更高（max weight ≈0.6），实现更选择性的跨层路由
- Attention Residuals：用学习到的 softmax 注意力替代标准加法残差连接，实现选择性跨层信息路由
- Stable LatentMoE：896 个路由专家中每 token 激活 16 个
- 效果：相比 K2 整体缩放效率提升约 2.5x

### MiniMax：Lightning Attention → MSA (M3)

两代演进：线性注意力 → 块级稀疏注意力。

**Lightning Attention** — Text-01 / M1 / M2

- 论文：[MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)（2025.06）
- 架构：混合注意力（hybrid-attention），部分层用 Lightning Attention（线性注意力），部分层用标准注意力
- Lightning Attention 原理：通过分块计算（tiling）和核函数技巧将注意力复杂度从 O(n²) 降至 O(n)，利用左乘右乘重排实现推理恒定速度
- 相关论文：[Lightning Attention-2](https://arxiv.org/abs/2401.04658)
- 上下文：M1 原生支持 1M token（8x DeepSeek R1）
- M2 系列：[The MiniMax-M2 Series](https://arxiv.org/abs/2605.26494)（2025），229.9B 总参数 / 9.8B 激活，面向 agentic 部署
- Ling and Ring 2.6：[技术报告](https://arxiv.org/abs/2606.15079)（2026.06），万亿参数规模混合架构，验证线性注意力在超大规模的可行性

**MSA (MiniMax Sparse Attention)** — MiniMax-M3

- 论文：[MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392)（2026.06）
- 上下文：1M，109B 参数，原生多模态训练。模型已开源：[HuggingFace](https://huggingface.co/MiniMaxAI/MiniMax-M3)
- 架构：基于 GQA 的块级稀疏注意力，双分支设计：
  - Index Branch（索引分支）：轻量级，对 KV 块打分，为每个 GQA group 独立选择 Top-k 子集。实现 group-specific 稀疏检索
  - Main Branch（主分支）：对选中的块执行精确 block-sparse attention
- 工程协同设计：exp-free Top-k 选择 + KV-outer sparse attention，提升 tensor-core 利用率
- 效果：1M 上下文下每 token 注意力计算量减少 28.4x；H800 上 prefill 加速 14.2x、decoding 加速 7.6x
- 与 Lightning Attention 的关系：M3 从线性注意力转向稀疏注意力路线，可能是为了在超长上下文下获得更好的精度-效率平衡

### Qwen (阿里)：标准注意力 + 工程优化

- 论文：[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)（2025.05）
- 架构：Dense + MoE 两种架构，参数 0.6B~235B。Qwen3 技术报告中未报告专门的注意力机制创新，主要依赖标准 GQA + RoPE + YaRN 进行上下文扩展
- 上下文：Qwen2.5-Turbo 支持 1M；Qwen3 系列支持 128K~256K（通过 YaRN 外推扩展）
- 特点：thinking/non-thinking 双模式，thinking budget 机制。模型能力侧重多语言（119 种语言）和推理，注意力层面以工程优化为主

### Meta Llama 4：10M 上下文

- 上下文：Llama 4 Scout 原生 10M token
- 架构细节：Meta 未发布 Llama 4 的学术论文。公开信息显示采用了 iRoPE（interleaved RoPE）和注意力稀疏化策略，但具体机制未公开
- 相关工程实践：[Arctic Long Sequence Training](https://arxiv.org/abs/2506.13996)（2025.06）讨论了多百万 token 序列训练的可扩展方案

### 智谱 GLM：标准注意力 → MLA+DSA (GLM-5/5.2)

**GLM-4** — 基础阶段

- 论文：[ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools](https://arxiv.org/abs/2406.12793)（2024.06）
- 架构：GLM-4-9B 支持 128K/1M 上下文，标准注意力 + RoPE 位置扩展

**GLM-5** — 引入 DSA

- 论文：[GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763)（2026.02）
- 架构：采用 DSA (DeepSeek Sparse Attention) 显著降低训练和推理成本，同时保持长上下文保真度。异步 RL 基础设施解耦生成与训练
- 意义：GLM-5 开始跟进 DeepSeek 的 DSA 路线，从标准注意力转向稀疏注意力

**GLM-5.2** — MLA + DSA + IndexShare

- 架构（来源：[LongStraw](https://arxiv.org/abs/2607.14952) 论文对 GLM-5.2 架构的描述）：
  - 78 层 MLA/DSA 注意力堆栈 + 256 专家 top-8 MoE 尾部
  - MLA latent pages 和 DSA indexer-key pages 在 CPU 内存中分片存储（CP-sharded）
  - IndexShare 机制：跨上下文并行 ranks（CP32）重建 IndexShare-aware 全局稀疏选择。DSA 的索引（决定哪些 KV 块参与注意力）在多个 CP rank 间共享，使分布式稀疏注意力能够全局协调
  - 响应 token 通过 EP32（专家并行 32 路）分发
- 特点：GLM-5.2 的注意力栈本质上和 DeepSeek V3.2 同源（MLA+DSA），但通过 IndexShare 解决了 DSA 在分布式训练/推理中的索引共享问题

### Google Gemini：2M 上下文

- 上下文：Gemini 1.5 Pro 2M，Gemini 2.0 Pro 2M+
- 架构：Google 未公开 Gemini 的注意力架构细节。业界推测采用了混合注意力（局部+全局）和高效的 KV cache 管理，但无公开论文确认

### Mistral：滑动窗口注意力

- 架构：Mistral 7B 首创 sliding window attention（窗口大小 4096），通过层级堆叠实现有效上下文扩展。每层只关注局部窗口，但通过多层堆叠，信息可以间接传播到更远
- 演进：Mistral Large 2 支持扩展上下文，具体注意力优化未公开

### 厂商方案对比

| 厂商 | 模型 | 注意力机制 | 复杂度 | 上下文 | 论文 |
|------|------|-----------|--------|--------|------|
| DeepSeek | V4 | CSA+HCA 混合 + MLA | 次线性 | 1M | [2606.19348](https://arxiv.org/abs/2606.19348) |
| DeepSeek | V3 | MLA (低秩压缩) | O(n) 缓存 | 128K | [2412.19437](https://arxiv.org/abs/2412.19437) |
| Kimi | K3 | KDA (Delta Attention Residuals) | 次线性 | 1M | [2607.24653](https://arxiv.org/abs/2607.24653) |
| Kimi | K2 | MoBA (块级稀疏) | O(n·k) | 1M+ | [2502.13189](https://arxiv.org/abs/2502.13189) |
| MiniMax | M3 | MSA (Index+Main 双分支稀疏) | 次线性 | 1M | [2606.13392](https://arxiv.org/abs/2606.13392) |
| MiniMax | M1 | Lightning Attention (线性) | O(n) | 1M | [2506.13585](https://arxiv.org/abs/2506.13585) |
| 智谱 | GLM-5.2 | MLA + DSA + IndexShare | 次线性 | 1M | [LongStraw](https://arxiv.org/abs/2607.14952) |
| 智谱 | GLM-5 | DSA | 次线性 | 1M | [2602.15763](https://arxiv.org/abs/2602.15763) |
| Qwen | Qwen3 | GQA + YaRN | O(n²) | 128K-1M | [2505.09388](https://arxiv.org/abs/2505.09388) |
| Meta | Llama 4 Scout | iRoPE + 稀疏化 (未公开) | 未知 | 10M | 无公开论文 |
| Google | Gemini 2.0 | 未公开 | 未知 | 2M+ | 无公开论文 |
| Mistral | Mistral | Sliding Window | O(n·w) | 128K | 无公开论文 |

> 趋势：国内厂商在注意力架构创新上全面铺开，DeepSeek/智谱走 MLA+DSA 路线，Kimi 走 Delta Attention 路线，MiniMax 从线性注意力转向稀疏注意力。一个明显的趋势是各家从 V1 的线性/标准注意力方案，在 V2/V3 转向稀疏注意力方案（Kimi: MoBA→KDA, MiniMax: Lightning→MSA, DeepSeek: MLA→DSA→CSA+HCA, GLM: 标准注意力→DSA+MLA）。

---

## 注意力机制优化：学术论文

### 稀疏注意力 (Sparse Attention)

#### 基础工作

- **Longformer** [2020.04] [论文](https://arxiv.org/abs/2004.05150) Allen AI。滑动窗口注意力 + 全局注意力，O(n²)→O(n)。局部 token 用滑动窗口，特定 token（如 CLS）用全局注意力
- **BigBird** [2020.07] [论文](https://arxiv.org/abs/2007.14062) Google。稀疏注意力 = 随机 + 窗口 + 全局。理论上证明是通用近似器且图灵完备

#### 最新进展

- [2025.02] [NSA: Native Sparse Attention](https://arxiv.org/abs/2502.11089) DeepSeek。动态分层稀疏策略：粗粒度 token 压缩 + 细粒度 token 选择。两大创新：(1) 算术强度均衡的硬件对齐设计；(2) 支持端到端训练。64k 序列上解码/前向/反向均显著加速，性能保持或超越 Full Attention
- [2025.02] [MoBA: Mixture of Block Attention](https://arxiv.org/abs/2502.13189) Moonshot AI。MoE 式块选择，门控动态路由，无参数设计，全/稀疏注意力无缝切换。与 NSA 同日发布，技术路径互补
- [2024.02] [InfLLM](https://arxiv.org/abs/2402.04617) 清华大学。免训练长上下文外推：块级记忆 + 相似度检索动态加载。内存随块数线性增长
- [2025.09] [InfLLM-V2](https://arxiv.org/abs/2509.24663) 清华大学。密集-稀疏可切换注意力，根据序列长度动态切换全/稀疏模式，解决短序列性能下降问题
- [2024.01] [Lightning Attention-2](https://arxiv.org/abs/2401.04658) 线性注意力变体，分块+核函数 O(n²)→O(n)，推理恒定速度
- [2026.06] [MiniMax Sparse Attention (MSA)](https://arxiv.org/abs/2606.13392) MiniMax。基于 GQA 的块级稀疏注意力，双分支：Index Branch 对 KV 块打分选 Top-k，Main Branch 做精确 block-sparse attention。exp-free Top-k + KV-outer 稀疏注意力协同 GPU 设计。1M 上下文计算量减少 28.4x，H800 prefill 14.2x / decoding 7.6x
- [2026.05] [Delta Attention Residuals](https://arxiv.org/abs/2605.18855) Attention Residuals 的改进。标准 Attention Residuals 对前层累积 hidden states 路由，深层冗余导致路由坍缩（max weight ≈0.2）。改为对 delta（每层变化量 v_i = h_{i+1} - h_i）路由，对比度更高（max weight ≈0.6）。220M~7.6B 规模 perplexity 提升 1.7~8.2%。Kimi K3 的 KDA 基于此
- [2026.08] [LongCat Sparse Attention](https://arxiv.org/abs/2608.01662) 美团。流式感知分层跨层索引，跨层共享索引信息
- [2025.11] [Optimizing MoBA](https://arxiv.org/abs/2511.11571) MIT。对 MoBA 的进一步优化
- [2025.11] [Optimizing NSA](https://arxiv.org/abs/2511.00819) 美团。潜在注意力和局部-全局交替策略优化 NSA

#### 分布式稀疏注意力工程

- [2026.07] [LongStraw](https://arxiv.org/abs/2607.14952) 长上下文 RL 系统，支持 2M+ token 的 GRPO 训练。对 GLM-5.2 实现 IndexShare-aware 全局稀疏选择（跨 CP32 ranks 共享 DSA 索引），对 Qwen3.6 实现 GDN 循环状态 + CP8 分片 KV 的跨 rank 合并

### 线性注意力与非 Transformer 架构

#### SSM (State Space Model)

- [2023.12] [Mamba](https://arxiv.org/abs/2312.00752) 选择性 SSM，参数为输入函数实现内容感知。推理吞吐 5x Transformer，序列线性扩展
- [2024.05] [Mamba-2](https://arxiv.org/abs/2405.21060) ICML 2024。状态空间对偶性 (SSD) 框架，揭示 SSM 和注意力的理论联系。核心层比 Mamba-1 快 2-8x

#### RNN 架构

- [2025.03] [RWKV-7 "Goose"](https://arxiv.org/abs/2503.14456) 表达式动态状态演化。纯 RNN 架构，推理恒定内存。RWKV-7 在状态跟踪和长序列建模上显著优于前代
  - [RWKV-X](https://arxiv.org/abs/2504.21463)：线性复杂度混合语言模型
  - [ARWKV](https://arxiv.org/abs/2501.15570)：从 Transformer 权重转换的 RNN-注意力混合模型，免预训练

#### 线性 Transformer

- [2023.07] [TransNormerLLM](https://arxiv.org/abs/2307.14995) 改进的线性注意力 LLM 架构
- [2025.10] [Kimi Linear](https://arxiv.org/abs/2510.26692) Moonshot AI。兼顾表达力和效率的线性注意力架构

### KV Cache 压缩与管理

#### 驱逐策略 (Eviction)

- [2023.06] [H2O (Heavy-Hitter Oracle)](https://arxiv.org/abs/2306.14048) UT Austin/Stanford。发现"重击者"token，动态保留高注意力分数 token，KV cache 降至 ~10%
- [2023.09] [StreamingLLM (Attention Sinks)](https://arxiv.org/abs/2309.17453) MIT/Meta。发现"注意力槽"现象：初始 token 获得不成比例的高注意力。保留注意力槽 + 最近窗口 → 无限流式推理。ICLR 2024
- [2024.04] [SnapKV](https://arxiv.org/abs/2404.14469) 利用 prefill 阶段注意力模式识别重要 KV 对，预先压缩
- [2024.06] [PyramidKV](https://arxiv.org/abs/2406.02069) 金字塔式层级 cache 分配：浅层多、深层少
- [2024.06] [QUEST](https://arxiv.org/abs/2406.10774) MIT。查询感知稀疏注意力：先近似估计再精确计算。ICML 2024
- [2024.07] [ThinK](https://arxiv.org/abs/2407.21018) 查询驱动 Key 通道维度剪枝

#### 量化 (Quantization)

- [2024.02] [KIVI](https://arxiv.org/abs/2402.02750) 免调参非对称 2bit 量化。Key 按通道、Value 按 token，KV cache 内存减少 8x
- [2024.01] [KVQuant](https://arxiv.org/abs/2401.18079) UC Berkeley。面向 10M token 的 KV cache 量化。非均匀量化 + 注意力槽保留。NeurIPS 2024
- [2024.03] [GEAR](https://arxiv.org/abs/2403.05527) Georgia Tech。量化 + 低秩近似补偿，近无损压缩

#### 最新趋势 (2025-2026)

- [EpiKV](https://arxiv.org/abs/2606.26472)：无需注意力矩阵的 KV 驱逐，"顿悟分数"
- [XQuant](https://arxiv.org/abs/2510.11236)：跨层压缩超低比特 KV 量化
- [H1B-KV](https://arxiv.org/abs/2510.05529)：混合 1bit cache

### 分布式注意力计算

- [2023.10] [Ring Attention](https://arxiv.org/abs/2310.01889) UC Berkeley。分块注意力 + 环形通信，GPU 间异步传递 KV 块，近无限上下文。通信与计算重叠
- [2023.10] [DistFlashAttn](https://arxiv.org/abs/2310.03294) FlashAttention 的多 GPU 扩展
- [2024.05] [USP (Unified Sequence Parallelism)](https://arxiv.org/abs/2405.07719) 统一 Ring Attention 和 Ulysses 的序列并行框架
- [2024.06] [LoongTrain](https://arxiv.org/abs/2406.18485) 头-上下文并行，沿注意力头和上下文维度同时切分
- [2024.11] [Context Parallelism](https://arxiv.org/abs/2411.01783) Meta。面向百万 token 推理的上下文并行，已生产部署
- [2024.12] [TokenRing](https://arxiv.org/abs/2412.20501) 双向通信改进 Ring Attention
- [2025.09] [TASP](https://arxiv.org/abs/2509.26541) 拓扑感知序列并行
- [2025.10] [MTraining](https://arxiv.org/abs/2510.18830) 分布式动态稀疏注意力，超长上下文训练
- [2026.02] [Untied Ulysses](https://arxiv.org/abs/2602.21196) 头级分块内存高效上下文并行

### 混合架构

- [2024.03] [Jamba](https://arxiv.org/abs/2403.19887) AI21 Labs。Transformer + Mamba SSM 交替（1:7 比例）+ MoE。256K 上下文
- [2024.08] [Jamba-1.5](https://arxiv.org/abs/2408.12570) 12B/94B 参数（94B MoE, 12B 活跃），256K 上下文
- [2026.02] [MiniCPM-SALA](https://arxiv.org/abs/2602.11761) 面壁智能。混合稀疏 + 线性注意力，端侧友好
- [2025.01] [ARWKV](https://arxiv.org/abs/2501.15570) 从 Transformer 演化的 RNN-注意力混合，可从已有权重转换

### 位置编码扩展

- [2023.09] [YaRN](https://arxiv.org/abs/2309.00071) NTK-aware 插值 + 温度缩放，~400 步微调扩展 4x 上下文。最广泛使用的 RoPE 扩展方法
- [2026.07] [AdaRoPE](https://arxiv.org/abs/2607.19363) ICML 2026。不同注意力头使用不同旋转/缩放参数
- [2026.07] [Jet-Long](https://arxiv.org/abs/2607.07740) 动态双焦 RoPE，近焦局部精确 + 远焦全局粗略
- [2025.04] [Dimension-Wise Positional Embeddings](https://arxiv.org/abs/2504.18857) 按维度操作 RoPE，比 YaRN 分组策略更细粒度
- [2026.06] [Randomized YaRN](https://arxiv.org/abs/2606.23687) YaRN + 随机化位置编码，改善长上下文推理的长度泛化
- RoPE (Rotary Position Embedding)：主流长上下文位置编码基础
- ALiBi (Attention with Linear Biases)：无需位置编码，线性偏置实现长度外推

---

## 记忆与推理增强

### 记忆模块

- [2025.05] [ATLAS: Learning to Optimally Memorize the Context at Test Time](https://arxiv.org/abs/2505.23735) Google。DeepTransformers 架构，可训练记忆模块替代注意力层。10M BABILong +80%。Titans 改进版
- [2024.04] [Infini-attention](https://arxiv.org/abs/2404.07143) Google。压缩记忆融入 vanilla attention，masked local + long-term linear。1B/8B 验证 1M passkey 和 500K 书籍摘要
- [2024.07] [Needle in the Haystack for Memory Based LLMs](https://arxiv.org/abs/2407.01437) IBM。Larimar 外部关联记忆，无需长上下文训练

### 情景记忆

- [2024.07] [EM-LLM](https://arxiv.org/abs/2407.09450) ICLR 2025。Bayesian surprise + 图论边界分割"事件"，无需微调，10M token 检索

### 多记忆系统

- **LIGHT (BEAM 框架)**：长时情景记忆 + 短期工作记忆 + 草稿本。10M 对话提升 3.5%-12.69%
- **MELODI**：记忆压缩，短上下文窗口处理长文档
- **Recursive Language Models (RLM)** MIT。编程方式分块、检索和处理

---

## 评估与基准测试

### 10M 级别

- [2026.01] [CorpusQA](https://arxiv.org/abs/2601.14952) 首个 10M token 语料库级基准。发现 SOTA 模型随长度退化，RAG 崩溃，记忆增强 agent 更鲁棒
- **BEAM**：10M token 连贯对话，含 LIGHT 多记忆系统评估
- **BABILong**：广泛采用的 10M 长度测试

### 百万 token 级别

- **∞-Bench (InftyBench)**：平均 100K+ token 样本
- **LongBench**：多任务中文/英文长上下文理解
- **Needle in a Haystack (NIAH)**：长文本检索标准测试
- **RULER**：合成长上下文基准（multi-key/multi-value/multi-query/variable tracing/aggregation）
- [2024.09] [LongMT-Bench](https://arxiv.org/abs/2409.12640) 长上下文多轮对话一致性

### RAG 与长上下文

- [2025.11] [Q-RAG](https://arxiv.org/abs/2511.07328) 长上下文多步检索，value-based embedder 训练

---

## 应用场景

- **软件工程**：整个代码仓库，全局代码审查、漏洞检测、跨模块重构
- **法律合规**：多年法律档案、诉讼记录，案例比对和风险分析
- **金融分析**：数年交易记录、财报、市场数据，风险评估和欺诈检测
- **生物医药**：临床试验档案、医学文献、基因组数据
- **学术研究**：领域论文全集，自动文献综述
- **长视频理解**：多模态模型处理数小时视频
- **Agent 工作流**：完整多步骤 agent 工作记忆
- **超长对话**：跨月对话记忆，个性化
