# LLM 推理性能分析

> 2026.08.31 | 基于理论推导 + arXiv:2502.02789 (SpecPrefill, ICML 2025) + NVIDIA 官方规格

## 1. 推理两阶段：根本差异

| | Prefill (→TTFT) | Decode (→TPOT) |
|---|---|---|
| 计算类型 | GEMM（矩阵×矩阵） | GEMV（矩阵×向量） |
| 瓶颈 | 算力（FLOPS） | 显存带宽（TB/s） |
| 每token读权重 | S个token分摊1次 | 每个token读1次完整权重 |
| Attention | O(S²) 对全序列 | O(n) 对单个token |
| 公式 | 2PS / FLOPS + attn(S²) | W_bytes / BW |
| 随prompt长度 | 线性 + 二次 | 不变 |

## 2. TPOT：为什么普遍在 10-20ms / 50-100 tok/s

### 2.1 硬公式

```
TPOT_min = 模型权重总字节数 / 显存总带宽
```

这是物理下限，无法通过优化绕过。

### 2.2 理论计算

| 配置 | 权重(GB) | 带宽(TB/s) | 理论TPOT(ms) | 理论tok/s |
|---|---|---|---|---|
| 7B FP16 / A100 | 14 | 2.0 | 7.0 | 143 |
| 7B FP16 / H100 | 14 | 3.4 | 4.2 | 239 |
| 13B FP16 / A100 | 26 | 2.0 | 13.0 | 77 |
| 13B FP16 / H100 | 26 | 3.4 | 7.8 | 129 |
| 70B FP16 / 4xA100 | 140 | 8.0 | 17.5 | 57 |
| 70B FP16 / 4xH100 | 140 | 13.4 | 10.4 | 96 |
| 70B FP8 / 4xH100 | 70 | 13.4 | 5.2 | 191 |
| 67B FP8 / 2xH100 | 67 | 6.7 | 10.0 | 100 |
| 405B FP8 / 8xH100 | 405 | 26.8 | 15.1 | 66 |

### 2.3 为什么收敛到这个区间

1. **模型越大→用的卡越多→带宽同步增大**：分子（权重大小）和分母（总带宽）基本同步增长，比值天然趋于常数。这不是巧合，而是部署逻辑决定的——用多少卡取决于模型能塞进多少显存。
2. **硬件代际锚定**：当前一代GPU的HBM带宽是物理常数（A100=2.0, H100=3.35, B200=8.0 TB/s），模型大小×量化精度的组合天然在"适配"这个带宽常数。
3. **固定开销叠加**：实际TPOT = 理论值 + KV cache读取 + attention计算 + kernel启动 + 采样 ≈ +3~8ms。所以理论5-17ms → 实际8-25ms → 40-125 tok/s，去掉极端值就是 10-20ms / 50-100 tok/s。

## 3. TTFT：取决于什么

### 3.1 公式

```
TTFT = (2PS + 4LH·S²) / FLOPS + 固定开销
```

- P = 参数量，S = prompt长度，L = 层数，H = head_dim × num_heads
- 线性层(GEMM)：2PS → 随prompt线性增长
- Attention：4LH·S² → 随prompt平方增长

### 3.2 Attention 占比随 prompt 长度变化（70B / 4xH100）

| Prompt长度 | GEMM(TF) | Attention(TF) | Attn占比 | 理论TTFT(ms) |
|---|---|---|---|---|
| 100 | 14.0 | 0.0 | 0.2% | 3.5 |
| 500 | 70.0 | 0.7 | 0.9% | 17.9 |
| 2000 | 280.0 | 10.5 | 3.6% | 73.4 |
| 4000 | 560.0 | 41.9 | 7.0% | 152.2 |
| 8000 | 1120.0 | 167.8 | 13.0% | 325.5 |
| 16000 | 2240.0 | 671.1 | 23.1% | 735.9 |
| 32000 | 4480.0 | 2684.4 | 37.5% | 1811.0 |

转折点：S≈4000 时 attention 开始显著，S>8000 后 attention 主导。

### 3.3 影响因素排序

| 因素 | 影响方式 | 程度 |
|---|---|---|
| Prompt长度(S) | GEMM线性 + Attn平方 | ★★★★★ 最大因素 |
| 模型大小(P) | GEMM线性增长 | ★★★★ |
| GPU算力(FLOPS) | 反比，直接决定下限 | ★★★★ |
| Attention O(S²) | 长上下文时主导 | ★★★ |
| TP通信(all-reduce) | 每层2次，层数越多越重 | ★★★ |
| KV Cache分配 | 需为S个token分配显存 | ★★ 5-20ms |
| 调度排队 | 取决于系统负载 | ★★~★★★★ |
| 网络延迟 | 远程调用时 | ★★ 10-50ms |

TP通信开销估算：7B(32层)≈64ms, 70B(80层)≈160ms, 405B(126层)≈252ms

### 3.4 算力 vs 带宽——不同阶段吃不同资源

| GPU | 算力(TF FP16) | 带宽(TB/s) | 算力/带宽比 |
|---|---|---|---|
| A100 | 312 | 2.0 | 156 |
| H100 | 989 | 3.35 | 295 |

H100 比 A100 算力提升 3.2x，带宽只提升 1.7x：
- 对 TTFT（算力受限）：快 ~3 倍
- 对 TPOT（带宽受限）：快 ~1.7 倍

## 4. NVIDIA 数据中心 GPU 谱系

| GPU | 显存 | 带宽(TB/s) | 算力 | 上市 |
|---|---|---|---|---|
| A100 | 80GB HBM2e | 2.0 | 312 TF FP16 | 2020 |
| H100 | 80GB HBM3 | 3.35 | 989 TF FP16 | 2022.03 |
| H200 | 141GB HBM3e | 4.8 | 989 TF FP16 | 2024.11 |
| B200 | 180GB HBM3e | 8.0 | 1191 TF FP4 | 2024.11 |
| B300 | 288GB HBM3e | 8.0 | 1232 TF FP4 | 2025.09 |
| Rubin | HBM4 | 待确认 | 50 PF FP4 | 2026 H2 |
| Rubin Ultra | 待确认 | 待确认 | 100 PF FP4 | 2027 |

注意：H200 相对 H100 是容量 80→141GB、带宽 3.35→4.8 TB/s（+43%），算力不变；B200/B300 的 8.0 TB/s 是双芯片聚合带宽。

算力/带宽比的代际演进决定 TPOT"引力井"位置：
- 当前：A100/H100 → 10-20ms / 50-100 tok/s
- 下一代：B200/Rubin(HBM4) → 预计 5-10ms / 100-200 tok/s

## 5. 推理性能优化全景

从硅片到 prompt，分六层，每层解决的问题不同：

```
Layer 0  芯片/硬件层     —— 物理上限
Layer 1  模型结构层       —— 减少计算量和参数量
Layer 2  算法层           —— 用策略绕过物理极限
Layer 3  推理引擎/系统层  —— 提高资源利用率
Layer 4  Prompt/请求层    —— 减少输入量
Layer 5  部署架构层       —— 系统级调度
```

优化目标说明：

| 目标 | 含义 | 主要受谁影响 |
|---|---|---|
| TTFT | 首token延迟 | 算力、prompt长度 |
| TPOT | 单token生成延迟 | 带宽、模型大小 |
| Throughput | 聚合吞吐(tokens/s) | batching、调度、并行 |
| Memory | 显存占用(KV cache+权重) | 量化、cache管理 |
| Cost | $/token | 以上所有的 trade-off |

### 5.1 Layer 0: 芯片/硬件层

物理地基，决定一切优化的天花板。

| 方向 | 原理 | 影响 |
|---|---|---|
| HBM带宽提升 | HBM2e→HBM3e→HBM4, A100=2.0→H100=3.35→B200=8.0 TB/s | 直接降TPOT（带宽引力井） |
| 互联带宽 | NVLink/NVSwitch, NVLink4=900GB/s vs PCIe5=64 | 降TP通信开销（TTFT的TP部分） |
| 计算密度 | Tensor Core代际, FP16→FP8→FP4支持 | 降TTFT（算力提升） |
| 显存容量 | 80GB→141GB→288GB | 能装更大模型/更长KV |
| 多芯片封装 | B200双die=8TB/s聚合带宽 | 带宽翻倍但单die不变 |

硬件层是"慢变量"，一代芯片2-3年，但锚定了TPOT的物理下限。是唯一能同时改善TTFT和TPOT的手段，但成本最高。

### 5.2 Layer 1: 模型结构层

不改硬件，改模型本身的"计算形状"。

**Attention 结构优化：**

| 方案 | 原理 | KV cache减少 | 精度影响 |
|---|---|---|---|
| MHA | 每个head独立QKV | 基准 | 基准 |
| MQA | 所有head共享1组KV | ~1/N_heads | 较大 |
| GQA | 分组共享KV(折中) | ~1/G | 小 |
| MLA | 低秩投影压缩KV(DeepSeek) | 大幅 | 极小 |
| Sparse Attn | 只关注局部/token子集 | 不变 | 中等 |
| Linear Attn | 用核函数近似,去softmax | 不变 | 较大 |

MLA 是目前最优解，DeepSeek-V2/V3 采用，KV cache压缩到1/4以上。

**FFN 结构优化：**

| 方案 | 原理 | 激活参数 | 影响 |
|---|---|---|---|
| Dense FFN | 全连接,所有参数参与 | 100% | 基准 |
| MoE | 稀疏激活,只激活Top-K专家 | 5-15% | TPOT↓（读的权重少），但总参数更大 |

MoE 对推理：TPOT降低（每token只激活部分专家，等效"小模型"），但总参数增大（需更多显存，TP切分更复杂）。DeepSeek-V3: 671B总参数, 37B激活 → TPOT接近37B模型。

**量化：**

| 精度 | 字节/参数 | TPOT影响 | 精度损失 |
|---|---|---|---|
| FP16/BF16 | 2 | 基准 | 0 |
| FP8 | 1 | TPOT减半 | <1% |
| INT8 | 1 | TPOT减半 | 1-3% |
| INT4/FP4 | 0.5 | TPOT降4倍 | 3-8% |

量化直接减少"权重字节数"，在 TPOT = W_bytes / BW 公式里分子变小。是性价比最高的TPOT优化手段。

两种路线：
- Weight-only（只量化权重）: W8A16, W4A16 → TPOT降，但计算时反量化
- Weight+Activation: W8A8, W4A4 → TPOT更优，但精度损失大

**架构创新：**

| 方案 | 原理 | 状态 |
|---|---|---|
| Mamba/SSM | 线性复杂度替代attention | 实验中,长序列有优势 |
| Hybrid | 前几层attention+后面SSM | Jamba等在探索 |
| RWKV | 线性RNN架构 | 小众但有社区 |

这些方案把 attention 的 O(S²) 降到 O(S)，理论上对长上下文 TTFT 有质的飞跃，但通用能力尚未追上 Transformer。

模型结构层的优化大多是训练时就决定的，推理时改不了。量化和蒸馏是例外——可以 post-training 做。

### 5.3 Layer 2: 算法层

不改模型、不改硬件，用算法策略"作弊"绕过物理极限。

**Speculative Decoding（投机解码）：**

小模型快速起草 K 个token，大模型一次性批量验证。等效TPOT降低2-5x（取决于接受率），且无损（大模型做最终判定）。代表：EAGLE-3(5.6x), Medusa, Lookahead。

本质：把多个串行的decode步骤，变成一个并行验证步骤。一次带宽读取验证K个token，等效TPOT/K。

**Speculative Prefill：**

小模型筛掉不重要token，压缩prompt。TTFT提升最高7.66x，有损(<5%)。详见第6节。

**其他算法策略：**

| 策略 | 原理 | 影响 |
|---|---|---|
| 投机解码+prefill | 同一小模型两阶段共用 | TTFT+TPOT同时优化 |
| Early Exit | 前几层就输出(如果自信) | 降TPOT,但精度风险 |
| Medusa多头 | 一个模型多个解码头并行预测 | 降TPOT,需训练 |
| Token Skipping | 低置信度token跳过某些层 | 降TPOT,实验阶段 |

算法层大多需要额外的小模型或训练，且投机解码的收益高度依赖接受率（任务相关）。

### 5.4 Layer 3: 推理引擎/系统层

工程优化最密集的层，不改模型不改硬件，纯粹提高资源利用率。

**Batching：**

| 方案 | 原理 | 效果 |
|---|---|---|
| Static Batching | 等齐再一起算 | 低效,padding浪费 |
| Continuous Batching | 请求随到随算,每步可加入/移除 | 吞吐提升22x(vLLM) |

Continuous batching 不影响单个请求的TPOT，但大幅提升聚合吞吐——不同请求的decode步骤可以交织，GPU不会闲着。

**KV Cache 管理：**

| 方案 | 原理 | 效果 |
|---|---|---|
| PagedAttention | 类似OS虚拟内存,分页管理KV | 碎片浪费从60%→<4% |
| KV Compression | 量化KV cache(INT8/FP8) | 显存减半,可服务更多并发 |
| KV Eviction | 丢掉不重要的KV(H2O等) | 长上下文不OOM,但有损 |
| RadixAttention | 前缀树共享公共前缀KV | RAG场景大幅省计算 |

主要影响显存利用率 → 能服务的并发数 → 吞吐。

**调度：**

| 方案 | 原理 | 效果 |
|---|---|---|
| Chunked Prefill | 长prompt分块,与decode交替 | 短请求不被长prompt阻塞 |
| Priority Scheduling | 按优先级调度 | SLA保障 |
| Prefill-Decode分离 | 不同阶段分配不同GPU | 各自最优配置 |

**算子优化：**

| 方案 | 原理 | 效果 |
|---|---|---|
| FlashAttention | 分块计算attention,减少HBM读写 | attn快2-4x |
| Kernel Fusion | QKV融合/bias融合/激活融合 | 减少kernel launch |
| CUDA Graph | 预录制计算图,免运行时开销 | 省2-5ms/step |
| FlashInfer | 可定制attention引擎 | 灵活适配各种attn变体 |

**并行：**

| 方案 | 原理 | 适用场景 |
|---|---|---|
| Tensor Parallel (TP) | 层内切分,每卡算一部分 | 单机多卡,需NVLink |
| Pipeline Parallel | 层间切分,流水线 | 跨机,有bubble |
| Expert Parallel | MoE专家分布到不同卡 | MoE模型 |
| Sequence Parallel | 长序列切分到多卡 | 超长上下文 |

TP对TTFT的影响：计算加速，但引入all-reduce通信开销。7B(32层,1卡)=0通信, 70B(80层,4卡)≈160ms, 405B(126层,8卡)≈252ms。

系统层优化大多是"无损"的——不降精度，只提高利用率。但复杂度高，且某些优化（如KV eviction）有精度风险。

### 5.5 Layer 4: Prompt/请求层

从"输入"端做优化，减少需要计算的内容。

| 方案 | 原理 | 效果 |
|---|---|---|
| Prompt压缩 | 删除冗余token(SpecPrefill等) | TTFT↓ 7.66x |
| Prefix Caching | 缓存共享前缀的KV | TTFT↓ (跳过已算前缀) |
| RAG替代长上下文 | 只检索相关片段,不塞全文 | TTFT大幅↓ |
| Prompt工程 | 精简prompt表述 | TTFT↓ (S减少) |
| 上下文窗口管理 | 滑动窗口/摘要替代全历史 | TTFT↓ + KV↓ |

直接作用于 TTFT 公式里的 S：S减半 → GEMM部分减半，attention部分减4倍。

最灵活（不需要改模型/硬件），但效果依赖具体任务，RAG可能影响准确性。

### 5.6 Layer 5: 部署架构层

系统级的资源调度和拓扑设计。

| 方案 | 原理 | 效果 |
|---|---|---|
| PD分离 | prefill和decode用不同GPU集群(Mooncake架构) | 各自最优配置 |
| 多级缓存 | KV cache跨节点共享 | 跨请求复用 |
| 弹性伸缩 | 按负载动态调整GPU分配 | 降成本 |
| 模型路由 | 简单query用小模型,复杂的用大 | 降平均延迟和成本 |
| 混合精度部署 | 不同层用不同精度 | 权衡精度和速度 |

PD分离是目前部署架构最热的方向：prefill是算力受限→用算力强但显存小的卡；decode是带宽受限→用带宽高但算力可以弱的卡。两者分离后各自配置最优硬件，不互相拖累。

架构层优化需要多节点协调，复杂度和成本最高，但能实现单机无法达到的规模。

### 5.7 全景总结：优化方向 × 目标矩阵

| 优化方向 | TTFT | TPOT | 吞吐 | 显存 | 精度 |
|---|---|---|---|---|---|
| **Layer 0 芯片** | | | | | |
| HBM带宽提升 | - | ★★★★★ | ★★ | - | - |
| 互联带宽 | ★★★ | ★★ | ★★ | - | - |
| 算力提升 | ★★★★★ | - | ★★ | - | - |
| 显存容量 | ★ | - | ★★ | ★★★★ | - |
| **Layer 1 模型结构** | | | | | |
| Attention优化 | - | - | ★★ | ★★★★ | ★ |
| MoE | - | ★★★ | ★ | ★ | ★ |
| 量化 | - | ★★★★ | ★★ | ★★★ | ★★ |
| 架构创新 | ★★★★ | - | ★ | - | ★★★ |
| **Layer 2 算法** | | | | | |
| 投机解码 | - | ★★★★ | ★★ | - | 无损 |
| Speculative Prefill | ★★★★ | - | ★★ | - | ★ |
| **Layer 3 引擎系统** | | | | | |
| Continuous Batching | - | - | ★★★★ | - | - |
| PagedAttention | - | - | ★★★ | ★★★★ | - |
| FlashAttention | ★★ | ★ | ★ | - | - |
| Chunked Prefill | ★★ | - | ★★★ | - | - |
| TP并行 | ★★ | ★★ | ★ | - | - |
| **Layer 4 Prompt** | | | | | |
| Prompt压缩 | ★★★★ | - | ★★ | - | ★ |
| Prefix Caching | ★★★ | - | ★★ | - | - |
| RAG | ★★★★ | - | ★ | - | ★ |
| **Layer 5 部署架构** | | | | | |
| PD分离 | ★★ | - | ★★★ | - | - |
| 模型路由 | ★★★ | ★★★ | ★★ | - | - |

关键规律：

1. **TPOT优化**主要靠：带宽提升（芯片）+ 量化（模型）+ 投机解码（算法）。三者分别从公式 TPOT = W_bytes / BW 的分子、分母、和"作弊"三个角度下手。
2. **TTFT优化**主要靠：算力提升（芯片）+ prompt压缩（prompt层）+ FlashAttention（引擎）。核心是减少计算量(2PS + S²)和提升算力。
3. **吞吐优化**主要靠：Continuous Batching + PagedAttention + PD分离。核心是提高GPU利用率。
4. **唯一无损的加速**：投机解码 + Continuous Batching + PagedAttention + FlashAttention。其他手段都有精度trade-off。
5. **越往下层**（芯片），影响越大但迭代越慢；**越往上层**（prompt），越灵活但效果有限。

## 6. Speculative Prefill (SpecPrefill)

> 论文：[arXiv:2502.02789](https://arxiv.org/abs/2502.02789), ICML 2025
> 后续：[Cross-Family版 arXiv:2603.02631](https://arxiv.org/abs/2603.02631), ICLR 2026 WS

### 6.1 核心思想

LLM 足够 robust，只给它 prompt 中"精心选择的子集"token，质量基本不掉。用小模型(speculator)跑一遍 prompt，通过 attention score 判断哪些 token 重要，只把这些重要 token 发给大模型做 prefill。本质是 prompt 压缩，S 降到 S/k，TTFT 降 k 倍。

### 6.2 四步流程

1. **小模型前向传播**：选一个轻量模型对完整prompt跑一遍
2. **Attention score 估计重要性**：取最后一个token对所有prompt token的attention score作代理
3. **Look-Ahead + 聚合去噪**：
   - Look-Ahead：再decode N步，用N个token的attention取平均，消除位置偏置（sink现象、近距偏置）
   - 聚合：attention tensor [N,L,S,H] → 对L和H取max，对N取mean
   - Chunk分块 + 1D average pooling 平滑，选Top-K块
4. **恢复原始位置ID**：选中token保留原始位置编号（非连续），送入大模型做prefill

### 6.3 与 Speculative Decoding 的关系

两者是"一体两面"，可共用同一个小模型：

| | Prefill 阶段 | Decode 阶段 |
|---|---|---|
| Speculative Prefill | 小模型选重要token压缩 | — |
| Speculative Decoding | — | 小模型起草token，大模型批量验证 |
| 两者结合 | 小模型做prefill压缩 ✓ | 同一小模型做decode起草 ✓ |

### 6.4 效果

- TTFT 提升：7.66x
- 最大 QPS 提升：7x
- 精度损失：<5%
- Training-free，可与量化/KV eviction/投机解码叠加
- 跨家族也可行（Qwen小模型给LLaMA大模型压缩，保留90-100%精度）
- 实验：Llama-3.1-405B-Instruct-FP8, 8xH200

## 7. Agent工程师实践指南

前面六层优化大多属于模型工程师和infra工程师的战场。Agent工程师控制不了芯片和模型结构，但在"如何用模型"这件事上有巨大优化空间。

### 7.1 Agent场景的本质区别

```
单轮对话:    1次 prefill + N次 decode → 用户等一次
Agent工作流: K次 prefill + 每次N_i次 decode + 工具调用 → 用户等K轮
```

Agent的延迟是"乘法级"的：每次LLM调用的TTFT和TPOT，乘以调用次数，再加上工具执行时间。核心矛盾是 **调用次数多 × 每次延迟 = 总延迟爆炸**。

### 7.2 模型选择：什么任务用什么模型

Agent工程师最大的杠杆。选错模型，后面所有优化都白费。

| 决策维度 | 小模型(7B-14B) | 大模型(70B+) |
|---|---|---|
| TPOT | 8-12ms / 80-125t/s | 12-20ms / 50-80t/s |
| TTFT(500tok) | ~25ms | ~70ms |
| 成本$/Mtok | $0.1-0.5 | $2-10 |
| 适合任务 | 分类/提取/路由/简单判断 | 推理/写作/复杂工具调用 |
| 工具调用能力 | 弱(需few-shot) | 强(零样本) |

多模型路由策略：

| 步骤类型 | 推荐模型 | 理由 |
|---|---|---|
| 意图识别/路由 | 7B-8B | 二分类/多分类，小模型够用 |
| 信息提取 | 7B-14B | 结构化输出，小模型+schema约束 |
| 简单判断(是否安全) | 7B | 规则化任务 |
| 工具调用决策 | 70B+ | 需要理解上下文+选择正确工具 |
| 多步推理 | 70B+ / 推理模型 | 需要CoT |
| 最终回答生成 | 70B+ | 质量要求高 |
| 结果格式化/校验 | 7B | 模式匹配 |

关键：不要只看单次延迟，要看"延迟×调用次数"的总和。路由用7B: 10次×30ms=300ms vs 路由用70B: 10次×80ms=800ms，多出的500ms对用户感知差别不大，但成本差10倍。

需要建立内部的"任务-模型-延迟-质量"映射表，持续benchmark。

### 7.3 上下文管理：agent最大的性能杀手

Agent和单轮对话最大的区别：上下文会累积。

```
第1轮:  prompt = 500 tokens → TTFT = 25ms
第5轮:  prompt = 3000 tokens → TTFT = 110ms
第10轮: prompt = 8000 tokens → TTFT = 326ms
第20轮: prompt = 20000 tokens → TTFT = 1000ms+
```

TPOT也会受影响，因为KV cache随上下文增长：70B模型约1.3MB/token，8000 token上下文→KV cache≈10.5GB，挤占显存→减少batch size→间接降低吞吐。

| 策略 | 原理 | 效果 |
|---|---|---|
| 对话摘要压缩 | 定期把历史摘要成短文本替换 | S大幅↓，但有信息损失 |
| 滑动窗口 | 只保留最近N轮+系统prompt | S可控，但丢失早期上下文 |
| 关键信息提取 | 从历史中提取关键事实单独存储 | S↓+信息保留，但需额外调用 |
| 工具结果裁剪 | 工具返回的长结果做摘要再入上下文 | S↓，最容易被忽视 |
| 结构化上下文 | 用JSON/精简格式而非自然语言 | S↓ 20-50% |

**最容易忽视的：工具返回值。** 一次搜索工具返回5000 token的网页内容，直接塞进上下文，后面每一轮都要为这5000 token付出TTFT和KV cache成本。应该在工具调用后立即做摘要/提取，把5000 token压到500。

### 7.4 调用编排：串行vs并行

Agent工作流里经常有多个独立的LLM调用。串行还是并行，对延迟影响巨大。

```
场景: 需要对3个文档分别做分析，然后综合

串行: 3 × (TTFT + 生成时间) = 3 × 200ms = 600ms
并行: max(3 × TTFT + 生成时间) ≈ 250ms (假设batch够大)
```

并行有前提：两个调用之间无数据依赖；serving框架支持continuous batching；GPU有足够显存做并发batch。

| 编排模式 | 适用场景 | 延迟 |
|---|---|---|
| 串行链式 | 后一步依赖前一步输出 | Σ(每步延迟) |
| 并行map-reduce | 多个独立任务→汇总 | max(各步延迟)+汇总 |
| 流水线 | 前一步出结果就开始下一步 | 约等于最长单步 |
| 投机并行 | 同时调大+小模型，快的先用 | 小模型延迟+可能的校验 |

在画工作流DAG时，标注每个节点的依赖关系，把无依赖的节点并行化。很多agent框架默认串行执行。

### 7.5 Prompt工程：性能视角

从性能角度，prompt工程的核心是：在保证质量的前提下最小化S。

| 手段 | 具体做法 | S减少幅度 |
|---|---|---|
| 去冗余 | 删除示例中不必要的解释 | 10-30% |
| 结构化 | 用JSON/YAML替代自然语言描述 | 20-50% |
| Few-shot精选 | 3个高质量示例>10个普通示例 | 50%+ |
| 系统prompt复用 | 公共前缀提取，利用prefix cache | 间接省TTFT |
| 输出格式约束 | 限制输出长度(max_tokens) | TPOT阶段↓ |
| 动态prompt | 根据任务难度动态调整prompt长度 | 视情况 |

**Prefix caching**：如果agent有固定的系统prompt（比如500 token的角色设定），确保所有请求共享这个前缀，且serving框架支持prefix caching。第1次全算，后续跳过前500 token → TTFT大幅降低。

### 7.6 实用性能预估速查

基于前面的公式，简化成实用规则：

```
TTFT ≈ prompt长度 × 0.05ms/token (70B级别)
     ≈ prompt长度 × 0.02ms/token (7B级别)

生成时间 ≈ 输出token数 × TPOT

总延迟 = TTFT + 输出token数 × TPOT + 网络RTT
```

实用速查（70B级别，API调用）：

| Prompt长度 | TTFT估算 | 输出500token总延迟 |
|---|---|---|
| 500 token | ~70ms | ~7.6s |
| 2000 token | ~140ms | ~7.6s |
| 8000 token | ~400ms | ~7.9s |
| 32000 token | ~2s | ~9.5s |

Agent工作流总延迟估算：

```
总延迟 = Σ(TTFT_i + 输出_i × TPOT) + Σ(工具执行_i) + Σ(网络RTT)

10步agent，每次平均输出200 token，70B模型:
10 × (TTFT + 200×15ms) = 10 × (TTFT + 3s) ≈ 30s+ (不含工具执行)
```

每多一步LLM调用，就是3-15秒的额外延迟。

### 7.7 成本控制

```
成本 = Σ(prompt_tokens × input_price + output_tokens × output_price)
```

典型价格（2026年，API）：

| 模型级别 | input $/Mtok | output $/Mtok |
|---|---|---|
| 7B级别 | $0.05-0.15 | $0.05-0.15 |
| 70B级别 | $0.5-3 | $0.5-5 |
| 旗舰(405B/GPT) | $3-15 | $10-75 |

三大成本陷阱：

1. **上下文累积**：第1轮500 token→$0.0015，第10轮8000 token→$0.024(16倍)，第20轮20000 token→$0.06(40倍)
2. **冗余调用**：能一次完成的任务拆成3次→成本3倍；能用小模型的任务用了大模型→成本10倍
3. **工具返回值不裁剪**：搜索返回5000 token→每轮多付5000 token input cost→10轮多付45000 token

| 成本优化策略 | 效果 |
|---|---|
| 模型路由(小+大) | 降成本50-80% |
| 上下文压缩 | 降成本30-60% |
| 结果缓存(相同query) | 降成本10-40% |
| 批量调用(合并prompt) | 降成本10-20% |
| 输出长度控制 | 降成本10-30% |

### 7.8 Agent工程师的杠杆点

从六层优化框架看，agent工程师的杠杆集中在 Layer 4(Prompt) 和 Layer 5(部署架构中的模型路由)：

| 层 | 你能做什么 | 杠杆大小 |
|---|---|---|
| Layer 0 芯片 | 选模型时间接选择 | ★ |
| Layer 1 模型结构 | 选模型时间接选择 | ★ |
| Layer 2 算法 | 几乎无法控制 | - |
| Layer 3 引擎系统 | 选择serving框架，利用cache | ★★ |
| Layer 4 Prompt | **核心战场** | ★★★★★ |
| Layer 5 部署架构 | 模型路由、PD分离、缓存 | ★★★★ |

还有一个不在六层框架里的关键维度——**编排层**：如何组织K次LLM调用的拓扑和依赖关系。这是agent工程师独有的：既是模型的使用者，也是工作流的设计师。同样一个任务，串行10步和并行3步的延迟可以差3倍，成本可以差2倍。

Agent工程师需要建立的核心能力：

1. 对每次LLM调用的延迟和成本有精确的心理模型
2. 对工作流拓扑有优化意识（并行化、减少步数）
3. 对上下文生命周期有管理意识（压缩、裁剪、缓存）
4. 对模型能力边界有清晰认知（什么任务需要多大模型）

### 7.9 Checklist

- [ ] 每个LLM调用节点是否选了合适的模型大小？
- [ ] 工具返回值是否做了摘要/裁剪再入上下文？
- [ ] 无依赖的LLM调用是否并行化了？
- [ ] 系统prompt是否复用（利用prefix caching）？
- [ ] 对话历史是否定期压缩？
- [ ] 是否监控了每次调用的实际TTFT/TPOT？
- [ ] 是否有结果缓存避免重复调用？
- [ ] 输出长度是否做了约束（max_tokens）？
- [ ] 是否能快速估算任意工作流的总延迟？
- [ ] 成本是否按"$/用户/次"做了核算？

## 8. 推理芯片

### 8.1 推理芯片 vs 训练GPU：本质区别

训练GPU（如H100/B200）的设计目标是"全能"：既要训练（GEMM密集）又要推理（GEMV密集）。但这两个场景的计算特征完全不同：

| 维度 | 训练 | 推理 |
|---|---|---|
| 计算类型 | GEMM（矩阵×矩阵） | GEMV（矩阵×向量）为主 |
| 计算强度 | 高（FLOPS充分利用） | 低（算力大量闲置） |
| 瓶颈 | 算力+显存+互联 | 显存带宽为主 |
| Batch size | 大（几百~几千） | 小（1~几十） |
| 精度需求 | 高（FP16/BF16，梯度需要） | 低（FP8/INT4/FP4够用） |
| 模型是否变化 | 是（每步更新权重） | 否（权重固定） |
| 互联需求 | 极高（梯度同步） | 中（TP切分） |
| 内存层次 | HBM+片外 | 片上SRAM最优 |

**关键洞察：训练GPU在做推理decode时，算力利用率极低。**

```
H100 decode阶段的计算强度：
  每读1 byte权重，只做 ~2 FLOPS（矩阵×向量）
  H100算力/带宽比 = 989TF / 3.35TB/s = 295 FLOPS/byte
  实际只需要 2 FLOPS/byte
  → 算力利用率 = 2/295 = 0.7%
```

99%以上的算力在decode阶段是闲置的。这就是推理芯片的机会——不需要那么多算力，但需要极致的带宽和内存访问效率。

核心设计哲学差异：
- 训练GPU：算力优先，带宽跟上
- 推理芯片：带宽优先，算力够用，架构匹配decode的GEMV特征

| 设计维度 | 训练GPU (H100) | 推理专用芯片 |
|---|---|---|
| SRAM | 少（~50MB） | 多（几百MB~几十GB） |
| HBM | 大容量（80GB） | 可选，有些用SRAM替代 |
| 计算单元 | 密集Tensor Core | 稀疏/定制化，够用即可 |
| 精度支持 | FP16/BF16/FP8 | INT4/INT8/FP4/FP8 |
| 互联 | NVLink（900GB/s） | 定制fabric或以太网 |
| 架构 | SIMD（通用） | Dataflow/脉动阵列/定制 |
| 灵活性 | 高（可编程） | 低（针对特定模型） |
| 权重存储 | 每次从HBM读 | 预加载到SRAM/片上 |

### 8.2 推理芯片主要玩家

#### 类别1：晶圆级/超大芯片

**Cerebras Systems（美国）**
- 芯片：WSE-3（Wafer-Scale Engine），已上市（Nasdaq:CBRS）
- 架构：整块晶圆做成一个芯片，215mm×215mm，4万亿晶体管，90万cores
- 内存：44GB 片上SRAM（不用HBM），125PB/s 片上内存带宽（vs H100的3.35TB/s）
- 推理优势：权重全在片上SRAM，带宽是H100的37000倍。Llama推理：1200 tokens/s（Perplexity Sonar）
- 核心思路：不是提升HBM带宽，而是把HBM直接做到芯片上（SRAM），消除"权重搬运"瓶颈。代价是芯片极大、良率挑战、成本高
- 2025年营收$510M，覆盖训练+推理

**Groq（美国）**
- 芯片：LPU（Language Processing Unit）/ TSP
- 架构：数据流架构，确定性执行（无缓存、无分支预测、无乱序执行），SRAM为主（230MB/芯片）
- 推理优势：Llama 70B 500+ tokens/s，单token延迟极低
- 核心思路：去掉所有"不确定"的硬件，做极致确定性的数据流。每个cycle都知道在算什么，没有浪费。但灵活性极低，基本只能跑矩阵乘法
- 创始人：Jonathan Ross（Google TPU设计者之一）
- 2025年营收$500M，2025年12月Nvidia以~$200亿license Groq技术
- 纯推理

#### 类别2：可重构/数据流架构

**SambaNova Systems（美国）**
- 芯片：SN40L（Reconfigurable Dataflow Unit, RDU）
- 架构：可重构数据流，软硬件协同设计。三层内存：SRAM + HBM + 外部DDR，精度可变（FP8/FP16/INT8混合）
- 推理优势：大模型可分布在DDR上，不挤HBM，适合超大模型推理
- 创始人：Kunle Olukotun（Stanford），Christopher Ré（Stanford）
- 累计融资$1.6B+，估值$5B，覆盖训练+推理

**Tenstorrent（美国/加拿大）**
- 芯片：Grayskull / Wormhole / Blackhole
- 架构：RISC-V + AI加速核（Tensix Core），片上以太网互联，模块化扩展
- 创始人：Jim Keller（传奇芯片架构师，AMD Zen / Apple A4 / Tesla FSD）
- 覆盖训练+推理，从边缘到数据中心

**Graphcore（英国）**
- 芯片：IPU（Intelligence Processing Unit）/ Bow
- 架构：大量小核（数千个），MIMD架构，纯SRAM（~900MB/芯片）
- 状态：2024年从公开市场收缩，转向政府/国防项目

#### 类别3：推理专用ASIC

**Etched（美国）**
- 芯片：Sohu
- 架构：纯Transformer ASIC，硬件固化attention+FFN，只能跑Transformer
- 推理优势：固化了Transformer结构，无需通用计算开销，宣称单芯片推理速度比H100快一个数量级
- 状态：2024年融$120M，芯片仍在流片阶段
- 纯推理（只支持Transformer）

**D-Matrix（美国）**
- 芯片：Corsair
- 架构：Chiplet设计，片上SRAM为主，数字in-memory computing，专为LLM推理的decode阶段优化
- 状态：2024年融$154M，纯推理

**FuriosaAI（韩国）**
- 芯片：Renegade（RNGD）
- 架构：数据流架构，8nm，专注推理，低功耗
- 状态：2025年被OpenAI考虑投资/收购，纯推理

**Rebellions（韩国）**
- 芯片：ATOM / REBEL
- 架构：定制指令集，针对推理优化，从边缘推理起家向数据中心扩展
- 状态：2025年与FuriosaAI讨论合并，纯推理

#### 类别4：大厂自研

| 公司 | 芯片 | 架构特点 | 状态 |
|---|---|---|---|
| Google | TPU v5e / v7（Ironwood） | 脉动阵列，MXU矩阵单元，v5e专门针对推理优化 | 自用+GCP对外 |
| AWS | Inferentia2 / Trainium2 | NeuronCore架构，支持FP8/INT8 | EC2 Inf2实例对外 |
| Microsoft | Maia 100 | 为Azure AI工作负载定制 | 2024年部署，Azure内部 |
| Meta | MTIA | 为推荐系统和LLM推理定制 | 2024年第二代，自用 |
| 华为 | 昇腾910B/910C | 达芬奇架构，3D Cube矩阵计算 | 国内主力，替代Nvidia |
| 寒武纪 | 思元590/MLU系列 | 专为AI计算设计 | 国产推理芯片上市 |

### 8.3 技术路线对比

各家解决"带宽瓶颈"的策略不同：

| 路线 | 代表 | 核心思路 | 优势 | 劣势 |
|---|---|---|---|---|
| 晶圆级SRAM | Cerebras | 全片上SRAM替代HBM | 带宽极高 | 成本/良率 |
| 纯SRAM数据流 | Groq | 去缓存+确定性执行 | 延迟极低 | 灵活性低 |
| 可重构数据流 | SambaNova | 软硬件协同+多层内存 | 灵活 | 复杂度高 |
| Transformer固化 | Etched | 硬件固化模型结构 | 效率极高 | 只能跑Transformer |
| In-memory计算 | D-Matrix | 计算直接在存储中做 | 省搬运 | 精度/工艺挑战 |
| 脉动阵列 | Google TPU | 经典矩阵加速架构 | 成熟稳定 | 对decode非最优 |
| 大厂定制 | AWS/Meta/MS | 和自有云生态绑定 | 规模效应 | 不对外/绑定生态 |

### 8.4 芯片层面的推理优化方法

| 方法 | 原理 | 代表 | 效果 |
|---|---|---|---|
| SRAM替代HBM | 权重放片上，消除搬运 | Cerebras（125PB/s vs 3.35TB/s） | TPOT降10-100x（理论） |
| 确定性数据流 | 去缓存/分支预测/乱序 | Groq TSP | 提高利用率 |
| 模型结构固化 | 把attention/FFN做到硬件里 | Etched Sohu | 去通用计算开销 |
| In-memory计算 | 计算在存储单元内完成 | D-Matrix | 省数据搬运 |
| 低精度计算 | 原生支持INT4/FP4 | 推理芯片普遍比GPU更大胆 | TPOT降2-4x |
| Chiplet封装 | 多小芯粒互联 | AMD/Groq/Cerebras | 降成本/提良率 |
| PD分离专用硬件 | prefill用算力芯片，decode用带宽芯片 | 部署架构层 | 各自最优 |
| 稀疏化硬件支持 | MoE专家路由硬件加速 | — | 只算激活参数 |
| KV cache压缩 | 硬件级量化KV | — | 省显存→更多并发 |
| 批处理优化 | 数据流架构天然支持 | Groq/Cerebras | 高吞吐 |

### 8.5 推理芯片 vs GPU 实际对比

| 指标 | H100（GPU） | Cerebras WSE-3 | Groq LPU |
|---|---|---|---|
| 片上SRAM | ~50MB | 44GB | 230MB |
| 内存带宽 | 3.35TB/s（HBM） | 125PB/s（SRAM） | ~80TB/s（SRAM） |
| Llama 70B推理 | ~80 tok/s | ~1200 tok/s | ~500 tok/s |
| 灵活性 | 高（通用） | 中（CUDA生态外） | 低（专用） |
| 生态 | CUDA，最完善 | 自有框架 | 自有框架 |
| 能否训练 | 是 | 是 | 否（纯推理） |
| 价格 | ~$30k | ~$2-3M（系统级） | ~$20k（芯片） |

注意：Cerebras和Groq的速度数字是单流延迟（低并发），GPU在batch大时吞吐更高。推理芯片在低延迟单请求场景优势最大。

### 8.6 推理芯片会取代GPU吗？

短期内不会，但会瓜分推理市场。

| 场景 | 最优选择 | 理由 |
|---|---|---|
| 大批量吞吐（API服务） | GPU（H100/B200） | Continuous batching利用率高 |
| 低延迟单流（实时对话） | 推理芯片（Cerebras/Groq） | SRAM带宽碾压 |
| 超大模型（405B+） | GPU集群 | 生态成熟，TP/PP完善 |
| 边缘/端侧推理 | 专用ASIC/NPU | 功耗敏感 |
| 成本敏感推理 | 推理芯片 | $/tok更低（带宽利用率高） |

核心逻辑：
- GPU的"算力/带宽比"对decode是浪费的（利用率<1%）
- 推理芯片通过架构匹配，把带宽利用率拉到极致
- 但GPU的生态（CUDA）和灵活性是巨大壁垒
- 训练市场GPU几乎垄断，推理市场将分化

这也是为什么Nvidia在2025年底花$200亿license Groq技术——它意识到了推理专用架构的价值，试图在保持GPU主导地位的同时吸收推理芯片的创新。

## 9. 总结

- **TPOT 是"带宽引力井"**——硬件带宽×模型部署配比把它锁在 10-20ms
- **TTFT 是"算力×长度的函数"**——从 30ms 到数秒，取决于 prompt 多长
- **优化分六层**：芯片（物理上限）→ 模型结构（减少计算量）→ 算法（绕过极限）→ 引擎系统（提高利用率）→ Prompt（减少输入）→ 部署架构（系统调度）
- TPOT 优化靠：带宽提升 + 量化 + 投机解码（分子/分母/作弊）
- TTFT 优化靠：算力提升 + prompt压缩 + FlashAttention（减计算量+提算力）
- 无损加速：投机解码 + Continuous Batching + PagedAttention + FlashAttention
- **Agent工程师的杠杆**在 Layer 4(Prompt) + Layer 5(模型路由) + 编排层（调用拓扑），核心是：选对模型、管好上下文、并行化无依赖调用
- **推理芯片**通过SRAM替代HBM、确定性数据流、模型结构固化等路线，解决GPU在decode阶段算力利用率<1%的问题。推理市场将分化：GPU主导大批量吞吐，推理芯片主导低延迟场景
- 下一代硬件（HBM4/Rubin）会让整个区间下移一倍

## 关联

- [Inference.md](./Inference.md) — 推理优化论文与工具索引
- [train.md](./train.md) — 训练框架与分布式训练
