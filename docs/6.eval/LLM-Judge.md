# LLM-as-a-Judge

> 2026.08.31 | 来源：arXiv 检索（经典奠基 + 2025.11-2026.09 新论文）
> 主题：用 LLM 做评估——打分、排序、选择、裁决

## 0. 领域概览

评估一直是 AI/NLP 的核心难题。传统方法（BLEU/ROUGE 等匹配式指标、小模型分类器）在开放式生成场景失效。LLM 的出现催生了 **LLM-as-a-Judge** 范式：用 LLM 对输出做 scoring / ranking / selection。

**I/O 形式空间**（judge LLM J，候选 C₁...Cₙ，结果 R = J(C₁,...,Cₙ)）：

| 输入格式 | 含义 | 输出格式 | 含义 |
|---|---|---|---|
| Point-wise (n=1) | 单独评一个候选 | Score | 每个候选一个分数 |
| Pair-wise (n=2) | 比较两个候选 | Ranking | 候选的全序排列 |
| List-wise (n>2) | 多个候选一起比较 | Selection | 选出最优的一个或多个 |

**领域的核心矛盾**：
1. 单次判定方差大（同模型重采样结果不稳）
2. 强裁判成本高（逐样本用最强模型不可扩展）
3. 一致性≠正确（多裁判都同意可能是共享盲区）
4. 系统性偏差（位置/长度/自我偏好/锚定等，稳定地偏）

后续研究线（投票聚合、级联升级、偏差去偏、元评估、置信度校准）都是对这四个矛盾的回应。

## 1. 奠基与范式

| # | 论文 | 年份 | 核心贡献 |
|---|---|---|---|
| 1 | [G-Eval](https://arxiv.org/abs/2303.16634) | 2023.03 | 首个系统化框架：任务表单填充 + CoT 评分 + 概率加权（token 概率加权平均，非单次采样）。与人工评价相关性超过 BLEU/ROUGE |
| 2 | [Judging LLM-as-a-Judge with MT-Bench](https://arxiv.org/abs/2306.05685) | 2023.06 | 领域开山之作。80 道多轮题 + GPT-4 裁判 + Chatbot Arena 验证。**首次系统定义四大偏差**：位置、自增强、冗长、方差 |
| 3 | [A Survey on LLM-as-a-Judge](https://arxiv.org/abs/2411.15594) | 2024.11 | 综述：多轮/多智能体/集成裁判分类 |
| 4 | [From Generation to Judgment](https://arxiv.org/abs/2411.16594) | 2024.11 | 综述：what to judge / how to judge / how to benchmark 三维分类法 |

## 2. 构建裁判

### 2.1 免训练（Prompting）

| # | 论文 | 年份 | 核心贡献 |
|---|---|---|---|
| 5 | Swapping Operation（MT-Bench §5.2 内） | 2023.06 | pair-wise 交换顺序跑两次，不一致判 tie——位置偏差的标准缓解手段 |
| 6 | Rule Augmentation（综述 §4.2.2 归纳） | — | 把评分原则/rubric 嵌入 prompt 引导裁判。rubric 质量直接决定裁判质量 |
| 7 | [Rubric Artifacts](https://arxiv.org/abs/2609.02942) | 2026.08 | 警示：看似源于 rubric 推理的判断，部分实为 rubric 存在本身引入的伪影 |

### 2.2 训练（Tuning）

| # | 论文 | 年份 | 核心贡献 |
|---|---|---|---|
| 8 | [JudgeLM](https://arxiv.org/abs/2310.17631) | 2023.10 | 10 万条判定数据微调裁判，研究裁判的 scaling law |
| 9 | [Prometheus](https://arxiv.org/abs/2310.08491) | 2023.10 | 开源 13B 裁判（GPT-4 反馈训练），direct + pairwise，1-5 分细粒度。自建裁判不依赖 GPT-4 成为可能 |
| 10 | [Prometheus 2](https://arxiv.org/abs/2405.01535) | 2024.05 | 识别"分布不匹配"：开源裁判对自己的参考答案有偏。同时支持两种模式且结论相关性强 |
| 11 | [M-Prometheus](https://arxiv.org/abs/2504.04953) | 2025.04 | 多语言裁判套件 |
| 12 | [Beyond Scalar Reward Model](https://arxiv.org/abs/2410.03742) | 2024.10 | 偏好数据直接训**生成式裁判**（给出推理依据而非单一标量分数） |
| 13 | [RecurSE](https://arxiv.org/abs/2608.24231) | 2026.08 | 无外部标注的有界递归自提升：裁判判候选（Pass 1）+ checker 用元规则审计裁判推理给过程奖励（Pass 2）。接口解耦防 token 复制捷径，给出"何时必须停止"的边界条件 |
| 14 | [Decoupled Analysis-Judging](https://arxiv.org/abs/2609.03432) | 2026.09 | 创造力评估：分析（找证据）与打分（下判断）解耦为两阶段，抑制冗长/宽松偏差 |

## 3. 聚合与协作

### 3.1 同模型多采样（降方差）

核心洞察：正确推理路径"殊途同归"，错误路径分散。多数投票=经验众数估计，方差随采样数幂律下降。

| # | 论文 | 年份 | 核心贡献 |
|---|---|---|---|
| 15 | [Self-Consistency](https://arxiv.org/abs/2203.11171) | 2022.03 | 多采样+多数投票，奠基 |
| 16 | [Universal Self-Consistency](https://arxiv.org/abs/2311.17311) | 2023.11 | 扩展到自由文本：用 LLM 判断候选间等价性再投票 |
| 17 | [Optimal Self-Consistency (Blend-ASC)](https://arxiv.org/abs/2511.12309) | 2025.11 | SC 的幂律 scaling 理论；动态分配采样数（简单题少采），省 4.8x |
| 18 | [RISC](https://arxiv.org/abs/2606.05054) | 2026.06 | 答案选择重构为排序问题：LambdaRank + 频率/语义中心性/轨迹一致性特征，找回多数投票丢掉的正确答案 |

### 3.2 异构模型共识（降系统偏差）

核心洞察（误差去相关）：独立训练的模型错得不一样，错误答案分散而正确答案聚集。同模型多采样只能降方差，**异构共识才能降系统偏差**。

| # | 论文 | 年份 | 核心贡献 |
|---|---|---|---|
| 19 | [PoLL](https://arxiv.org/abs/2404.18796) | 2024.04 | 一组小异构模型组成陪审团，比单个大裁判更便宜、偏差更低 |
| 20 | [LLMs as a Jury](https://arxiv.org/abs/2607.10139) | 2026.07 | 跨模型共识优于 self-consistency 和自评分。给出闭式定律：从 panel 统计量预测共识精度 |

### 3.3 辩论式

| # | 论文 | 年份 | 核心贡献 |
|---|---|---|---|
| 21 | [Combating Adversarial Attacks with MAD](https://arxiv.org/abs/2401.05998) | 2024.01 | 辩论作为对抗边界输入的二次消解机制 |
| 22 | [Safety Eval through MAD](https://arxiv.org/abs/2511.06396) | 2025.11 | 安全评估里用结构化辩论替代昂贵强裁判，按需动用 |
| 23 | [Judging with Many Minds](https://arxiv.org/abs/2505.19477) | 2025.05 | 警示：多智能体裁判可能放大而非减少偏差（从众、强势 agent 主导） |
| 24 | [Beyond Consensus: Downward Bias](https://arxiv.org/abs/2608.30373) | 2026.08 | 主观评分上 MAD 共识的 human alignment **不如单 judge**。根因：角色不对称——严格派压分在共识中无法纠正 |
| 25 | [Layered Analysis of Disagreement in MAD](https://arxiv.org/abs/2609.08016) | 2026.09 | MAD 里的"分歧"很大程度是**指令诱发的**：删掉诱发指令后 23.1% 立场回退；语气改变 50.4pp 一致性 |
| 26 | [Peacemaker or Troublemaker](https://arxiv.org/abs/2509.23055) | 2025.09 | 谄媚倾向在辩论中是双刃剑：过强从众消灭分歧，过弱无法收敛 |

### 3.4 Panel 组建与路由

| # | 论文 | 年份 | 核心贡献 |
|---|---|---|---|
| 27 | [Stopping and Routing LLM Judge Panels](https://arxiv.org/abs/2608.19802) | 2026.08 | panel 设计形式化：估计每个 judge 的角色（冗余副本/全局补充/切片专家），丢副本、加补充、条件路由专家、增益低于阈值停止 |
| 28 | [Beyond Majority Vote](https://arxiv.org/abs/2609.03953) | 2026.09 | 医疗幻觉检测：多候选源（人工+LLM judge）之上的仲裁提升完备性，但不替代专业判断 |

### 3.5 级联与升级触发（cascade / deferral）

| # | 论文 | 年份 | 核心贡献 |
|---|---|---|---|
| 29 | [FrugalGPT](https://arxiv.org/abs/2305.05176) | 2023.05 | cascade 奠基：便宜模型先答，置信不够才升级贵模型 |
| 30 | [Cost-Saving Cascades with Early Abstention](https://arxiv.org/abs/2502.09054) | 2025.02 | 引入"提前弃权"——某些样本干脆不判 |
| 31 | [Rational Tuning of Cascades](https://arxiv.org/abs/2501.09345) | 2025.01 | 概率模型联合调各级升级阈值 |
| 32 | [Cascade Failure under Attack](https://arxiv.org/abs/2605.17288) | 2026.05 | 对抗输入可专门骗过升级判断，让危险样本不被升级 |
| 33 | [Forced Deferral](https://arxiv.org/abs/2606.15308) | 2026.06 | 多模态 cascade 的路由决策本身可被操纵——升级触发器是对抗面 |
| 34 | [Conformal Cascade](https://arxiv.org/abs/2607.25018) | 2026.07 | conformal prediction 预测集大小做升级规则，给出分布无关的有限样本精度保证 ≥1-Kα |
| 35 | [Judge, Retrieve, or Abstain](https://arxiv.org/abs/2608.17994) | 2026.08 | 不确定性守卫：校准阈值使被接受裁决的 FDR ≤ α，带工具增强选项 |
| 36 | [Multi-Expert Conformal Risk Control](https://arxiv.org/abs/2608.26529) | 2026.08 | CRC（弃权控险）+ 多专家聚合（源头净化）；MC3 解决异构 judge 评分尺度不统一 |
| 37 | [Certified Deferral for Verbalized Uncertainty](https://arxiv.org/abs/2608.05064) | 2026.08 | 小模型何时该弃权给人类的可证明边界 |

### 3.6 分歧样本的性质

| # | 论文 | 年份 | 核心贡献 |
|---|---|---|---|
| 38 | [Is LLM an Overconfident Judge?](https://arxiv.org/abs/2502.06207) | 2025.02 | LLM 在"标注者本就分歧"的样本上过度自信、抹平本应保留的分歧 |

## 4. 偏差研究

### 4.1 偏差类型与测量

| # | 论文 | 年份 | 偏差类型 | 核心发现 |
|---|---|---|---|---|
| 39 | MT-Bench（§1 #2） | 2023.06 | 位置/自增强/冗长/方差 | 四大偏差的首次系统定义 |
| 40 | [Length-Controlled AlpacaEval](https://arxiv.org/abs/2404.04475) | 2024.04 | 冗长 | 长度控制胜率（回归调整）替代朴素胜率，解耦"更长=更好"混杂 |
| 41 | [Bidirectional Bias: Self- and Other-Labels](https://arxiv.org/abs/2608.18091) | 2026.06 | 自我偏好 | 盲评时自我偏好消失；但仅标注"自选/他选"（不出现模型名）就双向偏移：自标高估、他标低估 |
| 42 | [Self-Preference Is Weak or Absent](https://arxiv.org/abs/2606.20093) | 2026.06 | 自我偏好（反证） | 可验证任务+真实作者身份下自我偏好弱/不存在。自我偏好是任务依赖的：开放生成强、可验证弱 |
| 43 | [Anchoring Bias](https://arxiv.org/abs/2608.25869) | 2026.08 | 锚定 | **先前分数哪怕只作为上下文元数据，也系统性锚定判断**（d=0.71，7/8 模型显著，18.5万次评估） |
| 44 | [Style Wins, Substance Loses](https://arxiv.org/abs/2608.01666) | 2026.08 | 风格 | 科研想法评估中裁判实际评的是风格表达而非科学实质。Style Bias Index / Substance Recognition Rate 指标 |
| 45 | [BabelJudge](https://arxiv.org/abs/2606.22329) | 2026.06 | 语言 | 跨语言+agent 轨迹的裁判可靠性测量 |
| 46 | [Lower-Resource, Higher Scores](https://arxiv.org/abs/2607.14480) | 2026.07 | 语言 | 评估器对高资源语言内容打分系统性更高 |
| 47 | [Scoring Bias via RNG](https://arxiv.org/abs/2608.05726) | 2026.08 | 数字偏好 | 让 LLM 生成随机数暴露潜在数字偏好，用偏差校正评分 token 概率 |
| 48 | [Rank Reversal in Multilingual Judges](https://arxiv.org/abs/2608.22432) | 2026.08 | 语言×排序 | prompt 语言不同导致评估器排序结论翻转 |
| 49 | [LLM Judges as Raters](https://arxiv.org/abs/2608.29517) | 2026.08 | 评分者效应 | 预注册审计：严重度/光环/信度/版本不稳定（教育测量学视角） |

### 4.2 共识的风险（一致性≠正确）

| # | 论文 | 年份 | 核心贡献 |
|---|---|---|---|
| 50 | [JuryProbe](https://arxiv.org/abs/2608.20607) | 2026.08 | 多 judge 一致可能反映**共享假阴性盲区**而非独立证据（FN 相关性 0.402/0.368，假共识提升 3.13x/18.13x）。共识风险诊断+校准路由 |
| 51 | [Anchor-Judge Error Correlation](https://arxiv.org/abs/2609.08826) | 2026.09 | judge panel 误差分解为质量信号+共模误差；锚定集污染的诊断 |

### 4.3 对抗鲁棒性

| # | 论文 | 年份 | 核心贡献 |
|---|---|---|---|
| 52 | Cascade Failure under Attack（§3.5 #32） | 2026.05 | 攻击者构造"稳定一致判错"绕过升级 |
| 53 | [LLM Judges Verify Presence, Not Absence](https://arxiv.org/abs/2608.31016) | 2026.08 | 临床场景：LLM 裁判验证"存在"强、验证"缺失"弱（遗漏盲区） |

## 5. 元评估（评估裁判本身）

| # | 论文 | 年份 | 核心贡献 |
|---|---|---|---|
| 54 | [Reliability without Validity](https://arxiv.org/abs/2606.19544) | 2026.06 | 21 裁判、54.1 万判定的最大规模元评估。四发现：精确匹配一致率比 kappa 高估 33-41pp；裁判排名跨 benchmark 漂移 14 位；**信度>0.95 与严重位置偏差共存**（一致性-偏差悖论）；单一 pairwise rubric 下冗长偏差其实很小。提出 Minimum Viable Validation Protocol |
| 55 | [Four-Axis Trustworthiness Benchmark](https://arxiv.org/abs/2608.14329) | 2026.08 | 原则性监管场景（"公平/清晰/不误导"）的四轴可信度基准 |
| 56 | [Human-Anchored Factuality Evaluation](https://arxiv.org/abs/2609.00494) | 2026.08 | 策略性标注锚定事实性裁判，修正系统性偏差 |
| 57 | [IF-RewardBench](https://arxiv.org/abs/2603.04738) | 2026.03 | 指令遵循场景的裁判基准 |
| 58 | [Long-form RewardBench](https://arxiv.org/abs/2603.12963) | 2026.03 | 长文本生成的奖励模型评估 |
| 59 | [Personalized RewardBench](https://arxiv.org/abs/2604.07343) | 2026.04 | 多元对齐：不同人群偏好下的奖励模型评估 |
| 60 | [LLM-as-a-Judge Is Not an Oracle](https://arxiv.org/abs/2609.02246) | 2026.09 | 自改进循环中的裁判需要确定性护栏 |
| 61 | [trajectory-judge](https://arxiv.org/abs/2609.00038) | 2026.08 | 只看最终结果的裁判对 agent 轨迹结构性失明 |

## 6. 置信度与不确定性

| # | 论文 | 年份 | 核心贡献 |
|---|---|---|---|
| 62 | [Wired for Overconfidence](https://arxiv.org/abs/2604.01457) | 2026.04 | 机制可解释性视角：LLM 错误答案上口头置信度虚高的内部表征原因 |
| 63 | [Verbal Confidence Saturation](https://arxiv.org/abs/2604.22215) | 2026.04 | 3-9B 小模型口头置信度饱和：天花板>95%，AUROC 接近随机。**小模型自报置信度基本不可用** |
| 64 | [Distilling SC into Verbal Confidence](https://arxiv.org/abs/2604.24070) | 2026.04 | 预注册负结果：把 self-consistency 蒸馏成单次口头置信度失败 |
| 65 | [ORCE](https://arxiv.org/abs/2605.12446) | 2026.05 | 答案顺序对口头置信度的影响及对齐 |
| 66 | [DualStake](https://arxiv.org/abs/2609.00935) | 2026.09 | deep research agent 的双路置信度校准 |

## 7. 机制要点

**多采样降方差**：多数投票=众数估计，方差随 n 幂律下降。只解决"不稳定"，不解决"稳定地错"。

**异构共识降系统偏差**：误差去相关是核心机制——独立训练的模型错误分散、正确聚集。闭式定律可从 panel 统计量预测共识精度（LLMs as a Jury）。

**一致性门控是免费置信度信号**：投票一致性天然产出置信信息，无需额外打分模型（vs FrugalGPT 要训打分器）。代价：同参采样的"全一致"含金量低于跨模型一致。

**假共识**：多个裁判一致接受错误结论，可能是独立验证（可信）或共享盲区（不可信）——两者外观相同。区分方法：标注校准探针测 FN-only 相关性（JuryProbe）。

**一致性-偏差悖论**：信度>0.95 与严重偏差可以共存。三票一致完全可能是"稳定地偏"。裁判评估必须同时测 agreement / consistency / bias 三轴，不能只看一致率。

**锚定效应**：先前分数哪怕只作为元数据出现也会牵引判断（d=0.71）。喂历史票据给仲裁的设计，历史分数本身就是锚。

**升级规则的统计保证**：conformal prediction 把"何时接受/何时升级"放进分布无关的有限样本框架（≥1-Kα），替代拍脑袋硬阈值。

**小模型口头置信度不可用**：3-9B 的 verbal confidence 天花板>95%、判别力近随机。需要置信度信号时，用行为信号（多采样一致性）替代自报。

## 8. 实践检查清单

设计 LLM 裁判流程时：

- [ ] 输入格式选对了吗？point-wise（绝对质量）/ pair-wise（相对比较，受位置偏差影响）/ list-wise
- [ ] 需要降方差还是降系统偏差？同模型多采样只降方差；异构共识才降系统偏差
- [ ] pair-wise 做了 swapping（交换顺序跑两次，不一致判 tie）吗？
- [ ] 打分式裁判测过数字偏好偏差吗？（RNG 方法可检测+校正）
- [ ] 喂给裁判的历史信息会锚定它吗？（隐藏具体分数/做有/无票据 A/B 对照）
- [ ] 多裁判一致时，考虑过共享盲区吗？（高相关模型对的"一致"信息量趋零）
- [ ] 裁判的验证测了三轴（agreement/consistency/bias）还是只看一致率？
- [ ] 长文本/多语言场景下，测过冗长偏差和语言偏差吗？
- [ ] 级联升级的阈值是硬规则还是校准的？要不要统计保证（conformal）？
- [ ] 对抗场景下，升级触发器本身可能被攻击——高风险样本强制走重验证？
- [ ] 用小模型自报置信度做路由？——3-9B 的口头置信度基本没有判别力
- [ ] 分歧样本是被多数票压平了，还是被妥善保留（它们重合于"标注者分歧样本"）？

## 9. 发展脉络

```
2022.03  Self-Consistency（多采样投票）
2023.03  G-Eval（LLM做NLG评估的首个系统框架）
2023.06  MT-Bench（范式确立：四大偏差定义）
2023.05  FrugalGPT（级联奠基）
2023.10  JudgeLM / Prometheus（裁判训练线开启）
2023.11  Universal SC
2024.04  PoLL（异构陪审团）
2024.04  Length-Controlled AlpacaEval（冗长去偏）
2024.05  Prometheus 2
2024.10  生成式裁判（Beyond Scalar RM）
2024.11  两篇领域综述（范式成熟标志）
2025    ↑ 阈值调优（Rational Tuning）
        ↑ 弃权（Early Abstention）
        ↑ 分歧样本性质（Overconfident Judge）
        ↑ 谄媚对辩论的影响（Peacemaker）
2026    ↑ 统计保证爆发：Conformal Cascade / Judge-Abstain / Multi-Expert CRC
        ↑ 共识风险爆发：JuryProbe / Anchor-Judge / Stopping-Routing
        ↑ 辩论批判性反思：Beyond Consensus / Layered Analysis
        ↑ 异构共识理论：LLMs as a Jury（误差去相关闭式定律）
        ↑ 偏差细分化：锚定 / 风格 / 语言 / 数字偏好 / 自我偏好双向性
        ↑ 元评估规模化：Reliability without Validity（54.1万判定）
        ↑ 置信度机制化：Wired for Overconfidence / Saturation
```

**两条主趋势**：
1. **从启发式到统计保证**——升级阈值从拍脑袋→学出来→可证明；裁判验证从一致率→三轴协议
2. **从"更多裁判更好"到"裁判质量审计"**——从朴素加模型数，到 panel 内冗余/互补/专家角色分析，到共识本身的风险诊断

**一条暗线**：对"一致性"的态度从信任到审视——三票一致可能是把握，也可能是共享盲区或稳定偏差。区分这两者，是当前研究的核心战场。

## 关联

- [Benchmark.md](./Benchmark.md) — 评测基准与榜单
- [Metric.md](./Metric.md) — 评测指标
- [../3.infra/inference-perf-theory.md](../3.infra/inference-perf-theory.md) — 推理性能分析（多采样/级联的算力成本视角）
