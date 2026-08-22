# Physical AI 全景：公司、团队与人才地图

> 本文整理 Physical AI 领域的公司、研究团队、关键人物和分工模式。
> 配套学习笔记见 [PhysicalAI-Feynman.md](../8.tutorial/PhysicalAI-Feynman.md)
> 配套论文索引见 [PhysicalAI.md](../5.multimodal/PhysicalAI.md)

---

## 一、领域格局总览

Physical AI 领域的参与者可以分为四类：

```
① 基础设施层：提供芯片、仿真器、世界模型、开发平台
② 模型层：训练机器人基础模型（VLA、世界模型、策略模型）
③ 硬件层：造机器人本体（人形、机械臂、移动平台）
④ 学术层：大学实验室，发论文，培养人才
```

这四层不是孤立的——头部公司往往横跨多层。

---

## 二、基础设施层

### NVIDIA（英伟达）— 全栈基础设施

Physical AI 领域的"卖铲人"，不造机器人本体，但几乎所有机器人公司都用它的基础设施。

**关键人物：**
- 黄仁勋（Jensen Huang）— CEO，战略决策者。定调"Physical AI"为英伟达核心方向
- 刘洺堉（Ming-Yu Liu）— 研究副总裁，领导 Cosmos 世界基座模型。英伟达第一位从研究员擢升 VP 的管理者。直属团队不到 100 人，虚线汇报 200-300 人
- Dieter Fox — 高级首席研究科学家，NVIDIA 机器人感知方向
- Lerrel Pinto — 与 NVIDIA 合作的机器人学习研究者

**团队组成：**
- Cosmos 团队（世界模型）：~100 人核心
- Isaac Sim / Omniverse 团队（仿真器）：数百人
- GR00T 团队（机器人基础模型）：与 Cosmos 协作
- 硬件团队（Jetson 系列芯片）

**产品矩阵：**
```
硬件：Jetson Orin / Jetson Thor（边缘）→ DGX（云端）
平台：CUDA → Omniverse（3D 仿真）→ Isaac Sim（机器人仿真）→ Newton 物理引擎
模型：Cosmos 1/3（世界基座模型）→ GR00T N1（人形机器人基础模型）
```

**战略定位：** 不做机器人本体，做所有机器人的"大脑基础设施"。黄仁勋原话："你就做到 Cosmos 97"。

---

### Google DeepMind — 模型 + 研究

**关键人物：**
- Chelsea Finn — RT-2 共同作者（现为 Stanford 教授，但深度参与 Google 机器人项目）
- Karol Hausman — RT-2 共同作者
- Pierre Sermanet — 机器人学习团队
- 郑梦昊（Montse Gonzalez Arenas）— RT-2 团队成员

**核心工作：**
- RT-1 / RT-2：Robotics Transformer，VLA 的开山之作
- AlphaEvolve：LLM + 进化搜索 + 自动评估器，用于算法发现和 AI 基础设施优化
- Gemini Robotics：基于 Gemini 的机器人模型（未完全公开）

**团队组成：**
- 机器人基础模型团队（RT 系列）
- 进化搜索团队（AlphaEvolve）
- 与 Stanford、Berkeley 深度合作

---

## 三、模型层

### Physical Intelligence (π) — 通用机器人策略

2024 年成立，创始人来自 Google Brain 和 Stanford。

**关键人物：**
- Karol Hausman — 联合创始人（前 Google Brain）
- Chelsea Finn — 联合创始人（Stanford 教授）
- Sergey Levine — 早期顾问（Berkeley 教授，深度强化学习奠基人之一）

**核心工作：**
- π0 / π0.5：通用机器人策略模型，使用 flow matching 生成动作。论文未在 arXiv 公开，但被大量后续论文引用
- 核心创新：diffusion-based action generation + 大规模预训练

**团队组成：**
- 模型团队（策略训练）
- 数据团队（遥操作采集 + 数据管线）
- 硬件集成团队（策略部署到不同机器人）

**定位：** 不造机器人，做机器人的"通用大脑"，可以部署到不同厂家的机器人本体上。

---

### Skild AI — 通用机器人基础模型

2024 年成立，创始团队来自 Carnegie Mellon University。

**关键人物：**
- Deepak Pathak — 联合创始人（CMU 教授）
- Abhinav Gupta — 联合创始人（CMU 教授）

**定位：** 做通用机器人基础模型，强调泛化能力。论文未在 arXiv 大量公开。

---

### Sakana AI — 自进化 + AI Scientist

日本 AI 公司，虽然不是传统机器人公司，但在 Physical AI 的"自进化"方向有重要贡献。

**关键人物：**
- David Ha — 联合创始人/CTO（前 Google Brain 东京）
- Llion Jones — 联合创始人（Transformer 论文作者之一）

**核心工作：**
- AI Scientist：自动化完整科研流程（idea → 实验 → 论文 → 审稿），2026 年发表于 Nature
- Darwin Gödel Machine：Agent 修改自身代码的进化式自改进
- Recursive Self-Improvement Lab：明确把 RSI 作为公司方向

---

## 四、硬件层（机器人本体）

### 美国

| 公司 | 产品 | 特点 | 关键人物 |
|------|------|------|---------|
| Tesla | Optimus（人形） | 量产路线，成本控制 | Elon Musk |
| Figure AI | Figure 02（人形） | Helix VLA 模型（未公开论文） | Brett Adcock（CEO） |
| 1X Technologies | NEO（人形） | 挪威，家庭服务场景 | Bernt Børnich（CEO） |
| Apptronik | Apollo（人形） | 德州，工业场景 | Jeff Cardenas（CEO） |
| Agility Robotics | Digit（双足） | 俄勒冈，物流场景 | Damion Shelton（CEO） |
| Boston Dynamics | Atlas（人形） | 液压→电动转型，技术标杆 | Robert Playter（CEO） |

### 中国

#### 硬件本体公司

| 公司 | 产品 | 特点 | 城市 | 技术路线 |
|------|------|------|------|---------|
| 宇树科技（Unitree） | H1/G1（人形）+ 四足 | 成本控制极强，G1 消费级定价。2026.08.19 上市（人形机器人第一股） | 杭州 | 硬件成本领先→大规模部署→数据反哺模型 |
| 智元机器人（Agibot） | 远征 A1/远征 A2（人形） | 华为前天才少年彭志辉创立。已上宁德时代电池产线 | 上海 | 硬件+模型双路走，自研启智大模型 |
| 优必选（UBTECH） | Walker S（人形） | 上市最早的人形机器人公司。已入职极氪工厂 | 深圳 | 量产先行，工业场景验证 |
| 小米 | CyberOne（人形） | 生态整合，CyberDog 已消费级 | 北京 | 生态驱动，不大规模投入但做平台 |
| 追觅科技 | 通用人形 | 家电场景延伸（扫地机器人→人形） | 苏州 | 场景驱动，家电技能迁移 |
| 越疆科技 | Dobot 机械臂 | 已上市。从机械臂切入 | 深圳 | 机械臂起家，向上做人形 |

#### AI 模型公司

| 公司/团队 | 核心产品 | 技术路线 | 备注 |
|-----------|---------|---------|------|
| 银河通用（Galbot） | 通用抓取模型 | VLA + 大规模数据，不造本体只做大脑 | 2024 成立，融资规模大 |
| 星尘智能（Stardust AI） | Astribot S1 | 遥操作数据驱动 + VLA 模型。视频展示能力强 | 深圳 |
| 自变量机器人 | — | 聚焦操作策略学习 | 新公司 |
| 光轮智能 | — | 利用 NVIDIA Isaac Sim 生成合成数据 | 数据层切入 |

#### 中国学术 Embodied 模型团队

| 团队 | 模型 | 规模 | 特点 | 机构 |
|------|------|------|------|------|
| BAAI RoboBrain Team | RoboBrain 2.0/2.5 | 7B/32B | 统一脑模型，深度感知+时序建模。政府支持 | 北京智源 |
| ACE-Brain Team | ACE-Brain-0.5 | — | 理解→规划→执行→监控→改进完整闭环 | 多机构 |
| RynnBrain Team | RynnBrain 1.1 | 2B/9B/122B-A10B | 统一时空+物理接地框架，三规模 | — |
| Embodied-R1.5 团队 | Embodied-R1.5 | — | 统一 EFM：认知+任务规划+纠错+指向 | Yifu Yuan, Jianye Hao 等 |
| 上海交大 Cewu Lu 团队 | ForceVLA / ForceVLA2 | — | 力感知 VLA，接触丰富操作 | Cewu Lu, Wenqiang Zhang |
| 面壁智能 | MiniCPM-Robot | — | 端侧 VLA，在 PhyAI 引擎上适配 | 清华系 |
| 清华大学 | MoWM / RoboScape | — | 世界模型 + embodied 规划 | Yong Li, Wei Wu 等 |

#### 关键中国研究者

- **卢策吾（Cewu Lu）**— 上海交大教授，ForceVLA 系列。专注力感知操作，机器人视觉
- **李勇（Yong Li）**— 清华大学，MoWM/RoboScape/LongScape。embodied 世界模型
- **姚顺宇（Shunyu Yao）**— 前在 Princeton/Anthropic，2026 年加入面壁智能。姚班背景
- **彭志辉**— 智元机器人创始人，华为前天才少年。硬件+模型双路线

### 机器人本体公司的团队组成

```
典型人形机器人公司（500-1000 人）：

机械工程团队（30-40%）
  - 关节设计、结构强度、散热、材料
  - 核心壁垒：减速器、电机、编码器

电气团队（10-15%）
  - 电路设计、电源管理、传感器集成

嵌入式软件团队（15-20%）
  - 实时控制（PID/MPC）、电机驱动、通信
  - 控制频率 1000Hz+，延迟 <1ms

AI/算法团队（15-20%）
  - VLA 模型部署、感知、规划
  - 通常只有 10-30 人做模型
  - 大部分做工程化：数据管线、仿真、sim-to-real

产品/应用团队（10-15%）
  - 场景定义、客户需求、测试验证
```

**关键洞察：** 机器人本体公司中，做 AI 模型的人占比很小（15-20%），大部分人是做机械和嵌入式。这跟纯 AI 公司完全不同——Tesla 的 AI 团队可能几百人，但 Optimus 的机械团队可能比 AI 团队大。

---

## 五、学术层（大学实验室）

### Stanford（斯坦福）

Physical AI 学术圈的"半壁江山"。

**关键人物：**
- Chelsea Finn — 机器人学习，元学习。OpenVLA、Octo、RT-2 共同作者。创立 Physical Intelligence
- Sergey Levine — 深度强化学习奠基人。TRPO、SAC、 Offline RL。同时参与 Berkeley 和 Physical Intelligence
- Percy Liang — 基础模型评估，Stanford CRFM（基础模型研究中心）。OpenVLA 共同作者
- Russ Tedrake — 控制理论 + 机器人学习。OpenVLA 共同作者（MIT，但深度参与 Stanford 合作）
- Jiajun Wu — 物理场景理解，物体物理属性推理
- Shuran Song（宋舒然）— 机器人操作，抓取规划（现为 Columbia 教授但深度合作）

**核心贡献：** OpenVLA、Octo、DROID 数据集、Behavior-1K

### UC Berkeley

**关键人物：**
- Sergey Levine — （同时 affiliated with Stanford）深度强化学习，Offline RL
- Pieter Abbeel — 深度强化学习先驱，机器人学习。Deep RL+imitation learning
- Sudeep Dasari — 机器人操作，Octo 共同作者
- Ken Goldberg — 云机器人，Grasp Planning

**核心贡献：** Octo、DROID、Open X-Embodiment

### CMU（卡内基梅隆）

**关键人物：**
- Deepak Pathak — 机器人学习，Skild AI 联合创始人
- Abhinav Gupta — 视觉+机器人，Skild AI 联合创始人
- Katerina Fragkiadaki — 机器人学习
- David Held — 机器人感知，抓取

### Caltech（加州理工）

**关键人物：**
- Anima Anandkumar — Voyager、Eureka 共同作者。AI + 物理模拟
- Yisong Yue — 强化学习，机器人学习

**核心贡献：** Voyager（Minecraft 终身学习）、Eureka（LLM 自动设计 reward）

### MIT

**关键人物：**
- Russ Tedrake — 控制理论 + 机器人学习。OpenVLA 共同作者
- Leslie Kaelbling — 机器人规划，POMDP
- Pulkit Agrawal — 机器人操作，触觉感知

### 中国学术机构

**BAAI（北京智源人工智能研究院）：**
- RoboBrain Team — RoboBrain 2.0/2.5，统一脑模型，7B/32B 双版本
- 引入深度感知 + 时序建模

**其他中国团队：**
- ACE-Brain Team — ACE-Brain-0.5，统一 Embodied 基础模型。作者来自多个机构
- RynnBrain Team — RynnBrain 1.1，2B/9B/122B-A10B 三规模
- Embodied-R1.5 团队 — 作者包括 Yifu Yuan、Jianye Hao 等

---

## 六、关键大牛小传

### Sergey Levine（伯克利 + Stanford + Physical Intelligence）

深度强化学习领域的奠基人之一。如果你读过任何关于 Deep RL + 机器人的论文，大概率引用过他。

- 代表作：TRPO（信任区域策略优化）、SAC（Soft Actor-Critic）、Offline RL、DROID 数据集、Octo
- 特点：同时横跨学术界（Berkeley/Stanford）和工业界（Physical Intelligence 顾问）
- 影响力：他的学生遍布 Physical AI 各家公司

### Chelsea Finn（Stanford + Physical Intelligence）

机器人学习领域的核心人物。

- 代表作：MAML（模型无关元学习）、RT-2、OpenVLA、Octo
- 特点：元学习 → 机器人基础模型的演进路径，从"学会学习"到"学会操作"
- 同时是 Stanford 教授和 Physical Intelligence 联合创始人

### Anima Anandkumar（Caltech + NVIDIA）

AI + 物理模拟的桥梁人物。

- 代表作：Voyager（Minecraft 终身学习 Agent）、Eureka（LLM 自动设计 reward function）
- 特点：把 LLM 的推理能力引入机器人，不直接做端到端 VLA，而是用 LLM 做"元优化"（优化训练过程本身）

### Linxi "Jim" Fan（NVIDIA）

Voyager 和 Eureka 的 NVIDIA 方负责人。

- 代表作：Voyager、Eureka、MineDojo
- 特点：LLM + 环境交互的 Agent 路线，在 NVIDIA 内部推动 Embodied AI 研究

### Jeff Clune（不列颠哥伦比亚大学 + Sakana AI）

自进化 Agent 的核心推动者。

- 代表作：Darwin Gödel Machine、AI Scientist、Open-Endedness 理论
- 特点：不满足于"解决特定任务"，追求"开放式持续进化"

---

## 七、团队分工模式

### 模式一：大厂全栈（NVIDIA / Google）

```
NVIDIA 分工：
├── 芯片团队（Jetson Orin/Thor）
│   └── 硬件工程师 + 驱动软件工程师
├── 仿真平台团队（Isaac Sim / Omniverse）
│   └── 图形工程师 + 物理引擎工程师 + 工具开发
├── 世界模型团队（Cosmos，~100 人）
│   ├── 预训练组（视频数据 pipeline + 模型训练）
│   ├── 后训练组（特定场景微调）
│   └── 评估组（物理一致性验证）
├── 机器人模型团队（GR00T）
│   └── 策略学习 + VLA 模型
└── 生态系统团队
    └── 开发者关系 + 合作伙伴支持
```

### 模式二：AI 模型公司（Physical Intelligence / Skild AI）

```
Physical Intelligence 分工（估计 50-100 人）：
├── 模型团队
│   ├── 预训练（VLA backbone 训练）
│   ├── 策略学习（diffusion / flow matching 动作生成）
│   └── 评估（跨场景泛化测试）
├── 数据团队
│   ├── 遥操作采集（雇人操作机器人录制轨迹）
│   ├── 数据清洗 + 标注
│   └── 合成数据（仿真器生成）
├── 部署团队
│   ├── 模型压缩 + 量化
│   ├── 边缘推理优化
│   └── sim-to-real 验证
└── 硬件集成团队
    └── 策略部署到不同机器人平台
```

### 模式三：机器人本体公司（Figure / 宇树）

```
Figure AI 分工（估计 300-500 人）：
├── 机械团队（40%）
│   ├── 关节/执行器设计
│   ├── 结构 + 散热
│   └── 制造工程
├── 嵌入式团队（20%）
│   ├── 电机控制（1000Hz+ 实时）
│   ├── 传感器驱动
│   └── 通信协议
├── AI 团队（15-20%）
│   ├── 感知（VLM 部署）
│   ├── 规划（LLM 任务分解）
│   ├── 控制（VLA 策略）
│   └── 数据管线
├── 软件平台团队（10%）
│   ├── 机器人 OS
│   ├── 远程监控 + OTA
│   └── 开发者工具
└── 产品/商务团队（10%）
```

### 模式四：学术实验室（Stanford / Berkeley）

```
Stanford 机器人学习实验室（Finn 组，~15 人）：
├── 教授（1 人）— 方向决策 + 指导
├── 博士后（2-4 人）— 核心研究 + 论文写作
├── 博士生（8-12 人）— 具体实验 + 代码
├── 硕士生/本科生（2-4 人）— 辅助实验
└── 访问学者（1-2 人）

特点：
- 每人通常负责 1-2 个研究课题
- 论文导向：每篇论文 3-6 个作者
- 与工业界深度合作：学生去 NVIDIA/Google/PI 实习
- 数据共享：DROID、Open X-Embodiment 等数据集是多校联合成果
```

---

## 八、人才流动网络

Physical AI 领域的人才流动有几个显著模式：

```
1. 学术→创业
   Chelsea Finn（Stanford）→ Physical Intelligence
   Sergey Levine（Berkeley）→ Physical Intelligence（顾问）
   Deepak Pathak（CMU）→ Skild AI
   Abhinav Gupta（CMU）→ Skild AI
   David Ha（Google Brain）→ Sakana AI

2. 大厂→创业
   Karol Hausman（Google Brain）→ Physical Intelligence
   Llion Jones（Google Brain）→ Sakana AI

3. 学术→大厂
   Anima Anandkumar（Caltech）→ NVIDIA 顾问
   多位 Stanford/Berkeley 学生 → NVIDIA/Google 实习→全职

4. 跨机构合作（常态）
   OpenVLA：Stanford + Google + TRI（Toyota Research Institute）
   Octo：Berkeley + Stanford + Google
   DROID：Stanford + Berkeley + CMU + Google + TRI + 20+ 机构
   Open X-Embodiment：Google + 32 家机构
```

**关键洞察：** Physical AI 的人才池比 LLM 小得多。LLM 领域有几十万 NLP 研究者可以转化，但同时懂深度学习和机器人控制的人很少。这就是为什么头部公司都在抢同一批人——Chelsea Finn、Sergey Levine、Anima Anandkumar 这些名字出现在大量论文的作者列表中。

---

## 九、中国在 Physical AI 的位置

### 中国的优势

1. **硬件制造**：宇树、智元、优必选在机械设计和制造效率上有成本优势
2. **数据采集**：人力成本低，遥操作数据采集成本远低于美国
3. **应用场景**：制造业、物流、服务业的机器人需求量大

### 中国的劣势

1. **基础模型**：VLA、世界模型的原创性工作主要在美国。中国的 Embodied 基础模型（RoboBrain、ACE-Brain、RynnBrain）起步较晚
2. **仿真生态**：NVIDIA Isaac Sim / Omniverse 是事实标准，中国没有对标产品
3. **顶尖人才**：Sergey Levine、Chelsea Finn 级别的研究者在中国极度稀缺
4. **芯片**：Jetson 是主流选择，国产替代仍在早期

### 中国值得关注的方向

- RoboBrain 系列（BAAI 智源）：统一脑模型，有政府支持
- 宇树科技：硬件成本领先，可能走"硬件优势→数据优势→模型优势"路线
- 银河通用、星动纪元等新公司：大量融资但论文产出待观察

---

## 十、Physical AI 处在什么阶段

### 类比 LLM 的发展阶段

要判断 Physical AI 在哪一步，最好的方法是跟你熟悉的 LLM 做类比：

```
LLM 发展时间线：
  2018 GPT-2     → 模型能生成连贯文本，但不可靠
  2020 GPT-3     → 涌现能力，但需要 prompt engineering
  2022 ChatGPT   → 可用，进入产品化
  2024 GPT-4o    → 多模态，大规模商业化
  2025+          → Agent 化，自进化

Physical AI 发展时间线：
  2022 RT-1      → VLA 概念验证，类比 GPT-2 阶段
  2023 RT-2      → VLM 知识迁移到机器人，类比 GPT-3
  2024 π0/OpenVLA→ 开源 VLA 爆发，类比 LLM 开源潮
  2025 GR00T/Cosmos→ 基础设施成型，但可靠性仍不足
  2026 当前       → 大约在 LLM 的 2022 年（ChatGPT 前夜）
```

Physical AI 目前大约在 **"ChatGPT 前夜"**——技术已经证明可行，但还没有一个"ChatGPT 时刻"让所有人觉得机器人真的好用了。

### 当前阶段的具体特征

**已经实现的：**
- 单一任务可靠执行（抓取特定物体、放置到特定位置）
- 仿真器中大规模训练→真实世界部分迁移
- VLA 模型在 LIBERO 等标准 benchmark 上达到 90%+ 成功率
- 人形机器人在工厂执行重复性任务（优必选在极氪工厂、智元在宁德时代产线）

**还没实现的：**
- 跨场景泛化：换个没见过的厨房就不会做饭了
- 长程任务：能抓杯子但不能连续做一杯咖啡（10+ 步骤）
- 柔性物体操作：折衣服、切菜、拧毛巾
- 真正自主：目前都需要人在旁边监控，随时准备接管
- 成本可接受：一个能干活的人形机器人成本仍然 10 万美元+

**一句话总结当前阶段：**

  Physical AI 已经过了"能不能做"的阶段，但还在"能不能用"的阶段挣扎。类比 LLM，就是有了 GPT-3 但还没有 ChatGPT。

---

## 十一、缺少什么

### 缺少一：数据（最根本的瓶颈）

这是整个领域的第一瓶颈，所有其他问题都跟它有关。

```
LLM 训练数据量：~15 万亿 token（从互联网爬取，几乎零成本）
VLA 训练数据量：~100 万条轨迹（Open X-Embodiment，多机构联合几年才凑齐）

差距：1500 万倍
```

数据为什么缺？前面讲过——机器人数据必须用真机或仿真器录制，不能爬取。一条高质量操作轨迹需要：

1. 一个可用的机器人（成本 5-50 万人民币）
2. 一个训练有素的操作员（时薪 100-300 元）
3. 每条轨迹 1-5 分钟
4. 10 万条 = 2000-8000 小时 = 250-1000 人天

而且数据不通用——不同机器人的关节配置不同，A 机器人的数据不能直接给 B 机器人用。这跟文本数据完全不同（一段文本任何模型都能读）。

世界模型（Cosmos）和仿真器（Isaac Sim）都在试图解决这个问题，但目前合成数据的物理真实性还不够。

### 缺少二：评估标准

LLM 有 MMLU、HumanEval、GSM8K 等标准 benchmark，不同模型可以横向比较。

Physical AI 没有。目前的 benchmark 各自为政：

```
LIBERO：仿真桌面操作（简单，不接近真实）
DROID：真实机器人数据但只评估抓取
SWE-bench for robots？不存在
```

没有统一 benchmark 意味着：
- 你不知道你的模型比别人好还是差
- 你不知道改进了感知还是规划还是动作执行
- 投资人不知道哪家公司真有技术

你笔记里的 PAST-Bench、RADAR、Apple-π 都在试图填补这个空缺，但目前还没有像 MMLU 那样的共识标准。

### 缺少三：仿真到真实的闭环

目前 sim-to-real 是单向的：仿真器训练→部署到真实→如果不行，人类工程师调参→重新训练。

缺少的是自动闭环：部署到真实后，收集失败案例→自动反馈到仿真器→自动改进策略→自动重新部署。

你笔记里的 Real2Sim2Real 在做这个方向，但自动化程度还不够。大部分情况下仍然需要大量人工干预。

### 缺少四：硬件标准化

LLM 跑在 GPU 上，GPU 有 CUDA 统一编程。换一张 GPU 代码几乎不用改。

机器人不是。不同机器人的关节数、关节限位、力矩范围、传感器配置都不同。你在 Franka 机械臂上训练的策略，迁移到宇树 G1 上需要大量适配。

这导致每个机器人公司都在建自己的数据池，数据无法跨平台共享（虽然 Open X-Embodiment 在试图标准化，但还很早期）。

### 缺少五：安全验证方法

LLM 输出错误文本，最多被用户吐槽。机器人输出错误动作，可能伤人。

目前的安全保障主要靠：
- 物理围栏（把机器人关在笼子里）
- 急停按钮（人在旁边随时按）
- 力矩限制（超过阈值自动停）

这些是"硬安全"。但缺少"软安全"——机器人在执行前能否判断"这个动作是否有风险"？你笔记里的 Runtime Action Authorization、Initiation Safety 都在做这个方向，但远未成熟。

---

## 十二、缺少什么类型的人才

### 最稀缺：同时懂 AI 和物理的"翻译者"

Physical AI 最大的人才缺口不是纯 AI 研究者，也不是纯机械工程师，而是能同时理解两边的"翻译者"。

具体来说，需要一个人同时理解：
- 深度学习（Transformer、diffusion、RL）
- 控制理论（PID、MPC、系统辨识）
- 机器人学（运动学、动力学、接触力学）
- 硬件工程（电机、传感器、嵌入式系统）

这种人在全球都不多。Sergey Levine、Chelsea Finn 之所以值钱，正是因为他们横跨 AI 和机器人。大部分 AI 研究者不懂接触力学，大部分机器人工程师不懂 diffusion model。

### 五类急需人才

**1. Sim-to-Real 工程师（极度稀缺）**

能把仿真器里的策略成功迁移到真实机器人的工程师。需要同时懂仿真器物理引擎、域随机化、系统辨识。目前这类人主要在 NVIDIA 和几家头部公司。

**2. 机器人数据工程师（新兴岗位）**

不是标注数据，而是设计数据采集 pipeline：怎么遥操作最高效、怎么保证数据质量、怎么用仿真器扩增、怎么管理跨机器人数据格式。这个岗位在 LLM 领域不存在，是 Physical AI 独有的。

**3. 实时 AI 推理工程师**

能把 7B VLA 模型优化到在 Jetson 上跑 25Hz 的工程师。需要同时懂模型量化、CUDA 编程、机器人控制频率约束。你笔记里 PhyAI、Kairos、Jetson-PI 的作者就是这类人。

**4. 触觉与力感知研究者**

视觉 VLA 已经相对成熟，但触觉还非常原始。机器人不知道自己握力多大、不知道物体是否在滑动、不知道接触面是否稳定。需要懂触觉传感器硬件 + 信号处理 + 策略学习的复合人才。你笔记里 TouchWorld、Tactile Genesis 的作者在做这个方向。

**5. Physical AI 安全工程师**

能在机器人执行前评估动作风险的工程师。需要懂形式化验证、风险评估、人机协作安全标准。这个岗位几乎没有成熟的人才来源，需要从自动驾驶安全或航空航天安全领域转化。

### 中国特别缺的人才

在上述五类的基础上，中国还额外缺：

**6. 仿真生态建设者**

NVIDIA 有 Isaac Sim + Omniverse + CUDA，中国没有对标产品。这不是一个算法问题，而是一个需要几百人团队、数年投入的工程问题。中国目前没有团队在做这个。

**7. Embodied 基础模型研究者**

能从零训练 VLA 或世界模型的人。中国目前只有 BAAI RoboBrain、ACE-Brain 等少数团队在做，而且原创性架构不多，大部分是在已有 LLM backbone 上接动作头。

**8. 机器人硬件+AI 双栈工程师**

中国有大量优秀的机械工程师（宇树、智元的硬件水平不输美国），也有大量 AI 研究者，但能把两者打通的人极少。美国有 Chelsea Finn 这类横跨两端的人，中国目前没有同等影响力的角色。

---

## 十三、中国的机会与风险

### 机会

1. **硬件成本优势**：宇树 G1 价格已经打到消费级，美国同类产品价格 5-10 倍。如果数据量是关键瓶颈，低成本硬件 = 更多部署 = 更多数据
2. **场景丰富**：中国制造业、物流、餐饮场景全球最丰富。这些场景可以提供海量真实任务数据
3. **政策支持**：2026 年国家发布《人形机器人与具身智能实景实训专项行动》，政策力度大
4. **数据采集成本**：遥操作人员时薪远低于美国，同样预算可以采集更多数据

### 风险

1. **基础模型依赖**：几乎所有中国团队都在用 NVIDIA 的芯片、仿真器和（部分）模型。如果制裁加码，基础层会断
2. **论文导向 vs 产品导向**：中国学术团队倾向于发论文（RoboBrain、ACE-Brain 都很快发了技术报告），但在真实场景的长期验证不足
3. **仿真生态空白**：没有自主的仿真器和物理引擎，意味着数据生成基础设施完全依赖 NVIDIA
4. **顶尖人才流失**：最优秀的中国机器人研究者（如 Anima Anandkumar 级别）大多在美国机构。姚顺宇 2026 年加入面壁智能是少见的回流案例

---

## 十四、人才可能来自哪里

Physical AI 的人才池比 LLM 小得多——同时懂深度学习和机器人控制的人全球都不多。但并不是只能从机器人实验室招人。以下按转化难度从低到高排列。

### 第一梯队：直接可转化（转化周期 1-3 个月）

**1. 自动驾驶（最接近的领域）**

自动驾驶和 Physical AI 的技术栈重叠度极高：

```
自动驾驶                          Physical AI
──────────────────────────────────────────────
多摄像头感知                      多摄像头感知
激光雷达 3D 建图                  3D 空间理解
实时规划（10Hz+）                 实时规划（10-25Hz）
安全约束（撞了要命）              安全约束（碰了伤人）
仿真器训练（CARLA）               仿真器训练（Isaac Sim）
端到端模型（BEV→规划）            端到端模型（像素→动作）
```

小鹏、蔚来、Momenta、文远知行里的感知工程师和规控工程师，只要对机器人有兴趣，几乎可以平移。他们已经习惯了"在不可逆的物理世界里做实时决策"这个约束。

实际上已经发生了：Tesla 的 Autopilot 团队和 Optimus 团队共享了大量技术栈。Waymo 也有团队在研究机器人操作。

**2. 手术机器人（Intuitive Surgical / 国产腔镜机器人）**

可能是所有来源中最被低估的。他们每天都在解决 Physical AI 的核心问题：
- 毫米级精度操作（比抓杯子难 100 倍）
- 力反馈（医生需要"感觉"到组织阻力）
- 安全约束（切错了就是医疗事故）
- 人机协作（医生和机器人共同操作）

手术机器人领域的人转到通用 Physical AI，技术上是"降维打击"——他们处理的问题比抓杯子复杂得多。

**3. 工业自动化（KUKA / ABB / 发那科）**

这些公司的工程师懂传统机器人的一切——运动学、动力学、接触力学、安全标准。他们缺的是深度学习知识，但这个可以通过短期学习补上。

反过来让 AI 研究者学机器人学，比让机器人工程师学 AI 难得多——因为机器人学涉及大量物理直觉，而深度学习在很大程度上是"调参 + 数据"。

### 第二梯队：部分可转化（转化周期 3-12 个月）

**4. 游戏引擎 / 图形学（Unreal / Unity / NVIDIA Omniverse）**

NVIDIA 的 Isaac Sim 和 Omniverse 团队大量招的就是游戏引擎和图形学背景的人。他们懂物理引擎（碰撞检测、刚体动力学）、实时渲染、场景建模、传感器模拟。缺机器学习和控制理论，但仿真器团队不一定需要自己懂 VLA。

**5. 强化学习（游戏 AI / DeepMind / OpenAI）**

DeepMind 做 AlphaGo、StarCraft AI 的人，OpenAI 做 OpenAI Five（Dota 2）的人，技术上离 Physical AI 很近——已经在做"在复杂环境中通过 RL 学习策略"。

关键差异：游戏环境是完美模拟的，物理世界不是。AlphaGo 可以自我对弈几千万局因为围棋规则完美定义。机器人不能在真实世界试错几千万次。游戏 AI 的人需要补的核心一课是 sim-to-real——从"完美仿真器"到"不完美的仿真器"。

**6. 计算机视觉（3D 视觉 / SLAM）**

做 3D 重建、SLAM（同步定位与建图）、深度估计的 CV 研究者，可以直接转到机器人感知层。已理解 3D 几何、多视图几何、点云处理。缺的是下游知识（感知怎么接到动作）和控制理论。

**7. 航空航天 / 无人机（DJI / 大疆）**

无人机领域的人懂实时控制（姿态稳定 400Hz+）、传感器融合（IMU + GPS + 视觉）、安全约束（坠机要命）、嵌入式系统。大疆可能是中国 Physical AI 最大的隐藏人才库——几千名懂实时控制和嵌入式 AI 的工程师。

### 第三梯队：需要大量重新学习（转化周期 1-2 年）

**8. 纯 LLM / NLP 研究者**

看起来是最自然的来源，实际上转化难度最高。

LLM 研究者懂 Transformer、懂训练、懂 inference 优化。但通常缺少：
- 物理直觉（不知道摩擦力、接触力、惯量怎么影响动作）
- 控制理论（不知道 PID、MPC、状态空间）
- 硬件经验（从没碰过真实机器人）
- 安全思维（习惯了"输出错了改了重来"）

最关键的是思维方式的转变：LLM 的输出是文本，容错率极高。Physical AI 的输出是动作，容错率为零。很多优秀的 LLM 研究者转做机器人后会极度不适应——"为什么模型在仿真器里 97% 成功率，放到真机只有 12%？"

这就是为什么 Chelsea Finn 值钱——她先学机器人学再学 AI，顺序对了。纯 AI 背景的人转机器人，需要先"忘掉"容错思维。

**9. 机械工程 / 传统制造业**

机械工程师懂物理、材料、结构，但需要从头学深度学习。同时做过机械设计和编程的人（比如机器人竞赛背景）转化速度会快很多。

### 最被忽视的来源

**10. 动画 / VFX / 物理模拟（皮克斯 / 工业光魔）**

物理动画师是世界上最懂"柔性物体如何形变"的人。他们手动调过布料模拟、流体模拟、软体碰撞——这些恰恰是 Physical AI 仿真器最薄弱的环节。Newton 物理引擎的开发离不开这类人。

**11. 远程操作员 / 遥操作专家**

非常非传统的来源。有些人长期做机器人遥操作数据采集，虽然不做算法，但对"什么动作难、什么动作容易出问题"有最直接的体感。这种人转去做数据策略设计或评估标准，比 PhD 更有 practical insight。

### 中国的特殊人才来源

- **大疆（DJI）**：中国最大的 Physical AI 人才库。几千名懂飞控、嵌入式 AI、实时控制的工程师。农业无人机已经在做自主作业
- **RoboMaster 体系**：大疆办的机器人比赛，每年几万大学生参加。这些学生本科期间同时接触机械设计、嵌入式编程、视觉算法——"双栈"思维基础已打好
- **自动驾驶行业溢出**：中国自动驾驶 2023-2025 年经历收缩，大量感知和规控工程师在找工作。是 Physical AI 最直接的可转化人才
- **华大基因 / 医疗器械**：医疗领域的人懂精密操作和安全标准，转化路径跟手术机器人类似

### 人才转化路径总览

```
转化难度    来源领域               已有技能                    需要补的
────────────────────────────────────────────────────────────────────
低(1-3月)  自动驾驶              感知+规划+实时+安全          机器人控制
           手术机器人            精密操作+力反馈+安全         通用化
           工业自动化            运动学+动力学+安全标准       深度学习

中(3-12月) 游戏引擎/图形学       物理模拟+渲染+仿真器         ML+控制
           游戏AI/RL             RL+策略学习+自博弈           sim-to-real
           3D视觉/SLAM           感知+3D几何                  规划+控制
           无人机/航空航天       飞控+嵌入式+实时             操作+ML

高(1-2年)  LLM/NLP              Transformer+训练+推理        物理+控制+安全
           机械工程              物理+材料+结构               深度学习全栈
```

### 判断标准

最理想的 Physical AI 工程师简历不是"MIT 机器人学 PhD"，而是"大疆做了 3 年飞控，然后去自动驾驶公司做了 2 年感知"。这种人有真实的物理世界部署经验，懂实时约束，又已经转化过一次。

---

## 延伸阅读

- [PhysicalAI-Feynman.md](../8.tutorial/PhysicalAI-Feynman.md) — 费曼学习笔记
- [PhysicalAI.md](../5.multimodal/PhysicalAI.md) — 论文索引
- [Cosmos-Interview.md](../5.multimodal/Cosmos-Interview.md) — 英伟达刘洺堉访谈
- [SelfEvolving.md](../7.product/SelfEvolving.md) — 自进化 Agent（含 Sakana AI / Darwin Gödel Machine）
