# AI 小说生成 Agent

> 整理日期: 2026-08-06 | 更新: 2026-08-06

## 概览

AI 小说生成是 LLM 的重要应用方向。核心挑战：长文本一致性、角色/伏笔管理、上下文窗口限制、去 AI 味。

---

## 开源项目

### 1. InkOS — 故事创作 AI Agent 系统

- **Repo:** [Narcooo/inkos](https://github.com/Narcooo/inkos)
- **Stars:** 8.8k
- **语言:** TypeScript / Node.js (npm: `@actalk/inkos`)
- **License:** AGPL-3.0 | **最新版本:** v1.7

**定位:** 面向长短篇小说、剧本剧作、互动影游、IP 内容与多语言翻译的创作智能体系统。

**核心特性:**
- Studio (Web) / TUI / CLI 三种交互形式
- 长篇小说: 创作简报建书 → 世界观/角色/卷纲/章节意图 → 写作 → 审稿 → 修订 → 状态结算
- 剧情多线推演: 基于"正史"创建 2-5 条隔离候选未来，横向比较
- InkOS Play: 开放世界互动 (角色 agent、物品/证据/关系状态、HUD)
- InkOS Short: 独立短篇一键生成
- 完整翻译工作台: EPUB/PDF/TXT/MD，术语表 + 对照审校
- Skill 系统: 标准 `SKILL.md` 可插拔能力包
- 联网研究: `research_web` 带来源参考报告
- 上下文管理: protected/compressible 语义压缩 + SQLite 记忆
- 已发布为 OpenClaw Skill (`clawhub install inkos`)
- 支持 OpenAI 兼容 API、Moonshot (Kimi)、OpenRouter 等

**亮点:** 功能最全面，覆盖小说/剧本/互动影游/翻译多场景，Agent 架构成熟。

---

### 2. AI_NovelGenerator — 多章节长篇小说生成器

- **Repo:** [YILING0013/AI_NovelGenerator](https://github.com/YILING0013/AI_NovelGenerator)
- **Stars:** 高 (topic 下最热门之一)
- **语言:** Python | **License:** 开源
- **状态:** 重构中 (2026/03/25 宣布重构，dev 分支)

**定位:** 基于 LLM 的多章节长篇小说生成工具，自动衔接上下文和伏笔。

**核心特性:**
- 小说设定工坊: 世界观 / 角色设计 / 情节蓝图
- 智能章节生成: 多阶段生成保证情节连贯
- 状态追踪: 角色发展轨迹 / 伏笔管理
- 语义搜索引擎: 向量检索长程上下文一致性
- 知识库集成: 支持本地文档参考
- 自动校对: 检测情节矛盾和逻辑冲突
- 可视化工作台: GUI 全流程操作
- 多模型配置: 支持不同任务路由不同模型 (大纲/正文/校对各用不同模型)

**亮点:** 向量检索做长程一致性 + 多模型任务路由，工程化程度高。

---

### 3. MuMuAINovel — AI 智能小说创作助手

- **Repo:** [xiamuceer-j/MuMuAINovel](https://github.com/xiamuceer-j/MuMuAINovel)
- **Stars:** 中等 | **语言:** Python (FastAPI) + React
- **License:** GPL v3 | **最新版本:** v1.5.4

**定位:** 基于 AI 的智能小说创作助手，Web 应用 + Docker 部署。

**核心特性:**
- 多 AI 模型: OpenAI / Gemini / Claude
- 智能向导: AI 自动生成大纲、角色和世界观
- 角色管理: 人物关系、组织架构可视化
- 伏笔管理: 智能追踪 + 未回收线索提醒 + 时间线可视化
- 提示词工坊: 社区驱动 Prompt 模板分享
- 拆书功能: 一键拆书分析已有作品
- 职业等级体系: 支持修仙境界、魔法等级等
- PostgreSQL 多用户数据隔离 + Docker 一键部署

**亮点:** 产品化程度高，多用户支持，Docker 开箱即用，伏笔可视化。

---

### 4. Show Me The Story — 自托管 AI 小说生成器

- **Repo:** [Nigh/show-me-the-story](https://github.com/Nigh/show-me-the-story)
- **Stars:** 438 | **语言:** Go + Svelte | **License:** 开源

**定位:** 单文件运行的 AI 长篇小说创作工具，自托管，OpenAI 兼容 API。

**核心特性:**
- 单二进制文件: 无需数据库，一个 Go binary + 浏览器
- 两阶段创作: 先大纲后逐章写作
- 逐章审核 + 自动确认模式
- 伏笔系统: 埋设 → 推进 → 回收，超期告警
- 叙事记忆: 自动提取大纲未体现的关键细节，跨章注入
- 事实核查: 每章自动一致性检查，不通过自动重写
- 续写已有作品 + 去 AI 味润色
- 全书优化: 诊断 → 一致性核查 → 优化工单 → 逐章修订 → diff 对比
- 技能系统 + AI 助理 (聊天式操作项目)
- 中英双语，数据纯文本/JSON

**亮点:** 单文件部署最简，伏笔+叙事记忆+事实核查三件套，全书优化流程完整。

---

### 5. awesome-novel-skill — AI Agent 小说创作 Skill

- **Repo:** [modoojunko/awesome-novel-skill](https://github.com/modoojunko/awesome-novel-skill)
- **Stars:** 509 | **语言:** Python | **License:** GPL 3.0

**定位:** 让 AI Agent (Claude Code / OpenCode / Reasonix) 成为小说创作搭档的 Skill 包。

**核心特性:**
- 8 Agent 协作架构: novel-agent (总指挥) → updater / volume-planner / chapter-planner / prompt-crafter / writer / anti-ai / reader
- 24 套预置题材画像
- 四维规划: 情绪走向 / 冲突阶梯 / 信息差 / 场景卡
- 写作动态记忆: 卷/章/提示词/写作四级 memory
- 支持 Claude Code / OpenCode / Reasonix / OpenClaw

**亮点:** 纯 Skill 方案，多 Agent 协作，与 coding agent 生态深度集成。

---

### 6. oh-story-claudecode — 网文写作全流程 Skill 包 ⭐ 新发现

- **Repo:** [worldwonderer/oh-story-claudecode](https://github.com/worldwonderer/oh-story-claudecode)
- **Stars:** 5.1k | **语言:** JavaScript

**定位:** 覆盖长篇与短篇网络小说的扫榜、拆文、写作、去AI味、封面图全流程。

**核心特性:**
- 三步方法论: 扫榜 (分析热门榜单) → 拆文 (拆解大纲节奏) → 商业化写作
- 13 个 Skill: story-setup / story-long-write / story-long-analyze / story-long-scan / story-short-write / story-short-analyze / story-short-scan / story-deslop / story-import 等
- 7 个专业 Agent: story-architect / narrative-writer / consistency-checker 等
- 长篇追踪状态模型: 单一权威 `_tracking-state.json`，续写状态卡 + 伏笔/时间线/角色快照
- 本地工作台 Dashboard
- 支持 Claude Code / OpenCode / ZCode / OpenClaw / Codex CLI / Reasonix
- 核心理念: "套路 = 确定性的情绪满足"

**亮点:** 最贴近中文网文生态 (起点/番茄/晋江扫榜)，商业化写作导向，去 AI 味治理成熟。

---

### 7. AI-Novel-Writing-Assistant — AI 导演式长篇小说生产系统 ⭐ 新发现

- **Repo:** [ExplosiveCoderflome/AI-Novel-Writing-Assistant](https://github.com/ExplosiveCoderflome/AI-Novel-Writing-Assistant)
- **Stars:** 2.3k | **语言:** TypeScript | **桌面版:** Windows EXE

**定位:** 面向长篇小说完成度的 AI Native 生产系统，从一句灵感到完整小说。

**核心特性:**
- AI 自动导演开书: 一句灵感 → 整本方向/角色/卷章规划 → 正文生产
- Creative Hub: 统一创作中枢 (对话/规划/工具调用/任务状态)
- 整本生产主链: 章节生成 → AI 审核 → 修复 → 角色状态回灌
- 拆书工作台: 四档角色档案 + 角色形象演变追踪
- 写法引擎: 可保存/编辑/绑定的写法资产 + 反 AI 规则
- RAG: Qdrant 向量检索，拆书产物入索引
- 漫画与短剧衍生工坊
- 模型路由: 多提供商按任务拆分
- Windows 桌面版可直接下载安装

**技术栈:** React + Vite + Express + Prisma + LangChain + LangGraph + Qdrant

**亮点:** LangGraph 编排的生产链最完整，"导演式"全自动开书，角色资源账本精准上下文筛选。

---

### 8. Humanizer — 去 AI 味 Agent Skill ⭐ 新发现

- **Repo:** [blader/humanizer](https://github.com/blader/humanizer)
- **Stars:** 33.9k | **语言:** Python (SKILL.md)

**定位:** 移除 AI 写作痕迹的 Agent Skill，基于 Wikipedia "Signs of AI writing" 指南。

**核心特性:**
- 33 种 AI 写作模式检测 (意义膨胀、署名堆砌、推广语言、模糊归因等)
- 两轮重写: 第一轮去 AI 味 + 第二轮审计
- 声音校准: 可提供个人写作样本匹配风格
- 无捏造规则: 不添加原文没有的事实/名字/日期
- 适配所有支持 SKILL.md 的 Agent (Claude Code / OpenClaw 等)

**亮点:** Star 数最高 (33.9k)，专注去 AI 味这一垂直场景，基于维基百科 AI 清理项目的方法论。

---

### 9. novel-studio — 多智能体世界推演小说生成引擎

- **Repo:** [Xiaoyangy/novel-studio](https://github.com/Xiaoyangy/novel-studio)
- **Stars:** 69 | **语言:** Go

**定位:** 开源、本地优先的 AI 长篇小说创作引擎。

**核心特性:**
- 多智能体世界推演
- 按弧规划 (Story Arc Planning)
- RAG 长程记忆 (Qdrant)
- 逐章审核与断点恢复
- OpenAI 兼容 API

---

### 10. Recurrent-LLM — RecurrentGPT 论文实现

- **Repo:** [jackaduma/Recurrent-LLM](https://github.com/jackaduma/Recurrent-LLM)
- **Stars:** 206 | **语言:** Python

**定位:** RecurrentGPT 论文的开源实现，交互式任意长文本生成。

---

### 11. novel-writer-skills — 零成本 AI 中文小说

- **Repo:** [Shine8592/novel-writer-skills](https://github.com/Shine8592/novel-writer-skills)
- **Stars:** 15 | **语言:** Python

**定位:** 零成本 AI 中文小说写作，3 个 OpenClaw Skill，¥0 API 花费，已生成 90 万字+。

---

## 自动发布工具

> 整理日期: 2026-08-06

AI 写完小说后，手动上传各平台耗时耗力。以下工具可自动将章节发布到小说平台。

### 1. fanqie_auto_publish ⭐ 最成熟

- **Repo:** [hchcx/fanqie_auto_publish](https://github.com/hchcx/fanqie_auto_publish)
- **语言:** Python | **依赖:** Playwright
- **平台:** 番茄小说

**核心特性:**
- 基于 Playwright 全自动网页端发文与状态管理
- 批量 .txt 章节文件精准投递并发布
- 断点续检 + 弹窗斩杀机制
- 无人值守运行
- Linux.do 有开源讨论帖

### 2. fanqie-publisher-skill（OpenClaw Skill）

- **来源:** [ClawHub](https://clawhub.ai/followaf/skills/fanqie-publisher-skill)
- **语言:** Python | **依赖:** Playwright
- **平台:** 番茄小说

**核心特性:**
- OpenClaw Skill，可直接 `clawhub install` 安装使用
- 支持单章/批量发布、存草稿箱
- 登录番茄作家后台、查看作品列表和状态
- 触发词：番茄发布、番茄小说上传、发布章节等

### 3. tomato-novel-mcp-tool

- **Repo:** [WTW-NiuBi/tomato-novel-mcp-tool](https://github.com/WTW-NiuBi/tomato-novel-mcp-tool)
- **语言:** TypeScript | **协议:** MCP
- **平台:** 番茄小说

**核心特性:**
- MCP 协议工具，可接入任意支持 MCP 的 Agent
- AI 生成章节 + 原创性检测 + 自动发布
- 数据统计和分析
- 定时发布功能

### 4. webfiction-write

- **Repo:** [qingyabailu-git/webfiction-write](https://github.com/qingyabailu-git/webfiction-write)
- **语言:** Python
- **平台:** 番茄、起点、七猫、刺猬猫（面向这些平台优化）

**核心特性:**
- 基于 Codex 的网文创作工具集 Skill
- 从构思到完本完整工作流（偏创作侧，非纯发布）

### 5. OpenWrite（通用多平台发布）

- **网站:** [openwrite.cn](https://openwrite.cn)
- **类型:** SaaS
- **平台:** 掘金、CSDN、知乎等技术内容平台

**核心特性:**
- 写一次同步到多个内容平台
- Markdown 写作与内容管理
- ⚠️ 偏技术文章，非小说专属

### 平台覆盖情况

| 平台 | 专用工具 | 自动发布可行性 |
|------|---------|-------------|
| **番茄小说** | 3 个专用工具 | ✅ 最成熟 |
| **起点中文网** | 无现成工具 | ⚠️ 作家后台反爬严格 |
| **七猫** | 无专用工具 | ⚠️ 需自研 |
| **刺猬猫** | 无专用工具 | ⚠️ 需自研 |

### 技术方案建议

1. **番茄优先** — 装 `fanqie-publisher-skill`（OpenClaw 原生）或 `fanqie_auto_publish`（独立脚本）
2. **起点/七猫** — 基于 Playwright 自研，参考 `fanqie_auto_publish` 架构改造
3. **通用方案** — 各平台作家后台用浏览器自动化模拟登录 + 上传，配合定时任务
4. **注意事项** — 各平台对自动化发文有不同限制策略，需关注账号安全

---

## 论文资源

### 论文合集

**Awesome-Story-Generation** — LLM 时代故事生成论文合集 ⭐
- **Repo:** [yingpengma/Awesome-Story-Generation](https://github.com/yingpengma/Awesome-Story-Generation)
- **Stars:** 643
- **维护者:** Yingpeng Ma, Yan Ma
- 收录 LLM 时代故事生成相关论文，按方向分类，持续更新

### 重点论文（按方向）

#### Plan And Write (规划与写作)

| 论文 | 会议 | 核心思路 |
|------|------|---------|
| Learning to Reason for Long-Form Story Generation | ArXiv-2025 | 推理能力用于长篇故事生成 |
| Generating Long-form Story Using Dynamic Hierarchical Outlining with Memory-Enhancement | NAACL-2025 | 动态分层大纲 + 记忆增强 |
| Collective Critics for Creative Story Generation | EMNLP-2024 | 多批评家协作提升创意 |
| Ex3: Automatic Novel Writing by Extracting, Excelsior and Expanding | ACL-2024 | 提取-提升-扩展自动小说写作 |
| Navigating the Path of Writing: Outline-guided Text Generation | NAACL-2025 | 大纲引导文本生成 |
| EIPE-text: Evaluation-Guided Iterative Plan Extraction | ArXiv-2023 | 评估引导的迭代计划提取 |

#### Multi Agent (多智能体)

| 论文 | 会议 | 核心思路 |
|------|------|---------|
| HAMLET: Hierarchical and Adaptive Multi-Agent Framework | ICLR-2026 | 分层自适应多 Agent 框架，现场戏剧表演 |
| Agents' Room: Narrative Generation through Multi-step Collaboration | ICLR-2025 | 多步协作叙事生成 (Google DeepMind) |
| IBSEN: Director-Actor Agent Collaboration | ACL-2024 | 导演-演员 Agent 协作生成可控剧本 |
| HoLLMwood: Screenwriting via Role Playing | EMNLP-2024 | 角色扮演释放 LLM 编剧创意 |
| StoryVerse: Co-authoring Dynamic Plot via Narrative Planning | FDG-2024 | 虚拟角色 + 叙事规划协同创作 |

#### Better Storytelling (更好讲故事)

| 论文 | 会议 | 核心思路 |
|------|------|---------|
| Finding Flawed Fictions: Plot Hole Detection | ArXiv-2025 | 检测剧情漏洞评估推理能力 |
| Improving Pacing in Long-Form Story Planning | EMNLP-2023 | 长篇故事节奏控制 |
| GROVE: Retrieval-augmented Complex Story Generation | EMNLP-2023 | RAG 增强复杂故事生成 |
| SWAG: Storytelling With Action Guidance | EMNLP-2024 | 动作引导的叙事 (RL/SFT) |

#### More Controllable (可控性)

| 论文 | 会议 | 核心思路 |
|------|------|---------|
| SCORE: Story Coherence and Retrieval Enhancement | ArXiv-2025 | 故事连贯性 + 检索增强 |
| MoPS: Modular Story Premise Synthesis | ACL-2024 | 模块化故事前提合成 |
| RecurrentGPT: Interactive Generation of (Arbitrarily) Long Text | ArXiv-2023 | 交互式任意长文本生成 (64 引用) |
| RLCD: Reinforcement Learning from Contrast Distillation | ICLR-2024 | 对比蒸馏 RL 对齐 |

#### More Personalized (个性化)

| 论文 | 会议 | 核心思路 |
|------|------|---------|
| Pastiche Novel Generation: Fan Fiction in Favorite Author's Style | ArXiv-2025 | 仿写指定作者风格的同人小说 |
| Whose story is it? Personalizing by inferring author styles | ArXiv-2025 | 推断作者风格个性化生成 |
| MirrorStories: Reflecting Diversity through Personalized Narrative | EMNLP-2024 | 个性化叙事反映多样性 |

#### Evaluation (评估)

| 论文 | 会议 | 核心思路 |
|------|------|---------|
| What Makes a Good Story and How Can We Measure It? | ArXiv-2024 | 故事评估综合综述 |
| BooookScore: Book-length Summarization in the Era of LLMs | ICLR-2024 | 书长度摘要评估 (117 引用) |
| Are Large Language Models Capable of Generating Human-Level Narratives? | EMNLP-2024 | LLM 能否达到人类叙事水平 (33 引用) |
| Echoes in AI: Quantifying Lack of Plot Diversity | ArXiv-2025 | 量化 LLM 输出剧情多样性不足 |
| LongEval: Long-Text Generation Analysis | ArXiv-2025 | 基于计划的长文本生成评估 |
| FACTTRACK: Time-Aware World State Tracking | NAACL-2025 | 时间感知世界状态追踪 |
| Art or Artifice? LLMs and the False Promise of Creativity | CHI-2023 | LLM 创造力的虚假承诺 (136 引用) |

#### Foundation Models (基础模型)

| 论文 | 会议 | 核心思路 |
|------|------|---------|
| Weaver: Foundation Models for Creative Writing | ArXiv-2024 | 专注写作能力的基础模型 |

---

## 对比总结

| 项目 | Stars | 形态 | 部署难度 | 核心优势 |
|------|-------|------|---------|---------|
| **InkOS** | 8.8k | npm/Studio/TUI/CLI | 低 | 功能最全，多形态创作 |
| **oh-story** | 5.1k | Agent Skill | 低 | 网文全流程，扫榜+拆文 |
| **AI-Novel-Writing** | 2.3k | 桌面App/Web | 低 | LangGraph 导演式全自动 |
| **AI_NovelGenerator** | 高 | Python GUI | 中 | 向量检索+多模型路由 |
| **MuMuAINovel** | 中 | Docker Web | 低 | 产品化好，多用户 |
| **Show Me The Story** | 438 | 单Go binary | 最低 | 伏笔+核查+全书优化 |
| **awesome-novel-skill** | 509 | Agent Skill | 低 | 8 Agent 协作 |
| **Humanizer** | 33.9k | Agent Skill | 低 | 专注去 AI 味，33 模式 |
| **novel-studio** | 69 | Go 自托管 | 中 | 多智能体世界推演 |

---

## 关键技术方向

1. **长程一致性** — 向量检索 / 叙事记忆 / RAG / 全书脉络注入
2. **伏笔管理** — 埋设→推进→回收状态机 + 超期告警
3. **去 AI 味** — 33+ 模式检测 / 禁用套话 / 口语化改写 / AI 味检测流程
4. **多 Agent 协作** — 角色分工 (设定/大纲/写作/审稿/修订) / 导演-演员架构
5. **上下文管理** — 语义压缩 + 分层记忆 + 角色资源精准筛选
6. **全书优化** — 诊断→核查→工单→修订→diff
7. **多模型路由** — 不同任务用不同模型 (规划/正文/审阅/拆书)
8. **大纲引导生成** — 动态分层大纲 + 记忆增强 / 评估引导迭代计划
9. **个性化/风格** — 作者风格推断 / 仿写 / 声音校准

---

## GitHub Topics 导航

- [novel-generator](https://github.com/topics/novel-generator) — 31 个项目
- [ai-writing](https://github.com/topics/ai-writing) — 869 个项目
- [story-generation](https://github.com/topics/story-generation) — 136 个项目
