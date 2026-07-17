# 三大 Tuning 算法调研：TextGrad / GEPA / SkillOpt

调研日期: 2026-06-19

---

## 1. TextGrad — 文本空间的"自动微分"

### 论文

- **标题**: TextGrad: Automatic "Differentiation" via Text
- **arXiv**: 2406.07496
- **作者**: Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, James Zou (Stanford)
- **核心思想**: 借鉴深度学习的反向传播和自动微分，在文本空间中实现"梯度"传播。LLM 提供的自然语言反馈就是梯度，沿计算图反向传播到各个可优化变量。

### 代码仓库

- **GitHub**: https://github.com/zou-group/textgrad (★3617)
- **语言**: Python
- **依赖**: OpenAI API (或其他 LLM backend)

### 核心实现

#### Variable — 计算图节点（类似 PyTorch Tensor）

```python
class Variable:
    def __init__(self, value, predecessors=None, requires_grad=True, role_description=""):
        self.value = value                    # 文本内容
        self.predecessors = predecessors or [] # 前驱节点
        self.requires_grad = requires_grad     # 是否需要梯度
        self.role_description = role_description  # 角色描述（对优化性能影响巨大）
        self.gradients: Set[Variable] = set()  # 梯度集合（文本反馈）
        self.grad_fn = None                    # 反向函数
```

#### Function / Module — 可微分操作（类似 PyTorch autograd.Function）

```python
class Function(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs) -> Variable: pass
    
    @abstractmethod
    def backward(self, *args, **kwargs): pass

class BackwardContext:
    """封装反向函数和上下文，调用时传入 backward_engine"""
    def __call__(self, backward_engine: EngineLM):
        return self.backward_fn(*self.args, **self.kwargs, backward_engine=backward_engine)
```

#### 反向传播 — 从 Karpathy 的 micrograd 借鉴拓扑排序

```python
def backward(self, engine=None):
    # 1. 拓扑排序所有前驱节点
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for predecessor in v.predecessors:
                build_topo(predecessor)
            topo.append(v)
    build_topo(self)
    
    # 2. 反向遍历，逐节点聚合梯度 + 调用 grad_fn
    self.gradients = set()
    for v in reversed(topo):
        if v.requires_grad:
            v.gradients = _check_and_reduce_gradients(v, backward_engine)
            if v.get_grad_fn() is not None:
                v.grad_fn(backward_engine=backward_engine)
```

#### 梯度聚合 (Reduce) — 多个梯度通过 LLM 总结

- `_reduce_gradients_mean()`: 用 backward engine LLM 对多个梯度做摘要
- Sum 的 backward: 恒等传播，把反馈直接传递给子变量
- Aggregate 的 backward: 用 LLM 做归约

#### 核心洞察

- **"文本梯度"定义**: LLM 生成的自然语言反馈本身就是梯度
- 评估 LLM 检查输出，生成改进建议（"梯度"）
- 这个建议沿计算图反向传播到各个可优化变量
- 本质是用 LLM 替代数值微分，用自然语言替代标量梯度
- `role_description` 对优化性能影响极大——越具体越好

---

## 2. GEPA — 反思式进化 + Pareto 选择

### 论文

- **标题**: GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning
- **arXiv**: 2507.19457
- **作者**: Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems 等 (Berkeley, DSPy 团队)
- **核心思想**: LLM 读取完整的执行轨迹（而不仅是标量奖励），用自然语言反思诊断问题、提出修复，再通过 Pareto 前沿选择最优变体。比 RL 快 35 倍（100-500 次评估 vs GRPO 的 5000-25000+）。

### 代码仓库

- **GitHub**: https://github.com/gepa-ai/gepa (★5221)
- **PyPI**: `pip install gepa`
- **集成**: 也作为 `dspy.GEPA` 集成在 DSPy 框架中
- **依赖**: litellm, tqdm, cloudpickle

### 核心实现

#### GEPAEngine — 优化循环主控

```python
class GEPAEngine(Generic[DataId, DataInst, Trajectory, RolloutOutput]):
    def __init__(self,
        adapter: GEPAAdapter,           # 评估适配器
        reflective_proposer: ReflectiveMutationProposer,  # 反思变异提议器
        merge_proposer: MergeProposer,  # 合并提议器
        frontier_type: FrontierType,    # Pareto 前沿类型
        acceptance_criterion: AcceptanceCriterion,  # 接受策略
        num_parallel_proposals: int,    # 并行提案数
        ...
    ): ...
```

#### ReflectiveMutationProposer — 反思变异（核心算法）

1. 从训练集采样 minibatch (`BatchSampler`)
2. 用目标 LLM 执行任务（rollout），捕获完整轨迹
3. 用 reflection LLM 阅读轨迹，诊断失败原因，提出新的 prompt/代码修改
4. `propose_new_texts()`: 对每个待优化的组件，根据反思数据生成新文本

```python
class ReflectiveMutationProposer(ProposeNewCandidate[DataId]):
    def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        # 如果 adapter 有自定义提议，用 adapter
        if self.adapter.propose_new_texts is not None:
            return self.adapter.propose_new_texts(...)
        # 否则用 reflection_lm 逐组件生成
        for name in components_to_update:
            # 用 InstructionProposalSignature 构造反思 prompt
            # LLM 生成新文本
```

#### Pareto 前沿 (GEPAState) — 4 种前沿类型

```python
FrontierType = Literal["instance", "objective", "hybrid", "cartesian"]

@dataclass
class GEPAState:
    pareto_front_valset: dict[DataId, float]             # instance 前沿分数
    program_at_pareto_front_valset: dict[DataId, set[ProgramIdx]]  # 前沿上的候选
    objective_pareto_front: ObjectiveScores               # objective 前沿分数
    program_at_pareto_front_objectives: dict[str, set[ProgramIdx]]
    pareto_front_cartesian: dict[tuple[DataId, str], float]  # cartesian 前沿
    program_at_pareto_front_cartesian: dict[tuple[DataId, str], set[ProgramIdx]]
```

- **instance**: 每个验证样本一条前沿（谁的分数高谁在前沿上）
- **objective**: 每个目标指标一条前沿
- **hybrid**: 两者兼有
- **cartesian**: 每个样本×每个目标一条前沿

#### Merge Proposer — 互补变体融合

```python
def does_triplet_have_desirable_predictors(program_candidates, ancestor, id1, id2):
    """检查两个变体是否有互补预测器"""
    # 对每个预测器名称，检查是否一个变体与祖先相同、另一个不同
    # 只有存在互补性才有合并价值
```

- 找到两个变体的公共祖先
- 检查它们是否有"互补预测器"（一个在某维度强，另一个在另一维度强）
- 通过 `filter_ancestors()` 过滤不合适的祖先
- 合并产生新候选，尝试从两个父体继承各自的强项

#### EvaluationCache — 避免重复评估

```python
@dataclass
class EvaluationCache(Generic[RolloutOutput, DataId]):
    _cache: dict[CacheKey, CachedEvaluation]  # (candidate_hash, example_id) → 结果
    
    def get_batch(self, candidate, example_ids):
        """批量查询缓存，返回 (cached_results, uncached_ids)"""
```

#### 关键数据结构

- `Candidate = dict[str, str]` — 一个候选就是一组文本参数
- 哈希用 SHA256 (`_candidate_hash`)，用于缓存和去重
- `ObjectiveScores = dict[str, float]` — 多目标分数

### 生产验证

- Databricks: 开源模型 + GEPA 比 Claude Opus 4.1 便宜 90 倍
- Shopify, Dropbox, OpenAI, Pydantic, MLflow 等 50+ 生产使用
- Tobi Lutke (Shopify CEO): "GEPA 在 AI context engineering 世界被严重低估"

---

## 3. SkillOpt — 类深度学习的技能文档优化器

### 论文

- **标题**: SkillOpt: Executive Strategy for Self-Evolving Agent Skills
- **arXiv**: 2605.23904
- **作者**: Yifan Yang, Ziyang Gong, Weiquan Huang 等 (Microsoft Research)
- **核心思想**: 把 agent 的 skill 文档当作冻结模型的"可训练外部状态"，用深度学习的训练纪律（epoch、学习率、验证门控）来优化，部署时零额外推理开销。

### 代码仓库

- **GitHub**: https://github.com/microsoft/SkillOpt (★8301)
- **PyPI**: `pip install skillopt`
- **文档**: https://microsoft.github.io/SkillOpt/
- **依赖**: openai, pyyaml, numpy, openpyxl, azure-identity

### 核心实现 — 6 阶段管线 ReflACT

```
Rollout → Reflect → Aggregate → Select → Update → Evaluate
  ①         ②          ③          ④        ⑤        ⑥
```

#### ① Rollout — 目标模型用当前 skill 执行任务

```python
@dataclass
class RolloutResult:
    id: str
    hard: int           # 精确匹配得分
    soft: float          # 部分得分 (F1 / 部分信用)
    n_turns: int         # 交互轮次
    fail_reason: str     # 失败原因
    task_description: str
    predicted_answer: str
    question: str
    extras: dict         # 环境特定字段
```

#### ② Reflect — 优化器模型分析轨迹，生成 Edit 补丁

```python
EditOp = Literal["append", "insert_after", "replace", "delete"]

@dataclass
class Edit:
    op: EditOp
    content: str = ""
    target: str = ""           # 编辑目标位置
    support_count: int | None  # 支持此编辑的失败案例数
    source_type: Literal["failure", "success"] | None  # 来自失败还是成功
    merge_level: int | None    # 聚合层级
```

#### ③ Aggregate — 层级合并多个 Edit 为 Patch

```python
@dataclass
class Patch:
    edits: list[Edit]           # 编辑列表
    reasoning: str              # 合并理由
    ranking_details: dict       # 排名详情
```

#### ④ Select — 排名并选择 top edits

#### ⑤ Update — 应用 edits 到 skill 文档

- 有界编辑预算（cosine decay 学习率控制每轮最大编辑数）
- 早期 epoch: 大预算（4 edits）→ 晚期 epoch: 小预算（1 edit）

#### ⑥ Evaluate — Validation Gate（核心创新）

```python
GateAction = Literal["accept_new_best", "accept", "reject"]
GateMetric = Literal["hard", "soft", "mixed"]

@dataclass(frozen=True)
class GateResult:
    action: GateAction
    current_skill: str
    current_score: float
    best_skill: str
    best_score: float
    best_step: int

def evaluate_gate(candidate_skill, cand_hard, current_skill, current_score,
                  best_skill, best_score, best_step, global_step, *,
                  cand_soft=0.0, metric="hard", mixed_weight=0.5) -> GateResult:
    """纯决策函数：比较候选分数与当前/最佳，返回接受/拒绝"""
    cand_score = select_gate_score(cand_hard, cand_soft, metric, mixed_weight)
    if cand_score > current_score:
        if cand_score > best_score:
            return GateResult(action="accept_new_best", ...)
        return GateResult(action="accept", ...)
    return GateResult(action="reject", ...)
```

**三种比较指标**:
- `hard`: 精确匹配准确率
- `soft`: 部分分 (F1 / partial credit)
- `mixed`: 加权平均 `(1-w)*hard + w*soft`

#### Epoch 级 Slow Update (EMA/正则化)

```python
@dataclass
class SlowUpdateResult:
    """Epoch 级别的慢更新（EMA / 正则化）"""
    reasoning: str
    slow_update_content: str
    action: str
    prev_hard: float | None     # 上一步 hard 分数
    curr_hard: float | None     # 当前 hard 分数
    selection_hard: float | None  # 选择集 hard 分数
    selection_soft: float | None  # 选择集 soft 分数
    candidate_hash: str
```

#### 失败模式分析

```python
@dataclass
class FailureSummaryEntry:
    failure_type: str     # 失败类型
    count: int            # 出现次数
    description: str      # 描述
```

### 关键设计决策

1. **skill 文档是纯文本** `best_skill.md`（300-2000 tokens），部署时零额外模型调用
2. **优化器模型与目标模型分离**: 优化器分析轨迹、生成编辑；目标模型只执行
3. **验证门控是纯函数**: 不含副作用（缓存查找、rollout、状态变更都由 trainer 管理）
4. **部署产物**: 一个紧凑的 `best_skill.md`，直接挂载到未修改的目标模型

### 实验结果

- 6 个 benchmark, 7 个目标模型, 3 个执行环境 (direct chat, Codex CLI, Claude Code CLI)
- **所有 52 个评估单元 (model, benchmark, harness) 均为最佳或并列最佳**
- GPT-5.5: 平均无技能准确率提升 +23.5 (direct chat), +24.8 (Codex), +19.1 (Claude Code)
- 优化后的 skill 可跨模型规模、跨执行环境、跨相似 benchmark 迁移

---

## 横向对比

| 维度 | TextGrad | GEPA | SkillOpt |
|---|---|---|---|
| **论文** | 2406.07496 | 2507.19457 | 2605.23904 |
| **机构** | Stanford | Berkeley (DSPy) | Microsoft Research |
| **优化对象** | 计算图中的任意文本变量 | Prompt / 代码 / 配置 | Skill 文档（结构化自然语言） |
| **"梯度"定义** | LLM 生成的自然语言反馈 | LLM 反思诊断 + 变异 | LLM 反思 + 有界编辑(add/del/replace) |
| **选择机制** | 梯度传播+变量更新 | Pareto 前沿（4种类型） | 验证门控(严格提升才接受) |
| **并行度** | 单路径 | 多变体并行 + 合并提议 | 单路径 |
| **学习率控制** | 无 | 无 | Cosine decay（早多晚少） |
| **过拟合防护** | 无 | Pareto 前沿自然防过拟合 | 验证集门控 + rejected buffer + 回滚 |
| **训练信号** | 评估反馈→文本梯度 | 完整轨迹→反思→变异 | 失败模式→有界编辑→验证 |
| **部署开销** | 无额外开销 | 无额外开销 | 零推理时模型调用 |
| **灵感来源** | PyTorch autograd | 进化算法 + 多目标优化 | 深度学习训练流程 |
| **GitHub Stars** | 3617 | 5221 | 8301 |

---

## Coach 项目的工程适配

从 Coach 项目的 `tuning-algorithm-internals.md` 和 `tuning-algorithms.md` 可看出，Coach 对三个算法做了大量工程适配：

### TextGrad → 5 步信号管线

从通用计算图退化为确定性管线：
1. **信号提取**: 从 FP/FN 错误案例提取 attribution/keyword/pattern 三类信号
2. **信号聚合**: Counter 分组排序，FP/FN 分池
3. **RCA 增强**: Worker 提供的 RCA 可让信号计数 ×1.5
4. **方向映射**: 信号→梯度方向的确定性映射表（如 FP→add_exclusion, FN→add_trigger）
5. **摘要生成**: 人类可读的梯度摘要

与原始 TextGrad 的关键差异：梯度不再是 LLM 自由生成，而是从错误案例确定性提取。

### GEPA → VariantScope 隔离架构

- 引入 `VariantScope` 隔离（deep copy + contextvars + asyncio.gather）
- 原始 GEPA 用线程池 + 评估缓存
- Coach 的反思是确定性的（无 LLM 调用），变体方向有三个优先级来源：Reflection 建议 > Pareto 互补性 > Fallback 方向
- 变体数量自适应：首轮3个, F1<70%→3个, F1>85%→2个, 连续无改进+1

### SkillOpt → Strategy-Worker 分离

- Worker 是薄 LLM 调用（零工具调用），只负责 reflect + 生成 edits JSON
- Strategy 直接调用 SkillOptTool 控制编辑应用和回滚
- 原始 SkillOpt 的 6 阶段管线被简化为 9 步循环
- SkillOptTool 是必需依赖，不可移除（结构化 YAML 操作无法用通用 prompt 工具替代）
