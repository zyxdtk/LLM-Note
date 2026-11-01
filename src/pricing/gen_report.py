"""从 processed/ 下的 CSV 生成 docs/2.compute/LLM-Pricing.md (六厂商版)。

依赖: analyze_pricing.py 与 selfhost_h200.py 先运行产出当日 CSV。
表 1 (API 报价) 与表 2 (自部署对比) 均按厂商分节; 结论为脚本内模板。
"""

from datetime import date
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
TODAY = date.today().isoformat()

VENDOR_NAMES = {
    "qwen": "Qwen (阿里)",
    "deepseek": "DeepSeek",
    "z-ai": "GLM (智谱)",
    "minimax": "MiniMax",
    "moonshotai": "Kimi (月之暗面)",
    "tencent": "混元 (腾讯)",
    "anthropic": "Claude (Anthropic)",
    "google": "Gemini (Google)",
    "openai": "GPT (OpenAI)",
}

# 闭源厂商在表 1 中只展示主流模型 (排除 :free / :batch / audio / lyria / gpt-oss 等变体)
CLOSED_MAIN_FILTER = {
    "anthropic": lambda m: ":free" not in m and ":batch" not in m and "audio" not in m,
    "google": lambda m: ":free" not in m and ":batch" not in m and "audio" not in m and "lyria" not in m,
    "openai": lambda m: ":free" not in m and ":batch" not in m and "audio" not in m and "gpt-oss" not in m and "gpt-4-mini" not in m and "gpt-4o-mini" not in m,
}


def mod_short(m: str) -> str:
    if "audio" in m:
        return "omni"
    if "image" in m or "video" in m:
        return "VLM"
    return "LLM"


def f(v, d=2, dash="-"):
    if isinstance(v, str):
        return v
    if pd.isna(v):
        return dash
    return f"{v:.{d}f}".rstrip("0").rstrip(".")


def api_table(p: pd.DataFrame, vendor: str) -> str:
    rows = p[p.model.str.startswith(vendor + "/")].sort_values("cost_per_req_min", ascending=False)
    filt = CLOSED_MAIN_FILTER.get(vendor)
    if filt:
        rows = rows[rows.model.map(filt)]
    lines = []
    for _, r in rows.iterrows():
        size = "-" if (pd.isna(r.total_b) or r.total_b == "—") else f"{f(r.total_b)}/{f(r.active_b)}"
        lines.append(
            "| {model} | {mod} | {ctx} | {size} | {li}/{lo} | {mi}/{mo} | {cm} | {cl} | {np} | {lat} | {tps} |".format(
                model=r.model.removeprefix(vendor + "/"), mod=mod_short(str(r.modality)),
                ctx=int(r.ctx_k), size=size,
                li=f(r.in_per_m, 3), lo=f(r.out_per_m, 3),
                mi=f(r.min_prompt_per_m, 3), mo=f(r.min_completion_per_m, 3),
                cm=f(r.cost_per_req_min, 3), cl=f(r.cost_per_req_listed, 2),
                np=int(r.n_providers), lat=f(r.best_p50_latency_ms, 0),
                tps=f(r.best_p50_throughput_tps, 0),
            )
        )
    return "\n".join(lines)


def selfhost_table(s: pd.DataFrame, vendor: str) -> str:
    rows = s[s.model.str.startswith(vendor + "/")].sort_values("total_b")
    lines = []
    for _, r in rows.iterrows():
        lines.append(
            "| {m} | {tb}/{ab} | {ng} | {b} | {tt} | {tps} | {dtps} | {sc} | {dc} | {ac} | {mk} | {dmk} |".format(
                m=r.model.removeprefix(vendor + "/"), tb=f(r.total_b), ab=f(r.active_b),
                ng=int(r.n_gpu), b=int(r.batch), tt=r.ttft_s, tps=int(r.agg_tps),
                dtps=int(r.dflash_tps), sc=f(r.selfhost_cost_per_req, 3),
                dc=f(r.dflash_cost_per_req, 3), ac=f(r.api_cost_per_req, 3),
                mk=r.api_markup_x, dmk=r.api_markup_dflash_x,
            )
        )
    return "\n".join(lines)


def main() -> None:
    p = pd.read_csv(BASE_DIR / f"data/pricing/processed/pricing_{TODAY}.csv")
    s = pd.read_csv(BASE_DIR / f"data/pricing/processed/selfhost_h200_{TODAY}.csv")

    t1_header = "| 模型 | 类型 | ctx(k) | 参数B | 列表价in/out | 最低价in/out | $/千req(最低) | $/千req(列表) | prov | p50ms | tps |\n|:--|:--|--:|:--|:--|:--|--:|--:|--:|--:|--:|"
    t2_header = "| 模型 | 参数B | 卡数 | 并发 | TTFT(s) | AR tps | DFlash tps | 自建$/千req(AR) | 自建(DFlash) | API$/千req | 加价(AR) | 加价(DFlash) |\n|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"

    sec1, sec2 = [], []
    for v, name in VENDOR_NAMES.items():
        rows = p[p.model.str.startswith(v + "/")]
        filt = CLOSED_MAIN_FILTER.get(v)
        if filt:
            rows = rows[rows.model.map(filt)]
        n1 = len(rows)
        n2 = len(s[s.model.str.startswith(v + "/")])
        sec1.append(f"### {name}（{n1} 个）\n\n{t1_header}\n{api_table(p, v)}")
        if n2:
            sec2.append(f"### {name}\n\n{t2_header}\n{selfhost_table(s, v)}")

    doc = f"""# LLM 成本分析（OpenRouter 报价 vs H200 自部署）

> 数据快照：{TODAY} | 数据源：OpenRouter | 覆盖：Qwen / DeepSeek / GLM / MiniMax / Kimi / 混元（开源，含自部署估算） + Claude / Gemini / GPT（闭源，仅 API 报价）
> 目的：为「不同模型成本与收益核算」提供报价基线

## 方法

**标准化请求**：输入 15k tokens / 输出 100 tokens（长输入短输出，贴近客服 Agent、RAG 摘要等场景）。千请求成本 = `15 × 输入$/M + 0.1 × 输出$/M`。

**价格口径（重要）**：OpenRouter 同一模型有多个 provider，价差可达一个数量级。

- **列表价**：`/api/v1/models` 返回的官方口径，接近默认路由价（往往偏贵）
- **最低价**：全部 provider 中的最低单价，可控路由（指定 provider / 比价调度）能达到的成本下限。本报告主口径
- 注意：最低价 provider 多为 fp8/fp4 量化社区端点，稳定性与速度需实测权衡

延迟/吞吐取各 provider 近 30 分钟 p50 统计中的最优值（样本请求数 > 50）。

**H200 自部署模型 v3**（已经 vLLM/SGLang 实测报告校准）：

- 时价 $3.0/h；FP8 推理，权重冗余 5%；显存预留 20% 给 KV cache 与系统
- 解码并发感知：batch 由 KV cache 显存预算反推；每 step 字节量 = 激活权重 + Σ各序列 KV 全量（15k 长上下文下 KV 读取占主导）；带宽利用率取 40%（Qwen 官方 SGLang 实测反推 32%，NVIDIA vLLM/H200 实测反推 37~45%）
- prefill 按 MFU 30%（NVIDIA AIConfigurator 实测 TTFT 校准）
- **MLA 压缩 KV**（deepseek 全系、kimi-k2 系）：每 token 只存 576 维压缩潜变量（61 层，bf16），15k 上下文单序列 KV 仅 ~1.05GB，比 GQA 同档小一个数量级
- **卡数寻优**：prefill GPU·秒与卡数无关（理想 scaling），在 [恰好装下, +3] 区间按每请求成本寻优，仅当收益 >5% 采纳更多卡（修复单卡恰好装下但 KV 预算不足 → batch=1 的病态点）
- **DFlash 口径**：块扩散推测解码（Z Lab，已进 SGLang Spec V2），每 step 提交 τ=4 个 token（保守有效值；实测单请求 4~6×）。字节量不变、计算量 ×τ，转为算力瓶颈时加速比自动封顶。DFlash 目前仅有 Qwen 官方 checkpoint，其他家族列为理论参考
- 稳态成本摊薄，**理论下界**口径，不含空闲/运维/网络

参数列为「总参/激活」(B)，`-` = 闭源或未公开。Claude / Gemini / GPT 闭源模型只做 API 报价，不做自部署估算；闭源厂商表中已剔除 :free / audio / lyria / gpt-oss 等变体，仅保留主流对话模型。

## 表 1：OpenRouter API 报价对比（{len(p)} 个模型，闭源已过滤变体，按厂商分节，节内按最低价口径千请求成本降序）

价格单位 $/M tokens，成本单位 $/千请求。

<!-- 数据源: data/pricing/processed/pricing_{TODAY}.csv (src/pricing/analyze_pricing.py 生成) -->

{chr(10) + chr(10) + (chr(10) + chr(10)).join(sec1)}

## 表 2：H200 自部署 vs API 最低价（仅参数已公开的开源权重模型，{len(s)} 个）

`API加价倍数` = API 每请求成本 ÷ 自部署理论成本。**>1 表示自建更便宜，<1 表示 API 更便宜。** `并发`为 KV 显存预算反推的解码并发数。AR = 自回归解码，DFlash = 块扩散推测解码（τ=4）。

<!-- 数据源: data/pricing/processed/selfhost_h200_{TODAY}.csv (src/pricing/selfhost_h200.py 生成) -->

{chr(10) + chr(10) + (chr(10) + chr(10)).join(sec2)}

## 关键结论

> 校准说明：基于实测 Qwen3.8-27B 单卡 QPS=3 / E2E=0.8s 反推，模型 v7 校准项：① **KV 前缀缓存命中率 0.6**（prefill 计算量打 4 折）；② **prefill/decode 分离 MFU**（prefill 0.49，decode 0.30）；③ **prefill/decode 流水线重叠**（gpu_s=max 非 sum）；④ **DFlash 有效 τ=4**（接受长度，非草稿长度 8，端到端加速比≈3×）；⑤ **多卡 TP 效率 0.85**（AllReduce 开销）；⑥ **卡数向上取整到 8 卡整机**。自建成本为理论下界（不含空闲/运维/网络）。

1. **自建甜区 0.3~88 倍**（85 个开源模型）。GLM-5.3-Prime 88 倍、Qwen3-VL-30B-A3B-Thinking 67 倍、GLM-4.5V 63 倍、Qwen3-Next-80B-A3B 47 倍（自建仅 $0.031/千req）、MiniMax-M2.1 38 倍。前缀缓存把 prefill 计算量打 4 折，是自建性价比翻倍的主因。GLM-5.2 加价 0.31 倍（API 引流价 $0.15/千req 低于自建）。
2. **DeepSeek 不再是反例**：V4-Pro-0813 自建 $0.60 vs API $1.64，加价 2.75 倍；V4-Flash 自建 $0.19 vs API $0.58，3.1 倍。全家族自建均更划算。
3. **唯一 API 更便宜的是竞争充分的 dense 模型**：Qwen3.8-27B（22 个 provider）自建 $0.28 vs API $0.31，加价 1.10 倍——价格战已把 API 压到自建理论下界附近，中小流量可直接调 API。其余 dense 模型自建 1~3 倍空间，需权衡运维成本。
4. **MLA 的 KV 红利**：DeepSeek-V3.2（671B）单序列 15k KV 仅 ~1.05GB（GQA 同档 3~6GB），8 卡 H200 支撑大 batch；Kimi-K2（1T）16 卡。长上下文场景 MLA 自建可行性远好于 GQA。
5. **KV 预算决定部署形态**：卡数寻优按 8 卡整机取整，「装得下」不等于「跑得划算」——单卡恰好装下但 KV 预算只够 batch=1 时，加卡到 8 卡整机可大幅摊薄成本。
6. **闭源梯队价格战离谱**：DeepSeek-V4.1-Flash 最低输入 $0.0038/M（$0.072/千req，全表最低）、GLM-5.2 列表输入 $0.006/M 配输出 $6.8/M（长输入场景输入白送）。多为引流/不对称定价，按中位 provider 价重算结论会反转。
7. **DFlash 在本场景（15k 入/100 出）与 AR 成本相同**：prefill 占 88%+ GPU 时间是绝对瓶颈，DFlash 加速的 decode 被 prefill 掩盖。DFlash 价值在**长输出场景**（长 CoT、代码生成），此时 decode 成瓶颈，有效 τ=4 可降成本约 4×。端到端加速比≈3×（官方数据），因为 TTFT 不加速。目前仅 Qwen 有官方 checkpoint。

## 表 3：闭源模型 API 报价概览（Claude / Gemini / GPT，96 个主流模型）

闭源模型无公开权重，无法自部署，只能调 API。价格跨度极大：最便宜 GPT-5-Nano $0.395/千req，最贵 GPT-4 $456/千req（相差 1150 倍）。

**关键对比**：
- 闭源最便宜（GPT-5-Nano $0.395）比开源最便宜（DeepSeek-V4.1-Flash $0.072）贵 **5.5 倍**
- 闭源旗舰（Claude-Opus-4.1 $232.5、GPT-5.5-Pro $234）比同档开源旗舰（GLM-5.3-Prime API $43、自建 $0.49）贵 **5~480 倍**
- 同一能力档闭源 vs 开源：GPT-5-Nano（$0.395）≈ Qwen3.8-27B API（$0.31）但比其自建（$0.28）贵 40%
- **batch 变体**：闭源厂商普遍提供 5 折 batch 价（异步队列），高延迟容忍场景可直接减半成本

## 局限与下一步

- 自部署为**理论下界**：未计空闲冗余、运维人力、电力网络、折旧；低峰期并发上不去时成本显著劣化。实际成本按量级 ×1.5~3 估计。
- 前缀缓存命中率 0.6 为通用假设：实际取决于 prompt 结构（系统 prompt 越长命中率越高），可按业务 0.3~0.8 调整。
- 多卡 TP 效率 0.85 为 8 卡内经验值：更大规模集群（16+ 卡）通信开销更高，效率可能降到 0.7~0.8。
- 校准后仍有 ±2x 精度：GLM/MiniMax/混元 KV 配置为按架构推测的 [估] 值；MiniMax-01/M1 混合线性注意力仅按 1/8 层全注意力计 KV；未计 MoE 专家并行通信开销。
- 最低价 provider 的延迟/可用性通常弱于头部 provider，表 1 的 p50/tps 是全 provider 最优值，需按业务 SLA 取舍。
- 下一步：接入质量维度（benchmark 分数）做性价比帕累托；长输出场景（代码生成、长 CoT）重跑对比，DFlash 权重会显著上升。
"""

    out = BASE_DIR / "docs" / "2.compute" / "LLM-Pricing.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
