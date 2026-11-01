"""H200 自部署成本估算 v2: 与 OpenRouter API 最低价对齐同一请求分布 (15k 输入 / 100 输出)。

v2 修正 (2026-11-01, 依据 vLLM/SGLang 实测报告校准):
- BWU 70% -> 40%: Qwen 官方 SGLang/H20 实测反推 batch=1 BWU≈32%,
  NVIDIA AIConfigurator vLLM/8xH200 Qwen3-32B-FP8 反推 batch≈8 BWU≈37~45%
- prefill MFU 35% -> 30%: NVIDIA 实测 TTFT 453ms@4k vs 本模型 35% 时 378ms
- decode 吞吐改为并发感知: batch 由 KV cache 显存预算反推 (原来按 batch=1 算,
  高估 decode 成本 5~10 倍)
- decode 每 step 字节量 = 激活权重 + Σ各序列 KV 全量 (长上下文下 KV 读取占主导,
  v1 完全没算)

v3 (2026-11-01): 扩展到 deepseek/z-ai/minimax/moonshotai/tencent; MLA 压缩 KV
(deepseek/kimi-k2) 单列配置; 卡数在 [n_fit, n_fit+3] 内按每请求成本寻优
(prefill GPU·秒与卡数无关, 寻优只影响 decode, 修复单卡恰好装下但 batch=1 的病态点)。

硬件 (NVIDIA H200 SXM): HBM3e 141GB, 带宽 4.8TB/s, FP8 ~1979 TFLOPS。
成本口径: 稳态 serving, 仍为理论下界 (不含空闲/运维/网络)。
"""

import json
import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from model_sizes import parse_size  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[2] / "data" / "pricing"
TODAY = date.today().isoformat()

# 与 fetch_openrouter.py 一致的厂商范围
VENDORS = ("qwen/", "deepseek/", "z-ai/", "minimax/", "moonshotai/", "tencent/")

CONFIG = {
    "gpu_price_per_hour": 3.0,   # H200 按需租用市场价 (USD)
    "hbm_gb": 141,
    "bandwidth_bps": 4.8e12,
    "fp8_flops": 1979e12,
    "bw_util": 0.40,             # 带宽利用率, 实测校准 (batch>=8 区间 0.35~0.45)
    "mfu": 0.30,                 # decode 算力利用率 (KV 读取瓶颈, NVIDIA 实测校准)
    "mfu_prefill": 0.49,         # prefill 算力利用率 (纯 GEMM, 无 KV 读取; 实测 Qwen3.8-27B QPS=3 反推校准)
    "weight_bytes": 1.0,         # FP8
    "weight_overhead": 1.05,
    "mem_reserve": 0.20,         # 系统/激活/碎片预留
    "batch_cap": 256,            # 并发硬上限
    "gpus_per_node": 8,          # 单机卡数, 卡数寻优向上取整到整机
    "tp_efficiency": 0.85,       # 多卡 TP 通信效率 (AllReduce 开销, 8 卡内经验值)
    "in_tokens": 15_000,
    "out_tokens": 100,
    "prefix_hit_rate": 0.60,     # KV 前缀缓存命中率: 命中部分跳过 prefill 计算 (PagedAttention/prefix caching)
    # DFlash 块扩散推测解码 (Z Lab, ICML 2026): 用有效接受长度而非草稿长度
    # 配置 num_speculative_tokens=7~8 (草稿长度), 但实际接受长度 τ≈3~5 (论文表: greedy 均值 4.0, sampling 均值 3.5)
    # 取保守有效值 4。加速比 = τ ≈ 4× (非草稿长度 8×)
    "dflash_tau": 4.0,
}

# KV cache 配置: slug 片段 -> (层数, KV头数, head_dim); 按 key 长度最长优先匹配
# 未标 [估] 的为官方 config 值, [估] 为按同代架构推测
KV_CONFIG: dict[str, tuple[int, int, int]] = {
    # ---- 新家族 (GQA) ----
    "glm-4.5-air": (46, 8, 128),        # [估] 106B
    "glm-4.5v": (46, 8, 128),           # [估] 基于 4.5-Air
    "glm-4.5": (92, 8, 128),            # [估] 355B
    "glm-4.6": (92, 8, 128),            # [估] 同 4.5
    "glm-4.7": (92, 8, 128),            # [估] 同 4.5/4.6 架构
    "glm-5.3-flash": (80, 8, 128),      # [估] 320B/18B 混合架构
    "glm-5": (78, 8, 128),              # [估] 744B/40B, GLM-5.2 确认 78 层, DSA 稀疏注意力
    "minimax-01": (10, 8, 128),         # [估] 80 层混合线性注意力, 仅 1/8 层全注意力
    "minimax-m1": (10, 8, 128),         # [估] 同上
    "minimax-m2": (62, 8, 128),         # [估] 论文: M2 62 层, 256 专家
    "minimax-m3": (80, 8, 128),         # [估] 456B/23B
    "hunyuan-a13b": (48, 8, 128),       # [估]
    # ---- qwen ----
    "qwen3-coder": (62, 8, 128),
    "qwen3-next-80b-a3b": (12, 2, 256),   # [估] 混合线性注意力, 仅 12 层全注意力
    "qwen3-235b": (94, 4, 128),
    "qwen3-vl-235b": (94, 4, 128),
    "qwen3-30b-a3b": (48, 4, 128),
    "qwen3-vl-30b": (48, 4, 128),
    "qwen3-coder-30b": (48, 4, 128),
    "qwen3-coder-flash": (48, 4, 128),    # [估] 同 30B-A3B
    "qwen3-32b": (64, 8, 128),
    "qwen-2.5-coder-32b": (64, 8, 128),
    "qwen3-14b": (40, 8, 128),
    "qwen3.5-9b": (36, 8, 128),           # [估]
    "qwen3-vl-8b": (36, 8, 128),
    "qwen3.5-27b": (48, 8, 128),          # [估]
    "qwen3.6-27b": (48, 8, 128),          # [估]
    "qwen3.8-27b": (48, 8, 128),          # [估]
    "qwen3.5-35b-a3b": (48, 4, 128),      # [估] 同 30B-A3B 系
    "qwen3.6-35b-a3b": (48, 4, 128),      # [估]
    "qwen3.5-122b-a10b": (64, 4, 128),    # [估]
    "qwen3.5-397b-a17b": (80, 4, 128),    # [估]
    "qwen3.8-2.4t-a95b": (96, 4, 128),    # [估]
    "qwen2.5-vl-72b": (80, 8, 128),
    "qwen-2.5-72b": (80, 8, 128),
    "qwen-2.5-7b": (28, 4, 128),
}
DEFAULT_KV = (48, 4, 128)

# MLA 压缩 KV 配置: slug 片段 -> (层数, 压缩维 = kv_lora_rank + qk_rope_dim)
# MLA 每 token 只存压缩潜变量 (无独立 V), KV 占用比 GQA 小一个数量级
# deepseek-v3/v4 与 kimi-k2 均为 kv_lora_rank 512 + qk_rope 64 = 576 维, 61 层
MLA_CONFIG: dict[str, tuple[int, int]] = {
    "deepseek": (61, 576),   # v3 系官方; v4 系 [估] 同架构
    "kimi-k2": (61, 576),    # K2 官方 config
    "kimi-k3": (96, 576),    # [估] K3 2.8T, MLA 架构, 层数推测
}


def kv_config(model_id: str) -> tuple:
    """返回 ("gqa", 层数, KV头数, head_dim) 或 ("mla", 层数, 压缩维)。"""
    for key in sorted(MLA_CONFIG, key=len, reverse=True):
        if key in model_id:
            layers, cdim = MLA_CONFIG[key]
            return ("mla", layers, cdim)
    for key in sorted(KV_CONFIG, key=len, reverse=True):
        if key in model_id:
            layers, kv_heads, head_dim = KV_CONFIG[key]
            return ("gqa", layers, kv_heads, head_dim)
    return ("gqa", *DEFAULT_KV)


def estimate_at(total_b: float, active_b: float, kv_bytes_token: float, n_gpu: int, cfg: dict) -> dict:
    """固定卡数下的成本估算。"""
    usable_gb = cfg["hbm_gb"] * (1 - cfg["mem_reserve"])
    free_gb = n_gpu * usable_gb - total_b * cfg["weight_bytes"] * cfg["weight_overhead"]

    ctx_kv_gb = cfg["in_tokens"] * kv_bytes_token / 1e9  # 单序列 15k 上下文 KV
    b_mem = max(1, int(free_gb / ctx_kv_gb)) if ctx_kv_gb > 0 else cfg["batch_cap"]
    batch = min(b_mem, cfg["batch_cap"])

    bw_gbs = cfg["bandwidth_bps"] * cfg["bw_util"] / 1e9 * n_gpu   # GB/s
    tp_eff = cfg["tp_efficiency"] if n_gpu > 1 else 1.0
    flops = cfg["fp8_flops"] * cfg["mfu"] * n_gpu * tp_eff
    flops_prefill = cfg["fp8_flops"] * cfg["mfu_prefill"] * n_gpu * tp_eff

    # decode: 每 step 读 激活权重(集群合计) + 每序列 KV 全量
    w_act_gb = active_b * cfg["weight_bytes"]
    step_bytes = (w_act_gb + batch * ctx_kv_gb) * 1e9
    step_flops = 2 * active_b * 1e9 * batch
    step_s = max(step_bytes / (bw_gbs * 1e9), step_flops / flops)
    agg_tps = batch / step_s

    # prefill: 前缀缓存命中部分跳过计算, 仅算 (1-hit) 比例的新 token
    eff_in = cfg["in_tokens"] * (1 - cfg["prefix_hit_rate"])
    prefill_s = 2 * active_b * 1e9 * eff_in / flops_prefill
    # GPU-秒口径: 连续批处理 prefill 与 decode 流水线重叠, 取瓶颈阶段 max(非串行 sum)
    prefill_gpu_s = prefill_s * n_gpu
    decode_gpu_s = cfg["out_tokens"] * n_gpu / agg_tps
    gpu_s = max(prefill_gpu_s, decode_gpu_s)
    cost_per_req = gpu_s * cfg["gpu_price_per_hour"] / 3600
    out_per_m = n_gpu * cfg["gpu_price_per_hour"] / 3600 / agg_tps * 1e6

    # DFlash 口径: 每 step 提交 tau×batch 个 token, 字节量不变, 计算量 ×tau;
    # 加 5% 草稿模型开销 (DFlash draft 模型约 0.5B, 前向计算量约 target 的 1~5%)
    tau = cfg["dflash_tau"]
    draft_overhead = 1.05
    step_s_d = max(step_bytes / (bw_gbs * 1e9), tau * step_flops * draft_overhead / flops)
    dflash_tps = batch * tau / step_s_d
    decode_gpu_s_d = cfg["out_tokens"] * n_gpu / dflash_tps
    gpu_s_d = max(prefill_gpu_s, decode_gpu_s_d)
    cost_dflash = gpu_s_d * cfg["gpu_price_per_hour"] / 3600
    return {
        "n_gpu": n_gpu,
        "batch": batch,
        "prefill_s": prefill_s,
        "agg_tps": agg_tps,
        "dflash_tps": dflash_tps,
        "cost_per_req": cost_per_req,
        "cost_dflash": cost_dflash,
        "out_per_m": out_per_m,
    }


def estimate(total_b: float, active_b: float, kv: tuple, cfg: dict) -> dict:
    if kv[0] == "mla":
        kv_bytes_token = kv[1] * kv[2] * 2  # 压缩潜变量, bf16
    else:
        _, layers, kv_heads, head_dim = kv
        kv_bytes_token = 2 * layers * kv_heads * head_dim * 2  # K+V, bf16 KV cache

    weight_gb = total_b * cfg["weight_bytes"] * cfg["weight_overhead"]
    usable_gb = cfg["hbm_gb"] * (1 - cfg["mem_reserve"])
    n_fit = max(1, math.ceil(weight_gb / usable_gb))

    # 卡数寻优: 实际部署受限于整机规格 (gpus_per_node=8), 卡数向上取整到 8 的倍数
    # 单卡可独立部署的小模型保留 1 卡选项
    gpn = cfg["gpus_per_node"]
    def round_up_to_node(n: int) -> int:
        if n <= 1:
            return 1
        return math.ceil(n / gpn) * gpn

    # 候选卡数: n_fit 向上取整 + 后续整机档
    candidates = [round_up_to_node(n_fit)]
    while candidates[-1] < n_fit + gpn * 2:
        candidates.append(candidates[-1] + gpn)

    # 仅当收益 >5% 才采纳更多卡 (避免 1% 收益换 8 卡)
    best = estimate_at(total_b, active_b, kv_bytes_token, candidates[0], cfg)
    for n in candidates[1:]:
        cand = estimate_at(total_b, active_b, kv_bytes_token, n, cfg)
        if cand["cost_per_req"] < best["cost_per_req"] * 0.95:
            best = cand
    return {
        "n_gpu": best["n_gpu"],
        "batch": best["batch"],
        "ttft_s": round(best["prefill_s"], 2),
        "agg_tps": round(best["agg_tps"]),
        "dflash_tps": round(best["dflash_tps"]),
        "selfhost_cost_per_req": round(best["cost_per_req"] * 1000, 4),  # 毫美元, 自回归
        "dflash_cost_per_req": round(best["cost_dflash"] * 1000, 4),     # 毫美元, DFlash
        "selfhost_out_per_m": round(best["out_per_m"], 4),
    }


def min_provider_price(model_id: str, endpoints: dict) -> tuple[float, float] | None:
    """该模型全部 provider 中的最低输入/输出单价 ($/token)。"""
    eps = endpoints.get(model_id, [])
    pins = [float(e["prompt_price"]) for e in eps if e.get("prompt_price")]
    pouts = [float(e["completion_price"]) for e in eps if e.get("completion_price")]
    return (min(pins), min(pouts)) if pins and pouts else None


def main() -> None:
    models = json.loads((BASE_DIR / "raw" / f"models_{TODAY}.json").read_text())
    ep_path = BASE_DIR / "raw" / f"endpoints_{TODAY}.json"
    if not ep_path.exists():  # 兼容旧文件
        ep_path = BASE_DIR / "raw" / f"qwen_endpoints_{TODAY}.json"
    endpoints = json.loads(ep_path.read_text())
    rows = []
    for m in models:
        if not m["id"].startswith(VENDORS):
            continue
        total_b, active_b = parse_size(m["id"], m.get("name", ""))
        if total_b is None:
            continue  # 闭源/参数未公开模型无法估算
        # API 口径: 最低 provider 价 (自部署对比的严格口径), 无端点数据回落列表价
        best = min_provider_price(m["id"], endpoints)
        if best:
            p_in, p_out = best
        else:
            p_in = float(m["pricing"]["prompt"])
            p_out = float(m["pricing"]["completion"])
        api_cost = CONFIG["in_tokens"] * p_in + CONFIG["out_tokens"] * p_out
        est = estimate(total_b, active_b, kv_config(m["id"]), CONFIG)
        rows.append(
            {
                "model": m["id"],
                "total_b": total_b,
                "active_b": active_b,
                **est,
                "api_cost_per_req": round(api_cost * 1000, 4),
                "api_markup_x": round(api_cost / (est["selfhost_cost_per_req"] / 1000), 2),
                "api_markup_dflash_x": round(api_cost / (est["dflash_cost_per_req"] / 1000), 2),
            }
        )
    df = pd.DataFrame(rows).sort_values(["model"])
    out_csv = BASE_DIR / "processed" / f"selfhost_h200_{TODAY}.csv"
    out_md = BASE_DIR / "processed" / f"selfhost_h200_{TODAY}.md"
    df.to_csv(out_csv, index=False)
    out_md.write_text(df.to_markdown(index=False, floatfmt=".3f"))
    print(df.to_markdown(index=False, floatfmt=".3f"))
    print(f"\nsaved: {out_csv}\nsaved: {out_md}")


if __name__ == "__main__":
    main()
