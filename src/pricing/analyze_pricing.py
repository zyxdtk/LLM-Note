"""汇总目标厂商模型报价, 生成标准化成本对比表。

标准化请求: 输入 15k tokens, 输出 100 tokens
  成本 = 15 * 输入单价($/M) + 0.1 * 输出单价($/M)  (单位: 千分之一美元)

输出 (data/pricing/processed/):
- pricing_YYYY-MM-DD.csv
- pricing_YYYY-MM-DD.md   (markdown 表格, 供报告引用)
"""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from model_sizes import parse_size  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[2] / "data" / "pricing"
TODAY = date.today().isoformat()
IN_TOKENS, OUT_TOKENS = 15_000, 100

# 与 fetch_openrouter.py 一致的厂商范围
VENDORS = ("qwen/", "deepseek/", "z-ai/", "minimax/", "moonshotai/", "tencent/",
           "anthropic/", "google/", "openai/")


def load_raw() -> tuple[list[dict], dict[str, list[dict]]]:
    models = json.loads((BASE_DIR / "raw" / f"models_{TODAY}.json").read_text())
    ep_path = BASE_DIR / "raw" / f"endpoints_{TODAY}.json"
    if not ep_path.exists():  # 兼容旧文件
        ep_path = BASE_DIR / "raw" / f"qwen_endpoints_{TODAY}.json"
    endpoints = json.loads(ep_path.read_text())
    return models, endpoints


def endpoint_summary(eps: list[dict]) -> dict:
    """从 provider 端点明细中提炼: 有统计样本下的最优延迟 / 最高吞吐 / 最低/中位价。"""
    import statistics

    with_stats = [e for e in eps if e.get("p50_latency_ms") and (e.get("request_count") or 0) > 50]
    pins = [float(e["prompt_price"]) * 1e6 for e in eps if e.get("prompt_price")]
    pouts = [float(e["completion_price"]) * 1e6 for e in eps if e.get("completion_price")]
    return {
        "n_providers": len(eps),
        "best_p50_latency_ms": min(
            (e["p50_latency_ms"] for e in with_stats), default=None
        ),
        "best_p50_throughput_tps": max(
            (e["p50_throughput_tps"] for e in with_stats if e.get("p50_throughput_tps")),
            default=None,
        ),
        "min_prompt_per_m": min(pins) if pins else None,
        "min_completion_per_m": min(pouts) if pouts else None,
        "median_prompt_per_m": statistics.median(pins) if pins else None,
        "median_completion_per_m": statistics.median(pouts) if pouts else None,
    }


def main() -> None:
    models, endpoints = load_raw()
    rows = []
    for m in models:
        if not m["id"].startswith(VENDORS):
            continue
        pricing = m["pricing"]
        p_in = float(pricing["prompt"]) * 1e6  # $/M tokens, OpenRouter 列表价
        p_out = float(pricing["completion"]) * 1e6
        total_b, active_b = parse_size(m["id"], m.get("name", ""))
        ep = endpoint_summary(endpoints.get(m["id"], []))
        # 主口径: 最低 provider 价 (可控路由能达到的成本下限); 无端点数据时回落列表价
        eff_in = ep["min_prompt_per_m"] if ep["min_prompt_per_m"] else p_in
        eff_out = ep["min_completion_per_m"] if ep["min_completion_per_m"] else p_out
        cost_min = (IN_TOKENS * eff_in + OUT_TOKENS * eff_out) / 1e6 * 1000
        cost_listed = (IN_TOKENS * p_in + OUT_TOKENS * p_out) / 1e6 * 1000
        # 闭源模型无公开参数, total/active 显示为 —
        tb = total_b if total_b is not None else "—"
        ab = active_b if active_b is not None else "—"
        rows.append(
            {
                "model": m["id"],
                "modality": m.get("architecture", {}).get("modality", ""),
                "ctx_k": (m.get("context_length") or 0) // 1000,
                "total_b": tb,
                "active_b": ab,
                "in_per_m": round(p_in, 4),
                "out_per_m": round(p_out, 4),
                "cost_per_req_min": round(cost_min, 4),  # 毫美元/请求, 最低价口径
                "cost_per_req_listed": round(cost_listed, 4),  # 毫美元/请求, 列表价口径
                **ep,
            }
        )
    df = pd.DataFrame(rows).sort_values("cost_per_req_min", ascending=False)

    out_csv = BASE_DIR / "processed" / f"pricing_{TODAY}.csv"
    out_md = BASE_DIR / "processed" / f"pricing_{TODAY}.md"
    df.to_csv(out_csv, index=False)
    out_md.write_text(df.to_markdown(index=False, floatfmt=".3f"))
    print(df.to_markdown(index=False, floatfmt=".3f"))
    print(f"\nsaved: {out_csv}\nsaved: {out_md}")


if __name__ == "__main__":
    main()
