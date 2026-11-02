"""抓取 OpenRouter 上目标厂商模型的报价与各 provider 端点的延迟/吞吐数据。

数据源:
1. 公开 API: https://openrouter.ai/api/v1/models  (全部模型列表/定价/上下文)
2. 模型详情页 HTML 内嵌的 RSC 数据 (各 provider 端点的 p50 延迟、吞吐、请求数)

输出 (data/pricing/raw/):
- models_YYYY-MM-DD.json     全量模型列表原始快照
- endpoints_YYYY-MM-DD.json  每个目标厂商模型的 provider 端点明细

已抓过的 slug 会从当日 endpoints 文件 (及历史 qwen_endpoints 文件) 复用, 不重复请求。
"""

import json
import re
import time
from datetime import date
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "pricing" / "raw"
MODELS_API = "https://openrouter.ai/api/v1/models"
PAGE_URL = "https://openrouter.ai/{slug}"
HEADERS = {"User-Agent": "Mozilla/5.0 (pricing-analysis)"}
TODAY = date.today().isoformat()

# 目标厂商 slug 前缀 (GLM 在 OpenRouter 上的前缀是 z-ai)
VENDORS = ("qwen/", "deepseek/", "z-ai/", "minimax/", "moonshotai/", "tencent/",
           "anthropic/", "google/", "openai/")


def fetch_models() -> list[dict]:
    resp = requests.get(MODELS_API, timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]
    (RAW_DIR / f"models_{TODAY}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2)
    )
    return data


def parse_endpoint_page(html: str) -> list[dict]:
    """从模型详情页 HTML 中提取各 provider 端点的价格与延迟/吞吐统计。

    页面是 Next.js RSC payload, JSON 被转义成 \\"key\\":value 形式嵌在 HTML 里,
    先反转义再逐块正则提取。
    """
    text = html.replace('\\"', '"')
    endpoints = []
    # 每个端点块以 "name":"<Provider> | <slug>-<variant>" 开始
    blocks = re.split(r'(?="name":"[^"|]+ \| )', text)
    for block in blocks:
        m = re.match(r'"name":"(?P<provider>[^"|]+) \| (?P<variant>[^"]+)"', block)
        if not m:
            continue
        # 截取到下一个端点块之前, 避免越界匹配
        seg = block[:20000]

        def grab(pattern: str, cast=str):
            r = re.search(pattern, seg)
            return cast(r.group(1)) if r else None

        endpoints.append(
            {
                "provider": m.group("provider").strip(),
                "variant": m.group("variant"),
                "context_length": grab(r'"context_length":(\d+)', int),
                "quantization": grab(r'"quantization":"([^"]+)"'),
                "prompt_price": grab(r'"pricing":\{"prompt":"([\d.eE+-]+)"'),
                "completion_price": grab(
                    r'"pricing":\{"prompt":"[\d.eE+-]+","completion":"([\d.eE+-]+)"'
                ),
                "p50_latency_ms": grab(
                    r'"p50_latency":([\d.]+|null)',
                    lambda v: None if v == "null" else float(v),
                ),
                "p90_latency_ms": grab(
                    r'"p90_latency":([\d.]+|null)',
                    lambda v: None if v == "null" else float(v),
                ),
                "p50_throughput_tps": grab(
                    r'"p50_throughput":([\d.]+|null)',
                    lambda v: None if v == "null" else float(v),
                ),
                "request_count": grab(r'"request_count":(\d+)', int),
            }
        )
    # 同一 provider 会在页面多个 RSC 片段中重复出现, 且带 stats 的块往往在后面,
    # 因此去重时按字段合并 (优先保留非 None 值)
    merged: dict[tuple[str, str], dict] = {}
    for ep in endpoints:
        key = (ep["provider"], ep["variant"])
        if key not in merged:
            merged[key] = ep
        else:
            base = merged[key]
            for k, v in ep.items():
                if base.get(k) is None and v is not None:
                    base[k] = v
    return list(merged.values())


def load_cache() -> dict[str, list[dict]]:
    """已抓取的端点数据: 当日 endpoints 文件 + 历史 qwen_endpoints 文件。"""
    cache: dict[str, list[dict]] = {}
    legacy = RAW_DIR / f"qwen_endpoints_{TODAY}.json"
    if legacy.exists():
        cache.update(json.loads(legacy.read_text()))
    current = RAW_DIR / f"endpoints_{TODAY}.json"
    if current.exists():
        cache.update(json.loads(current.read_text()))
    return cache


def fetch_endpoints(slugs: list[str]) -> dict[str, list[dict]]:
    result = load_cache()
    todo = [s for s in slugs if s not in result]
    print(f"cached: {len(slugs) - len(todo)}, to fetch: {len(todo)}")
    for i, slug in enumerate(todo, 1):
        try:
            resp = requests.get(PAGE_URL.format(slug=slug), headers=HEADERS, timeout=60)
            resp.raise_for_status()
            vendor = slug.split("/")[0] + "/"
            eps = [e for e in parse_endpoint_page(resp.text) if e["variant"].startswith(vendor)]
            result[slug] = eps
            n_stats = sum(1 for e in eps if e["p50_latency_ms"] is not None)
            print(
                f"[{i}/{len(todo)}] {slug}: "
                f"{len(eps)} endpoints, {n_stats} with latency stats"
            )
        except Exception as e:  # noqa: BLE001 - 单模型失败不阻断整体抓取
            print(f"[{i}/{len(todo)}] {slug}: FAILED {e}")
            result[slug] = []
        time.sleep(0.5)
    result = {s: result.get(s, []) for s in slugs}
    (RAW_DIR / f"endpoints_{TODAY}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    return result


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    models = fetch_models()
    print(f"total models: {len(models)}")
    slugs = sorted(m["id"] for m in models if m["id"].startswith(VENDORS))
    # 闭源厂商模型众多, :batch 变体价格为主模型 0.5x, 从列表价可推, 不单独抓取 endpoints
    slugs = [s for s in slugs if ":batch" not in s]
    print(f"vendor models (excl batch): {len(slugs)} ({', '.join(v.rstrip('/') for v in VENDORS)})")
    fetch_endpoints(slugs)


if __name__ == "__main__":
    main()
