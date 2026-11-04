"""抓取 Azure 零售价 API 的 GPU 虚拟机租价(免登录公开接口)。

数据源: Azure Retail Prices API (https://prices.azure.com/api/retail/prices, 无需认证)。
超大规模云里唯一免登录、可脚本化的官方价源; AWS 免签 Pricing API 已失效, GCP 需 key。

输出 (data/pricing/):
- raw/azure_retail_YYYY-MM-DD.json           去重后的原始条目快照
- processed/gpu_rental_azure_YYYY-MM-DD.csv   sku × 计费档汇总
- processed/gpu_rental_azure_YYYY-MM-DD.md    单 sku 一行的汇总表

口径与坑详见 docs/2.compute/GPU-Price.md 的「数据源与采集方法」节:
- OData filter 不支持 not(contains(...)), 排除逻辑必须在客户端做
- 价格 > 10000 的 meter 是 reserved/许可类噪声(unitOfMeasure 同为 1 Hour), 直接剔除
- Spot / Low Priority 是独立 meterName(不是独立 priceType), 与 on-demand 分档
- NC* 为单卡整机; ND96* 为 8 卡整机需 ÷8; 未知 sku(如 ND128 GB200 v6)按整机输出不拆
"""

import csv
import json
import re
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote

import requests

BASE = Path(__file__).resolve().parents[2] / "data" / "pricing"
API = "https://prices.azure.com/api/retail/prices"
TODAY = date.today().isoformat()

# 覆盖 GPU-Price.md 主表的 Azure 在售系列; 查不到的(如 L40S/H20, Azure 无此命名)输出为空属正常
KEYWORDS = ("H100", "H200", "B200", "GB200", "MI300X", "MI355X", "A100", "L40S", "H20")
NOISE_PRICE = 10_000.0  # 数十万刀级 meter 是 reserved/许可口径, 非时租价


def fetch_keyword(keyword: str) -> list[dict]:
    """拉取一个关键词的全部分页条目(跟随 NextPageLink)。"""
    odata = f"serviceName eq 'Virtual Machines' and contains(productName,'{keyword}')"
    url = f"{API}?currencyCode=USD&%24filter={quote(odata, safe='(),$=')}"
    items: list[dict] = []
    while url:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items.extend(data.get("Items", []))
        url = data.get("NextPageLink")
    return items


def clean(item: dict) -> dict | None:
    """客户端过滤 + 计费档分类, 无效条目返回 None。"""
    if item.get("unitOfMeasure") != "1 Hour":
        return None
    if "Windows" in item.get("productName", ""):
        return None
    if "Dedicated Host" in item.get("productName", ""):
        return None  # 整机专用主机租约, 非按卡时租口径
    if item.get("retailPrice", 0) > NOISE_PRICE:
        return None
    name = item.get("meterName", "")
    if "Spot" in name:
        kind = "spot"
    elif "Low Priority" in name:
        kind = "low_priority"
    elif name == item.get("skuName"):
        kind = "on_demand"
    else:
        return None  # Windows/许可/reserved 等噪声 meter
    return {
        "sku": item["skuName"],
        "product": item["productName"],
        "region": item["armRegionName"],
        "kind": kind,
        "price": item["retailPrice"],
        "meter_id": item["meterId"],
    }


def gpu_count(sku: str) -> int | None:
    """整机 sku → 卡数; 未知的返回 None(不折算, 按整机口径输出)。"""
    s = sku.upper()
    if s.startswith("STANDARD_"):
        s = s[len("STANDARD_"):]
    if s.startswith("ND96"):
        return 8  # 8 卡整机
    if s.startswith("ND"):
        return None  # 如 ND128 GB200 v6: 多卡封装, 卡数以 Azure 官方文档为准
    if s.startswith("NC"):
        if "A100" in s:  # NCads A100 系: 24 vCPU/卡 (NC24=1, NC48=2, NC96=4)
            m = re.match(r"NC(\d+)", s)
            return max(1, int(m.group(1)) // 24) if m else 1
        return 1  # NC* Hopper/Blackwell 系为单卡整机
    return None


def main() -> None:
    raw_dir = BASE / "raw"
    proc_dir = BASE / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    seen: dict[str, dict] = {}  # meter_id -> 条目, 跨关键词去重(B200 会命中 GB200 产品)
    for kw in KEYWORDS:
        items = fetch_keyword(kw)
        kept = 0
        for it in items:
            c = clean(it)
            if c and c["meter_id"] not in seen:
                seen[c["meter_id"]] = c
                kept += 1
        print(f"{kw}: {len(items)} 条原始, 新增有效 {kept}")
        time.sleep(0.3)

    records = sorted(seen.values(), key=lambda r: (r["sku"], r["kind"], r["region"]))
    (raw_dir / f"azure_retail_{TODAY}.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"有效条目合计: {len(records)} -> raw/azure_retail_{TODAY}.json")

    # 聚合: 每 sku × 每档取全局最低价
    agg: dict[str, dict] = {}
    for r in records:
        a = agg.setdefault(r["sku"], {"product": r["product"], "regions": set(), "kinds": {}})
        a["regions"].add(r["region"])
        k = a["kinds"].setdefault(r["kind"], {"price": float("inf"), "region": ""})
        if r["price"] < k["price"]:
            k.update(price=r["price"], region=r["region"])

    # CSV: sku × 档 一行
    with open(proc_dir / f"gpu_rental_azure_{TODAY}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sku", "product", "gpu_count", "kind", "min_price_vm",
                    "min_price_gpu", "region_of_min", "n_regions"])
        for sku in sorted(agg):
            a = agg[sku]
            n = gpu_count(sku)
            for kind in ("on_demand", "spot", "low_priority"):
                if kind in a["kinds"]:
                    p = a["kinds"][kind]["price"]
                    w.writerow([sku, a["product"], n or "", kind, f"{p:.2f}",
                                f"{p / n:.2f}" if n else "", a["kinds"][kind]["region"],
                                len(a["regions"])])

    # MD: 单 sku 一行
    lines = [
        f"# GPU 租价 · Azure 零售 API 快照({TODAY})",
        "",
        "来源: prices.azure.com/api/retail/prices(免登录, USD/时, Linux)。"
        "整机时租价; 单卡 = 整机 ÷ 卡数。Spot = 竞价, LP = 低优先级。"
        "噪声 meter(>1 万刀)已剔; 未知卡数 sku 按整机输出。",
        "",
        "| sku | 产品 | 卡数 | on-demand整机 | on-demand单卡 | spot单卡 | LP单卡 | 区域数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sku in sorted(agg):
        a = agg[sku]
        n = gpu_count(sku)

        def cell(kind: str, per_gpu: bool = True) -> str:
            if kind not in a["kinds"]:
                return "-"
            p = a["kinds"][kind]["price"]
            if per_gpu and n:
                return f"{p / n:.2f}"
            return f"{p:.2f}"

        lines.append(
            f"| {sku} | {a['product']} | {n or '?'} | {cell('on_demand', per_gpu=False)} "
            f"| {cell('on_demand')} | {cell('spot')} | {cell('low_priority')} "
            f"| {len(a['regions'])} |"
        )
    (proc_dir / f"gpu_rental_azure_{TODAY}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"\n== on-demand 单卡价(有卡数映射的 sku)==")
    for sku in sorted(agg):
        n = gpu_count(sku)
        if n and "on_demand" in agg[sku]["kinds"]:
            p = agg[sku]["kinds"]["on_demand"]["price"]
            print(f"  {sku}: ${p:.2f}/整机 -> ${p / n:.2f}/卡 ({agg[sku]['kinds']['on_demand']['region']})")
    print(f"\n输出: processed/gpu_rental_azure_{TODAY}.csv / .md")


if __name__ == "__main__":
    main()
