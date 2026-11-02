"""模型参数量表 (总参数 / 激活参数, 单位 B)。覆盖 qwen/deepseek/z-ai/minimax/moonshotai/tencent。

来源优先级:
1. 官方命名惯例: slug/名称中的 "35b-a3b" => 总 35B, 激活 3B (MoE);
   "27b" => dense, 总=激活=27B; "2.4t" => 2400B。
2. 闭源商业模型 (max/plus/flash/omni/coder-flash 等) 未公开参数, 记 None。

已公开确切参数的特例在下面 SPECIAL 中登记。
新家族来源: OpenRouter 模型描述 (deepseek-v4-pro 1.6T/49B, hy3 295B/21B 等)
与官方公布值 (deepseek-v3 671B/37B, kimi-k2 1T/32B, glm-4.5 355B/32B)。
"""

import re

# 官方已公布确切参数的模型 (slug 片段 -> (总参数B, 激活参数B))
# 匹配规则: 按 key 长度从长到短匹配, 避免 "qwen3-coder" 抢在 "qwen3-coder-30b" 前面
SPECIAL: dict[str, tuple[float, float]] = {
    "qwen3-coder-30b-a3b-instruct": (30.5, 3.3),
    "qwen3-coder-next": (None, None),  # type: ignore[assignment] 参数未公开
    "qwen3-coder-flash": (30.5, 3.3),  # 官方定位 30B-A3B 档
    "qwen/qwen3-coder": (480.0, 35.0),  # Qwen3 Coder 480B A35B
    "qwen3-30b-a3b-instruct-2507": (30.5, 3.3),
    "qwen3-30b-a3b": (30.5, 3.3),
    "qwen3-235b-a22b-thinking-2507": (235.0, 22.0),
    "qwen3-235b-a22b-2507": (235.0, 22.0),
    "qwen3-next-80b-a3b-instruct": (80.0, 3.0),
    "qwen3-vl-235b-a22b-instruct": (235.0, 22.0),
    "qwen3-vl-30b-a3b-thinking": (30.5, 3.3),
    "qwen3-vl-30b-a3b-instruct": (30.5, 3.3),
    "qwen3-vl-8b-instruct": (8.0, 8.0),
    "qwen3-32b": (32.8, 32.8),
    "qwen3-14b": (14.8, 14.8),
    "qwen2.5-vl-72b-instruct": (72.0, 72.0),
    "qwen-2.5-coder-32b-instruct": (32.5, 32.5),
    "qwen-2.5-7b-instruct": (7.6, 7.6),
    "qwen-2.5-72b-instruct": (72.7, 72.7),
    # ---- deepseek ----
    "deepseek-chat-v3-0324": (685.0, 37.0),  # V3-0324 官方 685B
    "deepseek-chat": (671.0, 37.0),          # DeepSeek-V3
    "deepseek-v3.1": (671.0, 37.0),
    "deepseek-v3.2": (671.0, 37.0),          # V3.2-Exp, DSA 稀疏注意力
    "deepseek-r1": (671.0, 37.0),
    "deepseek-v4-pro": (1600.0, 49.0),       # 官方描述 1.6T/49B
    "deepseek-v4-flash": (284.0, 13.0),      # 官方描述 284B/13B
    # deepseek-v4.1-flash: 总参未公开 (激活非对称 8B/16B), None
    # ---- z-ai (GLM) ----
    "glm-4.5v": (106.0, 12.0),               # 基于 4.5-Air
    "glm-4.5-air": (106.0, 12.0),
    "glm-4.5": (355.0, 32.0),
    "glm-4.6v": (355.0, 32.0),               # 基于 4.6
    "glm-4.6": (355.0, 32.0),
    "glm-4.7-flash": (355.0, 32.0),
    "glm-4.7": (355.0, 32.0),                # 官方: 355B/32B, 同 4.5/4.6 架构
    "glm-5v-turbo": (744.0, 40.0),           # [估] 同 GLM-5 base
    "glm-5-turbo": (744.0, 40.0),
    "glm-5.1": (744.0, 40.0),                # [估] 同 GLM-5 base (744B/40B, 78层)
    "glm-5.2": (744.0, 40.0),                # NVIDIA NGC 确认 744B/40B, 78层, 256专家
    "glm-5.3-prime": (744.0, 40.0),
    "glm-5.3-flashx": (320.0, 18.0),
    "glm-5.3-flash": (320.0, 18.0),          # 官方: 320B/18B, 混合稀疏+线性注意力
    "glm-5.3": (744.0, 40.0),                # [估] 同 GLM-5.2 base
    "glm-5": (744.0, 40.0),                  # 官方确认 745B/44B (openlm 报 744B/40B), 取 744/40
    # ---- minimax ----
    "minimax-01": (456.0, 45.9),             # MiniMax-Text-01
    "minimax-m1": (456.0, 45.9),
    "minimax-m2": (230.0, 10.0),             # 论文: 229.9B/9.8B
    "minimax-m2.1": (230.0, 10.0),           # [估] M2 系列同架构
    "minimax-m2.5": (230.0, 10.0),           # [估] M2 系列同架构
    "minimax-m2.7": (230.0, 10.0),           # [估] M2 系列同架构 (论文确认 M2 系列)
    "minimax-m2-her": (230.0, 10.0),         # [估] M2 系列
    "minimax-m3.1-flash": (456.0, 23.0),     # [估] M3 系列, madebyagents 报 23B active
    "minimax-m3": (456.0, 23.0),             # [估] 456B/23B
    # ---- moonshotai (Kimi) ----
    "kimi-k2": (1000.0, 32.0),               # 1T/32B, MLA
    "kimi-k2.5": (1000.0, 32.0),             # [估] K2 架构继续预训练
    "kimi-k2.6": (1000.0, 32.0),             # [估] 同 K2 架构
    "kimi-k2.7-code": (1000.0, 32.0),        # madebyagents 确认 1000B/32B
    "kimi-k3.1-pro": (3000.0, 104.0),        # [估] 3T 级, 激活同 K3
    "kimi-k3": (2800.0, 104.0),              # 官方确认 2.8T/104B
    # ---- tencent (Hunyuan) ----
    "hunyuan-a13b": (80.0, 13.0),
    "hy4-preview": (770.0, 49.0),            # 官方描述
    "hy3-preview": (295.0, 21.0),            # [估] 同 hy3
    "hy3": (295.0, 21.0),                    # 官方描述 295B/21B, 192 专家 top-8
    # ---- deepseek v4.1 ----
    "deepseek-v4.1-flash": (284.0, 13.0),    # [估] 同 V4-Flash 架构
    # hy-mt2 系列按命名正则自动解析 (1.8b/7b dense, 30b-a3b MoE)
}

_MOE_RE = re.compile(r"(\d+\.?\d*)t-a(\d+\.?\d*)b|(\d+\.?\d*)b-a(\d+\.?\d*)b")
_DENSE_RE = re.compile(r"(?:^|[-\s])(\d+\.?\d*)b(?:[-\s]|$)")


def parse_size(slug: str, name: str = "") -> tuple[float | None, float | None]:
    """从 slug/名称解析 (总参数B, 激活参数B); 无法确定返回 (None, None)。"""
    for key in sorted(SPECIAL, key=len, reverse=True):
        if key in slug:
            return SPECIAL[key]
    text = f"{slug} {name}".lower()
    m = _MOE_RE.search(text)
    if m:
        if m.group(1):  # 2.4t-a95b
            return float(m.group(1)) * 1000, float(m.group(2))
        return float(m.group(3)), float(m.group(4))
    m = _DENSE_RE.search(text)
    if m:
        size = float(m.group(1))
        return size, size
    return None, None
