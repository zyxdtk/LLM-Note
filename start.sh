#!/usr/bin/env bash
# LLM-Note 一键启动脚本
# 用法: ./start.sh [端口]     默认端口 8600
#       ./start.sh build      # 构建静态站点到 site/（GitHub Pages 用）
set -euo pipefail
cd "$(dirname "$0")"

# ── 0. 构建模式（静态站点，GitHub Pages 用）──────────────────────────
if [ "${1:-}" = "build" ]; then
    shift
    exec uv run mkdocs build "$@"
fi

PORT="${1:-8600}"
URL="http://localhost:${PORT}/"

# ── 1. 确保 uv 可用（非交互 shell 可能缺 ~/.local/bin）──────────────
if ! command -v uv >/dev/null 2>&1; then
    export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "错误: 未找到 uv，请先安装: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# ── 2. 同步依赖（首次运行会安装 mkdocs 及分析依赖到 .venv）───────────
uv sync --quiet

# ── 3. 端口已被占用则直接复用，不重复启动 ────────────────────────────
if curl -s -o /dev/null --max-time 2 "$URL"; then
    echo "LLM-Note 已在运行: ${URL}"
    open "$URL" 2>/dev/null || true
    exit 0
fi

# ── 4. 启动（前台运行，Ctrl+C 停止；改动 docs/ 自动热更新）──────────
echo "启动 LLM-Note: ${URL}"
sleep 1 && open "$URL" 2>/dev/null || true
exec uv run mkdocs serve --dev-addr "0.0.0.0:${PORT}"
