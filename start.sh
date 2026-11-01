#!/usr/bin/env bash
# LLM-Note 本地文档服务
#   ./start.sh           # http://127.0.0.1:8000 预览（改动自动重载）
#   ./start.sh 9000      # 指定端口
#   ./start.sh build     # 构建到 site/（GitHub Pages 用）
set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "build" ]; then
    shift
    exec uvx mkdocs build "$@"
fi

PORT="${1:-8000}"
HOST="${HOST:-127.0.0.1}"

command -v uvx >/dev/null 2>&1 && exec uvx mkdocs serve -a "$HOST:$PORT"
command -v mkdocs >/dev/null 2>&1 && exec mkdocs serve -a "$HOST:$PORT"
echo "错误: 需要 mkdocs（安装 uv 后自动可用，或 pip install mkdocs）" >&2
exit 1
