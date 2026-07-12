#!/usr/bin/env bash
# 一鍵啟動:靈巧手 — 手套(JSON UDP 接收)模式
set -e
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"

c_cyan "==== 靈巧手 (手套 / JSON UDP 模式) ===="
pkill -f "glove_receiver.py" 2>/dev/null || true
cd "$REPO/hand"
exec python3 glove_receiver.py "$@"
