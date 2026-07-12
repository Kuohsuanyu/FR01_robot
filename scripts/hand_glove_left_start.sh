#!/usr/bin/env bash
# One-click: 靈巧手 LEFT hand — 手套 UDP 模式
set -e
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"

c_cyan "==== 靈巧手 LEFT (手套 / JSON UDP 模式) ===="
pkill -f "glove_receiver.py" 2>/dev/null || true
cd "$REPO/hand"
exec python3 glove_receiver.py --side left "$@"
