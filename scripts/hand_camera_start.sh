#!/usr/bin/env bash
# 一鍵啟動:靈巧手 — 相機影像模式(推薦)
set -e
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"

c_cyan "==== 靈巧手 (camera 影像模式) ===="
# 一次只留一個相機視窗 / MediaPipe 消耗者 / 馬達 bus 佔用者
pkill -f "camera_receiver.py" 2>/dev/null || true
cd "$REPO/hand"
exec python3 camera_receiver.py --show "$@"
