#!/usr/bin/env bash
# 一鍵啟動:motor_tool (Python 版 FD.exe — register 編輯 + 波形圖)
set -e
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"

c_cyan "==== motor_tool (Feetech register editor) ===="
if ls /dev/ttyACM* >/dev/null 2>&1; then
    c_green "  ✓ 偵測到 $(ls /dev/ttyACM* | tr '\n' ' ')"
else
    c_yellow "  ⚠ 沒有 /dev/ttyACM* (Scan 之後仍需要接上馬達才會有東西)"
fi

exec python3 "$REPO/arm/motor_tool.py"
