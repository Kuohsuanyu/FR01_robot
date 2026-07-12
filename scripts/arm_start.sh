#!/usr/bin/env bash
# 一鍵啟動:手臂鬼影 + IK GUI (本機直連 /dev/ttyACM*)
set -e
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"

c_cyan "==== Q-BOT arm IK GUI ===="

# 檢查串口
if ls /dev/ttyACM* >/dev/null 2>&1; then
    c_green "  ✓ 偵測到 $(ls /dev/ttyACM* | tr '\n' ' ')"
else
    c_yellow "  ⚠ 沒有 /dev/ttyACM*(馬達 USB 沒插?GUI 仍會開,鬼影可用)"
fi

exec python3 "$REPO/arm/qbot_ik_gui.py"
