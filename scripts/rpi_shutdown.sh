#!/usr/bin/env bash
# 遠端關機:先停 agents,再 SSH 送 poweroff 到樹莓派
set -e
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"

c_cyan "==== 遠端關機 $RPI_HOST ===="

if ! rpi_ping; then
    c_red "  ✗ SSH 不通,無法遠端關機。若 RPi 還亮著請直接拔電。"
    exit 1
fi

c_cyan "停 agent (head + arm) ..."
rpi_ssh 'pkill -f "agent.py" 2>/dev/null; sleep 1' || true

c_cyan "送 poweroff ..."
# sudo -n 不會提示密碼,失敗就 return 非 0 → 提示 user 手動處理
if rpi_ssh 'sudo -n poweroff 2>/dev/null'; then
    c_green "  ✓ 關機指令已送出。約 15 秒後 RPi 會斷。"
else
    c_yellow "  ⚠ sudo 需要密碼。改試 systemd-shutdown fallback ..."
    rpi_ssh 'sudo poweroff' || {
        c_red "  ✗ 遠端關機失敗。可能需要 RPi 上 visudo 設定 NOPASSWD:/sbin/poweroff"
        exit 1
    }
    c_green "  ✓ 關機指令已送出"
fi
