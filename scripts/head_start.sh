#!/usr/bin/env bash
# 一鍵啟動:樹莓派上的 head_rpi_agent + PC 上的 remote_gui.py
# Head end-to-end 全部帶起(agent → 相機 → 馬達 → PC 視覺化)

set -e
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"

echo
c_cyan "==== Q-BOT head one-click start ===="

# 1) 確認 RPi 連得到
c_cyan "[1/3] 檢查 RPi ($RPI_HOST) 連線 ..."
if ! rpi_ping; then
    c_red "  ✗ SSH 到 $RPI_HOST 失敗。檢查 wifi / 電源 / IP。"
    exit 1
fi
c_green "  ✓ SSH ok"

# 2) 確認 agent 是否已跑,不在就起
c_cyan "[2/3] 檢查 head agent ..."
if rpi_ssh 'pgrep -f "agent.py" >/dev/null'; then
    c_green "  ✓ agent 已在 RPi 上運作 (PID $(rpi_ssh 'pgrep -f agent.py' | head -1))"
else
    c_yellow "  agent 未在跑,啟動中 ..."
    rpi_ssh 'cd ~/head_rpi_agent && (nohup python3 agent.py > ~/agent.log 2>&1 &) && sleep 1'
    sleep 2
    if rpi_ssh 'pgrep -f "agent.py" >/dev/null'; then
        c_green "  ✓ agent 已啟動 (PID $(rpi_ssh 'pgrep -f agent.py' | head -1))"
    else
        c_red   "  ✗ 啟動失敗,查看 log:"
        rpi_ssh 'tail -20 ~/agent.log'
        exit 1
    fi
fi

# 3) 快速自檢:HTTP 狀態 + 馬達 IDs
c_cyan "[3/3] Health check ..."
STAT=$(curl -sk --max-time 3 "http://$RPI_HOST:8000/status" || echo "{}")
echo "  $STAT"
echo

c_green "── endpoints ──"
echo "  PC (HTTP  8000):  http://$RPI_HOST:8000/"
echo "  Phone (HTTPS 8443):  https://$RPI_HOST:8443/vr"
echo "  Camera stream:       http://$RPI_HOST:8000/mjpeg"
echo

# 4) 啟動 PC-side GUI
c_cyan "─── 啟動 PC 端 remote_gui.py ───"
cd "$REPO/head/pc_client"
exec python3 remote_gui.py --host "$RPI_HOST:8000"
