#!/usr/bin/env bash
# 一鍵啟動:樹莓派上的 arm_rpi_agent + PC 上的 qbot_ik_gui (Remote 模式)
# Arm end-to-end: RPi 拿手臂 USB → PC 端 MuJoCo GUI 走 WebSocket 控制

set -e
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"

ARM_PORT=8100

echo
c_cyan "==== Q-BOT arm remote one-click start ===="

# 1) SSH 通不通
c_cyan "[1/3] 檢查 RPi ($RPI_HOST) 連線 ..."
if ! rpi_ping; then
    c_red "  ✗ SSH 到 $RPI_HOST 失敗。檢查 wifi / 電源 / IP。"
    exit 1
fi
c_green "  ✓ SSH ok"

# 2) 部署最新 agent(每次都同步,避免 PC 端改過但 RPi 沒更新)
c_cyan "[2/3] 同步 agent 到 RPi ..."
rpi_ssh 'mkdir -p ~/arm_rpi_agent' >/dev/null
scp $RPI_SSH_OPTS \
    "$REPO/arm/rpi_agent/agent.py" \
    "$REPO/arm/rpi_agent/feetech_lib.py" \
    "$RPI_USER@$RPI_HOST:~/arm_rpi_agent/" >/dev/null
c_green "  ✓ 同步完成"

# 3) 起 agent — 手臂跟頭部共用同一條 Feetech bus (/dev/ttyACM0),
#    只能一個 agent 拿住 port,所以連 head agent 一起殺,交給 arm agent 掃全部 IDs
c_cyan "[3/3] 啟動 arm agent (port $ARM_PORT) ..."
rpi_ssh 'pkill -f "head_rpi_agent/agent.py" 2>/dev/null; pkill -f "arm_rpi_agent/agent.py" 2>/dev/null; sleep 1' || true
rpi_ssh "cd ~/arm_rpi_agent && (nohup python3 agent.py --port $ARM_PORT > ~/arm_agent.log 2>&1 &) && sleep 1"
sleep 2
if rpi_ssh "pgrep -f 'arm_rpi_agent/agent.py' >/dev/null"; then
    c_green "  ✓ arm agent PID $(rpi_ssh 'pgrep -f arm_rpi_agent/agent.py' | head -1)"
else
    c_red "  ✗ agent 沒起來 — log:"
    rpi_ssh 'tail -20 ~/arm_agent.log'
    exit 1
fi

# 4) health check
STAT=$(curl -sk --max-time 3 "http://$RPI_HOST:$ARM_PORT/status" || echo "{}")
echo "  $STAT"
echo

c_green "── endpoints ──"
echo "  HTTP status:   http://$RPI_HOST:$ARM_PORT/status"
echo "  WebSocket:     ws://$RPI_HOST:$ARM_PORT/ws"
echo

# 5) PC 端 GUI — 先寫入 host 到環境變數,GUI 讀進來當 default
c_cyan "─── 啟動 PC 端 qbot_ik_gui.py (Remote 模式) ───"
cd "$REPO/arm"
QBOT_ARM_REMOTE_HOST="$RPI_HOST:$ARM_PORT" \
QBOT_ARM_REMOTE_DEFAULT=1 \
    exec python3 qbot_ik_gui.py
