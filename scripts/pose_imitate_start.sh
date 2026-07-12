#!/usr/bin/env bash
# 一鍵啟動:pose_imitator + 自動重啟 (若被 ESC 誤關或 crash)
# 預設讀 RPi head_agent 的 MJPEG;加 --local 用筆電 webcam
set -u
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"

cd "$REPO/tools/pose_imitator"

if [[ "${1:-}" == "--local" ]]; then
    c_cyan "==== pose imitator (LOCAL webcam) — auto-restart on exit ===="
    while true; do
        python3 pose_imitator.py --cam 0 --target 127.0.0.1:9999 --show
        c_yellow "  pose_imitator ended, restart in 2s ..."
        sleep 2
    done
else
    if ! curl -sf --max-time 3 "http://$RPI_HOST:8000/status" >/dev/null; then
        c_yellow "  head agent 沒起來,先啟動 ..."
        rpi_ssh 'cd ~/head_rpi_agent && nohup python3 agent.py > ~/agent.log 2>&1 < /dev/null &' || true
        sleep 6
    fi
    c_cyan "==== pose imitator (RPi 相機, headless) — auto-restart ===="
    # No --show — arm GUI has its own inline camera panel now.
    while true; do
        python3 pose_imitator.py \
            --url "http://$RPI_HOST:8000/mjpeg" \
            --target 127.0.0.1:9999
        c_yellow "  pose_imitator ended, restart in 2s ..."
        sleep 2
    done
fi
