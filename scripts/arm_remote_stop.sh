#!/usr/bin/env bash
# 停 RPi 上的 arm_rpi_agent(不動 head agent)
set -e
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"

c_cyan "==== stopping arm agent on $RPI_HOST ===="
rpi_ssh 'pgrep -af arm_rpi_agent/agent.py'
rpi_ssh 'pkill -f "arm_rpi_agent/agent.py" && echo killed || echo none' || true
