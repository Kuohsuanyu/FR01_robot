#!/usr/bin/env bash
# 停 RPi 上的 head_rpi_agent(釋放 port 8000/8443 + 相機 + 馬達 bus)
set -e
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"

c_cyan "==== stopping head agent on $RPI_HOST ===="
if ! rpi_ping; then c_red "  ✗ RPi 連不到"; exit 1; fi

BEFORE=$(rpi_ssh 'pgrep -f agent.py | wc -l')
rpi_ssh 'pkill -f agent.py 2>/dev/null; sleep 1; pgrep -f agent.py >/dev/null && pkill -9 -f agent.py; sleep 0.3'
AFTER=$(rpi_ssh 'pgrep -f agent.py | wc -l')

echo "  agent processes: $BEFORE → $AFTER"
[ "$AFTER" -eq 0 ] && c_green "  ✓ agent stopped" || c_red "  ✗ 仍有殘留"
