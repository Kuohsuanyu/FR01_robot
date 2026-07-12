#!/usr/bin/env bash
# Stop exo-related processes both locally and (if reachable) on exo RPi.
set +e
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
source "$REPO/scripts/lib.sh"

pkill -f 'exo_gui.py'          2>/dev/null || true
pkill -f 'exo_rpi_agent/agent.py' 2>/dev/null || true

if [ -f "$REPO/.rpi_host_exo" ]; then
    HOST="$(cat "$REPO/.rpi_host_exo")"
    RPI_EXO_USER="${RPI_EXO_USER:-robot}"
    ssh -i "$RPI_KEY" -o BatchMode=yes -o ConnectTimeout=5 \
        "$RPI_EXO_USER@$HOST" \
        "pkill -f 'exo_rpi_agent/agent.py' 2>/dev/null; sleep 0.3" 2>/dev/null || true
fi
c_green "  ✓ exo pipeline stopped"
