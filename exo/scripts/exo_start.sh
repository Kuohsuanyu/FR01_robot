#!/usr/bin/env bash
# One-click: start exo agent on exo RPi + launch dual-ghost GUI locally.
#
# Env / flags:
#   --fake        skip real serial on RPi, generate sinusoidal fake pose
#                 (useful when the exoskeleton hardware isn't hooked up yet)
set -e
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
source "$REPO/scripts/lib.sh"

FAKE_FLAG=""
FAKE_ARG=""
for arg in "$@"; do
    case "$arg" in
        --fake) FAKE_FLAG="--fake"; FAKE_ARG="--fake" ;;
    esac
done

# ── discover exo RPi -----------------------------------------------------
if [ -z "${RPI_EXO_HOST:-}" ] && [ -f "$REPO/.rpi_host_exo" ]; then
    RPI_EXO_HOST="$(cat "$REPO/.rpi_host_exo")"
fi
if [ -z "${RPI_EXO_HOST:-}" ]; then
    c_yellow "  exo RPi IP unknown — running local fake agent only."
    FAKE_FLAG="--fake"
    # fake agent starts on this laptop so exo_gui.py can talk to it
    LOCAL_FAKE=1
fi

if [ -n "${LOCAL_FAKE:-}" ]; then
    c_cyan "  launching LOCAL fake exo agent on 127.0.0.1:8200 ..."
    pkill -f 'exo_rpi_agent/agent.py' 2>/dev/null || true
    (cd "$REPO/exo/exo_rpi_agent" && \
     nohup python3 agent.py --fake > /tmp/exo_agent.log 2>&1 & disown)
    sleep 1
    EXO_URL="127.0.0.1:8200"
else
    c_cyan "  starting exo agent on $RPI_EXO_HOST ..."
    RPI_EXO_KEY="${RPI_EXO_KEY:-$RPI_KEY}"
    RPI_EXO_USER="${RPI_EXO_USER:-robot}"
    ssh -i "$RPI_EXO_KEY" -o BatchMode=yes \
        "$RPI_EXO_USER@$RPI_EXO_HOST" \
        "pkill -f 'exo_rpi_agent/agent.py' 2>/dev/null; \
         cd ~/exo_rpi_agent && (nohup python3 -u agent.py $FAKE_ARG > ~/exo_agent.log 2>&1 &) && sleep 1"
    EXO_URL="$RPI_EXO_HOST:8200"
fi

c_cyan "  launching local dual-ghost GUI ..."
cd "$REPO/exo/exo_pc_client"
exec python3 exo_gui.py --exo-host "$EXO_URL" --robot-host "$RPI_HOST:8000"
