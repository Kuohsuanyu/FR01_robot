#!/usr/bin/env bash
# One-click: LEFT hand camera control.
#
# Architecture (all motor + camera stays on the RPi):
#   * RPi: head_agent must be running.  It serves the camera stream at
#          http://RPi:8000/mjpeg  AND writes ALL motors on /dev/ttyACM0
#          (SCS009 hand + STS arms/head, dispatched by model).
#   * Laptop: this script runs camera_receiver.py which
#          - reads the RPi MJPEG stream
#          - runs MediaPipe Hands locally (fast CPU)
#          - sends per-finger goal ticks via WebSocket back to head_agent
set -e
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"

c_cyan "==== LEFT hand — camera imitation (RPi cam + RPi motors) ===="

# Kill any prior camera_receiver instances (local) so only one window /
# one MediaPipe consumer / one WS client exists at a time.  Also sweep
# leftover RPi glove_receiver.py from earlier UDP-mode attempts (it would
# hold /dev/ttyACM0 and starve head_agent).
c_cyan "  cleaning up prior instances..."
pkill -f "camera_receiver.py" 2>/dev/null || true

if ! rpi_ping; then
    c_red "  ✗ SSH to $RPI_HOST failed."
    exit 1
fi
c_green "  ✓ SSH ok"

rpi_ssh 'pkill -f "glove_receiver.py" 2>/dev/null; sleep 0.3' || true

# Make sure head_agent is running (it provides both camera stream and motor
# write endpoint).  If not running, start it.
if ! rpi_ssh "pgrep -f 'head_rpi_agent/agent.py' >/dev/null"; then
    c_cyan "  head_agent not running — starting on RPi..."
    rpi_ssh "cd ~/head_rpi_agent && (nohup python3 -u agent.py \
             > ~/agent.log 2>&1 &) && sleep 1"
    sleep 5
fi
if rpi_ssh "pgrep -f 'head_rpi_agent/agent.py' >/dev/null"; then
    c_green "  ✓ head_agent running"
else
    c_red "  ✗ head_agent failed to start"
    rpi_ssh 'tail -30 ~/agent.log'
    exit 1
fi

c_cyan "  starting camera_receiver locally (RPi mjpeg → MediaPipe → head_agent)..."
cd "$REPO/hand"
# `exec` replaces this shell so nothing lingers when the GUI window closes.
exec python3 camera_receiver.py --side left --show \
     --source "http://${RPI_HOST}:8000/mjpeg" \
     --remote-hand "${RPI_HOST}:8000" "$@"
