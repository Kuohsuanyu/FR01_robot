#!/usr/bin/env bash
# One-click: launch the RIGHT hand slider GUI on the RPi with X11 forwarded
# back to this laptop.  Stops the arm/head agent first because it holds
# /dev/ttyACM0 open.
set -e
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"

c_cyan "==== Hand slider (RIGHT) via SSH X11 forwarding ===="

if ! rpi_ping; then
    c_red "  ✗ SSH to $RPI_HOST failed."
    exit 1
fi
c_green "  ✓ SSH ok"

# Free /dev/ttyACM0 by stopping any running motor agent.  Pattern matches
# both `python3 agent.py` and `python3 -u agent.py` forms (older narrower
# path-based pkill missed the argv-only-agent.py case and left the port
# held, giving "device busy" when the slider GUI tried to open it).
rpi_ssh 'pkill -f "python3.*agent.py" 2>/dev/null; sleep 1' || true

c_cyan "  launching python GUI on RPi (window opens on this laptop)..."
# -Y trusted X11 forwarding; DISPLAY must already point at the local X server
: "${DISPLAY:=:0}"
# Right hand SIDs 31-36 — scan window 30..40 skips head/arm motors on the
# shared daisy chain so the GUI only shows the six hand servos.
ssh -Y -i "$RPI_KEY" -o StrictHostKeyChecking=no \
    "$RPI_USER@$RPI_HOST" \
    "cd ~/hands_control && python3 -u hand_slider_gui.py --hand right --min-id 30 --max-id 40"
