#!/usr/bin/env bash
# One-click stop: kill any running hand-related processes on this laptop
# AND any RPi-side hand daemons.  Safe to run when nothing's up.
set +e
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"

c_cyan "==== 停止靈巧手 pipeline ===="

# Local: camera_receiver + glove_receiver
n_local=0
for pat in "camera_receiver.py" "glove_receiver.py"; do
    if pgrep -f "$pat" >/dev/null; then
        c_cyan "  killing local $pat ..."
        pkill -f "$pat"
        n_local=$((n_local+1))
    fi
done
[ "$n_local" -eq 0 ] && c_yellow "  no local hand process was running"

# Remote (RPi): glove_receiver daemon leftover from earlier UDP-mode tries.
# Head_agent is left running (arm/head still need it).
if rpi_ping; then
    if rpi_ssh "pgrep -f 'glove_receiver.py' >/dev/null"; then
        c_cyan "  killing RPi glove_receiver.py ..."
        rpi_ssh 'pkill -f "glove_receiver.py"; sleep 0.3' || true
    fi
    if rpi_ssh "pgrep -f 'hand_slider_gui.py' >/dev/null"; then
        c_cyan "  killing RPi hand_slider_gui.py ..."
        rpi_ssh 'pkill -f "hand_slider_gui.py"; sleep 0.3' || true
    fi
fi

c_green "  ✓ done"
