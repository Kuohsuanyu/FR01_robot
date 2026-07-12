# Shared bash helpers used by the launch scripts under this directory.
# Not executable on its own — `source` it from the other .sh files.

# Repo root (../ from scripts/)
REPO="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." && pwd )"

# ── RPi connection ──────────────────────────────────────────────────────────
RPI_USER="robot"          # head/arm/hand RPi login
RPI_LEG_USER="fr01"       # legs / kbot RPi login

# Head RPi IP resolution: env → cache → hard fallback
if [ -n "${QBOT_RPI_HOST:-}" ]; then
    RPI_HOST="$QBOT_RPI_HOST"
elif [ -f "$REPO/.rpi_host" ]; then
    RPI_HOST="$(cat "$REPO/.rpi_host")"
else
    RPI_HOST="192.168.0.123"
fi

# Leg RPi IP resolution: env → cache → blank
if [ -n "${QBOT_RPI_LEG_HOST:-}" ]; then
    RPI_LEG_HOST="$QBOT_RPI_LEG_HOST"
elif [ -f "$REPO/.rpi_host_leg" ]; then
    RPI_LEG_HOST="$(cat "$REPO/.rpi_host_leg")"
else
    RPI_LEG_HOST=""
fi

# Exo RPi IP resolution: env → cache → blank
if [ -n "${QBOT_RPI_EXO_HOST:-}" ]; then
    RPI_EXO_HOST="$QBOT_RPI_EXO_HOST"
elif [ -f "$REPO/.rpi_host_exo" ]; then
    RPI_EXO_HOST="$(cat "$REPO/.rpi_host_exo")"
else
    RPI_EXO_HOST=""
fi
RPI_KEY="$HOME/.ssh/qbot_rpi"
RPI_SSH_OPTS="-i $RPI_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=5"

rpi_ssh() {   # rpi_ssh "remote command"  — head RPi
  ssh $RPI_SSH_OPTS "$RPI_USER@$RPI_HOST" "$1"
}

rpi_ping() {  # returns 0 if head RPi reachable + ssh works
  ssh $RPI_SSH_OPTS -o BatchMode=yes "$RPI_USER@$RPI_HOST" true 2>/dev/null
}

rpi_leg_ssh() {   # rpi_leg_ssh "remote command"  — leg RPi
  [ -z "$RPI_LEG_HOST" ] && return 1
  ssh $RPI_SSH_OPTS "$RPI_LEG_USER@$RPI_LEG_HOST" "$1"
}

rpi_leg_ping() {  # returns 0 if leg RPi reachable + ssh works
  [ -z "$RPI_LEG_HOST" ] && return 1
  ssh $RPI_SSH_OPTS -o BatchMode=yes "$RPI_LEG_USER@$RPI_LEG_HOST" true 2>/dev/null
}

# Coloured echo (works in most terminals)
c_green()  { printf '\033[32m%s\033[0m\n' "$*"; }
c_red()    { printf '\033[31m%s\033[0m\n' "$*"; }
c_yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
c_cyan()   { printf '\033[36m%s\033[0m\n' "$*"; }
