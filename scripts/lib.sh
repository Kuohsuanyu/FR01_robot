# Shared bash helpers used by the launch scripts under this directory.
# Not executable on its own — `source` it from the other .sh files.

# Repo root (../ from scripts/)
REPO="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." && pwd )"

# ── RPi connection ──────────────────────────────────────────────────────────
RPI_USER="robot"          # head/arm/hand RPi login
RPI_LEG_USER="fr01"       # legs / kbot RPi login

# 穩定 mDNS 主機名(IP 怎麼變都用這個連;各 RPi 需設好唯一 hostname + avahi)
RPI_HEAD_NAME="fr01-head.local"
RPI_LEG_NAME="fr01-leg.local"
RPI_EXO_NAME="fr01-exo.local"

# 解析順序:env 覆寫 → mDNS 名稱(主)→ 最後已知快取 IP → hard 預設。
# 名稱能解析就用名稱(IP 變動免管);解不到才退回快取,再不行才靠 find_rpi.sh 掃描。
_resolve_host() {   # $1=mdns_name  $2=cache_file  $3=hard_default
    local name="$1" cache="$2" def="$3"
    if getent hosts "$name" >/dev/null 2>&1; then echo "$name"; return; fi
    if [ -f "$cache" ]; then
        local c; c="$(cat "$cache" 2>/dev/null)"
        [ -n "$c" ] && { echo "$c"; return; }
    fi
    echo "$def"
}

RPI_HOST="${QBOT_RPI_HOST:-$(_resolve_host "$RPI_HEAD_NAME" "$REPO/.rpi_host"     "$RPI_HEAD_NAME")}"
RPI_LEG_HOST="${QBOT_RPI_LEG_HOST:-$(_resolve_host "$RPI_LEG_NAME" "$REPO/.rpi_host_leg" "")}"
RPI_EXO_HOST="${QBOT_RPI_EXO_HOST:-$(_resolve_host "$RPI_EXO_NAME" "$REPO/.rpi_host_exo" "")}"
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
