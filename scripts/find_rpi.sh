#!/usr/bin/env bash
# Auto-discover a specific RPi on the current network by MAC address and
# cache the resulting IP.  Both the head RPi and the leg RPi share the
# hostname "raspberrypi" so mDNS is ambiguous — MAC scan is authoritative.
#
# Usage:
#   find_rpi.sh                # default: head RPi
#   find_rpi.sh --leg          # leg RPi
#   find_rpi.sh --mac AA:BB..  # arbitrary MAC (advanced)
#
# Writes the discovered IP to $REPO/.rpi_host (head) or .rpi_host_leg (leg)
# so lib.sh + launcher.py pick it up.  Exit 0 on success.
set -u
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

# ── known robot MACs ────────────────────────────────────────────────────────
HEAD_MAC="d8:3a:dd:d4:6b:76"          # wlan0 of upper-body RPi (head/arm/hand)
LEG_MAC="b8:27:eb:02:db:cf"           # wlan0 of lower-body RPi (legs / kbot)
# Exo MAC intentionally unknown at commit time — set once by:
#   echo "aa:bb:cc:dd:ee:ff" > $REPO/.rpi_exo_mac
# so the exoskeleton RPi can be discovered like the other two.
EXO_MAC="${EXO_MAC:-$(cat "$REPO/.rpi_exo_mac" 2>/dev/null || echo "")}"
HEAD_FALLBACK="192.168.0.123"

MODE="head"
CUSTOM_MAC=""
CACHE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --leg)  MODE="leg";  shift ;;
        --head) MODE="head"; shift ;;
        --exo)  MODE="exo";  shift ;;
        --mac)  CUSTOM_MAC="$2"; MODE="custom"; shift 2 ;;
        *) shift ;;
    esac
done

# 穩定 mDNS 主機名(名稱優先;各 RPi 需設唯一 hostname + avahi)
NAME=""
case "$MODE" in
    head)   MAC="$HEAD_MAC"; CACHE="$REPO/.rpi_host";     NAME="fr01-head.local" ;;
    leg)    MAC="$LEG_MAC";  CACHE="$REPO/.rpi_host_leg"; NAME="fr01-leg.local" ;;
    exo)    MAC="$EXO_MAC";  CACHE="$REPO/.rpi_host_exo";  NAME="fr01-exo.local"
            if [ -z "$MAC" ]; then
                c_r() { printf '\033[31m%s\033[0m\n' "$*"; }
                c_r "[find_rpi:exo] no EXO_MAC set." >&2
                c_r "  edit \$REPO/.rpi_exo_mac (one line: aa:bb:cc:dd:ee:ff)" >&2
                c_r "  or export EXO_MAC=aa:bb:... in your shell first." >&2
                exit 1
            fi ;;
    custom) MAC="$CUSTOM_MAC"; CACHE="$REPO/.rpi_host_custom" ;;
esac
MAC="$(echo "$MAC" | tr '[:upper:]' '[:lower:]')"

c_g() { printf '\033[32m%s\033[0m\n' "$*"; }
c_r() { printf '\033[31m%s\033[0m\n' "$*"; }
c_c() { printf '\033[36m%s\033[0m\n' "$*"; }

try_ping() { ping -c1 -W2 "$1" >/dev/null 2>&1; }

# --- Phase 0: mDNS 名稱優先(IP 變動免管)--------------------------------
# 名稱解析得到且連得上就直接用名稱,並把名稱寫進快取(下次 lib.sh 也用名稱)。
if [ -n "$NAME" ] && getent hosts "$NAME" >/dev/null 2>&1 && try_ping "$NAME"; then
    c_g "[find_rpi:$MODE] mDNS $NAME 可用(免掃描)" >&2
    echo "$NAME" > "$CACHE"; echo "$NAME"; exit 0
fi

# --- Phase 1: cached IP ping check ---------------------------------------
if [ -f "$CACHE" ]; then
    CACHED="$(cat "$CACHE" 2>/dev/null)"
    if [ -n "$CACHED" ] && try_ping "$CACHED"; then
        # Also verify MAC on that IP to avoid stale cache pointing at wrong host
        LEARNT_MAC="$(ip neigh 2>/dev/null | awk -v ip="$CACHED" '$1==ip {print tolower($5); exit}')"
        if [ "$LEARNT_MAC" = "$MAC" ] || [ -z "$LEARNT_MAC" ]; then
            c_g "[find_rpi:$MODE] cached $CACHED still up (mac ok)" >&2
            echo "$CACHED"; exit 0
        fi
    fi
fi

# --- Phase 2: try known fallback (head only) -----------------------------
if [ "$MODE" = "head" ] && try_ping "$HEAD_FALLBACK"; then
    LEARNT="$(ip neigh 2>/dev/null | awk -v ip="$HEAD_FALLBACK" '$1==ip {print tolower($5); exit}')"
    if [ "$LEARNT" = "$MAC" ]; then
        c_g "[find_rpi:$MODE] fallback $HEAD_FALLBACK matches mac" >&2
        echo "$HEAD_FALLBACK" > "$CACHE"; echo "$HEAD_FALLBACK"; exit 0
    fi
fi

# --- Phase 3: ARP scan the /24 subnet for the MAC ------------------------
c_c "[find_rpi:$MODE] ARP scan for $MAC ..." >&2
MY_IP="$(ip -o -4 addr show scope global 2>/dev/null | awk '{print $4}' | head -1 | cut -d/ -f1)"
if [ -z "$MY_IP" ]; then
    c_r "[find_rpi:$MODE] no local IPv4 — connect to a network first" >&2
    exit 1
fi
NET="${MY_IP%.*}"
for i in $(seq 1 254); do ping -c1 -W1 -n "${NET}.${i}" >/dev/null 2>&1 & done
wait 2>/dev/null

IP="$(ip neigh 2>/dev/null | awk -v mac="$MAC" 'tolower($5)==mac {print $1; exit}')"
if [ -z "$IP" ]; then
    c_r "[find_rpi:$MODE] MAC $MAC not seen on ${NET}.0/24" >&2
    exit 1
fi
c_g "[find_rpi:$MODE] found → $IP" >&2
echo "$IP" > "$CACHE"
echo "$IP"
