#!/usr/bin/env bash
# 幫某台 RPi 設定「唯一主機名 + avahi」,之後就能用 fr01-<x>.local 連,IP 免管。
#
# 用法(RPi 要先開機、同網段):
#   scripts/rpi_set_hostname.sh head     # → fr01-head  (robot@)
#   scripts/rpi_set_hostname.sh exo      # → fr01-exo   (robot@)
#   scripts/rpi_set_hostname.sh leg      # → fr01-leg   (fr01@)
#
# 會用 find_rpi.sh 以 MAC 找出目前 IP(此時名稱還沒設),再 ssh -t 進去設定。
# 需要 RPi 的 sudo 密碼(會在你終端機互動輸入)。設完重開一次最保險。
set -u
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"

MODE="${1:-}"
case "$MODE" in
    head) NAME="fr01-head"; USER="robot"; FLAG="--head" ;;
    exo)  NAME="fr01-exo";  USER="robot"; FLAG="--exo"  ;;
    leg)  NAME="fr01-leg";  USER="fr01";  FLAG="--leg"  ;;
    *) c_red "用法: $0 head|exo|leg"; exit 1 ;;
esac

c_cyan "[set-hostname:$MODE] 以 MAC 找出目前 IP ..."
IP="$(bash "$HERE/find_rpi.sh" "$FLAG" 2>/dev/null | tail -1)"
if [ -z "$IP" ]; then
    c_red "  找不到 $MODE RPi(確認已開機、同網段)。"; exit 1
fi
c_green "  目前 $MODE 在 $IP,將設定主機名為 '$NAME'"

# 在 RPi 上:設 hostname + /etc/hosts + 確保 avahi 啟用
ssh -t -i "$RPI_KEY" -o StrictHostKeyChecking=no "$USER@$IP" "
    set -e
    sudo hostnamectl set-hostname '$NAME'
    sudo sed -i 's/^127.0.1.1.*/127.0.1.1\t$NAME/' /etc/hosts || \
        echo '127.0.1.1\t$NAME' | sudo tee -a /etc/hosts >/dev/null
    if ! command -v avahi-daemon >/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y avahi-daemon
    fi
    sudo systemctl enable --now avahi-daemon
    echo '=== 完成:主機名 -> '\$(hostname)' ; avahi:' \$(systemctl is-active avahi-daemon)
"

c_green "[set-hostname:$MODE] 建議重開一次:ssh $USER@$IP 'sudo reboot'"
c_cyan  "  之後即可用  $NAME.local  連線(IP 變了也不用管)。"
