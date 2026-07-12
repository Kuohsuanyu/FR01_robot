#!/usr/bin/env bash
# 把一顆已驗證的腿部 policy 部署到 leg RPi,並把 model.kinfer 指向它。
#
# 流程:  先驗證(verify_kinfer.py)→ scp 到 leg RPi → 重指 symlink。
# firmware 讀取工作目錄的 model.kinfer(symlink),所以換 policy = 換 symlink。
#
# 用法:
#   leg/policies/deploy_leg_policy.sh leg/policies/verified/my_new_policy.kinfer
#   leg/policies/deploy_leg_policy.sh <檔> --force     # 跳過驗證(不建議)
set -u
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
source "$REPO/scripts/lib.sh"

POLICY="${1:-}"
FORCE=0; [ "${2:-}" = "--force" ] && FORCE=1
[ -z "$POLICY" ] && { c_red "用法: $0 <path/to.kinfer> [--force]"; exit 1; }
[ -f "$POLICY" ] || { c_red "找不到檔案:$POLICY"; exit 1; }

# 1) 驗證
if [ "$FORCE" -eq 0 ]; then
    c_cyan "[deploy] 先驗證 ..."
    if ! python3 "$HERE/verify_kinfer.py" "$POLICY"; then
        c_red "[deploy] 驗證未通過,中止(要強制請加 --force)。"; exit 1
    fi
fi

# 2) 找 leg RPi
LEG="${RPI_LEG_HOST:-fr01-leg.local}"
if [ -z "$LEG" ]; then c_red "[deploy] 找不到 leg RPi host"; exit 1; fi
c_cyan "[deploy] 目標 leg RPi:$RPI_LEG_USER@$LEG"

NAME="$(basename "$POLICY")"
REMOTE_DIR="~/robot_data/policy/FR01"
REMOTE_LINK="~/robot_data/kbot_deployment/model.kinfer"

# 3) scp + 重指 symlink
c_cyan "[deploy] 上傳 $NAME → $REMOTE_DIR ..."
scp $RPI_SSH_OPTS "$POLICY" "$RPI_LEG_USER@$LEG:$REMOTE_DIR/$NAME" || {
    c_red "[deploy] scp 失敗(leg RPi 開機了嗎?)"; exit 1; }

c_cyan "[deploy] 將 model.kinfer 指向 $NAME ..."
ssh $RPI_SSH_OPTS "$RPI_LEG_USER@$LEG" \
    "ln -sf $REMOTE_DIR/$NAME $REMOTE_LINK && ls -l $REMOTE_LINK" || {
    c_red "[deploy] 重指 symlink 失敗"; exit 1; }

c_green "[deploy] ✓ 完成。model.kinfer → $NAME"
c_yellow "  下一步:在 leg RPi 上重啟 firmware 讓新 policy 生效"
c_yellow "  (視你的啟動方式:重跑 kbot_deployment/run,或 systemctl restart <你的服務>)"
