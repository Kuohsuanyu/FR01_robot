#!/usr/bin/env python3
"""外骨骼 → 手臂 連動橋接(安全分階段)。

流程(和你的安全計畫一致):
  exo /status(關節角 rad)──對齊層(offset/sign)──夾手臂真實限位──arm_calib.q_to_step
     ──▶ dry-run 印對照表(預設,不送)  /  --drive-arm 才真的送 arm /ws

零位/方向/縮放:**原封重用手臂的 qbot_arm_calibration.json + arm_calib**,不重算零位。
exo↔arm 之間只有一層可調對齊(exo_arm_align.json,預設 identity),到時看 ghost 校。

安全預設:
  * 預設 DRY-RUN — 完全不碰馬達,只印「exo q → 手臂目標 q → tick → SID」給你核對
  * --drive-arm 才連 arm /ws;需再加 --torque 才啟力矩(否則只送位置、力矩你手動開)
  * --hz 限速、--max-step-delta 限制每周期步進(限速平滑)
  * Ctrl-C → 送 torque_all off(E-STOP)

用法:
  # 1) 目視核對(exo 開機、arm 可不通電):看對照表 + 對照 exo ghost
  python3 exo/exo_to_arm_bridge.py --exo fr01-exo.local:8200 --dry-run
  # 2) 兩側確認無誤後,真的驅動手臂:
  python3 exo/exo_to_arm_bridge.py --exo fr01-exo.local:8200 \
        --arm fr01-head.local:8100 --drive-arm --torque --hz 30
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys, signal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arm"))
import arm_calib as AC  # noqa: E402
import aiohttp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ALIGN_PATH = os.path.join(HERE, "exo_arm_align.json")


def load_align():
    """每關節 {sign:+1/-1, offset:rad}。預設 identity。到時看 ghost 校準後存檔。"""
    base = {name: {"sign": 1.0, "offset": 0.0} for name in AC.EXO_JOINT_TO_ARM}
    if os.path.exists(ALIGN_PATH):
        try:
            base.update(json.load(open(ALIGN_PATH, encoding="utf-8")))
        except Exception as e:
            print(f"[align] 讀取失敗,用 identity:{e}")
    return base


async def poll_exo(session, url):
    async with session.get(url + "/status", timeout=1.5) as r:
        return (await r.json()).get("q", {})


def build_commands(exo_q, cal, align):
    """回傳 (rows, cmds):rows 供顯示,cmds=[[sid,step],...] 供傳送。"""
    rows, cmds = [], []
    for name, (arm, i) in AC.EXO_JOINT_TO_ARM.items():
        if name not in exo_q:
            continue
        q_exo = float(exo_q[name])
        a = align[name]
        q_arm = a["sign"] * q_exo + a["offset"]
        q_c, clamped = AC.clamp_to_limits(arm, i, q_arm)
        step = AC.q_to_step(cal, arm, i, q_c)
        sid = AC.ARM_SPEC[arm][i][0]
        rows.append((name, q_exo, q_arm, q_c, clamped, sid, step))
        cmds.append([sid, step])
    return rows, cmds


def print_table(rows):
    print("\033[H\033[J", end="")   # 清畫面
    print(f"{'exo 關節':22s} {'q_exo':>8s} {'→q_arm':>8s} {'夾後':>8s} {'':>4s} {'SID':>4s} {'tick':>6s}")
    for name, qe, qa, qc, cl, sid, step in rows:
        flag = "夾!" if cl else "  "
        print(f"{name:22s} {qe:8.3f} {qa:8.3f} {qc:8.3f} {flag:>4s} {sid:4d} {step:6d}")
    print("\n(DRY-RUN:未送任何指令。對照 exo ghost 確認方向/零位/限位無誤後,再 --drive-arm)")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exo", default="fr01-exo.local:8200")
    ap.add_argument("--arm", default="fr01-head.local:8100")
    ap.add_argument("--drive-arm", action="store_true", help="真的送指令到手臂(否則 dry-run)")
    ap.add_argument("--torque", action="store_true", help="啟動時開力矩(需 --drive-arm)")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--max-step-delta", type=int, default=80,
                    help="每周期每馬達最大步進(限速;0=不限)")
    ap.add_argument("--speed", type=int, default=300)
    ap.add_argument("--acc", type=int, default=30)
    args = ap.parse_args()

    cal = AC.load_cal()
    align = load_align()
    exo_url = "http://" + args.exo
    print(f"[bridge] exo={args.exo}  arm={args.arm}  "
          f"mode={'DRIVE-ARM' if args.drive_arm else 'DRY-RUN'}")

    async with aiohttp.ClientSession() as session:
        ws = None
        if args.drive_arm:
            ws = await session.ws_connect("http://" + args.arm + "/ws")
            if args.torque:
                for arm in ("L", "R"):
                    for sid, *_ in AC.ARM_SPEC[arm]:
                        await ws.send_str(json.dumps({"op": "torque", "sid": sid, "on": True}))
                print("[bridge] 力矩已開")
            else:
                print("[bridge] 未開力矩(--torque 才開);目前只送位置")

        last = {}

        async def estop():
            if ws:
                await ws.send_str(json.dumps({"op": "torque_all", "on": False}))
                print("\n[bridge] E-STOP:torque_all off")
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, lambda: loop.create_task(_stop(estop)))

        period = 1.0 / args.hz
        while True:
            try:
                exo_q = await poll_exo(session, exo_url)
            except Exception as e:
                print(f"[bridge] exo 讀取失敗:{e}"); await asyncio.sleep(0.5); continue
            rows, cmds = build_commands(exo_q, cal, align)

            if args.max_step_delta > 0:              # 限速平滑
                for c in cmds:
                    sid, step = c
                    if sid in last:
                        d = step - last[sid]
                        if abs(d) > args.max_step_delta:
                            step = last[sid] + args.max_step_delta * (1 if d > 0 else -1)
                            c[1] = step
                    last[sid] = c[1]

            if args.drive_arm and ws:
                await ws.send_str(json.dumps({"op": "sync", "cmds": cmds,
                                              "speed": args.speed, "acc": args.acc}))
            else:
                print_table(rows)
            await asyncio.sleep(period)


async def _stop(estop):
    await estop()
    asyncio.get_running_loop().stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, RuntimeError):
        pass
