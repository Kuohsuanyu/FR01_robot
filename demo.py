#!/usr/bin/env python3
"""
RX1 Windows Motor Controller — Interactive Demo

Usage:
    python demo.py COM8

Commands (type at the prompt):
    home              - Move all joints to zero / home position
    right_arm         - Right arm slow wave (shoulder joint)
    left_arm          - Left arm slow wave
    head              - Head nod
    grip_r <0-1>      - Right gripper  (0=open, 1=closed)
    grip_l <0-1>      - Left gripper
    ping              - Check serial port is open
    quit / exit       - Disconnect and exit
"""

import sys
import time
import math

from rx1_motor_win import Rx1Motor


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM8"

    print(f"[RX1] Connecting to {port} @ 1 Mbaud ...")
    robot = Rx1Motor(port)
    try:
        robot.initialize()
    except RuntimeError as e:
        print(f"[RX1] ERROR: {e}")
        sys.exit(1)

    print("[RX1] Initialized.  All joints moved to home position.\n")
    print("Commands: home | right_arm | left_arm | head | "
          "grip_r <0-1> | grip_l <0-1> | ping | quit\n")

    while True:
        try:
            raw = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        tokens = raw.lower().split()
        if not tokens:
            continue
        cmd = tokens[0]

        if cmd in ("quit", "exit", "q"):
            break

        elif cmd == "ping":
            print(f"[OK] Port {port} is open.")

        elif cmd == "home":
            print("Moving to home ...")
            robot.right_arm([0.0] * 7)
            robot.left_arm([0.0] * 7)
            robot.torso([0.0] * 3)
            robot.head([0.0] * 5)
            print("Done.")

        elif cmd == "right_arm":
            print("Right arm wave (shoulder) ...")
            for t in range(30):
                angle = math.sin(t * 0.3) * 0.5
                robot.right_arm([angle, 0, 0, 0, 0, 0, 0])
                time.sleep(0.08)
            robot.right_arm([0.0] * 7)
            print("Done.")

        elif cmd == "left_arm":
            print("Left arm wave (shoulder) ...")
            for t in range(30):
                angle = math.sin(t * 0.3) * 0.5
                robot.left_arm([angle, 0, 0, 0, 0, 0, 0])
                time.sleep(0.08)
            robot.left_arm([0.0] * 7)
            print("Done.")

        elif cmd == "head":
            print("Head nod ...")
            for t in range(30):
                angle = math.sin(t * 0.3) * 0.25
                robot.head([0, angle, 0, 0, 0])
                time.sleep(0.08)
            robot.head([0.0] * 5)
            print("Done.")

        elif cmd == "grip_r":
            if len(tokens) < 2:
                print("Usage: grip_r <0.0-1.0>")
                continue
            try:
                ratio = max(0.0, min(1.0, float(tokens[1])))
            except ValueError:
                print("Invalid number.")
                continue
            robot.right_gripper(ratio)
            print(f"Right gripper → {ratio:.2f}")

        elif cmd == "grip_l":
            if len(tokens) < 2:
                print("Usage: grip_l <0.0-1.0>")
                continue
            try:
                ratio = max(0.0, min(1.0, float(tokens[1])))
            except ValueError:
                print("Invalid number.")
                continue
            robot.left_gripper(ratio)
            print(f"Left gripper → {ratio:.2f}")

        else:
            print(f"Unknown command: {cmd}")

    robot.close()
    print("[RX1] Disconnected.")


if __name__ == "__main__":
    main()
