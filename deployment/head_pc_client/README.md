# Q-BOT head PC client (上位機)

Runs on your Linux **PC/laptop**. Connects to the Raspberry Pi running
[`head_rpi_agent/`](../head_rpi_agent/) over LAN and provides the full
visualisation + control side:

  * Ghost robot render (MuJoCo, head highlighted orange)
  * Live camera feed from the RPi (MJPEG)
  * 3-DOF neck sliders (yaw / pitch / roll) → sent via WebSocket
  * Live telemetry (position / voltage / temperature per motor)
  * E-STOP (torque off everything on the RPi)

```
     LAN
[This PC]  ─────────────────────►  [Raspberry Pi]
 remote_gui.py                      head_rpi_agent/agent.py
   ▲                                  │
   │ MJPEG camera (http)              ├─ /dev/ttyACM* (motors)
   │ WS telemetry+events              └─ /dev/video*  (camera)
   ▼
 motor commands (WS)
```

## Install

```bash
cd deployment/head_pc_client
pip install -r requirements.txt
```

Also requires MuJoCo + numpy + pillow — the same deps
`upper_body_control/head_ghost_gui.py` uses (already installed on this PC).

## Run

Auto-connect on launch:
```bash
python3 remote_gui.py --host 192.168.0.42
```

Or launch and enter the address in the UI:
```bash
python3 remote_gui.py
```

## Latency budget (LAN, typical)

| Stage | ~ |
|---|---|
| Slider → WebSocket send   | <1 ms |
| Network hop (Wi-Fi/wired) | 1-5 ms |
| Agent → motor (SyncWrite) | 1-2 ms |
| **Total command latency** | **5-10 ms** |
| Camera frame → PC canvas  | 50-100 ms |

## Troubleshooting

- **"connect err"** → RPi agent not running, or wrong IP/port
- **Camera panel empty** → check `http://<pi>:8000/status` in a browser;
  if `cam_error` is set the RPi couldn't open the webcam
- **Motor slider moves ghost but not real head** →
  - Uncheck **Live** was disabled?  Turn it back on
  - Or `motor_error` in `/status` means RPi couldn't open `/dev/ttyACM*`
