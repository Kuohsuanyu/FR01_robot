# Q-BOT head Raspberry Pi agent

Headless agent that runs **on the Raspberry Pi** wired to the head
hardware. The PC (上位機) then connects over LAN, does all the
visualisation, and drives the motors through the agent.

```
[Raspberry Pi (headless)]                     [Linux PC (上位機)]
  Feetech gimbal (ttyACM*) ──┐
  USB camera (/dev/video*) ──┤   http+ws       head_ghost_gui.py
                              │◄──────────────┤ (ghost robot render,
                              │               │  camera panel,
                              │               │  phone IMU relay)
                              └─── agent.py ──┘
```

## Endpoints

| URL | Purpose | Latency (LAN typical) |
|---|---|---|
| `ws://<ip>:8000/ws` | Motor control + telemetry (JSON) | 1-5 ms round-trip |
| `http://<ip>:8000/mjpeg` | MJPEG camera stream (multipart) | 50-100 ms end-to-end |
| `http://<ip>:8000/snapshot` | Single JPEG for debug | one request |
| `http://<ip>:8000/status` | JSON health check | — |
| `http://<ip>:8000/` | Tiny HTML dashboard (browser test) | — |

## WebSocket protocol

**Client → Agent**

```jsonc
{"op":"sync",       "cmds":[[1,2048],[2,2200],[3,2000]], "speed":0, "acc":30}
{"op":"write",      "sid":1, "step":2048, "speed":0, "acc":30}
{"op":"torque",     "sid":1, "on":true}
{"op":"torque_all", "on":false}          // emergency stop
{"op":"read",       "sid":1, "addr":9, "size":2, "req_id":"x"}
{"op":"scan",       "range":[1,50], "req_id":"y"}
{"op":"tele_hz",    "hz":50}             // set telemetry rate 1..200 Hz
```

**Agent → Client** (streamed)

```jsonc
{"t":"tele","ts":1234.56,"motors":[
  {"sid":1,"pos":2048,"spd":0,"load":0.0,"volt":12.3,"temp":35}, ...
]}
{"t":"reply","req_id":"x","ok":true,"value":1800}
{"t":"reply","req_id":"y","ok":true,"ids":[1,2,3]}
{"t":"log","level":"warn","msg":"…"}
```

## Install on the Pi

```bash
sudo apt install -y python3-pip v4l-utils
git clone <this repo>  ~/qbot   # or rsync the folder
cd ~/qbot/deployment/head_rpi_agent
pip3 install -r requirements.txt
sudo usermod -aG dialout $USER   # for /dev/ttyACM*
# reboot / re-login so group change takes effect
```

## Run

Quick manual test:

```bash
python3 agent.py --port 8000
```

Then from the PC:

```bash
curl http://<pi-ip>:8000/status
xdg-open http://<pi-ip>:8000/           # visual sanity check
```

## Configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `HEAD_USB_SERIAL` | *(auto)* | CH343 serial number, e.g. `5B14115243` |
| `CAM_INDEX` | 0 | `/dev/videoN` |
| `CAM_W` / `CAM_H` | 640 / 480 | Capture resolution |
| `CAM_ROTATE` | 0 | 0 / 90 / 180 / 270 |
| `CAM_Q` | 70 | JPEG quality (0-100) |

## systemd (start on boot)

```bash
sudo cp qbot-head-agent.service /etc/systemd/system/
# edit User=, WorkingDirectory=, ExecStart= paths inside the file if needed
sudo systemctl daemon-reload
sudo systemctl enable --now qbot-head-agent
sudo systemctl status qbot-head-agent
journalctl -u qbot-head-agent -f          # live logs
```

## Security notice

There is **no authentication**. This is intended for a trusted LAN.
If you need to expose over the internet, put it behind a VPN or add
your own token check in `_on_cmd()`.
