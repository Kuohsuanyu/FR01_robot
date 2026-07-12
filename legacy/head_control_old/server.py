"""
VR 頭追遙現 - PC 端伺服器（WebRTC P2P 版，最低延遲）
  - 影像/IMU 走 WebRTC：靠 STUN 探到公網 IP，手機與本機直接 P2P，媒體不經中繼
  - 網頁與 signaling 仍可用 Cloudflare 隧道（受信任 HTTPS、手機零設定）
  - 攝影機可選/自癒；雲台 (neck_gimbal) 跟隨頭部；/status 即時狀態

用法（區網直連）:
  python server.py                       # 攝影機0, https://<本機IP>:8443
  python server.py --camera 0 --rotate 90 --width 640 --height 480

用法（外網）:
  1) python server.py
  2) ../remote_access/cloudflared.exe tunnel --url https://localhost:8443 --no-tls-verify
  3) 手機開 cloudflared 給的網址 → 進入 VR；WebRTC 會自動 P2P 直連你家公網 IP
"""
import argparse
import asyncio
import datetime
import ipaddress
import json
import os
import socket
import ssl
import threading
import time

import cv2
import numpy as np
from aiohttp import web
from aiortc import (RTCConfiguration, RTCIceServer, RTCPeerConnection,
                    RTCSessionDescription, VideoStreamTrack)
from av import VideoFrame

ROOT = os.path.dirname(os.path.abspath(__file__))
CERT = os.path.join(ROOT, "cert.pem")
KEY = os.path.join(ROOT, "key.pem")

pcs = set()
latest_pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
STATE = {"clients": 0, "poses": 0, "last_pose_t": 0.0, "frames": 0, "last_frame_t": 0.0}

# ── 接馬達開關 ──（給好 neck_gimbal.py 的 AXES 後設 True）
NECK_ENABLED = True
neck = None

# STUN：讓 WebRTC 探到公網 IP 以做 P2P 直連
ICE = RTCConfiguration(iceServers=[
    RTCIceServer(urls="stun:stun.l.google.com:19302"),
    RTCIceServer(urls="stun:stun1.l.google.com:19302"),
])


# ───────────────────────── 攝影機（可選/自癒） ─────────────────────────
class Camera:
    def __init__(self, index, width, height):
        self.index = index
        self.w, self.h = width, height
        self.cap = None
        self.frame = None
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _open(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    def _loop(self):
        while self._run:
            if self.cap is None:
                self.cap = self._open()
                if self.cap is None:
                    time.sleep(1.0)
                    continue
                print(f"\n[camera] 已連接 index={self.index}", flush=True)
            ret, f = self.cap.read()
            if ret:
                self.frame = f
            else:
                self.cap.release()
                self.cap = None
                self.frame = None
                print("\n[camera] 影像中斷，重連中…", flush=True)
                time.sleep(0.5)

    def stop(self):
        self._run = False
        if self.cap:
            self.cap.release()


def _no_cam_frame(w, h):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.putText(img, "NO CAMERA", (int(w * 0.18), int(h * 0.52)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2, cv2.LINE_AA)
    return img


_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class CameraTrack(VideoStreamTrack):
    def __init__(self, cam, rotate=0):
        super().__init__()
        self.cam = cam
        self.rotate = rotate

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        img = self.cam.frame
        if img is None:
            img = _no_cam_frame(self.cam.w, self.cam.h)
        if self.rotate in _ROT:
            img = cv2.rotate(img, _ROT[self.rotate])
        frame = VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts, frame.time_base = pts, time_base
        STATE["frames"] += 1
        STATE["last_frame_t"] = time.time()
        return frame


# ───────────────────────── WebRTC signaling ─────────────────────────
async def offer(request):
    params = await request.json()
    pc = RTCPeerConnection(configuration=ICE)
    pcs.add(pc)
    pc.addTrack(CameraTrack(request.app["camera"], request.app["rotate"]))
    STATE["clients"] += 1
    print(f"\n[webrtc] 新連線  目前={STATE['clients']}", flush=True)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            try:
                d = json.loads(message)
            except Exception:
                return
            latest_pose.update(d)
            STATE["poses"] += 1
            STATE["last_pose_t"] = time.time()
            if neck is not None:
                neck.set_head_pose(d.get("yaw", 0), d.get("pitch", 0), d.get("roll", 0))
            print(f"\rhead  yaw={d.get('yaw',0):7.1f}  "
                  f"pitch={d.get('pitch',0):7.1f}  roll={d.get('roll',0):7.1f}",
                  end="", flush=True)

    @pc.on("connectionstatechange")
    async def on_state():
        print(f"\n[webrtc] {pc.connectionState}", flush=True)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            pcs.discard(pc)
            STATE["clients"] = max(0, STATE["clients"] - 1)

    await pc.setRemoteDescription(RTCSessionDescription(params["sdp"], params["type"]))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


# ───────────────────────── 靜態檔 / 狀態 ─────────────────────────
async def index(request):
    return web.FileResponse(os.path.join(ROOT, "index.html"))


async def static_file(request):
    name = os.path.basename(request.match_info.get("name") or request.path)
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        return web.Response(status=404)
    return web.FileResponse(path)


async def pose_ws(request):
    """校準模式專用：輕量 WebSocket，只收姿態(yaw/pitch/roll)，不需影像。"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    STATE["clients"] += 1
    print("\n[cal] 校準連線", flush=True)
    try:
        async for msg in ws:
            if msg.type != web.WSMsgType.TEXT:
                continue
            try:
                d = json.loads(msg.data)
            except Exception:
                continue
            latest_pose.update(d)
            STATE["poses"] += 1
            STATE["last_pose_t"] = time.time()
            if neck is not None:
                neck.set_head_pose(d.get("yaw", 0), d.get("pitch", 0), d.get("roll", 0))
            print(f"\r[cal] yaw={d.get('yaw',0):7.1f}  pitch={d.get('pitch',0):7.1f}  "
                  f"roll={d.get('roll',0):7.1f}", end="", flush=True)
    finally:
        STATE["clients"] = max(0, STATE["clients"] - 1)
        print("\n[cal] 校準斷線", flush=True)
    return ws


async def status(request):
    age = time.time() - STATE["last_pose_t"] if STATE["last_pose_t"] else None
    fage = time.time() - STATE["last_frame_t"] if STATE["last_frame_t"] else None
    return web.json_response({
        "clients": STATE["clients"],
        "latest_pose": latest_pose,
        "poses_recv": STATE["poses"],
        "last_pose_age_sec": round(age, 2) if age is not None else None,
        "frames_sent": STATE["frames"],
        "last_frame_age_sec": round(fage, 2) if fage is not None else None,
    })


async def on_shutdown(app):
    await asyncio.gather(*[pc.close() for pc in pcs])
    pcs.clear()
    app["camera"].stop()
    if neck is not None:
        neck.stop()


# ───────────────────────── 工具 ─────────────────────────
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_cert(ip):
    if os.path.exists(CERT) and os.path.exists(KEY):
        return
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, ip)])
    san = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address(ip))]
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .sign(key, hashes.SHA256())
    )
    with open(KEY, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption()))
    with open(CERT, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    print(f"[cert] 已產生自簽憑證: {CERT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="影像旋轉角度（鏡頭橫裝就設 90 或 270）")
    args = ap.parse_args()

    ip = get_local_ip()
    ensure_cert(ip)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(CERT, KEY)

    global neck
    if NECK_ENABLED:
        try:
            from neck_gimbal import NeckGimbal
            neck = NeckGimbal()
            neck.start()
            print("[neck] 雲台已啟用，將跟隨頭部姿態")
        except Exception as e:
            neck = None
            print(f"[neck] 連線失敗（{e}）→ 先不接雲台，只串影像；檢查 COM 埠後重啟即可")

    app = web.Application()
    app["camera"] = Camera(args.camera, args.width, args.height)
    app["rotate"] = args.rotate
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_get("/status", status)
    app.router.add_get("/pose", pose_ws)
    app.router.add_get("/manifest.json", static_file)
    app.router.add_get("/{name:icon-.*\\.png}", static_file)
    app.on_shutdown.append(on_shutdown)

    print("=" * 60)
    print("  區網直連:  https://%s:%d" % (ip, args.port))
    print("  外網:      cloudflared tunnel --url https://localhost:%d --no-tls-verify" % args.port)
    print("            （WebRTC 會用 STUN 自動 P2P 直連你家公網 IP）")
    print("=" * 60)
    web.run_app(app, host="0.0.0.0", port=args.port, ssl_context=ctx)


if __name__ == "__main__":
    main()
