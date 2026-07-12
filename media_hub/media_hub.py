#!/usr/bin/env python3
"""FR01 Media Hub — 單一進程獨占相機,一次擷取、多方消費。

解決「多個 agent 搶同一顆 /dev/video」與「MJPEG 串流不順」兩大問題。
架構(參考 Open-TeleVision 的 single-owner + WebRTC 概念,改用 cv2 擷取):

    [capture thread]  cv2.VideoCapture (唯一持有相機)
          │  每幀寫入 ↓
          ├── FrameBus.latest_bgr   ── WebRTC 自訂 track ── H264/VP8 ─→ VR / 瀏覽器
          ├── FrameBus.latest_jpg   ── /mjpeg /snapshot ────────────→ 既有 client 相容
          └── shared_memory ndarray ── 本機其他 agent 直接讀 ───────→ 手部視覺等

端點:
    GET  /            — 測試用 WebRTC 檢視頁
    POST /offer       — WebRTC SDP 交換(每個 client 一條 PeerConnection,共用同一份影格)
    GET  /mjpeg       — MJPEG multipart(相容舊 client)
    GET  /snapshot    — 單張 JPEG
    GET  /status      — JSON:相機資訊 + 共享記憶體名稱/shape + WebRTC 連線數

用法:  python3 media_hub.py --cam 0 --width 640 --height 480 --fps 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time
from fractions import Fraction

import cv2
import numpy as np
from aiohttp import web
import aiohttp_cors
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import VideoFrame
from multiprocessing import shared_memory

HERE = os.path.dirname(os.path.abspath(__file__))
_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE}


# ────────────────────────────────────────────────────────────────────
# FrameBus — 單一影格來源,一寫多讀
# ────────────────────────────────────────────────────────────────────
class FrameBus:
    """一個 producer(擷取執行緒)、多個 consumer(WebRTC/MJPEG/shm)。
    只保留『最新一幀』,消費者各自以自己的速率取用,天生 fan-out、不搶裝置。"""

    def __init__(self, h: int, w: int, shm_name: str = "fr01_cam"):
        self.h, self.w = h, w
        self._lock = threading.Lock()
        self._bgr = np.zeros((h, w, 3), np.uint8)
        self._jpg: bytes | None = None
        self.seq = 0
        # 本機同機器其他 agent 用的共享記憶體(raw BGR)
        size = int(np.prod((h, w, 3)))
        try:
            self.shm = shared_memory.SharedMemory(
                name=shm_name, create=True, size=size)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=shm_name)
        self.shm_name = self.shm.name
        self.shm_array = np.ndarray((h, w, 3), np.uint8, buffer=self.shm.buf)

    def publish(self, bgr: np.ndarray, jpg: bytes):
        with self._lock:
            self._bgr = bgr
            self._jpg = jpg
            self.seq += 1
            np.copyto(self.shm_array, bgr)      # 給本機 consumer

    def latest_bgr(self) -> np.ndarray:
        with self._lock:
            return self._bgr

    def latest_jpg(self) -> bytes | None:
        with self._lock:
            return self._jpg

    def close(self):
        try:
            self.shm.close(); self.shm.unlink()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────
# Camera — 唯一持有 /dev/video 的擷取執行緒
# ────────────────────────────────────────────────────────────────────
class Camera:
    def __init__(self, bus: FrameBus, index: int, w: int, h: int,
                 fps: int, quality: int, rotate: int):
        self.bus = bus
        self.index, self.w, self.h = index, w, h
        self.fps, self.quality = fps, quality
        self.rotate = rotate
        self.error = ""
        self.frame_count = 0
        self._run = True
        self._cap = None
        self._t = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._t.start()

    def _open(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.index)
        # 直接向相機要 MJPG,少一次 YUV→RGB
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _loop(self):
        while self._run:
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._open()
                if not self._cap.isOpened():
                    self.error = f"cannot open camera {self.index}"
                    time.sleep(1.0)
                    continue
                self.error = ""
            ok, frame = self._cap.read()
            if not ok:
                self.error = "read failed — reopening"
                self._cap.release(); self._cap = None
                time.sleep(0.3)
                continue
            if self.rotate in _ROT:
                frame = cv2.rotate(frame, _ROT[self.rotate])
            # 尺寸保險(旋轉/相機不吃設定時)
            if frame.shape[0] != self.h or frame.shape[1] != self.w:
                frame = cv2.resize(frame, (self.w, self.h))
            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            self.bus.publish(frame, buf.tobytes() if ok else None)
            self.frame_count += 1

    def stop(self):
        self._run = False
        if self._cap:
            self._cap.release()


# ────────────────────────────────────────────────────────────────────
# WebRTC — 每個 client 一條 track,全部讀 FrameBus 的最新幀
# ────────────────────────────────────────────────────────────────────
class CameraTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, bus: FrameBus, fps: int):
        super().__init__()
        self.bus = bus
        self.frame_interval = 1.0 / fps
        self._next = time.time()
        self._t0 = time.time()

    async def recv(self):
        # 依 fps 節流,避免空轉
        now = time.time()
        wait = self._next - now
        if wait > 0:
            await asyncio.sleep(wait)
        self._next = time.time() + self.frame_interval

        bgr = self.bus.latest_bgr()
        vf = VideoFrame.from_ndarray(bgr, format="bgr24")
        vf.pts = int((time.time() - self._t0) * 90000)
        vf.time_base = Fraction(1, 90000)
        return vf


# ────────────────────────────────────────────────────────────────────
# Web app
# ────────────────────────────────────────────────────────────────────
class Hub:
    def __init__(self, bus: FrameBus, cam: Camera, fps: int):
        self.bus, self.cam, self.fps = bus, cam, fps
        self.pcs: set[RTCPeerConnection] = set()

    async def index(self, request):
        return web.FileResponse(os.path.join(HERE, "index.html"))

    async def status(self, request):
        return web.json_response({
            "camera_index": self.cam.index,
            "camera_error": self.cam.error,
            "width": self.bus.w, "height": self.bus.h,
            "fps": self.fps,
            "frame_count": self.cam.frame_count,
            "shm_name": self.bus.shm_name,      # 本機 consumer 用它 attach
            "shm_shape": [self.bus.h, self.bus.w, 3],
            "shm_dtype": "uint8",
            "webrtc_clients": len(self.pcs),
        })

    async def snapshot(self, request):
        jpg = self.bus.latest_jpg()
        if jpg is None:
            return web.Response(status=503, text="no frame yet")
        return web.Response(body=jpg, content_type="image/jpeg")

    async def mjpeg(self, request):
        resp = web.StreamResponse(status=200, headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame"})
        await resp.prepare(request)
        last = -1
        try:
            while True:
                if self.bus.seq != last:
                    last = self.bus.seq
                    jpg = self.bus.latest_jpg()
                    if jpg:
                        await resp.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(jpg)).encode() +
                            b"\r\n\r\n" + jpg + b"\r\n")
                await asyncio.sleep(1.0 / self.fps)
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        return resp

    async def offer(self, request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def _on_state():
            print(f"[webrtc] state={pc.connectionState} "
                  f"(clients={len(self.pcs)})", flush=True)
            if pc.connectionState in ("failed", "closed"):
                await pc.close()
                self.pcs.discard(pc)

        pc.addTrack(CameraTrack(self.bus, self.fps))
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return web.json_response(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    async def on_shutdown(self, app):
        await asyncio.gather(*[pc.close() for pc in self.pcs])
        self.pcs.clear()
        self.cam.stop()
        self.bus.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=70)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--shm-name", default="fr01_cam")
    args = ap.parse_args()

    bus = FrameBus(args.height, args.width, shm_name=args.shm_name)
    cam = Camera(bus, args.cam, args.width, args.height,
                 args.fps, args.quality, args.rotate)
    cam.start()
    hub = Hub(bus, cam, args.fps)

    app = web.Application()
    cors = aiohttp_cors.setup(app, defaults={"*": aiohttp_cors.ResourceOptions(
        allow_credentials=True, expose_headers="*",
        allow_headers="*", allow_methods="*")})
    app.router.add_get("/", hub.index)
    cors.add(app.router.add_get("/status", hub.status))
    cors.add(app.router.add_get("/snapshot", hub.snapshot))
    cors.add(app.router.add_get("/mjpeg", hub.mjpeg))
    cors.add(app.router.add_post("/offer", hub.offer))
    app.on_shutdown.append(hub.on_shutdown)

    print(f"[hub] camera {args.cam} {args.width}x{args.height}@{args.fps} "
          f"→ http://{args.host}:{args.port}  shm='{bus.shm_name}'", flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
