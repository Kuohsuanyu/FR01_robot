#!/usr/bin/env python3
"""MediaPipe Pose → 4 shoulder+elbow joint angles per arm → UDP to arm GUI.

Runs on the PC.  Reads webcam frames, does pose estimation, retargets the
subject's left+right arm to the robot's (J0 shoulder_rot, J1 shoulder_lift,
J2 arm_twist, J3 elbow) joint targets, applies a One-Euro smoothing filter,
and streams JSON packets over UDP for the arm GUI to consume.

Payload format (little JSON dict per frame):
  {"L": {"q": [q0,q1,q2,q3,q4], "conf": <0..1>},
   "R": {"q": [q0,q1,q2,q3,q4], "conf": <0..1>}}
q values are radians in the same frame as cal_points.q_lo/q_hi.

Run:
    python3 pose_imitator.py --cam 0 --target 127.0.0.1:9999 --show
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np


# ── MJPEG reader (for RPi remote camera) ─────────────────────────────────────
class MjpegReader:
    """Pull JPEG frames from an ``http://host:port/mjpeg`` multipart stream.
    Runs in a background thread; call ``.latest()`` to grab the most recent
    decoded BGR frame or None if not ready yet."""
    def __init__(self, url: str):
        self.url = url
        self._frame = None
        self._last_frame_t = 0.0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self):
        boundary = b"--frame"
        n_frames = 0
        n_reconnects = 0
        while not self._stop.is_set():
            try:
                req = urllib.request.Request(
                    self.url, headers={"User-Agent": "pose_imitator"})
                with urllib.request.urlopen(req, timeout=15) as r:
                    buf = b""
                    self._last_frame_t = time.monotonic()
                    while not self._stop.is_set():
                        # Detect stall: if no complete frame in 15 s (very
                        # generous — RPi at low fps + jpeg parse quirks),
                        # bail out so the outer loop can reconnect.
                        if time.monotonic() - self._last_frame_t > 15.0:
                            print(f"[mjpeg] stall (>15s), reconnect "
                                  f"#{n_reconnects} (had {n_frames} frames)",
                                  flush=True)
                            n_reconnects += 1
                            break
                        chunk = r.read(65536)
                        if not chunk: break
                        buf += chunk
                        while True:
                            a = buf.find(boundary)
                            if a < 0:
                                if len(buf) > 4_000_000: buf = b""
                                break
                            b = buf.find(boundary, a + len(boundary))
                            if b < 0:
                                buf = buf[a:]
                                break
                            part = buf[a:b]
                            buf = buf[b:]
                            hdr_end = part.find(b"\r\n\r\n")
                            if hdr_end < 0: continue
                            body = part[hdr_end + 4:].rstrip(b"\r\n")
                            arr = np.frombuffer(body, dtype=np.uint8)
                            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if img is not None:
                                self._frame = img
                                self._last_frame_t = time.monotonic()
                                n_frames += 1
                                if n_frames % 100 == 0:
                                    print(f"[mjpeg] {n_frames} frames total",
                                          flush=True)
            except Exception as e:
                print(f"[mjpeg] {e}, retry in 1s", flush=True)
                time.sleep(1.0)

    def latest(self):
        return self._frame

    def stop(self):
        self._stop.set()


# ── One-Euro filter ──────────────────────────────────────────────────────────
class OneEuro:
    """Simple 1-Euro low-pass filter (Casiez et al. 2012).  Great for
    smoothing hand/pose tracking without lag on fast motion."""
    def __init__(self, min_cutoff=1.0, beta=0.05, dcutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.prev_x = None
        self.prev_dx = 0.0
        self.prev_t = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: float) -> float:
        if self.prev_x is None:
            self.prev_x = x
            self.prev_t = t
            return x
        dt = max(t - self.prev_t, 1e-4)
        dx = (x - self.prev_x) / dt
        a_d = self._alpha(self.dcutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.prev_dx
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.prev_x
        self.prev_x = x_hat
        self.prev_dx = dx_hat
        self.prev_t = t
        return x_hat


# ── Vector helpers ───────────────────────────────────────────────────────────
def _pt(lm) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z], dtype=float)


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    d = float(np.dot(_norm(u), _norm(v)))
    return math.acos(max(-1.0, min(1.0, d)))


def _signed_angle(u: np.ndarray, v: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle from u to v around axis (right-hand rule)."""
    a = _angle(u, v)
    if float(np.dot(np.cross(u, v), axis)) < 0:
        a = -a
    return a


# ── Arm retargeting ──────────────────────────────────────────────────────────
def compute_arm_q(lm, shoulder_i, elbow_i, wrist_i, hip_i, other_shoulder_i):
    """Compute (J0 rot, J1 lift, J2 twist, J3 elbow, conf) for one arm.

    All landmark indices are into MediaPipe's 33-point pose_world_landmarks.
    Returns None when confidence too low.  Angles are in *raw* radians —
    the receiver applies cal_points clipping so we don't have to.
    """
    lm_s = lm[shoulder_i]; lm_e = lm[elbow_i]
    lm_w = lm[wrist_i];    lm_h = lm[hip_i]
    lm_os = lm[other_shoulder_i]
    conf = float(min(lm_s.visibility, lm_e.visibility,
                     lm_w.visibility, lm_h.visibility))
    if conf < 0.3:
        return None

    S = _pt(lm_s); E = _pt(lm_e); W = _pt(lm_w); H = _pt(lm_h); OS = _pt(lm_os)
    ua = E - S          # upper-arm vector (S → E)
    fa = W - E          # forearm vector   (E → W)
    down = H - S        # torso "down" (S → H)
    side = OS - S       # torso "sideways" (S → other shoulder)

    down_n = _norm(down)
    side_n = _norm(side)
    fwd = _norm(np.cross(side_n, down_n))    # torso forward normal
    ua_n = _norm(ua)
    fa_n = _norm(fa)

    # J3 elbow flex: 0 straight ↔ π fully folded.  Using angle_between(ua,
    # fa) directly — small when arm extended, large when folded.  (Earlier
    # `pi - angle` was inverted and made elbow track backwards.)
    elbow = _angle(ua, fa)

    # J1 shoulder lift: angle between upper-arm and torso down
    #    0 = arm hanging, π/2 = horizontal, π = raised straight up
    lift = _angle(ua_n, down_n)

    # J0 shoulder rot: azimuth in plane perpendicular to side axis
    #    project upper-arm onto (down, fwd) plane, measure vs fwd
    ua_in_sag = ua_n - float(np.dot(ua_n, side_n)) * side_n
    rot = _signed_angle(fwd, _norm(ua_in_sag), side_n)

    # J2 arm twist: forearm rotation around upper-arm axis
    #    reference = torso side projected onto plane perp to upper-arm
    fa_perp = fa_n - float(np.dot(fa_n, ua_n)) * ua_n
    ref     = side_n - float(np.dot(side_n, ua_n)) * ua_n
    if np.linalg.norm(fa_perp) < 1e-3 or np.linalg.norm(ref) < 1e-3:
        twist = 0.0
    else:
        twist = _signed_angle(_norm(ref), _norm(fa_perp), ua_n)

    return (rot, lift, twist, elbow, conf)


# ── Main loop ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0,
                    help="v4l2 index for local webcam (ignored if --url)")
    ap.add_argument("--url", default="",
                    help="MJPEG URL, e.g. http://192.168.0.123:8000/mjpeg — "
                         "when set, read from RPi camera instead of local cam")
    ap.add_argument("--target", default="127.0.0.1:9999")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--rate", type=float, default=25.0,
                    help="max UDP send rate (Hz)")
    ap.add_argument("--mirror", action="store_true", default=True,
                    help="swap L/R output so user's arm maps to the visually-"
                         "matching robot arm (default ON — natural puppet mode)")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false")
    args = ap.parse_args()

    host, sport = args.target.split(":")
    port = int(sport)
    port_img = int(sport) - 1     # 9998 for JPEG frames (paired with 9999)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Use larger send buffer for the JPEG channel
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    # Choose source: MJPEG URL (RPi) or local webcam.  A dedicated reader
    # thread continuously calls cap.read() and overwrites a shared "latest"
    # slot; the main loop always picks up the freshest frame.  This avoids
    # FFmpeg's internal MJPEG buffer accumulating (which was collapsing our
    # effective fps to 0.5 as backlog piled up).
    if args.url:
        print(f"[pose] reading MJPEG from {args.url} via OpenCV")
        cap = cv2.VideoCapture(args.url)
    else:
        print(f"[pose] opening local webcam /dev/video{args.cam}")
        cap = cv2.VideoCapture(args.cam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        print("[pose] failed to open capture source")
        return

    latest = {"frame": None, "ts": 0.0}
    stop_reader = threading.Event()
    def _reader():
        while not stop_reader.is_set():
            ok, f = cap.read()
            if ok and f is not None:
                latest["frame"] = f
                latest["ts"] = time.monotonic()
            else:
                time.sleep(0.02)
    threading.Thread(target=_reader, daemon=True).start()

    # One-Euro filter per (arm, joint)
    filt = {a: [OneEuro(min_cutoff=1.2, beta=0.02) for _ in range(4)]
            for a in ("L", "R")}

    period = 1.0 / max(1.0, args.rate)
    last_send = 0.0
    last_jpeg_send = 0.0
    jpeg_period = 1.0 / 12.0        # 12 fps cap for the JPEG preview
    n_sent = 0
    n_jpeg = 0
    n_processed = 0
    last_stat_t = time.monotonic()
    print(f"[pose] streaming to UDP {host}:{port} at {args.rate:.1f} Hz",
          flush=True)

    # MediaPipe landmark indices (subject-perspective naming)
    #   left shoulder=11, elbow=13, wrist=15, hip=23
    #   right shoulder=12, elbow=14, wrist=16, hip=24
    # After horizontal frame flip (mirror), subject's LEFT arm appears on
    # image's LEFT — but MediaPipe re-labels correctly so we still use 11/13
    # etc. for subject's left.

    last_ok_t = time.monotonic()
    last_ts_seen = 0.0
    while True:
        # Wait for a fresh frame from the reader thread (skip stale)
        if latest["ts"] == last_ts_seen:
            if time.monotonic() - last_ok_t > 10.0:
                print("[pose] no fresh frame in 10s, exiting for restart",
                      flush=True)
                stop_reader.set(); break
            time.sleep(0.01); continue
        last_ts_seen = latest["ts"]
        frame = latest["frame"]
        if frame is None:
            time.sleep(0.02); continue
        last_ok_t = time.monotonic()
        frame = cv2.flip(frame, 1)          # mirror so screen matches user
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        payload = {}
        # Attach 2D image-space landmarks so the GUI can draw the skeleton
        # overlay on its own camera panel (each entry is [x, y, visibility]
        # normalised to 0..1 image coords).
        if result.pose_landmarks:
            lm2d = result.pose_landmarks.landmark
            payload["lm2d"] = [[float(p.x), float(p.y), float(p.visibility)]
                               for p in lm2d]
        if result.pose_world_landmarks:
            lm = result.pose_world_landmarks.landmark
            t = time.monotonic()
            L = compute_arm_q(lm, 11, 13, 15, 23, 12)
            R = compute_arm_q(lm, 12, 14, 16, 24, 11)
            # Optionally swap L↔R output so subject's arm maps to the
            # visually-adjacent robot arm (natural puppet / mirror feeling).
            # This is the case both for laptop-webcam (user faces cam) and
            # robot-head-cam (robot faces user) — both are mirror views.
            mapping = (("R", L), ("L", R)) if args.mirror else (("L", L), ("R", R))
            for arm, res in mapping:
                if res is None:
                    continue
                rot, lift, twist, elbow, conf = res
                q = [
                    filt[arm][0].filter(rot,   t),
                    filt[arm][1].filter(lift,  t),
                    filt[arm][2].filter(twist, t),
                    filt[arm][3].filter(elbow, t),
                    0.0,   # J4 wrist — no source yet
                ]
                payload[arm] = {"q": q, "conf": conf}

        now = time.monotonic()
        n_processed += 1
        if payload and now - last_send >= period:
            try:
                # Payload can exceed 8KB with 33 landmarks — set send buffer
                # up to 64KB to avoid EMSGSIZE.
                sock.sendto(json.dumps(payload).encode(), (host, port))
                last_send = now
                n_sent += 1
            except Exception as e:
                print(f"[pose] send err: {e}", flush=True)
        # Send preview JPEG to arm GUI (port 9998).  Always send at 12fps
        # regardless of pose detection so the operator can see whatever
        # pose_imitator sees (camera checking / user positioning).
        if now - last_jpeg_send >= jpeg_period:
            last_jpeg_send = now
            annotated = cv2.resize(frame, (320, int(320 * frame.shape[0]
                                                    / frame.shape[1])),
                                    interpolation=cv2.INTER_AREA)
            h_disp, w_disp = annotated.shape[:2]
            if result.pose_landmarks is not None:
                # overlay skeleton when detected
                conns = [(11,13),(13,15),(12,14),(14,16),(11,12),(11,23),
                         (12,24),(23,24)]
                lm2d_pts = result.pose_landmarks.landmark
                for a, b in conns:
                    pa, pb = lm2d_pts[a], lm2d_pts[b]
                    if pa.visibility > 0.3 and pb.visibility > 0.3:
                        cv2.line(annotated,
                                 (int(pa.x*w_disp), int(pa.y*h_disp)),
                                 (int(pb.x*w_disp), int(pb.y*h_disp)),
                                 (0, 255, 0), 2)
                for p in lm2d_pts[:25]:
                    if p.visibility > 0.3:
                        cv2.circle(annotated,
                                   (int(p.x*w_disp), int(p.y*h_disp)),
                                   3, (0, 200, 255), -1)
                # text: "POSE OK"
                cv2.putText(annotated, "POSE OK", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)
            else:
                # no detection — annotate red
                cv2.putText(annotated, "NO POSE", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1)
            ok_enc, jpg = cv2.imencode(".jpg", annotated,
                                        [cv2.IMWRITE_JPEG_QUALITY, 55])
            if ok_enc:
                jbytes = jpg.tobytes()
                if len(jbytes) < 60000:
                    try:
                        sock.sendto(jbytes, (host, port_img))
                        n_jpeg += 1
                    except Exception as e:
                        print(f"[pose] jpeg send err: {e}", flush=True)
        if now - last_stat_t > 2.0:
            print(f"[pose] proc={n_processed}fps="
                  f"{n_processed/(now-last_stat_t):.1f}  "
                  f"json={n_sent}  jpeg={n_jpeg}", flush=True)
            n_processed = 0; n_sent = 0; n_jpeg = 0
            last_stat_t = now

        if args.show:
            if result.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # overlay joint targets
            y = 20
            for arm, data in payload.items():
                txt = f"{arm}: " + " ".join(f"{math.degrees(v):+6.1f}"
                                            for v in data["q"][:4])
                cv2.putText(frame, txt, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 0), 2)
                y += 22
            cv2.imshow("pose_imitator (ESC to quit)", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    if cap is not None: cap.release()
    if mjpeg is not None: mjpeg.stop()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
