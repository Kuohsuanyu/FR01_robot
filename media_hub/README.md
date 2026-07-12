# FR01 Media Hub

單一進程獨占相機、一次擷取多方消費。解決 **多 agent 搶鏡頭** 與 **MJPEG 串流不順**。

架構參考 [Open-TeleVision](https://github.com/OpenTeleVision/TeleVision) 的
single-owner + WebRTC 概念,擷取端改用 `cv2`(相容你現有 `head_rpi_agent` 的相機設定)。

```
                       ┌──────────── media_hub.py（唯一開相機）───────────┐
   /dev/video0 ───────▶│  Camera thread → FrameBus（最新一幀）            │
                       │        ├── CameraTrack ──WebRTC/H264/VP8──▶ VR 頭盔 / 瀏覽器（可多 client）
                       │        ├── /mjpeg /snapshot ───────────────▶ 既有 PC client（相容）
                       │        └── shared_memory('fr01_cam') ───────▶ 本機其他 agent（讀 raw BGR）
                       └──────────────────────────────────────────────────┘
   motor agents（head/arm/exo）維持原樣，只是「不再自己開相機」，改讀 shm
```

## 執行(本機測試)

```bash
python3 media_hub.py --cam 0 --width 640 --height 480 --fps 30 --port 8090
```

- 瀏覽器看 WebRTC:`http://<ip>:8090/`
- 舊 MJPEG:`http://<ip>:8090/mjpeg`
- 健康檢查(含 shm 名稱):`http://<ip>:8090/status`

另一個 agent 讀影格(證明不搶鏡頭):
```bash
python3 frame_client.py --hub http://127.0.0.1:8090
```

## 端點

| 端點 | 用途 |
|------|------|
| `GET /` | WebRTC 測試檢視頁 |
| `POST /offer` | WebRTC SDP 交換(每 client 一條 PeerConnection,共用同一份影格) |
| `GET /mjpeg` | MJPEG multipart(相容舊 client) |
| `GET /snapshot` | 單張 JPEG |
| `GET /status` | JSON:相機狀態 + `shm_name`/`shm_shape` + `webrtc_clients` |

## 怎麼把現有 agent 接進來

**改動原則:相機的擁有權從各 agent 收攏到 media_hub;agent 只讀不開。**

1. **head_rpi_agent**:拿掉自己的 `Camera` 類與 `/mjpeg`,VR 頁面的影像來源改指到 media_hub
   的 WebRTC(`WebRTCStereoVideoPlane` 或 `<video>`)。馬達 / `/ws` / `/status` 全部維持不動。
2. **需要影像的 agent**(手部視覺、錄影、AI 推論):用 `frame_client.py` 的方式
   `shared_memory.SharedMemory(name=meta['shm_name'])` attach,`np.ndarray` 讀 raw BGR。
3. **多顆相機**:每顆相機起一個 media_hub 實例(不同 `--port` 與 `--shm-name`)。

## 搬到樹莓派 / Jetson 前要驗證

- **編碼負載**:aiortc 預設軟體編碼。RPi 上先用 `--width 640 --height 480 --fps 15`
  觀察 CPU;若吃不消,降解析度或改走 Jetson(有 NVENC 硬體編碼,最順)。
- Jetson Nano 建議:優先讓 aiortc 走 H264 硬體編碼(NVENC),延遲/CPU 都最佳。
