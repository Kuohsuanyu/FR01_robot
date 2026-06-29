# 機器人頭部控制（VR 頭追雲台）— 部署版

可攜式整包：複製整個資料夾到另一台電腦即可運作。
手機戴 VR（Cardboard）→ 顯示機器人攝影機畫面 + 頭部轉向控制三軸脖子雲台。

## 內容物

| 檔案 | 用途 |
|---|---|
| `server.py` | 主程式：WebRTC 影像 + IMU 收發 + 雲台控制 + /status |
| `index.html` | 手機網頁（VR 雙眼、感測器、校準模式） |
| `neck_gimbal.py` | 三軸雲台控制（絕對跟隨＋濾波＋限速＋軟限位），自動偵測 CH343 |
| `feetech_lib.py` | Feetech STS3215 伺服驅動 |
| `manifest.json` / `icon-*.png` | PWA（iPhone 加入主畫面可全螢幕） |
| `cloudflared` | 外網隧道（免費、免帳號）— **未含在 repo，需自行下載**，見下 |
| `啟動_伺服器.bat` / `啟動_外網隧道.bat` | 一鍵啟動（Windows；Linux 見下方指令） |
| `requirements.txt` | Python 套件清單 |

## 新電腦第一次設定

1. 裝 **Python 3.10+**（安裝時勾選 Add to PATH）。
2. 在本資料夾開命令列，安裝套件：
   ```
   pip install -r requirements.txt
   ```
3. 接上硬體：
   - **USB 攝影機**（機器人頭上的）
   - **CH343 USB 轉接器**（接三顆 Feetech STS3215，ID 1/2/3）——程式會自動找它的 COM 埠
4. **下載 cloudflared**（要外網才需要，未含在 repo）：
   - Windows：<https://github.com/cloudflare/cloudflared/releases/latest> → `cloudflared-windows-amd64.exe`，改名為 `cloudflared.exe` 放本資料夾
   - Linux：下載 `cloudflared-linux-amd64`（或對應架構），`chmod +x` 後改名 `cloudflared`

## 每次啟動

### Windows
1. 雙擊 **`啟動_伺服器.bat`**（連攝影機/雲台、開 https 服務；方向不對改 bat 裡的 `--rotate`）
2. 要外網：雙擊 **`啟動_外網隧道.bat`** → 印出 `https://xxxx.trycloudflare.com`
3. 區網用：直接開 `https://<這台電腦IP>:8443`（憑證警告選繼續）

### Linux（小電腦）
```bash
python3 server.py --rotate 270            # 伺服器
./cloudflared tunnel --url https://localhost:8443 --no-tls-verify   # 另開一個終端
```

## 手機操作

- 開網址 → **進入 VR**（轉頭控制雲台 + 看畫面），或 **校準模式**（接雲台前先確認方向正負）。
- iPhone 要全螢幕：Safari 分享 → 加入主畫面 → 從圖示開啟。
- 雙擊畫面 = 歸正。

## 雲台設定（`neck_gimbal.py` 最上面 AXES）

實機限位（tick，中心 2047）：
- ID1 左右轉(yaw)：1000–3000，越低=往左轉
- ID2 上下(pitch)：1500–3000，1500=低頭最下
- ID3 側傾(roll)：1500–2500，越低=向右傾

裝反就改該軸的 `dir`（+1 / -1）。`python neck_gimbal.py --check` 只讀不動、確認連線；`--demo` 展示擺動。

## 常見問題

- **攝影機打不開**：換 `--camera 1`；或被卡住時拔插 USB（程式會自動重連）。
- **雲台連不到**：確認 CH343 有插穩（裝置管理員看得到 COM 埠）；程式會自動偵測埠號。
- **手機讀不到 IMU**：一定要走 `https`（外網用 cloudflared 的網址；iOS 要在點擊時授權感測器）。
- **外網連得上但沒畫面**：表示 P2P 沒打通（NAT 限制）。本機有公網 IP 的話設 port forward；否則需 TURN 中繼。
