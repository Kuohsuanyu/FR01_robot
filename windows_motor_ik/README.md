# windows_motor_ik — Windows 手臂伺服控制 + IK 工具

**Windows 獨立工具(無 ROS)**,從 Red-Rabbit `rx1_motor` 移植,透過 USB COM 串口
直接控制 Feetech STS/SCS 伺服,含逐關節測試與雙臂 IK 圖形介面。用於在 Windows
開發機上對手臂做 bring-up / 測試。

## 檔案
| 檔案 | 用途 |
|---|---|
| `feetech_lib.py` | Feetech SCServo 協定(純 Python `pyserial`,0xFF 0xFF 封包) |
| `rx1_motor_win.py` | 馬達控制器:關節角↔伺服步數(含 gear/dir),`Rx1Motor("COM8")` |
| `demo.py` | CLI 互動 demo:`python demo.py COM8`(home / right_arm / head …) |
| `joint_test_gui.py` | 逐關節測試 GUI(分頁 左臂/右臂/頭/軀幹/手掌,滑桿 + ±30° 掃描) |
| `gui.py` | 左臂視覺化控制 GUI |
| `arm_ik_gui.py` | **雙臂 IK 控制 + 3D 視窗**(LMB 拖曳末端執行器、RMB 旋轉、滾輪縮放) |
| `arm_ik_config.json` | 關節設定:每軸 `sid / dir / gear / label / 限位 / mode / ticks` + COM port |

## 安裝(Windows / Anaconda)
```bash
pip install -r requirements.txt        # pyserial
pip install numpy scipy matplotlib     # arm_ik_gui.py / gui.py 需要
```

## 使用
```bash
python demo.py COM8                     # CLI 快速測試
python joint_test_gui.py               # 逐關節測試(先確認每顆馬達)
python arm_ik_gui.py                   # 雙臂 IK 圖形控制
```
COM port 在 `arm_ik_config.json` 或命令列指定。

## ⚠️ 與 Q-BOT 的差異(重要)
- 這套是 **RX1 的設定**:`arm_ik_config.json` 用 **gear=3**(RX1),且含頭/軀幹/手掌。
- **Q-BOT 的肩部是 gear=4**(且只有肩 3 軸減速、手肘直驅)。要用在 Q-BOT 上,
  需把 `arm_ik_config.json` 的 `sid / dir / gear / 限位` 改成 Q-BOT 實機值
  (對照 `../upper_body_control/config.py`)。
- 換算邏輯與 `../upper_body_control/`(部署端 Linux 版)一致:
  `steps = angle/π × 2048 × gear × dir + offset`。

## 與專案其他部分的關係
- `../upper_body_control/` — Linux/部署端的 Feetech 手臂驅動(精簡、給 policy 即時控制用)
- `../rx1/`(未入庫)— 原始 RX1 ROS 套件(此工具的移植來源)
- `../deployment/` — 完整實機部署(腿 Robstride + IMU)
