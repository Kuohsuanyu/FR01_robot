# 從 USB 隨身碟複製過來的參考資料

**這裡的檔案是唯讀參考,不要當作正在維護的程式碼。**

需要用時直接讀,不會被 `hands_control/` 主流程 import。

## 檔案清單

### 巧手直接相關

| 檔案 | 來源 | 用途 |
|---|---|---|
| `old_hand_gui_range.py` | `USB:.../sms_sts/test2.py` | **最重要** — 舊 6 顆巧手 (ID 11-16) 的 tkinter GUI + 每顆 `(open_tick, close_tick)` 校正值。目前的 ID 31-36 EPROM min/max 與此完全對得起來(只是 ID 重編)。 |
| `old_move_single_id.py` | `USB:.../sms_sts/test1.py` | 用原始封包(不透過 SDK)控單顆馬達到指定位置的最小範例。 |
| `old_feetech_scan.py` | `USB:/feetech_scan.py` | 舊掃描工具,示範用序號固定的 by-id 串口路徑 + 多 baud 輪詢。 |

### SCS009 SDK 範例(model 1029 用這組)

`scscl_examples/` 目錄:

- `ping.py`, `read.py`, `read_write.py`, `write.py`, `sync_write.py`
- 來源:USB 上 `FTServo_Python/scscl/`
- 為什麼放這:巧手上 4 顆 model 1029 = SCS009,byte order 是 big-endian、位置 10-bit,
  跟 SMS/STS 不同。要控制 SCS009 需用 `from scservo_sdk import scscl`,不是 `sms_sts`。
  這些範例是 Feetech 官方 SDK 附帶的最小可用程式碼。

## 舊 SERVOS 對照表(供 config 撰寫參考)

```python
# 出自 old_hand_gui_range.py,舊 ID 11-16
# (low_tick, high_tick) — slider 0→1 時的目標 tick 值
# low>high 表示裝配方向反相
SERVOS = {
    11: (515, 780),
    12: (518, 285),
    13: (430, 700),
    14: (3400, 2700),
    15: (750, 500),
    16: (2660, 1980),
}
```

現在的 ID 31-36 對應:11→31, 12→32, 13→33, 14→34, 15→35, 16→36
(EPROM min/max 與上表數值一致,詳見上層 `../README.md`)
