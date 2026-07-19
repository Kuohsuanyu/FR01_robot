# FR01 VLA 資料採集

錄製「影像序列 + 對應手臂關節姿勢」的 episode,給 VLA(Vision-Language-Action)訓練用。

## 執行
```bash
python3 -m pip install -r vla_collect/requirements.txt   # 首次
python3 vla_collect/vla_collect.py \
    --cam http://192.168.0.123:8000/snapshot \
    --arm 192.168.0.123:8100 --fps 15
```
（相機也可用 media_hub:`http://<host>:8090/snapshot`）

## 操作
1. 上方填**任務名稱**(如 `pick`),右上顯示「下一個」自動遞增的 episode 名
2. 確認相機有畫面、手臂已連(按鈕變綠)
3. 按 **● 開始錄製** → 依 fps 連續存影像+姿勢 → 按 **■ 停止**
4. 每段自動存成 `data/<名稱><NN>/`,NN 從 01 遞增(pick01, pick02, ...)

## 每個 episode 資料夾
```
data/pick01/
├── meta.json          任務名 / fps / 幀數 / 關節名 / 時長
├── frames/frame_00000.jpg ...   每幀影像
└── trajectory.jsonl   每幀一行:{t, frame, q:{joint:rad}, ticks:{sid:tick}}
```
- `q`:手臂關節角度(rad,用 arm/qbot_arm_calibration.json 從 tick 換算)
- `ticks`:原始馬達 tick(保留給不同用途)

## 注意
- 姿勢來源是 **arm agent /ws 遙測**;需先啟動手臂 agent(:8100)
- `data/` 已 gitignore(影像大);資料自行備份
