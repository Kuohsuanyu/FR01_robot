# 腿部 Policy 上傳 / 驗證 / 部署

訓練好的腿部模型(`.kinfer`)透過這個資料夾流轉。**這裡會進 git**(和被排除的龐大
`leg/robot_data/` 分開),`.kinfer` 檔小(多半 <4 MB),適合版本控管。

## 流程

```
 [顯卡主機] 訓練 → 產生 xxx.kinfer
      │  放到 incoming/,git push
      ▼
 [這台 PC] git pull → verify_kinfer.py 驗證
      │  通過 → 移到 verified/,git push
      ▼
 [leg RPi] deploy_leg_policy.sh → scp + 重指 model.kinfer → 重啟 firmware 運行
```

## 資料夾

| 路徑 | 用途 |
|------|------|
| `incoming/` | 你從顯卡主機上傳**新訓練**的 `.kinfer` 放這 |
| `verified/` | 驗證通過、可安全部署的 `.kinfer` |
| `reference_spec.json` | 驗證基準(關節名稱/指令數;取自已知可運行的 policy) |
| `verify_kinfer.py` | 驗證單顆 `.kinfer` |
| `deploy_leg_policy.sh` | 部署到 leg RPi(會先驗證) |

## 各步驟指令

**① 顯卡主機(你)— 上傳新模型**
```bash
cp <訓練輸出>/model.kinfer  leg/policies/incoming/walk_v42.kinfer
git add leg/policies/incoming/walk_v42.kinfer && git commit -m "leg policy walk_v42" && git push
```

**② 這台 PC(我)— 拉下來驗證**
```bash
git pull
python3 leg/policies/verify_kinfer.py leg/policies/incoming/walk_v42.kinfer
# 通過後移到 verified/ 並 push
git mv leg/policies/incoming/walk_v42.kinfer leg/policies/verified/
git commit -m "leg policy walk_v42 verified" && git push
```
驗證會檢查:合法 kinfer 壓縮包、含 `init_fn.onnx`/`step_fn.onnx`/`metadata.json`、
`joint_names` 與基準**完全一致**(順序也要對)、`num_commands` 相符;
裝了 `onnx` 還會檢查兩個模型可載入。

**③ leg RPi — 下載並運行**
```bash
# leg RPi 開機後(用 fr01-leg.local 名稱,IP 免管):
leg/policies/deploy_leg_policy.sh leg/policies/verified/walk_v42.kinfer
# → scp 到 ~/robot_data/policy/FR01/,並把 model.kinfer symlink 指向它
# 然後在 leg RPi 上重啟 firmware 生效
```
或 leg RPi 端自己 `git pull` 後跑 deploy(兩種都行)。

## 更新驗證基準
換了機器人 DOF / 指令介面時,用新的已知可運行 policy 重產基準:
```bash
python3 -c "import tarfile,json; t=tarfile.open('verified/<好的>.kinfer'); \
m=json.load(t.extractfile('metadata.json')); \
json.dump({'required_members':['init_fn.onnx','step_fn.onnx','metadata.json'], \
'joint_names':m['joint_names'],'num_joints':len(m['joint_names']), \
'num_commands':m['num_commands']}, open('reference_spec.json','w'), ensure_ascii=False, indent=2)"
```
