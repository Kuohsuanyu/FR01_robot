# Q-BOT URDF（完整機器人）

由 `QBOT_MJCF/qbot.xml` 用 **mjcf-urdf-simple-converter** 轉出,扭力(effort)再依實際馬達修正。

## 結構
```
QBOT_URDF/
├── qbot.urdf            # URDF(相對 mesh 路徑)
└── meshes/              # converted_*.obj + .mtl（黑色已烘入），共 63 件
```

## 統計
- 載入後：21 個可動關節(revolute)+ 固定連結。
- 關節限位、扭力(effort)、軸向、慣性都已保留。

## 關節限位 + 扭力(effort = 馬達扭力上限 N·m)

| 部位 | 限位 | effort |
|---|---|---|
| 腿 hip_pitch / knee (RS04) | kbot 內建 | **120** |
| 腿 hip_roll / hip_yaw (RS03) | kbot 內建 | **60** |
| 腿 ankle (RS02) | kbot 內建 | **17** |
| 手臂 shoulder/lateral_raise/arm_twist | ±90° | **0.98**(10 kg·cm) |
| 手臂 elbow | 0~135° | **0.98** |
| 脖子 pitch/roll | ±30° | **0.98** |
| 脖子 yaw | ±90° | **0.98** |

velocity 暫設:腿 20、伺服 6 rad/s(可自行調整)。

## 開啟 / 驗證
```python
import mujoco
m = mujoco.MjModel.from_xml_path("qbot.urdf")   # MuJoCo 可直接載入
```
也可用 RViz / urdf-viewer 等標準 URDF 工具開啟(mesh 為相對路徑的 .obj)。

## 注意
- 轉換器把浮動 base 當固定根(URDF 慣例),根連結 = `Torso_Side_Right`。
- 轉換器預設 effort=100,已用腳本依馬達型號改成上表數值。
- 來源 MJCF 在 `../QBOT_MJCF/qbot.xml`。
