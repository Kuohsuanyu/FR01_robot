#!/usr/bin/env python3
"""驗證一顆訓練好的腿部 .kinfer 是否可安全部署到 leg RPi。

.kinfer = gzip-tar,內含 init_fn.onnx / step_fn.onnx / metadata.json。
本工具**不需 kinfer 套件**即可檢查:
  1. 是合法 kinfer 壓縮包、含必要成員
  2. metadata.json 可解析、欄位齊全
  3. joint_names 與 reference_spec.json 完全一致(避免部署到不相容機器)
  4. num_commands 相符
  5. (若裝了 onnx)兩個 onnx 模型可載入、且列出 I/O 形狀

用法:
  python3 verify_kinfer.py incoming/my_new_policy.kinfer
  python3 verify_kinfer.py incoming/my_new_policy.kinfer --ref reference_spec.json
退出碼 0 = 通過;非 0 = 有問題(細節印在上面)。
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
G = "\033[32m"; R = "\033[31m"; Y = "\033[33m"; C = "\033[36m"; Z = "\033[0m"


def fail(msg): print(f"{R}  ✗ {msg}{Z}"); return False
def ok(msg):   print(f"{G}  ✓ {msg}{Z}"); return True
def warn(msg): print(f"{Y}  ⚠ {msg}{Z}")


def verify(path: str, ref_path: str) -> bool:
    print(f"{C}驗證 {path}{Z}")
    if not os.path.isfile(path):
        return fail(f"檔案不存在:{path}")
    ref = json.load(open(ref_path, encoding="utf-8"))
    passed = True

    # 1) 合法 kinfer 壓縮包 + 必要成員
    try:
        tar = tarfile.open(path)          # 自動偵測 gzip
        members = tar.getnames()
    except Exception as e:
        return fail(f"不是合法的 kinfer(gzip-tar)壓縮包:{e}")
    for req in ref["required_members"]:
        if req in members:
            ok(f"含 {req}")
        else:
            passed = fail(f"缺少 {req}(成員:{members})") and passed

    if "metadata.json" not in members:
        return False
    # 2) metadata 可解析
    try:
        meta = json.load(tar.extractfile("metadata.json"))
    except Exception as e:
        return fail(f"metadata.json 解析失敗:{e}")

    # 3) joint_names 完全一致
    jn = meta.get("joint_names")
    if jn is None:
        passed = fail("metadata 缺 joint_names") and passed
    elif jn != ref["joint_names"]:
        passed = False
        print(f"{R}  ✗ joint_names 與基準不一致{Z}")
        rset, uset = set(ref["joint_names"]), set(jn)
        if uset - rset: print(f"      多出:{sorted(uset - rset)}")
        if rset - uset: print(f"      缺少:{sorted(rset - uset)}")
        if uset == rset: print("      (成員相同但順序不同 — 順序會影響動作對應!)")
    else:
        ok(f"joint_names 一致({len(jn)} 顆)")

    # 4) num_commands（依 enforce_num_commands 決定擋或僅提示）
    nc = meta.get("num_commands")
    if nc == ref["num_commands"]:
        ok(f"num_commands={nc}")
    elif ref.get("enforce_num_commands", True):
        passed = fail(f"num_commands={nc},基準要 {ref['num_commands']}") and passed
    else:
        warn(f"num_commands={nc}(基準 {ref['num_commands']};僅提示,依指令介面而定)")

    # 5) 可選:onnx 載入
    try:
        import onnx  # noqa
        import tempfile
        for name in ("init_fn.onnx", "step_fn.onnx"):
            data = tar.extractfile(name).read()
            with tempfile.NamedTemporaryFile(suffix=".onnx") as tf:
                tf.write(data); tf.flush()
                model = onnx.load(tf.name)
                onnx.checker.check_model(model)
            ok(f"{name} ONNX 有效")
    except ImportError:
        warn("未裝 onnx,略過模型結構檢查(pip install onnx 可啟用)")
    except Exception as e:
        passed = fail(f"ONNX 檢查失敗:{e}") and passed

    print(f"{(G+'通過 ✓' if passed else R+'未通過 ✗')}{Z}")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kinfer", help="要驗證的 .kinfer 路徑")
    ap.add_argument("--ref", default=os.path.join(HERE, "reference_spec.json"))
    args = ap.parse_args()
    sys.exit(0 if verify(args.kinfer, args.ref) else 1)


if __name__ == "__main__":
    main()
