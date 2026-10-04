"""
build_dataset.py — 把標註好的畫面整理成 YOLO 訓練格式
=====================================================
類別：0 = ped_red（行人紅燈），1 = ped_green（行人綠燈）
車輛號誌不標，會被當成背景，模型因此學到「那不是行人號誌」。
以影片為單位切訓練／驗證集（同一支影片不會同時出現在兩邊）。
今天實地測試的 results/field_1004 完全不拿來訓練，留作最後的考試。
"""

import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "dataset", "pedlight")
RAW, LAB, YOLO = (os.path.join(BASE, d) for d in ("raw", "labels", "yolo"))
CLASSES = ["ped_red", "ped_green"]


def build(every=2, val_every=4):
    vids = sorted(f[:-5] for f in os.listdir(LAB) if f.endswith(".json") and not f.startswith("_"))
    if not vids:
        raise SystemExit("沒有標註資料")
    shutil.rmtree(YOLO, ignore_errors=True)
    stats = {"train": [0, 0, 0], "val": [0, 0, 0]}
    for vi, vid in enumerate(vids):
        split = "val" if len(vids) >= 3 and vi % val_every == val_every - 1 else "train"
        labels = {int(k): v for k, v in json.load(open(os.path.join(LAB, vid + ".json"))).items()}
        frames = sorted(f for f in os.listdir(os.path.join(RAW, vid)) if f.endswith(".jpg"))
        for i in sorted(labels)[::every]:
            x1, y1, x2, y2, st = labels[i]
            src = os.path.join(RAW, vid, frames[i])
            from PIL import Image
            W, H = Image.open(src).size
            for sub in ("images", "labels"):
                os.makedirs(os.path.join(YOLO, sub, split), exist_ok=True)
            name = f"{vid}_{i:05d}"
            shutil.copy(src, os.path.join(YOLO, "images", split, name + ".jpg"))
            c = CLASSES.index("ped_" + st)
            with open(os.path.join(YOLO, "labels", split, name + ".txt"), "w") as f:
                f.write(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
            stats[split][0] += 1
            stats[split][1 + c] += 1
    val = "images/val" if stats["val"][0] else "images/train"     # 影片太少時沒有驗證集，先用訓練集代替
    with open(os.path.join(YOLO, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {YOLO.replace(os.sep, '/')}\ntrain: images/train\nval: {val}\n"
                f"names:\n  0: ped_red\n  1: ped_green\n")
    for s, (n, r, g) in stats.items():
        print(f"{s}: {n} 張（紅 {r}、綠 {g}）")
    return os.path.join(YOLO, "data.yaml")


if __name__ == "__main__":
    build()
