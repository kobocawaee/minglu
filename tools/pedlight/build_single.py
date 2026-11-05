"""
build_single.py — 單一類別資料集（只學「行人號誌在哪裡」，顏色交給 HSV）
=====================================================================
為什麼：第一版模型（紅／綠兩類）會「記住路口」— v2/v3/v7 的標註全是綠燈，
同一個路口紅燈時模型說綠燈、信心高達 0.9。改成只學位置，顏色由 app/ped_detector.lamp_color 判斷。

標註來源：
  1. labels/<影片>.json — 原本人工錨點＋追蹤的標註（只涵蓋每支影片開頭，追丟就停）
  2. 偽標註 — 舊模型（models/pedlight.pt）在其餘畫面上的框，條件：
       信心 ≥0.25、HSV 看得出亮紅或亮綠、框的大小像行人號誌（過濾整片紅色路面標線之類的大框）

用法：python tools/pedlight/build_single.py    → dataset/pedlight/yolo_single/
"""

import glob
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
BASE = os.path.join(ROOT, "dataset", "pedlight")
RAW, LAB = os.path.join(BASE, "raw"), os.path.join(BASE, "labels")
OUT = os.path.join(BASE, "yolo_single")
VIDEOS = ["v1", "v2", "v3", "v4", "v5", "v6", "v7"]
VAL = "v4"          # 只用來看訓練過程，最後的考試是 results/field_1004（不同路口、沒參與訓練）
PSEUDO_CONF = 0.25


def _plausible(box, W):
    """行人號誌的框：不能太大（畫面寬 15% 以內），而且是直的或接近正方形"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return 6 <= w <= 0.15 * W and 6 <= h and 0.8 <= h / w <= 4.0


def build(every=2):
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO
    from app.ped_detector import lamp_color

    model = YOLO(os.path.join(ROOT, "models", "pedlight.pt"))
    shutil.rmtree(OUT, ignore_errors=True)
    stats = {}
    for vid in VIDEOS:
        split = "val" if vid == VAL else "train"
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(OUT, sub, split), exist_ok=True)
        labels = {int(k): v for k, v in json.load(open(os.path.join(LAB, vid + ".json"))).items()}
        frames = sorted(glob.glob(os.path.join(RAW, vid, "*.jpg")))
        n_human = n_pseudo = 0
        for i in range(0, len(frames), every):
            im = Image.open(frames[i]).convert("RGB")
            W, H = im.size
            if i in labels:
                boxes = [labels[i][:4]]
                n_human += 1
            else:
                rgb = np.asarray(im)
                r = model.predict(im, imgsz=1536, conf=PSEUDO_CONF, verbose=False)[0]
                boxes = [b for b in ([float(v) for v in bb.xyxy[0]] for bb in r.boxes)
                         if _plausible(b, W) and lamp_color(rgb, b)]
                if not boxes:
                    continue                    # 沒把握的畫面不放（可能有號誌但沒偵測到，當背景會教錯）
                n_pseudo += 1
            name = f"{vid}_{i:05d}"
            shutil.copy(frames[i], os.path.join(OUT, "images", split, name + ".jpg"))
            with open(os.path.join(OUT, "labels", split, name + ".txt"), "w") as f:
                for x1, y1, x2, y2 in boxes:
                    f.write(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                            f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
        stats[vid] = (split, n_human, n_pseudo)
        print(f"{vid} ({split}): 人工 {n_human}、偽標註 {n_pseudo}", flush=True)
    yaml = os.path.join(OUT, "data.yaml")
    with open(yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {OUT.replace(os.sep, '/')}\ntrain: images/train\nval: images/val\n"
                "names:\n  0: ped_signal\n")
    return yaml


if __name__ == "__main__":
    build()
