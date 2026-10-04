"""
train_eval.py — 訓練臺灣行人號誌偵測模型，並用今天的實地照片考試
==================================================================
  python tools/pedlight/train_eval.py train      訓練（需要先關掉伺服器釋放顯示卡）
  python tools/pedlight/train_eval.py eval <權重>  用 results/field_1004（沒參與訓練）考試
考試規則：只要有任何一張「實際紅燈」被說成綠燈，就不及格，不放進系統。
"""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
BASE = os.path.join(ROOT, "dataset", "pedlight")
FIELD = os.path.join(ROOT, "results", "field_1004")


def train(epochs=60, imgsz=1536, batch=6):
    from ultralytics import YOLO
    sys.path.insert(0, os.path.dirname(__file__))
    from build_dataset import build
    data = build()
    model = YOLO(os.path.join(ROOT, "yolov8n.pt"))
    model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, project=os.path.join(BASE, "runs"),
                name="ped", exist_ok=True, patience=15, fliplr=0.5, hsv_h=0.0, workers=2, verbose=False)
    w = str(model.trainer.best)                     # 直接問訓練器存在哪，不猜路徑
    print("模型：", w)
    return w


def evaluate(weights):
    from PIL import Image
    from app import ped_detector as P
    P._WEIGHTS = type(P._WEIGHTS)(weights)
    P._MODEL = None
    truth = json.load(open(os.path.join(ROOT, "results", "field_1004_labels.json"), encoding="utf-8"))
    cache = {}
    for n, tv in sorted(truth.items()):
        fn = glob.glob(os.path.join(FIELD, n + "_*.jpg"))[0]
        cache[n] = (tv, P.detect(Image.open(fn).convert("RGB")))
    print(f"{'紅門檻':>6} {'綠門檻':>6} | 紅燈答對 | 綠燈答對 | 紅說成綠 | 看不到號誌時亂說")
    best = None
    for rmin in (0.3, 0.45, 0.6):
        for gmin in (0.45, 0.6, 0.75, 0.9):
            stat = {"red": [0, 0], "green": [0, 0]}
            false_green, unknown_said = 0, 0
            for n, (tv, dets) in cache.items():
                v = P.decide(dets, rmin, gmin)
                if tv in stat:
                    stat[tv][1] += 1
                    stat[tv][0] += v == tv
                    false_green += tv == "red" and v == "green"
                elif v in ("red", "green"):
                    unknown_said += 1
            r, g = stat["red"], stat["green"]
            print(f"{rmin:>6} {gmin:>6} | {r[0]:>2}/{r[1]}    | {g[0]:>2}/{g[1]}    | {false_green:>4}     | {unknown_said}")
            # 及格：沒有把紅燈說成綠燈，而且綠燈至少答對 1/3（不然對使用者沒有幫助）
            if false_green == 0 and g[0] >= g[1] / 3 and (best is None or r[0] + g[0] > best[0]):
                best = (r[0] + g[0], rmin, gmin)
    if best:
        print(f"→ 及格。建議門檻：紅 {best[1]}、綠 {best[2]}（總答對 {best[0]} 張）")
    else:
        print("→ 不及格：會把紅燈說成綠燈，或綠燈認得太少，不放進系統")
    return best


if __name__ == "__main__":
    if sys.argv[1] == "train":
        evaluate(train())
    else:
        evaluate(sys.argv[2])
