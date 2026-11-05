"""
train_single.py — 訓練單一類別的行人號誌偵測模型（YOLOv8s），再用 results/field_1004 考試
=========================================================================================
  python tools/pedlight/train_single.py            訓練（需要先關掉伺服器釋放顯示卡）＋考試
  python tools/pedlight/train_single.py eval <權重>  只考試

和第一版的差別：
  - 單一類別（顏色交給 HSV）→ 不會再「記住路口的燈號」
  - YOLOv8s（比 v8n 大，對小東西比較好）
  - scale=0.7：訓練時號誌大小變化 0.3~1.7 倍，裁切放大／不同距離都看過
考試規則和 train_eval.py 相同：紅燈說成綠燈 = 0 才能放進系統。
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
BASE = os.path.join(ROOT, "dataset", "pedlight")


def train(epochs=80, imgsz=1536, batch=4):
    from ultralytics import YOLO
    data = os.path.join(BASE, "yolo_single", "data.yaml")
    if not os.path.exists(data):
        from build_single import build
        data = build()
    model = YOLO("yolov8s.pt")
    model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, project=os.path.join(BASE, "runs"),
                name="single", exist_ok=True, patience=30, fliplr=0.5, scale=0.7, workers=2, verbose=False)
    w = str(model.trainer.last)          # 用最後一輪，不用驗證集挑（驗證集只有一個路口）
    print("模型：", w)
    return w


if __name__ == "__main__":
    from train_eval import evaluate
    evaluate(train() if len(sys.argv) < 2 else sys.argv[2])
