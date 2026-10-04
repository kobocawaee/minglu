"""
ped_detector.py — [資服版] 臺灣行人號誌偵測（以 YOLOv8n 用臺灣路口影片微調）
============================================================================
為什麼需要：LYTNetV2 用上海的號誌訓練，YOLOv8n 原本的 traffic light 類別又分不出車輛號誌與
行人號誌；實地測試 3 個臺灣路口 52 張有燈號的畫面，綠燈 0 張答對（results/field_1004）。
這個模型只學兩類：行人紅燈、行人綠燈，車輛號誌是背景。

安全規則（寧可不說，不可說錯）：
  - 同一張畫面同時看到行人紅燈和行人綠燈 → 不判斷（可能是不同方向的號誌，分不出是哪一個）
  - 綠燈的信心門檻比紅燈高（誤報綠燈會讓使用者走進車流）
  - 沒偵測到 → 回傳 None，交回原本 LYTNet 的流程（原流程只會說紅燈或無法確定，不會更危險）
模型檔不存在時 available() 為 False，系統行為和以前完全一樣。
"""

from pathlib import Path

from app import config
from app.messages import t

_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "pedlight.pt"
_MODEL = None


def available() -> bool:
    return getattr(config, "PED_DETECTOR", True) and _WEIGHTS.exists()


def _model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(str(_WEIGHTS))
    return _MODEL


def detect(image, conf=0.2):
    """回傳 [(類別 'red'/'green', 信心, 框)]，框是原圖座標。"""
    r = _model().predict(image, imgsz=config.PED_IMGSZ, conf=conf, verbose=False)[0]
    out = []
    for b in r.boxes:
        c = int(b.cls[0])
        out.append(("red" if c == 0 else "green", float(b.conf[0]), [float(v) for v in b.xyxy[0]]))
    return out


def decide(dets, red_min=None, green_min=None):
    """依安全規則把偵測結果變成 'red' / 'green' / 'conflict' / None（沒看到）。"""
    red_min = config.PED_RED_MIN if red_min is None else red_min
    green_min = config.PED_GREEN_MIN if green_min is None else green_min
    reds = [d for d in dets if d[0] == "red" and d[1] >= red_min]
    greens = [d for d in dets if d[0] == "green" and d[1] >= green_min]
    weak_red = any(d[0] == "red" and d[1] >= 0.2 for d in dets)
    if greens and weak_red:            # 有一點點紅燈的跡象就不說綠燈
        return "conflict"
    if reds and greens:
        return "conflict"
    if greens:
        return "green"
    if reds:
        return "red"
    return None


def phrase(image):
    """給 street_mode 用：回傳要說的話；沒偵測到行人號誌時回傳 None（交回原流程）。"""
    v = decide(detect(image))
    if v in ("red", "green"):
        return t("light_" + v)
    if v == "conflict":
        return t("light_unclear")
    return None
