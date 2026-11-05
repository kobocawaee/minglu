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

跨畫面投票（PedHistory）：實地考試 52 張只有 8 張單張就過門檻，但低信心（0.1 上下）的框
顏色從來沒錯過 → 連續幾張都在同一個顏色上看到弱訊號，就當成證據（規則見 PedHistory.vote）。
"""

import time
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


def lamp_color(rgb, box):
    """框裡亮的是紅還是綠（HSV，和 tools/pedlight/track_label.py 標註時的判斷相同）。
    框往外擴 20%：模型的框常常偏掉半個燈頭，只框到沒亮的那一半。"""
    import cv2
    import numpy as np
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = box
    px, py = 0.2 * (x2 - x1), 0.2 * (y2 - y1)
    crop = rgb[max(0, int(y1 - py)):min(H, int(y2 + py)), max(0, int(x1 - px)):min(W, int(x2 + px))]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    lit = (v > 140) & (s > 90)
    green = int((lit & (h >= 45) & (h <= 95)).sum())
    red = int((lit & ((h <= 7) | (h >= 170))).sum())   # 不含橘色倒數數字（H 8~25）
    need = max(5, int(0.006 * crop.shape[0] * crop.shape[1]))
    if green >= need and green > 2 * red:
        return "green"
    if red >= need and red > 2 * green:
        return "red"
    return None


def detect(image, conf=0.2):
    """回傳 [(顏色 'red'/'green', 信心, 框)]，框是原圖座標。

    顏色以框裡實際亮的顏色（HSV）為準，模型的類別只當參考 — 模型會「記住路口」：
    v2/v3/v7 的訓練標註全是綠燈，同一個路口紅燈時模型說綠燈、信心高達 0.9
    （框偏到下半部沒亮的綠燈位置）。規則依風險不對稱：
      - 綠：模型說綠 而且 HSV 也是綠
      - 紅：模型或 HSV 任一說紅（說成紅燈 = 使用者白等，不危險）
      - 模型說綠但 HSV 看不出亮什麼顏色 → 不採用
    單一類別的模型（只學「行人號誌」）→ 顏色完全由 HSV 決定。"""
    import numpy as np
    m = _model()
    r = m.predict(image, imgsz=config.PED_IMGSZ, conf=conf, verbose=False)[0]
    rgb = np.asarray(image.convert("RGB"))
    single = len(m.names) == 1
    out = []
    for b in r.boxes:
        box = [float(v) for v in b.xyxy[0]]
        if box[2] - box[0] > 0.15 * rgb.shape[1]:
            continue                    # 框太大不可能是行人號誌（舊模型會把整片紅色路面標線框成紅燈）
        hsv = lamp_color(rgb, box)
        if single:
            color = hsv
        else:
            cls = "red" if int(b.cls[0]) == 0 else "green"
            color = ("green" if cls == "green" and hsv == "green"
                     else "red" if "red" in (cls, hsv) else None)
        if color:
            out.append((color, float(b.conf[0]), box))
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


_VOTE_MIN = 0.10      # 投票用的弱訊號門檻（單張的門檻見 config.PED_RED_MIN / PED_GREEN_MIN）
_VOTE_KEEP = 3        # 看最近幾張
_VOTE_NEED = 2        # 其中幾張要看到同一個顏色（含這一張）
_VOTE_WINDOW_S = 12.0 # 連續模式約 4-10 秒一張（results/field_1004）；超過就不算同一段


class PedHistory:
    """過馬路模式跨畫面的短期記憶 — 讓「連續幾張的弱訊號」也能報燈號"""

    def __init__(self):
        self._frames = []     # [(時間, 偵測結果)]

    def add(self, dets, now=None):
        now = time.time() if now is None else now
        self._frames = [f for f in self._frames if now - f[0] <= _VOTE_WINDOW_S]
        self._frames = (self._frames + [(now, dets)])[-_VOTE_KEEP:]

    def vote(self):
        """回傳 'red' / 'green' / None。條件（全部都要符合）：
          1. 這一張就要看到該顏色（號誌離開畫面或燈號改變時，不會沿用舊的結果）
          2. 最近 _VOTE_KEEP 張中至少 _VOTE_NEED 張看到該顏色
          3. 依風險不對稱：
             green → 時間窗內任何一張都不能有紅燈的弱訊號（誤報綠燈 = 可能致命）
             red   → 只要這一張沒有綠燈的弱訊號（誤報紅燈 = 使用者白等，不危險）
        """
        if not self._frames:
            return None
        seen = [{d[0] for d in dets if d[1] >= _VOTE_MIN} for _, dets in self._frames]
        cur = seen[-1]
        if "green" in cur and "red" not in cur:
            if sum("green" in s for s in seen) >= _VOTE_NEED and not any("red" in s for s in seen):
                return "green"
        if "red" in cur and "green" not in cur:
            if sum("red" in s for s in seen) >= _VOTE_NEED:
                return "red"
        return None


def phrase(image, history=None):
    """給 street_mode 用：回傳要說的話；沒偵測到行人號誌時回傳 None（交回原流程）。
    history = PedHistory（若有）— 單張不夠時參考跨畫面投票"""
    dets = detect(image, conf=_VOTE_MIN)
    v = decide(dets)
    if history is not None:
        history.add(dets)
        if v is None:
            v = history.vote()
    if v in ("red", "green"):
        return t("light_" + v)
    if v == "conflict":
        return t("light_unclear")
    return None
