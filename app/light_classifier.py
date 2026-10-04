"""
light_classifier.py — 用 LYTNetV2（專用 CNN）判斷行人號誌燈號，取代 VLM
==========================================================================
原因：§5.5.1 證明所有 VLM 都讀不出號誌（分辨能力 ≈ 0，500M 模型把紅燈誤報可通行
達 77%）。LYTNetV2（Yu et al. 2019，MIT）在同一子集上：紅 29/30、綠 28/30
discrimination ~90pp, ~70ms CPU → `results/lytnet_validation.md`

⚠️ 原則「沒有號誌就要說沒有 — 不准亂猜顏色」→ 三層防護（全部驗證過）：
  第 1 層  模型本身的「none」類別（PTL 驗證集 171 張無號誌：回答 none 164/171 = 96%）
  第 2 層  確實有號誌的證據，任一即可（7/14 實地測試：YOLO 只在 3/45 張畫面
         看到行人號誌 — 號誌太小／太遠 → 改用時間上的證據 = 跨畫面彙整）：
         (甲) YOLO 在這張或過去 8 秒內看到「traffic light」（含兩階段放大）
         (乙) 一致：最近兩次預測是同一類別，而且信心都 ≥0.95
         (丙) 多數：時間窗內 ≥3/5 張是同一類別，且至少一張信心 ≥0.95
             而且沒有混到相反的顏色（紅：平均 ≥0.75 | 綠：每張都 ≥0.90 — 依 L4 風險不對稱）
  第 3 層  信心：紅／綠要 ≥0.95（搭配證據甲／乙），或單獨 ≥0.99
         倒數類別較不可信（訓練樣本少，在分布外的圖上會亂猜）→ 只在 YOLO 看到號誌時才採用
         （不接受一致／多數：在實際影片中 countdown_green 是紅燈期間閃爍的雜訊）

門檻來自 PTL 驗證集完整切分的實測（n=645，7/14 校準）：
  在沒有號誌的圖上：信心 ≥0.95 時亂猜紅／綠 = 3/171（1.8%）— 和 0.99 時完全一樣
  也就是說提高到 0.99 並不會多擋亂猜，卻會讓綠燈召回率從 92.9% 降到 86.2%
  危險方向（紅燈圖猜成綠）在 0.95 時 = 每張 2/235 → 要求連續兩張一致後更低

6 張標準測試圖的結果：有號誌的 3/3 顏色正確，沒有號誌的 3/3 回答「沒有號誌」（完全沒亂猜）

容錯：沒有 torch／權重檔（例如 NPU 環境）→ available() = False，street_mode
改回用 VLM 詢問燈號的舊方法
"""

import time
from pathlib import Path

from app.messages import t

_REPO = Path(__file__).resolve().parent.parent
_MODEL_DIR = _REPO / "external/ImVisible/Model"
_WEIGHTS = _MODEL_DIR / "LytNetV2_weights"

CLASSES = ["red", "green", "none", "countdown_blank", "countdown_green"]
CONF_HIGH = 0.99   # 非常有把握 → 即使沒有其他證據也可以報紅／綠
CONF_MIN = 0.95    # 有佐證時的最低信心（YOLO 看到號誌，或跨畫面一致）

_WINDOW_S = 8.0    # 記憶時長（遠短於實際號誌週期 — 燈號一變，舊類別的一致性自然失效）
_MAX_KEEP = 5


class LightHistory:
    """過馬路模式跨畫面的短期記憶（連續模式／使用者連續點擊）

    為什麼需要：臺灣號誌實地測試（7/14）YOLO 只在 3/45 張畫面看到 traffic light 框
    （行人號誌對 YOLOv8n 來說太小／太遠）→ 原本只靠 YOLO 的關卡
    讓程式在 LYTNet 其實判斷正確時也保持沉默。「連續兩張看到同一類別」
    是同樣可靠的「確實有號誌」證據（亂猜很難連續兩張都同色 — 見上方 docstring 的校準）
    """

    def __init__(self):
        self._preds = []      # [(時間, 類別, 信心)] 最近最多 _MAX_KEEP 筆，且在 _WINDOW_S 內
        self._light_t = 0.0   # 最近一次 YOLO 看到 traffic light 的時間（關卡保持開啟）

    def _fresh(self):
        now = time.time()
        self._preds = [p for p in self._preds if now - p[0] <= _WINDOW_S][-_MAX_KEEP:]
        return self._preds

    def add(self, cls, conf, yolo_light: bool):
        self._fresh()
        self._preds.append((time.time(), cls, conf))
        if yolo_light:
            self._light_t = time.time()

    def saw_light_recently(self) -> bool:
        """YOLO 在時間窗內是否看過號誌 — 號誌 8 秒內不會消失，視為關卡仍開啟"""
        return time.time() - self._light_t <= _WINDOW_S

    def consensus(self, cls: str, min_conf: float = CONF_MIN) -> bool:
        """最近兩次預測（含這一張）是同一類別，且信心都達到最低門檻"""
        p = self._fresh()
        if len(p) < 2:
            return False
        (_, c1, f1), (_, c2, f2) = p[-2], p[-1]
        return c1 == c2 == cls and min(f1, f2) >= min_conf

    def majority(self):
        """時間窗內的多數決（至少 3/5）— 救回信心中等但反覆一致的畫面，
        並防止雜訊類別（紅燈期間閃爍的 countdown_green）打斷判斷

        條件（全部都要符合）：
          1. 峰值：該類別至少有一張信心 ≥0.95 — 全部只是中等信心
             不算證據（防止站著不動、模型穩定猜錯在 ~0.86 的情況，
             例如 crosswalk_car 其實沒有號誌；真的有號誌時一定有清楚的畫面達到 1.00）
          2. 時間窗內完全不能有相反的顏色 → 燈號真的改變時會立刻推翻舊的多數
             （過時的「綠」碰到第一張紅燈畫面就立刻作廢 — 延遲只會往安全的方向）
          3. 依風險不對稱（指導建議：失效時偏向安全）：
             red   → 平均信心 ≥0.75 即可   （誤報「紅」= 使用者白等，不危險）
             green → 每一張都要 ≥0.90     （誤報「綠」= 可能致命，L4）
        """
        p = self._fresh()
        if len(p) < 3:
            return None
        from collections import Counter
        cls, n = Counter(c for _, c, _ in p).most_common(1)[0]
        if cls not in ("red", "green") or n < 3:
            return None
        opposite = "green" if cls == "red" else "red"
        if any(c == opposite for _, c, _ in p):
            return None
        confs = [f for _, c, f in p if c == cls]
        if max(confs) < CONF_MIN:                      # peak evidence
            return None
        if cls == "red" and sum(confs) / len(confs) >= 0.75:
            return cls
        if cls == "green" and min(confs) >= 0.90:
            return cls
        return None

_NET = None


def available() -> bool:
    return _WEIGHTS.exists()


def _net():
    global _NET
    if _NET is None:
        import sys
        sys.path.insert(0, str(_MODEL_DIR))
        import torch
        from LYTNetV2 import LYTNetV2
        net = LYTNetV2()
        net.load_state_dict(torch.load(_WEIGHTS, map_location="cpu", weights_only=True))
        net.eval()
        _NET = net
    return _NET


def predict(image):
    """PIL.Image → (類別名稱, 信心)。輸入 1024x768（V2 要求 ≥768x768），原始像素 0-255"""
    import torch
    import numpy as np
    im = image.convert("RGB").resize((1024, 768))
    x = torch.from_numpy(np.transpose(np.asarray(im, dtype=np.float32), (2, 0, 1))).unsqueeze(0)
    with torch.no_grad():
        cls, _direction = _net()(x)      # forward 內含 softmax
    p = cls[0].numpy()
    i = int(p.argmax())
    return CLASSES[i], float(p[i])


def light_phrase(image, yolo_sees_light: bool, history: "LightHistory | None" = None) -> str:
    """
    依三層規則，回傳「可以安全說出口」的燈號句子。
    yolo_sees_light = 這張畫面 YOLO 有沒有看到 'traffic light' 框（來自 street_mode）
    history = 跨畫面記憶（若有）— 啟用一致性判斷＋關卡保持開啟
    """
    try:
        cls, conf = predict(image)
    except Exception:
        return ""                                   # 模型出錯 → 什麼都不說（容錯）

    gate = yolo_sees_light
    if history is not None:
        history.add(cls, conf, yolo_sees_light)
        gate = yolo_sees_light or history.saw_light_recently()

    if cls in ("red", "green"):
        if (gate and conf >= CONF_MIN) or conf >= CONF_HIGH:
            return t("light_" + cls)
        if history is not None and history.consensus(cls):
            return t("light_" + cls)

    # countdown_blank / countdown_green — 只有在 YOLO 確認確實有號誌時才採信
    # （不可用一致／多數：在實際影片中 countdown_green 是紅燈期間閃爍的雜訊）
    elif cls != "none" and gate and conf >= CONF_MIN:
        return t("light_countdown")

    # 這張單獨看不夠 → 參考時間窗內的多數決（跨畫面彙整）：
    # 救回中等信心的畫面、平滑閃爍雜訊，並容許號誌被擋住一張（短暫出現 none）
    if history is not None:
        maj = history.majority()
        if maj:
            return t("light_" + maj)

    if cls == "none":
        return t("light_none")
    return t("light_unclear")
