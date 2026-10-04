"""
quality.py — 畫面品質檢查（推論之前）
=============================================
問題：SmolVLM-500M 是小模型 → 對輸入很敏感。畫面模糊或太暗（拿著手機邊走邊拍時
很常見）會讓描述「不穩定」（亂說／一直變）。我們的解碼已經是貪婪法
（結果固定）→ 不穩定來自「畫面品質差」，不是隨機性。

做法：送進 VLM／OCR 之前，先用 OpenCV 快速檢查畫面（約 1ms）。沒通過 → 不在壞畫面上推論，
而是請使用者拿穩或換到亮一點的地方再試（比用模糊畫面描述好）。

指標：
  - 模糊度：Laplacian 的變異數（越低越模糊）。量測前先縮放成固定寬度，
            讓同一個門檻在網路攝影機和手機（解析度不同）都適用
  - 亮度：平均灰階值（0-255）。太暗或太亮 = 模型會亂猜
門檻以 17 張標準測試圖校準（清晰的圖必須全部通過）— 見 config.QUALITY
"""

import numpy as np

from app.messages import t

# 量測模糊度前統一縮放的寬度（讓 Laplacian 變異數不受解析度影響）
_NORM_WIDTH = 640


def _to_gray_norm(image):
    """PIL.Image → 寬度縮放成 _NORM_WIDTH 的灰階 numpy（uint8）"""
    import cv2
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    if w != _NORM_WIDTH:
        scale = _NORM_WIDTH / w
        arr = cv2.resize(arr, (_NORM_WIDTH, max(1, int(h * scale))))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def blur_score(image) -> float:
    """Laplacian 變異數 — 越高越清晰，模糊則數值低"""
    import cv2
    gray = _to_gray_norm(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(image) -> float:
    """平均亮度 0-255（灰階平均）"""
    return float(_to_gray_norm(image).mean())


def assess(image, params: dict | None = None):
    """
    檢查畫面品質。回傳 (ok: bool, reason: str)。
    reason = 要唸給使用者聽的簡短提示（ok 時為 ""）。
    沒有 OpenCV（例如 NPU 環境）→ 所有畫面都放行（容錯），避免程式當掉。
    """
    p = params or {}
    min_blur = p.get("min_blur", 80.0)
    min_bright = p.get("min_brightness", 40.0)
    max_bright = p.get("max_brightness", 225.0)
    try:
        b = blur_score(image)
        lum = brightness(image)
    except Exception:
        return True, ""            # 沒有 cv2／無法量測 → 不擋

    if b < min_blur:
        return False, t("q_blur")
    if lum < min_bright:
        return False, t("q_dark")
    if lum > max_bright:
        return False, t("q_bright")
    return True, ""
