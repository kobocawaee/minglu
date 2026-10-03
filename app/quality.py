"""
quality.py — frame-quality gate (ก่อน infer)
=============================================
ปัญหา: SmolVLM-500M เป็นโมเดลจิ๋ว → เปราะต่อ input. เฟรมเบลอ/มืด (เจอบ่อยตอนถือ
กล้องมือถือเดินไป) ทำให้คำบรรยาย "ไม่เสถียร" (มั่ว/เปลี่ยนไปมา). decoder เราเป็น greedy
(deterministic) อยู่แล้ว → ความไม่เสถียรมาจาก "เฟรมคุณภาพแย่" ไม่ใช่การสุ่ม.

วิธี: ก่อนส่งเข้า VLM/OCR เช็คเฟรมเร็วๆ ด้วย OpenCV (~1ms). ถ้าไม่ผ่าน → ไม่ infer
บนภาพเสีย แต่บอกผู้ใช้ให้ถือนิ่ง/หาที่สว่าง แล้วลองใหม่ (ดีกว่าบรรยายจากภาพเบลอ).

เมตริก:
  - blur  : variance ของ Laplacian (ยิ่งต่ำ = ยิ่งเบลอ). resize กว้างคงที่ก่อนวัด
            เพื่อให้ threshold เดียวใช้ได้ทั้งเว็บแคม/มือถือ (ความละเอียดต่างกัน)
  - แสง   : ความสว่างเฉลี่ย (grayscale mean 0-255). มืด/จ้าเกิน = เดามั่ว
threshold คาลิเบรตจาก 17 รูป canonical (รูปคมทั้งหมดต้องผ่าน) — ดู config.QUALITY
"""

import numpy as np

from app.messages import t

# ความกว้างมาตรฐานที่ resize ไปก่อนวัด blur (ทำให้ Laplacian var ไม่ขึ้นกับ resolution)
_NORM_WIDTH = 640


def _to_gray_norm(image):
    """PIL.Image → grayscale numpy (uint8) ที่ resize กว้าง = _NORM_WIDTH"""
    import cv2
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    if w != _NORM_WIDTH:
        scale = _NORM_WIDTH / w
        arr = cv2.resize(arr, (_NORM_WIDTH, max(1, int(h * scale))))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def blur_score(image) -> float:
    """variance ของ Laplacian — ยิ่งสูง = ยิ่งคม. เบลอ = ค่าต่ำ"""
    import cv2
    gray = _to_gray_norm(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(image) -> float:
    """ความสว่างเฉลี่ย 0-255 (grayscale mean)"""
    return float(_to_gray_norm(image).mean())


def assess(image, params: dict | None = None):
    """
    ตรวจคุณภาพเฟรม. คืน (ok: bool, reason: str).
    reason = ข้อความสั้นสำหรับพูดให้ผู้ใช้ฟัง ("" ถ้า ok).
    ถ้าไม่มี OpenCV (เช่น env NPU) → ปล่อยผ่านทุกเฟรม (fail-open) กันแอปพัง.
    """
    p = params or {}
    min_blur = p.get("min_blur", 80.0)
    min_bright = p.get("min_brightness", 40.0)
    max_bright = p.get("max_brightness", 225.0)
    try:
        b = blur_score(image)
        lum = brightness(image)
    except Exception:
        return True, ""            # ไม่มี cv2 / วัดไม่ได้ → ไม่กั้น

    if b < min_blur:
        return False, t("q_blur")
    if lum < min_bright:
        return False, t("q_dark")
    if lum > max_bright:
        return False, t("q_bright")
    return True, ""
