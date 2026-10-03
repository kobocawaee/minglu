"""
light_classifier.py — อ่านสีไฟคนข้ามด้วย LYTNetV2 (dedicated CNN) แทน VLM
==========================================================================
ทำไม: §5.5.1 พิสูจน์ว่า VLM ทุกตัวอ่านไฟไม่ได้ (discrimination ≈ 0, 500M false-clear
ไฟแดง 77%). LYTNetV2 (Yu et al. 2019, MIT) บน subset เดียวกัน: แดง 29/30 เขียว 28/30
discrimination ~90pp, ~70ms CPU → `results/lytnet_validation.md`

⚠️ หลัก "ไม่มีไฟต้องบอกว่าไม่มีไฟ — ห้ามมั่วสี" → กัน 3 ชั้น (validate แล้วทั้งหมด):
  ชั้น 1  คลาส "none" ของโมเดลเอง (PTL val none 171 ใบ: ตอบ none 164/171 = 96%)
  ชั้น 2  หลักฐานว่ามีไฟจริง อย่างใดอย่างหนึ่ง (field test 14 ก.ค.: YOLO เห็นไฟคนข้าม
         แค่ 3/45 เฟรม — ไฟเล็ก/ไกลเกิน → ต้องมีหลักฐานเชิงเวลาแทน = temporal aggregation):
         (ก) YOLO เห็น "traffic light" ในเฟรมนี้หรือภายใน 8 วิที่ผ่านมา (+2-pass zoom)
         (ข) consensus: 2 prediction ล่าสุดคลาสเดียวกัน conf≥0.95 ทั้งคู่
         (ค) majority: ≥3/5 เฟรมใน window เป็นคลาสเดียวกัน + peak ≥0.95 อย่างน้อย 1 เฟรม
             โดยไม่มีสีตรงข้ามปน (แดง: mean≥0.75 | เขียว: ทุกเฟรม≥0.90 — อสมมาตรตาม L4)
  ชั้น 3  confidence: red/green ต้อง ≥0.95 (คู่หลักฐาน ก/ข) หรือ ≥0.99 เดี่ยวๆ
         คลาส countdown เชื่อยาก (ตัวอย่างเทรนน้อย, มั่วบน OOD) → ต้องมี YOLO เห็นไฟเท่านั้น
         (ไม่รับ consensus/majority — บนวิดีโอจริง countdown_green คือ noise กระพริบช่วงไฟแดง)

เกณฑ์มาจากการวัดจริงบน PTL val เต็ม split (n=645, scratch calibrate_gate 14 ก.ค.):
  บนภาพไม่มีไฟ: มั่ว red/green ที่ conf≥0.95 = 3/171 (1.8%) — เท่ากับที่ 0.99 เป๊ะ
  แปลว่าเข้มขึ้นเป็น 0.99 ไม่ได้กันมั่วเพิ่ม แต่เสีย recall เขียว 92.9%→86.2%
  ทิศอันตราย (ทายเขียวบนภาพแดง) ที่ 0.95 = 2/235 ต่อเฟรม → consensus 2 เฟรมยิ่งต่ำ

ผลบน 6 รูป canonical: ไฟจริง 3/3 สีถูก, รูปไม่มีไฟ 3/3 ตอบ "no light" (ไม่มั่วเลย)

fail-open: ไม่มี torch/weights (เช่น env NPU) → available() = False, street_mode
fallback ไปใช้ VLM ถามสีไฟแบบเดิม
"""

import time
from pathlib import Path

from app.messages import t

_REPO = Path(__file__).resolve().parent.parent
_MODEL_DIR = _REPO / "external/ImVisible/Model"
_WEIGHTS = _MODEL_DIR / "LytNetV2_weights"

CLASSES = ["red", "green", "none", "countdown_blank", "countdown_green"]
CONF_HIGH = 0.99   # มั่นใจมาก → รายงาน red/green ได้แม้ไม่มีหลักฐานอื่น
CONF_MIN = 0.95    # ขั้นต่ำเมื่อมีหลักฐานยืนยัน (YOLO เห็นไฟ หรือ consensus ข้ามเฟรม)

_WINDOW_S = 8.0    # อายุความจำ (สั้นกว่ารอบไฟจริงมาก — ไฟเปลี่ยนแล้ว consensus คลาสเก่าหลุดเอง)
_MAX_KEEP = 5


class LightHistory:
    """ความจำสั้นข้ามเฟรมของ street mode (โหมด continuous / ผู้ใช้กดถี่ๆ)

    ทำไมต้องมี: field test ไฟจริงไต้หวัน (14 ก.ค.) YOLO เห็นกล่อง traffic light แค่
    3/45 เฟรม (ไฟคนข้ามเล็ก/ไกลเกินสำหรับ YOLOv8n) → gate เดิมพึ่ง YOLO อย่างเดียว
    ทำให้แอปเงียบทั้งที่ LYTNet อ่านสีถูก. "เห็นคลาสเดิมซ้ำ 2 เฟรมติด" คือหลักฐานว่า
    มีไฟจริงที่แข็งพอๆ กัน (ภาพมั่วให้สีเดิมซ้ำติดกันยาก — ดู calibration ใน docstring บน)
    """

    def __init__(self):
        self._preds = []      # [(t, cls, conf)] ล่าสุดไม่เกิน _MAX_KEEP ภายใน _WINDOW_S
        self._light_t = 0.0   # เวลาเจอ YOLO traffic light ครั้งล่าสุด (sticky gate)

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
        """YOLO เคยเห็นไฟภายใน window ไหม — ไฟไม่หายไปไหนใน 8 วิ ถือว่า gate ยังเปิด"""
        return time.time() - self._light_t <= _WINDOW_S

    def consensus(self, cls: str, min_conf: float = CONF_MIN) -> bool:
        """2 prediction ล่าสุด (รวมเฟรมนี้) เป็นคลาสเดียวกัน + conf ถึงขั้นต่ำทั้งคู่"""
        p = self._fresh()
        if len(p) < 2:
            return False
        (_, c1, f1), (_, c2, f2) = p[-2], p[-1]
        return c1 == c2 == cls and min(f1, f2) >= min_conf

    def majority(self):
        """เสียงข้างมากใน window (อย่างน้อย 3/5) — กู้เฟรม conf กลางๆ ที่เห็นตรงกันซ้ำๆ
        และกันคลาส noise (countdown_green กระพริบช่วงไฟแดง) มาขัดจังหวะ

        เงื่อนไข (ทุกข้อ):
          1. peak: อย่างน้อย 1 เฟรมของคลาสนั้น conf ≥0.95 — สาย conf กลางๆ ล้วน
             ไม่นับเป็นหลักฐาน (กันเคสยืนนิ่งชี้ฉากที่โมเดลทายผิดคงที่ ~0.86
             เช่น crosswalk_car ซึ่งไม่มีไฟจริง; ฉากมีไฟจริงมีเฟรมชัดๆ แตะ 1.00 เสมอ)
          2. ห้ามมีสีตรงข้ามใน window เลย → ไฟเพิ่งเปลี่ยนจริงจะปิดเสียงข้างมากเก่าทันที
             (stale "เขียว" โดนเฟรมแดงเฟรมแรก kill ทันที — lag ไปทางปลอดภัยเท่านั้น)
          3. อสมมาตรตามความเสี่ยง (advisor: fail-safe street bias):
             red   → mean conf ≥0.75 พอ   (พูด "แดง" ผิด = ผู้ใช้รอเก้อ ไม่อันตราย)
             green → ทุกเฟรมต้อง ≥0.90     (พูด "เขียว" ผิด = อันตรายถึงชีวิต L4)
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
    """PIL.Image → (class_name, confidence). input 1024x768 (V2 บังคับ ≥768x768), pixel ดิบ 0-255"""
    import torch
    import numpy as np
    im = image.convert("RGB").resize((1024, 768))
    x = torch.from_numpy(np.transpose(np.asarray(im, dtype=np.float32), (2, 0, 1))).unsqueeze(0)
    with torch.no_grad():
        cls, _direction = _net()(x)      # forward มี softmax ในตัว
    p = cls[0].numpy()
    i = int(p.argmax())
    return CLASSES[i], float(p[i])


def light_phrase(image, yolo_sees_light: bool, history: "LightHistory | None" = None) -> str:
    """
    คืนประโยคสีไฟที่ 'ปลอดภัยที่จะพูด' ตามกฎ 3 ชั้น.
    yolo_sees_light = YOLO เจอกล่อง 'traffic light' ในเฟรมนี้ไหม (จาก street_mode)
    history = ความจำข้ามเฟรม (ถ้ามี) — เปิดทาง consensus + sticky YOLO gate
    """
    try:
        cls, conf = predict(image)
    except Exception:
        return ""                                   # โมเดลพัง → ไม่พูดอะไร (fail-open)

    gate = yolo_sees_light
    if history is not None:
        history.add(cls, conf, yolo_sees_light)
        gate = yolo_sees_light or history.saw_light_recently()

    if cls in ("red", "green"):
        if (gate and conf >= CONF_MIN) or conf >= CONF_HIGH:
            return t("light_" + cls)
        if history is not None and history.consensus(cls):
            return t("light_" + cls)

    # countdown_blank / countdown_green — เชื่อได้เฉพาะเมื่อ YOLO ยืนยันว่ามีไฟจริง
    # (ห้ามใช้ consensus/majority: บนวิดีโอจริง countdown_green คือ noise กระพริบช่วงไฟแดง)
    elif cls != "none" and gate and conf >= CONF_MIN:
        return t("light_countdown")

    # เฟรมนี้เดี่ยวๆ ไม่พอ → ฟังเสียงข้างมากใน window (temporal aggregation):
    # กู้เฟรม conf กลางๆ, เกลี่ย noise กระพริบ, และทนไฟโดนบัง 1 เฟรม (none แวบเดียว)
    if history is not None:
        maj = history.majority()
        if maj:
            return t("light_" + maj)

    if cls == "none":
        return t("light_none")
    return t("light_unclear")
