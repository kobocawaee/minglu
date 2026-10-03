"""
street_mode.py — hybrid pipeline สำหรับโหมด street (fully deterministic แล้ว)
=============================================================================
2 channel เฉพาะทาง — ไม่ใช้ VLM ในโหมดนี้แล้ว (เร็วขึ้นจาก ~2-4s เหลือ ~0.3s):
  - LYTNetV2 (light_classifier)  → สีไฟคนข้าม  (discrimination ~90pp vs VLM ~0, §5.5.1)
  - YOLOv8n + กฎตำแหน่ง (crossing) → รถในเส้นทาง (แก้อาการ VLM เดา/หลอนรถ, §5.5.2)
ผลรวม: "The light is red. Caution, a vehicle is in your path ahead. Wait."

กันมั่วสีไฟ 3 ชั้น (ดู light_classifier.py): คลาส none + YOLO ต้องเห็นไฟ + confidence

fallback: ถ้า LYTNet ใช้ไม่ได้ (เช่น env NPU ไม่มี torch) → ถามสีไฟจาก VLM แบบเดิม
(แม่นน้อยกว่า — ดู §5.5.1) / ถ้า YOLO ใช้ไม่ได้ → รายงานเท่าที่มี (fail-open ทุกชั้น)
"""

from app import config, postprocess, crossing, detect, light_classifier

# คลาสที่ street mode ต้องใช้จาก YOLO รันเดียว: รถ (กฎ crossing) + คน + ไฟ (gate สีไฟ)
_STREET_CLASSES = detect.VEHICLE_CLASSES | {detect.PERSON_CLASS, "traffic light"}

# ความจำสีไฟข้ามเฟรม (1 ผู้ใช้/instance — ทั้ง desktop และ server เสิร์ฟคนเดียว)
_LIGHT_HISTORY = light_classifier.LightHistory()

# night guard (rev. 07-21 หลัง field test 4 แยก + ตลาดกลางคืน):
# กลางคืน LYTNet มั่นใจ 1.00 แต่ผิดทั้งสองทาง — ป้ายไฟสี/ไฟถนนหลอก
# (ตลาดกลางคืนไม่มีไฟคนข้าม → ตอบ "green" 15/22 เฟรม = false clear ทิศอันตราย)
# → ฉากมืด: พูดสีไฟได้เฉพาะเมื่อ YOLO เห็นกล่องไฟ "ในเฟรมนี้" (ห้าม consensus/sticky แทน)
_NIGHT_LEVEL = 90.0   # rev. 07-27 (เดิม 80 — ต่ำกว่าช่วงกลางคืนที่วัดได้จริง)


def _is_night(image) -> bool:
    """ฉากมืด (กลางคืน) ไหม — mean gray < _NIGHT_LEVEL

    วัดจริง 175 เฟรมจากคลิปต้นฉบับ (`code/eval_night_threshold.py`):
      กลางวัน 2 แยกใหม่ 97-137 · **คลิป 14 ก.ค. 81-120** · **กลางคืน 65-87**
    → กลางวันกับกลางคืน**ซ้อนกันช่วง 81-87** ไม่มีเกณฑ์เดียวที่แยกได้ ต้องเลือกว่าจะเสียอะไร

    re-score ทั้งระบบ (`code/score_field_clips.py`, ยืนยัน reproduce ค่า 80 ได้เป๊ะก่อน):
      | | T=80 | T=90 |
      | พูดสีไฟ | 99/148 | 57/148 |
      | ผิดสี | 28 | 6 |
      | กลางคืน 2 คลิป | 29 พูด (ผิด 22) | 1 พูด (ผิด 0) |
      | 14 ก.ค. (กลางวัน) | 29/45 | 13/45 |
      | กลางวัน 2 แยกใหม่ | 43/63 | 43/63 (ไม่กระทบ) |

    เลือก 90 เพราะเล่มอ้างว่า "กลางคืนระบบเงียบ ไม่เดา" ซึ่งที่ 80 **ไม่จริงตามโค้ด**
    (พูด 29 ครั้ง ผิด 22) แลกกับเสียคำประกาศกลางวันบนคลิปที่มืดผิดปกติคลิปเดียว

    ⚠️ threshold **ไม่ได้แก้ false clearance ที่เหลือ 1/22** — เฟรมนั้นหลุดเพราะเงื่อนไข
    "YOLO เห็นกล่องไฟ" ไม่ใช่เพราะความสว่าง ไม่ว่าตั้งเท่าไหร่ก็ยังหลุด"""
    import numpy as np
    return float(np.asarray(image.convert("L")).mean()) < _NIGHT_LEVEL


def _sees_light(image, dets) -> bool:
    """YOLO เห็นไฟไหม — 2 pass: เฟรมเต็มก่อน, พลาดค่อย zoom ครึ่งบน 2 เท่า
    (field test: ไฟคนข้ามเล็ก/ไกล YOLOv8n บนเฟรมเต็มเห็นแค่ 3/45 เฟรม)"""
    if any(d["cls"] == "traffic light" for d in dets):
        return True
    w, h = image.size
    top = image.crop((0, 0, w, int(h * 0.55)))
    top = top.resize((w * 2, int(h * 0.55) * 2))
    dets2, _ = detect.detect(top, conf=0.20, classes={"traffic light"})
    return bool(dets2)


def describe(backend, image):
    """คืน (text, hazard, info). text = สีไฟ (CNN) + คำเตือนรถ (detector)"""
    # YOLO ครั้งเดียว ใช้ทั้ง 2 channel
    dets, (w, h) = detect.detect(image, conf=0.25, classes=_STREET_CLASSES)
    yolo_sees_light = _sees_light(image, dets)

    # channel 1: สีไฟ — LYTNet (fallback → VLM ถ้าไม่มี weights)
    if light_classifier.available():
        night = _is_night(image)
        box_now = any(d["cls"] == "traffic light" for d in dets)
        if night and not box_now:
            light = ""            # ฉากมืดไม่มีหลักฐานไฟในเฟรม → เงียบ (fail-safe)
        else:
            light = light_classifier.light_phrase(image, yolo_sees_light,
                                                  history=_LIGHT_HISTORY)
    else:
        light = postprocess.clean(
            backend.describe(image, config.get_prompt("street"),
                             max_new_tokens=config.get_max_tokens("street"))
        )

    # channel 2: รถในเส้นทาง — กฎตำแหน่งบน detection เดิม (conf ≥0.35 เท่ากฎเดิม)
    veh_dets = [d for d in dets if d["conf"] >= 0.35]
    veh_phrase, hazard, info = crossing.assess(image, dets=veh_dets, size=(w, h))
    info["yolo_sees_light"] = yolo_sees_light

    text = " ".join(p for p in (light, veh_phrase) if p).strip()
    return text, hazard, info
