"""
pipeline.py — จุดรวม logic "เฟรม → คำพูด" ของทุกโหมด (ใช้ทั้ง assistant.py และ server.py)
=========================================================================================
เดิม logic โหมดซ้ำ 2 ที่ (desktop/มือถือ) → รวมที่เดียว. ต่อโหมด:
  read        → OCR (RapidOCR)                                 — VLM อ่าน text ไม่ได้ (0/17)
  street      → hybrid: VLM สีไฟ + YOLO รถในเส้นทาง (street_mode)
  surrounding/indoor/object → VLM + postprocess
                + person-check: YOLO เจอคนใหญ่ชัดแต่ VLM ไม่พูดถึง → เติมประโยคเตือน
                  (คนจ่อใกล้กล้อง = กล่องใหญ่/เห็นบางส่วน VLM จิ๋วมักมองไม่ออกว่าเป็นคน
                   — พลาดคน = อันตราย จึงให้ detector เช็คซ้ำแบบ deterministic)
  auto        → router เลือกโหมดจากสิ่งที่ YOLO เห็น (คนตาบอดไม่ต้องเลือกโหมดเอง)

auto router (heuristic จาก COCO classes, ~50ms):
  เห็นรถ/ไฟจราจร → street | ของถือได้ชิ้นใหญ่จ่อกลางเฟรม → object |
  เฟอร์นิเจอร์ → indoor | อื่นๆ → surrounding
  (read ไม่อยู่ใน auto — "อยากอ่าน" เป็นความตั้งใจของผู้ใช้ เดาจากภาพไม่ได้)

rev. 07-19 — street bias (ทดลองบนชุด verified 60 ใบ, results/router_eval_v2.md):
  street cue รับที่ conf ≥0.25 (cue อื่นคง ≥0.40) — เอนเข้า street ทิศเดียว = fail-safe
  → street-recall 16/20 → 19/20 แลก benign misroute 1 ใบ. ส่วน "LYTNet เป็น street cue"
  ทดลองแล้ว *ไม่เอา*: นอก domain (รูป indoor/object) มันตอบ red@1.00 มั่ว 24/40 ใบ
"""

import re

from app import config, postprocess, detect
from app.messages import t, mode_name

# ---- คลาส COCO ที่ใช้ route ----
_STREET_CUES = detect.VEHICLE_CLASSES | {"traffic light", "stop sign"}
_INDOOR_CUES = {"chair", "couch", "bed", "dining table", "tv", "refrigerator",
                "microwave", "oven", "sink", "toilet", "potted plant"}
_HANDHELD = {"bottle", "cup", "remote", "cell phone", "book", "banana", "apple",
             "orange", "scissors", "toothbrush", "spoon", "fork", "knife", "bowl",
             "mouse", "keyboard", "vase", "clock", "teddy bear"}

_PERSON_RE = re.compile(r"\b(person|people|man|woman|men|women|pedestrian|someone|"
                        r"child|children|kid|boy|girl|human|figure)\b"
                        # [資服版] 中文描述裡提到人的說法（排除「沒有人」）
                        r"|(?<!沒)有人|行人|路人|人們|一個人|一名|一位|男子|女子|男人|女人|"
                        r"男性|女性|小孩|孩童|兒童|學生|騎士", re.I)


def route(image):
    """เลือกโหมดจากสิ่งที่ detector เห็น. คืน (mode, dets) — dets ส่งต่อไปใช้ได้เลย
    street bias: street cue รับตั้งแต่ conf 0.25 (พลาด street = เสี่ยง L4, พลาดโหมดอื่น = benign)
    cue ที่เหลือใช้ ≥0.40 เท่าเดิม"""
    dets, (w, h) = detect.detect(image, conf=0.25, classes="all")
    if any(d["cls"] in _STREET_CUES for d in dets):
        return "street", dets

    dets = [d for d in dets if d["conf"] >= 0.40]
    if not dets:                      # detector ใช้ไม่ได้/ไม่เห็นอะไร → บรรยายทั่วไป
        return "surrounding", dets

    classes = {d["cls"] for d in dets}

    # ของถือได้ชิ้นใหญ่ จ่อกลางเฟรม → โหมด object
    for d in dets:
        if d["cls"] in _HANDHELD:
            x1, y1, x2, y2 = d["box"]
            area = ((x2 - x1) * (y2 - y1)) / (w * h)
            cx = ((x1 + x2) / 2) / w
            if area > 0.12 and 0.2 < cx < 0.8:
                return "object", dets

    if classes & _INDOOR_CUES:
        return "indoor", dets
    return "surrounding", dets


def _person_clearly_present(image) -> bool:
    """YOLO เจอคนที่ 'ชัด' ไหม (conf สูง + กล่องใหญ่พอ = อยู่ใกล้/เด่น ไม่ใช่คนจิ๋วไกลๆ)"""
    dets, (w, h) = detect.detect(image, conf=0.50)
    for d in dets:
        if d["cls"] == detect.PERSON_CLASS:
            x1, y1, x2, y2 = d["box"]
            if ((x2 - x1) * (y2 - y1)) / (w * h) > 0.04:
                return True
    return False


def _vlm_read(backend, image) -> str:
    return postprocess.clean(
        backend.describe(image, config.get_prompt("read"),
                         max_new_tokens=config.get_max_tokens("read")))


def _read(backend, image) -> str:
    """[資服版] 讀字模式。engine 見 config.READ_ENGINE。"""
    engine = getattr(config, "READ_ENGINE", "easyocr")
    can_vlm = backend is not None and config.BACKEND != "smolvlm"   # SmolVLM 讀不了字（論文 4.3）
    if engine == "vlm" and can_vlm:
        return _vlm_read(backend, image)
    from app import ocr
    text, kept, total = ocr.read_detail(image)
    if total == 0:
        return t("no_text")
    if engine == "auto" and can_vlm and kept < total * config.OCR_MIN_SURE_RATIO:
        print(f"      [read] EasyOCR 有把握 {kept}/{total} 字 → 改由 VLM 補讀")
        return _vlm_read(backend, image)
    return text or t("no_text")


def describe(backend, image, mode: str):
    """
    ประมวลผล 1 เฟรมตามโหมด. คืน (text, mode_used).
    mode="auto" → route เลือกโหมดก่อน แล้วบอกชื่อโหมดนำหน้าให้ผู้ใช้รู้
    """
    auto = mode == "auto"
    if auto:
        mode, _ = route(image)
        # [資服版] 自動模式選到過馬路時，補做過馬路模式較嚴格的清晰度檢查
        if mode == "street" and config.QUALITY_GATE:
            from app import quality
            ok, reason = quality.assess(image, config.get_quality("street"))
            if not ok:
                return reason, mode

    if mode == "read":
        text = _read(backend, image)
    elif mode == "street":
        from app import street_mode
        text, _hazard, _info = street_mode.describe(backend, image)
    else:
        text = postprocess.clean(
            backend.describe(image, config.get_prompt(mode),
                             max_new_tokens=config.get_max_tokens(mode))
        )
        # person-check (hybrid): VLM ไม่พูดถึงคน แต่ detector เห็นคนชัด → เติมเตือน
        if mode in ("surrounding", "indoor") and not _PERSON_RE.search(text):
            if _person_clearly_present(image):
                text = (text + " " + t("person_ahead")).strip()

    if auto:
        text = f"{t('mode_prefix', mode=mode_name(mode))} {text}"
    return text, mode
