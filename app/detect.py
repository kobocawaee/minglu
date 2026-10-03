"""
detect.py — object detector (YOLOv8n) สำหรับ street mode hybrid
================================================================
ทำไมมี: VLM จิ๋วเดา "รถจอด/วิ่ง" ไม่ได้ (finding §5.5.2/5.5.1) และหลอนรถบนถนนว่าง.
detector เฉพาะทางหา "รถ/คน" เป็นกล่องพิกัดได้แม่นกว่ามาก → เอาพิกัดไปคำนวณกฎเชิง
ตำแหน่ง (app/crossing.py) แบบ deterministic. นี่คือ "hybrid pipeline" ที่เล่มเราชู
(VLM=บริบท/สีไฟ, detector=รถในเส้นทาง).

offline: YOLOv8n weights (~6MB) ดาวน์โหลดครั้งเดียว แล้ว cache — inference ไม่ต่อเน็ต.
fail-open: ถ้าไม่มี ultralytics → คืน [] (แอปยังรันได้ ไม่พัง).
"""

# คลาสที่สนใจใน street mode (ชื่อตาม COCO ของ YOLO)
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
PERSON_CLASS = "person"
_RELEVANT = VEHICLE_CLASSES | {PERSON_CLASS}

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO("yolov8n.pt")
    return _MODEL


def detect(image, conf: float = 0.35, classes=None):
    """
    รัน detector บน PIL.Image. คืน (dets, (w, h)).
    dets = list of {"cls": str, "conf": float, "box": (x1,y1,x2,y2)}.
    classes: set ของชื่อคลาสที่เอา (default = รถ+คน), "all" = ทุกคลาส COCO (ใช้โดย router).
    ถ้าไม่มี ultralytics → ([], size) (fail-open).
    """
    w, h = image.size
    try:
        m = _model()
    except Exception:
        return [], (w, h)

    keep = _RELEVANT if classes is None else classes
    import numpy as np
    res = m(np.asarray(image.convert("RGB")), conf=conf, verbose=False)[0]
    dets = []
    for b in res.boxes:
        cls = m.names[int(b.cls)]
        if keep != "all" and cls not in keep:
            continue
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
        dets.append({"cls": cls, "conf": float(b.conf), "box": (x1, y1, x2, y2)})
    return dets, (w, h)
