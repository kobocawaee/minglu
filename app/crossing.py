"""
crossing.py — กฎเชิงตำแหน่ง (deterministic) จาก detector boxes
=============================================================
แปลงกล่องรถจาก app/detect.py เป็นคำเตือน โดยใช้ "กฎเชิงตำแหน่ง" แทนการเดา moving/stopped
(เฟรมเดียวมองการเคลื่อนไหวไม่ได้ — แต่ "รถอยู่ตรงไหน" เห็นได้):

  โซนอันตราย (in path) = รถที่ **ใกล้ (กล่องใหญ่/อยู่ครึ่งล่าง) + อยู่กลางเฟรม (ข้างหน้าเรา)**
  → รถบน/ใกล้ทางข้ามข้างหน้า = "รอ"; รถไกลหรือริมขอบ = ยังไม่ใช่ภัยเฉพาะหน้า

⚠️ simplification: เราไม่ได้ตรวจ 'เส้นทางม้าลายจริง'. ลอง classical CV (threshold ขาว) แล้ว
เปราะเกิน — ปนกับของสว่างอื่น (รถขาว/ตึก) ทำให้ขอบเพี้ยน. ตรวจทางม้าลายแม่นจริงต้องใช้
trained segmentation model (มี dataset crosswalk) = future work. แทนที่ด้วย "รถใกล้+อยู่ข้างหน้า"
โดยใช้ **ขนาดกล่อง = ความใกล้** (robust ต่อมุมกล้อง กว่าใช้ตำแหน่ง y). bias ปลอดภัย (เตือนไว้ก่อน).
"""

from app import detect
from app.messages import t

# threshold (normalized 0-1) — คาลิเบรตจากรูป crosswalk 5 ใบ
#   in-path = อยู่กลางเฟรม (ข้างหน้า) AND (ใกล้พอ = กล่องใหญ่ | อยู่ที่เท้าเรา = ล่างมาก)
#   ขนาดกล่อง (area) = สัญญาณความใกล้ที่แข็งสุด → คัดรถ 'อีกฝั่งถนน' (เล็ก/ไกล) ออกได้
_CENTRAL = (0.12, 0.88)   # cx ในช่วงนี้ = อยู่ข้างหน้า (ไม่ใช่ไกลริมขอบ)
_NEAR_AREA = 0.03         # พื้นที่กล่อง > 3% ของเฟรม = ใกล้พอเป็นภัย
_AT_FEET = 0.75           # ขอบล่างกล่อง > นี้ = อยู่ตรงหน้าเรามาก (ใกล้แม้กล่องไม่ใหญ่)


def _in_path(box, w, h) -> bool:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / w
    bottom = y2 / h
    area = ((x2 - x1) * (y2 - y1)) / (w * h)
    central = _CENTRAL[0] < cx < _CENTRAL[1]
    near = area > _NEAR_AREA or bottom > _AT_FEET
    return central and near


def assess(image, conf: float = 0.35, dets=None, size=None):
    """
    คืน (phrase, hazard: bool, info: dict).
    phrase = ประโยคสั้นเรื่องรถ (พูดต่อจากสีไฟ), hazard = มีรถในเส้นทางไหม.
    dets/size: ส่ง detection ที่รันแล้วมา reuse ได้ (street_mode รัน YOLO ครั้งเดียว)
    ถ้า detector ใช้ไม่ได้ (ไม่มี ultralytics) → ("", False, ...) = ไม่เพิ่มอะไร (fail-open).
    """
    if dets is None:
        dets, (w, h) = detect.detect(image, conf=conf)
    else:
        w, h = size
    vehicles = [d for d in dets if d["cls"] in detect.VEHICLE_CLASSES]
    in_path = [d for d in vehicles if _in_path(d["box"], w, h)]
    n_people = sum(1 for d in dets if d["cls"] == detect.PERSON_CLASS)
    info = {"n_vehicles": len(vehicles), "n_in_path": len(in_path), "n_people": n_people}

    if not dets:                                   # detector ปิด/ไม่เจออะไร
        return "", False, info
    if in_path:
        return t("veh_in_path"), True, info
    if vehicles:
        return t("veh_nearby"), False, info
    return t("veh_none"), False, info
