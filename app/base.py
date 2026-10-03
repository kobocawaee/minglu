"""
base.py — interface กลางของ VLM backend
=======================================
ทุก backend (SmolVLM บน CPU/iGPU, Gemma บน NPU) ต้อง implement คลาสนี้
→ app logic (assistant.py) เรียกผ่าน interface เดียว ไม่ต้องรู้ว่าข้างใน
  เป็น transformers หรือ OGA หรือรันบน device ไหน

เหตุผลที่ต้องมี interface: Phase 1 พบว่า backend อยู่คนละ conda env
(transformers vs OGA) → รันพร้อมกันใน process เดียวไม่ได้. แอปจึงเลือก
backend 1 ตัวตอน start แล้วใช้ผ่าน abstraction นี้
"""

from abc import ABC, abstractmethod


class VLMBackend(ABC):
    """interface ที่ทุก VLM backend ต้องมี"""

    #: ชื่อสั้นๆ ไว้ log / แสดงผล เช่น "smolvlm-500m@cpu"
    name: str = "vlm-backend"

    @abstractmethod
    def load(self) -> None:
        """โหลด model เข้า memory (เรียกครั้งเดียวตอน start แอป).
        แยกจาก __init__ เพราะการโหลดช้า (หลายวินาที) อยากคุมจังหวะเอง
        """
        raise NotImplementedError

    @abstractmethod
    def describe(self, image, prompt: str, max_new_tokens: int | None = None) -> str:
        """รับภาพ (PIL.Image หรือ numpy BGR จาก OpenCV) + prompt → คืนคำบรรยาย (str)
        max_new_tokens: override เพดาน output ต่อโหมด (None = ใช้ default ของ backend).
        จาก finding §5.6 latency = decode-bound → ตัด output = เร็วขึ้น (street ใช้สั้น)
        backend แต่ละตัวจัดการ preprocessing/format เองภายใน
        """
        raise NotImplementedError

    def warmup(self) -> None:
        """(ออปชัน) รัน inference หลอก 1 ครั้งให้ JIT/compile เสร็จ
        ก่อนใช้งานจริง → ลด latency ของครั้งแรก. default ไม่ทำอะไร
        """
        pass
