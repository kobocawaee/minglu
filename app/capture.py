"""
capture.py — ดึงเฟรมจากกล้องด้วย OpenCV
========================================
แยกออกมาเป็น module เพื่อ: (1) test ง่าย — สลับเป็นไฟล์ภาพแทนกล้องได้
(2) ภายหลังเปลี่ยน source (กล้องมือถือ/IP cam) โดยไม่แตะ logic อื่น
"""

from PIL import Image

# หมายเหตุ: import cv2 แบบ lazy (ในเมธอด) เพื่อให้ load_image_file (ใช้แค่ PIL)
# ทำงานได้ใน env ที่ไม่มี opencv เช่น ryzen-ai-1.7.1 ตอนทดสอบ Gemma ด้วยไฟล์


class Camera:
    """wrapper รอบ cv2.VideoCapture — คืนเฟรมเป็น PIL.Image (RGB)"""

    def __init__(self, index: int = 0):
        self.index = index
        self.cap = None

    def open(self) -> None:
        import cv2
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise RuntimeError(f"เปิดกล้อง index={self.index} ไม่ได้")

    def grab(self) -> Image.Image:
        """ถ่าย 1 เฟรม → PIL.Image (RGB). OpenCV ให้ BGR ต้องแปลงก่อน"""
        import cv2
        if self.cap is None:
            raise RuntimeError("ยังไม่ได้เรียก open()")
        ok, frame_bgr = self.cap.read()
        if not ok:
            raise RuntimeError("อ่านเฟรมจากกล้องไม่สำเร็จ")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()


def load_image_file(path: str) -> Image.Image:
    """โหลดภาพจากไฟล์ (ไว้ test แทนกล้อง)"""
    return Image.open(path).convert("RGB")
