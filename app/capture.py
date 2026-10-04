"""
capture.py — 用 OpenCV 從攝影機擷取畫面
========================================
獨立成一個模組是為了：(1) 方便測試 — 可以改用圖片檔取代攝影機
(2) 之後要換來源（手機鏡頭／網路攝影機）時，不必動到其他邏輯
"""

from PIL import Image

# 注意：cv2 採延遲 import（放在方法裡），讓只用到 PIL 的 load_image_file
# 能在沒有 opencv 的環境執行，例如用圖片檔測試 Gemma 時的 ryzen-ai-1.7.1


class Camera:
    """包裝 cv2.VideoCapture — 回傳 PIL.Image（RGB）格式的畫面"""

    def __init__(self, index: int = 0):
        self.index = index
        self.cap = None

    def open(self) -> None:
        import cv2
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise RuntimeError(f"無法開啟攝影機 index={self.index}")

    def grab(self) -> Image.Image:
        """擷取一張畫面 → PIL.Image（RGB）。OpenCV 給的是 BGR，要先轉換"""
        import cv2
        if self.cap is None:
            raise RuntimeError("還沒呼叫 open()")
        ok, frame_bgr = self.cap.read()
        if not ok:
            raise RuntimeError("從攝影機讀取畫面失敗")
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
    """從檔案載入影像（測試時代替攝影機）"""
    return Image.open(path).convert("RGB")
