"""
extract.py — 從路口影片抽出畫面（訓練台灣行人號誌偵測模型用）
=================================================================
用法：python tools/pedlight/extract.py <影片資料夾> [每秒張數，預設 5]
輸出：dataset/pedlight/raw/<影片名>/00000.jpg …（直式 1080 寬，和手機傳給伺服器的畫面一樣）

每秒抽 5 張是為了讓追蹤穩定（相鄰畫面差很少）；之後訓練時會再挑其中一部分。
"""

import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "dataset", "pedlight", "raw")
VIDEO_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}


def extract(path, fps_out=5.0, width=1080):
    name = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(OUT, name)
    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(path)                    # OpenCV 會依影片的旋轉資訊自動轉正
    if not cap.isOpened():
        print(f"[失敗] 打不開 {path}（如果是 HEVC 格式，請把 iPhone 相機格式改成「最相容」重錄）")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = fps / fps_out
    i, nxt, n = 0, 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= nxt:
            h, w = frame.shape[:2]
            if w != width:
                frame = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
            cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(
                os.path.join(out, f"{n:05d}.jpg"))      # tofile：路徑有中文也能存
            n += 1
            nxt += step
        i += 1
    cap.release()
    print(f"{name}: {i} 幀 → 抽出 {n} 張（{fps:.0f} fps → {fps_out} fps）")
    return n


if __name__ == "__main__":
    src = sys.argv[1]
    fps_out = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    files = [os.path.join(src, f) for f in sorted(os.listdir(src))
             if os.path.splitext(f)[1].lower() in VIDEO_EXT]
    total = sum(extract(f, fps_out) for f in files)
    print(f"共 {len(files)} 支影片，{total} 張")
