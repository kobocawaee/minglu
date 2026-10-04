"""
grid.py — 在畫面上畫座標格線，方便人找出行人號誌的位置（設定追蹤的錨點用）
用法：python tools/pedlight/grid.py <影片名> <畫面編號> [x1 y1 x2 y2]
  不給範圍：整張縮小、每 100 像素一條線
  給範圍：  把該區域放大 3 倍、每 10 像素一條線（黃線每 50）
"""

import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "dataset", "pedlight", "raw")
OUT = os.path.join(ROOT, "dataset", "pedlight", "grid.png")


def grid(video, idx, box=None):
    im = Image.open(os.path.join(RAW, video, f"{idx:05d}.jpg")).convert("RGB")
    f = ImageFont.truetype(r"C:\Windows\Fonts\msjh.ttc", 14)
    if box is None:
        x0, y0, s, step, major = 0, 0, 0.5, 100, 100
        im = im.resize((int(im.width * s), int(im.height * s)))
    else:
        x0, y0, x1, y1 = box
        s, step, major = 3, 10, 50
        im = im.crop(box).resize(((x1 - x0) * s, (y1 - y0) * s), Image.NEAREST)
    d = ImageDraw.Draw(im)
    for v in range((x0 // step + 1) * step, x0 + int(im.width / s), step):
        X = (v - x0) * s
        d.line([(X, 0), (X, im.height)], fill=(255, 220, 0) if v % major == 0 else (120, 120, 120))
        if v % major == 0:
            d.text((X + 2, 2), str(v), font=f, fill=(255, 255, 0))
    for v in range((y0 // step + 1) * step, y0 + int(im.height / s), step):
        Y = (v - y0) * s
        d.line([(0, Y), (im.width, Y)], fill=(255, 220, 0) if v % major == 0 else (120, 120, 120))
        if v % major == 0:
            d.text((2, Y + 2), str(v), font=f, fill=(255, 255, 0))
    im.save(OUT)
    return OUT


if __name__ == "__main__":
    v, i = sys.argv[1], int(sys.argv[2])
    b = tuple(int(x) for x in sys.argv[3:7]) if len(sys.argv) >= 7 else None
    print(grid(v, i, b))
