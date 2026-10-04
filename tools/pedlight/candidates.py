"""
candidates.py — 自動找出畫面中所有「亮的紅點／綠點」（可能是號誌），裁切成一張圖給人挑出行人號誌
用法：python tools/pedlight/candidates.py <影片名> <畫面編號>
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from track_label import RAW, imread  # noqa: E402

OUT = os.path.join(os.path.dirname(RAW), "cand.png")


def find(video, idx, top=0.75):
    img = imread(os.path.join(RAW, video, f"{idx:05d}.jpg"))
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img[:int(H * top)], cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (v > 150) & (s > 100)
    mask = (lit & (((h <= 7) | (h >= 170)) | ((h >= 45) & (h <= 95)))).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
    cands = []
    for i in range(1, n):
        x, y, w, hh, area = st[i]
        if 25 <= area <= 4000 and w < 120 and hh < 160:
            cands.append((x, y, x + w, y + hh))
    cands.sort(key=lambda b: (b[1], b[0]))
    cell = 150
    cols = 8
    rows = max(1, (len(cands) + cols - 1) // cols)
    sheet = np.full((rows * (cell + 20), cols * cell, 3), 255, np.uint8)
    for j, (x1, y1, x2, y2) in enumerate(cands[:cols * 6]):
        pad = 45
        c = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        sc = (cell - 4) / max(c.shape[:2])
        c = cv2.resize(c, (int(c.shape[1] * sc), int(c.shape[0] * sc)), interpolation=cv2.INTER_NEAREST)
        r, q = divmod(j, cols)
        sheet[r * (cell + 20) + 2:r * (cell + 20) + 2 + c.shape[0], q * cell + 2:q * cell + 2 + c.shape[1]] = c
        cv2.putText(sheet, f"#{j} {x1},{y1}", (q * cell + 3, r * (cell + 20) + cell + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    cv2.imencode(".png", sheet)[1].tofile(OUT)
    return cands


if __name__ == "__main__":
    for j, b in enumerate(find(sys.argv[1], int(sys.argv[2]))):
        print(j, b)
