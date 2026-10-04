"""
track_label.py — 半自動標註：追蹤影片中的行人號誌，並用顏色判斷紅燈／綠燈
==========================================================================
人只需要在幾個「錨點畫面」框出行人號誌（整個號誌頭，含倒數數字），
程式用灰階模板比對往後追蹤（號誌外殼的形狀不會因為燈號改變而變），
再看框裡亮的是紅色還是綠色來決定類別。判斷不出來的畫面（閃爍熄滅、被擋住）直接略過，不硬標。

用法（在程式裡呼叫）：
    track("影片名", anchors=[(畫面編號, x1, y1, x2, y2), ...], end=結束畫面編號)
輸出：
    dataset/pedlight/labels/<影片名>.json   {畫面編號: [x1, y1, x2, y2, "red"/"green"]}
    dataset/pedlight/qa/<影片名>.png        抽樣裁切圖，給人檢查有沒有追錯、判錯
"""

import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "dataset", "pedlight", "raw")
LAB = os.path.join(ROOT, "dataset", "pedlight", "labels")
QA = os.path.join(ROOT, "dataset", "pedlight", "qa")


def imread(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def lamp_state(bgr_crop):
    """框裡亮的是紅還是綠。亮且飽和的像素才算（外殼的暗綠色不會被算進去）。"""
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    lit = (v > 140) & (s > 90)
    green = int((lit & (h >= 45) & (h <= 95)).sum())
    red = int((lit & ((h <= 7) | (h >= 170))).sum())   # 不含橘色倒數數字（H 8~25）
    need = max(5, int(0.006 * bgr_crop.shape[0] * bgr_crop.shape[1]))
    if green >= need and green > 2 * red:
        return "green", green, red
    if red >= need and red > 2 * green:
        return "red", green, red
    return None, green, red


def _match(gray, tmpl, cx, cy, radius):
    H, W = gray.shape
    th, tw = tmpl.shape
    best = (-1, None)
    for sc in (0.93, 1.0, 1.07):
        t = cv2.resize(tmpl, (max(6, round(tw * sc)), max(8, round(th * sc))))
        x0, y0 = max(0, int(cx - radius)), max(0, int(cy - radius))
        x1, y1 = min(W, int(cx + radius)), min(H, int(cy + radius))
        win = gray[y0:y1, x0:x1]
        if win.shape[0] < t.shape[0] or win.shape[1] < t.shape[1]:
            continue
        r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(r)
        if score > best[0]:
            best = (score, (x0 + loc[0], y0 + loc[1], x0 + loc[0] + t.shape[1], y0 + loc[1] + t.shape[0]))
    return best


def track(video, anchors, end=None, min_score=0.45):
    frames = sorted(f for f in os.listdir(os.path.join(RAW, video)) if f.endswith(".jpg"))
    end = len(frames) - 1 if end is None else min(end, len(frames) - 1)
    anchors = sorted(anchors)
    out, lost_at = {}, []
    for k, (start, *box) in enumerate(anchors):
        stop = anchors[k + 1][0] - 1 if k + 1 < len(anchors) else end
        img = imread(os.path.join(RAW, video, frames[start]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        x1, y1, x2, y2 = box
        # 追蹤用的是「號誌連同周圍」的大範圍：燈號由紅變綠時，只佔模板的一小部分，不會追丟
        pad = int(1.3 * max(x2 - x1, y2 - y1))
        cx1, cy1, cx2, cy2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
        off = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)   # 號誌在大範圍裡的相對位置（隨尺度縮放）
        cw0 = cx2 - cx1
        tmpl0 = gray[cy1:cy2, cx1:cx2].copy()           # 錨點模板：用來檢查有沒有越追越偏
        tmpl = tmpl0.copy()
        for i in range(start, stop + 1):
            img = imread(os.path.join(RAW, video, frames[i]))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if i > start:
                mx, my = (cx1 + cx2) / 2, (cy1 + cy2) / 2
                radius = 0.8 * max(cx2 - cx1, cy2 - cy1) + 40
                score, b = _match(gray, tmpl, mx, my, radius)
                if b is None or score < min_score:          # 用最初的模板、更大範圍再找一次
                    score, b = _match(gray, tmpl0, mx, my, radius * 1.6)
                if b is None or score < min_score:
                    lost_at.append(i)
                    break
                patch = gray[b[1]:b[3], b[0]:b[2]]
                t0 = cv2.resize(tmpl0, (patch.shape[1], patch.shape[0]))
                s0 = float(cv2.matchTemplate(patch, t0, cv2.TM_CCOEFF_NORMED)[0, 0])
                if s0 < 0.2:                           # 跟最初的畫面已經不像了 → 視為追丟
                    lost_at.append(i)
                    break
                cx1, cy1, cx2, cy2 = b
                tmpl = gray[cy1:cy2, cx1:cx2].copy()
            k = (cx2 - cx1) / cw0
            x1, y1 = int(cx1 + off[0] * k), int(cy1 + off[1] * k)
            x2, y2 = int(cx1 + off[2] * k), int(cy1 + off[3] * k)
            state, g, r = lamp_state(img[y1:y2, x1:x2])
            if state:
                out[i] = [int(x1), int(y1), int(x2), int(y2), state]
    os.makedirs(LAB, exist_ok=True)
    json.dump(out, open(os.path.join(LAB, video + ".json"), "w"))
    qa_sheet(video, frames, out)
    n_red = sum(1 for v in out.values() if v[4] == "red")
    print(f"{video}: 標註 {len(out)} 張（紅 {n_red}、綠 {len(out) - n_red}），追丟於 {lost_at or '無'}")
    return out


def qa_sheet(video, frames, out, every=4, cols=12, cell=110):
    keys = sorted(out)[::every]
    if not keys:
        return
    rows = (len(keys) + cols - 1) // cols
    sheet = np.full((rows * (cell + 18), cols * cell, 3), 255, np.uint8)
    for j, i in enumerate(keys):
        x1, y1, x2, y2, st = out[i]
        img = imread(os.path.join(RAW, video, frames[i]))
        pw, ph = (x2 - x1), (y2 - y1) // 2
        c = img[max(0, y1 - ph):y2 + ph, max(0, x1 - pw):x2 + pw]
        s = (cell - 4) / max(c.shape[:2])
        c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))))
        r, q = divmod(j, cols)
        y, x = r * (cell + 18), q * cell
        sheet[y + 2:y + 2 + c.shape[0], x + 2:x + 2 + c.shape[1]] = c
        color = (0, 0, 220) if st == "red" else (0, 170, 0)
        cv2.putText(sheet, f"{i} {st[0].upper()}", (x + 3, y + cell + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    os.makedirs(QA, exist_ok=True)
    cv2.imencode(".png", sheet)[1].tofile(os.path.join(QA, video + ".png"))
