# -*- coding: utf-8 -*-
"""Blur the faces of bystanders in a field frame before it is published.

The people in our street footage did not consent to appear in a paper. Figure 6's
screenshots were blurred by hand; this does the same job reproducibly, so the
published image can be regenerated from the source clip at any time and the
unblurred original never has to be committed.

The detector is run at a deliberately low confidence, because a distant
pedestrian is exactly the detection most likely to be missed and exactly the one
that still shows a face. The blurred region is the head: the top fraction of each
person box, widened, since a box that is slightly wrong should still cover the
face.

    python code/blur_faces.py in.png out.png [--conf 0.15] [--check]

--check writes a copy with the blurred regions outlined, to confirm by eye that
every person was covered before the clean version is used.
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
from PIL import Image

from app import detect

HEAD_FRACTION = 0.42   # of the person box height, from the top
WIDEN = 0.30           # of the box width, added on each side


def blur_people(img_bgr, conf: float):
    """Return (blurred image, list of blurred rectangles)."""
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    dets, (w, h) = detect.detect(pil, conf=conf, classes={detect.PERSON_CLASS})
    out = img_bgr.copy()
    boxes = []
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        pad = int((x2 - x1) * WIDEN)
        hx1 = max(0, x1 - pad)
        hx2 = min(w, x2 + pad)
        hy1 = max(0, y1 - pad // 2)
        hy2 = min(h, y1 + int((y2 - y1) * HEAD_FRACTION) + pad // 2)
        if hx2 - hx1 < 3 or hy2 - hy1 < 3:
            continue
        region = out[hy1:hy2, hx1:hx2]
        # kernel scaled to the region, and forced odd, so a small face is blurred
        # as thoroughly as a large one rather than merely softened
        k = max(9, (min(region.shape[:2]) // 2) | 1)
        out[hy1:hy2, hx1:hx2] = cv2.GaussianBlur(region, (k, k), 0)
        boxes.append((hx1, hy1, hx2, hy2, d["conf"]))
    return out, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.src)
    if img is None:
        sys.exit(f"cannot read {args.src}")
    out, boxes = blur_people(img, args.conf)
    cv2.imwrite(args.dst, out)
    print(f"{Path(args.src).name}: blurred {len(boxes)} head region(s) at conf >= {args.conf}")
    for x1, y1, x2, y2, c in boxes:
        print(f"    ({x1:4},{y1:4})-({x2:4},{y2:4})  person conf {c:.2f}")
    print(f"saved {args.dst}")

    if args.check:
        marked = out.copy()
        for x1, y1, x2, y2, _ in boxes:
            cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 200, 255), 2)
        chk = str(Path(args.dst).with_name(Path(args.dst).stem + "_check.png"))
        cv2.imwrite(chk, marked)
        print(f"saved {chk} for visual confirmation")


if __name__ == "__main__":
    main()
