# -*- coding: utf-8 -*-
"""Score the field clips end to end through the deployed street channel.

Promoted from the scratchpad script that produced results/field_multi_intersection.csv
on 21 July, with two changes: paths are relative, and the night-guard threshold is
a parameter so the same code can score both values and the comparison is
apples to apples.

    python code/score_field_clips.py                      # deployed threshold
    python code/score_field_clips.py --night-level 90     # try another
    python code/score_field_clips.py --night-level 80 --out results/field_T80.csv

Deliberately *not* changed: LightHistory ages its window against the wall clock,
and this loop runs far faster than real time, so the 8-second window behaves as
"the last few predictions" rather than a true 8 seconds. That was true of the
original scoring too. Introducing a simulated clock would change the numbers for
a reason unrelated to the threshold and make the two runs incomparable.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
from PIL import Image

from app import config, street_mode, light_classifier as lc

config.LANG = "en"   # [資服版] 評分靠比對英文句子，所以固定用英文輸出

# Ground truth read by hand from magnified per-second crops, before the pipeline
# was ever run on these clips. Ranges are inclusive, in seconds.
GT = {
    "crosssing1": [(0, 48, "red"), (49, 50, "green")],
    "crosssing2": [(0, 7, "red"), (8, 11, "green")],
    "crosssing3": [(0, 11, "green"), (12, 13, "unknown")],
    "crosssing4": [(0, 18, "green"), (19, 27, "red")],
    "crosssingnolight1": [(0, 99, "none")],
    "croossingnolight2": [(0, 99, "none")],
    "video-testing": [(0, 40, "red"), (41, 44, "green")],   # the 14 July clip
}


def gt_at(clip, sec):
    for a, b, label in GT[clip]:
        if a <= sec <= b:
            return label
    return "unknown"


def score(night_level):
    street_mode._NIGHT_LEVEL = night_level
    rows = []
    for clip in GT:
        path = ROOT / "video" / f"{clip}.mp4"
        if not path.exists():
            print(f"  missing {path.name}, skipped")
            continue
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps)))
        street_mode._LIGHT_HISTORY = lc.LightHistory()      # fresh memory per clip
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                sec = int(round(i / fps))
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if img.height > 1200:        # the 8K clip, downscaled like a phone stream
                    img = img.resize((int(img.width * 1000 / img.height), 1000))
                text, _hazard, _info = street_mode.describe(None, img)
                low = text.lower()
                said = ("red" if "light is red" in low
                        else "green" if "light is green" in low else "")
                rows.append(dict(clip=clip, sec=sec, gt=gt_at(clip, sec),
                                 said=said, text=text))
            i += 1
        cap.release()
    return rows


def summarise(rows, label):
    lit = [r for r in rows if r["gt"] in ("red", "green")]
    nolight = [r for r in rows if r["gt"] == "none"]
    spoke = [r for r in lit if r["said"]]
    wrong = [r for r in spoke if r["said"] != r["gt"]]
    danger = [r for r in wrong if r["gt"] == "red" and r["said"] == "green"]
    false_colour = [r for r in nolight if r["said"]]

    print(f"\n=== {label} ===")
    print(f"{'clip':22} {'lit':>5} {'said':>5} {'wrong':>6} {'danger':>7}")
    for clip in GT:
        cr = [r for r in rows if r["clip"] == clip]
        cl = [r for r in cr if r["gt"] in ("red", "green")]
        cs = [r for r in cl if r["said"]]
        cw = [r for r in cs if r["said"] != r["gt"]]
        cd = [r for r in cw if r["gt"] == "red" and r["said"] == "green"]
        if cl:
            print(f"{clip:22} {len(cl):5d} {len(cs):5d} {len(cw):6d} {len(cd):7d}")
        else:
            fc = sum(1 for r in cr if r["said"])
            print(f"{clip:22} {'no light':>5} {fc:5d} false colour of {len(cr)}")
    pct = 100 * len(spoke) / len(lit) if lit else 0
    print(f"TOTAL  light present {len(lit)} · announced {len(spoke)} ({pct:.0f}%) "
          f"· wrong {len(wrong)} · DANGEROUS {len(danger)}")
    print(f"       no-light frames {len(nolight)} · false colour {len(false_colour)}")
    return {"lit": len(lit), "said": len(spoke), "wrong": len(wrong),
            "danger": len(danger), "nolight": len(nolight),
            "false_colour": len(false_colour)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--night-level", type=float, default=street_mode._NIGHT_LEVEL)
    ap.add_argument("--out", default="results/field_multi_intersection.csv")
    args = ap.parse_args()

    print(f"night guard threshold: mean grey < {args.night_level:g}")
    rows = score(args.night_level)
    out = ROOT / args.out
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    summarise(rows, f"night guard at {args.night_level:g}")
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
