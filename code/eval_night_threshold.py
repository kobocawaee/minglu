# -*- coding: utf-8 -*-
"""Sensitivity of the night guard to its brightness threshold.

The advisor asked us to avoid unjustified predefined thresholds, or to run a
sensitivity analysis for the ones we keep. This is that analysis for the most
safety-critical constant in the system: the mean grey level below which street
mode refuses to name a light colour unless YOLO sees a light box in the same
frame.

It was prompted by a discrepancy rather than by the request. app/street_mode.py
sets _NIGHT_LEVEL = 80.0, but the docstring beside the constant says 90 and
records the calibration as "daytime clips >= 97, night scenes <= 86" -- if night
really reaches 86, then 80 leaves part of the night unguarded. Rather than argue
from the comment, this measures the mean grey level of all 175 scored field
frames straight from the source clips and sweeps the threshold.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import csv
import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader(open(REPO / "results/field_multi_intersection.csv",
                                encoding="utf-8-sig")))
NIGHT_CLIPS = {"crosssing3", "crosssing4", "crosssingnolight1", "croossingnolight2"}

by_clip = {}
for r in rows:
    by_clip.setdefault(r["clip"], []).append(r)

print(f"{'clip':22} {'frames':>6} {'min':>6} {'mean':>6} {'max':>6}   frames with gray >= 80")
print("-" * 78)
summary = {}
for clip, items in sorted(by_clip.items()):
    path = REPO / "video" / f"{clip}.mp4"
    if not path.exists():
        print(f"{clip:22} (no video)")
        continue
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    grays = []
    for r in items:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(float(r["sec"]) * fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        g = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        grays.append((g, r))
    cap.release()
    if not grays:
        continue
    vals = np.array([g for g, _ in grays])
    over = int((vals >= 80).sum())
    summary[clip] = grays
    print(f"{clip:22} {len(vals):6d} {vals.min():6.1f} {vals.mean():6.1f} "
          f"{vals.max():6.1f}   {over:3d}/{len(vals)}")

print("\nnight clips only, per threshold: how many frames escape the guard")
night = [(g, r) for c in NIGHT_CLIPS for g, r in summary.get(c, [])]
print(f"night frames measured: {len(night)}")
for thr in (70, 75, 80, 85, 90, 95, 100):
    escaped = [(g, r) for g, r in night if g >= thr]
    said = [r for _, r in escaped if r["said"].strip()]
    false_clear = [r for _, r in escaped
                   if "green" in r["said"].lower() and r["gt"].strip().lower() != "green"]
    print(f"  threshold {thr:3d}: {len(escaped):3d} frames escape, "
          f"{len(said):3d} spoke a colour, {len(false_clear):3d} false 'green'")

print("\nday clips: how many frames a higher threshold would wrongly silence")
day = [(g, r) for c, items in summary.items() if c not in NIGHT_CLIPS for g, r in items]
print(f"day frames measured: {len(day)}")
for thr in (70, 75, 80, 85, 90, 95, 100):
    caught = [(g, r) for g, r in day if g < thr]
    spoke = [r for _, r in caught if r["said"].strip()]
    print(f"  threshold {thr:3d}: {len(caught):3d} day frames treated as night, "
          f"{len(spoke):3d} of them had spoken a colour")
