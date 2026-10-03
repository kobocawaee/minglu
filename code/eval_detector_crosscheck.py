# -*- coding: utf-8 -*-
"""Can a detector catch the VLM's fabrications before they are spoken?

Section 4.3 reports that all three models fabricate on 4 of 17 images, and that
scale does not change it. This asks whether the fabrications are *catchable*
rather than preventable, by checking what the VLM says against what the detectors
we already run can see.

The idea is the inverse of a guard the assistant already has. app/pipeline.py
adds a warning when YOLO sees a person the VLM did not mention. Nothing does the
opposite: nothing removes a claim about a person YOLO cannot find.

Looking at the 12 fabrications, six invent a person, three invent readable text
and three invent a vehicle or hazard. All three are things YOLOv8n or RapidOCR
can be asked about, so the question is worth measuring rather than assuming.

Two levels are reported, because they answer different questions:

  oracle      claims are checked against the hand-written ground truth for each
              image. This is the ceiling: how many fabrications are catchable in
              principle by this kind of check.
  deployable  claims are checked against what YOLOv8n and RapidOCR actually
              return on the image. This is what a shipped system would get, and
              the gap between the two is the detector's own error.

Both are scored the same way: of the 12 known fabrications, how many are flagged
(recall), and of the 39 outputs judged clean, how many are flagged anyway (false
alarms). A guard that flags everything scores perfect recall and is useless, so
both numbers have to be read together.

    python code/eval_detector_crosscheck.py

Writes results/detector_crosscheck.csv.
"""
import csv
import math
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image

from app import detect

IMAGES = ROOT / "data/test_images"
OUT = ROOT / "results/detector_crosscheck.csv"

# What the assistant might claim, and the words that signal each claim.
PERSON = re.compile(r"\b(person|people|pedestrian|man|woman|men|women|someone|"
                    r"child|children|kid|boy|girl|crowd|guard|wheelchair)\b", re.I)
VEHICLE = re.compile(r"\b(car|cars|truck|bus|van|motorcycle|bicycle|vehicle|vehicles)\b", re.I)
# A claim to have read specific text: quoted text, or capitalised strings, or digits
TEXT_CLAIM = re.compile(r"[\"“'']([^\"“'']{2,40})[\"“'']|\b([A-Z]{3,}[A-Z\s]{2,})\b")

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * max(0.0, c - m), 100 * min(1.0, c + m)


def detector_view(path):
    """What YOLO can see in this image, at the confidences the app uses."""
    img = Image.open(path).convert("RGB")
    dets, (w, h) = detect.detect(img, conf=0.35, classes="all")
    classes = {d["cls"] for d in dets}
    person = any(d["cls"] == detect.PERSON_CLASS and
                 ((d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])) / (w * h) > 0.004
                 for d in dets)
    from app import ocr
    try:
        found = ocr.read_text(img) or ""
    except Exception as e:                       # OCR is optional, never fatal
        found = ""
        print(f"    (OCR unavailable: {e})")
    return {"person": person, "vehicle": bool(classes & VEHICLE_CLASSES),
            "classes": sorted(classes),
            "text": re.sub(r"[^a-z0-9]", "", found.lower())}


def oracle_view(row):
    """What the hand-written ground truth says the image contains."""
    blob = (row["description"] + " " + row["key_objects"] + " " + row["ground_truth"]).lower()
    return {"person": bool(PERSON.search(blob)), "vehicle": bool(VEHICLE.search(blob)),
            "text": re.sub(r"[^a-z0-9]", "", blob)}


def quoted_strings(text):
    """Strings the output presents as read off the scene: quoted, or shouted in
    capitals. Three of the twelve fabrications are of this kind ("Exit 647",
    "7TH STREET", a clock face), so a check that ignores them can reach at best
    three quarters of the target."""
    out = []
    for m in TEXT_CLAIM.finditer(text):
        s = (m.group(1) or m.group(2) or "").strip()
        if len(s) >= 3:
            out.append(s)
    return out


def unsupported(text, view):
    """Which claims in this output the view cannot support."""
    flags = []
    if PERSON.search(text) and not view["person"]:
        flags.append("person")
    if VEHICLE.search(text) and not view["vehicle"]:
        flags.append("vehicle")
    for s in quoted_strings(text):
        # the claimed string has to appear in what the reader of the scene found;
        # compare on letters and digits only, since OCR spacing is unreliable
        want = re.sub(r"[^a-z0-9]", "", s.lower())
        if want and want not in view.get("text", ""):
            flags.append(f"text:{s[:22]}")
            break
    return flags


def score(rows, key, label):
    fab = [r for r in rows if r["is_fabrication"]]
    clean = [r for r in rows if not r["is_fabrication"]]
    caught = [r for r in fab if r[key]]
    alarms = [r for r in clean if r[key]]
    lo, hi = wilson(len(caught), len(fab))
    flo, fhi = wilson(len(alarms), len(clean))
    print(f"\n  {label}")
    print(f"    caught      {len(caught):2d}/{len(fab)}  "
          f"({100*len(caught)/len(fab):.0f}%, 95% CI [{lo:.0f}, {hi:.0f}])")
    print(f"    false alarm {len(alarms):2d}/{len(clean)}  "
          f"({100*len(alarms)/len(clean):.0f}%, 95% CI [{flo:.0f}, {fhi:.0f}])")
    return len(caught), len(fab), len(alarms), len(clean)


def main():
    manifest = {r["filename"]: r for r in
                csv.DictReader(open(ROOT / "data/dataset_manifest.csv", encoding="utf-8-sig"))}
    outputs = list(csv.DictReader(open(ROOT / "data/safety_eval.csv", encoding="utf-8-sig")))

    print("running YOLOv8n on the 17 canonical images ...")
    seen = {}
    for name in manifest:
        path = IMAGES / name
        if path.exists():
            seen[name] = detector_view(path)
        else:
            print(f"  missing {name}")

    rows = []
    for o in outputs:
        name = o["filename"]
        if name not in seen or name not in manifest:
            continue
        text = o["output"]
        orc = unsupported(text, oracle_view(manifest[name]))
        dep = unsupported(text, seen[name])
        rows.append({
            "filename": name, "model": o["model"],
            "is_fabrication": o["safety_critical_halluc"] == "Y",
            "oracle_flag": bool(orc), "oracle_why": " ".join(orc),
            "deploy_flag": bool(dep), "deploy_why": " ".join(dep),
            "yolo_sees": " ".join(seen[name]["classes"]),
            "output": text[:200],
        })

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print(f"\noutputs checked: {len(rows)}  "
          f"(fabrications {sum(r['is_fabrication'] for r in rows)})")
    score(rows, "oracle_flag", "oracle: claims checked against hand-written ground truth")
    score(rows, "deploy_flag", "deployable: claims checked against YOLOv8n")

    print("\n  fabrications the deployable check misses:")
    for r in rows:
        if r["is_fabrication"] and not r["deploy_flag"]:
            print(f"    {r['filename'][:28]:30} {r['output'][:70]}")
    print(f"\nsaved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
