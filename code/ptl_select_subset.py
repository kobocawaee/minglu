"""
เลือก subset สมดุลจาก ImVisible/PTL (LYTNet) annotation → manifest schema เดียวกับ
data/dataset_manifest.csv พร้อม ground_truth ของ street-crossing ที่ derive จาก class label

ข้อได้เปรียบ: class ไฟจราจรคนข้าม (red/green/...) = ground-truth ความปลอดภัยโดยตรง
→ ลด manual judgment สำหรับ scenario street (ดู docs/street_crossing_dataset_prep.md)

วิธีใช้ (หลังโหลด annotation จาก github.com/samuelyu2002/ImVisible):
    python code/ptl_select_subset.py path/to/training_file.csv
    python code/ptl_select_subset.py path/to/training_file.csv --per-class 12 --seed 42 \
        --image-col file --class-col mode

⚠️ ต้อง confirm การ map "เลข class -> ชื่อไฟ" กับ README ของ repo ก่อนเชื่อ ground_truth
   (ดีฟอลต์ด้านล่างอิงลำดับที่พบบ่อยใน LYTNet — verify ก่อนลงเล่ม)
"""
import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parent.parent

# ⚠️ VERIFY กับ repo README — ลำดับ class index ที่ LYTNet ใช้บ่อย (0..4)
CLASS_MAP = {
    "0": ("red", "NOT SAFE — red pedestrian light, must wait"),
    "1": ("green", "SAFE — green pedestrian light, may cross"),
    "2": ("countdown_green", "CAUTION — green countdown running, cross only if enough time"),
    "3": ("countdown_blank", "NOT SAFE/CAUTION — pedestrian signal not clearly 'walk'"),
    "4": ("none", "no pedestrian signal present — rely on other cues"),
}
# รองรับกรณี annotation เก็บเป็นชื่อ string ตรง ๆ ด้วย
ALIASES = {
    "red": "0", "green": "1", "countdown_green": "2", "green_countdown": "2",
    "countdown_blank": "3", "blank_countdown": "3", "none": "4", "off": "4",
}


def norm_class(raw):
    raw = str(raw).strip().lower()
    if raw in CLASS_MAP:
        return raw
    return ALIASES.get(raw)  # None ถ้า map ไม่ได้


def sniff_rows(path):
    """อ่านไฟล์ annotation เป็น list[dict] รองรับทั้ง comma/space-separated + มี/ไม่มี header"""
    text = Path(path).read_text(encoding="utf-8-sig").splitlines()  # utf-8-sig ตัด BOM (ไฟล์ PTL มี BOM)
    if not text:
        return []
    delim = "," if "," in text[0] else None  # None = whitespace split
    first = (text[0].split(",") if delim else text[0].split())
    has_header = not any(c.strip().isdigit() for c in first[1:2])  # col2 เป็นเลข = ไม่มี header
    rows = []
    if has_header:
        header = [h.strip() for h in first]
        for line in text[1:]:
            cells = (line.split(",") if delim else line.split())
            if cells:
                rows.append(dict(zip(header, [c.strip() for c in cells])))
    else:
        for line in text:
            cells = (line.split(",") if delim else line.split())
            if cells:
                rows.append({"_0": cells[0].strip(), "_1": cells[1].strip() if len(cells) > 1 else ""})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation_path", help="PTL annotation file (training_file.csv ฯลฯ)")
    ap.add_argument("--per-class", type=int, default=12, help="จำนวนรูปต่อ class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image-col", default="_0", help="ชื่อคอลัมน์ไฟล์รูป (ดีฟอลต์ col แรก)")
    ap.add_argument("--class-col", default="_1", help="ชื่อคอลัมน์ class (ดีฟอลต์ col ที่สอง)")
    ap.add_argument("--out", default=str(ROOT / "data" / "ptl_subset_candidate.csv"))
    args = ap.parse_args()

    rows = sniff_rows(args.annotation_path)
    buckets = defaultdict(list)
    skipped = 0
    for r in rows:
        fn = r.get(args.image_col)
        cls = norm_class(r.get(args.class_col, ""))
        if not fn or cls is None:
            skipped += 1
            continue
        buckets[cls].append(fn)

    random.seed(args.seed)
    out_rows = []
    for cls, (name, gt) in CLASS_MAP.items():
        pool = buckets.get(cls, [])
        random.shuffle(pool)
        for fn in pool[: args.per_class]:
            out_rows.append({
                "filename": fn,
                "scenario": "street",
                "description": f"pedestrian traffic light = {name}; zebra crossing in view",
                "key_objects": "pedestrian traffic light, crosswalk",
                "ground_truth": gt,                          # derive จาก class label
                "source_type": "ImVisible/PTL (LYTNet)",
                "license_risk": "LOW (MIT)",
                "resolution": "",                            # ทราบหลังโหลดรูป
                "format": "JPEG",
                "ptl_class": name,
                "gt_source": "auto-from-label" if cls in ("0", "1", "2") else "label-but-verify",
            })

    out = Path(args.out)
    fields = ["filename", "scenario", "description", "key_objects", "ground_truth",
              "source_type", "license_risk", "resolution", "format", "ptl_class", "gt_source"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    avail = {CLASS_MAP[c][0]: len(buckets.get(c, [])) for c in CLASS_MAP}
    print(f"Available per class: {avail}  (skipped {skipped} unparseable rows)")
    print(f"Wrote {len(out_rows)} rows -> {out}")
    print("⚠️ VERIFY class-index->name mapping vs repo README before trusting ground_truth.")
    print("   red/green/countdown_green = ground-truth ชัด; countdown_blank/none ควร manual review")


if __name__ == "__main__":
    main()
