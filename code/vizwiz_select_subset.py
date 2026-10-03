"""
เลือก subset จาก VizWiz-Captions annotation JSON ให้คลุม 3 use case (street/indoor/surrounding)
สำหรับ final eval — เขียน candidate manifest schema เดียวกับ data/dataset_manifest.csv

ทำไมต้องมี script นี้: VizWiz ไม่มี label "scenario" + ส่วนใหญ่เป็นรูป close-up วัตถุ
(คนตาบอดถ่ายเพื่อระบุของ) ไม่ใช่ scene → ต้อง bucket ด้วย keyword จาก captions + กรอง close-up ออก
ผลที่ได้เป็น "candidate" ต้อง manual verify scenario + เขียน ground_truth/safety เองทีละรูป

วิธีใช้ (หลังโหลด annotations.zip จาก vizwiz.org แล้วได้ val.json):
    python code/vizwiz_select_subset.py path/to/val.json
    python code/vizwiz_select_subset.py path/to/val.json --per-scenario 20 --seed 42

ดูบริบทเต็มใน docs/vizwiz_prep.md
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

try:  # กัน UnicodeEncodeError บน Windows console (cp1252) เวลา print ไทย/emoji
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parent.parent

# keyword สำหรับ bucket scenario (จับจาก captions ที่ผ่านการกรอง)
SCENARIO_KEYWORDS = {
    "street": ["street", "road", "crosswalk", "cross walk", "traffic", "sidewalk",
               "intersection", "vehicle", " car ", "cars", "bus", "crossing", "pavement",
               "parking lot", "highway", "lane"],
    "indoor": ["room", "kitchen", "indoor", "inside", "wall", "floor", "desk", "counter",
               "sofa", "couch", "bed", "refrigerator", "fridge", "microwave", "cabinet",
               "table", "stairs", "staircase", "hallway", "living room", "office"],
    "surrounding": ["outside", "outdoor", "park", "tree", "trees", "grass", "building",
                    "sky", "path", "walkway", "people", "crowd", "garden", "yard",
                    "field", "fence", "bench"],
}

# รูป close-up วัตถุ/เอกสาร — ตัดออก (ไม่ใช่ scene สำหรับ navigation)
CLOSEUP_KEYWORDS = ["screen", "label", "bottle", " can ", "cans", "box", "package",
                    "document", "paper", "book", "remote", "monitor", "phone", "card",
                    "bag of", "jar", "package of", "menu", "receipt", "barcode", "tag",
                    "container", "wrapper", "close-up", "close up", "hand holding"]

PRECANNED = "quality issues are too severe"


def load_captions(json_path):
    """อ่าน VizWiz caption JSON (COCO-style) → {file_name: [captions ที่ผ่านการกรอง]}"""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    id2file = {img["id"]: img["file_name"] for img in data.get("images", [])}
    caps = defaultdict(list)
    text_flag = defaultdict(bool)
    for ann in data.get("annotations", []):
        # กรอง spam / precanned (รูปเบลอ ดูไม่ออก) ตามที่ eval ทางการทำ
        if ann.get("is_rejected") or ann.get("is_precanned"):
            continue
        cap = (ann.get("caption") or "").strip()
        if not cap or PRECANNED in cap.lower():
            continue
        fn = id2file.get(ann["image_id"])
        if fn:
            caps[fn].append(cap)
            if ann.get("text_detected"):
                text_flag[fn] = True
    return caps, text_flag


def classify(captions):
    """คืน scenario ที่เข้ากับ captions มากสุด หรือ None ถ้าเป็น close-up/ไม่เข้าเกณฑ์"""
    blob = " " + " ".join(captions).lower() + " "
    closeup_hits = sum(blob.count(k) for k in CLOSEUP_KEYWORDS)
    scores = {sc: sum(blob.count(k) for k in kws) for sc, kws in SCENARIO_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    # ต้องมี scene keyword ชัด และไม่ถูก close-up ครอบงำ
    if scores[best] == 0:
        return None
    if closeup_hits > scores[best]:
        return None
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="VizWiz caption annotation JSON (เช่น val.json)")
    ap.add_argument("--per-scenario", type=int, default=20, help="จำนวนรูปต่อ scenario")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data" / "vizwiz_subset_candidate.csv"))
    args = ap.parse_args()

    caps, text_flag = load_captions(args.json_path)
    buckets = defaultdict(list)
    for fn, cs in caps.items():
        sc = classify(cs)
        if sc:
            buckets[sc].append((fn, cs))

    random.seed(args.seed)
    rows = []
    for sc in ("street", "indoor", "surrounding"):
        pool = buckets.get(sc, [])
        random.shuffle(pool)
        picked = pool[: args.per_scenario]
        for fn, cs in picked:
            rows.append({
                "filename": fn,
                "scenario": sc,
                "description": " | ".join(cs[:5]),     # captions ตั้งต้น (ก่อน manual)
                "key_objects": "",                       # เติมเอง
                "ground_truth": "",                      # ⚠️ ต้องเขียนเอง (manual)
                "source_type": "VizWiz-Captions (val)",
                "license_risk": "LOW (CC BY 4.0)",
                "resolution": "",                        # ทราบหลังโหลดรูป
                "format": "JPEG",
                "text_in_image": "Y" if text_flag[fn] else "N",
                "needs_manual_review": "Y",              # scenario เป็น heuristic
            })

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["filename", "scenario", "description", "key_objects",
                            "ground_truth", "source_type", "license_risk", "resolution",
                            "format", "text_in_image", "needs_manual_review"])
        w.writeheader()
        w.writerows(rows)

    avail = {sc: len(buckets.get(sc, [])) for sc in ("street", "indoor", "surrounding")}
    print(f"Scene candidates available per scenario: {avail}")
    print(f"Wrote {len(rows)} rows -> {out}")
    print("⚠️ candidate only — manual-verify scenario + write ground_truth before use "
          "(see docs/vizwiz_prep.md §4-5)")


if __name__ == "__main__":
    main()
