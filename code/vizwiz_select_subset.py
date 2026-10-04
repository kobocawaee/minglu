"""
從 VizWiz-Captions 標註 JSON 挑出涵蓋 3 種使用情境（street/indoor/surrounding）的子集
用於最終評估 — 寫出和 data/dataset_manifest.csv 相同格式的候選清單

為什麼需要這支腳本：VizWiz 沒有「情境」標籤＋大多是物品特寫
（視障者拍來辨識東西）而不是場景 → 要用描述裡的關鍵字分類＋濾掉特寫
得到的只是「候選」，需要人工確認情境＋逐張填寫正確答案／安全性

使用方式（從 vizwiz.org 下載 annotations.zip 取得 val.json 後）：
    python code/vizwiz_select_subset.py path/to/val.json
    python code/vizwiz_select_subset.py path/to/val.json --per-scenario 20 --seed 42

完整背景見 docs/vizwiz_prep.md
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

try:  # 避免 Windows 主控台（cp1252）在 print 非英文字／emoji 時出現 UnicodeEncodeError
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parent.parent

# 分類情境用的關鍵字（從過濾後的描述中比對）
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

# 物品／文件特寫的圖 — 排除（不是導航用的場景）
CLOSEUP_KEYWORDS = ["screen", "label", "bottle", " can ", "cans", "box", "package",
                    "document", "paper", "book", "remote", "monitor", "phone", "card",
                    "bag of", "jar", "package of", "menu", "receipt", "barcode", "tag",
                    "container", "wrapper", "close-up", "close up", "hand holding"]

PRECANNED = "quality issues are too severe"


def load_captions(json_path):
    """讀 VizWiz 描述 JSON（COCO 格式）→ {file_name: [過濾後的描述]}"""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    id2file = {img["id"]: img["file_name"] for img in data.get("images", [])}
    caps = defaultdict(list)
    text_flag = defaultdict(bool)
    for ann in data.get("annotations", []):
        # 依官方評估的做法，濾掉垃圾／預設回覆（模糊、看不出來的圖）
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
    """回傳和描述最吻合的情境；如果是特寫或不符合條件則回傳 None"""
    blob = " " + " ".join(captions).lower() + " "
    closeup_hits = sum(blob.count(k) for k in CLOSEUP_KEYWORDS)
    scores = {sc: sum(blob.count(k) for k in kws) for sc, kws in SCENARIO_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    # 必須有明確的場景關鍵字，而且不能被特寫的特徵蓋過
    if scores[best] == 0:
        return None
    if closeup_hits > scores[best]:
        return None
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="VizWiz 描述標註 JSON（例如 val.json）")
    ap.add_argument("--per-scenario", type=int, default=20, help="每個情境幾張圖")
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
                "description": " | ".join(cs[:5]),     # 初始描述（人工檢查前）
                "key_objects": "",                       # 人工填寫
                "ground_truth": "",                      # ⚠️ 必須人工填寫
                "source_type": "VizWiz-Captions (val)",
                "license_risk": "LOW (CC BY 4.0)",
                "resolution": "",                        # 載入圖片後才知道
                "format": "JPEG",
                "text_in_image": "Y" if text_flag[fn] else "N",
                "needs_manual_review": "Y",              # 情境是經驗法則判斷的
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
