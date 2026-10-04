"""
從 ImVisible/PTL（LYTNet）的標註挑出各類別平均的子集 → 輸出和
data/dataset_manifest.csv 相同格式的清單，附上由類別標籤推得的過馬路正確答案

優點：行人號誌的類別（red/green/...）直接就是安全性的正確答案
→ 減少過馬路情境的人工判斷（見 docs/street_crossing_dataset_prep.md）

使用方式（先從 github.com/samuelyu2002/ImVisible 下載標註）：
    python code/ptl_select_subset.py path/to/training_file.csv
    python code/ptl_select_subset.py path/to/training_file.csv --per-class 12 --seed 42 \
        --image-col file --class-col mode

⚠️ 採信正確答案之前，必須先對照 repo 的 README 確認「類別編號 -> 燈號名稱」的對應
   （下方預設值依 LYTNet 常見的順序 — 寫進論文前要先驗證）
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

# ⚠️ 要和 repo README 核對 — LYTNet 常用的類別索引順序（0..4）
CLASS_MAP = {
    "0": ("red", "NOT SAFE — red pedestrian light, must wait"),
    "1": ("green", "SAFE — green pedestrian light, may cross"),
    "2": ("countdown_green", "CAUTION — green countdown running, cross only if enough time"),
    "3": ("countdown_blank", "NOT SAFE/CAUTION — pedestrian signal not clearly 'walk'"),
    "4": ("none", "no pedestrian signal present — rely on other cues"),
}
# 也支援標註直接存成名稱字串的情況
ALIASES = {
    "red": "0", "green": "1", "countdown_green": "2", "green_countdown": "2",
    "countdown_blank": "3", "blank_countdown": "3", "none": "4", "off": "4",
}


def norm_class(raw):
    raw = str(raw).strip().lower()
    if raw in CLASS_MAP:
        return raw
    return ALIASES.get(raw)  # 對應不到時回傳 None


def sniff_rows(path):
    """把標註檔讀成 list[dict]，支援逗號／空白分隔＋有無標題列"""
    text = Path(path).read_text(encoding="utf-8-sig").splitlines()  # utf-8-sig 去掉 BOM（PTL 檔案有 BOM）
    if not text:
        return []
    delim = "," if "," in text[0] else None  # None = whitespace split
    first = (text[0].split(",") if delim else text[0].split())
    has_header = not any(c.strip().isdigit() for c in first[1:2])  # 第 2 欄是數字 = 沒有標題列
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
    ap.add_argument("annotation_path", help="PTL 標註檔（training_file.csv 等）")
    ap.add_argument("--per-class", type=int, default=12, help="每個類別幾張圖")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image-col", default="_0", help="圖檔欄位名稱（預設第 1 欄）")
    ap.add_argument("--class-col", default="_1", help="類別欄位名稱（預設第 2 欄）")
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
                "ground_truth": gt,                          # 由類別標籤推得
                "source_type": "ImVisible/PTL (LYTNet)",
                "license_risk": "LOW (MIT)",
                "resolution": "",                            # 載入圖片後才知道
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
    print("   red/green/countdown_green = 正確答案明確；countdown_blank/none 應人工檢查")


if __name__ == "__main__":
    main()
