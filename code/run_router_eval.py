"""
run_router_eval.py — รัน auto-mode router (app/pipeline.route) บนชุด labeled ~60 รูป
(คนละตัวกับ code/eval_router.py ของ Aomam ซึ่งเป็นตัว *วิเคราะห์* CSV ที่ไฟล์นี้สร้าง)

แผนตาม handoff 07-10 (Aomam confirm แล้ว) — ไม่ต้องถ่ายรูปใหม่ ทุกใบ license สะอาด:
  street 20      = PTL subset (มุมมองคนเดินที่ทางข้าม -> true=street แน่นอน)  [MIT]
  indoor 15      = VizWiz candidate scenario=indoor                          [CC BY 4.0]
  surrounding 15 = VizWiz candidate scenario=surrounding                     [CC BY 4.0]
  object 10      = VizWiz close-up (keyword ที่ script คัด scene ทิ้ง — เลือกกลับมา) [CC BY 4.0]
หมายเหตุ: read ไม่อยู่ใน eval นี้ (read = manual intent, ไม่ route จากภาพ — ดู app/pipeline.py)

output: results/router_eval.csv (filename,true_mode,routed_mode,correct)
        -> วิเคราะห์ต่อ: python code/eval_router.py results/router_eval.csv (script ของ Aomam)

รัน (env vlm_research): python code/run_router_eval.py
"""

import sys, os, csv, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PTL_CSV = ROOT / "data/ptl/ptl_subset150.csv"
PTL_IMGDIR = ROOT / "data/ptl/images/PTL_Dataset_876x657"
VIZ_MANIFEST = ROOT / "data/vizwiz_subset_candidate.csv"
VIZ_IMGDIR = ROOT / "data/vizwiz/images"
VAL_JSON = ROOT / "data/vizwiz/annotations/val.json"
OUT = ROOT / "results/router_eval.csv"

CLOSEUP_KW = ["bottle", " can ", "box of", "package", "remote", "cell phone", "phone",
              "jar", "container", "cup ", "mug", "book cover", "cd ", "dvd"]
SEED = 42


def pick_street(n=20):
    rows = list(csv.DictReader(open(PTL_CSV, encoding="utf-8-sig")))
    rng = random.Random(SEED)
    # คละคลาสไฟ (แดง/เขียว) ให้หลากหลายฉาก
    reds = [r for r in rows if r["ptl_class"] == "red"]
    greens = [r for r in rows if r["ptl_class"] == "green"]
    picked = rng.sample(reds, n // 2) + rng.sample(greens, n - n // 2)
    return [(PTL_IMGDIR / r["filename"], "street") for r in picked]


def pick_vizwiz_scenes():
    rows = list(csv.DictReader(open(VIZ_MANIFEST, encoding="utf-8-sig")))
    out = []
    for r in rows:
        sc = r["scenario"]
        if sc in ("indoor", "surrounding"):
            out.append((VIZ_IMGDIR / os.path.basename(r["filename"]), sc))
    return out


def pick_object(n=10):
    """เลือกรูป close-up จาก val.json (caption มี keyword ของถือได้) + ดึงจาก HF ถ้ายังไม่มี"""
    data = json.load(open(VAL_JSON, encoding="utf-8"))
    id2name = {im["id"]: os.path.basename(im["file_name"]) for im in data["images"]}
    cands = {}
    for a in data["annotations"]:
        if a.get("is_rejected") or a.get("is_precanned"):
            continue
        cap = " " + a["caption"].lower() + " "
        if any(k in cap for k in CLOSEUP_KW):
            cands.setdefault(a["image_id"], 0)
            cands[a["image_id"]] += 1
    # เอาเฉพาะรูปที่ >=2 captions บอกว่าเป็นของ close-up (มั่นใจกว่า)
    strong = sorted([i for i, c in cands.items() if c >= 2])
    rng = random.Random(SEED)
    chosen = rng.sample(strong, min(n * 2, len(strong)))[:n * 2]  # เผื่อดึงไม่เจอ

    VIZ_IMGDIR.mkdir(parents=True, exist_ok=True)
    have = [i for i in chosen if (VIZ_IMGDIR / id2name[i]).exists()]
    need = [i for i in chosen if i not in have]
    got = list(have)
    if need and len(got) < n:
        print(f"[info] streaming up to {n - len(got)} object images from HF ...")
        from datasets import load_dataset
        ds = load_dataset("lmms-lab/VizWiz-Caps", split="val", streaming=True)
        need_set = set(need)
        for ex in ds:
            iid = int(ex["image_id"])
            if iid in need_set:
                ex["image"].convert("RGB").save(VIZ_IMGDIR / id2name[iid])
                got.append(iid); need_set.discard(iid)
                print(f"  [{len(got)}] {id2name[iid]}")
            if len(got) >= n:
                break
    return [(VIZ_IMGDIR / id2name[i], "object") for i in got[:n]]


def main():
    from PIL import Image
    from app import pipeline

    cases = pick_street(20) + pick_vizwiz_scenes() + pick_object(10)
    print(f"[info] labeled set: {len(cases)} images")

    rows, correct = [], 0
    from collections import Counter
    per_mode = Counter(); per_mode_ok = Counter()
    for i, (path, true_mode) in enumerate(cases):
        if not path.exists():
            print(f"  skip missing: {path.name}"); continue
        im = Image.open(path).convert("RGB")
        routed, _dets = pipeline.route(im)
        ok = routed == true_mode
        correct += ok; per_mode[true_mode] += 1; per_mode_ok[true_mode] += ok
        rows.append({"filename": path.name, "true_mode": true_mode,
                     "routed_mode": routed, "correct": ok})
        print(f"  [{i+1}/{len(cases)}] {path.name:34s} true={true_mode:12s} routed={routed:12s} {'OK' if ok else 'X'}")

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "true_mode", "routed_mode", "correct"])
        w.writeheader(); w.writerows(rows)

    n = len(rows)
    print("\n" + "=" * 60)
    print(f"routing accuracy: {correct}/{n} = {100*correct/max(n,1):.0f}%")
    for m in ("street", "indoor", "surrounding", "object"):
        if per_mode[m]:
            print(f"  {m:12s} recall: {per_mode_ok[m]}/{per_mode[m]}"
                  + ("   <-- STREET-RECALL (L4)" if m == "street" else ""))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
