"""
vizwiz_generate_captions.py — ดึงรูป VizWiz subset + gen captions ส่งให้ Aomam รัน CIDEr/SPICE
==============================================================================================
ขั้นตอน:
  1. อ่าน candidate manifest (data/vizwiz_subset_candidate.csv จาก script ของ Aomam)
  2. ดึง "เฉพาะรูปที่เลือก" จาก HuggingFace lmms-lab/VizWiz-Caps (streaming — ไม่โหลด zip 39k)
     เซฟลง data/vizwiz/images/ (CC BY 4.0 — ใส่ attribution ในเล่มได้)
  3. รัน SmolVLM-500M 2 แบบต่อรูป:
       app     = prompt โหมด surrounding ของแอปจริง (วัด "แอปเราทำได้แค่ไหน")
       caption = prompt caption กลางๆ ("Describe this image in one sentence.")
                 (แฟร์กว่าเวลาเทียบ CIDEr กับ caption baselines)
  4. เขียน 2 ไฟล์ให้ Aomam:
       results/vizwiz_generated.csv   (filename, scenario, gen_app, gen_caption)
       results/vizwiz_references.csv  (filename, ref_1..ref_5 — กรอง rejected/precanned แล้ว)

รัน (env vlm_research, ~5-8 นาทีสำหรับ 45 รูป):
    python code/vizwiz_generate_captions.py
"""

import sys, csv, json, io, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MANIFEST = ROOT / "data/vizwiz_subset_candidate.csv"
VAL_JSON = ROOT / "data/vizwiz/annotations/val.json"
IMGDIR = ROOT / "data/vizwiz/images"
OUT_GEN = ROOT / "results/vizwiz_generated.csv"
OUT_REF = ROOT / "results/vizwiz_references.csv"

CAPTION_PROMPT = "Describe this image in one sentence."
PRECANNED = "quality issues are too severe"


def load_manifest():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))
    print(f"[info] manifest: {len(rows)} images")
    return rows


def fetch_images(wanted: set):
    """stream จาก HF แล้วเซฟเฉพาะรูปที่อยู่ใน manifest
    ⚠️ HF lmms-lab/VizWiz-Caps ไม่มี field ชื่อไฟล์ — มีแต่ image_id (ตรงกับ images[].id
    ใน val.json ทางการ) → สร้าง map id→file_name จาก val.json ก่อน"""
    IMGDIR.mkdir(parents=True, exist_ok=True)
    missing = {w for w in wanted if not (IMGDIR / w).exists()}
    if not missing:
        print("[info] all images already local")
        return
    data = json.load(open(VAL_JSON, encoding="utf-8"))
    id2name = {im["id"]: os.path.basename(im["file_name"]) for im in data["images"]}
    wanted_ids = {i for i, n in id2name.items() if n in missing}
    print(f"[info] streaming {len(missing)} images from HF (match by image_id) ...")
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/VizWiz-Caps", split="val", streaming=True)
    got = 0
    for ex in ds:
        iid = int(ex["image_id"])
        if iid in wanted_ids:
            fn = id2name[iid]
            ex["image"].convert("RGB").save(IMGDIR / fn)
            wanted_ids.discard(iid); missing.discard(fn); got += 1
            print(f"  [{got}] {fn}")
        if not wanted_ids:
            break
    if missing:
        print(f"⚠️ not found on HF: {sorted(missing)[:5]} (+{max(0,len(missing)-5)})")


def write_references(wanted: set):
    """อ่าน val.json → ref captions ต่อรูป (กรอง rejected/precanned ตาม protocol ทางการ)"""
    data = json.load(open(VAL_JSON, encoding="utf-8"))
    id2name = {im["id"]: os.path.basename(im["file_name"]) for im in data["images"]}
    refs = {}
    for a in data["annotations"]:
        fn = id2name.get(a["image_id"])
        if fn not in wanted:
            continue
        if a.get("is_rejected") or a.get("is_precanned"):
            continue
        if PRECANNED in a["caption"].lower():
            continue
        refs.setdefault(fn, []).append(a["caption"].strip())
    with open(OUT_REF, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["filename"] + [f"ref_{i+1}" for i in range(5)])
        for fn in sorted(wanted):
            r = (refs.get(fn, []) + [""] * 5)[:5]
            w.writerow([fn] + r)
    n_ok = sum(1 for fn in wanted if refs.get(fn))
    print(f"[info] references: {n_ok}/{len(wanted)} images have >=1 clean ref -> {OUT_REF}")


def main():
    rows = load_manifest()
    wanted = {os.path.basename(r["filename"]) for r in rows}
    fetch_images(wanted)
    write_references(wanted)

    from app import config
    from app.backends import get_backend
    backend = get_backend("smolvlm", model_id=config.SMOLVLM_MODEL,
                          device="cpu", gen_params=config.GEN_PARAMS)
    print("[info] loading SmolVLM-500M ..."); backend.load(); backend.warmup()

    from PIL import Image
    from app import postprocess
    out = []
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        fn = os.path.basename(r["filename"])
        p = IMGDIR / fn
        if not p.exists():
            print(f"  skip (no image): {fn}"); continue
        im = Image.open(p).convert("RGB")
        gen_app = postprocess.clean(backend.describe(
            im, config.get_prompt("surrounding"),
            max_new_tokens=config.get_max_tokens("surrounding")))
        gen_cap = postprocess.clean(backend.describe(im, CAPTION_PROMPT, max_new_tokens=45))
        out.append({"filename": fn, "scenario": r.get("scenario", ""),
                    "gen_app": gen_app, "gen_caption": gen_cap})
        print(f"  [{i+1}/{len(rows)}] {fn}: {gen_cap[:60]}")
    dt = time.perf_counter() - t0

    with open(OUT_GEN, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "scenario", "gen_app", "gen_caption"])
        w.writeheader(); w.writerows(out)
    print(f"[done] {len(out)} images in {dt/60:.1f} min -> {OUT_GEN}")


if __name__ == "__main__":
    main()
