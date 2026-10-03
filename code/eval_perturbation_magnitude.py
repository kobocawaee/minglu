"""
eval_perturbation_magnitude.py — "input เปลี่ยนน้อยแค่ไหน" (advisor request 2026-07-22)
=========================================================================================
§4.3 บอกว่า perturbation เบาๆ (หมุน ±4°, สว่าง ±15%, crop 92%) ทำ safety verdict พลิก 40%
แต่ reviewer แย้งได้ว่า "±4° เบาจริงหรือ — คุณเลือกเอง". สคริปต์นี้ตอบด้วยตัวเลข:
วัดว่าภาพ *เปลี่ยนไปเท่าไหร่จริง* ด้วย metric มาตรฐาน 3 ระดับ

  SSIM      — ระดับโครงสร้างพิกเซล (1.0 = เหมือนกันเป๊ะ) [Wang et al. 2004]
  LPIPS     — ระดับการรับรู้เชิงลึก (0.0 = เหมือนกันเป๊ะ) [Zhang et al. 2018]
  CLIP-cos  — ระดับความหมาย: cosine ของ CLIP image embedding (1.0 = เหมือนกันเป๊ะ)

หมายเหตุ metric ที่อาจารย์เสนอ: LPIPS/SSIM/DINO วัด *ภาพ* ไม่ใช่ *คำบรรยาย* จึงใช้ตอบคำถาม
"input เปลี่ยนน้อยแค่ไหน" (ไม่ใช่ "คำบรรยายเสถียรไหม" ซึ่งวัดด้วย verdict-flip/Jaccard อยู่แล้ว)
CLIP-cos ใช้แทนบทบาทของ DINO (semantic feature alignment) โดยใช้โมเดลที่โหลดไว้แล้วจาก CLIPScore

รัน (env vlm_research + pylibs ที่มี lpips/scipy):
    python code/eval_perturbation_magnitude.py
output: results/perturbation_magnitude.csv
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

IMG_DIR = ROOT / "data/test_images"
PER_IMAGE = ROOT / "results/robustness/robustness_per_image.csv"
OUT = ROOT / "results/perturbation_magnitude.csv"
CLIP_ID = "openai/clip-vit-base-patch32"


# ---------------------------------------------------------------- SSIM (ของเราเอง)
def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM แบบ global บน grayscale float [0,1] — สูตรมาตรฐาน Wang et al. 2004
    ใช้ Gaussian window 11x11 sigma 1.5 ผ่าน cv2 (ไม่ต้องพึ่ง skimage)"""
    import cv2
    C1, C2 = (0.01 ** 2), (0.03 ** 2)
    a = a.astype(np.float64); b = b.astype(np.float64)
    k = (11, 11); s = 1.5
    mu_a = cv2.GaussianBlur(a, k, s); mu_b = cv2.GaussianBlur(b, k, s)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = cv2.GaussianBlur(a * a, k, s) - mu_a2
    sb = cv2.GaussianBlur(b * b, k, s) - mu_b2
    sab = cv2.GaussianBlur(a * b, k, s) - mu_ab
    m = ((2 * mu_ab + C1) * (2 * sab + C2)) / ((mu_a2 + mu_b2 + C1) * (sa + sb + C2))
    return float(m.mean())


def main():
    import cv2
    import torch
    from PIL import Image
    import lpips as lpips_mod
    from transformers import CLIPModel, CLIPProcessor
    # ใช้ make_variants ตัวเดียวกับ eval_robustness.py เป๊ะ (ไม่ก๊อปโค้ดซ้ำ กัน drift)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "evrob", str(Path(__file__).resolve().parent / "eval_robustness.py"))
    evrob = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evrob)
    make_variants = evrob.make_variants

    torch.set_num_threads(8)
    print("[info] loading LPIPS (AlexNet) and CLIP ...")
    loss_fn = lpips_mod.LPIPS(net="alex", verbose=False)
    clip = CLIPModel.from_pretrained(CLIP_ID).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID)

    def clip_embed(pil):
        # transformers 5.x: get_image_features คืน output object ไม่ใช่ tensor
        # → เดินผ่าน vision_model + visual_projection เอง (ได้ image_embeds ตัวเดียวกัน)
        with torch.no_grad():
            vis = clip.vision_model(**proc(images=pil, return_tensors="pt"))
            e = clip.visual_projection(vis.pooler_output)
        return (e / e.norm(dim=-1, keepdim=True))[0]

    def to_lpips(pil):
        x = np.asarray(pil.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

    flips = {r["image"]: r for r in
             csv.DictReader(open(PER_IMAGE, encoding="utf-8-sig"))}

    rows = []
    files = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in
                   {".jpg", ".jpeg", ".png", ".webp"})
    for n, path in enumerate(files, 1):
        base = Image.open(path).convert("RGB")
        variants = make_variants(base)
        orig = variants["orig"]
        g0 = cv2.cvtColor(np.asarray(orig), cv2.COLOR_RGB2GRAY) / 255.0
        e0 = clip_embed(orig)
        t0 = to_lpips(orig)
        meta = flips.get(path.name, {})
        for vname, vimg in variants.items():
            if vname == "orig":
                continue
            if vimg.size != orig.size:
                vimg = vimg.resize(orig.size)
            g1 = cv2.cvtColor(np.asarray(vimg), cv2.COLOR_RGB2GRAY) / 255.0
            with torch.no_grad():
                lp = float(loss_fn(t0, to_lpips(vimg)))
            rows.append({
                "image": path.name, "mode": meta.get("mode", ""), "variant": vname,
                "ssim": round(ssim(g0, g1), 4),
                "lpips": round(lp, 4),
                "clip_cos": round(float((e0 * clip_embed(vimg)).sum()), 4),
                "stance_flip": meta.get("stance_flip", ""),
                "person_flip": meta.get("person_flip", ""),
                "text_consistency": meta.get("consistency", ""),
            })
        print(f"  [{n}/{len(files)}] {path.name}")

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # ------------------------------------------------------------ summary
    print("\n" + "=" * 72)
    print(f"{'variant':10s} {'SSIM':>18s} {'LPIPS':>18s} {'CLIP-cos':>18s}")
    for v in ("rot+4", "rot-4", "bright+", "bright-", "crop92"):
        sel = [r for r in rows if r["variant"] == v]
        f = lambda k: np.array([r[k] for r in sel])
        print(f"{v:10s} {f('ssim').mean():>10.3f} ±{f('ssim').std():.3f} "
              f"{f('lpips').mean():>10.3f} ±{f('lpips').std():.3f} "
              f"{f('clip_cos').mean():>10.3f} ±{f('clip_cos').std():.3f}")
    a = lambda k: np.array([r[k] for r in rows])
    print("-" * 72)
    print(f"{'ALL':10s} {a('ssim').mean():>10.3f} ±{a('ssim').std():.3f} "
          f"{a('lpips').mean():>10.3f} ±{a('lpips').std():.3f} "
          f"{a('clip_cos').mean():>10.3f} ±{a('clip_cos').std():.3f}")

    st = [r for r in rows if r["mode"] == "street"]
    print(f"\nstreet subset ({len(st)} variant pairs): "
          f"SSIM {np.mean([r['ssim'] for r in st]):.3f} · "
          f"LPIPS {np.mean([r['lpips'] for r in st]):.3f} · "
          f"CLIP-cos {np.mean([r['clip_cos'] for r in st]):.3f}")
    flipped = sorted({r["image"] for r in rows if r["stance_flip"] == "True"})
    print(f"images whose safety verdict flipped: {len(flipped)}")
    for im in flipped:
        sel = [r for r in rows if r["image"] == im]
        print(f"   {im:34s} SSIM {np.mean([r['ssim'] for r in sel]):.3f}  "
              f"LPIPS {np.mean([r['lpips'] for r in sel]):.3f}  "
              f"CLIP-cos {np.mean([r['clip_cos'] for r in sel]):.3f}")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
