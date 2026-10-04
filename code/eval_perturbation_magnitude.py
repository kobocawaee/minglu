"""
eval_perturbation_magnitude.py — 「輸入到底變了多少」（指導教授 2026-07-22 的要求）
=========================================================================================
§4.3 指出輕微的擾動（旋轉 ±4°、亮度 ±15%、裁切 92%）會讓安全判斷翻轉 40%
但審稿人可能質疑「±4° 真的算輕微嗎 — 是你自己選的」。這支腳本用數字回答：
用 3 個層次的標準指標，量測影像*實際改變了多少*

  SSIM      — 像素結構層次（1.0 = 完全相同）[Wang et al. 2004]
  LPIPS     — 深度感知層次（0.0 = 完全相同）[Zhang et al. 2018]
  CLIP-cos  — 語意層次：CLIP 影像 embedding 的 cosine（1.0 = 完全相同）

關於教授建議的指標：LPIPS/SSIM/DINO 衡量的是*影像*而不是*描述*，所以用來回答
「輸入改變了多少」（不是「描述穩不穩定」，那個已經用判斷翻轉率／Jaccard 衡量）
CLIP-cos 取代 DINO 的角色（語意特徵對齊），使用 CLIPScore 已經載入的模型

執行（vlm_research 環境＋有 lpips/scipy 的 pylibs）：
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


# ---------------------------------------------------------------- SSIM（自己實作）
def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """在灰階浮點數 [0,1] 上計算整體 SSIM — Wang et al. 2004 的標準公式
    用 cv2 做 11x11、sigma 1.5 的高斯視窗（不依賴 skimage）"""
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
    # 使用和 eval_robustness.py 完全相同的 make_variants（不複製程式碼，避免兩邊不一致）
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
        # transformers 5.x：get_image_features 回傳的是輸出物件而不是 tensor
        # → 自己走過 vision_model + visual_projection（得到相同的 image_embeds）
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
