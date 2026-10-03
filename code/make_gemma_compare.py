"""
make_gemma_compare.py — สร้างภาพเปรียบเทียบ 500M vs Gemma-4B บนฉากเดียวกัน
=========================================================================
ไว้โชว์ในพรีเซนต์ (แทนการรัน Gemma สด 37 วินาที). สไตล์ navy+teal เหมือนเด็ค
อ่าน output "จริง" จาก CSV ที่รันไว้ → results/gemma_compare.png

วิธีใช้:
    conda activate vlm_research
    python code/make_gemma_compare.py
    python code/make_gemma_compare.py --image "crosswalk_car.jpg"   # เปลี่ยนฉาก
"""
import argparse
import csv
import os
import sys
import textwrap
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAVY = (14, 36, 56); CARD_L = (241, 244, 248); TEALC = (16, 60, 55)
TEAL = (18, 165, 148); CYAN = (42, 166, 206); WHITE = (255, 255, 255)
GRAY = (200, 210, 220); DGRAY = (80, 96, 120); TITLE = (255, 255, 255)
FD = "C:/Windows/Fonts/"

def font(name, size):
    for f in [name, "arial.ttf"]:
        try:
            return ImageFont.truetype(FD + f, size)
        except Exception:
            continue
    return ImageFont.load_default()

def read_out(csvpath, fn, model_filter=None, col="output"):
    for r in csv.DictReader(open(csvpath, encoding="utf-8")):
        if r.get("filename") == fn and (model_filter is None or r.get("model") == model_filter):
            return r[col]
    return "(no output found)"

def wrap(draw, text, fnt, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= maxw:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines

def rrect(d, box, r, fill):
    d.rounded_rectangle(box, radius=r, fill=fill)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="images (2).jpg")
    ap.add_argument("--smol", default=None, help="override 500M text")
    ap.add_argument("--gemma", default=None, help="override Gemma text")
    ap.add_argument("--smol-time", default="~2 s  ·  iGPU")
    ap.add_argument("--gemma-time", default="~37 s  ·  NPU")
    ap.add_argument("--gt", default="Ground truth: crowded indoor space — many people seated, tables are obstacles")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "gemma_compare.png"))
    args = ap.parse_args()

    smol = args.smol or read_out(os.path.join(ROOT, "data/safety_eval.csv"), args.image, "SmolVLM-500M-Instruct")
    gemma = args.gemma or read_out(os.path.join(ROOT, "results/gemma_npu_17.csv"), args.image)

    W, H = 1280, 720
    img = Image.new("RGB", (W, H), NAVY)
    d = ImageDraw.Draw(img)
    f_title = font("georgiab.ttf", 34); f_sub = font("arialbd.ttf", 15)
    f_h = font("georgiab.ttf", 24); f_badge = font("arialbd.ttf", 15)
    f_body = font("arial.ttf", 20); f_small = font("arial.ttf", 16)

    d.text((40, 34), "The Quality Tier — Same Scene, Two Models", font=f_title, fill=WHITE)
    d.text((42, 82), "SMALL & FAST  vs  LARGE & ACCURATE (BUT SLOW)", font=f_sub, fill=TEAL)

    # left: scene photo
    ph = Image.open(os.path.join(ROOT, "data/test_images", args.image)).convert("RGB")
    pw, phh = 560, 380
    ph = ph.resize((pw, phh))
    img.paste(ph, (40, 130))
    d.rectangle([40, 130, 40 + pw, 130 + phh], outline=(40, 60, 80), width=2)
    d.text((40, 522), os.path.splitext(args.image)[0], font=f_small, fill=(120, 140, 160))

    # right: two cards
    cx, cw = 640, 600
    # card 1 — 500M (light)
    rrect(d, [cx, 130, cx + cw, 130 + 250], 16, CARD_L)
    d.text((cx + 26, 152), "SmolVLM-500M", font=f_h, fill=(19, 39, 61))
    rrect(d, [cx + cw - 170, 158, cx + cw - 26, 190], 14, (220, 227, 235))
    d.text((cx + cw - 158, 164), args.smol_time, font=f_badge, fill=DGRAY)
    for i, ln in enumerate(wrap(d, smol, f_body, cw - 56)[:5]):
        d.text((cx + 26, 205 + i * 30), ln, font=f_body, fill=(70, 82, 100))

    # card 2 — Gemma (dark teal highlight)
    rrect(d, [cx, 400, cx + cw, 400 + 250], 16, (16, 43, 62))
    d.rectangle([cx, 400, cx + cw, 402], fill=(16, 43, 62))
    rrect(d, [cx, 400, cx + cw, 650], 16, (15, 42, 61))
    d.text((cx + 26, 422), "Gemma-3-4b  @ NPU", font=f_h, fill=WHITE)
    rrect(d, [cx + cw - 175, 428, cx + cw - 26, 460], 14, CYAN)
    d.text((cx + cw - 163, 434), args.gemma_time, font=f_badge, fill=WHITE)
    for i, ln in enumerate(wrap(d, gemma, f_body, cw - 56)[:5]):
        d.text((cx + 26, 475 + i * 30), ln, font=f_body, fill=(215, 226, 236))

    # bottom strip: ground truth + takeaway
    d.text((40, 640), args.gt, font=f_small, fill=(150, 165, 180))
    d.text((40, 672), "Takeaway:  the 4B model is more accurate and detailed — but ~15-20x slower. No free lunch.",
           font=font("arialbd.ttf", 17), fill=CYAN)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    img.save(args.out)
    print("saved ->", args.out)
    print("500M :", smol[:80])
    print("Gemma:", gemma[:80])

if __name__ == "__main__":
    main()
