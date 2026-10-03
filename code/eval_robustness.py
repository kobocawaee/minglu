"""
eval_robustness.py — วัดความเสถียรของคำบรรยายต่อ input perturbation เล็กๆ
=========================================================================
คำถามวิจัย: โมเดลจิ๋วให้คำตอบ "นิ่ง" แค่ไหน เมื่อ input เปลี่ยนนิดเดียวแบบมือถือสั่นจริง
(หมุน ±4°, สว่าง ±15%, ครอป 92%) — *ทั้งที่ภาพยังคุณภาพดี (ผ่าน frame-quality gate)*?

decoder เป็น greedy (deterministic) → ถ้าคำตอบเปลี่ยนเมื่อ input ขยับนิดเดียว = ความเปราะ
ของ *ตัวโมเดล* ต่อ input ไม่ใช่การสุ่ม และไม่ใช่เฟรมแย่ (perturbation ทุกตัวผ่าน gate).

เมตริก:
  1) Safety-verdict flip rate (รูป crosswalk, street mode) — verdict {safe/not-safe/unsure} พลิกไหม
  2) Person-mention flip rate (ทั้งชุด, surrounding mode) — "มีคน/ไม่มีคน" พลิกไหม
  3) Content consistency (word-Jaccard เฉลี่ยทุกคู่ของ variant) — คำอธิบายเหมือนเดิมแค่ไหน

รัน (env vlm_research หรือ vlm_dml):
    python code/eval_robustness.py --device cpu --out results/robustness
"""

import sys, os, re, csv, argparse, itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root → import app
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from PIL import Image, ImageEnhance
from app import config, quality
from app.backends import get_backend


# ---------------------------------------------------------------------------
# Perturbations — เลียนแบบมือถือสั่น/แสงเปลี่ยน. deterministic (reproducible).
# ตั้งใจให้ "เบา" พอที่ความหมายไม่เปลี่ยน + ผ่าน frame-quality gate
# ---------------------------------------------------------------------------
def make_variants(img: Image.Image) -> dict:
    w, h = img.size
    c = int(w * 0.04), int(h * 0.04)   # crop 92% กลางภาพ
    return {
        "orig":     img,
        "rot+4":    img.rotate(4, resample=Image.BICUBIC, expand=False),
        "rot-4":    img.rotate(-4, resample=Image.BICUBIC, expand=False),
        "bright+":  ImageEnhance.Brightness(img).enhance(1.15),
        "bright-":  ImageEnhance.Brightness(img).enhance(0.85),
        "crop92":   img.crop((c[0], c[1], w - c[0], h - c[1])).resize((w, h)),
    }


# ---------------------------------------------------------------------------
# ตัวจำแนกคำตอบ (keyword-based, โปร่งใส) — reuse แนวจาก eval_ptl.py
# ---------------------------------------------------------------------------
_SAFE = re.compile(r"\b(safe to cross|you (can|may) cross|looks clear|way is clear|"
                   r"no vehicles|no cars|clear to cross|safe to walk)\b", re.I)
_NOTSAFE = re.compile(r"\b(not safe|do not cross|don'?t cross|wait|unsafe|"
                      r"vehicles? (are|is) (moving|coming|approaching)|car is (moving|coming)|"
                      r"stop|not clear)\b", re.I)
_PERSON = re.compile(r"\b(person|people|man|woman|men|women|pedestrian|someone|"
                     r"child|children|kid|boy|girl|human|figure)\b", re.I)


def safety_stance(text: str) -> str:
    safe, notsafe = bool(_SAFE.search(text)), bool(_NOTSAFE.search(text))
    if safe and not notsafe:
        return "safe"
    if notsafe and not safe:
        return "not-safe"
    return "unsure"          # ทั้งคู่/ไม่มีเลย = กำกวม


def has_person(text: str) -> bool:
    return bool(_PERSON.search(text))


_STOP = set("a an the is are was were be been of to in on at and or with this that it "
            "i you he she they we my your me front looks look see seeing there here as "
            "so if not no yes am".split())


def content_tokens(text: str) -> set:
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in _STOP and len(w) > 2}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def mean_pairwise_jaccard(texts: list) -> float:
    toks = [content_tokens(t) for t in texts]
    pairs = list(itertools.combinations(toks, 2))
    if not pairs:
        return 1.0
    return sum(jaccard(x, y) for x, y in pairs) / len(pairs)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", default="data/test_images")
    ap.add_argument("--device", default="cpu", choices=["cpu", "igpu"])
    ap.add_argument("--out", default="results/robustness")
    args = ap.parse_args()

    imgs = [p for p in sorted(Path(args.image_dir).glob("*"))
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    print(f"[info] {len(imgs)} images, device={args.device}")

    backend = get_backend("smolvlm", model_id=config.SMOLVLM_MODEL,
                          device=args.device, gen_params=config.GEN_PARAMS)
    print("[info] loading SmolVLM-500M ...")
    backend.load(); backend.warmup()

    os.makedirs(args.out, exist_ok=True)
    raw_rows = []                    # ทุก output (image,variant,mode,text)
    per_img = []                     # สรุปต่อรูป

    for p in imgs:
        is_street = p.name.lower().startswith("crosswalk")
        mode = "street" if is_street else "surrounding"
        prompt = config.get_prompt(mode)
        mx = config.get_max_tokens(mode)

        base = Image.open(p).convert("RGB")
        variants = make_variants(base)

        # ยืนยันว่า variant ทุกตัวผ่าน quality gate (แยก finding นี้ออกจากเรื่องเฟรมแย่)
        gate_pass = sum(quality.assess(im, config.QUALITY)[0] for im in variants.values())

        texts, stances, persons = [], [], []
        for vname, vimg in variants.items():
            txt = backend.describe(vimg, prompt, max_new_tokens=mx)
            texts.append(txt)
            stances.append(safety_stance(txt))
            persons.append(has_person(txt))
            raw_rows.append({"image": p.name, "variant": vname, "mode": mode, "text": txt})

        consistency = mean_pairwise_jaccard(texts)
        stance_flip = len(set(stances)) > 1
        person_flip = len(set(persons)) > 1
        per_img.append({
            "image": p.name, "mode": mode, "gate_pass": f"{gate_pass}/{len(variants)}",
            "consistency": round(consistency, 3),
            "n_distinct_stance": len(set(stances)), "stance_flip": stance_flip,
            "stances": "|".join(stances),
            "n_distinct_person": len(set(int(x) for x in persons)), "person_flip": person_flip,
            "persons": "".join("P" if x else "-" for x in persons),
        })
        flag = "  <-- FLIP" if (stance_flip or person_flip) else ""
        print(f"  {p.name:42s} [{mode:11s}] consist={consistency:.2f} "
              f"stance={'|'.join(stances):28s} person={''.join('P' if x else '-' for x in persons)}{flag}")

    # ---- เขียนไฟล์ ----
    raw_csv = os.path.join(args.out, "robustness_raw.csv")
    with open(raw_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image", "variant", "mode", "text"])
        w.writeheader(); w.writerows(raw_rows)

    sum_csv = os.path.join(args.out, "robustness_per_image.csv")
    with open(sum_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(per_img[0].keys()))
        w.writeheader(); w.writerows(per_img)

    # ---- aggregate ----
    street = [r for r in per_img if r["mode"] == "street"]
    allr = per_img
    def rate(rows, key): return 100.0 * sum(r[key] for r in rows) / len(rows) if rows else 0.0
    mean_consist = sum(r["consistency"] for r in allr) / len(allr)
    total_gate = sum(int(r["gate_pass"].split("/")[0]) for r in allr)
    total_var = sum(int(r["gate_pass"].split("/")[1]) for r in allr)

    print("\n" + "=" * 64)
    print(f"variants/image = 6 (orig + rot±4, bright±15%, crop92)")
    print(f"quality-gate pass: {total_gate}/{total_var} variants  "
          f"(perturbation ไม่ได้ทำให้เฟรมแย่ → วัดความเปราะของโมเดลล้วน)")
    print(f"mean content consistency (word-Jaccard, all {len(allr)}): {mean_consist:.3f}")
    print(f"safety-verdict flip rate (street n={len(street)}): {rate(street,'stance_flip'):.0f}%")
    print(f"person-mention flip rate (all n={len(allr)}):      {rate(allr,'person_flip'):.0f}%")
    print("=" * 64)
    print(f"[saved] {raw_csv}\n[saved] {sum_csv}")


if __name__ == "__main__":
    main()
