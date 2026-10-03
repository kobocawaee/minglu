"""
ทดสอบ SmolVLM บน local (CPU) สำหรับ VLM Offline Visual Assistant
- โหลด model จาก HuggingFace (ครั้งแรกต้องต่อเน็ต หลังจากนั้น cache ไว้)
- รับรูปเดียว, ทั้งโฟลเดอร์ หรือใช้รูปตัวอย่าง
- บรรยายภาพ + วัด latency (แยก prefill/decode) และ RAM
- รองรับ resolution sweep เพื่อหาจุดสมดุล speed vs quality
- บันทึกผลลง data/benchmark.csv อัตโนมัติ

วิธีรัน:
    python code/test_smolvlm.py                                # รูปตัวอย่าง + prompt default
    python code/test_smolvlm.py --image-dir data/test_images   # วนทุกรูปในโฟลเดอร์
    python code/test_smolvlm.py --longest-edge 384 512 768      # sweep หลาย resolution
    python code/test_smolvlm.py --model HuggingFaceTB/SmolVLM-500M-Instruct
    python code/test_smolvlm.py --no-split                      # ปิด image splitting
"""

import argparse
import csv
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# Windows console ใช้ cp1252 พิมพ์ภาษาไทยไม่ได้ บังคับเป็น UTF-8 ก่อน
sys.stdout.reconfigure(encoding="utf-8")

import torch
import psutil
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
# รูป standard ในวงการ CV (แมว 2 ตัวบนโซฟา) URL เสถียร ใช้เป็น smoke test (indoor scene)
SAMPLE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# prompt เน้นความปลอดภัย + กระชับ + กัน hallucination (สำหรับ surrounding awareness)
SAFETY_PROMPT = (
    "I am visually impaired. In two short sentences, describe only what you clearly "
    "see and warn me about nearby people or obstacles. Do not guess."
)


def collect_images(image_arg, image_dir_arg):
    """คืน list ของ (PIL image, ชื่อไฟล์) — รองรับทั้งโฟลเดอร์, รูปเดียว, หรือ default sample"""
    if image_dir_arg:
        paths = sorted(
            p for p in Path(image_dir_arg).iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not paths:
            raise SystemExit(f"ไม่พบรูปใน {image_dir_arg}")
        return [(Image.open(p).convert("RGB"), p.name) for p in paths]
    if image_arg:
        p = Path(image_arg)
        return [(Image.open(p).convert("RGB"), p.name)]
    # default: รูปตัวอย่าง
    DATA_DIR.mkdir(exist_ok=True)
    sample = DATA_DIR / "sample_indoor.jpg"
    if not sample.exists():
        print(f"กำลังโหลดรูปตัวอย่างจาก {SAMPLE_URL} ...")
        urllib.request.urlretrieve(SAMPLE_URL, sample)
    return [(Image.open(sample).convert("RGB"), sample.name)]


def run_once(model, processor, image, label, prompt, max_new_tokens, longest_edge,
             no_split, rep_penalty, no_repeat_ngram):
    """รัน inference 1 ครั้งตาม config ที่กำหนด แล้วคืน dict ของ metrics"""
    # คุมจำนวน vision tokens ผ่าน 2 ปุ่ม: image splitting และ resolution
    processor.image_processor.do_image_splitting = not no_split
    if longest_edge is not None:
        processor.image_processor.size = {"longest_edge": longest_edge}

    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt")
    n_prompt = inputs["input_ids"].shape[1]  # input tokens (สะท้อนว่ารูปกลายเป็นกี่ token)

    # prefill = เวลาสร้าง token แรก ≈ เวลาประมวลผลรูป
    t_pf = time.perf_counter()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1)
    prefill = time.perf_counter() - t_pf

    # full generation — ใส่ params กัน loop/พูดซ้ำ
    t1 = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=rep_penalty,
            no_repeat_ngram_size=no_repeat_ngram,
        )
    latency = time.perf_counter() - t1

    n_new = generated_ids.shape[1] - n_prompt
    output = processor.batch_decode(
        generated_ids[:, n_prompt:], skip_special_tokens=True
    )[0].strip()

    return {
        "image": label,
        "longest_edge": longest_edge if longest_edge is not None else "default",
        "vision_tokens": n_prompt,
        "prefill": prefill,
        "latency": latency,
        "out_tokens": n_new,
        "output": output,
        "peak_ram_mb": psutil.Process().memory_info().rss / 1024**2,
    }


CSV_FIELDS = [
    "timestamp", "device", "model", "image", "longest_edge", "split", "vision_tokens",
    "rep_penalty", "no_repeat_ngram", "prefill_s", "latency_s", "out_tokens",
    "peak_ram_mb", "prompt", "output",
]


def append_csv(csv_path, device, model, split, prompt, rep_penalty, no_repeat_ngram, results):
    """append ผลแต่ละ run ลง CSV — เขียน header ให้ถ้าไฟล์ยังไม่มี"""
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "device": device,
                "model": model,
                "image": r["image"],  # ชื่อไฟล์เท่านั้น (portable บน GitHub)
                "longest_edge": r["longest_edge"],
                "split": "OFF" if split else "ON",
                "vision_tokens": r["vision_tokens"],
                "rep_penalty": rep_penalty,
                "no_repeat_ngram": no_repeat_ngram,
                "prefill_s": round(r["prefill"], 2),
                "latency_s": round(r["latency"], 2),
                "out_tokens": r["out_tokens"],
                "peak_ram_mb": round(r["peak_ram_mb"]),
                "prompt": prompt,
                "output": r["output"],
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="HuggingFace model id")
    parser.add_argument(
        "--device",
        default="CPU",
        help='ป้ายระบุเครื่อง เช่น "AMD-Ryzen-CPU" (Mhiu) หรือ "Intel-CPU" (Aomam)',
    )
    parser.add_argument("--image", default=None, help="path ของรูปเดียว")
    parser.add_argument("--image-dir", default=None, help="โฟลเดอร์รูป (วนทุกรูป)")
    parser.add_argument("--prompt", default=SAFETY_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--repetition-penalty", type=float, default=1.2, help="ลงโทษ token ซ้ำ (กัน loop)")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3, help="ห้ามวลี n คำซ้ำ (กัน loop)")
    parser.add_argument("--no-split", action="store_true", help="ปิด image splitting")
    parser.add_argument(
        "--longest-edge",
        type=int,
        nargs="+",
        default=[None],
        help="resolution ที่จะ sweep เช่น --longest-edge 384 512 768 1024",
    )
    args = parser.parse_args()

    proc = psutil.Process()
    images = collect_images(args.image, args.image_dir)
    print(f"รูปทั้งหมด {len(images)} รูป | PROMPT: {args.prompt}\n")

    # ---------- โหลด model ครั้งเดียว ----------
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    print(f"MODEL: {args.model}")
    print(f"โหลด model เสร็จใน {time.perf_counter() - t0:.1f}s\n")

    # ---------- วนทุกรูป × ทุก resolution ----------
    results = []
    for image, label in images:
        for le in args.longest_edge:
            r = run_once(
                model, processor, image, label, args.prompt,
                args.max_new_tokens, le, args.no_split,
                args.repetition_penalty, args.no_repeat_ngram_size,
            )
            results.append(r)
            print(f"[{label} | le={r['longest_edge']}] {r['output']}")

    # ---------- สรุปเป็นตาราง ----------
    print("\n" + "=" * 90)
    print(f"{'image':>20} | {'le':>6} | {'vis_tok':>7} | {'prefill':>8} | {'latency':>8} | {'out_tok':>7}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['image'][:20]:>20} | {str(r['longest_edge']):>6} | {r['vision_tokens']:>7} | "
            f"{r['prefill']:>7.2f}s | {r['latency']:>7.2f}s | {r['out_tokens']:>7}"
        )
    print("=" * 90)
    print(f"Peak RAM: {proc.memory_info().rss / 1024**2:.0f} MB | split={'OFF' if args.no_split else 'ON'}")

    # ---------- บันทึกผลลง CSV ----------
    csv_path = DATA_DIR / "benchmark.csv"
    append_csv(
        csv_path, args.device, args.model, args.no_split, args.prompt,
        args.repetition_penalty, args.no_repeat_ngram_size, results,
    )
    print(f"บันทึกผล {len(results)} แถวลง {csv_path}")


if __name__ == "__main__":
    main()
