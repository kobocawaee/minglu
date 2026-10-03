"""
Benchmark SmolVLM บน AMD Radeon iGPU (DirectML) เทียบ CPU — platform ที่ 2
- รันใน env `vlm_dml` เท่านั้น! (torch 2.4.1 + torch-directml + transformers 4.49)
  เรียก python ของ env นั้นตรง ๆ: <conda>/envs/vlm_dml/python.exe
- โหลด model ครั้งเดียว → รันทุกรูปบน CPU ก่อน แล้วย้าย model ไป iGPU รันทุกรูป (ย้าย weight 2 ครั้ง)
- append ผลลง data/benchmark.csv (schema เดียวกับ test_smolvlm.py) เพื่อทำกราฟ CPU vs iGPU ร่วมกันได้
  device label: CPU = "AMD-Ryzen-CPU-t2.4" (env เดียวกัน เทียบแฟร์), iGPU = "AMD-Radeon-iGPU"

วิธีรัน:
    python code/test_directml.py --image-dir data/test_images
    python code/test_directml.py --image-dir data/test_images --model HuggingFaceTB/SmolVLM-500M-Instruct
    python code/test_directml.py --skip-cpu          # รันเฉพาะ iGPU
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch_directml
import psutil
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LONGEST_EDGE = 384
MAX_NEW = 50
REP_PENALTY = 1.2
NO_REPEAT = 3
PROMPT = (
    "I am visually impaired. In two short sentences, describe only what you clearly "
    "see and warn me about nearby people or obstacles. Do not guess."
)
CPU_LABEL = "AMD-Ryzen-CPU-t2.4"
IGPU_LABEL = "AMD-Radeon-iGPU"

CSV_FIELDS = [
    "timestamp", "device", "model", "image", "longest_edge", "split", "vision_tokens",
    "rep_penalty", "no_repeat_ngram", "prefill_s", "latency_s", "out_tokens",
    "peak_ram_mb", "prompt", "output",
]


def collect_images(image_arg, image_dir_arg):
    """คืน list ของ (PIL image, ชื่อไฟล์)"""
    if image_dir_arg:
        paths = sorted(p for p in Path(image_dir_arg).iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not paths:
            raise SystemExit(f"ไม่พบรูปใน {image_dir_arg}")
        return [(Image.open(p).convert("RGB"), p.name) for p in paths]
    p = Path(image_arg)
    return [(Image.open(p).convert("RGB"), p.name)]


def build_inputs(processor, image, target_device):
    """เตรียม input + ตั้ง longest_edge=384 (sweet spot) ให้ตรงกับ benchmark CPU"""
    processor.image_processor.do_image_splitting = True
    processor.image_processor.size = {"longest_edge": LONGEST_EDGE}
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(text=prompt_text, images=[image], return_tensors="pt").to(target_device)


def run_once(model, processor, image, label, target_device):
    """generate 1 รูป บน device ที่ model อยู่แล้ว — คืน dict metrics"""
    inputs = build_inputs(processor, image, target_device)
    n_prompt = inputs["input_ids"].shape[1]

    t_pf = time.perf_counter()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1)
    prefill = time.perf_counter() - t_pf

    t1 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW, repetition_penalty=REP_PENALTY,
            no_repeat_ngram_size=NO_REPEAT,
        )
    latency = time.perf_counter() - t1

    text = processor.batch_decode(out[:, n_prompt:], skip_special_tokens=True)[0].strip()
    return {
        "image": label,
        "vision_tokens": n_prompt,
        "prefill": prefill,
        "latency": latency,
        "out_tokens": out.shape[1] - n_prompt,
        "output": text,
        "peak_ram_mb": psutil.Process().memory_info().rss / 1024**2,
    }


def run_all(model, processor, images, label, target_device):
    """ย้าย model ไป device ครั้งเดียว warm-up แล้ววนทุกรูป"""
    print(f"\n=== {label} ===")
    model.to(target_device)
    # warm-up: ดูดเวลา compile ครั้งแรกของ DirectML ออกไป (ใช้รูปแรก)
    with torch.no_grad():
        model.generate(**build_inputs(processor, images[0][0], target_device), max_new_tokens=1)
    results = []
    for image, name in images:
        r = run_once(model, processor, image, name, target_device)
        results.append(r)
        print(f"[{name[:22]:>22}] prefill={r['prefill']:.2f}s latency={r['latency']:.2f}s "
              f"out={r['out_tokens']}tok")
        print(f"    -> {r['output']}")
    return results


def append_csv(csv_path, device, model_id, results):
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "device": device,
                "model": model_id,
                "image": r["image"],
                "longest_edge": LONGEST_EDGE,
                "split": "ON",
                "vision_tokens": r["vision_tokens"],
                "rep_penalty": REP_PENALTY,
                "no_repeat_ngram": NO_REPEAT,
                "prefill_s": round(r["prefill"], 2),
                "latency_s": round(r["latency"], 2),
                "out_tokens": r["out_tokens"],
                "peak_ram_mb": round(r["peak_ram_mb"]),
                "prompt": PROMPT,
                "output": r["output"],
            })


def main():
    global LONGEST_EDGE, MAX_NEW, PROMPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(DATA_DIR / "sample_indoor.jpg"))
    ap.add_argument("--image-dir", default=None, help="โฟลเดอร์รูป (วนทุกรูป)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model id")
    ap.add_argument("--skip-cpu", action="store_true", help="รันเฉพาะ iGPU")
    ap.add_argument("--longest-edge", type=int, default=LONGEST_EDGE)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--csv", default=None, help="path CSV (default data/benchmark.csv)")
    args = ap.parse_args()

    # override ค่าคงที่จาก CLI (สำหรับทดลอง tuning เช่น le=256 + max_new สั้น)
    LONGEST_EDGE = args.longest_edge
    MAX_NEW = args.max_new_tokens
    PROMPT = args.prompt
    print(f"[config] longest_edge={LONGEST_EDGE} max_new={MAX_NEW}")

    dml = torch_directml.device()
    print(f"DirectML device: {torch_directml.device_name(0)}")

    images = collect_images(args.image, args.image_dir)
    print(f"รูปทั้งหมด {len(images)} รูป | model: {args.model}")

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    print(f"โหลด model เสร็จใน {time.perf_counter() - t0:.1f}s")

    csv_path = Path(args.csv) if args.csv else DATA_DIR / "benchmark.csv"
    if not args.skip_cpu:
        cpu_res = run_all(model, processor, images, CPU_LABEL, "cpu")
        append_csv(csv_path, CPU_LABEL, args.model, cpu_res)
    dml_res = run_all(model, processor, images, IGPU_LABEL, dml)
    append_csv(csv_path, IGPU_LABEL, args.model, dml_res)

    # สรุปเฉลี่ย
    print("\n" + "=" * 60)
    if not args.skip_cpu:
        cpu_lat = sum(r["latency"] for r in cpu_res) / len(cpu_res)
        cpu_pf = sum(r["prefill"] for r in cpu_res) / len(cpu_res)
        print(f"{CPU_LABEL:>20}: avg prefill {cpu_pf:.2f}s | avg latency {cpu_lat:.2f}s")
    dml_lat = sum(r["latency"] for r in dml_res) / len(dml_res)
    dml_pf = sum(r["prefill"] for r in dml_res) / len(dml_res)
    print(f"{IGPU_LABEL:>20}: avg prefill {dml_pf:.2f}s | avg latency {dml_lat:.2f}s")
    if not args.skip_cpu:
        print(f"iGPU vs CPU latency speedup: {cpu_lat / dml_lat:.2f}x")
    print("=" * 60)
    print(f"append ผลลง {csv_path}")


if __name__ == "__main__":
    main()
