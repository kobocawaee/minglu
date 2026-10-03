"""
bench_gemma_npu.py — รัน Gemma-3-4b บน AMD NPU (Ryzen AI / OGA) วนทั้ง 17 รูป
=============================================================================
จุดประสงค์: เติม Table 4 (safety) + คุณภาพ ให้ Gemma มีตัวเลขบนชุด 17 รูปเดียวกัน
            กับ SmolVLM-256M/500M (เดิม Gemma รันแค่ 1 รูป crosswalk_car)

⚠️ ต้องรันใน env NPU เท่านั้น:
    conda activate ryzen-ai-1.7.1
    python code/bench_gemma_npu.py

หมายเหตุสำคัญ:
- Gemma build นี้เป็น NPU-only (provider_options=[{"RyzenAI":{}}]) → รันบน CPU/iGPU ไม่ได้
- ใช้ prompt เดียวกับ SmolVLM (concise + anti-guess) เพื่อให้เทียบ safety แฟร์
  (รอบแรกเคยใช้ "Describe this image in detail" → latency จะต่างนิดหน่อย ไม่ต้องตกใจ)
- คาดเวลา ~50 วิ/รูป (TTFT ~18s + decode) × 17 ≈ 15–20 นาที
- ผลออกที่ results/gemma_npu_17.csv → เอาไปให้คะแนน safety ด้วยเกณฑ์ 2-metric เดียวกัน

API note: onnxruntime_genai เปลี่ยน API ข้ามเวอร์ชัน. ถ้าบรรทัดที่ติดป้าย
[API] error ให้ลองสลับเป็นแบบที่คอมเมนต์ไว้ข้างๆ
"""

import sys
import os
import csv
import time
import argparse

# Windows console = cp1252 → กันพิมพ์ไทย/emoji แล้ว error (ชื่อไฟล์มีภาษาไทย)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import psutil

try:
    import onnxruntime_genai as og
except ImportError:
    print("[ERROR] import onnxruntime_genai ไม่ได้ — รันใน env ryzen-ai-1.7.1 หรือยัง?")
    print("        conda activate ryzen-ai-1.7.1")
    sys.exit(1)

# prompt เดียวกับ SmolVLM benchmark (ดู data/benchmark.csv) เพื่อเทียบ safety แฟร์
SAFETY_PROMPT = (
    "I am visually impaired. In two short sentences, describe only what you "
    "clearly see and warn me about nearby people or obstacles. Do not guess."
)


def build_prompt(user_text: str) -> str:
    """สร้าง prompt ตาม chat template ของ Gemma-3 (จาก chat_template.jinja).
    รูปแบบ: <bos><start_of_turn>user\n<start_of_image>{text}<end_of_turn>\n<start_of_turn>model\n
    """
    return (
        "<start_of_turn>user\n"
        "<start_of_image>"
        f"{user_text}"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def load_manifest(manifest_path):
    """อ่าน filename, scenario, ground_truth จาก dataset_manifest.csv"""
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "filename": r["filename"].strip(),
                    "scenario": r.get("scenario", "").strip(),
                    "ground_truth": r.get("ground_truth", "").strip(),
                }
            )
    return rows


def run_one(model, processor, tokenizer_stream, image_path, user_text, max_new_tokens):
    """รัน Gemma 1 รูป → คืน dict {ttft_s, decode_tps, out_tokens, peak_ram_mb, output}"""
    proc = psutil.Process(os.getpid())

    images = og.Images.open(image_path)
    prompt = build_prompt(user_text)
    inputs = processor(prompt, images=images)

    params = og.GeneratorParams(model)
    # do_sample=False = greedy (เทียบ deterministic เหมือน SmolVLM best config)
    params.set_search_options(do_sample=False, max_length=16384)

    generator = og.Generator(model, params)

    out_text = []
    out_tokens = 0
    peak_rss = proc.memory_info().rss

    # --- TTFT = prefill (image encode บน NPU) + token แรก ---
    # ⚠️ ใน OGA 0.11 การ prefill เกิดตอน set_inputs() ไม่ใช่ generate_next_token()
    #    → ต้องจับเวลาครอบทั้ง 2 ขั้น ไม่งั้นได้ TTFT=0
    t0 = time.perf_counter()
    # [API] เวอร์ชันใหม่: generator.set_inputs(inputs)
    #       เวอร์ชันเก่าบางตัว: params.set_inputs(inputs) ก่อนสร้าง generator
    generator.set_inputs(inputs)
    generator.generate_next_token()
    ttft_s = time.perf_counter() - t0

    tok = generator.get_next_tokens()[0]
    out_text.append(tokenizer_stream.decode(tok))
    out_tokens += 1
    peak_rss = max(peak_rss, proc.memory_info().rss)

    # --- decode tokens ที่เหลือ ---
    t_decode0 = time.perf_counter()
    while not generator.is_done() and out_tokens < max_new_tokens:
        generator.generate_next_token()
        tok = generator.get_next_tokens()[0]
        out_text.append(tokenizer_stream.decode(tok))
        out_tokens += 1
        peak_rss = max(peak_rss, proc.memory_info().rss)
    decode_elapsed = time.perf_counter() - t_decode0

    decoded_after_first = out_tokens - 1
    decode_tps = (decoded_after_first / decode_elapsed) if decode_elapsed > 0 else 0.0

    return {
        "ttft_s": round(ttft_s, 3),
        "decode_tps": round(decode_tps, 3),
        "out_tokens": out_tokens,
        "peak_ram_mb": round(peak_rss / (1024 * 1024), 1),
        # clean: ▁ = SentencePiece space marker หลุดมาบางที, newline → space
        "output": " ".join("".join(out_text).replace("▁", " ").split()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/gemma3_4b_npu")
    ap.add_argument("--image-dir", default="data/test_images")
    ap.add_argument("--manifest", default="data/dataset_manifest.csv")
    ap.add_argument("--out", default="results/gemma_npu_17.csv")
    ap.add_argument("--prompt", default=SAFETY_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=50)
    ap.add_argument(
        "--warmup",
        action="store_true",
        help="รัน warmup 1 รูปก่อน (TTFT รูปแรกจะรวมเวลา compile/load — แนะนำเปิด)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="รันแค่ N รูปแรก (0=ทั้งหมด). ใช้ smoke test เช่น --limit 1",
    )
    args = ap.parse_args()

    print(f"[info] onnxruntime_genai version: {getattr(og, '__version__', '?')}")
    # ⚠️ ต้องเป็น absolute path! ถ้าส่ง relative OGA จะเอา model-dir ไปต่อซ้ำ 2 ครั้ง
    #    (Cannot read header from models/gemma3_4b_npu/models/gemma3_4b_npu/...pb.bin)
    model_dir_abs = os.path.abspath(args.model_dir)
    print(f"[info] โหลด model จาก {model_dir_abs} (NPU/RyzenAI) ...")
    t_load = time.perf_counter()
    model = og.Model(model_dir_abs)
    processor = model.create_multimodal_processor()
    tokenizer_stream = processor.create_stream()
    print(f"[info] โหลดเสร็จใน {time.perf_counter() - t_load:.1f}s")

    manifest = load_manifest(args.manifest)
    if args.limit > 0:
        manifest = manifest[: args.limit]
        print(f"[info] --limit {args.limit} → รันแค่ {len(manifest)} รูปแรก (smoke test)")
    print(f"[info] รูปใน manifest: {len(manifest)} รูป | prompt: {args.prompt!r}")

    # warmup (TTFT call แรกรวม compile → ทิ้งผลทิ้ง ไม่บันทึก)
    if args.warmup and manifest:
        wpath = os.path.join(args.image_dir, manifest[0]["filename"])
        if os.path.exists(wpath):
            print("[info] warmup run (ไม่บันทึก) ...")
            try:
                run_one(model, processor, tokenizer_stream, wpath, args.prompt, args.max_new_tokens)
            except Exception as e:
                print(f"[warn] warmup error (ข้าม): {e}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = [
        "filename", "scenario", "ground_truth", "model",
        "ttft_s", "decode_tps", "out_tokens", "peak_ram_mb", "prompt", "output",
    ]
    results = []
    t_all = time.perf_counter()

    for i, row in enumerate(manifest, 1):
        img_path = os.path.join(args.image_dir, row["filename"])
        if not os.path.exists(img_path):
            print(f"[{i}/{len(manifest)}] ⚠️ ไม่พบไฟล์ {img_path} — ข้าม")
            continue
        print(f"[{i}/{len(manifest)}] {row['filename']} ...", flush=True)
        try:
            r = run_one(model, processor, tokenizer_stream, img_path, args.prompt, args.max_new_tokens)
        except Exception as e:
            print(f"    ❌ error: {e}")
            continue
        rec = {
            "filename": row["filename"],
            "scenario": row["scenario"],
            "ground_truth": row["ground_truth"],
            "model": "gemma-3-4b-npu",
            "prompt": args.prompt,
            **r,
        }
        results.append(rec)
        print(f"    TTFT {r['ttft_s']}s | decode {r['decode_tps']} tok/s | "
              f"RAM {r['peak_ram_mb']}MB | out: {r['output'][:80]}...")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in results:
            writer.writerow(rec)

    elapsed = time.perf_counter() - t_all
    if results:
        avg_ttft = sum(r["ttft_s"] for r in results) / len(results)
        avg_ram = sum(r["peak_ram_mb"] for r in results) / len(results)
        print(f"\n✅ เสร็จ {len(results)} รูป ใน {elapsed/60:.1f} นาที")
        print(f"   AVG TTFT {avg_ttft:.1f}s | AVG peak RAM {avg_ram:.0f}MB")
        print(f"   → {args.out}")
        print("\n[next] เอา column output ไปให้คะแนน safety ด้วยเกณฑ์ 2-metric "
              "(Metric A directional / Metric B fabrication) แล้วเติม Table 4 แถว Gemma")
    else:
        print("\n⚠️ ไม่มีผลเลย — เช็ค path รูป/manifest")


if __name__ == "__main__":
    main()
