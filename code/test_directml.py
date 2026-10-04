"""
在 AMD Radeon iGPU（DirectML）上效能測試 SmolVLM，和 CPU 比較 — 第 2 個平台
- 只能在 `vlm_dml` 環境執行！（torch 2.4.1 + torch-directml + transformers 4.49）
  直接呼叫該環境的 python：<conda>/envs/vlm_dml/python.exe
- 模型只載入一次 → 先在 CPU 跑完所有圖，再把模型移到 iGPU 跑所有圖（權重移動 2 次）
- 結果附加到 data/benchmark.csv（格式和 test_smolvlm.py 相同），方便一起畫 CPU vs iGPU 的圖
  裝置標籤：CPU = "AMD-Ryzen-CPU-t2.4"（同一環境，比較公平），iGPU = "AMD-Radeon-iGPU"

執行方式：
    python code/test_directml.py --image-dir data/test_images
    python code/test_directml.py --image-dir data/test_images --model HuggingFaceTB/SmolVLM-500M-Instruct
    python code/test_directml.py --skip-cpu          # 只跑 iGPU
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
    """回傳 (PIL 影像, 檔名) 的 list"""
    if image_dir_arg:
        paths = sorted(p for p in Path(image_dir_arg).iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not paths:
            raise SystemExit(f"在 {image_dir_arg} 找不到圖片")
        return [(Image.open(p).convert("RGB"), p.name) for p in paths]
    p = Path(image_arg)
    return [(Image.open(p).convert("RGB"), p.name)]


def build_inputs(processor, image, target_device):
    """準備輸入＋設定 longest_edge=384（最佳點），和 CPU 的效能測試一致"""
    processor.image_processor.do_image_splitting = True
    processor.image_processor.size = {"longest_edge": LONGEST_EDGE}
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(text=prompt_text, images=[image], return_tensors="pt").to(target_device)


def run_once(model, processor, image, label, target_device):
    """在模型所在的裝置上對 1 張圖產生輸出 — 回傳指標 dict"""
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
    """把模型移到裝置一次、暖機，然後逐張處理所有圖"""
    print(f"\n=== {label} ===")
    model.to(target_device)
    # 暖機：先把 DirectML 第一次編譯的時間消化掉（用第一張圖）
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
    ap.add_argument("--image-dir", default=None, help="圖片資料夾（逐張處理）")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model id")
    ap.add_argument("--skip-cpu", action="store_true", help="只跑 iGPU")
    ap.add_argument("--longest-edge", type=int, default=LONGEST_EDGE)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--csv", default=None, help="path CSV (default data/benchmark.csv)")
    args = ap.parse_args()

    # 從命令列覆寫常數（用來實驗調整，例如 le=256 + 較短的 max_new）
    LONGEST_EDGE = args.longest_edge
    MAX_NEW = args.max_new_tokens
    PROMPT = args.prompt
    print(f"[config] longest_edge={LONGEST_EDGE} max_new={MAX_NEW}")

    dml = torch_directml.device()
    print(f"DirectML device: {torch_directml.device_name(0)}")

    images = collect_images(args.image, args.image_dir)
    print(f"共 {len(images)} 張圖 | model: {args.model}")

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    print(f"模型載入完成，用時 {time.perf_counter() - t0:.1f}s")

    csv_path = Path(args.csv) if args.csv else DATA_DIR / "benchmark.csv"
    if not args.skip_cpu:
        cpu_res = run_all(model, processor, images, CPU_LABEL, "cpu")
        append_csv(csv_path, CPU_LABEL, args.model, cpu_res)
    dml_res = run_all(model, processor, images, IGPU_LABEL, dml)
    append_csv(csv_path, IGPU_LABEL, args.model, dml_res)

    # 平均摘要
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
    print(f"結果附加到 {csv_path}")


if __name__ == "__main__":
    main()
