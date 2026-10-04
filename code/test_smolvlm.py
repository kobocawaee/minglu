"""
在本機（CPU）測試 SmolVLM，用於 VLM 離線視覺助理
- 從 HuggingFace 載入模型（第一次需要網路，之後會快取）
- 可以輸入單張圖、整個資料夾，或使用範例圖
- 描述影像＋量測延遲（分開預填／解碼）與記憶體
- 支援解析度掃描，找出速度和品質的平衡點
- 結果自動存到 data/benchmark.csv

執行方式：
    python code/test_smolvlm.py                                # 範例圖 + 預設提示詞
    python code/test_smolvlm.py --image-dir data/test_images   # 逐張處理資料夾中的圖
    python code/test_smolvlm.py --longest-edge 384 512 768      # 掃描多種解析度
    python code/test_smolvlm.py --model HuggingFaceTB/SmolVLM-500M-Instruct
    python code/test_smolvlm.py --no-split                      # 關閉影像切塊
"""

import argparse
import csv
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# Windows 主控台用 cp1252，無法印出非英文字，先強制改成 UTF-8
sys.stdout.reconfigure(encoding="utf-8")

import torch
import psutil
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
# 電腦視覺領域常用的標準圖（沙發上的兩隻貓），網址穩定，用來快速測試（室內場景）
SAMPLE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 強調安全＋簡潔＋避免幻覺的提示詞（用於周遭環境感知）
SAFETY_PROMPT = (
    "I am visually impaired. In two short sentences, describe only what you clearly "
    "see and warn me about nearby people or obstacles. Do not guess."
)


def collect_images(image_arg, image_dir_arg):
    """回傳 (PIL 影像, 檔名) 的 list — 支援資料夾、單張圖或預設範例"""
    if image_dir_arg:
        paths = sorted(
            p for p in Path(image_dir_arg).iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not paths:
            raise SystemExit(f"在 {image_dir_arg} 找不到圖片")
        return [(Image.open(p).convert("RGB"), p.name) for p in paths]
    if image_arg:
        p = Path(image_arg)
        return [(Image.open(p).convert("RGB"), p.name)]
    # 預設：範例圖
    DATA_DIR.mkdir(exist_ok=True)
    sample = DATA_DIR / "sample_indoor.jpg"
    if not sample.exists():
        print(f"正在從 {SAMPLE_URL} 下載範例圖 ...")
        urllib.request.urlretrieve(SAMPLE_URL, sample)
    return [(Image.open(sample).convert("RGB"), sample.name)]


def run_once(model, processor, image, label, prompt, max_new_tokens, longest_edge,
             no_split, rep_penalty, no_repeat_ngram):
    """依指定設定跑一次推論，回傳指標 dict"""
    # 用兩個開關控制 vision token 數：影像切塊和解析度
    processor.image_processor.do_image_splitting = not no_split
    if longest_edge is not None:
        processor.image_processor.size = {"longest_edge": longest_edge}

    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt")
    n_prompt = inputs["input_ids"].shape[1]  # 輸入 token 數（反映圖片變成多少 token）

    # 預填 = 產生第一個 token 的時間 ≈ 處理圖片的時間
    t_pf = time.perf_counter()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1)
    prefill = time.perf_counter() - t_pf

    # 完整產生 — 加上參數避免無限重複
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
    """把每次執行的結果附加到 CSV — 檔案不存在時會先寫標題列"""
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
                "image": r["image"],  # 只存檔名（放上 GitHub 也能用）
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
        help='裝置標籤，例如 "AMD-Ryzen-CPU"（Mhiu）或 "Intel-CPU"（Aomam）',
    )
    parser.add_argument("--image", default=None, help="單張圖的路徑")
    parser.add_argument("--image-dir", default=None, help="圖片資料夾（逐張處理）")
    parser.add_argument("--prompt", default=SAFETY_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--repetition-penalty", type=float, default=1.2, help="懲罰重複的 token（避免無限重複）")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3, help="禁止 n 個字的片語重複（避免無限重複）")
    parser.add_argument("--no-split", action="store_true", help="關閉影像切塊")
    parser.add_argument(
        "--longest-edge",
        type=int,
        nargs="+",
        default=[None],
        help="要掃描的解析度，例如 --longest-edge 384 512 768 1024",
    )
    args = parser.parse_args()

    proc = psutil.Process()
    images = collect_images(args.image, args.image_dir)
    print(f"共 {len(images)} 張圖 | PROMPT: {args.prompt}\n")

    # ---------- 模型只載入一次 ----------
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    print(f"MODEL: {args.model}")
    print(f"模型載入完成，用時 {time.perf_counter() - t0:.1f}s\n")

    # ---------- 逐張圖 × 每種解析度 ----------
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

    # ---------- 整理成表格 ----------
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

    # ---------- 結果存到 CSV ----------
    csv_path = DATA_DIR / "benchmark.csv"
    append_csv(
        csv_path, args.device, args.model, args.no_split, args.prompt,
        args.repetition_penalty, args.no_repeat_ngram_size, results,
    )
    print(f"已存 {len(results)} 列結果到 {csv_path}")


if __name__ == "__main__":
    main()
