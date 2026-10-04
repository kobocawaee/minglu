"""
bench_gemma_npu.py — 在 AMD NPU（Ryzen AI / OGA）上執行 Gemma-3-4b，跑完全部 17 張圖
=============================================================================
目的：補齊 Table 4（安全性）＋品質，讓 Gemma 在同一組 17 張圖上有數字
          可和 SmolVLM-256M/500M 比較（原本 Gemma 只跑了 crosswalk_car 這 1 張）

⚠️ 只能在 NPU 環境中執行：
    conda activate ryzen-ai-1.7.1
    python code/bench_gemma_npu.py

重要說明：
- 這個 Gemma 版本只能在 NPU 執行（provider_options=[{"RyzenAI":{}}]）→ 無法在 CPU/iGPU 上跑
- 使用和 SmolVLM 相同的提示詞（簡潔＋不要猜），安全性比較才公平
  （第一輪曾用 "Describe this image in detail" → 延遲會有些不同，不必驚訝）
- 預計每張約 50 秒（首字延遲約 18 秒＋解碼）× 17 ≈ 15–20 分鐘
- 結果輸出到 results/gemma_npu_17.csv → 用同樣的雙指標標準評安全性

API 注意：onnxruntime_genai 不同版本的 API 不一樣。如果標有
[API] 的那幾行出錯，請改用旁邊註解起來的寫法
"""

import sys
import os
import csv
import time
import argparse

# Windows 主控台 = cp1252 → 避免印出非英文字／emoji 時出錯（檔名含非英文字）
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import psutil

try:
    import onnxruntime_genai as og
except ImportError:
    print("[ERROR] 無法 import onnxruntime_genai — 是不是還沒在 ryzen-ai-1.7.1 環境中執行？")
    print("        conda activate ryzen-ai-1.7.1")
    sys.exit(1)

# 和 SmolVLM 效能測試相同的提示詞（見 data/benchmark.csv），安全性比較才公平
SAFETY_PROMPT = (
    "I am visually impaired. In two short sentences, describe only what you "
    "clearly see and warn me about nearby people or obstacles. Do not guess."
)


def build_prompt(user_text: str) -> str:
    """依 Gemma-3 的對話模板建立提示詞（取自 chat_template.jinja）。
    格式：<bos><start_of_turn>user\n<start_of_image>{text}<end_of_turn>\n<start_of_turn>model\n
    """
    return (
        "<start_of_turn>user\n"
        "<start_of_image>"
        f"{user_text}"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def load_manifest(manifest_path):
    """從 dataset_manifest.csv 讀取 filename、scenario、ground_truth"""
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
    """對 1 張圖執行 Gemma → 回傳 dict {ttft_s, decode_tps, out_tokens, peak_ram_mb, output}"""
    proc = psutil.Process(os.getpid())

    images = og.Images.open(image_path)
    prompt = build_prompt(user_text)
    inputs = processor(prompt, images=images)

    params = og.GeneratorParams(model)
    # do_sample=False = 貪婪解碼（結果固定，和 SmolVLM 的最佳設定一樣）
    params.set_search_options(do_sample=False, max_length=16384)

    generator = og.Generator(model, params)

    out_text = []
    out_tokens = 0
    peak_rss = proc.memory_info().rss

    # --- 首字延遲 = 預填（在 NPU 上編碼圖片）＋第一個 token ---
    # ⚠️ 在 OGA 0.11 中，預填發生在 set_inputs()，不是 generate_next_token()
    #    → 計時必須包住這兩個步驟，否則首字延遲會量成 0
    t0 = time.perf_counter()
    # [API] 新版本：generator.set_inputs(inputs)
    #       某些舊版本：建立 generator 之前先呼叫 params.set_inputs(inputs)
    generator.set_inputs(inputs)
    generator.generate_next_token()
    ttft_s = time.perf_counter() - t0

    tok = generator.get_next_tokens()[0]
    out_text.append(tokenizer_stream.decode(tok))
    out_tokens += 1
    peak_rss = max(peak_rss, proc.memory_info().rss)

    # --- 解碼剩下的 token ---
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
        # 清理：▁ = 偶爾跑出來的 SentencePiece 空白符號，換行 → 空白
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
        help="先暖機跑 1 張圖（第一張的首字延遲會包含編譯／載入時間 — 建議開啟）",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只跑前 N 張圖（0 = 全部）。快速測試用，例如 --limit 1",
    )
    args = ap.parse_args()

    print(f"[info] onnxruntime_genai version: {getattr(og, '__version__', '?')}")
    # ⚠️ 必須是絕對路徑！傳相對路徑的話 OGA 會把 model-dir 重複接兩次
    #    (Cannot read header from models/gemma3_4b_npu/models/gemma3_4b_npu/...pb.bin)
    model_dir_abs = os.path.abspath(args.model_dir)
    print(f"[info] 從 {model_dir_abs} 載入模型（NPU/RyzenAI）...")
    t_load = time.perf_counter()
    model = og.Model(model_dir_abs)
    processor = model.create_multimodal_processor()
    tokenizer_stream = processor.create_stream()
    print(f"[info] 載入完成，用時 {time.perf_counter() - t_load:.1f}s")

    manifest = load_manifest(args.manifest)
    if args.limit > 0:
        manifest = manifest[: args.limit]
        print(f"[info] --limit {args.limit} → 只跑前 {len(manifest)} 張圖（快速測試）")
    print(f"[info] 清單中共 {len(manifest)} 張圖 | prompt: {args.prompt!r}")

    # 暖機（第一次呼叫的首字延遲含編譯時間 → 結果丟掉，不記錄）
    if args.warmup and manifest:
        wpath = os.path.join(args.image_dir, manifest[0]["filename"])
        if os.path.exists(wpath):
            print("[info] 暖機執行（不記錄）...")
            try:
                run_one(model, processor, tokenizer_stream, wpath, args.prompt, args.max_new_tokens)
            except Exception as e:
                print(f"[warn] 暖機出錯（略過）：{e}")

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
            print(f"[{i}/{len(manifest)}] ⚠️ 找不到檔案 {img_path} — 略過")
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
        print(f"\n✅ 完成 {len(results)} 張圖，用時 {elapsed/60:.1f} 分鐘")
        print(f"   AVG TTFT {avg_ttft:.1f}s | AVG peak RAM {avg_ram:.0f}MB")
        print(f"   → {args.out}")
        print("\n[next] 把 output 欄位拿去用雙指標標準評安全性 "
              "（指標 A 方向性／指標 B 捏造），再補上 Table 4 的 Gemma 那一列")
    else:
        print("\n⚠️ 完全沒有結果 — 請檢查圖片路徑／清單")


if __name__ == "__main__":
    main()
