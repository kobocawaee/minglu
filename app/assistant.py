"""
assistant.py — 影像描述助理的主迴圈（Phase 2，電腦端版本）
=========================================================
把所有元件串起來：攝影機 → VLM（依模式）→ 語音

執行方式（依 config.py 選的後端使用對應環境）：
    conda activate vlm_research          # DEVICE="cpu" 時
    python -m app.assistant --mode street
    python -m app.assistant --mode surrounding --source data/test_images/crosswalk_car.jpg

模式：street | surrounding | indoor   （提示詞不同 — 見 config.py）
觸發：按空白鍵拍一張並描述，按 q 離開（config.TRIGGER="key"）
"""

import sys
import time
import argparse

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app import config
from app.backends import get_backend
from app.capture import Camera, load_image_file
from app.speak import Speaker
from app.messages import t, mode_name


def build_backend(backend=None, device=None):
    backend = backend or config.BACKEND
    device = device or config.DEVICE
    config.BACKEND = backend          # [資服版] 讓提示詞語言跟著實際使用的後端走
    if backend == "gemma_hf":
        return get_backend("gemma_hf", model_id=config.GEMMA_MODEL,
                           gen_params=config.GEMMA_GEN_PARAMS)
    if backend == "smolvlm":
        return get_backend(
            "smolvlm",
            model_id=config.SMOLVLM_MODEL,
            device=device,
            gen_params=config.GEN_PARAMS,
        )
    elif backend == "gemma_npu":
        return get_backend("gemma_npu", gen_params=config.GEN_PARAMS)
    raise ValueError(f"後端名稱錯誤：{backend}")


def describe_once(backend, speaker, image, mode):
    """拍一張 → 描述 → 唸出來，並量測時間"""
    # 畫面品質檢查：畫面模糊或太暗 → 不推論（避免亂描述），請使用者再試一次
    if config.QUALITY_GATE:
        from app import quality
        ok, reason = quality.assess(image, config.get_quality(mode))
        if not ok:
            print(f"[{mode}] (skip) {reason}")
            speaker.say(reason)
            return reason
    t0 = time.perf_counter()
    from app import pipeline              # 所有模式的邏輯集中在這裡（OCR／過馬路混合流程／自動）
    text, _mode_used = pipeline.describe(backend, image, mode)
    dt = time.perf_counter() - t0
    print(f"[{mode}] ({dt:.1f}s) {text}")
    speaker.say(text)
    return text


def run_on_file(backend, speaker, path, mode):
    """測試模式：描述圖片檔（不用攝影機）"""
    image = load_image_file(path)
    describe_once(backend, speaker, image, mode)


def run_live(backend, speaker, mode):
    """實際使用模式：攝影機＋觸發"""
    import cv2

    with Camera(config.CAMERA_INDEX) as cam:
        speaker.say(t("ready", mode=mode_name(mode)))
        print("按空白鍵 = 描述 | q = 離開")
        last = 0.0
        while True:
            image = cam.grab()
            # 顯示預覽畫面（給開發者看；實際使用者不需要看螢幕）
            import numpy as np
            cv2.imshow("assistant", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF

            if config.TRIGGER == "interval":
                if time.perf_counter() - last >= config.INTERVAL_SEC:
                    describe_once(backend, speaker, image, mode)
                    last = time.perf_counter()
            elif key == ord(" "):
                describe_once(backend, speaker, image, mode)

            if key == ord("q"):
                break
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=config.DEFAULT_MODE,
                    choices=list(config.MODE_PROMPTS.keys()) + ["auto"])
    ap.add_argument("--source", default=None,
                    help="圖片檔路徑（不用攝影機測試）。不給 = 使用即時攝影機")
    ap.add_argument("--backend", default=None, choices=["gemma_hf", "smolvlm", "gemma_npu"],
                    help="覆寫 config.BACKEND（gemma_npu 需要 ryzen-ai-1.7.1 環境）")
    ap.add_argument("--device", default=None, choices=["cpu", "igpu"],
                    help="覆寫 config.DEVICE（igpu 需要 vlm_dml 環境）")
    ap.add_argument("--read-engine", default=None, choices=["auto", "easyocr", "vlm"],
                    help="[資服版] 讀字模式的引擎：auto（預設）、easyocr、vlm")
    ap.add_argument("--no-quality-gate", action="store_true",
                    help="關閉畫面品質檢查（模糊／太暗）— 展示或測試用")
    args = ap.parse_args()

    if args.no_quality_gate:
        config.QUALITY_GATE = False
    if args.read_engine:
        config.READ_ENGINE = args.read_engine

    backend_name = args.backend or config.BACKEND
    _dev = "cuda/cpu（自動）" if backend_name == "gemma_hf" else (args.device or config.DEVICE)
    print(f"[info] backend={backend_name} device={_dev} mode={args.mode}")
    backend = build_backend(args.backend, args.device)
    print("[info] 載入模型 ...")
    backend.load()
    backend.warmup()
    print(f"[info] 實際使用：{backend.name}")
    speaker = Speaker()

    if args.source:
        run_on_file(backend, speaker, args.source, args.mode)
    else:
        run_live(backend, speaker, args.mode)


if __name__ == "__main__":
    main()
