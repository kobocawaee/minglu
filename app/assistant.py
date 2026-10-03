"""
assistant.py — main loop ของแอปผู้ช่วยบรรยายภาพ (Phase 2)
=========================================================
ร้อยทุกชิ้นเข้าด้วยกัน:  กล้อง → VLM (ตามโหมด) → เสียงพูด

วิธีรัน (env ตาม backend ที่เลือกใน config.py):
    conda activate vlm_research          # ถ้า DEVICE="cpu"
    python -m app.assistant --mode street
    python -m app.assistant --mode surrounding --source data/test_images/crosswalk_car.jpg

โหมด: street | surrounding | indoor   (prompt ต่างกัน — ดู config.py)
trigger: กด SPACE เพื่อถ่าย+บรรยาย 1 ครั้ง, กด q เพื่อออก (config.TRIGGER="key")
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
    raise ValueError(f"backend ไม่ถูกต้อง: {backend}")


def describe_once(backend, speaker, image, mode):
    """ถ่าย 1 รูป → บรรยาย → พูด + วัดเวลา"""
    # frame-quality gate: เฟรมเบลอ/มืด → ไม่ infer (กันคำบรรยายมั่ว) บอกผู้ใช้ลองใหม่
    if config.QUALITY_GATE:
        from app import quality
        ok, reason = quality.assess(image, config.get_quality(mode))
        if not ok:
            print(f"[{mode}] (skip) {reason}")
            speaker.say(reason)
            return reason
    t0 = time.perf_counter()
    from app import pipeline              # logic ทุกโหมดรวมที่เดียว (OCR/street hybrid/auto)
    text, _mode_used = pipeline.describe(backend, image, mode)
    dt = time.perf_counter() - t0
    print(f"[{mode}] ({dt:.1f}s) {text}")
    speaker.say(text)
    return text


def run_on_file(backend, speaker, path, mode):
    """โหมดทดสอบ: บรรยายภาพจากไฟล์ (ไม่ใช้กล้อง)"""
    image = load_image_file(path)
    describe_once(backend, speaker, image, mode)


def run_live(backend, speaker, mode):
    """โหมดใช้งานจริง: กล้อง + trigger"""
    import cv2

    with Camera(config.CAMERA_INDEX) as cam:
        speaker.say(t("ready", mode=mode_name(mode)))
        print("กด SPACE = บรรยาย | q = ออก")
        last = 0.0
        while True:
            image = cam.grab()
            # โชว์ preview (สำหรับ dev; ผู้ใช้จริงไม่ต้องมองจอ)
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
                    help="path ไฟล์ภาพ (ทดสอบโดยไม่ใช้กล้อง). ไม่ใส่ = ใช้กล้องสด")
    ap.add_argument("--backend", default=None, choices=["gemma_hf", "smolvlm", "gemma_npu"],
                    help="override config.BACKEND (gemma_npu ต้อง env ryzen-ai-1.7.1)")
    ap.add_argument("--device", default=None, choices=["cpu", "igpu"],
                    help="override config.DEVICE (igpu ต้อง env vlm_dml)")
    ap.add_argument("--read-engine", default=None, choices=["auto", "easyocr", "vlm"],
                    help="[資服版] 讀字模式的引擎：auto（預設）、easyocr、vlm")
    ap.add_argument("--no-quality-gate", action="store_true",
                    help="ปิด frame-quality gate (เบลอ/มืด) — ไว้ demo/test")
    args = ap.parse_args()

    if args.no_quality_gate:
        config.QUALITY_GATE = False
    if args.read_engine:
        config.READ_ENGINE = args.read_engine

    backend_name = args.backend or config.BACKEND
    _dev = "cuda/cpu（自動）" if backend_name == "gemma_hf" else (args.device or config.DEVICE)
    print(f"[info] backend={backend_name} device={_dev} mode={args.mode}")
    backend = build_backend(args.backend, args.device)
    print("[info] โหลด model ...")
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
