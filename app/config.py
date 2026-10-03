"""
config.py — ค่าตั้งกลางของแอป (mode→prompt, เลือก backend, params)
==================================================================
รวมการตัดสินใจจาก Phase 1 ไว้ที่เดียว: prompt ที่ validate แล้ว, best config
(le=384, max_new_tokens=50, rep_penalty=1.2), และ mapping use case → backend
"""

# ---------------------------------------------------------------------------
# Prompt ต่อโหมด — ต่อยอดจาก safety prompt ที่ผ่านการ eval ใน Phase 1
# หลักการ: concise + "only what you clearly see" + "Do not guess"
#          (ลด hallucination + สั้นเหมาะ TTS — ดู docs/results_section.md §5.3)
# ---------------------------------------------------------------------------
# ⚠️ หลัก prompt (จาก live test 1 ก.ค.): balance ระหว่าง 2 error
#   - เอ่ย "people" นำ → หลอนคนในฉากว่าง (phantom-actor)
#   - ห้ามเอ่ยคนเลย → พลาดคนที่ยืนอยู่จริง (อันตราย!)
# → สูตรที่ผ่านเทสต์ 8/8: "include any person who is CLEARLY present. Do not INVENT people
#   who are not there" (report คนจริง + ไม่หลอน) + "if unsure say so"
MODE_PROMPTS = {
    # ⚠️ ไม่ให้ตัดสิน "safe to cross" (ต้นตอ "I don't know about crossing" + อันตราย:
    #   finding §5.5.1 โมเดลอ่านไฟผิดบ่อย). ให้รายงานสีไฟแบบ "สังเกตเห็น" + รถ เท่านั้น
    #   ไม่ใส่ persona "I am visually impaired" (ต้นตอ echo "as I am impaired")
    # HYBRID (street): VLM ตอบ "สีไฟ" อย่างเดียว — เรื่องรถให้ detector (app/crossing.py)
    #   ทำ เพราะ VLM เดา moving/stopped ไม่ได้ + หลอนรถ (§5.5.2). แยกงานตามจุดแข็ง
    "street": (
        "At a crosswalk, in one short sentence, name the colour of the pedestrian or traffic "
        "light: say 'The light is red', 'The light is green', or 'No traffic light visible'. "
        "Only the light colour, nothing else."
    ),
    "surrounding": (
        "I am visually impaired. In one clear sentence, describe the main things in front "
        "of me and roughly where they are. If a person is clearly present, say a person is "
        "in front of me. Do not invent people who are not there. If unsure, say you are not sure."
    ),
    "indoor": (
        "I am visually impaired and navigating indoors. In two short sentences, describe the "
        "layout in front of me and any obstacle in my path that you clearly see. Mention a "
        "person only if one is clearly present; do not invent people. If unsure, say you are not sure."
    ),
    # reading = VizWiz "reading" — ⚠️ โหมด read ใช้ OCR (app/ocr.py) ไม่ใช่ VLM/prompt นี้
    # (VLM อ่าน text ไม่แม่น → แยกไปใช้ OCR engine). prompt นี้เก็บไว้เป็น fallback เฉยๆ
    "read": (
        "I am visually impaired. Read any text, letters, or numbers you see in the image, "
        "exactly as written. Do not guess. If you truly see no text, say 'No text visible.'"
    ),
    # identification = VizWiz "identification" — จ่อของชิ้นเดียวใกล้ๆ (เคสที่ตัวจิ๋วน่าจะทำได้ดี)
    "object": (
        "I am visually impaired and holding an object up to the camera. In one short sentence, "
        "say what the object is. If it has a clear label or text, read it. Only what you clearly "
        "see. If unsure, say you are not sure."
    ),
}

DEFAULT_MODE = "surrounding"

# ---------------------------------------------------------------------------
# [資服版] 輸出語言與 OCR 設定
#   LANG = "zh" → 固定句子用繁體中文（app/messages.py）；"en" → 與論文相同的英文句子
#   OCR 改用 EasyOCR（JaidedAI，Apache-2.0）取代 RapidOCR
# ---------------------------------------------------------------------------
LANG = "zh"
OCR_LANGS = ["ch_tra", "en"]   # 繁體中文 + 英文
OCR_MIN_CONF = 0.30            # 低於此信心的文字不唸（EasyOCR 的雜訊比較多）
# 讀字模式用哪個引擎：
#   "auto"    （預設）先用 EasyOCR；它偵測到文字但大半沒把握時，改由 Gemma 補讀。
#             EasyOCR 完全沒偵測到文字時不叫 Gemma，直接說沒有文字（避免 VLM 憑空編出文字）
#   "easyocr" 只用專用 OCR
#   "vlm"     只用 Gemma
READ_ENGINE = "auto"
OCR_MIN_SURE_RATIO = 0.5       # auto：有把握的字數佔偵測到字數的比例低於此值 → 交給 Gemma

# max_new_tokens ต่อโหมด — จาก finding §5.6: latency ของ SmolVLM = decode-bound
# → ตัด output ให้สั้น = เร็วขึ้นมาก. street ต้องการเร็วสุด (~1.6s@iGPU ที่ 35 tokens)
# ⚠️ caveat: สั้นไป (เช่น <25) จะ drop nuance ความปลอดภัย (ไฟแดง→"Stopped." เฉยๆ) — 35 คือจุดสมดุล
# ลด cap ให้ output สั้นสม่ำเสมอ + เร็วขึ้น (decode-bound §5.6) — จาก live test
# "one short sentence, name objects" ให้ผลสั้นอยู่แล้ว, cap 40 กันเคสพล่ามนานๆ (เคยเจอ 6.7s)
MODE_MAX_TOKENS = {
    "street": 42,        # {สีไฟ}.{สถานะรถ} — ต้องมีที่พอ 2 ท่อน ไม่ตัดกลางประโยค
    "surrounding": 55,   # 1 ประโยคมีรายละเอียด+ตำแหน่ง (~2-3s@iGPU)
    "indoor": 42,        # 2 ประโยคสั้น — เร็วขึ้น (จาก 55, decode-bound §5.6)
    "read": 60,          # text อาจยาว
    "object": 30,        # ชื่อของสั้นๆ
}

# ---------------------------------------------------------------------------
# Backend / device — เลือกตาม use case (สรุปจาก Phase 1 Table 5)
#   real-time  → SmolVLM-500M (CPU พอดี 8GB / iGPU เร็วกว่าแต่ RAM 6.6GB)
#   คุณภาพสูง  → Gemma-3-4b @ NPU (ช้า ~18s ใช้แบบ non-realtime)
# ---------------------------------------------------------------------------
BACKEND = "gemma_hf"         # "gemma_hf"（資服版預設，繁中，NVIDIA 顯示卡）| "smolvlm"（英文）| "gemma_npu"
GEMMA_MODEL = "google/gemma-3-4b-it"
GEMMA_4BIT = True            # 顯示卡用 4-bit 載入（約 3GB）；bf16 約 8.6GB，8GB 顯示卡放不下

# [資服版] 語音指令／語音提問（app/voice.py）
ASR_MODEL = "openai/whisper-large-v3-turbo"   # 在筆電上做語音辨識，聲音不送雲端
ASK_MAX_TOKENS = 80                           # 語音提問的回答長度上限
ASK_HISTORY_TURNS = 3                         # 追問時最多沿用前幾輪問答
GEMMA_GEN_PARAMS = {"max_new_tokens": 60, "repetition_penalty": 1.05}
SMOLVLM_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
DEVICE = "cpu"               # "cpu" | "igpu"  (igpu ต้อง env vlm_dml)

# best config จาก Phase 1 (sweet spot สำหรับ model จิ๋ว)
GEN_PARAMS = {
    "longest_edge": 384,
    "max_new_tokens": 50,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 3,
}

# ---------------------------------------------------------------------------
# Frame-quality gate (app/quality.py) — กันเฟรมเบลอ/มืดเข้า VLM
# ความไม่เสถียรของคำบรรยายส่วนใหญ่มาจากเฟรมคุณภาพแย่ (decoder เป็น greedy อยู่แล้ว)
# threshold คาลิเบรตจาก 17 รูป canonical: blur 192-5424, bright 87-140
#   → ตั้งเผื่อ (min_blur 80 < 192, bright 40-225 กว้างกว่า 87-140) กัน false-reject
# ปิด gate ได้ด้วย QUALITY_GATE = False (หรือ --no-quality-gate)
# ---------------------------------------------------------------------------
QUALITY_GATE = True
QUALITY = {
    "min_blur": 80.0,        # variance ของ Laplacian ต่ำกว่านี้ = เบลอเกินไป
    "min_brightness": 40.0,  # มืดเกิน
    "max_brightness": 225.0, # จ้า/ล้างขาวเกิน
}

# ---------------------------------------------------------------------------
# Capture / loop
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0             # กล้องตัวแรก (เปลี่ยนถ้ามีหลายตัว)
TRIGGER = "key"             # "key" = กด space ถ่าย 1 ครั้ง | "interval" = ถ่ายอัตโนมัติ
INTERVAL_SEC = 5.0          # ใช้เมื่อ TRIGGER="interval"


# ---------------------------------------------------------------------------
# [資服版] 繁體中文提示詞 — 內容與上面的英文版逐句對應，只是換語言
# SmolVLM 不會中文，所以只有在後端不是 smolvlm 而且 LANG = "zh" 時才使用
# ---------------------------------------------------------------------------
MODE_PROMPTS_ZH = {
    "street": (
        "這是行人穿越道。請用一句短句說出行人號誌或紅綠燈的顏色："
        "只能回答「行人號誌是紅燈」、「行人號誌是綠燈」或「沒有看到號誌」。只說燈號，不要說別的。"
    ),
    "surrounding": (
        "我是視障者。請用一句話說明我前方主要有什麼，以及大概在哪個方位。"
        "如果清楚看到有人，就說前方有人；不要編造不存在的人。不確定就說不確定。"
    ),
    "indoor": (
        "我是視障者，正在室內行走。請用兩句短句說明前方的空間配置，"
        "以及你清楚看到、擋在我路上的障礙物。只有清楚看到人時才提到人，不要編造。不確定就說不確定。"
    ),
    "read": (
        "我是視障者。請把畫面中看到的文字、字母或數字照原樣唸出來，不要猜。"
        "如果真的沒有文字，就說「沒有看到文字」。"
    ),
    "object": (
        "我是視障者，正把一樣東西拿到鏡頭前。請用一句短句說出這是什麼；"
        "如果上面有清楚的標籤或文字，請唸出來。只說清楚看到的，不確定就說不確定。"
    ),
}

# 中文一個字大約一個 token，上限比英文版略高，避免句子被截斷
MODE_MAX_TOKENS_ZH = {"street": 20, "surrounding": 60, "indoor": 70, "read": 80, "object": 40}


def _use_zh() -> bool:
    return LANG == "zh" and BACKEND != "smolvlm"


# [資服版] 清晰度門檻依模式調整。
# 原門檻 80 是用 17 張網路照片定的（最低 192 分）；實測手機即時畫面清楚時常只有 36–135 分，
# 近拍平滑物品尤其低，導致物品模式幾乎全被擋。實測與模擬：嚴重模糊 ≤ 23 分、可用畫面 ≥ 36 分。
# 過馬路模式攸關安全，維持原本的 80 不動。
QUALITY_MIN_BLUR_BY_MODE = {"street": 80.0}
QUALITY_MIN_BLUR_DEFAULT = 30.0


def get_quality(mode: str) -> dict:
    """回傳該模式的畫面品質門檻（auto 在選出模式前先用預設值）。"""
    q = dict(QUALITY)
    q["min_blur"] = QUALITY_MIN_BLUR_BY_MODE.get(mode, QUALITY_MIN_BLUR_DEFAULT)
    return q


def get_prompt(mode: str) -> str:
    """คืน prompt ของโหมด (fallback เป็น DEFAULT_MODE ถ้าชื่อผิด)"""
    table = MODE_PROMPTS_ZH if _use_zh() else MODE_PROMPTS
    return table.get(mode, table[DEFAULT_MODE])


def get_max_tokens(mode: str) -> int:
    """คืน max_new_tokens ของโหมด (fallback = ค่าใน GEN_PARAMS)"""
    table = MODE_MAX_TOKENS_ZH if _use_zh() else MODE_MAX_TOKENS
    return table.get(mode, GEN_PARAMS["max_new_tokens"])
