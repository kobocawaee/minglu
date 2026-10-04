"""
config.py — 應用程式的集中設定（模式→提示詞、選擇後端、參數）
==================================================================
把 Phase 1 的決定集中在一處：驗證過的提示詞、最佳設定
（le=384、max_new_tokens=50、rep_penalty=1.2），以及使用情境 → 後端的對應
"""

# ---------------------------------------------------------------------------
# 各模式的提示詞 — 延續 Phase 1 評測過的安全提示詞
# 原則：簡短 + 「只說清楚看到的」+「不要猜」
#       （減少幻覺 + 短句適合語音 — 見 docs/results_section.md §5.3）
# ---------------------------------------------------------------------------
# ⚠️ 提示詞原則（7/1 實測）：要在兩種錯誤之間取得平衡
#   - 先提到「people」→ 在空無一人的場景幻想出人（phantom-actor）
#   - 完全不准提到人 → 漏掉真的站在那裡的人（危險！）
# → 通過 8/8 測試的寫法："include any person who is CLEARLY present. Do not INVENT people
#   who are not there"（真的有人才說＋不幻想）+ "if unsure say so"
MODE_PROMPTS = {
    # ⚠️ 不讓模型判斷「能不能安全過馬路」（"I don't know about crossing" 的來源，而且危險：
    #   §5.5.1 發現模型常讀錯燈號）。只回報「看到的」燈號＋車輛
    #   不加 "I am visually impaired" 的角色設定（會被照抄成 "as I am impaired"）
    # 混合式（street）：VLM 只回答「燈號顏色」— 車輛交給偵測模型（app/crossing.py）
    #   處理，因為 VLM 分不出車在動還是停著，還會幻想出車（§5.5.2）。依專長分工
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
    # reading = VizWiz 的「reading」— ⚠️ 讀字模式用 OCR（app/ocr.py），不是 VLM／這個提示詞
    # （VLM 讀文字不準 → 改用 OCR 引擎）。這個提示詞只是保留作為備援
    "read": (
        "I am visually impaired. Read any text, letters, or numbers you see in the image, "
        "exactly as written. Do not guess. If you truly see no text, say 'No text visible.'"
    ),
    # identification = VizWiz 的「identification」— 把單一物品拿近鏡頭（小模型應該做得到的情況）
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

# 各模式的 max_new_tokens — 依 §5.6 的發現：SmolVLM 的延遲卡在逐字產生（decode-bound）
# → 輸出越短越快。過馬路最需要快（iGPU 上 35 tokens 約 1.6 秒）
# ⚠️ 注意：太短（例如 <25）會丟掉安全相關的細節（紅燈 → 只剩「Stopped.」）— 35 是平衡點
# 降低上限讓輸出一致地短＋更快（decode-bound §5.6）— 來自實測
# 「one short sentence, name objects」本來就很短，上限 40 是防止偶爾長篇大論（曾經遇過 6.7 秒）
MODE_MAX_TOKENS = {
    "street": 42,        # {燈號}.{車輛狀態} — 要留夠兩段的長度，不在句子中間截斷
    "surrounding": 55,   # 一句有細節＋方位的描述（iGPU 上約 2-3 秒）
    "indoor": 42,        # 兩個短句 — 比較快（原本 55，decode-bound §5.6）
    "read": 60,          # 文字可能很長
    "object": 30,        # 物品名稱很短
}

# ---------------------------------------------------------------------------
# 後端／裝置 — 依使用情境選擇（整理自 Phase 1 Table 5）
#   即時     → SmolVLM-500M（CPU 剛好 8GB／iGPU 較快但佔記憶體 6.6GB）
#   高品質   → NPU 上的 Gemma-3-4b（約 18 秒，非即時使用）；資服版預設 gemma_hf（NVIDIA 顯示卡）
# ---------------------------------------------------------------------------
BACKEND = "gemma_hf"         # "gemma_hf"（資服版預設，繁中，NVIDIA 顯示卡）| "smolvlm"（英文）| "gemma_npu"
GEMMA_MODEL = "google/gemma-3-4b-it"
GEMMA_4BIT = True            # 顯示卡用 4-bit 載入（約 3GB）；bf16 約 8.6GB，8GB 顯示卡放不下

# [資服版] 臺灣行人號誌偵測（app/ped_detector.py；模型檔 models/pedlight.pt 不存在時自動停用）
PED_DETECTOR = True
PED_IMGSZ = 1536          # 號誌很小，用較大的輸入尺寸
PED_RED_MIN = 0.30        # 依 results/field_1004 實地考試（3 個路口 52 張）：紅 13/31、綠 1/21、紅說成綠 0
PED_GREEN_MIN = 0.60      # 綠燈門檻較高：誤報綠燈最危險

# [資服版] 語音指令／語音提問（app/voice.py）
ASR_MODEL = "openai/whisper-large-v3-turbo"   # 在筆電上做語音辨識，聲音不送雲端
ASK_MAX_TOKENS = 80                           # 語音提問的回答長度上限
ASK_HISTORY_TURNS = 3                         # 追問時最多沿用前幾輪問答
GEMMA_GEN_PARAMS = {"max_new_tokens": 60, "repetition_penalty": 1.05}
SMOLVLM_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
DEVICE = "cpu"               # "cpu" | "igpu"  （igpu 需要 vlm_dml 環境）

# Phase 1 的最佳設定（小模型的最佳點）
GEN_PARAMS = {
    "longest_edge": 384,
    "max_new_tokens": 50,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 3,
}

# ---------------------------------------------------------------------------
# 畫面品質檢查（app/quality.py）— 不讓模糊或太暗的畫面進到 VLM
# 描述不穩定大多來自畫面品質差（解碼已經是貪婪法）
# 門檻以 17 張標準測試圖校準：模糊度 192-5424、亮度 87-140
#   → 留有餘裕（min_blur 80 < 192，亮度 40-225 比 87-140 寬），避免誤擋
# 可設 QUALITY_GATE = False 關閉（或用 --no-quality-gate）
# ---------------------------------------------------------------------------
QUALITY_GATE = True
QUALITY = {
    "min_blur": 80.0,        # Laplacian 變異數低於這個值 = 太模糊
    "min_brightness": 40.0,  # 太暗
    "max_brightness": 225.0, # 太亮／過曝
}

# ---------------------------------------------------------------------------
# Capture / loop
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0             # 第一台攝影機（有多台時可以改）
TRIGGER = "key"             # "key" = 按空白鍵拍一張 | "interval" = 定時自動拍
INTERVAL_SEC = 5.0          # TRIGGER="interval" 時使用


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
    """回傳該模式的提示詞（名稱錯誤時改用 DEFAULT_MODE）"""
    table = MODE_PROMPTS_ZH if _use_zh() else MODE_PROMPTS
    return table.get(mode, table[DEFAULT_MODE])


def get_max_tokens(mode: str) -> int:
    """回傳該模式的 max_new_tokens（沒有設定時用 GEN_PARAMS 的值）"""
    table = MODE_MAX_TOKENS_ZH if _use_zh() else MODE_MAX_TOKENS
    return table.get(mode, GEN_PARAMS["max_new_tokens"])
