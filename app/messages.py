"""
messages.py — 助理會說出口的固定句子（繁體中文 / 英文）
=====================================================
[資服版新增] 原本各模組把英文句子寫死在程式裡；這裡集中管理，由 config.LANG 切換。
  config.LANG = "zh" → 繁體中文（預設，參賽版）
  config.LANG = "en" → 與論文完全相同的英文句子（評測腳本重現論文數字時使用）
"""

from app import config

MESSAGES = {
    # ---- 過馬路模式：號誌 ----
    "light_red":       {"en": "The light is red.",
                        "zh": "行人號誌是紅燈。"},
    "light_green":     {"en": "The light is green.",
                        "zh": "行人號誌是綠燈。"},
    # [資服版] 中文改成只報證據：臺灣的行人號誌在綠燈期間也會倒數（例如小綠人＋49 秒），
    #   LYTNet 的「倒數」類別分不出是紅燈還是綠燈在倒數，說「請不要過馬路」會誤導。
    #   英文維持論文原句（評測腳本靠比對英文句子計分）
    "light_countdown": {"en": "The light is counting down. Do not start crossing.",
                        "zh": "看到行人號誌正在倒數，但無法確定是紅燈還是綠燈，請再確認。"},
    "light_none":      {"en": "No pedestrian light detected.",
                        "zh": "沒有偵測到行人號誌。"},
    "light_unclear":   {"en": "No clear pedestrian light detected.",
                        "zh": "無法確定行人號誌的燈號。"},
    # ---- 過馬路模式：車輛 ----
    "veh_in_path":     {"en": "Caution, a vehicle is in your path ahead. Wait.",
                        "zh": "注意，前方路徑上有車，請等待。"},
    "veh_nearby":      {"en": "Vehicles are nearby but not in your path.",
                        "zh": "附近有車，但不在你的行進路徑上。"},
    "veh_none":        {"en": "No vehicles ahead.",
                        "zh": "前方沒有車。"},
    # ---- 畫面品質檢查 ----
    "q_blur":          {"en": "Image is blurry. Hold the camera steady.",
                        "zh": "畫面模糊，請拿穩鏡頭。"},
    "q_dark":          {"en": "Too dark. Find better lighting.",
                        "zh": "太暗了，請找亮一點的地方。"},
    "q_bright":        {"en": "Too bright or washed out. Adjust the angle.",
                        "zh": "畫面太亮，請調整角度。"},
    # ---- 其他 ----
    "person_ahead":    {"en": "A person is in front of you.",
                        "zh": "有人在你前方。"},
    "no_text":         {"en": "No text visible.",
                        "zh": "沒有看到文字。"},
    "mode_prefix":     {"en": "{mode} mode.",
                        "zh": "{mode}模式。"},
    "ready":           {"en": "Assistant ready. Mode {mode}.",
                        "zh": "助理已就緒，目前是{mode}模式。"},
}

MODE_NAMES = {
    "street":      {"en": "Street",      "zh": "過馬路"},
    "surrounding": {"en": "Surrounding", "zh": "周遭"},
    "indoor":      {"en": "Indoor",      "zh": "室內"},
    "read":        {"en": "Read",        "zh": "讀字"},
    "object":      {"en": "Object",      "zh": "物品"},
    "auto":        {"en": "Auto",        "zh": "自動"},
}


def _lang() -> str:
    return "zh" if getattr(config, "LANG", "zh") == "zh" else "en"


def t(key: str, **kwargs) -> str:
    """取出目前語言的句子；kwargs 用來填入 {mode} 這類空格。"""
    text = MESSAGES[key][_lang()]
    return text.format(**kwargs) if kwargs else text


def mode_name(mode: str) -> str:
    return MODE_NAMES.get(mode, {}).get(_lang(), mode)
