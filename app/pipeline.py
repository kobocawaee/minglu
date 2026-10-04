"""
pipeline.py — 所有模式「畫面 → 語句」的邏輯集中處（assistant.py 和 server.py 共用）
=========================================================================================
原本各模式的邏輯在電腦版和手機版各寫一份 → 集中到這裡。各模式：
  read        → OCR（資服版改用 EasyOCR）                     — VLM 讀不出文字（0/17）
  street      → 混合式：號誌模型判斷燈號 + YOLO 判斷路徑上的車（street_mode）
  surrounding/indoor/object → VLM + postprocess
                + 行人複查：YOLO 清楚看到人、但 VLM 沒提到 → 補一句提醒
                  （人貼近鏡頭 = 框很大／只看到一部分，小型 VLM 常認不出是人
                   — 漏掉人很危險，所以用偵測模型再確認一次，結果可預期）
  auto        → router 依 YOLO 看到的東西自動選模式（視障者不必自己選）

自動選模式的規則（依 COCO 類別的經驗法則，約 50ms）：
  看到車／紅綠燈 → street | 畫面中間有大的手持物品 → object |
  家具 → indoor | 其他 → surrounding
  （read 不在自動選項裡 —「想讀字」是使用者的意圖，從畫面猜不出來）

rev. 07-19 — 偏向過馬路模式（在 60 張驗證過的圖上實驗，results/router_eval_v2.md）：
  過馬路線索的信心門檻放寬到 ≥0.25（其他線索維持 ≥0.40）— 只往過馬路偏 = 失效時偏安全
  → 過馬路的召回率 16/20 → 19/20，代價是 1 張無害的誤判。至於「用 LYTNet 當過馬路線索」
  實驗後*不採用*：在非街景的圖（室內／物品）上它會亂回答 red@1.00，40 張錯 24 張
"""

import re

from app import config, postprocess, detect
from app.messages import t, mode_name

# ---- 自動選模式用到的 COCO 類別 ----
_STREET_CUES = detect.VEHICLE_CLASSES | {"traffic light", "stop sign"}
_INDOOR_CUES = {"chair", "couch", "bed", "dining table", "tv", "refrigerator",
                "microwave", "oven", "sink", "toilet", "potted plant"}
_HANDHELD = {"bottle", "cup", "remote", "cell phone", "book", "banana", "apple",
             "orange", "scissors", "toothbrush", "spoon", "fork", "knife", "bowl",
             "mouse", "keyboard", "vase", "clock", "teddy bear"}

_PERSON_RE = re.compile(r"\b(person|people|man|woman|men|women|pedestrian|someone|"
                        r"child|children|kid|boy|girl|human|figure)\b"
                        # [資服版] 中文描述裡提到人的說法（排除「沒有人」）
                        r"|(?<!沒)有人|行人|路人|人們|一個人|一名|一位|男子|女子|男人|女人|"
                        r"男性|女性|小孩|孩童|兒童|學生|騎士", re.I)


def route(image):
    """依偵測模型看到的東西選模式。回傳 (mode, dets) — dets 可以直接接著用
    偏向過馬路：過馬路線索從信心 0.25 就接受（漏判過馬路 = L4 風險，漏判其他模式 = 無害）
    其他線索維持 ≥0.40"""
    dets, (w, h) = detect.detect(image, conf=0.25, classes="all")
    if any(d["cls"] in _STREET_CUES for d in dets):
        return "street", dets

    dets = [d for d in dets if d["conf"] >= 0.40]
    if not dets:                      # 偵測模型不能用／什麼都沒看到 → 一般描述
        return "surrounding", dets

    classes = {d["cls"] for d in dets}

    # 畫面中間有大的手持物品 → 物品模式
    for d in dets:
        if d["cls"] in _HANDHELD:
            x1, y1, x2, y2 = d["box"]
            area = ((x2 - x1) * (y2 - y1)) / (w * h)
            cx = ((x1 + x2) / 2) / w
            if area > 0.12 and 0.2 < cx < 0.8:
                return "object", dets

    if classes & _INDOOR_CUES:
        return "indoor", dets
    return "surrounding", dets


def _person_clearly_present(image) -> bool:
    """YOLO 有沒有「清楚」看到人（信心高＋框夠大 = 近或明顯，不是遠方很小的人）"""
    dets, (w, h) = detect.detect(image, conf=0.50)
    for d in dets:
        if d["cls"] == detect.PERSON_CLASS:
            x1, y1, x2, y2 = d["box"]
            if ((x2 - x1) * (y2 - y1)) / (w * h) > 0.04:
                return True
    return False


def _vlm_read(backend, image) -> str:
    return postprocess.clean(
        backend.describe(image, config.get_prompt("read"),
                         max_new_tokens=config.get_max_tokens("read")))


def _read(backend, image) -> str:
    """[資服版] 讀字模式。engine 見 config.READ_ENGINE。"""
    engine = getattr(config, "READ_ENGINE", "easyocr")
    can_vlm = backend is not None and config.BACKEND != "smolvlm"   # SmolVLM 讀不了字（論文 4.3）
    if engine == "vlm" and can_vlm:
        return _vlm_read(backend, image)
    from app import ocr
    text, kept, total = ocr.read_detail(image)
    if total == 0:
        return t("no_text")
    if engine == "auto" and can_vlm and kept < total * config.OCR_MIN_SURE_RATIO:
        print(f"      [read] EasyOCR 有把握 {kept}/{total} 字 → 改由 VLM 補讀")
        return _vlm_read(backend, image)
    return text or t("no_text")


def describe(backend, image, mode: str):
    """
    處理一張畫面。回傳 (text, mode_used)。
    mode="auto" → 先自動選模式，再在句首說出模式名稱讓使用者知道
    """
    auto = mode == "auto"
    if auto:
        mode, _ = route(image)
        # [資服版] 自動模式選到過馬路時，補做過馬路模式較嚴格的清晰度檢查
        if mode == "street" and config.QUALITY_GATE:
            from app import quality
            ok, reason = quality.assess(image, config.get_quality("street"))
            if not ok:
                return reason, mode

    if mode == "read":
        text = _read(backend, image)
    elif mode == "street":
        from app import street_mode
        text, _hazard, _info = street_mode.describe(backend, image)
    else:
        text = postprocess.clean(
            backend.describe(image, config.get_prompt(mode),
                             max_new_tokens=config.get_max_tokens(mode))
        )
        # 行人複查（混合式）：VLM 沒提到人，但偵測模型清楚看到人 → 補一句提醒
        if mode in ("surrounding", "indoor") and not _PERSON_RE.search(text):
            if _person_clearly_present(image):
                text = (text + " " + t("person_ahead")).strip()

    if auto:
        text = f"{t('mode_prefix', mode=mode_name(mode))} {text}"
    return text, mode
