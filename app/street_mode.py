"""
street_mode.py — 過馬路模式的混合式流程（結果完全可預期）
=============================================================================
兩個專用管道 — 這個模式已經不使用 VLM（從約 2-4 秒加快到約 0.3 秒）：
  - 號誌模型（資服版：臺灣號誌 YOLOv8n 優先，再交給 LYTNetV2）→ 行人號誌燈號（分辨能力約 90 個百分點，VLM 約 0，§5.5.1）
  - YOLOv8n ＋ 位置規則（crossing）→ 路徑上的車（解決 VLM 亂猜／幻想出車的問題，§5.5.2）
合併結果：「行人號誌是紅燈。注意，前方路徑上有車，請等待。」

防止亂猜燈號的三層防護（見 light_classifier.py）：none 類別 ＋ YOLO 要看到號誌 ＋ 信心門檻

備援：LYTNet 不能用時（例如 NPU 環境沒有 torch）→ 改用舊方法問 VLM 燈號
（較不準 — 見 §5.5.1）／YOLO 不能用時 → 有多少說多少（每一層都容錯）
"""

from app import config, postprocess, crossing, detect, light_classifier

# 過馬路模式只跑一次 YOLO，用到的類別：車（位置規則）＋人＋號誌（燈號關卡）
_STREET_CLASSES = detect.VEHICLE_CLASSES | {detect.PERSON_CLASS, "traffic light"}

# 跨畫面的燈號記憶（每個程序一位使用者 — 電腦版和伺服器都只服務一個人）
_LIGHT_HISTORY = light_classifier.LightHistory()

# 夜間防護（rev. 07-21，四個路口＋夜市實地測試後）：
# 夜間 LYTNet 信心 1.00 卻兩個方向都會錯 — 被彩色招牌／路燈誤導
# （夜市沒有行人號誌 → 22 張裡 15 張回答「green」= 往危險方向誤報可通行）
# → 暗的場景：只有 YOLO 在「這一張」看到號誌框才報燈號（不可用一致性／記憶代替）
_NIGHT_LEVEL = 90.0   # rev. 07-27（原本 80 — 低於實際量到的夜間範圍）


def _is_night(image) -> bool:
    """是不是暗的場景（夜間）— 平均灰階 < _NIGHT_LEVEL

    實際量測原始影片的 175 張畫面（`code/eval_night_threshold.py`）：
      白天兩個新路口 97-137 · **7/14 的影片 81-120** · **夜間 65-87**
    → 白天和夜間**在 81-87 重疊**，沒有單一門檻能分開，只能選擇要犧牲什麼

    重新評分整個系統（`code/score_field_clips.py`，先確認能精確重現 80 的結果）：
      | | T=80 | T=90 |
      | 說出燈號 | 99/148 | 57/148 |
      | 顏色錯誤 | 28 | 6 |
      | 夜間 2 段影片 | 說了 29 次（錯 22）| 說了 1 次（錯 0）|
      | 7/14（白天）| 29/45 | 13/45 |
      | 白天兩個新路口 | 43/63 | 43/63（不受影響）|

    選 90，因為論文宣稱「夜間系統保持沉默、不亂猜」，而設 80 時程式**實際上做不到**
    （說了 29 次、錯 22 次），代價是在一段異常偏暗的白天影片上少報一些燈號

    ⚠️ 這個門檻**沒有解決剩下的 1/22 誤報可通行** — 那張是因為
    「YOLO 看到號誌框」這個條件漏掉的，跟亮度無關，門檻設多少都擋不住"""
    import numpy as np
    return float(np.asarray(image.convert("L")).mean()) < _NIGHT_LEVEL


def _sees_light(image, dets) -> bool:
    """YOLO 有沒有看到號誌 — 分兩階段：先看整張，沒看到再把上半部放大兩倍
    （實地測試：行人號誌又小又遠，YOLOv8n 在整張畫面上只看到 3/45 張）"""
    if any(d["cls"] == "traffic light" for d in dets):
        return True
    w, h = image.size
    top = image.crop((0, 0, w, int(h * 0.55)))
    top = top.resize((w * 2, int(h * 0.55) * 2))
    dets2, _ = detect.detect(top, conf=0.20, classes={"traffic light"})
    return bool(dets2)


def describe(backend, image):
    """回傳 (text, hazard, info)。text = 燈號（號誌模型）＋車輛提醒（偵測模型）"""
    # YOLO 只跑一次，兩個管道共用
    dets, (w, h) = detect.detect(image, conf=0.25, classes=_STREET_CLASSES)
    yolo_sees_light = _sees_light(image, dets)

    # [資服版] 白天先用臺灣行人號誌偵測模型；沒偵測到才交回下面原本的 LYTNet 流程
    #   （訓練資料都是白天，夜間不用它）
    from app import ped_detector
    light = None
    if ped_detector.available() and not _is_night(image):
        light = ped_detector.phrase(image)

    # 管道 1：燈號 — LYTNet（沒有權重檔時改問 VLM）
    if light is not None:
        pass
    elif light_classifier.available():
        night = _is_night(image)
        box_now = any(d["cls"] == "traffic light" for d in dets)
        if night and not box_now:
            light = ""            # 暗的場景、這張畫面沒有號誌的證據 → 保持沉默（偏安全）
        else:
            light = light_classifier.light_phrase(image, yolo_sees_light,
                                                  history=_LIGHT_HISTORY)
    else:
        light = postprocess.clean(
            backend.describe(image, config.get_prompt("street"),
                             max_new_tokens=config.get_max_tokens("street"))
        )

    # 管道 2：路徑上的車 — 在同一份偵測結果上套用位置規則（信心 ≥0.35，和原規則相同）
    veh_dets = [d for d in dets if d["conf"] >= 0.35]
    veh_phrase, hazard, info = crossing.assess(image, dets=veh_dets, size=(w, h))
    info["yolo_sees_light"] = yolo_sees_light

    text = " ".join(p for p in (light, veh_phrase) if p).strip()
    return text, hazard, info
