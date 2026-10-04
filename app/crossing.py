"""
crossing.py — 依偵測框位置做的規則判斷（結果可預期）
=============================================================
把 app/detect.py 偵測到的車輛框轉成提醒，用「位置規則」取代猜測車子是否在移動
（單張畫面看不出動態 — 但看得出「車在哪裡」）：

  危險區（路徑上）= **近（框大／在畫面下半）＋ 在畫面中間（正前方）** 的車
  → 在前方斑馬線上或附近的車 = 「請等待」；遠處或靠邊的車 = 還不是立即的危險

⚠️ 簡化：我們沒有偵測「真正的斑馬線範圍」。試過傳統影像處理（白色門檻），
太不穩定 — 會和其他亮的東西（白色車、建築物）混在一起，邊界偏掉。要準確偵測斑馬線需要
訓練過的分割模型（要有斑馬線資料集）= 未來工作。目前改用「車很近＋在正前方」，
以**框的大小 = 距離遠近**（比用 y 位置更不受相機角度影響）。偏向安全（寧可先提醒）。
"""

from app import detect
from app.messages import t

# 門檻（正規化 0-1）— 以 5 張斑馬線照片校準
#   路徑上 = 在畫面中間（正前方）而且（夠近 = 框大 | 就在腳前 = 位置很低）
#   框的面積 = 最可靠的遠近訊號 → 可以排除「對面車道」（小／遠）的車
_CENTRAL = (0.12, 0.88)   # 中心 x 在這個範圍內 = 在正前方（不是靠邊）
_NEAR_AREA = 0.03         # 框面積 > 畫面的 3% = 近到構成危險
_AT_FEET = 0.75           # 框下緣 > 這個值 = 就在我們面前（即使框不大也算近）


def _in_path(box, w, h) -> bool:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / w
    bottom = y2 / h
    area = ((x2 - x1) * (y2 - y1)) / (w * h)
    central = _CENTRAL[0] < cx < _CENTRAL[1]
    near = area > _NEAR_AREA or bottom > _AT_FEET
    return central and near


def assess(image, conf: float = 0.35, dets=None, size=None):
    """
    回傳 (phrase, hazard: bool, info: dict)。
    phrase = 關於車輛的短句（接在燈號之後唸），hazard = 路徑上是否有車。
    dets/size：可傳入已經跑好的偵測結果重複使用（street_mode 只跑一次 YOLO）
    偵測模型不能用（沒有 ultralytics）→ ("", False, ...) = 不額外說什麼（容錯）。
    """
    if dets is None:
        dets, (w, h) = detect.detect(image, conf=conf)
    else:
        w, h = size
    vehicles = [d for d in dets if d["cls"] in detect.VEHICLE_CLASSES]
    in_path = [d for d in vehicles if _in_path(d["box"], w, h)]
    n_people = sum(1 for d in dets if d["cls"] == detect.PERSON_CLASS)
    info = {"n_vehicles": len(vehicles), "n_in_path": len(in_path), "n_people": n_people}

    if not dets:                                   # 偵測關閉或什麼都沒看到
        return "", False, info
    if in_path:
        return t("veh_in_path"), True, info
    if vehicles:
        return t("veh_nearby"), False, info
    return t("veh_none"), False, info
