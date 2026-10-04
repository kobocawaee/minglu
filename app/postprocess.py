"""
postprocess.py — 唸出來（TTS）之前，先清理 VLM 的輸出
==========================================================
小模型常常吐出沒用的贅詞，或照抄提示詞裡的角色設定：
  - "I do not know about crossing but i can tell that ..."（舊版過馬路提示詞造成的保留語氣）
  - "as I am visually impaired ..."、"I am blind ..."（照抄提示詞的角色設定）
新的提示詞已經從源頭處理，但小模型還是可能冒出來 → 在輸出端再清一次當保險。
只用在 VLM 的模式（不用在讀字／OCR）。
"""

import re

# 每個 pattern = 要刪掉的片語（不分大小寫）。由長排到短
_STRIP = [
    r"\bi\s+do\s*not\s+know\s+about\s+crossing[,.]?\s*(but\s+i\s+can\s+tell\s+that\s+)?",
    r"\bi\s+can'?t\s+tell\s+if\s+it'?s\s+safe\s+to\s+cross[,.]?\s*",
    r"\bas\s+i\s+am\s+(a\s+)?(visually\s+)?(impaired|blind)(\s+person)?[,.]?\s*",
    r"\bi\s+am\s+(a\s+)?(visually\s+)?(impaired|blind)(\s+person)?[,.]?\s*",
    r"\bi'?m\s+(a\s+)?(visually\s+)?(impaired|blind)(\s+person)?[,.]?\s*",
    r"\bsince\s+i\s+am\s+(visually\s+)?(impaired|blind)[,.]?\s*",
    # "I can't read (the text/sign)..." — VLM 讀不出文字時的贅詞（研究發現：讀字要用 OCR）
    # 整句刪掉（到句號為止），因為對使用者沒有任何資訊
    r"\bi\s+can'?t\s+read\b[^.]*\.?\s*",
    r"\bi\s+cannot\s+read\b[^.]*\.?\s*",
    r"\bi\s+am\s+unable\s+to\s+read\b[^.]*\.?\s*",
]
_STRIP_RE = [re.compile(p, re.I) for p in _STRIP]


# 句尾殘留的字，表示句子被中途截斷（碰到 max_new_tokens）— 刪掉不影響內容
_DANGLING = re.compile(
    r"[,;:\s]+(?:and|or|but|with|including|such as|like|near|behind|beside|next to|"
    r"of|in|on|at|to|for|the|a|an|is|are|was|were|which|that|who|"
    r"leading|containing|holding|carrying|heading|pointing|connected|attached)\s*$", re.I)


def _finish_sentence(t: str) -> str:
    """結尾不完整的句子（在 token 上限被截斷）：刪掉句尾殘字 → 在內容真正結束的地方收尾
    不增加長度 — 只是讓它「聽起來是刻意結束的」（實測回饋：『話沒說完，但也不要變長』）"""
    if not t or t[-1] in ".!?":
        return t
    # 1) 反覆刪掉句尾的連接詞／介系詞，例如 "...pens, tissues, and" → "...pens, tissues"
    for _ in range(4):
        new = _DANGLING.sub("", t).strip().rstrip(",;: ")
        if new == t:
            break
        t = new
    if not t:
        return t
    # 2) 如果結尾殘段很短（≤3 個字）而且前面已經有完整的句子 → 整段刪掉
    last = max(t.rfind("."), t.rfind("!"), t.rfind("?"))
    if last != -1:
        frag = t[last + 1:].strip()
        if frag and len(frag.split()) <= 3:
            return t[:last + 1]
    return t + "."


_CJK = re.compile(r"[\u3400-\u9fff]")


def _clean_zh(text: str) -> str:
    """[資服版] 中文描述的整理：去掉符號與多餘空白；句子被長度上限截斷時，收在最後一個完整句子。"""
    t = re.sub(r"[*#`>]+", "", text)                 # 模型偶爾輸出的 Markdown 符號
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^[，。、；：\s]+", "", t)
    if not t or t[-1] in "。！？":
        return t
    last = max(t.rfind("。"), t.rfind("！"), t.rfind("？"))
    if last != -1 and len(t) - last - 1 <= 12:       # 尾巴只是一小段沒講完的話 → 丟掉
        return t[:last + 1]
    return t.rstrip("，、；： ") + "。"


def clean(text: str) -> str:
    """刪掉贅詞／照抄的角色設定＋修剪被截斷的句子＋整理格式，再交給語音"""
    if not text:
        return text
    if _CJK.search(text):
        return _clean_zh(text)
    t = text
    for rx in _STRIP_RE:
        t = rx.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()          # 合併連續空白
    t = re.sub(r"^[,.;:\s]+", "", t)             # 句首殘留的標點
    if not t:
        return t
    t = _finish_sentence(t)                       # 句子被截斷 → 收尾
    if not t:
        return t
    t = t[0].upper() + t[1:]                     # 開頭大寫
    if t[-1] not in ".!?":                        # 結尾補上句號（保留的片段）
        t += "."
    return t
