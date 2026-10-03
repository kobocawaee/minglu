"""
postprocess.py — ทำความสะอาด output ของ VLM ก่อนพูด (TTS)
==========================================================
โมเดลจิ๋วชอบพ่น filler ที่ไม่มีประโยชน์/ซ้ำ persona ใน prompt:
  - "I do not know about crossing but i can tell that ..." (hedge จาก street prompt เดิม)
  - "as I am visually impaired ...", "I am blind ..." (echo persona จาก prompt)
prompt ใหม่ตัดต้นตอไปแล้ว แต่โมเดลจิ๋วเดาได้ตลอด → ล้างซ้ำที่ output เป็น safety net.
ใช้กับโหมด VLM เท่านั้น (ไม่ใช้กับ read/OCR).
"""

import re

# แต่ละ pattern = วลีที่จะลบทิ้ง (case-insensitive). เรียงจากยาว→สั้น
_STRIP = [
    r"\bi\s+do\s*not\s+know\s+about\s+crossing[,.]?\s*(but\s+i\s+can\s+tell\s+that\s+)?",
    r"\bi\s+can'?t\s+tell\s+if\s+it'?s\s+safe\s+to\s+cross[,.]?\s*",
    r"\bas\s+i\s+am\s+(a\s+)?(visually\s+)?(impaired|blind)(\s+person)?[,.]?\s*",
    r"\bi\s+am\s+(a\s+)?(visually\s+)?(impaired|blind)(\s+person)?[,.]?\s*",
    r"\bi'?m\s+(a\s+)?(visually\s+)?(impaired|blind)(\s+person)?[,.]?\s*",
    r"\bsince\s+i\s+am\s+(visually\s+)?(impaired|blind)[,.]?\s*",
    # "I can't read (the text/sign)..." — filler จาก VLM ที่อ่าน text ไม่ได้ (finding read=OCR)
    # ตัดทั้งประโยค (ถึงจุด) เพราะไม่มีข้อมูลอะไรให้ผู้ใช้
    r"\bi\s+can'?t\s+read\b[^.]*\.?\s*",
    r"\bi\s+cannot\s+read\b[^.]*\.?\s*",
    r"\bi\s+am\s+unable\s+to\s+read\b[^.]*\.?\s*",
]
_STRIP_RE = [re.compile(p, re.I) for p in _STRIP]


# คำห้อยท้ายที่บ่งว่าประโยคโดนตัดกลางคัน (ชน max_new_tokens) — เล็มทิ้งได้โดยเนื้อหาไม่หาย
_DANGLING = re.compile(
    r"[,;:\s]+(?:and|or|but|with|including|such as|like|near|behind|beside|next to|"
    r"of|in|on|at|to|for|the|a|an|is|are|was|were|which|that|who|"
    r"leading|containing|holding|carrying|heading|pointing|connected|attached)\s*$", re.I)


def _finish_sentence(t: str) -> str:
    """ข้อความที่จบไม่สวย (โดนตัดที่เพดาน token): เล็มคำห้อยท้าย → ปิดประโยคตรงที่เนื้อหาจบจริง
    ไม่เพิ่มความยาว — แค่ทำให้ 'ฟังเหมือนตั้งใจจบ' (feedback field test: 'พูดไม่จบ แต่ไม่เอายาว')"""
    if not t or t[-1] in ".!?":
        return t
    # 1) เล็มคำเชื่อม/บุพบทห้อยท้ายซ้ำๆ เช่น "...pens, tissues, and" → "...pens, tissues"
    for _ in range(4):
        new = _DANGLING.sub("", t).strip().rstrip(",;: ")
        if new == t:
            break
        t = new
    if not t:
        return t
    # 2) ถ้าเศษท้ายสั้นมาก (≤3 คำ) และมีประโยคจบสมบูรณ์อยู่ก่อนแล้ว → ตัดเศษทิ้งทั้งก้อน
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
    """ลบ filler/persona echo + เล็มประโยคโดนตัด + จัดรูปก่อนพูด"""
    if not text:
        return text
    if _CJK.search(text):
        return _clean_zh(text)
    t = text
    for rx in _STRIP_RE:
        t = rx.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()          # ยุบช่องว่างซ้ำ
    t = re.sub(r"^[,.;:\s]+", "", t)             # เศษเครื่องหมายวรรคตอนต้นประโยค
    if not t:
        return t
    t = _finish_sentence(t)                       # โดนตัดกลางประโยค → จบให้สวย
    if not t:
        return t
    t = t[0].upper() + t[1:]                     # ขึ้นต้นตัวใหญ่
    if t[-1] not in ".!?":                        # ปิดท้ายด้วยจุด (กรณี fragment ที่เก็บไว้)
        t += "."
    return t
