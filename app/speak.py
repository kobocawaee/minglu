"""
speak.py — 把文字唸出來（筆電端，離線；英文 + 台灣口音中文）
==========================================================
使用者是視障者，主要輸出是聲音，而且必須完全離線。

[資服版] 中文語音不再使用 Piper 的 zh_CN-huayan（中國口音語音模型），
改用 Windows 內建的台灣中文語音（SAPI，例如 Microsoft Hanhan / Yating / Zhiwei）。

  英文片段 → Piper en_US-lessac-medium（若有安裝），否則 Windows 英文語音
  中文片段 → Windows 台灣中文語音（zh-TW）
  都沒有   → 印在畫面上（最後的備援）

Windows 若沒有台灣中文語音：設定 → 時間與語言 → 語音 → 新增語音 → 中文（台灣）。
手機端不走這個檔案，而是用手機系統語音（見 server.py）。

一句話裡中英混雜時，會切成中文段和英文段，各用對應的語音唸，避免英文語音硬唸中文字。
"""

import re
import wave
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PIPER_DIR = _ROOT / "models/piper"
_PIPER_EN = _PIPER_DIR / "en_US-lessac-medium.onnx"

_CJK = re.compile(r"[㐀-鿿豈-﫿]")
_HAS_DIGIT_OR_CJK = re.compile(r"[㐀-鿿豈-﫿0-9]")
# EN run = ต้องมี "คำละติน" จริง (อักษร A-Z + ช่องว่าง/ขีด/apostrophe ภายใน) เท่านั้น
# เครื่องหมาย /（）： และตัวเลข "ไม่ใช่" เหตุให้สลับไปเสียง EN — ไม่งั้นประโยคจีนที่มี
# วันที่/สแลชโดนหั่นเป็นชิ้นๆ สลับสำเนียงไปมา (feedback จากเทสต์จริง 11 ก.ค.)
# ตัวเลขในบริบทจีนให้เสียงจีนอ่าน (9月 → 九月 ถูกต้อง)
_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z'\- ]*[A-Za-z]|[A-Za-z]")


def _segments(text):
    """แยกข้อความเป็น [(ท่อน, is_zh)] — สลับภาษาเฉพาะเมื่อเจอคำละตินจริง"""
    text = str(text)
    if not _CJK.search(text):
        return [(text.strip(), False)] if text.strip() else []
    out = []
    pieces = _LATIN_RUN.split(text)              # ชิ้นระหว่างคำละติน
    matches = _LATIN_RUN.findall(text)           # คำละตินที่เจอ
    for i, piece in enumerate(pieces):
        piece = piece.strip()
        if piece and _HAS_DIGIT_OR_CJK.search(piece):    # ข้ามเศษ punctuation ล้วน
            out.append((piece, True))
        if i < len(matches):
            m = matches[i].strip()
            if m:
                out.append((m, False))
    return out


# Windows 語音名稱／ID 裡出現這些字樣就視為台灣中文語音
_TW_HINTS = ("zh-tw", "taiwan", "hanhan", "yating", "zhiwei", "hsiaochen", "yunjhe", "_404", "tts_ms_zh-tw")
_EN_HINTS = ("english", "en-us", "zira", "david")


class Speaker:
    """把文字唸出喇叭（離線；依片段語言自動選語音）"""

    def __init__(self, rate: int = 170, lang_hint: str = "zh"):
        self._piper_en = None
        self.engine = None
        self.voice_en = self.voice_zh = None

        # 1) Piper：只用英文語音
        try:
            from piper import PiperVoice
            if _PIPER_EN.exists():
                self._piper_en = PiperVoice.load(str(_PIPER_EN))
        except Exception:
            pass

        # 2) Windows SAPI：台灣中文語音（必要）＋英文語音（Piper 不在時的備援）
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", rate)
            for v in self.engine.getProperty("voices"):
                meta = f"{v.name} {v.id}".lower()
                if self.voice_zh is None and any(h in meta for h in _TW_HINTS):
                    self.voice_zh = v.id
                if self.voice_en is None and any(h in meta for h in _EN_HINTS):
                    self.voice_en = v.id
            if self.voice_zh is None:
                print("[TTS] 找不到台灣中文語音，中文會只顯示不發音。"
                      "請到 Windows 設定 → 時間與語言 → 語音 新增「中文（台灣）」。")
        except Exception:
            self.engine = None

        if self._piper_en is not None and self.engine is not None:
            self.mode = "piper+sapi"
        elif self.engine is not None:
            self.mode = "sapi"
        elif self._piper_en is not None:
            self.mode = "piper"
        else:
            self.mode = "print"

    # ---- 英文：Piper ------------------------------------------------------
    def _piper_say(self, part: str) -> None:
        import winsound
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            with wave.open(tmp, "wb") as w:
                self._piper_en.synthesize_wav(part, w)
            winsound.PlaySound(tmp, winsound.SND_FILENAME)   # 會等到唸完才返回
        finally:
            try:
                Path(tmp).unlink()
            except OSError:
                pass

    # ---- 中文與備援英文：Windows SAPI ---------------------------------------
    def _sapi_say(self, part: str, voice_id) -> None:
        self.engine.setProperty("voice", voice_id)
        self.engine.say(part)
        self.engine.runAndWait()

    def _say_part(self, part: str, zh: bool) -> None:
        if zh:
            if self.engine is not None and self.voice_zh:
                self._sapi_say(part, self.voice_zh)
            else:
                print(f"[TTS-fallback] {part}")
            return
        if self._piper_en is not None:
            try:
                self._piper_say(part)
                return
            except Exception:
                pass                                  # Piper 出錯 → 改用 SAPI
        if self.engine is not None and self.voice_en:
            self._sapi_say(part, self.voice_en)
        else:
            print(f"[TTS-fallback] {part}")

    def say(self, text: str) -> None:
        if not text:
            return
        for part, zh in _segments(text):
            self._say_part(part, zh)

    # 讓原本檢查 .enabled 的程式繼續可用
    @property
    def enabled(self) -> bool:
        return self.mode != "print"
