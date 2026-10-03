"""
voice.py — [資服版新增] 語音指令與語音提問
==========================================
手機錄音（16kHz 單聲道 PCM）→ 筆電上的 Whisper 轉成文字 → 判斷是「指令」還是「問題」：
  指令：切換模式（「切換到過馬路模式」）、開始／停止連續、再說一次、說明
  問題：
    問號誌、能不能過馬路 → 走過馬路模式原本的流程（LYTNetV2 + YOLO），不讓 VLM 判斷能不能過
    問上面寫什麼         → 走讀字模式（EasyOCR）
    其他問題             → 把問題連同畫面交給 Gemma 回答（視覺問答）

語音辨識也在筆電上跑（Whisper），聲音不會送到第三方雲端，和影像一樣維持離線。
模型：openai/whisper-large-v3-turbo（MIT 授權），第一次執行時下載約 1.6GB。
"""

import re

import numpy as np

from app import config, postprocess
from app.messages import mode_name

_ASR = None
_CC = None          # 簡體 → 臺灣繁體（Whisper 中文常輸出簡體）

# Whisper 在幾乎沒聲音時常見的幻覺字幕
_HALLU_RE = re.compile(r"字幕|訂閱|點贊|點讚|按讚|謝謝觀看|謝謝收看|感謝觀看|请不吝|明鏡|Amara")
_PUNCT_RE = re.compile(r"[\s，。、！？!?,.：:；;「」『』（）()…~～-]+")


def load_asr():
    """載入 Whisper（伺服器啟動時呼叫一次）。"""
    global _ASR, _CC
    import torch
    import transformers
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32
    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    # 不用 transformers 的 pipeline：它會忽略 language="zh"，短句常被誤判成別的語言（實測「說快一點」變冰島文）
    processor = AutoProcessor.from_pretrained(config.ASR_MODEL)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(config.ASR_MODEL, **{dtype_kw: dtype})
    model = model.to("cuda" if cuda else "cpu").eval()
    model.generation_config.forced_decoder_ids = None
    _ASR = (processor, model, dtype)
    try:
        from opencc import OpenCC
        _CC = OpenCC("s2twp")
    except Exception:
        _CC = None
    transcribe(np.zeros(16000, dtype=np.float32))     # warmup


def transcribe(pcm: np.ndarray) -> str:
    """16kHz float32 單聲道 → 繁體中文文字（固定用中文辨識）。聽不出內容時回傳空字串。"""
    import torch
    processor, model, dtype = _ASR
    feats = processor(pcm, sampling_rate=16000, return_tensors="pt").input_features
    feats = feats.to(model.device, dtype)
    with torch.inference_mode():
        ids = model.generate(feats, language="zh", task="transcribe", max_new_tokens=120)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    if _CC:
        text = _CC.convert(text)
    if _HALLU_RE.search(text):
        return ""
    return text


# ---------------------------------------------------------------------------
# 指令判斷
# ---------------------------------------------------------------------------
# 每個模式的說法（簡繁都收，以防轉換漏掉）
_MODE_WORDS = {
    "street":      ["過馬路", "过马路", "馬路", "马路", "紅綠燈", "红绿灯", "號誌", "斑馬線", "路口"],
    "read":        ["讀字", "读字", "讀", "唸", "念", "文字"],
    "object":      ["辨識物品", "物品", "東西", "辨識"],
    "indoor":      ["室內", "室内", "房間", "屋內"],
    "surrounding": ["周遭", "周圍", "周围", "四周", "環境", "环境"],
    "auto":        ["自動", "自动"],
}
_SWITCH_VERB_RE = re.compile(r"(切換|切换|換成|换成|換到|换到|改成|改用|切到|轉到|转到|進入|进入|轉成|转成)(到|成)?")
_WHICH_MODE_RE = re.compile(r"(什麼|什么|哪個|哪个|目前|現在|现在)(的)?模式")
_FILLER_RE = re.compile(r"^(請|请|幫我|帮我|麻煩|一下|的)+|(的|一下|吧|好了|好嗎|好吗)+$")
_CONT_ON_RE = re.compile(r"(開始|開啟|打開|開)(連續|持續)|(連續|持續)(模式)?(開|開始|開啟)")
_CONT_OFF_RE = re.compile(r"(停止|關閉|關掉|關|結束|暫停)(連續|持續)|(連續|持續)(模式)?(關|停|關閉|停止)")
_REPEAT_RE = re.compile(r"再說一次|再说一次|重複|重复|再講一次|剛剛說什麼|剛才說什麼")
# 個人化設定（實際調整在手機上做，這裡只判斷是哪個設定、往哪邊調）
#   短句容易聽成同音字：「字」→「自、子」，「語速」→「雨速、與速」，所以比對時把同音字也算進去
_Z = "[字自子紫]"                                   # 字
_YS = "(?:[語语雨與与於于魚鱼予宇羽][速素宿])"         # 語速（「速度」只在「速度調到快」這種明確說法才算，免得問「車速度快嗎」被當成調語速）
_SAY = "(?:說|说|講|讲|唸|念|講話|说话|說話)"
_SET_VERB = "(?:調|调|設|设|改|變|变|換|换)(?:到|成|為|为)?"
_RATE_VALUES = {"很快": 3, "最快": 3, "快": 2, "標準": 1, "标准": 1, "正常": 1, "普通": 1, "一般": 1,
                "預設": 1, "预设": 1, "慢": 0, "最慢": 0}
_FONT_VALUES = {"特大": 2, "最大": 2, "大": 1, "標準": 0, "标准": 0, "正常": 0, "一般": 0, "預設": 0,
                "预设": 0, "小": 0, "最小": 0}
_RATE_SET_RE = re.compile("(?:" + _YS + "|速度)(?:也)?" + _SET_VERB + "(很快|最快|快|標準|标准|正常|普通|一般|預設|预设|最慢|慢)")
_FONT_SET_RE = re.compile(_Z + "(?:體|体|型)?(?:的)?(?:大小)?" + _SET_VERB + "(特大|最大|大|標準|标准|正常|一般|預設|预设|最小|小)")
_SETTING_WORDS_RE = re.compile(_YS + "|" + _Z + "(?:體|体|型)|設定|设定|震動|震动|搖一搖|摇一摇")

_SETTING_RULES = [
    (re.compile(r"(恢復|還原|还原|重設|重设)(成|到)?(預設|预设|原本|原來|原来)|重設設定|重设设定"), ("reset", None)),
    (re.compile("(?:" + _SAY + "|" + _YS + ")(?:話)?(?:再|可以)?(?:快|加快)(?:一點|一点|一些|點|点)?"
                "|" + _YS + "(?:調|调)?(?:快|加快)|^(?:再)?快(?:一點|一点|一些)$"), ("rate", "up")),
    (re.compile("(?:" + _SAY + "|" + _YS + ")(?:話)?(?:再|可以)?(?:慢|放慢)(?:一點|一点|一些|點|点)?"
                "|" + _YS + "(?:調|调)?(?:慢|放慢)|^(?:再)?慢(?:一點|一点|一些)$"), ("rate", "down")),
    (re.compile(_Z + "(?:體|体|型)?(?:再|可以)?(?:大|放大)(?:一點|一点|一些|點|点)|(?:放大|加大)" + _Z
                + "|" + _Z + "(?:體|体|型)(?:再)?(?:大|放大)"), ("font", "up")),
    (re.compile(_Z + "(?:體|体|型)?(?:再|可以)?(?:小|縮小|缩小)(?:一點|一点|一些|點|点)|縮小" + _Z + "|缩小" + _Z
                + "|" + _Z + "(?:體|体|型)(?:再)?(?:小|縮小|缩小)"), ("font", "down")),
    (re.compile(r"(關掉|关掉|關閉|关闭|關|关|停止|取消|不要)(震動|振動|震动|振动)|(震動|振動|震动|振动)(關掉|关掉|關閉|关闭|關|关)"), ("vib", False)),
    (re.compile(r"(打開|打开|開啟|开启|開|开)(震動|振動|震动|振动)|(震動|振動|震动|振动)(打開|打开|開啟|开启|開|开)"), ("vib", True)),
    (re.compile(r"(關掉|关掉|關閉|关闭|關|关|停止|取消|不要)(搖一搖|摇一摇|搖動|摇动|搖晃|摇晃)|(搖一搖|摇一摇)(關掉|关掉|關閉|关闭|關|关)"), ("shake", False)),
    (re.compile(r"(打開|打开|開啟|开启|開|开)(搖一搖|摇一摇|搖動|摇动|搖晃|摇晃)|(搖一搖|摇一摇)(打開|打开|開啟|开启|開|开)"), ("shake", True)),
]

_HELP_RE = re.compile(r"怎麼用|怎么用|有哪些(模式|功能)|說明|幫助|帮助|你可以做什麼|可以說什麼")

# 問題要交給哪個模式的固定流程
_ASK_STREET_RE = re.compile(r"過馬路|过马路|能不能過|可不可以過|可以過|能過|紅燈|綠燈|红灯|绿灯|燈號|"
                            r"號誌|紅綠燈|红绿灯|斑馬線|有車|有没有车|有沒有車|車子|車輛")
_ASK_READ_RE = re.compile(r"寫什麼|寫了什麼|寫著|寫的是|写什么|上面寫|什麼字|哪些字|"
                          r"(唸|念|讀|读)(給我|一下|出來)|幫我(唸|念|讀|读)")

HELP_TEXT = ("你可以說：切換到過馬路模式、讀字模式、物品模式、室內模式、周遭模式或自動模式；"
             "也可以直接問問題，例如：現在是紅燈嗎、前面有什麼、這上面寫什麼。"
             "說開始連續或停止連續，可以開關連續描述。"
             "也可以說：說快一點、說慢一點、字大一點、字小一點、恢復預設設定。"
             "不想按按鈕的話，搖一搖手機也可以開始提問。")


def _find_mode(s: str):
    """找出句子裡最早出現的模式說法。"""
    best = None
    for mode, words in _MODE_WORDS.items():
        for w in words:
            i = s.find(w)
            if i >= 0 and (best is None or i < best[0]):
                best = (i, mode)
    return best[1] if best else None


def _pinyin(s: str) -> list:
    from pypinyin import lazy_pinyin
    return [p for p in lazy_pinyin(s) if p.strip()]


def _sound_mode(seg: str, min_score: float):
    """用發音找最像的模式：Whisper 常把「讀字」聽成同音的「獨自」「度字」。"""
    from difflib import SequenceMatcher
    seg_py = _pinyin(seg)
    if not seg_py:
        return None
    joined = " " + " ".join(seg_py) + " "
    best, best_score = None, 0.0
    for mode, words in _MODE_WORDS.items():
        for w in words:
            w_py = _pinyin(w)
            if " " + " ".join(w_py) + " " in joined:          # 發音完全包含在句子裡
                score = 1.0
            else:
                score = SequenceMatcher(None, seg_py, w_py).ratio()
            if score > best_score:
                best, best_score = mode, score
    return best if best_score >= min_score else None


def parse_command(text: str):
    """回傳 ('switch', mode) / ('switch_unknown', 聽到的詞) / ('which_mode', None) /
    ('continuous', bool) / ('repeat', None) / ('help', None)；不是指令回傳 None。"""
    s = _PUNCT_RE.sub("", text)
    if not s:
        return None
    if _CONT_OFF_RE.search(s):
        return ("continuous", False)
    if _CONT_ON_RE.search(s):
        return ("continuous", True)
    m = _RATE_SET_RE.search(s)                     # 「把語速調到快」
    if m:
        return ("setting", ("rate", _RATE_VALUES[m.group(1)]))
    m = _FONT_SET_RE.search(s)                     # 「字體調成特大」
    if m:
        return ("setting", ("font", _FONT_VALUES[m.group(1)]))
    for rx, setting in _SETTING_RULES:
        if rx.search(s):
            return ("setting", setting)
    if _SETTING_WORDS_RE.search(s) and not _ASK_READ_RE.search(s):
        return ("setting_unknown", None)           # 有提到設定但聽不出要怎麼調 → 不要當成問題去描述畫面
    if _REPEAT_RE.search(s):
        return ("repeat", None)
    if _HELP_RE.search(s):
        return ("help", None)
    if _WHICH_MODE_RE.search(s) and not _SWITCH_VERB_RE.search(s):
        return ("which_mode", None)

    # 有「切換到…」或以「模式」結尾 → 一定是切換指令，只需要判斷是哪個模式
    m = _SWITCH_VERB_RE.search(s)
    if m or s.endswith("模式"):
        seg = s[m.end():] if m else s
        seg = _FILLER_RE.sub("", seg.replace("模式", ""))
        mode = _find_mode(seg) or _sound_mode(seg, 0.5)
        return ("switch", mode) if mode else ("switch_unknown", seg)

    # 只說模式名稱的短句（例如「讀字」）→ 切換；發音比對要幾乎一樣才算，避免短問題被誤判
    if len(s) <= 4:
        mode = _find_mode(s) or _sound_mode(s, 0.99)
        if mode:
            return ("switch", mode)
    return None


# ---------------------------------------------------------------------------
# 回答問題
# ---------------------------------------------------------------------------
def question_mode(question: str) -> str:
    """問題該走哪個模式的流程（vqa = 交給 Gemma 自由回答）。"""
    if _ASK_STREET_RE.search(question):
        return "street"
    if _ASK_READ_RE.search(question):
        return "read"
    return "vqa"


# ---------------------------------------------------------------------------
# 對話歷史：追問（「那顏色呢？」「它旁邊有什麼？」）才沿用前面的問答，換話題就清掉
#   用規則判斷而不是再問一次 Gemma：快、結果可預期，判斷錯了頂多是少了上下文
# ---------------------------------------------------------------------------
_FOLLOWUP_RE = re.compile(
    r"^(那|那麼|那么|還有|还有|另外|然後|然后|所以|再|它|他|她|那個|那个|那些|剛剛|刚刚|剛才|刚才)"
    r"|呢$|它|那個|那个|那些|剛剛|刚刚|剛才|刚才|你說的|你说的|上一個|上一个|前一個|"
    r"第[一二三四五1-5]個|左邊那|右邊那|中間那|旁邊那|其他的|其它的|還有別的|詳細|仔細|更多")


def is_followup(question: str) -> bool:
    """新的語音問題是不是在追問上一題。"""
    s = _PUNCT_RE.sub("", question)
    return bool(s) and bool(_FOLLOWUP_RE.search(s))


# 語音對話專用的系統提示：和描述模式不同，要「回答問題」而不是「描述畫面」
SYSTEM_ASK = (
    "你是視障者的視覺助理，正在用語音和使用者對話。一律使用臺灣的繁體中文。"
    "回答規則："
    "一、先直接回答使用者問的問題；是非題第一個詞就回答「有」、「沒有」、「是」或「不是」。"
    "二、只回答被問到的事，不要另外描述畫面裡其他東西，除非使用者要求。"
    "三、用一到兩句口語短句，像在跟對方說話，稱呼使用者為「你」，不要用條列或符號。"
    "四、只根據畫面中清楚看到的內容回答；看不清楚就說不確定，不要猜。"
    "五、使用者問「有沒有某樣東西」時，畫面裡沒有清楚看到就回答「沒有看到」，"
    "不要因為使用者問了就說有。答錯「有」比答「沒有看到」更危險。"
    "六、不要判斷現在能不能安全過馬路。"
)


def answer(backend, image, question: str, current_mode: str, history=None):
    """回答語音問題。history 是要沿用的前幾輪問答（已判斷過是追問）。回傳 (text, target)。"""
    from app import pipeline

    target = question_mode(question)
    # 追問上一題的號誌／讀字（例如「那現在呢？」）→ 再跑一次同樣的固定流程
    if target == "vqa" and history and history[-1].get("t") in ("street", "read"):
        target = history[-1]["t"]
    if target in ("street", "read"):
        text, _ = pipeline.describe(backend, image, target)
        return text, target

    # SmolVLM 看不懂中文問題 → 退回目前模式的描述
    if not config._use_zh():
        mode = current_mode if current_mode != "auto" else "surrounding"
        text, _ = pipeline.describe(backend, image, mode)
        return text, "vqa"

    turns = [(h.get("q", ""), h.get("a", "")) for h in (history or [])[-config.ASK_HISTORY_TURNS:]]
    prompt = question
    if turns:
        prompt += "\n（鏡頭可能已經移動，以現在這張畫面為準。）"
    kwargs = {"system_prompt": SYSTEM_ASK, "history": turns} if config.BACKEND == "gemma_hf" else {}
    text = postprocess.clean(backend.describe(image, prompt, max_new_tokens=config.ASK_MAX_TOKENS, **kwargs))

    # 問到人時，和描述模式一樣用 YOLO 再確認一次（漏掉人比較危險）
    if "人" in question and not pipeline._PERSON_RE.search(text):
        if pipeline._person_clearly_present(image):
            from app.messages import t
            text = (text + " " + t("person_ahead")).strip()
    return text, "vqa"


def command_reply(cmd, arg, current_mode="auto") -> str:
    if cmd == "switch":
        return f"已切換到{mode_name(arg)}模式。"
    if cmd == "switch_unknown":
        return "沒聽清楚要切換到哪個模式。可以說：過馬路、周遭、室內、讀字、物品或自動。"
    if cmd == "setting_unknown":
        return "沒聽懂要怎麼調整。可以說：說快一點、說慢一點、字大一點、字小一點，或把語速調到快。"
    if cmd == "which_mode":
        return f"目前是{mode_name(current_mode)}模式。"
    if cmd == "continuous":
        return "連續模式已開啟。" if arg else "連續模式已關閉。"
    if cmd == "help":
        return HELP_TEXT
    return ""
