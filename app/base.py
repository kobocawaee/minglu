"""
base.py — VLM 後端的共用介面
=======================================
每個後端（CPU／iGPU 上的 SmolVLM、NPU 上的 Gemma）都要實作這個類別
→ 應用邏輯（assistant.py）只透過這一個介面呼叫，不需要知道裡面
  用的是 transformers 還是 OGA，或跑在哪個裝置上

為什麼需要介面：Phase 1 發現各後端裝在不同的 conda 環境
（transformers 和 OGA），無法在同一個程序裡同時執行。所以程式在啟動時
只選一個後端，之後都透過這層抽象來使用
"""

from abc import ABC, abstractmethod


class VLMBackend(ABC):
    """每個 VLM 後端都必須提供的介面"""

    #: 簡短名稱，用於紀錄與顯示，例如 "smolvlm-500m@cpu"
    name: str = "vlm-backend"

    @abstractmethod
    def load(self) -> None:
        """把模型載入記憶體（程式啟動時呼叫一次）。
        和 __init__ 分開，因為載入很慢（好幾秒），想自己控制時機
        """
        raise NotImplementedError

    @abstractmethod
    def describe(self, image, prompt: str, max_new_tokens: int | None = None) -> str:
        """輸入影像（PIL.Image 或 OpenCV 的 numpy BGR）＋提示詞 → 回傳描述文字（str）
        max_new_tokens：覆寫各模式的輸出長度上限（None = 用後端的預設值）。
        依 §5.6 的發現，延遲主要卡在逐字產生（decode-bound）→ 輸出越短越快（street 用最短）
        前處理與格式由各後端自行處理
        """
        raise NotImplementedError

    def warmup(self) -> None:
        """（可選）先空跑一次推論，讓 JIT／編譯完成，
        降低第一次使用的延遲。預設不做任何事
        """
        pass
