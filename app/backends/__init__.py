"""
backends — 集中管理所有 VLM 後端，並提供選擇後端的工廠函式
"""

from app.base import VLMBackend


def get_backend(name: str, **kwargs) -> VLMBackend:
    """工廠函式：依名稱回傳後端物件（延遲 import，避免載入用不到的後端套件
    — 這很重要，因為 smolvlm 和 gemma 分別裝在不同的環境）
    """
    if name == "smolvlm":
        from app.backends.smolvlm import SmolVLMBackend
        return SmolVLMBackend(**kwargs)
    if name == "gemma_npu":
        from app.backends.gemma_npu import GemmaNPUBackend
        return GemmaNPUBackend(**kwargs)
    if name == "gemma_hf":
        from app.backends.gemma_hf import GemmaHFBackend
        return GemmaHFBackend(**kwargs)
    raise ValueError(f"不認得的後端：{name!r}（可用：smolvlm、gemma_npu、gemma_hf）")
