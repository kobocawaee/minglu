"""
backends — รวม VLM backend + factory เลือกตัวที่จะใช้
"""

from app.base import VLMBackend


def get_backend(name: str, **kwargs) -> VLMBackend:
    """factory: คืน backend instance ตามชื่อ (lazy import เพื่อไม่ดึง dep
    ของ backend ที่ไม่ได้ใช้ — สำคัญเพราะ smolvlm/gemma อยู่คนละ env)
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
    raise ValueError(f"ไม่รู้จัก backend: {name!r} (มี: smolvlm, gemma_npu)")
