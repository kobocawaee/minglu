"""
smolvlm.py — 主要後端（即時）：在 CPU／iGPU 上透過 transformers 執行 SmolVLM
============================================================================
採用 Phase 1 的最佳設定（le=384、max_new_tokens=50、rep_penalty=1.2）。
載入與推論邏輯取自 code/test_smolvlm.py（已做過效能測試）

env:
  - DEVICE="cpu"  → 環境 vlm_research（transformers 5.x，使用 dtype=）
  - DEVICE="igpu" → 環境 vlm_dml（transformers 4.49、torch-directml；使用 torch_dtype=）
    ⚠️ iGPU 較快但佔用約 6.6GB 記憶體（見 Phase 1 §5.2），電腦需 16GB 以上
"""

import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor

from app.base import VLMBackend

# transformers 5.x 用 dtype= ／ 4.49（環境 vlm_dml）用 torch_dtype=（5.x 已不建議使用）
_DTYPE_KW = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"


class SmolVLMBackend(VLMBackend):
    def __init__(self, model_id, device="cpu", gen_params=None):
        self.model_id = model_id
        self.device = device
        self.gen_params = gen_params or {}
        self.name = f"{model_id.split('/')[-1]}@{device}"
        self.model = None
        self.processor = None
        self._torch_device = "cpu"

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # 依最佳設定控制解析度（le=384 是小模型的最佳點）
        le = self.gen_params.get("longest_edge", 384)
        self.processor.image_processor.size = {"longest_edge": le}
        self.processor.image_processor.do_image_splitting = True

        # 依 transformers 版本選參數名稱（5.x 用 dtype／4.49 用 torch_dtype）
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, **{_DTYPE_KW: torch.float32}
        )

        if self.device == "igpu":
            # iGPU 路徑要在環境 vlm_dml 中執行（需有 torch_directml）
            import torch_directml
            self._torch_device = torch_directml.device()
            self.model = self.model.to(self._torch_device)
        else:
            self._torch_device = "cpu"
        self.model.eval()

    def describe(self, image, prompt: str, max_new_tokens: int | None = None) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=chat, images=[image], return_tensors="pt")
        inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}

        max_new = max_new_tokens or self.gen_params.get("max_new_tokens", 50)
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                repetition_penalty=self.gen_params.get("repetition_penalty", 1.2),
                no_repeat_ngram_size=self.gen_params.get("no_repeat_ngram_size", 3),
                do_sample=False,
            )
        # 去掉提示詞的 token，只留下回答
        gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return text.strip()

    def warmup(self) -> None:
        from PIL import Image
        dummy = Image.new("RGB", (384, 384), (127, 127, 127))
        try:
            self.describe(dummy, "test")
        except Exception:
            pass
