"""
smolvlm.py — backend หลัก (real-time): SmolVLM บน CPU/iGPU ผ่าน transformers
============================================================================
ใช้ best config จาก Phase 1 (le=384, max_new_tokens=50, rep_penalty=1.2).
logic การโหลด/inference ยกมาจาก code/test_smolvlm.py (ที่ benchmark แล้ว)

env:
  - DEVICE="cpu"  → env vlm_research (transformers 5.x, ใช้ dtype=)
  - DEVICE="igpu" → env vlm_dml (transformers 4.49, torch-directml; ใช้ torch_dtype=)
    ⚠️ iGPU เร็วกว่าแต่ RAM ~6.6GB (ดู Phase 1 §5.2) ต้องเครื่อง ≥16GB
"""

import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor

from app.base import VLMBackend

# transformers 5.x ใช้ dtype= / 4.49 (env vlm_dml) ใช้ torch_dtype= (deprecated ใน 5.x)
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
        # คุม resolution ตาม best config (le=384 = sweet spot ของ model จิ๋ว)
        le = self.gen_params.get("longest_edge", 384)
        self.processor.image_processor.size = {"longest_edge": le}
        self.processor.image_processor.do_image_splitting = True

        # เลือก kwarg ตามเวอร์ชัน transformers (dtype 5.x / torch_dtype 4.49)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, **{_DTYPE_KW: torch.float32}
        )

        if self.device == "igpu":
            # iGPU path ต้องรันใน env vlm_dml (มี torch_directml)
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
        # ตัด prompt tokens ออก เหลือแต่คำตอบ
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
