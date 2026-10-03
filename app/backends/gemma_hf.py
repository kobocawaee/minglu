"""
gemma_hf.py — [資服版新增] Gemma 3 透過 transformers 跑在 NVIDIA 顯示卡（CUDA）上
==============================================================================
為什麼需要：SmolVLM 只會輸出英文；原本的 gemma_npu 後端只能跑在 AMD Ryzen AI 的 NPU 上。
這個後端讓 Gemma（會繁體中文）跑在一般的 NVIDIA 顯示卡上，沒有顯示卡時退回 CPU（會很慢）。

模型：google/gemma-3-4b-it（Google，Gemma 使用條款）。
第一次使用前要：
  1. 在 Hugging Face 的模型頁面登入並同意 Gemma 的使用條款
  2. 在這台電腦執行 `huggingface-cli login` 貼上存取權杖
之後模型會快取在本機，執行時不需要連網。
"""

import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor

from app.base import VLMBackend

_DTYPE_KW = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"

SYSTEM_ZH = ("你是視障者的視覺助理。一律使用臺灣的繁體中文回答。"
             "回答要簡短、直接，只說你清楚看到的東西，不要猜測，不要使用條列或符號。")


class GemmaHFBackend(VLMBackend):
    def __init__(self, model_id, gen_params=None, system_prompt=SYSTEM_ZH):
        self.model_id = model_id
        self.gen_params = gen_params or {}
        self.system_prompt = system_prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = f"{model_id.split('/')[-1]}@{self.device}"
        self.model = None
        self.processor = None

    def load(self) -> None:
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        from app import config
        if self.device == "cuda" and getattr(config, "GEMMA_4BIT", False):
            # 4-bit（bitsandbytes NF4）：8GB 顯示卡放得下
            from transformers import BitsAndBytesConfig
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16),
                device_map="cuda", **{_DTYPE_KW: dtype},
            )
            self.name += "-4bit"
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, **{_DTYPE_KW: dtype}
            ).to(self.device)
        self.model.eval()

    def describe(self, image, prompt: str, max_new_tokens: int | None = None,
                 system_prompt: str | None = None, history=None) -> str:
        """system_prompt：換掉預設的系統提示（語音對話用）。
        history：[(問, 答), ...] 前幾輪的文字對話，放在這次（帶圖片）的提問之前。"""
        if not hasattr(image, "convert"):            # OpenCV 的 BGR 陣列 → PIL
            from PIL import Image
            image = Image.fromarray(image[:, :, ::-1])
        messages = []
        system = system_prompt or self.system_prompt
        if system:
            messages.append({"role": "system",
                             "content": [{"type": "text", "text": system}]})
        for q, a in history or []:
            messages.append({"role": "user", "content": [{"type": "text", "text": q}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": a}]})
        messages.append({"role": "user",
                         "content": [{"type": "image", "image": image.convert("RGB")},
                                     {"type": "text", "text": prompt}]})
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)
        if self.device == "cuda" and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        max_new = max_new_tokens or self.gen_params.get("max_new_tokens", 60)
        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                repetition_penalty=self.gen_params.get("repetition_penalty", 1.05),
                do_sample=False,
            )
        gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    def warmup(self) -> None:
        from PIL import Image
        try:
            self.describe(Image.new("RGB", (384, 384), (127, 127, 127)), "這是什麼？", max_new_tokens=8)
        except Exception as e:
            print(f"[warn] Gemma warmup 失敗：{e}")
