"""
gemma_npu.py — backend คุณภาพสูง (non-realtime): Gemma-3-4b บน NPU ผ่าน OGA
===========================================================================
สำหรับโหมด indoor ที่รับ latency สูงได้ (ดู Phase 1 Table 5).
⚠️ ช้า: TTFT ~17s, decode ~4.6 tok/s — prefill-bound (§5.6) ไม่เหมาะ real-time

env: ryzen-ai-1.7.1 (OGA + RyzenAI EP) — คนละ env กับ smolvlm!
    conda activate ryzen-ai-1.7.1
    python -m app.assistant --mode indoor   (ตั้ง BACKEND="gemma_npu" ใน config.py)

logic port มาจาก code/bench_gemma_npu.py (ที่ทดสอบรันจริงแล้ว 17 รูป)
"""

import os
import tempfile

from app.base import VLMBackend


class GemmaNPUBackend(VLMBackend):
    def __init__(self, model_dir="models/gemma3_4b_npu", gen_params=None):
        # ⚠️ ต้อง absolute path ไม่งั้น OGA ต่อ path ซ้ำ (ดู bench_gemma_npu.py)
        self.model_dir = os.path.abspath(model_dir)
        self.gen_params = gen_params or {}
        self.name = "gemma-3-4b@npu"
        self.og = None
        self.model = None
        self.processor = None
        self.tokenizer_stream = None

    def load(self) -> None:
        # import ในนี้ (lazy) เพราะ onnxruntime_genai มีแค่ใน env ryzen-ai-1.7.1
        import onnxruntime_genai as og
        self.og = og
        self.model = og.Model(self.model_dir)
        self.processor = self.model.create_multimodal_processor()
        self.tokenizer_stream = self.processor.create_stream()

    @staticmethod
    def _build_prompt(user_text: str) -> str:
        """chat template ของ Gemma-3 (<start_of_image> = ตำแหน่งรูป)"""
        return (
            "<start_of_turn>user\n"
            "<start_of_image>"
            f"{user_text}"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

    def describe(self, image, prompt: str, max_new_tokens: int | None = None) -> str:
        og = self.og
        max_new = max_new_tokens or self.gen_params.get("max_new_tokens", 64)

        # OGA Images.open รับ path → เซฟ PIL image ลง temp file ชั่วคราว
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            image.save(tmp_path)
            images = og.Images.open(tmp_path)
            inputs = self.processor(self._build_prompt(prompt), images=images)

            params = og.GeneratorParams(self.model)
            params.set_search_options(do_sample=False, max_length=16384)
            generator = og.Generator(self.model, params)
            # ⚠️ prefill เกิดที่ set_inputs (ดู §5.6) ไม่ใช่ generate_next_token
            generator.set_inputs(inputs)

            out = []
            n = 0
            while not generator.is_done() and n < max_new:
                generator.generate_next_token()
                tok = generator.get_next_tokens()[0]
                out.append(self.tokenizer_stream.decode(tok))
                n += 1
            # clean: ▁ = SentencePiece space marker
            return " ".join("".join(out).replace("▁", " ").split())
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
