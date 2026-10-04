"""
gemma_npu.py — 高品質後端（非即時）：在 NPU 上透過 OGA 執行 Gemma-3-4b
===========================================================================
用於可以接受較高延遲的室內模式（見 Phase 1 Table 5）。
⚠️ 很慢：首字延遲約 17 秒、產生速度約 4.6 tok/s — 卡在預填（prefill-bound，§5.6），不適合即時使用

環境：ryzen-ai-1.7.1（OGA + RyzenAI EP）— 和 smolvlm 不同環境！
    conda activate ryzen-ai-1.7.1
    python -m app.assistant --mode indoor   （在 config.py 設定 BACKEND="gemma_npu"）

邏輯移植自 code/bench_gemma_npu.py（已實際跑過 17 張圖）
"""

import os
import tempfile

from app.base import VLMBackend


class GemmaNPUBackend(VLMBackend):
    def __init__(self, model_dir="models/gemma3_4b_npu", gen_params=None):
        # ⚠️ 必須用絕對路徑，否則 OGA 會把路徑重複接上（見 bench_gemma_npu.py）
        self.model_dir = os.path.abspath(model_dir)
        self.gen_params = gen_params or {}
        self.name = "gemma-3-4b@npu"
        self.og = None
        self.model = None
        self.processor = None
        self.tokenizer_stream = None

    def load(self) -> None:
        # 在這裡才 import（延遲載入），因為 onnxruntime_genai 只裝在 ryzen-ai-1.7.1 環境
        import onnxruntime_genai as og
        self.og = og
        self.model = og.Model(self.model_dir)
        self.processor = self.model.create_multimodal_processor()
        self.tokenizer_stream = self.processor.create_stream()

    @staticmethod
    def _build_prompt(user_text: str) -> str:
        """Gemma-3 的對話模板（<start_of_image> = 圖片的位置）"""
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

        # OGA 的 Images.open 只吃路徑 → 先把 PIL 影像存成暫存檔
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
            # ⚠️ 預填發生在 set_inputs（見 §5.6），不是 generate_next_token
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
