# 資服參賽版：改了什麼、怎麼跑

這份是原 `code_package` 的修改版。原始檔案沒有被覆蓋，仍在 `code_package/`。

## 改了什麼

| 檔案 | 修改 |
|---|---|
| `app/ocr.py` | RapidOCR → EasyOCR（繁體中文 + 英文） |
| `app/speak.py` | 拿掉 Piper `zh_CN-huayan`；中文改用 Windows 內建台灣中文語音 |
| `app/messages.py`（新增） | 所有固定句子的中英對照表 |
| `app/config.py` | 新增 `LANG`、`OCR_LANGS`、`OCR_MIN_CONF` |
| `app/quality.py`、`crossing.py`、`light_classifier.py`、`pipeline.py`、`assistant.py` | 寫死的英文句子改成查對照表 |
| `app/server.py` | 手機頁面改繁中，語音優先選台灣中文，中文警示詞也會觸發強震動 |
| `code/score_field_clips.py` | 加一行固定用英文輸出（它靠比對英文句子計分） |

沒有動：LYTNetV2、YOLOv8n、SmolVLM、Gemma、所有門檻與判斷邏輯。

`app/config.py` 的 `LANG = "en"` 可以切回和論文完全相同的英文句子。

## 全中文輸出：Gemma 後端

周遭、室內、物品三個模式的描述原本由 SmolVLM 產生，它只會英文。
現在預設後端改成 `gemma_hf`：Gemma-3-4b 透過 transformers 跑在 NVIDIA 顯示卡上，用繁體中文提示詞，輸出繁體中文。

| 新增／修改 | 內容 |
|---|---|
| `app/backends/gemma_hf.py`（新增） | Gemma 的顯示卡後端 |
| `app/config.py` | `BACKEND = "gemma_hf"`、`GEMMA_MODEL`、繁中提示詞 `MODE_PROMPTS_ZH` |
| `app/postprocess.py`、`app/pipeline.py` | 中文描述的整理、中文裡「提到人」的判斷 |

第一次使用 Gemma 前：到 Hugging Face 的 `google/gemma-3-4b-it` 頁面登入並同意使用條款，
然後在電腦上執行 `huggingface-cli login`。

要切回原本的英文 SmolVLM：執行時加 `--backend smolvlm`。

## 安裝（Windows，Python 3.10）

```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision          # 要用顯示卡請依 pytorch.org 的指令裝 CUDA 版
pip install transformers huggingface_hub ultralytics easyocr opencv-python pillow numpy pyttsx3
huggingface-cli login                  # Gemma 需要，先在模型頁面同意條款
git clone https://github.com/samuelyu2002/ImVisible external/ImVisible
```

- 第一次執行會自動下載 Gemma、YOLOv8n、EasyOCR 的模型，需要連網一次。
- 手機模式要用 `--https`，需要 `openssl` 指令（裝 Git for Windows 就有）。
- 中文語音：Windows 設定 → 時間與語言 → 語音，確認有「中文（台灣）」。
- 筆電端英文語音要用 Piper 的話：`pip install piper-tts`，
  再 `python -m piper.download_voices en_US-lessac-medium --data-dir models/piper`。不裝也能跑。

## 執行（都在這個資料夾下）

```
python -m app.assistant --mode read --source 某張圖.jpg     # 單張圖測試
python -m app.assistant --mode street                      # 用電腦的攝影機
python -m app.server --https                               # 手機當鏡頭和喇叭
```

手機和電腦連同一個 Wi-Fi 或熱點，手機瀏覽器打開終端機印出的網址。

## 不帶電腦出門：讓手機從外面連回實驗室電腦

伺服器就是一個網頁，手機只要連得到電腦就能用。電腦留在實驗室開著，手機用行動網路連回來。
代價是不再「完全離線」：畫面會經過網路傳回你自己的電腦（不是第三方雲端）。

**做法 A：Tailscale（私人網路，只有你的裝置連得到）**

1. 電腦和手機都安裝 Tailscale，登入同一個帳號。
2. 電腦執行 `python -m app.server --https`。
3. 電腦執行 `tailscale ip -4` 查出它的位址（100 開頭）。
4. 手機瀏覽器打開 `https://<那個位址>:8443`，憑證警告按「繼續前往」。

**做法 B：Cloudflare 臨時通道（不用在手機裝東西，但網址是公開的）**

1. 電腦執行 `python -m app.server`（不要加 `--https`）。
2. 另開一個終端機執行 `cloudflared tunnel --url http://localhost:8000`。
3. 手機打開它印出的 `https://….trycloudflare.com` 網址。

知道網址的人都能連，畫面會經過 Cloudflare，只適合測試。
