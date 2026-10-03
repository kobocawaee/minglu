"""
setup_helper.py — [資服版] 安裝.bat 的後半段（套件裝好之後執行）
================================================================
1. 檢查顯示卡
2. 在桌面建立「視覺助理」捷徑
3. 一步一步帶使用者登入 Hugging Face、同意 Gemma 條款（這步沒辦法自動化）
4. 先把模型下載好，第一次啟動才不用等

可以重複執行：已經完成的步驟會自動跳過。
"""

import os
import sys
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from app import config  # noqa: E402

GEMMA_PAGE = f"https://huggingface.co/{config.GEMMA_MODEL}"
TOKEN_PAGE = "https://huggingface.co/settings/tokens"


def title(n, text):
    print()
    print("=" * 56)
    print(f"  步驟 {n}：{text}")
    print("=" * 56)


def ask_enter(msg="完成後按 Enter 繼續…"):
    try:
        input(msg)
    except EOFError:
        pass


def check_gpu():
    title(1, "檢查顯示卡")
    import torch
    if not torch.cuda.is_available():
        print("  [注意] 沒有偵測到可用的 NVIDIA 顯示卡。")
        print("         程式還是能跑，但 Gemma 會改用 CPU，每次回答可能要好幾十秒。")
        print("         有 NVIDIA 顯示卡的話，請確認已安裝最新的顯示卡驅動程式。")
        return
    p = torch.cuda.get_device_properties(0)
    gb = p.total_memory / 1024 ** 3
    print(f"  顯示卡：{p.name}（{gb:.1f} GB）")
    if gb < 7.5:
        print("  [注意] 顯示記憶體少於 8 GB，同時載入 Gemma 和語音辨識可能會不夠。")
        print("         不夠的話，可以在啟動時關掉語音功能：啟動.bat --no-voice")
    else:
        print("  OK")


def make_shortcut():
    title(2, "建立桌面捷徑")
    try:
        import win32com.client
        shell = win32com.client.Dispatch("WScript.Shell")
        desktop = shell.SpecialFolders("Desktop")
        lnk = shell.CreateShortcut(os.path.join(desktop, "視覺助理.lnk"))
        lnk.TargetPath = os.path.join(ROOT, ".venv", "Scripts", "pythonw.exe")
        lnk.Arguments = f'"{os.path.join(ROOT, "launcher.pyw")}"'
        lnk.WorkingDirectory = ROOT
        lnk.IconLocation = os.path.join(ROOT, "app", "static", "icon.ico") + ",0"
        lnk.Description = "視覺助理啟動器"
        lnk.Save()
        print(f"  已在桌面建立「視覺助理」：{desktop}")
    except Exception as e:
        print(f"  [注意] 建立捷徑失敗（{e}）")
        print(f"         可以直接雙擊這個檔案啟動：{os.path.join(ROOT, 'launcher.pyw')}")


def hf_login():
    title(3, "登入 Hugging Face（Gemma 模型需要）")
    from huggingface_hub import get_token, login, auth_check
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    if not get_token():
        print("  Gemma 是 Google 的模型，每個人都要用自己的帳號同意使用條款。請照下面做：")
        print()
        print("   (1) 到 https://huggingface.co/join 註冊帳號（已經有就直接登入）")
        print(f"   (2) 打開 {GEMMA_PAGE}")
        print("       按頁面上的「Acknowledge license」並同意條款")
        print(f"   (3) 打開 {TOKEN_PAGE}")
        print("       按「Create new token」→ 類型選「Read」→ 建立後按複製")
        print("   (4) 回到這個視窗，在下面貼上權杖（按滑鼠右鍵就能貼上，畫面不會顯示，是正常的），按 Enter")
        print()
        webbrowser.open(GEMMA_PAGE)
        webbrowser.open(TOKEN_PAGE)
        while not get_token():
            try:
                login(add_to_git_credential=False)
            except Exception as e:
                print(f"  登入失敗：{e}")
                print("  請確認權杖有完整複製，再試一次。")

    while True:   # 確認真的有權限下載 Gemma（同意條款要在網頁上按）
        try:
            auth_check(config.GEMMA_MODEL)
            print("  已登入，也有 Gemma 的使用權限。OK")
            return True
        except GatedRepoError:
            print()
            print("  [還差一步] 帳號已登入，但還沒同意 Gemma 的使用條款。")
            print(f"  請到 {GEMMA_PAGE} 按「Acknowledge license」並同意。")
            webbrowser.open(GEMMA_PAGE)
            ask_enter("  同意完成後按 Enter 再檢查一次（按 Ctrl+C 可以先跳過）…")
        except HfHubHTTPError as e:
            if "401" in str(e):
                print("  權杖無效，請重新登入。")
                login(add_to_git_credential=False)
            else:
                print(f"  [注意] 無法確認權限（{e}），可能是網路問題，先跳過。")
                return False
        except Exception as e:
            print(f"  [注意] 無法確認權限（{e}），先跳過。")
            return False


def download_models(gemma_ok):
    title(4, "先下載模型（約 11 GB，只有第一次需要）")
    from huggingface_hub import snapshot_download
    jobs = [(config.ASR_MODEL, "語音辨識 Whisper（約 1.6 GB）")]
    if gemma_ok:
        jobs.insert(0, (config.GEMMA_MODEL, "影像模型 Gemma（約 8.6 GB）"))
    for repo, name in jobs:
        print(f"  下載 {name} …")
        try:
            snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken",
                                                    "*.jinja"])
            print("  OK")
        except Exception as e:
            print(f"  [注意] 下載失敗（{e}），第一次啟動時會再自動下載。")
    print("  下載讀字模型 EasyOCR …")
    try:
        import easyocr
        easyocr.Reader(config.OCR_LANGS, verbose=False)
        print("  OK")
    except Exception as e:
        print(f"  [注意] 下載失敗（{e}），第一次使用讀字模式時會再自動下載。")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    check_gpu()
    make_shortcut()
    try:
        gemma_ok = hf_login()
    except KeyboardInterrupt:
        print("\n  先跳過登入。之後可以重新執行 安裝.bat 完成這一步。")
        gemma_ok = False
    download_models(gemma_ok)
    print()
    print("=" * 56)
    if gemma_ok:
        print("  安裝完成！雙擊桌面的「視覺助理」→ 按「啟動」→ 用手機掃 QR Code。")
    else:
        print("  大致安裝完成，但 Hugging Face 這步還沒好，Gemma 還不能用。")
        print("  完成登入和同意條款後，重新執行 安裝.bat 即可（已完成的步驟會跳過）。")
    print("=" * 56)


if __name__ == "__main__":
    main()
