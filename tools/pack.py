"""
pack.py — [資服版] 打包成乾淨的 zip，可以直接給別人
==================================================
排除：
  .venv/    虛擬環境（寫死這台電腦的路徑，換電腦不能用；對方用 安裝.bat 自己裝）
  certs/    憑證和私鑰（給別人等於讓別人能冒充你的伺服器）
  .git/、__pycache__/、results/ 等開發用的暫存檔
輸出到上一層資料夾：視覺助理_資服版_日期.zip
"""

import os
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP = os.path.basename(ROOT)                       # zip 裡的最上層資料夾名稱
EXCLUDE_DIRS = {".venv", "certs", ".git", "__pycache__", "results", ".claude"}
EXCLUDE_EXT = {".pyc", ".pyo", ".log", ".zip"}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    out = os.path.join(os.path.dirname(ROOT), f"視覺助理_資服版_{time.strftime('%Y%m%d')}.zip")
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for d, dirs, files in os.walk(ROOT):
            dirs[:] = sorted(x for x in dirs if x not in EXCLUDE_DIRS)
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in EXCLUDE_EXT:
                    continue
                p = os.path.join(d, f)
                z.write(p, os.path.join(TOP, os.path.relpath(p, ROOT)))
                n += 1
    mb = os.path.getsize(out) / 1024 ** 2
    print(f"已打包 {n} 個檔案（{mb:.1f} MB）")
    print(f"→ {out}")
    print("已排除：虛擬環境 .venv、憑證與私鑰 certs、開發暫存檔")
    print("對方解壓縮後，雙擊 安裝.bat 即可。")


if __name__ == "__main__":
    main()
