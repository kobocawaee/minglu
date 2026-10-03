@echo off
chcp 65001 >nul
rem 視覺助理：第一次使用前雙擊這個檔案安裝。可以重複執行，已完成的步驟會跳過。
cd /d "%~dp0"
title 視覺助理 安裝程式

echo ============================================================
echo   視覺助理 安裝程式
echo   第一次安裝要下載約 12 GB，視網速需要半小時到一小時以上
echo   過程中請不要關閉這個視窗
echo ============================================================
echo.

rem ---- 找 Python 3.10 ~ 3.12 ----
set "PY="
for %%v in (3.12 3.11 3.10) do (
  if not defined PY (
    py -%%v -c "import sys" >nul 2>&1 && set "PY=py -%%v"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo [需要先安裝 Python]
  echo   這台電腦沒有 Python 3.10 ~ 3.12。請到即將打開的網頁下載 Python 3.12，
  echo   安裝時記得勾選「Add python.exe to PATH」，裝好後再雙擊一次 安裝.bat。
  start "" "https://www.python.org/downloads/release/python-3128/"
  pause
  exit /b 1
)
echo [OK] 使用 Python：%PY%

rem ---- 資料夾路徑不能太長：Windows 的路徑上限是 260 字，套件裡有些檔案路徑很深 ----
%PY% -c "import os,sys; sys.exit(1 if len(os.getcwd()) > 90 else 0)"
if errorlevel 1 (
  echo.
  echo [資料夾路徑太長]
  echo   目前位置：%CD%
  echo   Windows 的路徑長度有上限，放在太深的資料夾會安裝失敗。
  echo   請把整個資料夾搬到短一點的地方，例如 C:\視覺助理 或 D:\視覺助理，再雙擊一次 安裝.bat。
  pause
  exit /b 1
)

rem ---- 建立虛擬環境 ----
if not exist ".venv\Scripts\python.exe" (
  echo.
  echo 建立虛擬環境 .venv …
  %PY% -m venv .venv
  if errorlevel 1 goto :fail
)
set "VPY=.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip -q
if errorlevel 1 goto :fail

rem ---- 安裝 PyTorch：有 NVIDIA 顯示卡就裝 CUDA 版 ----
"%VPY%" -c "import torch" >nul 2>&1
if errorlevel 1 (
  echo.
  where nvidia-smi >nul 2>&1
  if errorlevel 1 (
    echo 沒有偵測到 NVIDIA 顯示卡，安裝 CPU 版 PyTorch …
    "%VPY%" -m pip install torch==2.14.1 torchvision==0.29.1
  ) else (
    echo 安裝顯示卡版 PyTorch，約 3 GB …
    "%VPY%" -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
  )
  if errorlevel 1 goto :fail
) else (
  echo [OK] PyTorch 已安裝
)

rem ---- 其他套件 ----
echo.
echo 安裝其他套件 …
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

rem ---- 捷徑、Hugging Face 登入、下載模型 ----
set PYTHONIOENCODING=utf-8
"%VPY%" tools\setup_helper.py
echo.
pause
exit /b 0

:fail
echo.
echo [安裝失敗] 常見原因：
echo   1. 網路中斷：確認網路後，再雙擊一次 安裝.bat，已裝好的部分會跳過
echo   2. 硬碟空間不足：需要約 20 GB
echo   3. 防毒軟體擋住：暫時允許後再試一次
echo 如果一直失敗，請把這個視窗的畫面截圖給開發者。
pause
exit /b 1
