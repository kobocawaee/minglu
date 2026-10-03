@echo off
chcp 65001 >nul
rem 視覺助理：雙擊這個檔案就會啟動伺服器（手機當鏡頭和喇叭）
cd /d "%~dp0"
title 視覺助理伺服器

if not exist ".venv\Scripts\python.exe" (
  echo [錯誤] 找不到 .venv 虛擬環境，請先依 README_資服版.md 安裝。
  pause
  exit /b 1
)

echo ============================================
echo   視覺助理伺服器啟動中，載入模型約需 30 秒
echo   看到「伺服器已啟動」後，手機打開下面的網址
echo   要關閉：直接關掉這個視窗，或按 Ctrl+C
echo ============================================
echo.

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
".venv\Scripts\python.exe" -m app.server --https %*

echo.
echo 伺服器已停止。
pause
