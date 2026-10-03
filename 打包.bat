@echo off
chcp 65001 >nul
rem 視覺助理：打包成乾淨的 zip，可以直接給別人（會排除虛擬環境與私鑰）
cd /d "%~dp0"
title 視覺助理 打包

set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" set "VPY=python"
set PYTHONIOENCODING=utf-8
"%VPY%" tools\pack.py
echo.
pause
