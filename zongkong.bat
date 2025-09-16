@echo off
setlocal
cd /d "%~dp0"

REM 后端与前端分别拉起到两个窗口
start "backend" "%~dp0backend_start.bat"
REM 给后端2秒起服务（可按需调大）
timeout /t 2 >nul
start "frontend" "%~dp0frontend_start.bat"

REM 再等8秒打开前端（避免 vite 还没 ready）
timeout /t 8 >nul
start "" "http://127.0.0.1:5173/"

endlocal
