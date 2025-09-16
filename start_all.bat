@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

REM === 启动后端（新窗口）===
if exist "%~dp0backend_start.bat" (
  start "backend" "%~dp0backend_start.bat"
) else (
  echo [WARN] 未找到 backend_start.bat
)

REM === 稍等 2 秒再启动前端 ===
timeout /t 2 >nul

REM === 前端直接跑 npm run dev ===
set "NVM_HOME=%LOCALAPPDATA%\nvm"
set "NVM_SYMLINK=%LOCALAPPDATA%\nodejs"
set "PATH=%NVM_HOME%;%NVM_SYMLINK%;%PATH%"
"%NVM_HOME%\nvm.exe" use 20.14.0

start "frontend" cmd /k "npm run dev -- --host 0.0.0.0 --port 5173"

REM === 再等 8 秒后自动打开浏览器 ===
timeout /t 8 >nul
start "" "http://127.0.0.1:5173/"

endlocal
