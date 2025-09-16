@echo off
setlocal
title Trading Dashboard - Start (MIN)

REM ===== 可改参数 =====
set "PROXY_URL=http://127.0.0.1:7890"
set "NODE_VERSION=20.14.0"
set "BACKEND_PORT=8889"
set "FRONTEND_PORT=5173"
REM ====================

set "PROJECT_DIR=%~dp0"
set "PATH=C:\nvm4w\nodejs;%PATH%"

echo [INFO] 项目: %PROJECT_DIR%
echo [INFO] 代理: %PROXY_URL%

REM ---------- 后端 ----------
start "backend" cmd /k ^
"cd /d ""%PROJECT_DIR%"" ^
 && call .venv\Scripts\activate ^
 && set HTTP_PROXY=%PROXY_URL% ^
 && set HTTPS_PROXY=%PROXY_URL% ^
 && set NO_PROXY=localhost,127.0.0.1 ^
 && set FLASK_APP=api_server.py ^
 && echo [INFO] 启动后端... ^
 && flask run --host 0.0.0.0 --port %BACKEND_PORT%"

REM ---------- 前端 ----------
start "frontend" cmd /k ^
"cd /d ""%PROJECT_DIR%"" ^
 && if exist frontend (cd frontend) ^
 && nvm use %NODE_VERSION% ^
 && echo [INFO] 启动前端... ^
 && npm run dev -- --host 0.0.0.0 --port %FRONTEND_PORT%"

REM ---------- 浏览器 ----------
timeout /t 5 >nul
start http://127.0.0.1:%FRONTEND_PORT%/

endlocal
