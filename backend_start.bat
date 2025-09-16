@echo off
setlocal
chcp 65001 >nul

REM 到项目根目录
cd /d "%~dp0"

REM 激活虚拟环境（不存在就提示）
if not exist ".venv\Scripts\activate" (
  echo [ERROR] 找不到 .venv\Scripts\activate
  echo 请先在本目录创建虚拟环境：py -3 -m venv .venv  或  python -m venv .venv
  pause
  exit /b 1
)
call .venv\Scripts\activate

REM 代理（用 Clash：7890；如你用别的端口，改这里）
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890
set NO_PROXY=localhost,127.0.0.1

REM 启动 Flask
set FLASK_APP=api_server.py
echo [INFO] 启动后端...
flask run --host 0.0.0.0 --port 8889

endlocal
