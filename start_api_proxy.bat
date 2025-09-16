@echo off
echo Starting API Proxy Server...
echo This server will handle CORS issues for the frontend
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM 检查Flask是否安装
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo Installing Flask and dependencies...
    pip install flask flask-cors requests
)

REM 启动API代理服务器
echo Starting API proxy server on http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python api_proxy_server.py

pause

