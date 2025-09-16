# PowerShell启动脚本 - API代理服务器
Write-Host "Starting API Proxy Server..." -ForegroundColor Green
Write-Host "This server will handle CORS issues for the frontend" -ForegroundColor Yellow
Write-Host ""

# 检查Python是否安装
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python version: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and try again" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# 检查Flask是否安装
try {
    python -c "import flask" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing Flask and dependencies..." -ForegroundColor Yellow
        pip install flask flask-cors requests
    } else {
        Write-Host "Flask and dependencies are already installed" -ForegroundColor Green
    }
} catch {
    Write-Host "Error checking Flask installation" -ForegroundColor Red
}

# 启动API代理服务器
Write-Host ""
Write-Host "Starting API proxy server on http://localhost:5000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# 设置环境变量
$env:FLASK_ENV = "development"
$env:FLASK_DEBUG = "1"

# 启动服务器
python api_proxy_server.py

Read-Host "Press Enter to exit"

