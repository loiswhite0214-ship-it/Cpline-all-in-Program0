@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

REM === nvm & npm 环境 ===
set "NVM_HOME=%LOCALAPPDATA%\nvm"
set "NVM_SYMLINK=%LOCALAPPDATA%\nodejs"
set "PATH=%NVM_HOME%;%NVM_SYMLINK%;%PATH%"
"%NVM_HOME%\nvm.exe" use 20.14.0

REM === 安裝依賴（第一次需要，之後可註釋掉加速）===
if exist package-lock.json (
  npm ci
) else (
  npm install
)

REM === 啟動 Vite 開發伺服器 ===
echo [INFO] 啟動前端...
npm run dev -- --host 0.0.0.0 --port 5173

endlocal
