@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Trading Dashboard - SAFE (一键准备环境+启动，不改项目文件)

REM ===== 可改参数 =====
set "PROXY_URL=http://127.0.0.1:7890"
set "NODE_VERSION=20.14.0"
set "BACKEND_PORT=8889"
set "FRONTEND_PORT=5173"
REM ====================

set "PROJECT_DIR=%~dp0"
set "PATH=C:\nvm4w\nodejs;%PATH%"

echo [SAFE] 项目: %PROJECT_DIR%
echo [SAFE] 代理: %PROXY_URL%

cd /d "%PROJECT_DIR%"

REM ---------- 创建/激活 venv ----------
if not exist .venv (
  echo [SAFE] 创建虚拟环境 .venv...
  py -3 -m venv .venv || python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install -U pip setuptools wheel

REM ---------- 安装依赖（过滤 requirements.txt 中的 pandas_ta 行；原文件不改） ----------
if exist requirements.txt (
  echo [SAFE] 解析 requirements.txt（忽略 pandas_ta）...
  powershell -NoProfile -Command ^
    "(Get-Content 'requirements.txt' | Where-Object {$_ -notmatch '^\s*pandas[_-]ta'}) | Set-Content '_req.no-pta.txt'"
  echo [SAFE] 安装依赖（不含 pandas_ta）...
  pip install --no-cache-dir -r "_req.no-pta.txt"
) else (
  echo [SAFE] 未找到 requirements.txt，安装常用依赖...
  pip install --no-cache-dir flask flask-cors requests pandas numpy ccxt streamlit ta
)

REM ---------- 如缺少 'pandas-ta' 分发，则在 .venv 内安装兼容包（仅影响 .venv，可随时删除） ----------
python -c "import importlib.metadata as m; print(m.version('pandas-ta'))" 1>nul 2>nul
if errorlevel 1 (
  echo [SAFE] 未检测到 'pandas-ta' 分发，安装兼容包（基于 ta 指标库）...
  set "_SHIM=%PROJECT_DIR%_pta_shim"
  rmdir /S /Q "%_SHIM%" 2>nul
  mkdir "%_SHIM%\pandas_ta" 2>nul

  >"%_SHIM%\pyproject.toml" (
    echo [build-system]
    echo requires = ["setuptools^>=61.0"]
    echo build-backend = "setuptools.build_meta"
  )

  >"%_SHIM%\setup.cfg" (
    echo [metadata]
    echo name = pandas-ta
    echo version = 0.3.14.post99
    echo description = Compatibility shim for pandas_ta built on ta
    echo license = MIT
    echo.
    echo [options]
    echo packages = find:
    echo install_requires =
    echo^ ^ ^ pandas^>=1.3
    echo^ ^ ^ numpy^>=1.20
    echo^ ^ ^ ta^>=0.11.0
    echo python_requires = ^>=3.8
  )

  >"%_SHIM%\pandas_ta\__init__.py" (
    echo import pandas as pd
    echo import numpy as np
    echo from ta.trend import EMAIndicator, SMAIndicator, ADXIndicator, MACD
    echo from ta.momentum import RSIIndicator
    echo from ta.volatility import AverageTrueRange
    echo __version__ = "0.3.14.post99-shim"
    echo.
    echo def ema(close, length=10):
    echo^    return EMAIndicator(close=close, window=length).ema_indicator()
    echo.
    echo def sma(close, length=10):
    echo^    return SMAIndicator(close=close, window=length).sma_indicator()
    echo.
    echo def rsi(close, length=14):
    echo^    return RSIIndicator(close=close, window=length).rsi()
    echo.
    echo def macd(close, fast=12, slow=26, signal=9):
    echo^    m = MACD(close=close, window_fast=fast, window_slow=slow, window_sign=signal)
    echo^    return pd.DataFrame({"MACD": m.macd(), "MACDs": m.macd_signal(), "MACDh": m.macd_diff()})
    echo.
    echo def adx(high, low, close, length=14):
    echo^    return ADXIndicator(high=high, low=low, close=close, window=length).adx()
    echo.
    echo def adx_di(high, low, close, length=14):
    echo^    a = ADXIndicator(high=high, low=low, close=close, window=length)
    echo^    return pd.DataFrame({"ADX": a.adx(), "+DI": a.adx_pos(), "-DI": a.adx_neg()})
    echo.
    echo def atr(high, low, close, length=14):
    echo^    return AverageTrueRange(high=high, low=low, close=close, window=length).average_true_range()
    echo.
    echo def donchian_channel(high, low, close=None, length=20, offset=0):
    echo^    upper = high.rolling(window=length, min_periods=1).max()
    echo^    lower = low.rolling(window=length, min_periods=1).min()
    echo^    middle = (upper + lower) / 2.0
    echo^    if offset:
    echo^        upper = upper.shift(offset); lower = lower.shift(offset); middle = middle.shift(offset)
    echo^    return pd.DataFrame({"DCH": upper, "DCL": lower, "DCM": middle})
    echo.
    echo def supertrend(high, low, close, length=10, multiplier=3.0):
    echo^    atr = AverageTrueRange(high=high, low=low, close=close, window=length).average_true_range()
    echo^    hl2 = (high + low) / 2.0
    echo^    upper = hl2 + multiplier * atr
    echo^    lower = hl2 - multiplier * atr
    echo^    st = pd.Series(index=close.index, dtype=float)
    echo^    direction = pd.Series(index=close.index, dtype=int)
    echo^    prev_fu = np.nan; prev_fl = np.nan; prev_st = np.nan
    echo^    for i in range(len(close)):
    echo^        cu, cl = upper.iloc[i], lower.iloc[i]
    echo^        if i == 0:
    echo^            fu, fl = cu, cl
    echo^            st.iloc[i] = fl
    echo^            direction.iloc[i] = 1
    echo^        else:
    echo^            fu = cu if (np.isnan(prev_fu) or cu ^< prev_fu or close.iloc[i-1] ^> prev_fu) else prev_fu
    echo^            fl = cl if (np.isnan(prev_fl) or cl ^> prev_fl or close.iloc[i-1] ^< prev_fl) else prev_fl
    echo^            if prev_st == prev_fu:
    echo^                st_i = fu if close.iloc[i] ^<= fu else fl
    echo^            else:
    echo^                st_i = fl if close.iloc[i] ^>= fl else fu
    echo^            st.iloc[i] = st_i
    echo^            direction.iloc[i] = 1 if st_i == fl else -1
    echo^        prev_fu, prev_fl, prev_st = fu, fl, st.iloc[i]
    echo^    return pd.DataFrame({"SUPERT": st, "SUPERTd": direction, "SUPERTl": lower, "SUPERTh": upper})
  )

  pip install --no-cache-dir "%_SHIM%"
) else (
  echo [SAFE] 已检测到 'pandas-ta' 分发，跳过兼容包安装。
)

REM ---------- 启动后端 ----------
start "backend" cmd /k ^
"cd /d ""%PROJECT_DIR%"" ^
 && call .venv\Scripts\activate ^
 && set HTTP_PROXY=%PROXY_URL% ^
 && set HTTPS_PROXY=%PROXY_URL% ^
 && set NO_PROXY=localhost,127.0.0.1 ^
 && set FLASK_APP=api_server.py ^
 && echo [SAFE] 启动后端... ^
 && flask run --host 0.0.0.0 --port %BACKEND_PORT%"

REM ---------- 启动前端 ----------
start "frontend" cmd /k ^
"cd /d ""%PROJECT_DIR%"" ^
 && if exist frontend (cd frontend) ^
 && nvm use %NODE_VERSION% ^
 && echo [SAFE] 启动前端... ^
 && npm run dev -- --host 0.0.0.0 --port %FRONTEND_PORT%"

REM ---------- 打开浏览器 ----------
timeout /t 5 >nul
start http://127.0.0.1:%FRONTEND_PORT%/

endlocal
