# strategies.py  — 改良版：ADX过滤 + ATR动态TP/SL + 当根触发 + 小数位控制
import numpy as np
import pandas as pd
import pandas_ta as ta

# ---------- 工具 ----------
def _ss(x):  # safe series
    return pd.Series(x).astype(float)

def _fmt(symbol: str, v: float) -> float:
    if v is None or not np.isfinite(v): return v
    s = symbol.upper()
    if any(k in s for k in ["BTC","ETH","BNB","SOL"]): return round(float(v), 2)
    if any(k in s for k in ["XRP","DOGE","SHIB","TRX"]): return round(float(v), 6)
    return round(float(v), 4)

def _atr(high, low, close, n=14):
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    atrp = (atr / close) * 100.0
    return atr, atrp

def _adx(high, low, close, n=14):
    adxdf = ta.adx(high=_ss(high), low=_ss(low), close=_ss(close), length=n)
    return adxdf[f"ADX_{n}"]

# --- HTF 重采样：把当前 df 按更大周期聚合成 OHLCV ---
def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    # 需要 ts 是 datetime；dashboard 的 to_ohlcv_df 已经转换过
    g = df.set_index("ts").resample(rule, label="right", closed="right")
    o = g["open"].first()
    h = g["high"].max()
    l = g["low"].min()
    c = g["close"].last()
    v = g["volume"].sum()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna()
    out.reset_index(inplace=True)
    return out

def strategy_diag(df: pd.DataFrame, tf: str) -> dict:
    """返回最近一根K的关键指标，前端诊断用"""
    try:
        close=_ss(df["close"]) ; high=_ss(df["high"]) ; low=_ss(df["low"])
        ema12=close.ewm(span=12).mean(); ema26=close.ewm(span=26).mean()
        dif=ema12-ema26; dea=dif.ewm(span=9).mean(); hist=dif-dea
        ema55=close.ewm(span=55).mean(); ema144=close.ewm(span=144).mean()
        up=np.maximum(ema55, ema144); dn=np.minimum(ema55, ema144)
        sma20=close.rolling(20).mean(); sma60=close.rolling(60).mean()
        atr, atrp=_atr(high,low,close,14); adx=_adx(high,low,close,14)
        last=float(close.iloc[-1])
        last_atr=float(atr.iloc[-1]) if np.isfinite(atr.iloc[-1]) else None
        vegas_dist_atr = (
            (last - float(up.iloc[-1])) / last_atr if last_atr and np.isfinite(up.iloc[-1]) and last > up.iloc[-1] else
            (float(dn.iloc[-1]) - last) / last_atr if last_atr and np.isfinite(dn.iloc[-1]) and last < dn.iloc[-1] else
            None
        )
        return {
            "close": last,
            "adx": float(adx.iloc[-1]) if np.isfinite(adx.iloc[-1]) else None,
            "atrp%": float(atrp.iloc[-1]) if np.isfinite(atrp.iloc[-1]) else None,
            "hist_now": float(hist.iloc[-1]) if np.isfinite(hist.iloc[-1]) else None,
            "hist_prev": float(hist.iloc[-2]) if np.isfinite(hist.iloc[-2]) else None,
            "sma20>sma60": bool(sma20.iloc[-1] > sma60.iloc[-1]) if np.isfinite(sma20.iloc[-1]) and np.isfinite(sma60.iloc[-1]) else None,
            "prev_in_vegas": bool(dn.iloc[-2] <= close.iloc[-2] <= up.iloc[-2]) if np.isfinite(up.iloc[-2]) and np.isfinite(dn.iloc[-2]) else None,
            "vegas_dist_atr": vegas_dist_atr,
        }
    except Exception:
        return {}

# 运行时放松模式（用于快速恢复信号量）
RELAX = False

def set_relax_mode(flag: bool):
    global RELAX
    RELAX = bool(flag)

# 不同周期参数（可按需再调）
PARAMS = {
    "4h": {"adx_min": 18, "atrp_min": 0.35, "tp_atr": {"trend":2.0,"revert":1.5}, "sl_atr": {"trend":1.4,"revert":1.2}},
    "1d": {"adx_min": 16, "atrp_min": 0.30, "tp_atr": {"trend":2.6,"revert":2.0}, "sl_atr": {"trend":1.8,"revert":1.5}},
    "1w": {"adx_min": 14, "atrp_min": 0.50, "tp_atr": {"trend":3.0,"revert":2.3}, "sl_atr": {"trend":2.5,"revert":2.0}},
}
DEF = {"adx_min":16,"atrp_min":0.4,"tp_atr":{"trend":2.2,"revert":1.8},"sl_atr":{"trend":1.6,"revert":1.3}}

def _cfg(tf:str):
    tf = (tf or "").lower()
    return PARAMS.get(tf, DEF)

# ---------- 策略 ----------
def vegas_tunnel(symbol: str, df: pd.DataFrame, tf: str):
    """
    趋势突破：上一根在通道内，本根收在通道外 + ADX过滤 + 超出一定距离
    通道：EMA55 / EMA144；距离门槛：0.3*ATR
    """
    if len(df) < 160: return None
    close = _ss(df["close"]); high=_ss(df["high"]); low=_ss(df["low"])
    ema55 = close.ewm(span=55).mean()
    ema144= close.ewm(span=144).mean()
    up = np.maximum(ema55, ema144); dn = np.minimum(ema55, ema144)

    atr, atrp = _atr(high, low, close, 14)
    adx = _adx(high, low, close, 14)
    if not np.isfinite(adx.iloc[-1]): return None
    if adx.iloc[-1] < _cfg(tf)["adx_min"]: return None
    if atrp.iloc[-1] < _cfg(tf)["atrp_min"]: return None

    prev_in = (dn.iloc[-2] <= close.iloc[-2] <= up.iloc[-2])
    if not prev_in: return None

    last = float(close.iloc[-1]); last_atr = float(atr.iloc[-1])
    if not np.isfinite(last_atr) or last_atr <= 0: return None

    side=None
    # 要求超出通道的幅度至少 X*ATR；放松模式降到 0.15
    dist_mult = 0.15 if RELAX else 0.3
    if last > up.iloc[-1] + dist_mult*last_atr:
        side="BUY"
    elif last < dn.iloc[-1] - dist_mult*last_atr:
        side="SELL"
    else:
        return None

    tp_mult = _cfg(tf)["tp_atr"]["trend"]; sl_mult=_cfg(tf)["sl_atr"]["trend"]
    if side=="BUY":
        target= last + tp_mult*last_atr; stop= last - sl_mult*last_atr
    else:
        target= last - tp_mult*last_atr; stop= last + sl_mult*last_atr

    return {
        "symbol":symbol, "strategy":"vegas_tunnel", "side":side,
        "entry":_fmt(symbol,last), "target":_fmt(symbol,target), "stop":_fmt(symbol,stop),
        "confidence":45, "reason":f"建议单：{symbol}（{tf}）{'做多' if side=='BUY' else '做空'}；收盘有效突破 Vegas 通道 + ADX{int(adx.iloc[-1])} 过滤。"
    }

def chan_simplified(symbol: str, df: pd.DataFrame, tf: str):
    """
    最终优化版（趋势跟随，减少假信号）：
    - 仅当 SMA20/60 在"当根"发生金叉/死叉才触发（去抖动）
    - ADX 更严格：4h≥22, 1d≥20, 1w≥18
    - 长周期方向确认：
        4h → 参考 1d（SMA20 > SMA60 才能做多，< 才能做空）
        1d → 参考 1w
        1w → 无更高周期约束
    - 非对称 ATR 止盈/止损（提高盈亏比）：
        4h:  TP=3.0*ATR,  SL=1.2*ATR
        1d:  TP=3.0*ATR,  SL=1.5*ATR
        1w:  TP=3.5*ATR,  SL=2.0*ATR
    """
    if len(df) < 80:
        return None

    close = _ss(df["close"]); high = _ss(df["high"]); low = _ss(df["low"])
    sma20 = close.rolling(20).mean(); sma60 = close.rolling(60).mean()
    atr, _ = _atr(high, low, close, 14)
    adx = _adx(high, low, close, 14)

    if not np.isfinite(sma20.iloc[-1]) or not np.isfinite(sma60.iloc[-1]):
        return None

    # ADX 门槛（放松模式下降低）
    adx_min_map = {"4h": (16 if RELAX else 22), "1d": (15 if RELAX else 20), "1w": (13 if RELAX else 18)}
    adx_min = adx_min_map.get(tf, 20)
    if not np.isfinite(adx.iloc[-1]) or adx.iloc[-1] < adx_min:
        return None

    # 当根交叉
    cross_up = (sma20.iloc[-2] <= sma60.iloc[-2]) and (sma20.iloc[-1] > sma60.iloc[-1])
    cross_down = (sma20.iloc[-2] >= sma60.iloc[-2]) and (sma20.iloc[-1] < sma60.iloc[-1])
    if not (cross_up or cross_down):
        return None

    # 更高周期方向确认
    htf_rule_map = {"4h": "1D", "1d": "1W"}
    htf_rule = htf_rule_map.get(tf)
    if htf_rule is not None:
        try:
            df_htf = _resample_ohlc(df, htf_rule)
            c_htf = _ss(df_htf["close"]) if len(df_htf) else None
            if c_htf is None or len(c_htf) < 60:
                pass  # 软退化：数据不足时跳过 HTF 确认
            else:
                sma20_h = c_htf.rolling(20).mean()
                sma60_h = c_htf.rolling(60).mean()
                if not (np.isfinite(sma20_h.iloc[-1]) and np.isfinite(sma60_h.iloc[-1])):
                    pass  # 软退化
                else:
                    if cross_up and not (sma20_h.iloc[-1] > sma60_h.iloc[-1]):
                        return None
                    if cross_down and not (sma60_h.iloc[-1] > sma20_h.iloc[-1]):
                        return None
        except Exception:
            return None

    side = "BUY" if cross_up else "SELL"
    last_close = float(close.iloc[-1])
    last_atr = float(atr.iloc[-1]) if np.isfinite(atr.iloc[-1]) else 0.0
    if last_atr <= 0:
        return None

    # 非对称 ATR TP/SL
    mult_map = {
        "4h": (3.0, 1.2),
        "1d": (3.0, 1.5),
        "1w": (3.5, 2.0),
    }
    tp_atr, sl_atr = mult_map.get(tf, (3.0, 1.2))
    if side == "BUY":
        target = last_close + tp_atr * last_atr
        stop = last_close - sl_atr * last_atr
    else:
        target = last_close - tp_atr * last_atr
        stop = last_close + sl_atr * last_atr

    return {
        "symbol": symbol,
        "strategy": "chan_simplified",
        "side": side,
        "entry": _fmt(symbol, last_close),
        "target": _fmt(symbol, target),
        "stop": _fmt(symbol, stop),
        "confidence": 42,
        "reason": f"建议单：{symbol}（{tf}）{'做多' if side=='BUY' else '做空'}；SMA20/60 当根交叉 + HTF 确认 + ADX{int(adx.iloc[-1])}。"
    }

def macd(symbol: str, df: pd.DataFrame, tf: str):
    """
    MACD 交叉 + 基线确认：
    - DIF/DEA 由负转正→BUY（并且收盘在 EMA200 上方）
    - 由正转负→SELL（并且收盘在 EMA200 下方）
    - ADX/ATR% 过滤，小幅抖动不触发
    """
    if len(df) < 220: return None
    close=_ss(df["close"]); high=_ss(df["high"]); low=_ss(df["low"])
    ema12=close.ewm(span=12).mean(); ema26=close.ewm(span=26).mean()
    dif=ema12-ema26; dea=dif.ewm(span=9).mean(); hist=dif-dea
    ema200=close.ewm(span=200).mean()
    atr, atrp=_atr(high,low,close,14); adx=_adx(high,low,close,14)

    # finite 检查，避免 NaN 绕过过滤或行为不一致
    if not (np.isfinite(atrp.iloc[-1]) and np.isfinite(adx.iloc[-1])):
        return None
    if atrp.iloc[-1] < _cfg(tf)["atrp_min"] or adx.iloc[-1] < _cfg(tf)["adx_min"]:
        return None
    if not (np.isfinite(hist.iloc[-1]) and np.isfinite(hist.iloc[-2]) and np.isfinite(ema200.iloc[-1])):
        return None

    use_ema200_filter = not RELAX
    cond_buy = (hist.iloc[-2] <= 0 and hist.iloc[-1] > 0)
    cond_sell= (hist.iloc[-2] >= 0 and hist.iloc[-1] < 0)
    if use_ema200_filter:
        cond_buy = cond_buy and (close.iloc[-1] > ema200.iloc[-1])
        cond_sell= cond_sell and (close.iloc[-1] < ema200.iloc[-1])

    side=None
    if cond_buy:
        side="BUY"
    elif cond_sell:
        side="SELL"
    else:
        return None

    last=float(close.iloc[-1]); last_atr=float(atr.iloc[-1])
    if not np.isfinite(last_atr) or last_atr <= 0:
        return None
    tp_mult=_cfg(tf)["tp_atr"]["trend"]; sl_mult=_cfg(tf)["sl_atr"]["trend"]
    target = last + (tp_mult*last_atr if side=="BUY" else -tp_mult*last_atr)
    stop   = last - (sl_mult*last_atr if side=="BUY" else -sl_mult*last_atr)

    return {
        "symbol":symbol,"strategy":"macd","side":side,
        "entry":_fmt(symbol,last),"target":_fmt(symbol,target),"stop":_fmt(symbol,stop),
        "confidence":30,"reason":f"建议单：{symbol}（{tf}）{'做多' if side=='BUY' else '做空'}；MACD交叉 + EMA200 基线 + ADX过滤。"
    }

# 统一注册表
STRATEGY_REGISTRY = {
    "vegas_tunnel": vegas_tunnel,
    "chan_simplified": chan_simplified,
    "macd": macd,
}
