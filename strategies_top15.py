import numpy as np
import pandas as pd
import pandas_ta as ta

# ------- helpers -------
def _xover(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


def _xunder(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


def _pack(df, long, short, entry=None, atr=None, atr_mult=2.0, rr=1.5):
    entry = df["close"] if entry is None else entry
    atr = ta.atr(df["high"], df["low"], df["close"]) if atr is None else atr
    sl = np.where(long, df["low"] - atr * atr_mult,
                  np.where(short, df["high"] + atr * atr_mult, np.nan))
    tp = np.where(long, entry + atr * rr,
                  np.where(short, entry - atr * rr, np.nan))
    sig = np.where(long, 1, np.where(short, -1, 0))
    return pd.DataFrame({"signal": sig, "entry": entry, "sl": sl, "tp": tp})


# 强韧版：确保 tz-naive 的 DatetimeIndex，且仅保留 OHLCV
def _ensure_dt_index(df: pd.DataFrame, default_freq: str = "4H") -> pd.DataFrame:
    """
    确保是按时间排序的 tz-naive DatetimeIndex，并只保留 OHLCV 列。
    若没有时间列，则造一条等距时间轴做兜底，避免 RangeIndex 导致的 to_period 等问题。
    """
    dfi = df.copy()

    # 1) 拿到 DatetimeIndex
    if not isinstance(dfi.index, pd.DatetimeIndex):
        idx = None
        if "timestamp" in dfi.columns:
            idx = pd.to_datetime(dfi["timestamp"], errors="coerce", utc=True)
        elif "ts" in dfi.columns:
            idx = pd.to_datetime(dfi["ts"], errors="coerce", utc=True)
        if idx is None or idx.isna().all():
            # 兜底：造一个等距索引，避免 RangeIndex
            start = pd.Timestamp.utcnow().floor("D") - pd.Timedelta(hours=len(dfi) * 4)
            idx = pd.date_range(start=start, periods=len(dfi), freq=default_freq)
        dfi.index = idx

    # 2) 去时区，排序
    if getattr(dfi.index, "tz", None) is not None:
        dfi.index = dfi.index.tz_convert(None)
    dfi = dfi.sort_index()

    # 3) 只保留 OHLCV，volume 缺失时兜底为 1
    cols = ["open", "high", "low", "close", "volume"]
    for c in cols:
        if c not in dfi.columns:
            dfi[c] = np.nan if c != "volume" else 1.0
    dfi[["open", "high", "low", "close", "volume"]] = dfi[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    dfi = dfi.dropna(subset=["open", "high", "low", "close"])
    return dfi[["open", "high", "low", "close", "volume"]].copy()


# 1) EMA 20/50 + ADX 过滤（趋势跟随）
def strat_ema_adx(df, fast=20, slow=50, adx_len=14, adx_min=20, atr_mult=2.0, rr=2.0):
    e1 = ta.ema(df["close"], length=fast)
    e2 = ta.ema(df["close"], length=slow)
    adx = ta.adx(df["high"], df["low"], df["close"], length=adx_len)[f"ADX_{adx_len}"]
    long = _xover(e1, e2) & (adx >= adx_min)
    short = _xunder(e1, e2) & (adx >= adx_min)
    return _pack(df, long, short, atr_mult=atr_mult, rr=rr)


# 2) MACD 信号线交叉 + 直方图确认
def strat_macd(df, fast=12, slow=26, sig=9, atr_mult=2.0, rr=2.0):
    macd = ta.macd(df["close"], fast=fast, slow=slow, signal=sig)
    macd_line = macd.columns[0]
    sig_line = macd.columns[1]
    hist_col = macd.columns[2]
    m = macd[macd_line]
    s = macd[sig_line]
    h = macd[hist_col]
    long = _xover(m, s) & (h > 0)
    short = _xunder(m, s) & (h < 0)
    return _pack(df, long, short, atr_mult=atr_mult, rr=rr)


# 3) RSI 反转 + EMA200 大势过滤（均值回归）
def strat_rsi_reversion(df, rsi_len=14, low=30, high=70, ema_len=200, atr_mult=1.8, rr=1.2):
    rsi = ta.rsi(df["close"], length=rsi_len)
    ema200 = ta.ema(df["close"], length=ema_len)
    long = (df["close"] > ema200) & _xover(rsi, pd.Series([low] * len(df), index=df.index))
    short = (df["close"] < ema200) & _xunder(rsi, pd.Series([high] * len(df), index=df.index))
    return _pack(df, long, short, atr_mult=atr_mult, rr=rr)


# 4) 布林带均值回归（BB mean-revert）
def strat_bb_mean(df, length=20, std=2.0, atr_mult=1.6, rr=1.2):
    bb = ta.bbands(df["close"], length=length, std=std)
    lower = bb.filter(like="BBL").iloc[:, 0]
    mid = bb.filter(like="BBM").iloc[:, 0]
    upper = bb.filter(like="BBU").iloc[:, 0]
    rsi = ta.rsi(df["close"], length=14)
    long = (df["close"] < lower) & (rsi < 40)
    short = (df["close"] > upper) & (rsi > 60)
    out = _pack(df, long, short, atr_mult=atr_mult, rr=rr)
    out.loc[long, "tp"] = mid[long]
    out.loc[short, "tp"] = mid[short]
    return out


# 5) 布林带挤压突破（Squeeze Breakout）
def strat_bb_squeeze(df, length=20, std=2.0, bw_th=0.05, atr_mult=2.2, rr=2.2):
    bb = ta.bbands(df["close"], length=length, std=std)
    lower = bb.filter(like="BBL").iloc[:, 0]
    mid = bb.filter(like="BBM").iloc[:, 0]
    upper = bb.filter(like="BBU").iloc[:, 0]
    bandwidth = (upper - lower) / mid
    long = (bandwidth < bw_th) & _xover(df["close"], upper)
    short = (bandwidth < bw_th) & _xunder(df["close"], lower)
    return _pack(df, long, short, atr_mult=atr_mult, rr=rr)


# 6) Donchian 20 突破 + ATR 止损（海龟）
def strat_donchian(df, length=20, atr_mult=2.5, rr=2.5):
    dc = ta.donchian(df["high"], df["low"], lower_length=length, upper_length=length)
    upper = dc.filter(like="DCU").iloc[:, 0]
    lower = dc.filter(like="DCL").iloc[:, 0]
    long = _xover(df["close"], upper)
    short = _xunder(df["close"], lower)
    return _pack(df, long, short, atr_mult=atr_mult, rr=rr)


# 7) Supertrend 趋势跟随
def strat_supertrend(df, length=10, multiplier=3.0, atr_mult=2.0, rr=2.0):
    st = ta.supertrend(df["high"], df["low"], df["close"], length=length, multiplier=multiplier)
    dcol = [c for c in st.columns if c.startswith("SUPERTd")][0]
    long = _xover(st[dcol], pd.Series(0, index=df.index))
    short = _xunder(st[dcol], pd.Series(0, index=df.index))
    return _pack(df, long, short, atr_mult=atr_mult, rr=rr)


# 8) Keltner 通道突破
def strat_keltner_break(df, length=20, mult=2.0, atr_mult=2.0, rr=2.0):
    kc = ta.kc(df["high"], df["low"], df["close"], length=length, scalar=mult)
    upper = kc.filter(like="KCU").iloc[:, 0]
    lower = kc.filter(like="KCL").iloc[:, 0]
    long = _xover(df["close"], upper)
    short = _xunder(df["close"], lower)
    return _pack(df, long, short, atr_mult=atr_mult, rr=rr)


# 9) Ichimoku 基准线交叉 + 云层过滤
def strat_ichimoku(df, tenkan=9, kijun=26, senkou=52, atr_mult=2.2, rr=2.0):
    # 手算 Ichimoku（不依赖 pandas_ta.ichimoku，避免“close 必填”的版本差异）
    df = _ensure_dt_index(df)

    hh_t = df["high"].rolling(tenkan, min_periods=tenkan).max()
    ll_t = df["low"].rolling(tenkan, min_periods=tenkan).min()
    its = (hh_t + ll_t) / 2.0  # Tenkan

    hh_k = df["high"].rolling(kijun, min_periods=kijun).max()
    ll_k = df["low"].rolling(kijun, min_periods=kijun).min()
    iks = (hh_k + ll_k) / 2.0  # Kijun

    isa = (its + iks) / 2.0  # Span A
    hh_s = df["high"].rolling(senkou, min_periods=senkou).max()
    ll_s = df["low"].rolling(senkou, min_periods=senkou).min()
    isb = (hh_s + ll_s) / 2.0  # Span B

    cloud_top = pd.concat([isa, isb], axis=1).max(axis=1)
    cloud_bot = pd.concat([isa, isb], axis=1).min(axis=1)

    long = _xover(its, iks) & (df["close"] > cloud_top)
    short = _xunder(its, iks) & (df["close"] < cloud_bot)

    return _pack(df, long, short, atr_mult=atr_mult, rr=rr)


# 10) Parabolic SAR + EMA 趋势过滤
def strat_psar(df, af=0.02, afmax=0.2, ema_len=50, atr_mult=2.0, rr=2.0):
    ps = ta.psar(df["high"], df["low"], df["close"], af=af, max_af=afmax)
    ps_val = ps.filter(like="PSAR").iloc[:, 0]
    ema = ta.ema(df["close"], length=ema_len)
    long = _xover(df["close"], ps_val) & (df["close"] > ema)
    short = _xunder(df["close"], ps_val) & (df["close"] < ema)
    return _pack(df, long, short, atr_mult=atr_mult, rr=rr)


# 11) Stochastic RSI 极值反转
def strat_stochrsi(df, rsi_len=14, stoch_len=14, k=3, d=3, low=0.2, high=0.8, atr_mult=1.8, rr=1.2):
    sr = ta.stochrsi(df["close"], length=rsi_len, rsi_length=rsi_len, k=k, d=d)
    kcol = sr.filter(like="STOCHRSIk").iloc[:, 0]
    dcol = sr.filter(like="STOCHRSId").iloc[:, 0]
    long = _xover(kcol, dcol) & (kcol < low)
    short = _xunder(kcol, dcol) & (kcol > high)
    return _pack(df, long, short, atr_mult=atr_mult, rr=rr)


# 12) CCI 极值 + 零轴回归
def strat_cci(df, length=20, lo=-100, hi=100, atr_mult=1.8, rr=1.2):
    cci = ta.cci(df["high"], df["low"], df["close"], length=length)
    zero = pd.Series(0, index=df.index)
    long = (cci < lo) & _xover(cci, zero)
    short = (cci > hi) & _xunder(cci, zero)
    return _pack(df, long, short, atr_mult=atr_mult, rr=rr)


# 13) ADX + DI 交叉（趋势确定）
def strat_adx_di(df, length=14, adx_min=20, atr_mult=2.0, rr=2.0):
    adx = ta.adx(df["high"], df["low"], df["close"], length=length)
    dmp = adx[f"DMP_{length}"]
    dmn = adx[f"DMN_{length}"]
    ad = adx[f"ADX_{length}"]
    long = _xover(dmp, dmn) & (ad >= adx_min)
    short = _xunder(dmp, dmn) & (ad >= adx_min)
    return _pack(df, long, short, atr_mult=atr_mult, rr=rr)


# 14) Heikin-Ashi 反转 + EMA 基线
def strat_heikin_ema(df, ema_len=50, atr_mult=2.0, rr=1.6):
    ha = ta.ha(df["open"], df["high"], df["low"], df["close"])
    ha_o = ha.filter(like="HA_open").iloc[:, 0]
    ha_c = ha.filter(like="HA_close").iloc[:, 0]
    ema = ta.ema(df["close"], length=ema_len)
    long = _xover(ha_c, ha_o) & (df["close"] > ema)
    short = _xunder(ha_c, ha_o) & (df["close"] < ema)
    return _pack(df, long, short, atr_mult=atr_mult, rr=rr)


# 15) VWAP 回踩/突破（日内或多周期）
def strat_vwap_pullback(df, anchor="D", atr_mult=1.8, rr=1.5):
    """
    手算 VWAP：
      vwap = 累计(典型价*成交量) / 累计(成交量)
    按 anchor（默认日 D）分组，每个锚点周期内独立累加。
    不使用 pandas_ta.vwap，规避 to_period 和排序告警。
    """
    df = _ensure_dt_index(df)

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]

    try:
        grp = df.index.floor(anchor)
    except Exception:
        grp = pd.Series(0, index=df.index)

    cum_pv = pv.groupby(grp).cumsum()
    cum_v  = df["volume"].groupby(grp).cumsum().replace(0, np.nan)
    vwap = cum_pv / cum_v

    long = _xover(df["close"], vwap)
    short = _xunder(df["close"], vwap)

    return _pack(df, long, short, entry=df["close"], atr_mult=atr_mult, rr=rr)


# ------- 注册表（名字 -> 函数）-------
REGISTRY = {
    "ema_adx": strat_ema_adx,
    "macd_plus": strat_macd,
    "rsi_reversion": strat_rsi_reversion,
    "bb_mean": strat_bb_mean,
    "bb_squeeze": strat_bb_squeeze,
    "donchian": strat_donchian,
    "supertrend": strat_supertrend,
    "keltner_break": strat_keltner_break,
    "ichimoku_kijun": strat_ichimoku,
    "psar_trend": strat_psar,
    "stochrsi": strat_stochrsi,
    "cci_reversion": strat_cci,
    "adx_di": strat_adx_di,
    "heikin_ema": strat_heikin_ema,
    "vwap_pullback": strat_vwap_pullback,
}


def run_strategy(name: str, df: pd.DataFrame, **params) -> pd.DataFrame:
    """统一入口：返回含 signal/entry/sl/tp 的 DataFrame"""
    fn = REGISTRY[name]
    return fn(df.copy(), **params)



