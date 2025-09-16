import streamlit as st
st.caption("THIS IS macro_btc_monitor_v2.py")

# macro_btc_monitor_v2.py
# 环境基线：Python 3.8；依赖仅用 streamlit / pandas / numpy / requests / pandas_ta / dateutil
# 功能：统一拉取免Token公开数据 → 对齐日频 → 滚动Z分数（win=90）+ 轻平滑(EWMA=2) → 单页双区图
# 配色：荧光绿系（Web3风），移动端适配（高度收紧、fit容器宽度）

import io, re, time, json
import datetime as dt
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ======================
# 页面 & 主题
# ======================
st.set_page_config(page_title="BTC 多因子宏观监控（v2）", layout="wide")
st.markdown("""
<style>
/* 暗底+荧光绿系 */
:root { --fg:#CFFFB0; --fg2:#91FBB2; --fg3:#55E6A5; --fg4:#22C786; --neg:#0F3D3E; --grid:#1e2a36; }
html,body { background:#0B0F14; color:#E6EDF6; }
section[data-testid="stSidebar"] { background:#0D141B; }
hr{border:0;border-top:1px solid #14202a;}
</style>
""", unsafe_allow_html=True)
st.title("BTC 多因子宏观监控（v2，免 Token 公共数据）")

# ======================
# 工具：HTTP & 缓存
# ======================
HEADERS = {"User-Agent":"Mozilla/5.0"}

@st.cache_data(ttl=3600, show_spinner=False)
def _get(url, params=None, timeout=30, as_text=False):
    r = requests.get(url, params=params or {}, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text if as_text else r.json()

@st.cache_data(ttl=3600, show_spinner=False)
def _get_text(url, params=None, timeout=30):
    r = requests.get(url, params=params or {}, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text

def to_daily(df: pd.DataFrame, date_col="DATE"):
    if df is None or df.empty: return df
    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], utc=True)
    x = x.dropna(subset=[date_col]).sort_values(date_col)
    x["DATE"] = x[date_col].dt.floor("D")
    return x.groupby("DATE").last().reset_index()

def rolling_z(s: pd.Series, win=90):
    m  = s.rolling(win, min_periods=max(10, win//5)).mean()
    sd = s.rolling(win, min_periods=max(10, win//5)).std(ddof=0).clip(lower=1e-12)
    return (s - m) / sd

def ewma(s: pd.Series, span=2):
    return s.ewm(span=span, adjust=False, min_periods=1).mean()

def safe_merge(a, b):
    if a is None or a.empty: return b
    if b is None or b.empty: return a
    return pd.merge(a, b, on="DATE", how="outer")

# ======================
# 数据源（免 Token）
# ======================

# 1) FRED：联邦基金利率 / CPI / 失业率（CSV）
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_csv(series_id: str) -> pd.DataFrame:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    txt = _get_text(url, params={"id":series_id}, timeout=30)
    df = pd.read_csv(io.StringIO(txt))
    df["DATE"] = pd.to_datetime(df["DATE"], utc=True, errors="coerce")
    df = df.rename(columns={series_id:"value"}).dropna()
    return df[["DATE","value"]]

def cpi_yoy_from_monthly() -> pd.DataFrame:
    m = fetch_fred_csv("CPIAUCSL")  # 月度
    if m.empty: return pd.DataFrame(columns=["DATE","cpi_yoy"])
    m["DATE"] = m["DATE"].dt.to_period("M").dt.to_timestamp("M", tz="UTC")
    m = m.drop_duplicates("DATE").sort_values("DATE").rename(columns={"value":"cpi"})
    m["cpi_yoy"] = m["cpi"].pct_change(12) * 100.0
    return to_daily(m[["DATE","cpi_yoy"]]).ffill()

# 2) CoinGecko：BTC 日线
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_btc(days=1825):
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    js = _get(url, params={"vs_currency":"usd","days":str(days),"interval":"daily"})
    arr = js.get("prices", [])
    if not arr: return pd.DataFrame(columns=["DATE","btc"])
    a = np.array(arr)
    return pd.DataFrame({"DATE":pd.to_datetime(a[:,0], unit="ms", utc=True), "btc":pd.to_numeric(a[:,1])})

# 3) Alternative.me：Fear & Greed
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fng():
    js = _get("https://api.alternative.me/fng/", params={"limit":"0","format":"json"})
    data = js.get("data", [])
    rows = [(pd.to_datetime(int(d["timestamp"]), unit="s", utc=True), float(d["value"])) for d in data]
    return pd.DataFrame(rows, columns=["DATE","fng"])

# 4) Farside：ETF 净流入
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_etf_flows():
    cands = [
        "https://farside.co.uk/wp-content/uploads/bitcoin_etf_flows.csv",
        "https://www.farside.co.uk/wp-content/uploads/bitcoin_etf_flows.csv",
    ]
    text = None
    for u in cands:
        try:
            t = _get_text(u, timeout=30)
            if "date" in t.lower(): text = t; break
        except Exception: continue
    if text is None:
        return pd.DataFrame(columns=["DATE","etf_net"])
    raw = text.replace("\ufeff","").replace("$","").replace("£","").replace(",","")
    raw = re.sub(r"\(([^)]+)\)", r"-\1", raw)  # (123) -> -123
    df = pd.read_csv(io.StringIO(raw))
    # 日期列
    dcol = None
    for c in df.columns:
        if str(c).strip().lower() in ("date","day"): dcol = c; break
    if dcol is None: dcol = df.columns[0]
    # 净流入列
    fcol = None
    for key in ["net flow","net_flow","netflow","net"]:
        for c in df.columns:
            if key in str(c).strip().lower(): fcol = c; break
        if fcol: break
    if fcol is None:
        num_cols = [c for c in df.columns if c != dcol]
        df["_sum"] = df[num_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        fcol = "_sum"
    df["DATE"] = pd.to_datetime(df[dcol], utc=True, errors="coerce")
    df["etf_net"] = pd.to_numeric(df[fcol], errors="coerce")
    return df.dropna(subset=["DATE"]).sort_values("DATE")[["DATE","etf_net"]]

# 5) Yahoo Finance（非官方）：^IXIC / ^GSPC / DX-Y.NYB / GC=F
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_daily(symbol: str, days=1825):
    now = int(time.time()); period1 = now - days*24*3600
    js = _get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={"period1":str(period1),"period2":str(now),"interval":"1d","includePrePost":"false"},
    )
    try:
        res = js["chart"]["result"][0]
        ts  = res["timestamp"]
        close = res["indicators"]["quote"][0]["close"]
    except Exception:
        return pd.DataFrame(columns=["DATE","close"])
    s = pd.Series(close, index=pd.to_datetime(ts, unit="s", utc=True)).dropna()
    return pd.DataFrame({"DATE":s.index, "close":s.values})

# 6) DefiLlama：稳定币总市值（汇总）
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stablecap():
    js = _get("https://stablecoins.llama.fi/stablecoincharts/all")
    arr = js.get("total", [])
    rows = [(pd.to_datetime(x["date"], unit="s", utc=True), float(x["totalCirculatingUSD"])) for x in arr]
    return pd.DataFrame(rows, columns=["DATE","stablecap"])

# 7) Blockchain.info：算力（TH/s）
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hashrate():
    js = _get("https://api.blockchain.info/charts/hash-rate", params={"timespan":"5years","format":"json","sampled":"true"})
    vals = js.get("values", [])
    rows = [(pd.to_datetime(v["x"], unit="s", utc=True), float(v["y"])) for v in vals]
    return pd.DataFrame(rows, columns=["DATE","hashrate"])

# 8) Binance：资金费率（BTCUSDT 永续）
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_binance_funding(days=365):
    end = int(time.time()*1000)
    start = end - days*24*3600*1000
    js = _get("https://fapi.binance.com/fapi/v1/fundingRate", params={"symbol":"BTCUSDT","startTime":start,"endTime":end,"limit":1000})
    rows = [(pd.to_datetime(int(x["fundingTime"]), unit="ms", utc=True), float(x["fundingRate"])) for x in js]
    return pd.DataFrame(rows, columns=["DATE","funding"])

# ======================
# 拉取 & 合并
# ======================
with st.spinner("拉取公开数据中…"):
    df_rate   = to_daily(fetch_fred_csv("DFF").rename(columns={"value":"rate"}))
    df_unemp  = to_daily(fetch_fred_csv("UNRATE").rename(columns={"value":"unemp"}))
    df_cpiyy  = to_daily(cpi_yoy_from_monthly())  # cpi_yoy
    df_btc    = to_daily(fetch_btc(1825)).rename(columns={"btc":"btc"})
    df_fng    = to_daily(fetch_fng())
    df_etf    = to_daily(fetch_etf_flows())
    df_ixic   = to_daily(fetch_yahoo_daily("^IXIC")).rename(columns={"close":"ixic"})
    df_gspc   = to_daily(fetch_yahoo_daily("^GSPC")).rename(columns={"close":"gspc"})
    df_dxy    = to_daily(fetch_yahoo_daily("DX-Y.NYB")).rename(columns={"close":"dxy"})
    df_gold   = to_daily(fetch_yahoo_daily("GC=F")).rename(columns={"close":"gold"})
    df_stcap  = to_daily(fetch_stablecap())
    df_hash   = to_daily(fetch_hashrate())
    df_fund   = to_daily(fetch_binance_funding())

# 合表
df = None
for x in [df_rate, df_cpiyy, df_unemp, df_btc, df_fng, df_etf, df_ixic, df_gspc, df_dxy, df_gold, df_stcap, df_hash, df_fund]:
    df = safe_merge(df, x)
if df is None or df.empty:
    st.error("数据为空（可能网络受限）。"); st.stop()

# 统一索引 & 填充策略
df = df.sort_values("DATE")
full_idx = pd.date_range(df["DATE"].min(), df["DATE"].max(), freq="D", tz="UTC")
df = df.set_index("DATE").reindex(full_idx).rename_axis("DATE").reset_index()

num_cols = ["rate","cpi_yoy","unemp","btc","fng","etf_net","ixic","gspc","dxy","gold","stablecap","hashrate","funding"]
for c in num_cols:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

# 宏观：仅前向填充（不补0）
for c in ["rate","cpi_yoy","unemp","dxy","gold","ixic","gspc","stablecap","hashrate"]:
    if c in df.columns: df[c] = df[c].ffill()

# 市场价：前向填充
if "btc" in df: df["btc"] = df["btc"].ffill()
# F&G：原值；ETF：缺就缺（不要补 0）
# Funding：前向填充更合理（8h一次）
if "funding" in df: df["funding"] = df["funding"].ffill()

# ======================
# 指标计算
# ======================
# DriverIndex = 0.4*(-Z(rate)) + 0.4*(-Z(cpi_yoy)) + 0.2*(-Z(unemp))
z_rate  = rolling_z(df["rate"])    if "rate"   in df else pd.Series(index=df.index, dtype=float)
z_cpi   = rolling_z(df["cpi_yoy"]) if "cpi_yoy" in df else pd.Series(index=df.index, dtype=float)
z_unemp = rolling_z(df["unemp"])   if "unemp"  in df else pd.Series(index=df.index, dtype=float)

valid = z_rate.notna() & z_cpi.notna() & z_unemp.notna()
driver = pd.Series(np.nan, index=df.index, dtype=float)
driver[valid] = 0.4*(-z_rate[valid]) + 0.4*(-z_cpi[valid]) + 0.2*(-z_unemp[valid])
driver = ewma(driver, 2)  # 轻平滑

# 相关区标准化（尽量少平滑）
def z_of(col, log=False):
    if col not in df: return None
    s = df[col].replace(0, np.nan)
    if log: s = np.log(s)
    return ewma(rolling_z(s), 2)

z_btc   = z_of("btc", log=True)
z_fng   = z_of("fng", log=False)
z_ixic  = z_of("ixic", log=True)
z_gspc  = z_of("gspc", log=True)
z_dxy   = z_of("dxy",  log=False)
z_gold  = z_of("gold", log=True)
z_stcap = z_of("stablecap", log=True)
z_hash  = z_of("hashrate",  log=False)
funding = df["funding"] if "funding" in df else None
etf     = df["etf_net"] if "etf_net" in df else None

# 滚动相关性（BTC vs 其它）
def rolling_corr(a: pd.Series, b: pd.Series, win=60):
    return a.rolling(win, min_periods=win//3).corr(b)

corrs = {}
if z_btc is not None:
    for name, series in {
        "IXIC": z_ixic, "GSPC": z_gspc, "DXY": z_dxy, "Gold": z_gold,
        "Stablecap": z_stcap, "Hashrate": z_hash, "F&G": z_fng
    }.items():
        if series is not None:
            corrs[name] = rolling_corr(z_btc, series, 60)

# 展示区间
with st.sidebar:
    st.header("显示设置")
    years = st.slider("显示年数", 1, 5, 2)
    mode  = st.selectbox("布局", ["桌面","手机"], index=0)
    H1, H2 = (260, 260) if mode=="桌面" else (180, 190)
    st.subheader("数据新鲜度")
    def last(d, name):
        if d is None or d.empty: st.write(f"• {name}: 无")
        else: st.write(f"• {name}: {pd.to_datetime(d['DATE']).max().date()}")
    for pair in [("rate","联邦基金利率"),("cpi_yoy","CPI同比"),("unemp","失业率"),("btc","BTC"),
                 ("fng","F&G"),("etf_net","ETF净流入"),("ixic","纳指"),("gspc","标普"),
                 ("dxy","美元指数"),("gold","黄金"),("stablecap","稳定币总市值"),("hashrate","算力"),("funding","资金费率")]:
        col, name = pair
        last(df[["DATE",col]].dropna() if col in df else None, name)

end_dt = pd.Timestamp.utcnow().normalize()
start_dt = end_dt - pd.DateOffset(years=years)
mask = (df["DATE"] >= start_dt) & (df["DATE"] <= end_dt)
D = df.loc[mask, "DATE"]

# ======================
# 绘图（Altair，随 Streamlit 内置，无需额外安装）
# ======================
def plot_altair():
    import altair as alt
    autosize = {"type":"fit","contains":"padding"}
    # 颜色：荧光绿系
    C_DRIVER = "#22C786"  # 主驱动线（荧光绿）
    C_BTC    = "#CFFFB0"  # 亮荧光线
    C_FNG    = "#91FBB2"  # 次荧光
    C_IDX    = "#55E6A5"  # 指数族
    C_DXY    = "#20E3B2"  # 美元
    C_GOLD   = "#8AF5AD"  # 金
    C_STABLE = "#7CF7B2"  # 稳定币
    C_HASH   = "#66E6B0"  # 算力
    C_NEG    = "#0F3D3E"  # 负柱（深青）

    # 上半区：DriverIndex（线+轻填充）
    d1 = pd.DataFrame({"DATE":D, "DriverIndex": driver.loc[mask].values})
    base = alt.Chart(d1).encode(x=alt.X("DATE:T", axis=alt.Axis(title=None, grid=True, gridColor="#1e2a36"))).properties(width="container")
    area = base.mark_area(opacity=0.18, color=C_DRIVER).encode(y="DriverIndex:Q")
    line = base.mark_line(strokeWidth=2, color=C_DRIVER).encode(y="DriverIndex:Q",
        tooltip=[alt.Tooltip("DATE:T","日期"), alt.Tooltip("DriverIndex:Q",".2f")])
    upper = (area + line).properties(height=H1, title="宏观驱动：DriverIndex = 0.4*(-Z利率)+0.4*(-Z通胀同比)+0.2*(-Z失业率)")

    # 下半区：多曲线 + ETF 柱 + Funding 细线（独立右轴）
    series = {
        "BTC(z)": z_btc.loc[mask].values if z_btc is not None else None,
        "F&G(z)": z_fng.loc[mask].values if z_fng is not None else None,
        "IXIC(z)": z_ixic.loc[mask].values if z_ixic is not None else None,
        "GSPC(z)": z_gspc.loc[mask].values if z_gspc is not None else None,
        "DXY(z)": z_dxy.loc[mask].values  if z_dxy  is not None else None,
        "Gold(z)": z_gold.loc[mask].values if z_gold is not None else None,
        "Stablecap(z)": z_stcap.loc[mask].values if z_stcap is not None else None,
        "Hashrate(z)": z_hash.loc[mask].values if z_hash is not None else None,
    }
    series = {k:v for k,v in series.items() if v is not None}
    d2 = pd.DataFrame({"DATE": D, **series})
    d2m = d2.melt("DATE", var_name="series", value_name="value")

    color_scale = alt.Scale(
        domain=list(series.keys()),
        range=[C_BTC, C_FNG, C_IDX, C_IDX, C_DXY, C_GOLD, C_STABLE, C_HASH][:len(series)]
    )
    lines = alt.Chart(d2m).mark_line(strokeWidth=2).encode(
        x="DATE:T",
        y=alt.Y("value:Q", axis=alt.Axis(title=None, grid=True, gridColor="#1e2a36")),
        color=alt.Color("series:N", scale=color_scale),
        opacity=alt.value(0.95),
        tooltip=["DATE:T","series:N", alt.Tooltip("value:Q",".2f")]
    ).properties(width="container", height=H2)

    # ETF 柱（正=荧光绿，负=深青）
    d3 = None
    if etf is not None:
        d3 = pd.DataFrame({"DATE": D, "etf": etf.loc[mask].values})
        bars = alt.Chart(d3).mark_bar(opacity=0.75).encode(
            x="DATE:T",
            y=alt.Y("etf:Q", axis=alt.Axis(title="ETF净流入(USD)", grid=False)),
            color=alt.condition(alt.datum.etf >= 0, alt.value(C_DRIVER), alt.value(C_NEG)),
            tooltip=["DATE:T", alt.Tooltip("etf:Q", ".2s")]
        )
    else:
        bars = None

    # Funding 细线（次轴）
    fund_layer = None
    if funding is not None:
        d4 = pd.DataFrame({"DATE": D, "funding": funding.loc[mask].values})
        fund_layer = alt.Chart(d4).mark_line(strokeWidth=1.2, strokeDash=[4,2], color="#A6F7C5").encode(
            x="DATE:T",
            y=alt.Y("funding:Q", axis=alt.Axis(title="Funding", grid=False))
        )

    lower = lines
    if bars is not None:
        lower = alt.layer(bars, lines).resolve_scale(y="independent")
    if fund_layer is not None:
        lower = alt.layer(lower, fund_layer).resolve_scale(y="independent")

    lower = lower.properties(title="相关性因子：多曲线(z) + ETF柱 + Funding(细线)")

    chart = alt.vconcat(upper, lower)\
             .configure_view(strokeOpacity=0.15)\
             .configure_axis(labelColor="#9FB0C0", grid=True)\
             .configure(autosize=autosize)

    st.altair_chart(chart, use_container_width=True)

try:
    plot_altair()
except Exception as e:
    st.error(f"绘图失败：{e}")

# 相关性热力条（简版）
st.markdown("#### BTC vs 其它因子：60D 滚动相关系数（简版热力）")
corr_df = pd.DataFrame({"DATE": df["DATE"]})
for k,v in corrs.items():
    corr_df[k] = v
corr_df = corr_df.loc[mask].set_index("DATE")
# 用 st.dataframe 展示（移动端可横滑）
st.dataframe(corr_df.tail(60).style.background_gradient(cmap="Greens"), use_container_width=True)

st.caption("数据源：FRED(DFF/CPIAUCSL/UNRATE)、CoinGecko(BTC)、Alternative.me(F&G)、Farside(ETF flows)、"
           "Yahoo Finance(^IXIC,^GSPC,DX-Y.NYB,GC=F)、DefiLlama(Stablecoins)、Blockchain.info(Hashrate)、Binance(Funding)。"
           "处理：日频对齐→滚动Z=90→EWMA=2（轻平滑）。DriverIndex=0.4×(-Z利率)+0.4×(-Z通胀同比)+0.2×(-Z失业率)。")