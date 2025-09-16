import streamlit as st
st.caption("THIS IS driver_corr_dashboard.py")

# driver_corr_dashboard.py
# Streamlit single-page: DriverIndex（宏观驱动） × 相关性因子（BTC/F&G/ETF）
import io
import math
import json
import time
import datetime as dt
from typing import Optional

import pandas as pd
import numpy as np
import requests
import streamlit as st

# ====== 基本配置 ======
st.set_page_config(page_title="BTC 驱动因子 × 相关性因子", layout="wide")
st.title("BTC 驱动因子 × 相关性因子（免 Token 公共数据）")

# 颜色（与你约定的 UI token）
COLOR_DRIVER = "#1F77B4"   # DriverIndex 深蓝
COLOR_BTC    = "#FF7F0E"   # BTC 亮橙
COLOR_FNG    = "#8C9C68"   # Fear&Greed 灰绿
COLOR_ETF_IN = "#2CA02C"
COLOR_ETF_OUT= "#D62728"

# ====== 工具函数：缓存 ======
@st.cache_data(ttl=3600, show_spinner=False)
def _get(url, headers=None, params=None, timeout=20):
    r = requests.get(url, headers=headers or {}, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_csv(series_id: str) -> pd.DataFrame:
    # FRED 免 token CSV：例如 https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    r = _get(url, params={"id": series_id})
    df = pd.read_csv(io.StringIO(r.text))
    # 列：DATE, <series_id>
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce", utc=True)
    df = df.rename(columns={series_id: "value"}).dropna()
    return df[["DATE", "value"]]

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_btc_from_coingecko(days: int = 1825) -> pd.DataFrame:
    # CoinGecko 免 token：/market_chart
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    r = _get(url, params={"vs_currency": "usd", "days": str(days), "interval": "daily"})
    data = r.json()
    prices = data.get("prices", [])  # [ [ms, price], ... ]
    if not prices:
        return pd.DataFrame(columns=["DATE", "close"])
    arr = np.array(prices)
    ts = pd.to_datetime(arr[:,0], unit="ms", utc=True)
    px = pd.to_numeric(arr[:,1], errors="coerce")
    df = pd.DataFrame({"DATE": ts, "close": px}).dropna()
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fear_greed() -> pd.DataFrame:
    # Alternative.me 免 token
    # https://api.alternative.me/fng/?limit=0&format=json
    url = "https://api.alternative.me/fng/"
    r = _get(url, params={"limit": "0", "format": "json"})
    data = r.json().get("data", [])
    if not data:
        return pd.DataFrame(columns=["DATE", "fng"])
    rows = []
    for d in data:
        # d: {"value":"64","value_classification":"Greed","timestamp":"1726099200","time_until_update":"..."}
        ts = pd.to_datetime(int(d.get("timestamp", "0")), unit="s", utc=True)
        val = pd.to_numeric(d.get("value", None), errors="coerce")
        rows.append((ts, val))
    df = pd.DataFrame(rows, columns=["DATE", "fng"]).dropna()
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_etf_flows() -> pd.DataFrame:
    """
    Farside 的 CSV 链接偶尔变动；这里做多源尝试，失败则回退为全 0。
    你之后若有稳定 URL，可直接替换 primary_url。
    """
    candidates = [
        # 常见历史路径（可能更换）
        "https://farside.co.uk/wp-content/uploads/bitcoin_etf_flows.csv",
        "https://www.farside.co.uk/wp-content/uploads/bitcoin_etf_flows.csv",
    ]
    text = None
    for url in candidates:
        try:
            r = _get(url, timeout=15)
            if r.status_code == 200 and "date" in r.text.lower():
                text = r.text
                break
        except Exception:
            continue

    if text is None:
        # 回退：生成空的日频 0 流入
        end = pd.Timestamp.utcnow().normalize()
        start = end - pd.Timedelta(days=365*2)
        idx = pd.date_range(start, end, freq="D", tz="UTC")
        return pd.DataFrame({"DATE": idx, "etf_netflow": np.zeros(len(idx))})

    df = pd.read_csv(io.StringIO(text))
    # 兼容列名：Date / date, Net Flow / net_flow / NetFlow...
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("day") or list(df.columns)[0]
    # 尝试寻找净流入列（不同表头命名不统一）
    flow_col = None
    for key in ["net flow", "net_flow", "netflow", "flow", "net"]:
        for c in df.columns:
            if key.lower() in c.lower():
                flow_col = c
                break
        if flow_col:
            break
    if flow_col is None:
        # 若没有净流入列，则把所有 ETF 单列相加（正负）
        numeric_cols = [c for c in df.columns if c != date_col]
        df["_sum"] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        flow_col = "_sum"

    df["DATE"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df["etf_netflow"] = pd.to_numeric(df[flow_col], errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values("DATE")[["DATE", "etf_netflow"]]
    return df

# ====== 处理函数 ======
def to_daily(df: pd.DataFrame, date_col="DATE") -> pd.DataFrame:
    # 保证日频，去重取最后值
    if df.empty:
        return df
    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], utc=True)
    x = x.dropna(subset=[date_col]).sort_values(date_col)
    x["DATE"] = x[date_col].dt.floor("D")
    x = x.groupby("DATE").last().reset_index()
    return x

def rolling_zscore(s: pd.Series, win: int = 180) -> pd.Series:
    m = s.rolling(win, min_periods=max(20, win//6)).mean()
    sd = s.rolling(win, min_periods=max(20, win//6)).std(ddof=0)
    z = (s - m) / sd.replace(0, np.nan)
    return z

def ewma(s: pd.Series, span_days: int = 7) -> pd.Series:
    return s.ewm(span=span_days, adjust=False, min_periods=1).mean()

def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on="DATE") -> pd.DataFrame:
    if left is None or left.empty:
        return right
    if right is None or right.empty:
        return left
    return pd.merge(left, right, on=on, how="outer")

# ====== 拉取数据 ======
with st.spinner("拉取公共数据中（免 Token）..."):
    # 宏观：利率、CPI、失业率（FRED）
    df_dff   = to_daily(fetch_fred_csv("DFF"))         # 联邦基金利率（水平）
    df_cpi   = to_daily(fetch_fred_csv("CPIAUCSL"))    # CPI 总指数
    df_unemp = to_daily(fetch_fred_csv("UNRATE"))      # 失业率 %

    # CPI 同比（基于 CPIAUCSL 计算 YoY %）
    if not df_cpi.empty:
        tmp = df_cpi.set_index("DATE")["value"].astype(float)
        cpi_yoy = (tmp.pct_change(365) * 100).replace([np.inf, -np.inf], np.nan)
        df_cpi_yoy = cpi_yoy.reset_index().rename(columns={"value": "cpi_yoy", 0:"cpi_yoy"})
        df_cpi_yoy.columns = ["DATE", "cpi_yoy"]
    else:
        df_cpi_yoy = pd.DataFrame(columns=["DATE", "cpi_yoy"])

    # BTC（CoinGecko）
    df_btc = to_daily(fetch_btc_from_coingecko(days=1825))  # 近 5 年
    # Fear & Greed
    df_fng = to_daily(fetch_fear_greed())
    # ETF 净流入（Farside）
    df_etf = to_daily(fetch_etf_flows())

# ====== 对齐 & 计算 DriverIndex ======
# 合并为大表
df = None
df = safe_merge(df, df_dff.rename(columns={"value": "rate"}))
df = safe_merge(df, df_cpi_yoy)
df = safe_merge(df, df_unemp.rename(columns={"value": "unemp"}))
df = safe_merge(df, df_btc.rename(columns={"close": "btc"}))
df = safe_merge(df, df_fng)
df = safe_merge(df, df_etf)
if df is None or df.empty:
    st.error("数据为空，可能是网络受限或上游变更。")
    st.stop()

df = df.sort_values("DATE").reset_index(drop=True)

# 缺失填充（前向填充→再填 0），再日频完整索引
full_idx = pd.date_range(df["DATE"].min(), df["DATE"].max(), freq="D", tz="UTC")
df = df.set_index("DATE").reindex(full_idx).rename_axis("DATE").reset_index()
for col in ["rate", "cpi_yoy", "unemp", "btc", "fng", "etf_netflow"]:
    if col in df.columns:
        df[col] = df[col].astype(float)
df[["rate","cpi_yoy","unemp"]] = df[["rate","cpi_yoy","unemp"]].ffill()
df[["btc","fng","etf_netflow"]] = df[["btc","fng","etf_netflow"]].fillna(0.0)

# 计算滚动 Z 分数并方向取反（利率、通胀、失业率）
z_rate   = rolling_zscore(df["rate"])
z_cpi    = rolling_zscore(df["cpi_yoy"])
z_unemp  = rolling_zscore(df["unemp"])
driver_raw = 0.4*(-z_rate) + 0.4*(-z_cpi) + 0.2*(-z_unemp)
driver = ewma(driver_raw, span_days=7)

# 相关区：标准化后的 BTC、F&G、ETF（柱）
z_btc  = ewma(rolling_zscore(np.log(df["btc"].replace(0, np.nan))), 7)  # 用 log 价格更稳
z_fng  = ewma(rolling_zscore(df["fng"]), 7)
# ETF 柱不做 zscore，保留真实"净流入/流出"量级的直觉；可选：做 z 后配色
etf = df["etf_netflow"].fillna(0.0)

# 裁剪展示区间（侧边栏）
with st.sidebar:
    st.header("显示设置")
    years = st.slider("显示年数", min_value=1, max_value=5, value=3, step=1)
    end_dt = pd.Timestamp.utcnow().normalize()
    start_dt = end_dt - pd.DateOffset(years=years)
    # 取该区间
mask = (df["DATE"] >= start_dt) & (df["DATE"] <= end_dt)
D = df.loc[mask, "DATE"]

# ====== 绘图（Altair，若缺失则回退 st.line_chart） ======
def plot_altair():
    import altair as alt
    # 上半区：DriverIndex（线 + 填充）
    d1 = pd.DataFrame({"DATE": D, "DriverIndex": driver.loc[mask].values})
    base = alt.Chart(d1).encode(x=alt.X("DATE:T", axis=alt.Axis(title=None, grid=True)))
    area = base.mark_area(opacity=0.25).encode(y="DriverIndex:Q", color=alt.value(COLOR_DRIVER))
    line = base.mark_line(strokeWidth=2).encode(y="DriverIndex:Q", color=alt.value(COLOR_DRIVER))
    upper = (area + line).properties(height=260, title="宏观驱动：DriverIndex（-Z利率/-Z通胀/-Z失业）")

    # 下半区：BTC/F&G 折线 + ETF 柱
    d2 = pd.DataFrame({
        "DATE": D,
        "BTC(z)": z_btc.loc[mask].values,
        "F&G(z)": z_fng.loc[mask].values,
    })
    d2m = d2.melt("DATE", var_name="series", value_name="value")
    lines = alt.Chart(d2m).mark_line(strokeWidth=2).encode(
        x="DATE:T",
        y=alt.Y("value:Q", axis=alt.Axis(title=None, grid=True)),
        color=alt.Color("series:N", scale=alt.Scale(domain=["BTC(z)","F&G(z)"],
                                                   range=[COLOR_BTC, COLOR_FNG])),
        tooltip=["DATE:T","series:N","value:Q"]
    ).properties(height=240)

    d3 = pd.DataFrame({"DATE": D, "etf": etf.loc[mask].values})
    bars = alt.Chart(d3).mark_bar(opacity=0.6).encode(
        x="DATE:T",
        y=alt.Y("etf:Q", axis=alt.Axis(title=None)),
        color=alt.condition(alt.datum.etf >= 0, alt.value(COLOR_ETF_IN), alt.value(COLOR_ETF_OUT)),
        tooltip=["DATE:T","etf:Q"]
    )
    lower = alt.layer(bars, lines).resolve_scale(y="independent").properties(title="相关性因子：BTC(z)/F&G(z) + ETF净流入(柱)")

    chart = alt.vconcat(upper, lower).configure_axis(labelColor="#9FB0C0", grid=True).configure_view(strokeOpacity=0.15)
    st.altair_chart(chart, use_container_width=True)

try:
    plot_altair()
except Exception as e:
    st.warning(f"Altair 绘图失败，已回退为简易折线：{e}")
    sub = pd.DataFrame({"DATE": D, "DriverIndex": driver.loc[mask].values}).set_index("DATE")
    st.line_chart(sub)
    sub2 = pd.DataFrame({"DATE": D, "BTC(z)": z_btc.loc[mask].values, "F&G(z)": z_fng.loc[mask].values}).set_index("DATE")
    st.line_chart(sub2)
    st.bar_chart(pd.DataFrame({"DATE": D, "ETF": etf.loc[mask].values}).set_index("DATE"))

# 说明与数据来源
st.caption("数据源：FRED(DFF/CPIAUCSL/UNRATE CSV)、CoinGecko(BTC)、Alternative.me(Fear&Greed)、Farside(ETF 流入，若不可用则回退为 0)。\
处理：日频化→180D 滚动Z分数→7D EWMA 平滑；DriverIndex=0.4×(-Z利率)+0.4×(-Z通胀同比)+0.2×(-Z失业率)。")

