import streamlit as st
st.caption("THIS IS macro_btc_monitor_v3.py")

# macro_btc_monitor_v3.py
# Python 3.8, 仅用：streamlit / pandas / numpy / requests / pandas_ta / dateutil
# 多因子：DriverIndex、BTC(z)、F&G、ETF净流入、Funding、^IXIC、^GSPC、DXY、Gold、
# 稳定币总市值、Hashrate、10Y/2Y 收益率及其利差
# 公开数据：FRED / Farside / CoinGecko / Alternative.me / Yahoo Finance / DefiLlama / Blockchain.info / Binance

import io, re, time
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ========= 主题 & 页面 =========
st.set_page_config(page_title="BTC 多因子宏观监控（v3）", layout="wide")
st.markdown("""
<style>
:root { --fg:#CFFFB0; --fg2:#91FBB2; --fg3:#55E6A5; --fg4:#22C786; --neg:#0F3D3E; --grid:#1e2a36; }
html,body { background:#0B0F14; color:#E6EDF6; }
section[data-testid="stSidebar"] { background:#0D141B; }
hr{border:0;border-top:1px solid #14202a;}
</style>
""", unsafe_allow_html=True)
st.title("BTC 多因子宏观监控（v3）")

# ========= 网络设置（若用 Clash，确认端口 7890）=========
HEADERS = {"User-Agent":"Mozilla/5.0"}
PROXIES = {
    "http":"http://127.0.0.1:7890",
    "https":"http://127.0.0.1:7890"
}

@st.cache_data(ttl=3600, show_spinner=False)
def _get_json(url, params=None, timeout=30):
    r = requests.get(url, params=params or {}, headers=HEADERS, timeout=timeout, proxies=PROXIES)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=3600, show_spinner=False)
def _get_text(url, params=None, timeout=30):
    r = requests.get(url, params=params or {}, headers=HEADERS, timeout=timeout, proxies=PROXIES)
    r.raise_for_status()
    return r.text

def to_daily(df: pd.DataFrame, date_col="DATE"):
    if df is None or df.empty: return df
    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], utc=True, errors="coerce")
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

# ========= 数据源 =========

# FRED: 通用 CSV
@st.cache_data(ttl=3600, show_spinner=False)
def fred_series(series_id: str) -> pd.DataFrame:
    txt = _get_text("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id":series_id}, timeout=30)
    # 若失败会是 HTML，这里兜底
    if not txt.strip().upper().startswith("DATE,"):
        return pd.DataFrame(columns=["DATE","value"])
    df = pd.read_csv(io.StringIO(txt))
    if "DATE" not in df.columns or series_id not in df.columns:
        return pd.DataFrame(columns=["DATE","value"])
    df["DATE"] = pd.to_datetime(df["DATE"], utc=True, errors="coerce")
    df = df.rename(columns={series_id:"value"}).dropna()
    return df[["DATE","value"]]

# CPI 同比（基于月度 CPIAUCSL 计算）
@st.cache_data(ttl=3600, show_spinner=False)
def fred_cpi_yoy():
    m = fred_series("CPIAUCSL")
    if m.empty: return pd.DataFrame(columns=["DATE","cpi_yoy"])
    m["DATE"] = m["DATE"].dt.to_period("M").dt.to_timestamp("M", tz="UTC")
    m = m.drop_duplicates("DATE").sort_values("DATE").rename(columns={"value":"cpi"})
    m["cpi_yoy"] = m["cpi"].pct_change(12) * 100.0
    return to_daily(m[["DATE","cpi_yoy"]]).ffill()

# CoinGecko: BTC 日线
@st.cache_data(ttl=3600, show_spinner=False)
def coingecko_btc(days=1825):
    js = _get_json("https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
                   params={"vs_currency":"usd","days":str(days),"interval":"daily"})
    arr = js.get("prices", [])
    if not arr: return pd.DataFrame(columns=["DATE","btc"])
    a = np.array(arr)
    return pd.DataFrame({"DATE":pd.to_datetime(a[:,0], unit="ms", utc=True), "btc":pd.to_numeric(a[:,1])})

# Alternative.me: Fear & Greed
@st.cache_data(ttl=3600, show_spinner=False)
def alt_fng():
    js = _get_json("https://api.alternative.me/fng/", params={"limit":"0","format":"json"})
    data = js.get("data", [])
    rows = [(pd.to_datetime(int(d["timestamp"]), unit="s", utc=True), float(d["value"])) for d in data]
    return pd.DataFrame(rows, columns=["DATE","fng"])

# Farside: ETF flows
@st.cache_data(ttl=3600, show_spinner=False)
def farside_etf():
    for u in [
        "https://farside.co.uk/wp-content/uploads/bitcoin_etf_flows.csv",
        "https://www.farside.co.uk/wp-content/uploads/bitcoin_etf_flows.csv",
    ]:
        try:
            t = _get_text(u, timeout=30)
            if "date" in t.lower():
                raw = t.replace("\ufeff","").replace("$","").replace("£","").replace(",","")
                raw = re.sub(r"\(([^)]+)\)", r"-\1", raw)
                df = pd.read_csv(io.StringIO(raw))
                # 找日期列
                dcol = next((c for c in df.columns if str(c).lower().strip() in ("date","day")), df.columns[0])
                # 找净流入列（否则求和）
                fcol = None
                for key in ["net flow","net_flow","netflow","net"]:
                    for c in df.columns:
                        if key in str(c).lower().strip(): fcol=c; break
                    if fcol: break
                if fcol is None:
                    num_cols = [c for c in df.columns if c != dcol]
                    df["_sum"] = df[num_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
                    fcol = "_sum"
                df["DATE"] = pd.to_datetime(df[dcol], utc=True, errors="coerce")
                df["etf_net"] = pd.to_numeric(df[fcol], errors="coerce")
                return df.dropna(subset=["DATE"])[["DATE","etf_net"]].sort_values("DATE")
        except Exception:
            continue
    return pd.DataFrame(columns=["DATE","etf_net"])

# Yahoo Finance v8: 指数/大宗
@st.cache_data(ttl=3600, show_spinner=False)
def yahoo_daily(symbol: str, days=1825):
    now = int(time.time()); period1 = now - days*24*3600
    js = _get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                   params={"period1":str(period1),"period2":str(now),"interval":"1d","includePrePost":"false"})
    try:
        res = js["chart"]["result"][0]
        ts  = res["timestamp"]
        close = res["indicators"]["quote"][0]["close"]
    except Exception:
        return pd.DataFrame(columns=["DATE","close"])
    s = pd.Series(close, index=pd.to_datetime(ts, unit="s", utc=True)).dropna()
    return pd.DataFrame({"DATE":s.index, "close":s.values})

# 稳定币总市值
@st.cache_data(ttl=3600, show_spinner=False)
def defillama_stablecap():
    js = _get_json("https://stablecoins.llama.fi/stablecoincharts/all")
    arr = js.get("total", [])
    rows = [(pd.to_datetime(x["date"], unit="s", utc=True), float(x["totalCirculatingUSD"])) for x in arr]
    return pd.DataFrame(rows, columns=["DATE","stablecap"])

# 算力
@st.cache_data(ttl=3600, show_spinner=False)
def blockchain_hashrate():
    js = _get_json("https://api.blockchain.info/charts/hash-rate",
                   params={"timespan":"5years","format":"json","sampled":"true"})
    vals = js.get("values", [])
    rows = [(pd.to_datetime(v["x"], unit="s", utc=True), float(v["y"])) for v in vals]
    return pd.DataFrame(rows, columns=["DATE","hashrate"])

# 资金费率
@st.cache_data(ttl=3600, show_spinner=False)
def binance_funding(days=365):
    end = int(time.time()*1000)
    start = end - days*24*3600*1000
    js = _get_json("https://fapi.binance.com/fapi/v1/fundingRate",
                   params={"symbol":"BTCUSDT","startTime":start,"endTime":end,"limit":1000})
    rows = [(pd.to_datetime(int(x["fundingTime"]), unit="ms", utc=True), float(x["fundingRate"])) for x in js]
    return pd.DataFrame(rows, columns=["DATE","funding"])

# ========= 拉取 & 合并 =========
with st.spinner("拉取公开数据中…"):
    # 宏观：利率/通胀/失业率
    dff     = to_daily(fred_series("DFF").rename(columns={"value":"rate"}))
    cpiyoy  = to_daily(fred_cpi_yoy())            # cpi_yoy
    unrate  = to_daily(fred_series("UNRATE").rename(columns={"value":"unemp"}))
    dgs10   = to_daily(fred_series("DGS10").rename(columns={"value":"us10y"}))
    dgs2    = to_daily(fred_series("DGS2").rename(columns={"value":"us2y"}))

    # 市场
    btc     = to_daily(coingecko_btc(1825)).rename(columns={"btc":"btc"})
    fng     = to_daily(alt_fng())
    etf     = to_daily(farside_etf())
    ixic    = to_daily(yahoo_daily("^IXIC")).rename(columns={"close":"ixic"})
    gspc    = to_daily(yahoo_daily("^GSPC")).rename(columns={"close":"gspc"})
    dxy     = to_daily(yahoo_daily("DX-Y.NYB")).rename(columns={"close":"dxy"})
    gold    = to_daily(yahoo_daily("GC=F")).rename(columns={"close":"gold"})
    stcap   = to_daily(defillama_stablecap())
    hashp   = to_daily(blockchain_hashrate())
    fund    = to_daily(binance_funding())

frames = [dff, cpiyoy, unrate, dgs10, dgs2, btc, fng, etf, ixic, gspc, dxy, gold, stcap, hashp, fund]
df = None
for x in frames:
    df = safe_merge(df, x)
if df is None or df.empty:
    st.error("数据为空（可能网络/代理有问题）。"); st.stop()

df = df.sort_values("DATE")
full_idx = pd.date_range(df["DATE"].min(), df["DATE"].max(), freq="D", tz="UTC")
df = df.set_index("DATE").reindex(full_idx).rename_axis("DATE").reset_index()

# 数值列
cols = ["rate","cpi_yoy","unemp","us10y","us2y","btc","fng","etf_net","ixic","gspc","dxy","gold","stablecap","hashrate","funding"]
for c in cols:
    if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")

# 填充策略
for c in ["rate","cpi_yoy","unemp","us10y","us2y","ixic","gspc","dxy","gold","stablecap","hashrate"]:
    if c in df: df[c] = df[c].ffill()
if "btc" in df: df["btc"] = df["btc"].ffill()
if "funding" in df: df["funding"] = df["funding"].ffill()
# ETF 缺就缺（不补 0）

# ========== 指标计算 ==========
# DriverIndex：0.4*(-Z(rate)) + 0.4*(-Z(cpi_yoy)) + 0.2*(-Z(unemp))
z_rate  = rolling_z(df["rate"])    if "rate"   in df else pd.Series(index=df.index, dtype=float)
z_cpi   = rolling_z(df["cpi_yoy"]) if "cpi_yoy" in df else pd.Series(index=df.index, dtype=float)
z_unemp = rolling_z(df["unemp"])   if "unemp"  in df else pd.Series(index=df.index, dtype=float)
valid = z_rate.notna() & z_cpi.notna() & z_unemp.notna()
driver = pd.Series(np.nan, index=df.index, dtype=float)
driver[valid] = 0.4*(-z_rate[valid]) + 0.4*(-z_cpi[valid]) + 0.2*(-z_unemp[valid])
driver = ewma(driver, 2)

# Z 序列（轻平滑，保留波动）
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

# 收益率利差（10Y-2Y）
spread = None
if "us10y" in df and "us2y" in df:
    spread = df["us10y"] - df["us2y"]

# ========== UI 控件 ==========
with st.sidebar:
    st.header("显示设置")
    rng = st.radio("区间", ["90D","1Y","2Y","3Y"], index=2, horizontal=True)
    years = {"90D":0.25, "1Y":1, "2Y":2, "3Y":3}[rng]
    mode  = st.selectbox("布局", ["桌面","手机"], index=0)
    H1, H2, H3 = ((260, 240, 220) if mode=="桌面" else (170, 180, 170))

    st.subheader("数据新鲜度")
    def last(col, name):
        if col not in df or df[col].dropna().empty: st.write(f"• {name}: 无"); return
        st.write(f"• {name}: {pd.to_datetime(df['DATE'][df[col].last_valid_index()]).date()}")
    for c, n in [("btc","BTC"), ("fng","F&G"), ("etf_net","ETF流入"), ("funding","资金费率"),
                 ("ixic","纳指"), ("gspc","标普"), ("dxy","美元指数"), ("gold","黄金"),
                 ("stablecap","稳定币总市值"), ("hashrate","算力"),
                 ("rate","联邦基金利率"), ("cpi_yoy","CPI同比"), ("unemp","失业率"),
                 ("us10y","美债10Y"), ("us2y","美债2Y")]:
        last(c, n)

end_dt = pd.Timestamp.utcnow().normalize()
start_dt = end_dt - pd.DateOffset(years=years)
mask = (df["DATE"] >= start_dt) & (df["DATE"] <= end_dt)
D = df.loc[mask, "DATE"]

# ========== 绘图 ==========
def draw_all():
    import altair as alt
    autosize = {"type":"fit","contains":"padding"}

    # 颜色
    C_DRIVER = "#22C786"; C_BTC = "#CFFFB0"; C_FNG = "#91FBB2"
    C_IDX = "#55E6A5"; C_DXY = "#20E3B2"; C_GOLD="#8AF5AD"
    C_STABLE="#7CF7B2"; C_HASH="#66E6B0"; C_NEG="#0F3D3E"

    # --- 顶部：DriverIndex ---
    d1 = pd.DataFrame({"DATE":D, "DriverIndex": driver.loc[mask].values})
    base = alt.Chart(d1).encode(x=alt.X("DATE:T", axis=alt.Axis(title=None, grid=True, gridColor="#1e2a36"))).properties(width="container")
    upper = base.mark_area(opacity=0.18, color=C_DRIVER).encode(y="DriverIndex:Q") + \
            base.mark_line(strokeWidth=2, color=C_DRIVER).encode(
                y="DriverIndex:Q", tooltip=["DATE:T", alt.Tooltip("DriverIndex:Q",".2f")]
            )
    upper = upper.properties(height=H1, title="DriverIndex ＝ 0.4×(-Z利率) + 0.4×(-Z通胀同比) + 0.2×(-Z失业率)")

    # --- 中部：相关因子多线（默认少量，避免糊） ---
    series = {
        "BTC(z)": z_btc.loc[mask].values if z_btc is not None else None,
        "F&G(z)": z_fng.loc[mask].values if z_fng is not None else None,
        "Stablecap(z)": z_stcap.loc[mask].values if z_stcap is not None else None,
        "IXIC(z)": z_ixic.loc[mask].values if z_ixic is not None else None,
        "GSPC(z)": z_gspc.loc[mask].values if z_gspc is not None else None,
        "DXY(z)": z_dxy.loc[mask].values if z_dxy is not None else None,
        "Gold(z)": z_gold.loc[mask].values if z_gold is not None else None,
        "Hashrate(z)": z_hash.loc[mask].values if z_hash is not None else None,
    }
    series = {k:v for k,v in series.items() if v is not None}
    d2 = pd.DataFrame({"DATE": D, **series}).melt("DATE", var_name="series", value_name="value")
    color_scale = alt.Scale(
        domain=list(series.keys()),
        range=[C_BTC, C_FNG, C_STABLE, C_IDX, C_IDX, C_DXY, C_GOLD, C_HASH][:len(series)]
    )
    mid = alt.Chart(d2).mark_line(strokeWidth=1.8).encode(
        x="DATE:T",
        y=alt.Y("value:Q", axis=alt.Axis(title=None, grid=True, gridColor="#1e2a36")),
        color=alt.Color("series:N", scale=color_scale),
        tooltip=["DATE:T","series:N", alt.Tooltip("value:Q",".2f")]
    ).properties(width="container", height=H2, title="相关因子（z 分数，轻平滑）")

    # --- 底部：ETF 柱 + Funding 次轴 +（可选）利差线 ---
    layers = []
    if "etf_net" in df:
        d3 = pd.DataFrame({"DATE": D, "etf": df.loc[mask, "etf_net"].values})
        bars = alt.Chart(d3).mark_bar(opacity=0.75).encode(
            x="DATE:T",
            y=alt.Y("etf:Q", axis=alt.Axis(title="ETF净流入(USD)")),
            color=alt.condition(alt.datum.etf >= 0, alt.value(C_DRIVER), alt.value(C_NEG)),
            tooltip=["DATE:T", alt.Tooltip("etf:Q",".2s")]
        )
        layers.append(bars)

    if "funding" in df:
        d4 = pd.DataFrame({"DATE": D, "funding": df.loc[mask, "funding"].values})
        fund_line = alt.Chart(d4).mark_line(strokeWidth=1.2, strokeDash=[4,2], color="#A6F7C5").encode(
            x="DATE:T",
            y=alt.Y("funding:Q", axis=alt.Axis(title="Funding", grid=False))
        )
        layers.append(fund_line)

    if spread is not None:
        d5 = pd.DataFrame({"DATE": D, "spread": spread.loc[mask].values})
        spr_line = alt.Chart(d5).mark_line(strokeWidth=1.2, color="#B9F7C2").encode(
            x="DATE:T",
            y=alt.Y("spread:Q", axis=alt.Axis(title="10Y-2Y(%)", grid=False))
        )
        layers.append(spr_line)

    if layers:
        lower = alt.layer(*layers).resolve_scale(y="independent").properties(
            width="container", height=H3, title="流入与利差：ETF柱 / Funding(次轴 虚线) / 10Y-2Y(次轴)"
        )
    else:
        lower = alt.Chart(pd.DataFrame({"DATE":[], "v":[]})).mark_line().properties(height=H3)

    chart = alt.vconcat(upper, mid, lower)\
        .configure_view(strokeOpacity=0.15)\
        .configure_axis(labelColor="#9FB0C0", grid=True)\
        .configure(autosize=autosize)

    st.altair_chart(chart, use_container_width=True)

try:
    draw_all()
except Exception as e:
    st.error(f"绘图失败：{e}")

# 简版相关系数表（60D）
st.markdown("#### BTC vs 其它因子：60D滚动相关系数（简表）")
def rolling_corr(a: pd.Series, b: pd.Series, win=60):
    return a.rolling(win, min_periods=max(10, win//3)).corr(b)

corrs = {}
if z_btc is not None:
    for name, series in {
        "IXIC": z_ixic, "GSPC": z_gspc, "DXY": z_dxy, "Gold": z_gold,
        "Stablecap": z_stcap, "Hashrate": z_hash, "F&G": z_fng
    }.items():
        if series is not None:
            corrs[name] = rolling_corr(z_btc, series, 60)

corr_df = pd.DataFrame({"DATE": df["DATE"]})
for k,v in corrs.items():
    corr_df[k] = v
mask2 = (corr_df["DATE"] >= start_dt) & (corr_df["DATE"] <= end_dt)
st.dataframe(corr_df.loc[mask2].set_index("DATE").tail(60).style.background_gradient(cmap="Greens"),
             use_container_width=True)

st.caption("数据源：FRED(DFF/CPIAUCSL/UNRATE/DGS10/DGS2)、CoinGecko(BTC)、Alternative.me(F&G)、"
           "Farside(ETF flows)、Yahoo Finance(^IXIC,^GSPC,DX-Y.NYB,GC=F)、DefiLlama(Stablecoins)、"
           "Blockchain.info(Hashrate)、Binance(Funding)。处理：日频对齐→滚动Z(90)→EWMA(2)。")

