import streamlit as st
st.caption("THIS IS simple_dashboard.py")

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
from collections import defaultdict

# 模拟数据生成
def generate_mock_data():
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
               "ADA/USDT", "DOGE/USDT", "TRX/USDT", "AVAX/USDT", "DOT/USDT"]

    # 模拟价格数据
    base_prices = {
        "BTC/USDT": 65000, "ETH/USDT": 3200, "BNB/USDT": 590,
        "SOL/USDT": 140, "XRP/USDT": 0.52, "ADA/USDT": 0.45,
        "DOGE/USDT": 0.12, "TRX/USDT": 0.08, "AVAX/USDT": 28, "DOT/USDT": 6.5
    }

    quotes = []
    for symbol in symbols:
        base = base_prices[symbol]
        variation = np.random.uniform(-0.03, 0.03)  # ±3% 变化
        close = base * (1 + variation)
        high = close * (1 + abs(variation) * 0.5)
        low = close * (1 - abs(variation) * 0.5)

        quotes.append({
            "Symbol": symbol,
            "Close": f"{close:.6f}".rstrip("0").rstrip("."),
            "High": f"{high:.6f}".rstrip("0").rstrip("."),
            "Low": f"{low:.6f}".rstrip("0").rstrip("."),
            "Time": datetime.now().strftime("%H:%M:%S")
        })

    return quotes

def generate_mock_signals():
    strategies = ["vegas_tunnel", "chan_simplified", "macd"]
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
    sides = ["BUY", "SELL"]

    signals = []
    # 随机生成一些信号
    for _ in range(np.random.randint(2, 6)):
        symbol = np.random.choice(symbols)
        strategy = np.random.choice(strategies)
        side = np.random.choice(sides)

        base_price = {"BTC/USDT": 65000, "ETH/USDT": 3200, "SOL/USDT": 140, "XRP/USDT": 0.52}[symbol]
        entry = base_price * (1 + np.random.uniform(-0.02, 0.02))

        if side == "BUY":
            target = entry * 1.03
            stop = entry * 0.98
        else:
            target = entry * 0.97
            stop = entry * 1.02

        signals.append({
            "symbol": symbol,
            "strategy": strategy,
            "side": side,
            "entry": f"{entry:.6f}".rstrip("0").rstrip("."),
            "target": f"{target:.6f}".rstrip("0").rstrip("."),
            "stop": f"{stop:.6f}".rstrip("0").rstrip("."),
            "confidence": np.random.randint(30, 50),
            "tf": "4h",
            "eta_text": "≈4 小时",
            "ts": datetime.now(),
            "reason": f"建议单：{symbol}（4h）{'做多' if side=='BUY' else '做空'}；{strategy} 策略触发"
        })

    return signals

# 页面配置
st.set_page_config(page_title="熬鹰计划（4h/1d/1w）", layout="wide")
st.title("熬鹰计划（4h/1d/1w）")

# 自检探针
st.markdown("**自检（Probe）**：用于检查多策略是否被渲染覆盖。")

# 顶部状态栏
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.caption("心跳时间")
    st.write(time.strftime("%Y-%m-%d %H:%M:%S"))
with col2:
    st.caption("交易所/周期")
    tf = st.radio("选择周期", ["4h", "1d", "1w"], index=0, horizontal=True, label_visibility="collapsed")
    st.write(f"binance / {tf}")
with col3:
    st.caption("错误计数")
    st.write("0")
with col4:
    st.caption("监控币种数")
    st.write("10")

st.divider()

# 侧边栏设置
with st.sidebar:
    with st.expander("回测工具", expanded=False):
        st.caption("快回测会基于历史K线在本地快速评估策略胜率（不下单）。")
        look_map = {"4h": 12, "1d": 10, "1w": 8}
        default_look = look_map.get(tf, 12)
        la = st.number_input("向前看的K线数（lookahead）", min_value=4, max_value=60, value=default_look, step=1)
        run_bt = st.button("运行快回测")

    diag_mode = st.checkbox("诊断模式（显示 ADX / ATR% / 交叉等）", value=False)
    relax = st.checkbox("放松过滤（先出信号，后再收紧）", value=True)

    active_strats = st.multiselect(
        "启用策略",
        options=["vegas_tunnel", "chan_simplified", "macd", "sma_cross", "rsi_reversal"],
        default=["vegas_tunnel", "chan_simplified", "macd"]
    )

# 实时报价
st.subheader("💹 实时报价（仅 8 行，超出可滚动）")
quotes = generate_mock_data()
qdf = pd.DataFrame(quotes)
st.dataframe(qdf.head(8), height=240, use_container_width=True)

# 当前形态与推荐策略
st.subheader("🧭 当前形态与推荐策略（简版）")
st.caption("说明：此处按简单指标展示'震荡/趋势'倾向与建议策略名称，仅作演示。")

# 生成模拟信号
signals_all = generate_mock_signals()

# 探针检查
if signals_all:
    pairs = sorted({(s.get("symbol","?"), s.get("strategy","?")) for s in signals_all})
    st.info(f"Probe → 收到 {len(signals_all)} 条；唯一对数：{len(pairs)}")
    st.code("\n".join([f"{a} | {b}" for a,b in pairs]), language="text")

if len(quotes):
    sym_pick = st.selectbox("选择标的查看推荐", [q["Symbol"] for q in quotes])
    rec = [s["strategy"] for s in signals_all if s["symbol"] == sym_pick]
    rec = list(dict.fromkeys(rec))[:3] if rec else ["—"]
    st.write(f"**{sym_pick}** · 推荐：", ", ".join(rec))

# 当前信号
st.subheader("🔔 当根信号（收盘确认）")
if not signals_all:
    st.info("当根无触发。")
else:
    # 按symbol分组显示信号
    by_sym = defaultdict(list)
    for s in signals_all:
        by_sym[s.get("symbol","—")].append(s)

    for sym, items in sorted(by_sym.items()):
        st.markdown(f"### {sym}")
        for s in items:
            side = s.get("side","—")
            title = f"{side} {sym}"
            st.markdown(f"**{title}** ｜ 策略：`{s.get('strategy','—')}`")
            st.caption(f"信心 {s.get('confidence','—')}｜周期：{s.get('tf','—')}｜时间：{s.get('ts','—')}")
            st.write(f"入场：{s.get('entry')} ｜ 目标：{s.get('target')} ｜ 止损：{s.get('stop')} ｜ ETA：{s.get('eta_text','—')}")
            if s.get("reason"):
                st.write(s["reason"])
        st.divider()

# 刷新按钮
if st.button("🔄 刷新数据"):
    st.rerun()

# 页脚信息
st.markdown("---")
st.caption("本面板为演示版本，数据为模拟生成。实际使用需要配置真实的API密钥。")
