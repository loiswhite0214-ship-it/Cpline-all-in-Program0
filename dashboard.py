# --- MUST be top-two lines ---
import streamlit as st
st.set_page_config(page_title="Cpline · Crypto Dashboard", layout="wide", initial_sidebar_state="collapsed")


def main():
    # === All imports AFTER set_page_config ===
    import json
    from pathlib import Path
    import time
    import ccxt
    import pandas as pd
    import numpy as np
    from collections import defaultdict
    from datetime import datetime, timezone

    # your modules
    from utils import fmt_price, to_ohlcv_df
    from strategies import STRATEGY_REGISTRY, strategy_diag, set_relax_mode
    from strategies_top15 import REGISTRY as TOP15_REGISTRY

    # ===== Merge strategies (safe) =====
    try:
        STRATEGY_REGISTRY.update(TOP15_REGISTRY)
    except Exception:
        pass

    # ===== Quick Backtest =====
    def _first_hit_future(df_slice, start_i, lookahead, entry, target, stop, side):
        highs = df_slice["high"].astype(float).iloc[start_i + 1 : start_i + 1 + lookahead].tolist()
        lows  = df_slice["low"].astype(float).iloc[start_i + 1 : start_i + 1 + lookahead].tolist()
        if side == "BUY":
            for h, l in zip(highs, lows):
                if l <= stop:  return "SL"
                if h >= target:return "TP"
        else:  # SELL
            for h, l in zip(highs, lows):
                if h >= stop:  return "SL"
                if l <= target:return "TP"
        return None

    def backtest_symbol_with_strategies(df_full: pd.DataFrame, tf: str, enabled: list, symbol: str, lookahead=12):
        results = []
        wins = losses = opens = 0
        warmup = 220  # cover EMA200 etc.
        for i in range(max(warmup, 60), len(df_full) - 1):
            df_slice = df_full.iloc[: i + 1].copy()
            for name in enabled:
                fn = STRATEGY_REGISTRY.get(name)
                if not fn:
                    continue
                sig = None
                try:
                    sig = fn(symbol, df_slice, tf)
                except Exception:
                    sig = None
                if not sig:
                    continue
                entry  = float(sig["entry"]) if sig.get("entry") is not None else float(df_slice["close"].iloc[-1])
                target = float(sig["target"])
                stop   = float(sig["stop"])
                side   = sig.get("side")
                outcome = _first_hit_future(df_full, i, lookahead, entry, target, stop, side)
                if outcome == "TP": wins += 1
                elif outcome == "SL": losses += 1
                else: opens += 1
                results.append({
                    "Time": str(df_slice["ts"].iloc[-1]),
                    "TF": tf, "Strategy": name, "Side": side,
                    "Entry": entry, "Target": target, "Stop": stop,
                    "Outcome": outcome or "None",
                    "R/R": 1 if outcome == "TP" else (-1 if outcome == "SL" else None)
                })
        total = wins + losses
        winrate = (wins / total * 100) if total > 0 else 0.0
        valid_R = [r["R/R"] for r in results if r["R/R"] is not None]
        avg_r = float(np.nan) if not valid_R else np.nanmean(valid_R)
        summary = {"win%": round(winrate, 2), "trades": total, "open_or_none": opens,
                   "avg_R": None if np.isnan(avg_r) else round(float(avg_r), 3)}
        return results, summary
    # ===== /Quick Backtest =====

    # ===== Probe（覆盖自检） =====
    def probe_signals_from_list(signals: list):
        try:
            pairs = sorted({(s.get("symbol","?"), s.get("strategy","?")) for s in signals})
            st.info(f"Probe → 收到 {len(signals)} 条；唯一对数：{len(pairs)}")
            st.code("\n".join([f"{a} | {b}" for a, b in pairs]), language="text")
            cnt = defaultdict(int)
            for _, strat in pairs:
                cnt[strat] += 1
            st.caption("每个策略出现次数（全局）：" + ", ".join([f"{k}:{v}" for k, v in sorted(cnt.items())]))
            by_sym = defaultdict(set)
            for sym, strat in pairs:
                by_sym[sym].add(strat)
            offenders = [sym for sym, s in by_sym.items() if len(s) <= 1]
            if offenders:
                st.warning("这些 symbol 仅有 1 条策略结果（或被覆盖）： " + ", ".join(offenders))
        except Exception:
            pass

    # ===== Timeframe helpers =====
    TF_SEC = {"4h": 4*3600, "1d": 24*3600, "1w": 7*24*3600}

    def drop_unclosed_last_bar(df: pd.DataFrame, tf: str) -> pd.DataFrame:
        if not len(df):
            return df
        sec = TF_SEC.get((tf or "").lower())
        if not sec:
            return df
        last_ts = int(pd.to_datetime(df["ts"].iloc[-1]).tz_localize("UTC").timestamp())
        now = int(datetime.now(timezone.utc).timestamp())
        return df.iloc[:-1].copy() if (now - last_ts) < sec else df

    def last_closed_index(df: pd.DataFrame, tf: str) -> int:
        return len(df) - 1 if len(df) else -1

    def signal_at(symbol: str, df: pd.DataFrame, tf: str, strat_name: str, i: int):
        if i < 0 or i >= len(df):
            return None
        view = df.iloc[: i + 1].copy()
        fn = STRATEGY_REGISTRY.get(strat_name)
        if not fn:
            return None
        try:
            sig = fn(symbol, view, tf)
            if sig:
                sig["strategy"] = strat_name
                sig["ts"] = pd.to_datetime(df["ts"].iloc[i])
                return sig
        except Exception:
            return None
        return None

    def latest_live_and_recent(symbol_to_df: dict, tf: str, strategies: list, lookahead: int):
        live_signals = []
        recent_window_signals = []
        for sym, df0 in symbol_to_df.items():
            df = drop_unclosed_last_bar(df0, tf)
            i_last = last_closed_index(df, tf)
            if i_last < 0:
                continue
            # 当根触发
            for strat in strategies:
                s = signal_at(sym, df, tf, strat, i_last)
                if s:
                    live_signals.append(s)
            # 近窗口逐根
            start = max(0, i_last - int(lookahead) + 1)
            for i in range(start, i_last + 1):
                for strat in strategies:
                    s = signal_at(sym, df, tf, strat, i)
                    if s:
                        recent_window_signals.append(s)
        # 去重并排序
        live_keys = {(s.get("symbol"), s.get("strategy"), s.get("ts")) for s in live_signals}
        recent_window_signals = [r for r in recent_window_signals
                                 if (r.get("symbol"), r.get("strategy"), r.get("ts")) not in live_keys]
        recent_window_signals.sort(key=lambda x: x.get("ts"), reverse=True)
        return live_signals, recent_window_signals

    def _render_cards(signals_list: list, bt_result_cache: dict, run_bt_flag: bool):
        _by_sym = defaultdict(list)
        for _s in signals_list:
            _by_sym[_s.get("symbol", "—")].append(_s)
        for _sym, _items in sorted(_by_sym.items()):
            st.markdown(f"### {_sym}")
            for s in sorted(_items, key=lambda x: (x.get("strategy",""), x.get("ts",""))):
                side = s.get("side","—")
                title = f"{side} {_sym}"
                st.markdown(f"**{title}** ｜ 策略：`{s.get('strategy','—')}`")
                st.caption(f"信心 {s.get('confidence','—')}｜周期：{s.get('tf','—')}｜时间：{s.get('ts','—')}")
                st.write(f"入场：{fmt_price(s.get('entry'))} ｜ 目标：{fmt_price(s.get('target'))} ｜ 止损：{fmt_price(s.get('stop'))} ｜ ETA：{s.get('eta_text','—')}")
                if s.get("reason"):
                    st.write(s["reason"])
                # 关联该 symbol 的快回测摘要
                bt_summ = bt_result_cache.get(_sym, [None, None])[1] if run_bt_flag else None
                if run_bt_flag and isinstance(bt_summ, dict) and "win%" in bt_summ:
                    st.caption(f"快回测：胜率 {bt_summ['win%']}%｜样本 {bt_summ['trades']}｜平均R {bt_summ.get('avg_R','—')}")
            st.divider()

    # ===== Config =====
    CFG_PATH = Path(__file__).with_name("config.json")
    USER_STATE_PATH = Path(__file__).with_name("user_state.json")
    try:
        cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    except Exception:
        cfg = {}

    PRODUCT = cfg.get("product_name", "熬鹰计划")
    EXCHANGE_NAME = cfg.get("exchange", "binance")
    SYMBOLS = (cfg.get("symbols", []) or [])[:15]
    TIMEFRAMES = cfg.get("timeframes", ["4h", "1d", "1w"])
    ENABLED = [s["name"] for s in (cfg.get("strategies", []) or []) if s.get("enabled")]
    ETA_TEXT = {"4h": "≈4 小时", "1d": "≈1 天", "1w": "≈1 周"}

    # ===== UI Top =====
    st.title(PRODUCT)
    st.caption("THIS IS dashboard.py")

    st.markdown("**自检（Probe）**：用于检查多策略是否被渲染覆盖。")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.caption("心跳时间")
        st.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    with col2:
        st.caption("交易所/周期")
        tf = st.radio("选择周期", TIMEFRAMES, index=0, horizontal=True, label_visibility="collapsed")
        st.write(f"{EXCHANGE_NAME} / {tf}")
    with col3:
        st.caption("错误计数")
        err_box = st.empty()
        err_box.write("0")
    with col4:
        st.caption("监控币种数")
        st.write(len(SYMBOLS))

    st.divider()

    with st.sidebar.expander("回测工具", expanded=False):
        st.caption("快回测会基于历史K线在本地快速评估策略胜率（不下单）。")
        look_map = {"4h": 12, "1d": 10, "1w": 8}
        default_look = look_map.get(tf, 12)
        la = st.number_input("向前看的K线数（lookahead）", min_value=4, max_value=60, value=default_look, step=1)
        run_bt = st.button("运行快回测")

    diag_mode = st.sidebar.checkbox("诊断模式（显示 ADX / ATR% / 交叉等）", value=False)
    relax = st.sidebar.checkbox("放松过滤（先出信号，后再收紧）", value=True)
    set_relax_mode(relax)

    # ===== User state (persist active strategies) =====
    def _load_user_state():
        try:
            if USER_STATE_PATH.exists():
                return json.loads(USER_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def _save_user_state(state: dict):
        try:
            USER_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    options = sorted(list(STRATEGY_REGISTRY.keys()))
    saved = _load_user_state()
    saved_active = saved.get("active_strategies")
    if "active_strategies" not in st.session_state:
        if isinstance(saved_active, list) and saved_active:
            st.session_state["active_strategies"] = [s for s in saved_active if s in options] or options
        else:
            st.session_state["active_strategies"] = options

    active_strats = st.sidebar.multiselect(
        "启用策略",
        options=options,
        default=st.session_state["active_strategies"],
    )
    st.session_state["active_strategies"] = active_strats
    _save_user_state({"active_strategies": active_strats})
    st.session_state["lookahead"] = int(la)
    st.session_state["timeframe"] = tf

    # 过滤变化时清空快回测缓存
    curr_filters = (tuple(sorted(active_strats)), tf, int(la))
    prev_filters = st.session_state.get("_prev_filters")
    if prev_filters != curr_filters:
        st.session_state["_prev_filters"] = curr_filters
        bt_result_cache = {}
    else:
        bt_result_cache = {}

    # ===== Exchange =====
    def build_exchange(name: str):
        klass = getattr(ccxt, name)
        return klass({
            "enableRateLimit": True,
            "timeout": 20000,
            # 如果无本地代理，删除 proxies 字段
            "proxies": {
                "http":  "http://127.0.0.1:7890",
                "https": "http://127.0.0.1:7890",
            },
        })

    try:
        ex = build_exchange(EXCHANGE_NAME)
    except Exception as e:
        st.error(f"初始化交易所失败：{e}")
        st.stop()

    # ===== Fetch & Compute =====
    def fetch_df(symbol: str, timeframe: str, limit=500) -> pd.DataFrame:
        raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return to_ohlcv_df(raw)

    def compute_signals(symbol: str, df: pd.DataFrame, timeframe: str):
        out = []
        active = st.session_state.get("active_strategies", ENABLED) or []
        for name, fn in STRATEGY_REGISTRY.items():
            if name not in active:
                continue
            try:
                sig = fn(symbol, df, timeframe)
                if sig:
                    sig["tf"] = timeframe
                    sig["eta_text"] = {"4h": "≈4 小时", "1d": "≈1 天", "1w": "≈1 周"}.get(timeframe, "—")
                    out.append(sig)
            except Exception as e:
                out.append({
                    "symbol": symbol, "strategy": name, "side": "—",
                    "entry": None, "target": None, "stop": None,
                    "confidence": "—", "tf": timeframe,
                    "eta_text": {"4h": "≈4 小时", "1d": "≈1 天", "1w": "≈1 周"}.get(timeframe, "—"),
                    "reason": f"{name} 失败：{e}"
                })
        return out

    st.subheader("💹 实时报价（仅 8 行，超出可滚动）")
    quotes = []
    signals_all = []
    err_count = 0
    symbol_to_df = {}

    for sym in SYMBOLS:
        try:
            df = fetch_df(sym, tf, limit=300)
            symbol_to_df[sym] = df
            last = df.iloc[-1]
            quotes.append({
                "Symbol": sym,
                "Close": fmt_price(last["close"]),
                "High": fmt_price(last["high"]),
                "Low": fmt_price(last["low"]),
                "Time": str(last["ts"])
            })
            # 策略信号
            signals_all.extend(compute_signals(sym, df, tf))

            # 诊断模式
            if diag_mode:
                try:
                    diag = strategy_diag(df, tf)
                    if diag:
                        with st.expander(f"📊 诊断 · {sym}", expanded=False):
                            st.write(diag)
                except Exception:
                    pass

            # 快回测
            if run_bt:
                try:
                    res, summ = backtest_symbol_with_strategies(
                        df, tf, st.session_state.get("active_strategies", ENABLED), symbol=sym, lookahead=int(la)
                    )
                    # 初始化/更新缓存
                    if "bt_cache" not in st.session_state: st.session_state["bt_cache"] = {}
                    st.session_state["bt_cache"][sym] = (res, summ)
                except Exception as e:
                    if "bt_cache" not in st.session_state: st.session_state["bt_cache"] = {}
                    st.session_state["bt_cache"][sym] = ([], {"error": str(e)})
        except Exception as e:
            err_count += 1
            quotes.append({
                "Symbol": sym, "Close": "—", "High": "—", "Low": "—", "Time": f"ERR: {e}"
            })

    # 回测结果展示
    if run_bt and "bt_cache" in st.session_state:
        st.subheader("🧪 快回测结果")
        cols = st.columns(3)
        with cols[0]: st.write(f"周期：**{tf}**")
        with cols[1]: st.write(f"启用策略：**{', '.join(st.session_state.get('active_strategies', ENABLED)) or '—'}**")
        with cols[2]: st.write(f"Lookahead：**{la}** 根K线")

        for sym in SYMBOLS:
            if sym not in st.session_state["bt_cache"]:
                continue
            res, summ = st.session_state["bt_cache"][sym]
            st.markdown(f"**{sym}**  → 胜率：{summ.get('win%',0)}%｜样本：{summ.get('trades',0)}｜未触发：{summ.get('open_or_none',0)}｜平均R：{summ.get('avg_R','—')}")
            if res:
                dfv = pd.DataFrame(res).tail(10)
                st.dataframe(dfv, use_container_width=True, height=260)
            st.divider()

        # 一致性校验
        bt_strats = set()
        for sym in st.session_state["bt_cache"]:
            res, _ = st.session_state["bt_cache"][sym]
            bt_strats.update({r.get("Strategy") for r in res if r.get("Strategy")})
        sig_strats = {s.get("strategy") for s in signals_all if s.get("strategy")}
        if len(bt_strats) > 1 and len(sig_strats) <= 1:
            st.error("检测到不一致：回测含多策略，但首页只展示了单策略。请检查渲染路径或缓存。")

    # 更新错误计数
    err_box.write(str(err_count))

    # 报价表（仅 8 行）
    qdf = pd.DataFrame(quotes)
    st.dataframe(qdf.head(8), height=240, use_container_width=True)

    # 推荐
    st.subheader("🧭 当前形态与推荐策略（简版）")
    st.caption("说明：此处按简单指标展示'震荡/趋势'倾向与建议策略名称，仅作演示。")
    probe_signals_from_list(signals_all)
    if len(quotes):
        sym_pick = st.selectbox("选择标的查看推荐", [q["Symbol"] for q in quotes])
        rec = [s["strategy"] for s in signals_all if s.get("symbol") == sym_pick]
        rec = list(dict.fromkeys(rec))[:3] if rec else ["—"]
        st.write(f"**{sym_pick}** · 推荐：", ", ".join(rec))

    # 当根与窗口内触发
    st.subheader("🔔 当根信号（收盘确认）")
    lookahead_val = st.session_state.get("lookahead", int(la))
    live_signals, recent_window_signals = latest_live_and_recent(
        symbol_to_df, tf, st.session_state.get("active_strategies", ENABLED), lookahead_val
    )
    if not live_signals:
        st.info("当根无触发。")
    else:
        _render_cards(live_signals, st.session_state.get("bt_cache", {}), run_bt)

    st.subheader(f"🕒 近 {lookahead_val} 根内的触发（与回测窗口一致）")
    if not recent_window_signals:
        st.caption("近窗口内无触发记录。")
    else:
        _render_cards(recent_window_signals, st.session_state.get("bt_cache", {}), run_bt)


if __name__ == "__main__":
    main()
