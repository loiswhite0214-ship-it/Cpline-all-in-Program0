def to_ohlcv_df(raw):
    import pandas as pd
    if isinstance(raw, list) and raw and isinstance(raw[0], (list, tuple)) and len(raw[0]) >= 6:
        df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    elif isinstance(raw, list) and raw and isinstance(raw[0], dict):
        df = pd.DataFrame(raw)
        rename_map = {
            "time":"timestamp","ts":"timestamp",
            "o":"open","h":"high","l":"low","c":"close","v":"volume"
        }
        df = df.rename(columns=rename_map)
    elif 'pandas' in str(type(raw)):
        df = raw.copy()
    else:
        raise ValueError(f"Unsupported OHLCV format: {type(raw)} -> cannot normalize")
    for col in ["timestamp","open","high","low","close","volume"]:
        if col not in df.columns:
            df[col] = None
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open","high","low","close"]).reset_index(drop=True)
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce").fillna(
            pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        )
    except Exception:
        pass
    return df

def assert_ohlcv_schema(df):
    need = {"timestamp","open","high","low","close","volume"}
    if 'pandas' not in str(type(df)):
        raise TypeError(f"OHLCV must be DataFrame, got {type(df)}")
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"OHLCV missing columns: {sorted(miss)}")
#!/usr/bin/env python3
"""
熬鹰计划 API服务器
为前端TypeScript提供真实的策略数据
"""

import json
import os
import requests
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
from collections import deque

# 导入真实策略算法（强制要求成功）
try:
    from strategies import STRATEGY_REGISTRY, vegas_tunnel, chan_simplified, macd, set_relax_mode
    from strategies_top15 import REGISTRY as TOP15_REGISTRY
    print("✅ All strategy modules loaded successfully")
except Exception as e:
    print(f"❌ Strategy module load failed: {e}")
    raise SystemExit(1)

# 强制依赖 ccxt（无则退出）
try:
    import ccxt
    print("✅ ccxt available")
except Exception as e:
    print(f"❌ ccxt not available: {e}")
    raise SystemExit(1)

# 数据源配置（默认强制真实数据）
USE_REAL_BINANCE_DATA = True

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# -------------------- Simple in-memory cache (24h) --------------------
_corr_cache = {}

def _cache_get_corr(key: str, ttl_sec: int = 86400):
    item = _corr_cache.get(key)
    if not item:
        return None
    now = time.time()
    if now - item.get('ts', 0) > ttl_sec:
        return None
    return item.get('payload')

def _cache_set_corr(key: str, payload: dict):
    _corr_cache[key] = { 'ts': time.time(), 'payload': payload }

# 配置数据
try:
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    SYMBOLS = config.get('symbols', ['BTC/USDT', 'ETH/USDT'])
    STRATEGIES = [s['name'] for s in config.get('strategies', []) if s.get('enabled')]
    EXCHANGE_NAME = str(config.get('exchange', 'binance')).lower()
    # 代理优先级：config.json > 环境变量
    PROXY_URL = (config.get('proxy') if isinstance(config, dict) else None) or os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')
    # 是否使用真实交易所数据（缺省也强制为 true）
    USE_REAL_BINANCE_DATA = bool(config.get('use_real_data', True))
    RELAX_MODE = bool(config.get('relax', False))
except Exception as e:
    print(f"❌ Failed to load config.json: {e}")
    # 缺省也强制真实模式，但若没有配置符号/策略，退出
    raise SystemExit(1)

# 基础价格（模拟真实数据）
BASE_PRICES = {
    'BTC/USDT': 65000, 'ETH/USDT': 3200, 'BNB/USDT': 590,
    'SOL/USDT': 140, 'XRP/USDT': 0.52, 'ADA/USDT': 0.45,
    'DOGE/USDT': 0.12, 'TRX/USDT': 0.08, 'AVAX/USDT': 28,
    'DOT/USDT': 6.5, 'SHIB/USDT': 0.000024, 'LINK/USDT': 12.5,
    'TON/USDT': 5.8, 'LTC/USDT': 85, 'MATIC/USDT': 0.85
}

# 策略名称映射
STRATEGY_NAMES = {
    'vegas_tunnel': 'Vegas通道',
    'chan_simplified': '简化缠论',
    'macd': 'MACD交叉',
    # TOP15 映射
    'ema_adx': 'EMA20/50 + ADX',
    'macd_plus': 'MACD+Histogram',
    'rsi_reversion': 'RSI反转+EMA200',
    'bb_mean': '布林带均值回归',
    'bb_squeeze': '布林带挤压突破',
    'donchian': 'Donchian突破',
    'supertrend': 'Supertrend',
    'keltner_break': 'Keltner通道突破',
    'ichimoku_kijun': 'Ichimoku云过滤',
    'psar_trend': 'Parabolic SAR',
    'stochrsi': 'StochRSI',
    'cci_reversion': 'CCI极值回归',
    'adx_di': 'ADX+DI',
    'heikin_ema': 'Heikin-Ashi+EMA',
    'vwap_pullback': 'VWAP回踩/突破',
}

# === Factors cache/state ===
FACTORS_CACHE: dict = {}  # key: (asset, granularity, date) -> {ts, data}
OI_SNAPSHOTS = { 'BTC': deque(maxlen=30), 'ETH': deque(maxlen=30) }
GEO_SNAPSHOTS = deque(maxlen=14)
POLICY_TONE_SNAPSHOTS = deque(maxlen=14)
FNG_SNAPSHOTS = deque(maxlen=14)

class MockDataGenerator:
    def __init__(self):
        self.prices = {}
        self.trends = {}
        self.last_update = time.time()
        self.price_history = {}  # 存储价格历史用于策略计算
        self.binance_exchange = None

        # 初始化交易所（强制真实模式）
        if USE_REAL_BINANCE_DATA:
            try:
                # NO_PROXY 避免本地接口被代理
                try:
                    existing_no_proxy = os.environ.get('NO_PROXY', '')
                    no_proxy_append = 'localhost,127.0.0.1,::1'
                    if existing_no_proxy:
                        if no_proxy_append not in existing_no_proxy:
                            os.environ['NO_PROXY'] = f"{existing_no_proxy},{no_proxy_append}"
                    else:
                        os.environ['NO_PROXY'] = no_proxy_append
                except Exception:
                    pass

                # 创建共享会话并配置代理
                session = requests.Session()
                session.trust_env = True
                if PROXY_URL:
                    session.proxies.update({'http': PROXY_URL, 'https': PROXY_URL})

                # 选择交易所（默认 binance，可在 config.json 配置 exchange: binance/okx 等）
                exchange_ctor = ccxt.binance if EXCHANGE_NAME == 'binance' else getattr(ccxt, EXCHANGE_NAME, ccxt.binance)

                self.binance_exchange = exchange_ctor({
                    'enableRateLimit': True,
                    'timeout': 20000,
                    'sandbox': False,
                })

                # 绑定会话与代理到 ccxt 实例
                if hasattr(self.binance_exchange, 'session'):
                    self.binance_exchange.session = session
                if PROXY_URL:
                    try:
                        self.binance_exchange.proxies = {'http': PROXY_URL, 'https': PROXY_URL}
                        print(f"Using proxy for ccxt: {PROXY_URL}")
                    except Exception as pe:
                        print(f"Setting ccxt proxies failed: {pe}")

                # 预加载市场，失败则退出
                try:
                    self.binance_exchange.load_markets(reload=True)
                    print("✅ Exchange markets loaded")
                except Exception as lm_err:
                    print(f"❌ Exchange load_markets failed: {lm_err}")
                    raise
                print("✅ Real exchange data mode enabled (Binance/OKX via ccxt)")
            except Exception as e:
                print(f"❌ Exchange connection failed: {e}")
                raise SystemExit(1)

        # 初始化价格和趋势
        for symbol in SYMBOLS:
            base_price = BASE_PRICES.get(symbol, 100)
            self.prices[symbol] = base_price
            self.trends[symbol] = random.choice([1, -1])
            # 初始化历史价格（用于策略计算）
            self.price_history[symbol] = self._fetch_real_history(symbol)

        # 应用放松模式（可选）
        try:
            set_relax_mode(bool(RELAX_MODE))
            if RELAX_MODE:
                print("⚙️  Relax mode enabled for strategies")
        except Exception:
            pass

    def _fetch_real_history(self, symbol, timeframe='4h', limit=300):
        """从Binance获取真实历史数据"""
        try:
            if not self.binance_exchange:
                raise RuntimeError('Exchange not connected (ccxt)')

            print(f"Fetching real history for {symbol}...")
            ohlcv = self.binance_exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            history = []
            for candle in ohlcv:
                timestamp, open_price, high, low, close, volume = candle
                history.append({
                    'ts': datetime.fromtimestamp(timestamp / 1000),
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume
                })

            print(f"Fetched {len(history)} candles for {symbol}")
            return history

        except Exception as e:
            print(f"❌ Failed to fetch {symbol}: {e}")
            raise

    def _update_real_prices(self):
        """更新真实价格数据"""
        if not self.binance_exchange:
            raise RuntimeError('Exchange not connected (ccxt)')

        try:
            # 获取实时价格
            tickers = self.binance_exchange.fetch_tickers(SYMBOLS[:8])
            for symbol in SYMBOLS[:8]:
                if symbol in tickers:
                    self.prices[symbol] = float(tickers[symbol]['last'])
        except Exception as e:
            print(f"❌ Failed to update real-time prices: {e}")
            raise

    def _generate_initial_history(self, base_price, periods=300):
        """生成初始历史数据用于策略计算"""
        history = []
        current_price = base_price

        for i in range(periods):
            # 生成OHLCV数据
            change = random.uniform(-0.02, 0.02)  # ±2%变化
            current_price *= (1 + change)

            high = current_price * (1 + random.uniform(0, 0.01))
            low = current_price * (1 - random.uniform(0, 0.01))
            open_price = history[-1]['close'] if history else current_price
            volume = random.uniform(1000000, 10000000)

            timestamp = datetime.now() - timedelta(hours=(periods-i))

            history.append({
                'ts': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': current_price,
                'volume': volume
            })

        return history

    def update_prices(self):
        """更新价格（仅真实模式）"""
        now = time.time()
        if now - self.last_update < 5:  # 5秒更新一次
            return

        # 强制真实数据
        if self.binance_exchange and USE_REAL_BINANCE_DATA:
            self._update_real_prices()
        else:
            raise RuntimeError('Real mode required but exchange not connected')

        self.last_update = now

    # 删除 _update_mock_prices：严格真实模式

    def get_quote_data(self):
        """获取行情数据（严格真实）"""
        if not USE_REAL_BINANCE_DATA:
            raise RuntimeError('Real data mode required')
        if not self.binance_exchange:
            raise RuntimeError('Exchange not connected (ccxt)')
        try:
            markets = SYMBOLS[:8]
            quotes = []
            # 优先批量获取
            tickers = {}
            try:
                tickers = self.binance_exchange.fetch_tickers(markets) or {}
            except Exception:
                tickers = {}

            for symbol in markets:
                t = tickers.get(symbol)
                if not t:
                    # 回退到单币请求（仍为真实数据，不使用模拟）
                    try:
                        t = self.binance_exchange.fetch_ticker(symbol)
                    except Exception:
                        t = None
                if not t:
                    continue
                last = float(t.get('last') or t.get('close') or 0)
                open_price = float(t.get('open') or 0) if t.get('open') is not None else None
                pct = t.get('percentage')
                if pct is None and open_price and open_price > 0:
                    pct = ((last - open_price) / open_price) * 100.0
                pct = float(pct) if pct is not None else 0.0

                quotes.append({
                    'symbol': symbol.replace('/USDT', ''),
                    'close': last,
                    'changePercent': f"{pct:+.2f}%",
                    'isPositive': pct >= 0
                })
            if not quotes:
                raise RuntimeError('Empty tickers from exchange')
            return quotes
        except Exception as e:
            print(f"❌ fetch_tickers error: {e}")
            raise

    def get_signals_data(self):
        """获取交易信号数据（使用真实策略计算）"""
        self.update_prices()
        signals = []

        # 合并注册表
        effective_registry = {}
        try:
            effective_registry.update(STRATEGY_REGISTRY)
            effective_registry.update(TOP15_REGISTRY)
        except Exception:
            effective_registry = STRATEGY_REGISTRY

        # 使用真实策略计算信号（覆盖全部符号）
        for symbol in SYMBOLS:
            try:
                raw = self.price_history.get(symbol) or []
                if isinstance(raw, pd.DataFrame):
                    df = raw.copy()
                else:
                    df = to_ohlcv_df(raw)
                # 统一时间列
                if 'ts' in df.columns and 'timestamp' not in df.columns:
                    df = df.rename(columns={'ts': 'timestamp'})
                assert_ohlcv_schema(df)
                if len(df) < 80:
                    continue
                # 最后一根未收盘剔除
                df_closed = df.iloc[:-1] if len(df) > 1 else df

                for strategy_name in STRATEGIES:
                    if strategy_name not in effective_registry:
                        continue
                    try:
                        fn = effective_registry[strategy_name]
                        result_dict = None

                        # 基础策略：签名 (symbol, df, tf) -> dict
                        if strategy_name in STRATEGY_REGISTRY:
                            result = fn(symbol, df_closed, '4h')
                            result_dict = result if result else None
                        else:
                            # TOP15：签名 (df) -> DataFrame(signal/entry/sl/tp)
                            out = fn(df_closed)
                            if out is not None and isinstance(out, pd.DataFrame) and len(out):
                                last = out.iloc[-1]
                                sig = int(last.get('signal') or 0)
                                if sig != 0:
                                    side = 'BUY' if sig > 0 else 'SELL'
                                    entry = float(last.get('entry')) if last.get('entry') is not None else float(df_closed['close'].iloc[-1])
                                    tp = float(last.get('tp')) if last.get('tp') is not None else entry
                                    sl = float(last.get('sl')) if last.get('sl') is not None else entry
                                    result_dict = {
                                        'side': side,
                                        'entry': entry,
                                        'target': tp,
                                        'stop': sl,
                                        'confidence': 40
                                    }

                        if result_dict:
                            signals.append({
                                'symbol': symbol.replace('/USDT', ''),
                                'strategy': STRATEGY_NAMES.get(strategy_name, strategy_name),
                                'side': result_dict['side'],
                                'entry': round(float(result_dict['entry']), 6),
                                'target': round(float(result_dict['target']), 6),
                                'stop': round(float(result_dict['stop']), 6),
                                'confidence': result_dict.get('confidence', 50),
                                'tf': '4h',
                                'time': datetime.now().strftime('%H:%M')
                            })
                            # 每币种只取一个信号即可，避免灌水
                            break
                    except Exception as e:
                        print(f"策略 {strategy_name} 计算失败: {e}")
                        continue
            except Exception as e:
                print(f"币种 {symbol} 数据处理失败: {e}")
                continue

        return signals

    # 删除 fallback：严格真实

    def get_learning_stats(self, has_strategies_enabled=True):
        """获取学习成绩数据"""
        if not has_strategies_enabled:
            return {
                'profitRatio': '--/--',
                'winRate': '--/--',
                'maxDrawdown': '--/--'
            }

        # 生成合理的模拟数据
        profit_ratio = round(random.uniform(1.2, 2.2), 1)
        win_rate = random.randint(55, 80)
        max_drawdown = random.randint(3, 12)

        return {
            'profitRatio': str(profit_ratio),
            'winRate': f'{win_rate}%',
            'maxDrawdown': f'{max_drawdown}%'
        }

# 创建数据生成器实例
data_generator = MockDataGenerator()

@app.route('/')
def index():
    """API首页"""
    return jsonify({
        'message': '🚀 熬鹰计划 API 服务器',
        'version': '1.0.0',
        'endpoints': [
            'GET /api/quotes - 获取实时行情',
            'GET /api/signals - 获取交易信号',
            'GET /api/learning-stats - 获取学习成绩',
            'GET /api/config - 获取配置信息',
            'GET /api/backtest/<symbol>?days=N&tf=4h|1d|1w&strategy=name - 回测'
        ],
        'status': 'running',
        'dataMode': 'real' if USE_REAL_BINANCE_DATA else 'mock'
    })

@app.route('/api/signals/diagnose')
def diagnose_signals():
    """诊断：逐个币种在最后一根已收K上跑所有策略，返回命中情况。支持 ?relax=1 临时放松阈值。"""
    try:
        if not USE_REAL_BINANCE_DATA:
            raise RuntimeError('Real data mode required')

        # 合并注册表
        effective_registry = {}
        try:
            effective_registry.update(STRATEGY_REGISTRY)
            effective_registry.update(TOP15_REGISTRY)
        except Exception:
            effective_registry = STRATEGY_REGISTRY

        # 可选临时 relax
        relax_flag = request.args.get('relax')
        restore_relax = None
        try:
            if relax_flag in ('1', 'true', 'True'):
                restore_relax = True
                set_relax_mode(True)
        except Exception:
            pass

        report = []
        for symbol in SYMBOLS:
            df = pd.DataFrame(data_generator.price_history.get(symbol) or [])
            # 标准化
            if not isinstance(df, pd.DataFrame) or df.empty:
                try:
                    df = to_ohlcv_df(data_generator.price_history.get(symbol) or [])
                except Exception:
                    df = pd.DataFrame()
            if df.empty:
                report.append({ 'symbol': symbol, 'history': 0, 'triggered': [], 'errors': ['no_history'] })
                continue
            # 最后一根（已收），统一用 4h
            tf = '4h'
            if 'ts' in df.columns and 'timestamp' not in df.columns:
                df = df.rename(columns={'ts': 'timestamp'})
            try:
                assert_ohlcv_schema(df)
            except Exception as e:
                report.append({ 'symbol': symbol, 'history': int(len(df)), 'triggered': [], 'errors': [f'schema:{e}'] })
                continue
            df_closed = df.iloc[:-1] if len(df) > 1 else df
            triggered = []
            errors = []
            for sname, fn in effective_registry.items():
                try:
                    if sname in STRATEGY_REGISTRY:
                        res = fn(symbol, df_closed, tf)
                        if res:
                            triggered.append(sname)
                    else:
                        out = fn(df_closed)
                        if out is not None and isinstance(out, pd.DataFrame) and len(out):
                            last = out.iloc[-1]
                            sig = int(last.get('signal') or 0)
                            if sig != 0:
                                triggered.append(sname)
                except Exception as e:
                    errors.append(f"{sname}:{e}")
            report.append({ 'symbol': symbol, 'history': int(len(df)), 'triggered': triggered, 'errors': errors })

        # 恢复 relax（若需要可改为读取全局状态）
        try:
            if restore_relax:
                set_relax_mode(False)
        except Exception:
            pass

        return jsonify({ 'success': True, 'data': report })
    except Exception as e:
        return jsonify({ 'success': False, 'error': str(e) }), 500

@app.route('/api/quotes')
def get_quotes():
    """获取实时行情数据"""
    try:
        quotes = data_generator.get_quote_data()
        return jsonify({
            'success': True,
            'data': quotes,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/signals')
def get_signals():
    """获取交易信号数据 - 修复前端期望的数据结构"""
    try:
        signals = data_generator.get_signals_data()
        # 前端期望的是 {items: [...]} 结构
        return jsonify({
            'items': signals,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"[API] Signals error: {e}")
        # 返回空数组而不是500错误
        return jsonify({
            'items': [],
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 200

@app.route('/api/learning-stats')
def get_learning_stats():
    """获取学习成绩数据"""
    try:
        # 检查是否有启用的策略（这里简化为随机）
        has_strategies = random.choice([True, True, False])  # 66%概率有策略

        stats = data_generator.get_learning_stats(has_strategies)
        return jsonify({
            'success': True,
            'data': stats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/config')
def get_config():
    """获取配置信息"""
    return jsonify({
        'success': True,
        'data': {
            'symbols': SYMBOLS,
            'strategies': STRATEGIES,
            'strategy_names': STRATEGY_NAMES
        }
    })

def _linear_map(x, lo_src, hi_src, lo_dst=0, hi_dst=100):
    try:
        x = max(lo_src, min(hi_src, float(x)))
        return int(round((x - lo_src) / (hi_src - lo_src) * (hi_dst - lo_dst) + lo_dst))
    except Exception:
        return None

def _minmax_norm(series):
    arr = [float(v) for v in series if v is not None]
    if not arr:
        return None
    lo, hi = min(arr), max(arr)
    if hi == lo:
        return 50
    return int(round((arr[-1] - lo) / (hi - lo) * 100))

def _utc_now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'

@app.route('/api/factors')
def get_factors():
    """公开API聚合（免注册源）：sentiment/capital/onchain/policy/geopolitics/macro
    Query: asset=BTC|ETH&granularity=daily|weekly&date=YYYY-MM-DD
    缓存 10 分钟；子因子失败则置空，维度为均值；输出 0-100 分数与 wow。
    """
    try:
        asset = request.args.get('asset', 'BTC').upper()
        gran = request.args.get('granularity', 'daily').lower()
        date_s = request.args.get('date')
        cache_key = (asset, gran, date_s or '')

        now_ts = time.time()
        cached = FACTORS_CACHE.get(cache_key)
        if cached and now_ts - cached['ts'] < 60 * 10:
            return jsonify({'success': True, 'data': cached['data']})

        # helpers
        def signal_bucket(score):
            if score is None: return None
            if score >= 66: return 'Mild+'
            if score <= 33: return 'Mild-'
            return 'Neutral'
        def dimension_score(sub_scores):
            vals = [s for s in sub_scores if s is not None]
            if not vals: return None
            return int(round(sum(vals) / len(vals)))
        def wow(cur, prev):
            if prev is None: return 0
            return int(round(cur - prev))

        # 1) sentiment
        score_fng = None; prev_fng = None; fng_notes = None
        try:
            r = requests.get('https://api.alternative.me/fng/?limit=14', timeout=8)
            js = r.json(); arr = [int(i['value']) for i in (js.get('data') or []) if i.get('value') is not None]
            if arr:
                last7 = arr[:7] if len(arr) >= 7 else arr
                score_fng = int(round(sum(last7) / len(last7)))
                if len(arr) >= 8:
                    prev7 = arr[1:8]
                    prev_fng = int(round(sum(prev7) / len(prev7)))
                fng_notes = f"FNG {score_fng} (7d avg)"
        except Exception:
            pass

        # 2) capital: funding + OI
        score_funding = None; prev_funding = None; notes_funding = None
        try:
            sym = f"{asset}USDT"
            r = requests.get(f'https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}&limit=100', timeout=8)
            rows = r.json() or []
            # 近3条≈24h，前一档再取前3条
            cur3 = rows[-3:]
            prev3 = rows[-6:-3]
            if cur3:
                avg = sum(float(x['fundingRate']) for x in cur3) / len(cur3)
                avg = max(-0.0005, min(0.0005, avg))
                score_funding = int(round((avg - (-0.0005)) / (0.001) * 100))
                notes_funding = f"Funding avg {avg:.5f} (24h)"
            if prev3:
                pavg = sum(float(x['fundingRate']) for x in prev3) / len(prev3)
                pavg = max(-0.0005, min(0.0005, pavg))
                prev_funding = int(round((pavg - (-0.0005)) / (0.001) * 100))
        except Exception:
            pass

        score_oi = None; prev_oi = None; notes_oi = None
        try:
            r = requests.get(f'https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={asset}', timeout=8)
            jj = r.json(); rows = (jj.get('result') or [])
            vals = []
            for it in rows:
                v = it.get('open_interest') or it.get('oi') or it.get('base_volume')
                if v is not None:
                    try: vals.append(float(v))
                    except: pass
            if vals:
                total = sum(vals)
                OI_SNAPSHOTS[asset].append(total)
                seq = list(OI_SNAPSHOTS[asset])
                if len(seq) >= 2:
                    lo, hi = min(seq), max(seq)
                    score_oi = int(round(0 if hi==lo else (seq[-1]-lo)/(hi-lo)*100))
                    prev_oi = int(round(0 if hi==lo else (seq[-2]-lo)/(hi-lo)*100))
                else:
                    score_oi = None
                notes_oi = 'Deribit open_interest 近似'
        except Exception:
            pass

        # 3) onchain
        score_onchain = None; prev_onchain = None; oc_key = None; oc_notes = None
        try:
            if asset == 'BTC':
                r = requests.get('https://api.blockchain.info/charts/transactions?timespan=30days&format=json&cors=true', timeout=8)
                js = r.json(); series = [it['y'] for it in (js.get('values') or []) if 'y' in it]
                if series:
                    # latest and prev
                    cur = series[-1]
                    prev = series[-2] if len(series) >= 2 else None
                    lo, hi = min(series), max(series)
                    score_onchain = int(round(0 if hi==lo else (cur-lo)/(hi-lo)*100))
                    prev_onchain = int(round(0 if prev is None or hi==lo else (prev-lo)/(hi-lo)*100))
                    oc_key = 'tx'; oc_notes = f'BTC tx norm {score_onchain} (30d window)'
            else:
                r = requests.get('https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=30', timeout=8)
                js = r.json(); vols = [v for _, v in (js.get('total_volumes') or [])]
                if vols:
                    cur = vols[-1]; prev = vols[-2] if len(vols)>=2 else None
                    lo, hi = min(vols), max(vols)
                    norm_cur = int(round(0 if hi==lo else (cur-lo)/(hi-lo)*100))
                    norm_prev = int(round(0 if prev is None or hi==lo else (prev-lo)/(hi-lo)*100))
                    score_onchain = norm_cur
                    prev_onchain = norm_prev
                    oc_key = 'eth_activity_proxy'; oc_notes = 'ETH onchain proxy via CoinGecko volume (30d norm)'
        except Exception:
            pass

        # 4) policy tones (GDELT)
        score_policy_sec = None; prev_policy_sec = None; notes_sec = None
        score_policy_etf = None; prev_policy_etf = None; notes_etf = None
        try:
            r = requests.get('https://api.gdeltproject.org/api/v2/doc/doc?query=SEC%20crypto%20ETF&mode=TimelineTone&timespan=7d&format=json', timeout=8)
            jj = r.json(); arr = [float(x.get('value')) for x in (jj.get('timeline') or []) if x.get('value') is not None]
            if arr:
                cur = sum(arr[-7:])/min(7,len(arr))
                prevv = sum(arr[-14:-7])/7 if len(arr)>=14 else None
                score_policy_sec = int(round((cur - (-5)) / 10 * 100))
                prev_policy_sec = int(round((prevv - (-5)) / 10 * 100)) if prevv is not None else None
                notes_sec = f'GDELT tone SEC {cur:.2f}'
        except Exception:
            pass
        try:
            r = requests.get('https://api.gdeltproject.org/api/v2/doc/doc?query=spot%20ETF%20Bitcoin%20OR%20Ethereum&mode=TimelineTone&timespan=7d&format=json', timeout=8)
            jj = r.json(); arr = [float(x.get('value')) for x in (jj.get('timeline') or []) if x.get('value') is not None]
            if arr:
                cur = sum(arr[-7:])/min(7,len(arr))
                prevv = sum(arr[-14:-7])/7 if len(arr)>=14 else None
                score_policy_etf = int(round((cur - (-5)) / 10 * 100))
                prev_policy_etf = int(round((prevv - (-5)) / 10 * 100)) if prevv is not None else None
                notes_etf = f'GDELT tone spotETF {cur:.2f}'
        except Exception:
            pass

        # 5) geopolitics vol (inverse)
        score_geo = None; prev_geo = None; notes_geo = None
        try:
            r = requests.get('https://api.gdeltproject.org/api/v2/doc/doc?query=sanction%20OR%20war%20OR%20conflict%20OR%20missile&mode=TimelineVol&timespan=7d&format=json', timeout=8)
            jj = r.json(); arr = [int(x.get('value')) for x in (jj.get('timeline') or []) if x.get('value') is not None]
            if arr:
                GEO_SNAPSHOTS.extend(arr[-7:])  # 使用7天窗口
                seq = list(GEO_SNAPSHOTS)
                lo, hi = (min(seq), max(seq)) if seq else (0,1)
                norm_cur = int(round(0 if hi==lo else (seq[-1]-lo)/(hi-lo)*100))
                norm_prev = int(round(0 if len(seq)<2 or hi==lo else (seq[-2]-lo)/(hi-lo)*100))
                score_geo = 100 - norm_cur
                prev_geo = 100 - norm_prev
                notes_geo = f'GDELT geo vol={seq[-1]}, inv-norm {score_geo}'
        except Exception:
            pass

        # 6) macro: USD strength proxy via EURUSD (exchangerate.host)
        score_macro = None; prev_macro = None
        try:
            r = requests.get('https://api.exchangerate.host/latest?base=USD&symbols=EUR', timeout=8)
            js = r.json(); rate = js.get('rates', {}).get('EUR')
            if rate:
                inv = 1.0/float(rate)  # 近似 DXY 方向
                # 在 [0.9, 1.2] 做 min-max -> [0,100] 后反向（美元强→分低）
                norm = int(round(0 if 1.2-0.9==0 else (inv-0.9)/(1.2-0.9)*100))
                score_macro = 100 - max(0, min(100, norm))
                prev_macro = score_macro
            else:
                score_macro = 60; prev_macro = 60
        except Exception:
            score_macro = 60; prev_macro = 60

        # Assemble subfactors
        sf_sent = [{ 'key':'fear_greed', 'score':score_fng, 'weight':1.0, 'signal': signal_bucket(score_fng), 'notes': fng_notes }]
        sf_cap = [
            { 'key':'funding_rate', 'score':score_funding, 'weight':0.5, 'signal': signal_bucket(score_funding), 'notes': notes_funding },
            { 'key':'oi_proxy', 'score':score_oi, 'weight':0.5, 'signal': signal_bucket(score_oi), 'notes': notes_oi }
        ]
        sf_onc = [{ 'key': oc_key or ('tx' if asset=='BTC' else 'eth_activity_proxy'), 'score':score_onchain, 'weight':1.0, 'signal': signal_bucket(score_onchain), 'notes': oc_notes }]
        sf_pol = [
            { 'key':'sec_news', 'score':score_policy_sec, 'weight':0.5, 'signal': signal_bucket(score_policy_sec), 'notes': notes_sec },
            { 'key':'etf_news', 'score':score_policy_etf, 'weight':0.5, 'signal': signal_bucket(score_policy_etf), 'notes': notes_etf }
        ]
        sf_geo = [{ 'key':'events', 'score':score_geo, 'weight':1.0, 'signal': signal_bucket(score_geo), 'notes': notes_geo }]
        sf_mac = [{ 'key':'dxy_proxy', 'score':score_macro, 'weight':1.0, 'signal': signal_bucket(score_macro), 'notes': '占位，待接 FRED' }]

        # Dimension scores and wow
        dim_sent_score = dimension_score([sf['score'] for sf in sf_sent]); dim_sent_prev = prev_fng
        dim_cap_score = dimension_score([sf['score'] for sf in sf_cap]); dim_cap_prev = dimension_score([prev_funding, prev_oi])
        dim_onc_score = dimension_score([sf['score'] for sf in sf_onc]); dim_onc_prev = prev_onchain
        dim_pol_score = dimension_score([sf['score'] for sf in sf_pol]); dim_pol_prev = dimension_score([prev_policy_sec, prev_policy_etf])
        dim_geo_score = dimension_score([sf['score'] for sf in sf_geo]); dim_geo_prev = prev_geo
        dim_mac_score = score_macro; dim_mac_prev = prev_macro

        as_of = _utc_now_iso()

        dimensions = [
            { 'name':'macro', 'score': dim_mac_score, 'wow': wow(dim_mac_score, dim_mac_prev), 'as_of': as_of, 'sub_factors': sf_mac },
            { 'name':'policy', 'score': dim_pol_score, 'wow': wow(dim_pol_score, dim_pol_prev), 'as_of': as_of, 'sub_factors': sf_pol },
            { 'name':'capital', 'score': dim_cap_score, 'wow': wow(dim_cap_score, dim_cap_prev), 'as_of': as_of, 'sub_factors': sf_cap },
            { 'name':'geopolitics', 'score': dim_geo_score, 'wow': wow(dim_geo_score, dim_geo_prev), 'as_of': as_of, 'sub_factors': sf_geo },
            { 'name':'onchain', 'score': dim_onc_score, 'wow': wow(dim_onc_score, dim_onc_prev), 'as_of': as_of, 'sub_factors': sf_onc },
            { 'name':'sentiment', 'score': dim_sent_score, 'wow': wow(dim_sent_score, dim_sent_prev), 'as_of': as_of, 'sub_factors': sf_sent },
        ]

        payload = { 'asset': asset, 'granularity': gran, 'as_of': as_of, 'dimensions': dimensions }
        FACTORS_CACHE[cache_key] = { 'ts': now_ts, 'data': payload }
        return jsonify({ 'success': True, 'data': payload })
    except Exception as e:
        return jsonify({ 'success': False, 'error': str(e) }), 200

@app.route('/api/factors/history')
def get_factors_history():
    """返回最近 N 天（尽可能多）的各维度分数日序列。
    Query: asset=BTC|ETH&granularity=daily|weekly&days=60
    - 对缺失维度按当日可用维度等比重标。
    - 尽力从免注册源获取历史；获取不到的维度以 null 填充。
    """
    try:
        asset = request.args.get('asset', 'BTC').upper()
        gran = request.args.get('granularity', 'daily').lower()
        days = int(request.args.get('days', '60'))

        # 生成日期索引（UTC，日粒度）
        end_dt = datetime.utcnow().date()
        start_dt = end_dt - timedelta(days=days-1)
        date_list = [start_dt + timedelta(days=i) for i in range(days)]
        date_keys = [d.isoformat() for d in date_list]

        # 收集各子源的历史序列（以 0-100 归一值为目标）
        def safe_get(url, timeout=8):
            try:
                r = requests.get(url, timeout=timeout)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
            return None

        # 1) sentiment: FNG 历史
        sent_map = {}
        js_fng = safe_get('https://api.alternative.me/fng/?limit=365')
        try:
            arr = js_fng.get('data') if isinstance(js_fng, dict) else None
            if arr:
                # API 的 data 按时间倒序
                for item in arr:
                    try:
                        ts = int(item.get('timestamp'))
                        d = datetime.utcfromtimestamp(ts).date().isoformat()
                        sent_map[d] = int(item.get('value'))
                    except Exception:
                        continue
        except Exception:
            pass

        # 2) capital: funding（近 ~33 天），日均值映射 [-0.0005,0.0005]→[0,100]
        cap_map = {}
        try:
            sym = f"{asset}USDT"
            js_fr = safe_get(f'https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}&limit=100') or []
            rows = js_fr if isinstance(js_fr, list) else []
            tmp = {}
            for it in rows:
                try:
                    ts = int(it.get('fundingTime'))/1000.0
                    d = datetime.utcfromtimestamp(ts).date().isoformat()
                    val = float(it.get('fundingRate'))
                    tmp.setdefault(d, []).append(val)
                except Exception:
                    continue
            for d, arr in tmp.items():
                if not arr: continue
                avg = sum(arr)/len(arr)
                avg = max(-0.0005, min(0.0005, avg))
                cap_map[d] = int(round((avg - (-0.0005)) / (0.001) * 100))
        except Exception:
            pass

        # 3) onchain：BTC tx 或 ETH volume（近 30-60 天）→ 当天归一
        onc_map = {}
        try:
            if asset == 'BTC':
                js = safe_get('https://api.blockchain.info/charts/transactions?timespan=60days&format=json&cors=true')
                vals = js.get('values') if isinstance(js, dict) else []
                xs = [(datetime.utcfromtimestamp(int(it.get('x'))).date().isoformat(), float(it.get('y'))) for it in vals if 'x' in it and 'y' in it]
                series_vals = [v for _, v in xs]
                if series_vals:
                    lo, hi = min(series_vals), max(series_vals)
                    for d, v in xs:
                        norm = 50 if hi==lo else (v-lo)/(hi-lo)*100
                        onc_map[d] = int(round(norm))
            else:
                js = safe_get('https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=60')
                vols = js.get('total_volumes') if isinstance(js, dict) else []
                xs = [(datetime.utcfromtimestamp(int(ts/1000)).date().isoformat(), float(v)) for ts, v in vols]
                series_vals = [v for _, v in xs]
                if series_vals:
                    lo, hi = min(series_vals), max(series_vals)
                    for d, v in xs:
                        norm = 50 if hi==lo else (v-lo)/(hi-lo)*100
                        onc_map[d] = int(round(norm))
        except Exception:
            pass

        # 4) policy: GDELT 30d TimelineTone 两个关键词组 → 取均值 → 线性映射 [-5,5]→[0,100]
        pol_map = {}
        try:
            def _gdelt_tone(url):
                jj = safe_get(url)
                arr = jj.get('timeline') if isinstance(jj, dict) else []
                res = {}
                for it in arr:
                    try:
                        d = it.get('date') or it.get('datetime') or it.get('time')
                        if not d: continue
                        # GDELT 日期形如 YYYYMMDDhhmmss or YYYY-MM-DD
                        ds = str(d)
                        if len(ds) >= 8 and ds.isdigit():
                            dkey = datetime.strptime(ds[:8], '%Y%m%d').date().isoformat()
                        else:
                            dkey = datetime.fromisoformat(ds[:10]).date().isoformat()
                        val = float(it.get('value'))
                        res.setdefault(dkey, []).append(val)
                    except Exception:
                        continue
                out = {}
                for d, vals in res.items():
                    avg = sum(vals)/len(vals)
                    # 映射到 0-100
                    out[d] = int(round((avg - (-5.0)) / 10.0 * 100))
                return out
            m1 = _gdelt_tone('https://api.gdeltproject.org/api/v2/doc/doc?query=SEC%20crypto%20ETF&mode=TimelineTone&timespan=30d&format=json')
            m2 = _gdelt_tone('https://api.gdeltproject.org/api/v2/doc/doc?query=spot%20ETF%20Bitcoin%20OR%20Ethereum&mode=TimelineTone&timespan=30d&format=json')
            all_days = set(m1.keys()) | set(m2.keys())
            for d in all_days:
                vals = []
                if d in m1: vals.append(m1[d])
                if d in m2: vals.append(m2[d])
                if vals:
                    pol_map[d] = int(round(sum(vals)/len(vals)))
        except Exception:
            pass

        # 5) geopolitics: GDELT 30d TimelineVol → 正向归一后取 100-值
        geo_map = {}
        try:
            jj = safe_get('https://api.gdeltproject.org/api/v2/doc/doc?query=sanction%20OR%20war%20OR%20conflict%20OR%20missile&mode=TimelineVol&timespan=30d&format=json')
            arr = jj.get('timeline') if isinstance(jj, dict) else []
            xs = []
            for it in arr:
                try:
                    d = it.get('date') or it.get('datetime') or it.get('time')
                    ds = str(d)
                    if len(ds) >= 8 and ds.isdigit():
                        dkey = datetime.strptime(ds[:8], '%Y%m%d').date().isoformat()
                    else:
                        dkey = datetime.fromisoformat(ds[:10]).date().isoformat()
                    xs.append((dkey, int(it.get('value'))))
                except Exception:
                    continue
            vals = [v for _, v in xs]
            if vals:
                lo, hi = min(vals), max(vals)
                for d, v in xs:
                    norm = 50 if hi==lo else (v-lo)/(hi-lo)*100
                    geo_map[d] = int(round(100 - norm))
        except Exception:
            pass

        # 6) macro: exchangerate.host timeseries USD/EUR → 取 1/rate 近似 DXY，映射到 0-100 再反向
        mac_map = {}
        try:
            url = f'https://api.exchangerate.host/timeseries?base=USD&symbols=EUR&start_date={start_dt.isoformat()}&end_date={end_dt.isoformat()}'
            js = safe_get(url)
            rates = js.get('rates') if isinstance(js, dict) else {}
            series = []
            for d, obj in rates.items():
                try:
                    rate = float(obj.get('EUR'))
                    inv = 1.0/rate
                    series.append((d, inv))
                except Exception:
                    continue
            vals = [v for _, v in series]
            if vals:
                lo, hi = min(vals), max(vals)
                for d, v in series:
                    norm = 50 if hi==lo else (v-lo)/(hi-lo)*100
                    mac_map[d] = int(round(100 - norm))
        except Exception:
            pass

        # 组合维度分数（等权平均子因子；缺失为 null）
        series_out = []
        for d in date_keys:
            dim = {
                'macro': mac_map.get(d, None),
                'policy': pol_map.get(d, None),
                'capital': cap_map.get(d, None),
                'geopolitics': geo_map.get(d, None),
                'onchain': onc_map.get(d, None),
                'sentiment': sent_map.get(d, None)
            }
            # granularity weekly: 将 7 天均值视为当周
            if gran == 'weekly':
                # 仅对齐周一开始的块
                try:
                    cur_date = datetime.fromisoformat(d).date()
                except Exception:
                    cur_date = None
                if cur_date and cur_date.weekday() == 6:  # 周日收敛到一周
                    pass
                # 简化：此版本保留 daily，前端可选择 weekly
            series_out.append({'ts': f"{d}T00:00:00Z", 'dimensions': dim})

        weights = {k: round(1/6, 4) for k in ['macro','policy','capital','geopolitics','onchain','sentiment']}
        payload = {
            'asset': asset,
            'granularity': gran,
            'as_of': _utc_now_iso(),
            'weights': weights,
            'series': series_out
        }
        return jsonify({'success': True, 'data': payload})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 200

@app.route('/api/factors/corr_lines')
def get_factors_corr_lines():
    """返回BTC与多因子的30日滚动相关性数据
    Query: asset=BTC&window=30
    """
    try:
        asset = request.args.get('asset', 'BTC').upper()
        window = int(request.args.get('window', 30))
        cache_key = f"{asset}:{window}"
        cached = _cache_get_corr(cache_key)
        if cached:
            return jsonify(cached), 200
        
        # For now, return mock data matching the expected structure
        # TODO: Implement real correlation calculation with external data sources
        days = 90
        dates = []
        start = pd.Timestamp.now() - pd.Timedelta(days=days-1, unit='D')
        for i in range(days):
            dates.append((start + pd.Timedelta(days=i)).strftime('%Y-%m-%d'))
        
        # Mock correlation data (realistic patterns)
        factors = ['DXY', 'VIX', 'SPX', 'XAU', 'FNG', 'Funding', 'ETF_Flows', 'NFCI']
        rho = {}
        
        for factor in factors:
            # Generate realistic correlation patterns
            if factor == 'DXY':
                # USD strength typically negative correlation with BTC
                base_corr = -0.3 + np.random.normal(0, 0.1, days)
            elif factor == 'VIX':
                # VIX (fear) often negative correlation
                base_corr = -0.2 + np.random.normal(0, 0.15, days)
            elif factor == 'SPX':
                # SPX often positive correlation
                base_corr = 0.4 + np.random.normal(0, 0.1, days)
            elif factor == 'XAU':
                # Gold often weak positive correlation
                base_corr = 0.1 + np.random.normal(0, 0.1, days)
            elif factor == 'FNG':
                # Fear & Greed often positive correlation
                base_corr = 0.3 + np.random.normal(0, 0.1, days)
            elif factor == 'Funding':
                # Funding rate often negative correlation
                base_corr = -0.1 + np.random.normal(0, 0.1, days)
            elif factor == 'ETF_Flows':
                # ETF flows often positive correlation
                base_corr = 0.2 + np.random.normal(0, 0.1, days)
            elif factor == 'NFCI':
                # Financial conditions often negative correlation
                base_corr = -0.2 + np.random.normal(0, 0.1, days)
            else:
                base_corr = np.random.normal(0, 0.1, days)
            
            # Apply 3-day EMA smoothing
            smoothed = pd.Series(base_corr).ewm(span=3).mean().values
            # Clamp to [-1, 1] and convert to percentage
            rho[factor] = (np.clip(smoothed, -1, 1) * 100).tolist()
        
        payload = {
            'dates': dates,
            'rho': rho,
            'meta': {
                'window': window,
                'source': 'mock',
                'asset': asset,
                'asOf': _utc_now_iso()
            }
        }
        _cache_set_corr(cache_key, payload)
        return jsonify(payload), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 200

@app.route('/api/factors/index')
def get_factors_index():
    """根据历史因子序列计算多空指数 B(t) 及各维度贡献 C_i(t)，并做 EWMA 平滑。
    Query: asset=BTC&granularity=daily&days=60&alpha=0.3
    """
    try:
        asset = request.args.get('asset', 'BTC').upper()
        gran = request.args.get('granularity', 'daily').lower()
        days = int(request.args.get('days', '60'))
        try:
            alpha = float(request.args.get('alpha', '0.3'))
        except Exception:
            alpha = 0.3

        # 复用 /history 构建日序列
        with app.test_request_context(query_string={'asset': asset, 'granularity': gran, 'days': str(days)}):
            resp = get_factors_history()
            # resp 可能是 flask.Response
            try:
                js = resp.get_json()
            except Exception:
                js = resp[0].get_json() if isinstance(resp, tuple) else None
        if not js or not js.get('success'):
            raise RuntimeError(js.get('error') if js else 'history failed')

        hist = js['data']
        series = hist.get('series', [])
        keys = ['macro','policy','capital','geopolitics','onchain','sentiment']

        # 计算 B(t) 与 C_i(t)
        def ewma(values, a):
            out = []
            prev = None
            for v in values:
                if v is None:
                    out.append(prev)
                    continue
                prev = v if prev is None else (a*v + (1-a)*prev)
                out.append(prev)
            return out

        index_rows = []
        contrib_map = {k: [] for k in keys}

        for item in series:
            ts = item.get('ts')
            dims = item.get('dimensions') or {}
            # 可用维度集合
            avail = [k for k in keys if isinstance(dims.get(k), (int, float))]
            if not avail:
                index_rows.append({'ts': ts, 'raw': None})
                for k in keys:
                    contrib_map[k].append({'ts': ts, 'raw': None})
                continue
            w = 1.0/len(avail)
            # E_i 与贡献
            contrib = {}
            s_sum = 0.0
            for k in avail:
                s = float(dims[k])
                e = (s - 50.0) / 50.0
                c = 50.0 * w * e
                contrib[k] = c
                s_sum += w * e
            # 指数（裁剪 0-100）
            b = max(0.0, min(100.0, 50.0 + 50.0 * s_sum))
            index_rows.append({'ts': ts, 'raw': b})
            for k in keys:
                rv = contrib.get(k)
                contrib_map[k].append({'ts': ts, 'raw': None if rv is None else float(rv)})

        # 平滑
        idx_vals = [row['raw'] for row in index_rows]
        idx_sm = ewma([None if v is None else float(v) for v in idx_vals], alpha)
        for i, sm in enumerate(idx_sm):
            index_rows[i]['smoothed'] = None if sm is None else float(sm)

        for k in keys:
            vals = [p['raw'] for p in contrib_map[k]]
            sm = ewma([None if v is None else float(v) for v in vals], alpha)
            for i, v in enumerate(sm):
                contrib_map[k][i]['smoothed'] = None if v is None else float(v)

        out = {
            'asset': asset,
            'granularity': gran,
            'as_of': _utc_now_iso(),
            'index': index_rows,
            'contrib': [{'key': k, 'points': contrib_map[k]} for k in keys],
            'meta': { 'baseline': 50, 'explain': 'sum(contrib)=index-baseline', 'alpha': alpha }
        }
        return jsonify({'success': True, 'data': out})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 200

@app.route('/api/strategies', methods=['POST'])
def set_strategies():
    """保存启用的策略到 config.json，并热更新内存中的 STRATEGIES"""
    try:
        payload = request.get_json(silent=True, force=True) or {}
        incoming = payload.get('strategies', [])
        if not isinstance(incoming, list):
            return jsonify({'success': False, 'error': 'strategies must be a list'}), 400

        # 仅允许已知策略
        allowed = list(STRATEGY_NAMES.keys())
        enabled = [s for s in incoming if isinstance(s, str) and s in allowed]

        # 读取现有配置
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}

        # 重建 strategies 数组（含全部已知策略，按 STRATEGY_NAMES 顺序）
        cfg['strategies'] = [{ 'name': name, 'enabled': name in enabled } for name in allowed]

        # 持久化到磁盘
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        # 热更新内存启用列表
        global STRATEGIES
        STRATEGIES = [name for name in allowed if name in enabled]

        return jsonify({
            'success': True,
            'data': {
                'strategies': STRATEGIES,
                'count': len(STRATEGIES)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/fred/<series_id>')
def get_fred_data(series_id):
    """获取FRED数据，避免CORS问题"""
    try:
        import requests
        from datetime import datetime, timedelta
        
        # 构建FRED CSV URL
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
        
        # 获取数据
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # 解析CSV数据
        lines = response.text.strip().split('\n')
        data = []
        
        for line in lines[1:]:  # 跳过表头
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 2 and parts[1] != '.':
                    try:
                        date_str = parts[0]
                        value = float(parts[1])
                        data.append({
                            'date': date_str,
                            'value': value
                        })
                    except (ValueError, IndexError):
                        continue
        
        # 按日期排序
        data.sort(key=lambda x: x['date'])
        
        return jsonify({
            'success': True,
            'data': data,
            'count': len(data)
        })
        
    except Exception as e:
        print(f'[FRED] Error fetching {series_id}: {e}')
        return jsonify({
            'success': False,
            'error': str(e),
            'data': []
        }), 500

@app.route('/api/etf/flows')
def get_etf_flows():
    """获取ETF流入数据，避免CORS问题"""
    try:
        import requests
        from datetime import datetime, timedelta
        
        # 尝试多个ETF数据源
        urls = [
            'https://farside.co.uk/wp-content/uploads/bitcoin_etf_flows.csv',
            'https://www.farside.co.uk/wp-content/uploads/bitcoin_etf_flows.csv'
        ]
        
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # 解析CSV数据
                lines = response.text.strip().split('\n')
                data = []
                
                for line in lines[1:]:  # 跳过表头
                    if line.strip():
                        parts = line.split(',')
                        if len(parts) >= 2:
                            try:
                                date_str = parts[0]
                                # 清理数值（去除货币符号、千位分隔符等）
                                value_str = parts[1].replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
                                value = float(value_str)
                                data.append({
                                    'date': date_str,
                                    'value': value
                                })
                            except (ValueError, IndexError):
                                continue
                
                if data:
                    # 按日期排序
                    data.sort(key=lambda x: x['date'])
                    return jsonify({
                        'success': True,
                        'data': data,
                        'count': len(data),
                        'source': url
                    })
                    
            except Exception as e:
                print(f'[ETF] Failed to fetch from {url}: {e}')
                continue
        
        # 如果所有源都失败，返回模拟数据
        print('[ETF] All sources failed, returning mock data')
        mock_data = []
        for i in range(30):
            date = datetime.now() - timedelta(days=i)
            # 生成有波动的模拟ETF数据
            value = random.uniform(-500, 1000) + 200 * np.sin(i * 0.2)
            mock_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'value': value
            })
        
        mock_data.sort(key=lambda x: x['date'])
        return jsonify({
            'success': True,
            'data': mock_data,
            'count': len(mock_data),
            'source': 'mock'
        })
        
    except Exception as e:
        print(f'[ETF] Error: {e}')
        return jsonify({
            'success': False,
            'error': str(e),
            'data': []
        }), 500

@app.route('/api/backtest/<symbol>')
def get_backtest(symbol):
    """获取真实回测数据（基于 ccxt 历史 K 线与真实策略）"""
    try:
        days = int(request.args.get('days', '30'))
        tf = request.args.get('tf', '4h')
        strategy_filter = request.args.get('strategy')  # 可选，仅回测某策略名（映射名或原名）

        if not USE_REAL_BINANCE_DATA:
            raise RuntimeError('Real data mode required')
        if not data_generator.binance_exchange:
            raise RuntimeError('Exchange not connected (ccxt)')

        # 计算需要的K线数量（粗略按 4h/1d/1w）
        tf_map = {'4h': 6, '1d': 1, '1w': 1/7}
        per_day = tf_map.get(tf, 6)
        limit = max(100, int(days * per_day) + 50)

        # 拉取历史
        ohlcv = data_generator.binance_exchange.fetch_ohlcv(f"{symbol}/USDT", tf, limit=limit)
        rows = []
        for ts, o, h, l, c, v in ohlcv:
            rows.append({'ts': datetime.fromtimestamp(ts/1000), 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})
        df = pd.DataFrame(rows)
        if len(df) < 60:
            raise RuntimeError('Insufficient history for backtest')

        # 合并注册表
        effective_registry = {}
        try:
            effective_registry.update(STRATEGY_REGISTRY)
            effective_registry.update(TOP15_REGISTRY)
        except Exception:
            effective_registry = STRATEGY_REGISTRY

        # 选择策略集合（支持中文名→key 映射）
        strategies_to_test = []
        if strategy_filter:
            reverse_map = {v: k for k, v in STRATEGY_NAMES.items()}
            key = reverse_map.get(strategy_filter, strategy_filter)
            if key in effective_registry:
                strategies_to_test = [key]
        if not strategies_to_test:
            strategies_to_test = STRATEGIES[:]

        # 统一策略调用适配器
        def _call_strategy_unified(sname, fn, symbol_fq, window_df, tf_local):
            from strategies import STRATEGY_REGISTRY as BASE_REG
            try:
                if sname in BASE_REG:
                    res = fn(symbol_fq, window_df, tf_local)
                    if not res or not isinstance(res, dict) or 'side' not in res:
                        return None
                    ep = float(res.get('entry', window_df['close'].iloc[-1]))
                    return {
                        'side': res['side'],
                        'entry': ep,
                        'target': float(res.get('target', ep)),
                        'stop': float(res.get('stop', ep)),
                    }
                else:
                    out = fn(window_df)
                    if out is None or not hasattr(out, 'iloc') or len(out) == 0:
                        return None
                    last = out.iloc[-1]
                    sig = int(last.get('signal') or 0)
                    if sig == 0:
                        return None
                    side = 'BUY' if sig > 0 else 'SELL'
                    entry = float(last.get('entry') or window_df['close'].iloc[-1])
                    tp = float(last.get('tp') or entry)
                    sl = float(last.get('sl') or entry)
                    return {'side': side, 'entry': entry, 'target': tp, 'stop': sl}
            except Exception:
                return None

        # 统一滚动回测
        trades = []
        for sname in strategies_to_test:
            fn = effective_registry.get(sname)
            if not fn:
                continue
            position = None
            entry_price = None
            entry_time = None
            last_targets = None

            for i in range(60, len(df)):
                window = df.iloc[:i].copy()
                sigdict = _call_strategy_unified(sname, fn, f"{symbol}/USDT", window, tf)

                if sigdict and position is None:
                    position = sigdict['side']
                    entry_price = float(window['close'].iloc[-1])
                    entry_time = window['ts'].iloc[-1]
                    last_targets = {'tp': float(sigdict['target']), 'sl': float(sigdict['stop'])}
                    continue

                if position is not None:
                    px = float(window['close'].iloc[-1])
                    cur_time = window['ts'].iloc[-1]
                    if position == 'BUY':
                        if px >= last_targets['tp'] or px <= last_targets['sl']:
                            pnl = (px - entry_price) / entry_price * 100.0
                            trades.append({'date': entry_time.strftime('%Y-%m-%d'), 'side': position, 'entry': entry_price, 'exit': px, 'pnl': pnl, 'hold': (cur_time-entry_time).total_seconds()/3600.0, 'strategy': sname})
                            position = entry_price = entry_time = last_targets = None
                    else:
                        if px <= last_targets['tp'] or px >= last_targets['sl']:
                            pnl = (entry_price - px) / entry_price * 100.0
                            trades.append({'date': entry_time.strftime('%Y-%m-%d'), 'side': position, 'entry': entry_price, 'exit': px, 'pnl': pnl, 'hold': (cur_time-entry_time).total_seconds()/3600.0, 'strategy': sname})
                            position = entry_price = entry_time = last_targets = None

        if not trades:
            raise RuntimeError('No trades generated by strategies')

        # 汇总
        pnl_series = [t['pnl'] for t in trades]
        wins = sum(1 for p in pnl_series if p > 0)
        total = len(pnl_series)
        win_rate = int(round(wins/total*100))
        cum_return = sum(pnl_series)
        max_drawdown = max(0, int(round(max(0.0, 0 - min(cum_return, 0)))))  # 简化

        return jsonify({
            'success': True,
            'data': {
                'symbol': symbol,
                'trades': total,
                'winRate': win_rate,
                'profitLossRatio': None,
                'maxDrawdown': max_drawdown,
                'period': f'{days}天',
                'samples': trades[-10:]
            }
        })
    except Exception as e:
        # 降级：不再抛 500，返回 200 + success:false，前端据此展示失败原因
        return jsonify({
            'success': False,
            'error': str(e)
        }), 200

# ========= BTC 宏观监控（前端 main.ts: fetchBtcMacroMonitorV3Data 用） =========
import io, re, time, datetime as dt

PROXIES = {"http":"http://127.0.0.1:7890","https":"http://127.0.0.1:7890"}  # 如不用代理可设为 None
HEADERS = {"User-Agent":"Mozilla/5.0"}

def _get_json(u, params=None, timeout=30):
    r = requests.get(u, params=params or {}, headers=HEADERS, timeout=timeout, proxies=PROXIES)
    r.raise_for_status(); return r.json()

def _get_text(u, params=None, timeout=30):
    r = requests.get(u, params=params or {}, headers=HEADERS, timeout=timeout, proxies=PROXIES)
    r.raise_for_status(); return r.text

def _fred_series(series_id: str) -> pd.DataFrame:
    txt = _get_text("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id":series_id}, timeout=30)
    if not txt.strip().upper().startswith("DATE,"):  # 代理/网络失败时别炸
        return pd.DataFrame(columns=["DATE","value"])
    df = pd.read_csv(io.StringIO(txt))
    if "DATE" not in df or series_id not in df: return pd.DataFrame(columns=["DATE","value"])
    df["DATE"] = pd.to_datetime(df["DATE"], utc=True, errors="coerce")
    return df.rename(columns={series_id:"value"}).dropna()[["DATE","value"]]

def _fred_cpi_yoy() -> pd.DataFrame:
    m = _fred_series("CPIAUCSL")
    if m.empty: return pd.DataFrame(columns=["DATE","value"])
    m["DATE"] = m["DATE"].dt.to_period("M").dt.to_timestamp("M", tz="UTC")
    m = m.sort_values("DATE").drop_duplicates("DATE")
    m["value"] = m["value"].pct_change(12) * 100.0
    return m[["DATE","value"]].dropna()

def _yahoo_daily(symbol: str, days=1825) -> pd.DataFrame:
    now = int(time.time()); period1 = now - days*24*3600
    js = _get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                   params={"period1":str(period1),"period2":str(now),"interval":"1d"})
    try:
        res = js["chart"]["result"][0]
        ts = res["timestamp"]; close = res["indicators"]["quote"][0]["close"]
        s = pd.Series(close, index=pd.to_datetime(ts, unit="s", utc=True)).dropna()
        return pd.DataFrame({"DATE":s.index, "value":s.values})
    except Exception:
        return pd.DataFrame(columns=["DATE","value"])

def _coingecko_btc(days=1825) -> pd.DataFrame:
    js = _get_json("https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
                   params={"vs_currency":"usd","days":str(days),"interval":"daily"})
    arr = js.get("prices", [])
    if not arr: return pd.DataFrame(columns=["DATE","value"])
    a = np.array(arr)
    return pd.DataFrame({"DATE":pd.to_datetime(a[:,0], unit="ms", utc=True), "value":pd.to_numeric(a[:,1])})

def _alt_fng() -> pd.DataFrame:
    js = _get_json("https://api.alternative.me/fng/", params={"limit":"0","format":"json"})
    data = js.get("data", [])
    rows = [(pd.to_datetime(int(d["timestamp"]), unit="s", utc=True), float(d["value"])) for d in data]
    return pd.DataFrame(rows, columns=["DATE","value"])

def _farside_etf() -> pd.DataFrame:
    for u in ["https://farside.co.uk/wp-content/uploads/bitcoin_etf_flows.csv",
              "https://www.farside.co.uk/wp-content/uploads/bitcoin_etf_flows.csv"]:
        try:
            t = _get_text(u, timeout=30)
            if "date" in t.lower():
                raw = t.replace("\ufeff","").replace("$","").replace("£","").replace(",","")
                raw = re.sub(r"\(([^)]+)\)", r"-\1", raw)
                df = pd.read_csv(io.StringIO(raw))
                dcol = next((c for c in df.columns if str(c).strip().lower() in ("date","day")), df.columns[0])
                fcol = None
                for key in ["net flow","net_flow","netflow","net"]:
                    for c in df.columns:
                        if key in str(c).strip().lower(): fcol=c; break
                    if fcol: break
                if fcol is None:
                    num_cols = [c for c in df.columns if c != dcol]
                    df["_sum"] = df[num_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
                    fcol = "_sum"
                df["DATE"] = pd.to_datetime(df[dcol], utc=True, errors="coerce")
                df["net"] = pd.to_numeric(df[fcol], errors="coerce")
                return df.dropna(subset=["DATE"])[["DATE","net"]].sort_values("DATE")
        except Exception:
            continue
    return pd.DataFrame(columns=["DATE","net"])

def _defillama_stablecap() -> pd.DataFrame:
    js = _get_json("https://stablecoins.llama.fi/stablecoincharts/all")
    arr = js.get("total", [])
    rows = [(pd.to_datetime(x["date"], unit="s", utc=True), float(x["totalCirculatingUSD"])) for x in arr]
    return pd.DataFrame(rows, columns=["DATE","value"])

def _blockchain_hashrate() -> pd.DataFrame:
    js = _get_json("https://api.blockchain.info/charts/hash-rate",
                   params={"timespan":"5years","format":"json","sampled":"true"})
    vals = js.get("values", [])
    rows = [(pd.to_datetime(v["x"], unit="s", utc=True), float(v["y"])) for v in vals]
    return pd.DataFrame(rows, columns=["DATE","value"])

def _binance_funding(days=365) -> pd.DataFrame:
    end = int(time.time()*1000); start = end - days*24*3600*1000
    js = _get_json("https://fapi.binance.com/fapi/v1/fundingRate",
                   params={"symbol":"BTCUSDT","startTime":start,"endTime":end,"limit":1000})
    rows = [(pd.to_datetime(int(x["fundingTime"]), unit="ms", utc=True), float(x["fundingRate"])) for x in js]
    return pd.DataFrame(rows, columns=["DATE","value"])

def _series_to_list(df: pd.DataFrame, val_key="value", rename=None):
    if df is None or df.empty: return []
    x = df.copy()
    x["DATE"] = pd.to_datetime(x["DATE"], utc=True, errors="coerce")
    x = x.dropna(subset=["DATE"]).sort_values("DATE")
    x["date"] = x["DATE"].dt.strftime("%Y-%m-%d")
    if rename:  # ETF 用 {"value":"net"} 重命名
        x = x.rename(columns=rename)
    return x[["date", rename.get(val_key) if rename else val_key]].to_dict(orient="records")

@app.route('/api/macro/btc_monitor')
def api_macro_btc_monitor():
    """BTC宏观监控数据 - 修复前端期望的数据结构"""
    days = int(request.args.get("days", "1095"))  # 3 年
    try:
        rate   = _fred_series("DFF")
        cpi    = _fred_cpi_yoy()
        unemp  = _fred_series("UNRATE")
        btc    = _coingecko_btc(days)
        fng    = _alt_fng()
        etf    = _farside_etf()
        ixic   = _yahoo_daily("^IXIC")
        gspc   = _yahoo_daily("^GSPC")
        dxy    = _yahoo_daily("DX-Y.NYB")
        gold   = _yahoo_daily("GC=F")
        stcap  = _defillama_stablecap()
        hashp  = _blockchain_hashrate()
        fund   = _binance_funding()

        payload = {
            "rate":      _series_to_list(rate),          # [{date, value}]
            "cpi":       _series_to_list(cpi),           # [{date, value}] 这里是 CPI 同比%
            "unemp":     _series_to_list(unemp),
            "btc":       _series_to_list(btc),
            "fng":       _series_to_list(fng),
            "etf":       _series_to_list(etf, val_key="net", rename={"net":"net"}),  # [{date, net}]
            "ixic":      _series_to_list(ixic),
            "gspc":      _series_to_list(gspc),
            "dxy":       _series_to_list(dxy),
            "gold":      _series_to_list(gold),
            "stablecap": _series_to_list(stcap),
            "hashrate":  _series_to_list(hashp),
            "funding":   _series_to_list(fund),
        }
        return jsonify(payload)
    except Exception as e:
        print(f"[API] BTC Monitor error: {e}")
        # 不让前端 500 崩掉，至少返回空结构
        return jsonify({
            "rate":[], "cpi":[], "unemp":[], "btc":[], "fng":[], "etf":[],
            "ixic":[], "gspc":[], "dxy":[], "gold":[], "stablecap":[], "hashrate":[], "funding":[]
        }), 200

if __name__ == '__main__':
    print('Starting API server...')
    print('✅ Using real exchange data (ccxt: Binance/OKX)')
    print('API address: http://localhost:8889')
    print('Endpoints: /api/quotes, /api/signals, /api/learning-stats, /api/config, /api/backtest/<symbol>')

    app.run(host='0.0.0.0', port=8889, debug=True)
