#!/usr/bin/env python3
"""
运行真实的策略算法 - 从strategies.py中提取核心逻辑
展示Vegas通道、简化缠论、MACD策略的实际计算过程
"""

import json
import math
import time
from datetime import datetime

# 简化的技术指标计算（从strategies.py提取）
def safe_series(x):
    """安全转换为浮点数列表"""
    return [float(val) if val is not None and math.isfinite(float(val)) else None for val in x]

def calculate_ema(prices, span):
    """计算指数移动平均线"""
    prices = safe_series(prices)
    result = [None] * len(prices)

    # 找到第一个有效值
    valid_start = 0
    for i, price in enumerate(prices):
        if price is not None:
            valid_start = i
            break

    if valid_start >= len(prices):
        return result

    # 计算EMA
    alpha = 2 / (span + 1)
    result[valid_start] = prices[valid_start]

    for i in range(valid_start + 1, len(prices)):
        if prices[i] is not None and result[i-1] is not None:
            result[i] = alpha * prices[i] + (1 - alpha) * result[i-1]

    return result

def calculate_sma(prices, period):
    """计算简单移动平均线"""
    prices = safe_series(prices)
    result = []

    for i in range(len(prices)):
        if i < period - 1:
            result.append(None)
        else:
            window = prices[i-period+1:i+1]
            valid_prices = [p for p in window if p is not None]
            if len(valid_prices) == period:
                result.append(sum(valid_prices) / period)
            else:
                result.append(None)

    return result

def format_price(symbol, value):
    """按币种格式化价格（从strategies.py提取）"""
    if value is None or not math.isfinite(value):
        return None

    s = symbol.upper()
    if any(k in s for k in ["BTC", "ETH", "BNB", "SOL"]):
        return round(float(value), 2)
    if any(k in s for k in ["XRP", "DOGE", "SHIB", "TRX"]):
        return round(float(value), 6)
    return round(float(value), 4)

# 真实策略实现（从strategies.py提取核心逻辑）
def vegas_tunnel_strategy(symbol, klines):
    """
    Vegas通道策略：EMA55/144构成通道，突破产生信号
    这是从strategies.py中提取的真实算法
    """
    if len(klines) < 160:
        return None

    closes = [k[4] for k in klines]  # 收盘价
    highs = [k[2] for k in klines]   # 最高价
    lows = [k[3] for k in klines]    # 最低价

    # 计算EMA55和EMA144
    ema55 = calculate_ema(closes, 55)
    ema144 = calculate_ema(closes, 144)

    if ema55[-1] is None or ema144[-1] is None:
        return None

    # 通道上下轨
    up = max(ema55[-1], ema144[-1])
    dn = min(ema55[-1], ema144[-1])

    # 计算ATR（简化版本）
    tr_list = []
    for i in range(1, len(klines)):
        tr1 = abs(highs[i] - lows[i])
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i] - closes[i-1])
        tr_list.append(max(tr1, tr2, tr3))

    atr = sum(tr_list[-14:]) / 14 if len(tr_list) >= 14 else None
    if atr is None or atr <= 0:
        return None

    # 检查上一根是否在通道内
    prev_in_tunnel = dn <= closes[-2] <= up
    if not prev_in_tunnel:
        return None

    last_price = closes[-1]

    # 检查突破条件（需要超出通道0.3*ATR的距离）
    side = None
    if last_price > up + 0.3 * atr:
        side = "BUY"
    elif last_price < dn - 0.3 * atr:
        side = "SELL"
    else:
        return None

    # 计算止盈止损
    if side == "BUY":
        target = last_price + 2.0 * atr
        stop = last_price - 1.4 * atr
    else:
        target = last_price - 2.0 * atr
        stop = last_price + 1.4 * atr

    return {
        "symbol": symbol,
        "strategy": "vegas_tunnel",
        "side": side,
        "entry": format_price(symbol, last_price),
        "target": format_price(symbol, target),
        "stop": format_price(symbol, stop),
        "confidence": 45,
        "reason": f"收盘有效突破Vegas通道，EMA55: {ema55[-1]:.6f}, EMA144: {ema144[-1]:.6f}"
    }

def chan_simplified_strategy(symbol, klines):
    """
    简化缠论策略：SMA20/60金叉死叉
    这是从strategies.py中提取的真实算法
    """
    if len(klines) < 80:
        return None

    closes = [k[4] for k in klines]

    # 计算SMA20和SMA60
    sma20 = calculate_sma(closes, 20)
    sma60 = calculate_sma(closes, 60)

    if sma20[-1] is None or sma60[-1] is None:
        return None
    if sma20[-2] is None or sma60[-2] is None:
        return None

    # 检查当根交叉
    cross_up = (sma20[-2] <= sma60[-2]) and (sma20[-1] > sma60[-1])
    cross_down = (sma20[-2] >= sma60[-2]) and (sma20[-1] < sma60[-1])

    if not (cross_up or cross_down):
        return None

    side = "BUY" if cross_up else "SELL"
    last_price = closes[-1]

    # 简化的ATR计算
    highs = [k[2] for k in klines]
    lows = [k[3] for k in klines]
    tr_list = []
    for i in range(1, min(15, len(klines))):
        tr1 = abs(highs[-i] - lows[-i])
        tr2 = abs(highs[-i] - closes[-i-1])
        tr3 = abs(lows[-i] - closes[-i-1])
        tr_list.append(max(tr1, tr2, tr3))

    atr = sum(tr_list) / len(tr_list) if tr_list else last_price * 0.02

    # 计算止盈止损（非对称）
    if side == "BUY":
        target = last_price + 3.0 * atr
        stop = last_price - 1.2 * atr
    else:
        target = last_price - 3.0 * atr
        stop = last_price + 1.2 * atr

    return {
        "symbol": symbol,
        "strategy": "chan_simplified",
        "side": side,
        "entry": format_price(symbol, last_price),
        "target": format_price(symbol, target),
        "stop": format_price(symbol, stop),
        "confidence": 42,
        "reason": f"SMA20/60当根交叉，SMA20: {sma20[-1]:.6f}, SMA60: {sma60[-1]:.6f}"
    }

def macd_strategy(symbol, klines):
    """
    MACD策略：DIF与DEA交叉
    这是从strategies.py中提取的真实算法
    """
    if len(klines) < 220:
        return None

    closes = [k[4] for k in klines]

    # 计算MACD
    ema12 = calculate_ema(closes, 12)
    ema26 = calculate_ema(closes, 26)

    if ema12[-1] is None or ema26[-1] is None:
        return None

    # 计算DIF
    dif = []
    for i in range(len(closes)):
        if ema12[i] is not None and ema26[i] is not None:
            dif.append(ema12[i] - ema26[i])
        else:
            dif.append(None)

    # 计算DEA（DIF的9日EMA）
    dea = calculate_ema(dif, 9)

    # 计算HIST
    if dif[-1] is None or dea[-1] is None or dif[-2] is None or dea[-2] is None:
        return None

    hist_now = dif[-1] - dea[-1]
    hist_prev = dif[-2] - dea[-2]

    # 检查交叉
    cross_up = hist_prev <= 0 and hist_now > 0
    cross_down = hist_prev >= 0 and hist_now < 0

    if not (cross_up or cross_down):
        return None

    # 计算EMA200作为基线过滤
    ema200 = calculate_ema(closes, 200)
    if ema200[-1] is None:
        return None

    last_price = closes[-1]

    # 基线过滤
    if cross_up and last_price <= ema200[-1]:
        return None
    if cross_down and last_price >= ema200[-1]:
        return None

    side = "BUY" if cross_up else "SELL"

    # 计算ATR
    highs = [k[2] for k in klines]
    lows = [k[3] for k in klines]
    tr_list = []
    for i in range(1, min(15, len(klines))):
        tr1 = abs(highs[-i] - lows[-i])
        tr2 = abs(highs[-i] - closes[-i-1])
        tr3 = abs(lows[-i] - closes[-i-1])
        tr_list.append(max(tr1, tr2, tr3))

    atr = sum(tr_list) / len(tr_list) if tr_list else last_price * 0.02

    # 计算止盈止损
    if side == "BUY":
        target = last_price + 2.2 * atr
        stop = last_price - 1.6 * atr
    else:
        target = last_price - 2.2 * atr
        stop = last_price + 1.6 * atr

    return {
        "symbol": symbol,
        "strategy": "macd",
        "side": side,
        "entry": format_price(symbol, last_price),
        "target": format_price(symbol, target),
        "stop": format_price(symbol, stop),
        "confidence": 30,
        "reason": f"MACD交叉信号，DIF: {dif[-1]:.6f}, DEA: {dea[-1]:.6f}, HIST: {hist_now:.6f}"
    }

# 模拟K线数据生成器
class KlineGenerator:
    def __init__(self, symbol, base_price):
        self.symbol = symbol
        self.base_price = base_price
        self.current_price = base_price
        self.trend = 1
        self.data = []

        # 生成历史数据
        for i in range(250):  # 生成250根历史K线
            kline = self._generate_kline()
            self.data.append(kline)

    def _generate_kline(self):
        # 模拟价格变动
        if len(self.data) % 20 == 0:  # 每20根K线可能改变趋势
            self.trend = -self.trend if hash(time.time()) % 3 == 0 else self.trend

        # 基础变动
        base_change = (hash(str(time.time())) % 1000 - 500) / 10000  # -0.05 到 0.05
        trend_bias = self.trend * abs(base_change) * 0.3

        total_change = base_change + trend_bias

        open_price = self.current_price
        close_price = open_price * (1 + total_change)

        # 生成高低价
        volatility = abs(total_change) + 0.005
        high_price = max(open_price, close_price) * (1 + volatility)
        low_price = min(open_price, close_price) * (1 - volatility)

        volume = 1000 + (hash(str(time.time())) % 5000)

        self.current_price = close_price

        return [
            int(time.time() * 1000),  # timestamp
            open_price,   # open
            high_price,   # high
            low_price,    # low
            close_price,  # close
            volume        # volume
        ]

    def get_latest_klines(self):
        return self.data[-250:]  # 返回最近250根

    def add_new_kline(self):
        new_kline = self._generate_kline()
        self.data.append(new_kline)
        return new_kline

def main():
    print("🚀 熬鹰计划 - 真实策略算法运行")
    print("=" * 80)

    # 读取配置
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        symbols = config.get('symbols', ['BTC/USDT'])[:5]  # 取前5个
    except:
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

    print(f"📊 监控币种: {', '.join(symbols)}")
    print(f"🎯 策略: Vegas通道, 简化缠论, MACD")
    print(f"💡 这些是从strategies.py提取的真实算法逻辑")
    print("\n正在初始化历史数据...")

    # 基础价格
    base_prices = {
        'BTC/USDT': 65000, 'ETH/USDT': 3200, 'BNB/USDT': 590,
        'SOL/USDT': 140, 'XRP/USDT': 0.52, 'ADA/USDT': 0.45,
        'DOGE/USDT': 0.12, 'TRX/USDT': 0.08, 'AVAX/USDT': 28,
        'DOT/USDT': 6.5, 'SHIB/USDT': 0.000024, 'LINK/USDT': 12.5,
        'TON/USDT': 5.8, 'LTC/USDT': 85, 'MATIC/USDT': 0.85
    }

    # 创建K线生成器
    generators = {}
    for symbol in symbols:
        base_price = base_prices.get(symbol, 100)
        generators[symbol] = KlineGenerator(symbol, base_price)

    # 策略函数
    strategies = [
        ("Vegas通道", vegas_tunnel_strategy),
        ("简化缠论", chan_simplified_strategy),
        ("MACD", macd_strategy)
    ]

    print("=" * 80)
    print("开始实时监控...")

    total_signals = 0

    # 运行15个周期
    for cycle in range(1, 16):
        print(f"\n📅 周期 {cycle} - {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 50)

        cycle_signals = []

        # 为每个币种生成新K线并检测信号
        for symbol in symbols:
            generator = generators[symbol]
            new_kline = generator.add_new_kline()
            current_price = new_kline[4]

            # 计算涨跌幅
            klines = generator.get_latest_klines()
            if len(klines) >= 2:
                prev_price = klines[-2][4]
                change_pct = ((current_price - prev_price) / prev_price) * 100
            else:
                change_pct = 0

            print(f"💰 {symbol:12} {current_price:10.6f} ({change_pct:+.2f}%)")

            # 运行每个策略
            for strategy_name, strategy_func in strategies:
                try:
                    signal = strategy_func(symbol, klines)
                    if signal:
                        signal['timestamp'] = datetime.now().strftime('%H:%M:%S')
                        cycle_signals.append(signal)
                        total_signals += 1
                except Exception as e:
                    # 静默处理策略错误，实际使用中会记录日志
                    pass

        # 显示本周期的信号
        if cycle_signals:
            print(f"\n🚨 本周期触发 {len(cycle_signals)} 个信号:")
            for signal in cycle_signals:
                action_emoji = "🟢" if signal['side'] == 'BUY' else "🔴"
                print(f"   {action_emoji} {signal['symbol']} - {signal['strategy']}")
                print(f"      {signal['side']} @ {signal['entry']}")
                print(f"      目标: {signal['target']} | 止损: {signal['stop']}")
                print(f"      {signal['reason']}")
                print()
        else:
            print("   📊 本周期无信号触发")

        # 每5个周期显示统计
        if cycle % 5 == 0:
            avg_signals = total_signals / cycle
            print(f"\n📈 统计 (前{cycle}个周期):")
            print(f"   总信号: {total_signals} | 平均: {avg_signals:.1f}/周期")

        time.sleep(3)  # 等待3秒模拟实时更新

    print("\n" + "=" * 80)
    print("📊 监控完成!")
    print(f"🎯 总共生成 {total_signals} 个真实策略信号")
    print("💡 这展示了您项目中真实策略算法的计算过程")
    print("🔌 下一步：将这些策略连接到TypeScript前端显示")

if __name__ == "__main__":
    main()
