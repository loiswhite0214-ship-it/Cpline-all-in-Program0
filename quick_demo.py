import streamlit as st
st.caption("THIS IS quick_demo.py")

#!/usr/bin/env python3
"""
熬鹰计划策略核心逻辑演示
无需外部依赖，快速展示三大策略算法
"""

import json
import random
from datetime import datetime

def demo_vegas_tunnel():
    """Vegas通道策略演示"""
    print("🎯 策略1: Vegas通道 (EMA55/144)")
    print("-" * 40)

    # 模拟价格数据
    prices = [65000 + random.uniform(-1000, 1000) for _ in range(150)]

    # 简化EMA计算
    def simple_ema(data, period):
        alpha = 2 / (period + 1)
        ema = [data[0]]
        for price in data[1:]:
            ema.append(alpha * price + (1 - alpha) * ema[-1])
        return ema

    ema55 = simple_ema(prices, 55)
    ema144 = simple_ema(prices, 144)

    current_price = prices[-1]
    upper_bound = max(ema55[-1], ema144[-1])
    lower_bound = min(ema55[-1], ema144[-1])

    print(f"当前价格: ${current_price:,.2f}")
    print(f"通道上轨: ${upper_bound:,.2f} (EMA55/144较大值)")
    print(f"通道下轨: ${lower_bound:,.2f} (EMA55/144较小值)")

    if current_price > upper_bound:
        print("🟢 信号: BUY - 价格突破通道上轨")
        print(f"   策略逻辑: 当价格有效突破EMA55和EMA144形成的通道时，表明趋势强劲")
    elif current_price < lower_bound:
        print("🔴 信号: SELL - 价格跌破通道下轨")
        print(f"   策略逻辑: 当价格有效跌破通道时，表明下跌趋势确立")
    else:
        print("⚪ 无信号 - 价格在通道内震荡")

    return current_price > upper_bound or current_price < lower_bound

def demo_chan_simplified():
    """简化缠论策略演示"""
    print("\n🎯 策略2: 简化缠论 (SMA20/60金叉死叉)")
    print("-" * 40)

    # 模拟SMA数据
    sma20_prev, sma20_now = 3180, 3205
    sma60_prev, sma60_now = 3190, 3195

    print(f"SMA20: {sma20_prev:.2f} → {sma20_now:.2f}")
    print(f"SMA60: {sma60_prev:.2f} → {sma60_now:.2f}")

    # 检查交叉
    golden_cross = sma20_prev <= sma60_prev and sma20_now > sma60_now
    death_cross = sma20_prev >= sma60_prev and sma20_now < sma60_now

    if golden_cross:
        print("🟢 信号: BUY - SMA20上穿SMA60 (金叉)")
        print("   策略逻辑: 短期均线上穿长期均线，表明上涨动能增强")
        return True
    elif death_cross:
        print("🔴 信号: SELL - SMA20下穿SMA60 (死叉)")
        print("   策略逻辑: 短期均线下穿长期均线，表明下跌动能增强")
        return True
    else:
        print("⚪ 无信号 - 均线未发生交叉")
        return False

def demo_macd():
    """MACD策略演示"""
    print("\n🎯 策略3: MACD交叉")
    print("-" * 40)

    # 模拟MACD数据
    dif_prev, dif_now = -15.2, 8.7
    dea_prev, dea_now = 12.3, 5.8
    hist_prev = dif_prev - dea_prev  # -27.5
    hist_now = dif_now - dea_now     # 2.9

    print(f"DIF: {dif_prev:.2f} → {dif_now:.2f}")
    print(f"DEA: {dea_prev:.2f} → {dea_now:.2f}")
    print(f"HIST: {hist_prev:.2f} → {hist_now:.2f}")

    # 检查MACD交叉
    golden_cross = hist_prev <= 0 and hist_now > 0
    death_cross = hist_prev >= 0 and hist_now < 0

    if golden_cross:
        print("🟢 信号: BUY - MACD金叉 (HIST由负转正)")
        print("   策略逻辑: DIF上穿DEA，表明短期上涨动量强于长期")
        return True
    elif death_cross:
        print("🔴 信号: SELL - MACD死叉 (HIST由正转负)")
        print("   策略逻辑: DIF下穿DEA，表明短期下跌动量强于长期")
        return True
    else:
        print("⚪ 无信号 - MACD未发生交叉")
        return False

def demo_backtest():
    """回测引擎演示"""
    print("\n🧪 回测引擎演示")
    print("=" * 50)

    # 模拟回测结果
    results = {
        "vegas_tunnel": {"wins": 23, "losses": 12, "total": 35, "winrate": 65.7},
        "chan_simplified": {"wins": 18, "losses": 14, "total": 32, "winrate": 56.3},
        "macd": {"wins": 31, "losses": 19, "total": 50, "winrate": 62.0}
    }

    print("策略回测结果 (最近100根K线):")
    for strategy, data in results.items():
        print(f"\n📊 {strategy}:")
        print(f"   胜率: {data['winrate']:.1f}% ({data['wins']}/{data['total']})")
        print(f"   盈亏比: {random.uniform(1.2, 2.8):.1f}")
        print(f"   最大回撤: {random.uniform(3, 12):.1f}%")

    return results

def main():
    print("🚀 熬鹰计划 - 策略核心逻辑演示")
    print("=" * 60)
    print(f"⏰ 演示时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 演示三个策略
    signals = []
    signals.append(demo_vegas_tunnel())
    signals.append(demo_chan_simplified())
    signals.append(demo_macd())

    # 演示回测
    demo_backtest()

    # 总结
    print("\n" + "=" * 60)
    print("📋 演示总结:")
    print(f"🎯 本次演示触发信号数: {sum(signals)}/3")
    print("\n💡 核心算法说明:")
    print("   • Vegas通道: 基于EMA55/144的趋势突破系统")
    print("   • 简化缠论: 双均线金叉死叉，简化版缠中说禅理论")
    print("   • MACD交叉: 经典动量指标，捕捉趋势转换点")
    print("\n🔄 实际系统:")
    print("   • 连接币安API获取实时K线数据")
    print("   • 每根K线收盘后自动计算策略信号")
    print("   • ADX/ATR过滤减少假信号")
    print("   • 支持4H/1D/1W多周期分析")

if __name__ == "__main__":
    main()
