# 保存为 probe.py
from dashboard import latest_signals, SYMBOLS   # ⚠ 如果 latest_signals 不在 dashboard.py，就改成对应文件名

def probe_cli():
    strategies = ["macd", "vegas_tunnel", "chan_simplified"]
    timeframe = "4h"
    sigs = latest_signals(SYMBOLS, timeframe, strategies)
    print(f"[Probe] Got {len(sigs)} signals")
    for s in sigs:
        print(f"{s['symbol']} | {s.get('strategy')} | {s.get('side')} | entry={s.get('entry')}")

if __name__ == "__main__":
    probe_cli()
