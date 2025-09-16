  }
}

// 启动应用
const app = new MobileTradingDashboard();

// 桥接：若未定义全局 openQuickBacktest，则提供稳定版
if (!(window as any).openQuickBacktest) {
  (window as any).openQuickBacktest = (symbol: string, strategy: string) => openQuickBacktestStable(symbol, strategy);
}

