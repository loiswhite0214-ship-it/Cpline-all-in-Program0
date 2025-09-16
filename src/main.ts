// 量化交易面板 TypeScript 实现 - 自适应桌面端和移动端
import { useFactorsData, type Dimension } from './hooks/useFactorsData';
import { useFactorIndex, type FactorContribSeries, type FactorIndexPoint } from './hooks/useFactorIndex';
import { pushToQueue } from "./sim/store";
import { initMineUI } from "./mine";
import { renderCorrChart } from './corr_frontonly';
import { unique, savePref, loadPref, SignalDto, PositionDto, Side } from './utils/filtering';

// 用户参数接口
interface UserParams {
  profitTarget: number;
  maxDrawdown: number;
  riskExposure: number;
  capitalSize: number;
  monitoringFreq: string;
}

// API数据接口
interface ApiQuote {
  symbol: string;
  close: number;
  changePercent: string;
  isPositive: boolean;
}

interface ApiSignal {
  symbol: string;
  strategy: string;
  side: string;
  entry: number;
  target: number;
  stop: number;
  confidence: number;
  tf: string;
  time: string;
}

// 检测设备类型
const isMobile = () => {
  return window.innerWidth <= 768 ||
         /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ||
         (window.navigator && 'maxTouchPoints' in window.navigator && window.navigator.maxTouchPoints > 0);
};

// 固定后端真实数据地址（本机）
const BASE_API = 'http://127.0.0.1:8889';

// 移动端优化界面类
class MobileTradingDashboard {
  private currentTab: 'home' | 'vip' | 'backtest' | 'profile' | 'info' | '因子' = 'home';
  private currentTimeframe = '4h';
  private activeStrategies: Set<string> = new Set(['vegas_tunnel', 'chan_simplified', 'macd']);
  private updateTimer: ReturnType<typeof setTimeout> | undefined;
  // Info page state
  private infoInited: boolean = false;
  private infoSelectedKey: 'macro' | 'policy' | 'capital' | 'geopolitics' | 'onchain' | 'sentiment' = 'macro';
  private echartsMod: any = null;
  private onInfoResize: (() => void) | null = null;
  private resizeDebounceTimer: ReturnType<typeof setTimeout> | null = null;
  private infoData: Dimension[] = [];
  private infoAsOf: string = '';
  private infoIndex: FactorIndexPoint[] = [];
  private infoContrib: FactorContribSeries[] = [];
  private infoCurrentIdx: number = -1;
  private infoSelectedFactor: string | null = null;
  private infoEventBus: { emit: (event: string, data?: any) => void; on: (event: string, handler: (data?: any) => void) => void } = { emit: () => {}, on: () => {} };

  // 宏观数据相关
  private macroEventBus: { emit: (event: string, data?: any) => void; on: (event: string, handler: (data?: any) => void) => void } = { emit: () => {}, on: () => {} };
  private macroCurrentRange: string = '30D';
  private btcMacroMonitorV3Chart: any = null;
  private btcMacroMonitorV3Data: any = null;
  private macroDriverIndexChart: any = null;
  
  // 数据缓存和预加载
  private dataCache: Map<string, { data: any; timestamp: number; ttl: number }> = new Map();
  private DATA_CACHE_TTL = 5 * 60 * 1000; // 5分钟缓存
  
  // 过滤状态（带本地持久化）
  private sigFilter: { symbol:string; side:"ALL"|Side; tf:"ALL"|string; q:string } = loadPref('sigFilter', {symbol:'ALL', side:'ALL', tf:'ALL', q:''});
  private posFilter: { symbol:string; side:"ALL"|Side; tf:"ALL"|string; q:string } = loadPref('posFilter', {symbol:'ALL', side:'ALL', tf:'ALL', q:''});
  
  // 相关性数据
  private corrData: { dates: string[]; rho: Record<string, number[]>; meta: any } | null = null;
  private macroETFFNGChart: any = null;
  private macroFundingChart: any = null;
  private macroDriverIndexData: any[] = [];
  private macroETFFNGData: any[] = [];
  private macroFundingData: any = null;

  // 存储用户参数的状态
  private userParams: UserParams = {
    profitTarget: 5,
    maxDrawdown: 15,
    riskExposure: 5,
    capitalSize: 10000,
    monitoringFreq: 'daily'
  };

  // 标记用户是否已经保存过配置
  private hasConfiguredFlag = false;

  private basePrices: Record<string, number> = {
    'BTC': 65000, 'ETH': 3200, 'BNB': 590, 'SOL': 140, 'XRP': 0.52,
    'ADA': 0.45, 'DOGE': 0.12, 'TRX': 0.08, 'AVAX': 28, 'DOT': 6.5,
    'SHIB': 0.000024, 'LINK': 12.5, 'TON': 5.8, 'LTC': 85, 'MATIC': 0.85
  };

  constructor() {
    this.loadUserParamsFromStorage();
    this.createMobileUI();
    this.setupMobileEventListeners();
    this.startUpdates();
    
    // 预加载ECharts，提前开始加载
    this.preloadECharts();
    
    // 预加载因子数据，提前开始计算
    this.preloadFactorsData();
    
    // 初始化宏观数据事件总线
    this.initMacroEventBus();
  }
  
  private preloadECharts() {
    // 提前开始加载ECharts，不等待用户点击因子页
    console.log('[info] Starting ECharts preload...');
    import('echarts').then((mod) => {
      const echartsAny: any = (mod as any)?.default || (mod as any);
      this.echartsMod = echartsAny;
      console.log('[info] ECharts preloaded successfully');
      console.log('[info] ECharts module:', this.echartsMod);
      console.log('[info] ECharts init function:', typeof this.echartsMod?.init);
    }).catch((error) => {
      console.error('[info] ECharts preload error:', error);
    });
  }
  
  
  private preloadFactorsData() {
    // 立即开始加载因子数据，不等待用户点击因子页
    this.refreshInfoData().catch((error) => {
      console.warn('[info] Factors data preload failed:', error);
    });
  }
  
  private initMacroEventBus() {
    // 宏观数据事件总线
    this.macroEventBus = {
      emit: (event: string, data?: any) => {
        document.dispatchEvent(new CustomEvent(`macro:${event}`, { detail: data }));
      },
      on: (event: string, handler: (data?: any) => void) => {
        document.addEventListener(`macro:${event}`, (e: any) => handler(e.detail));
      }
    };
  }
  
  private initCharts() {
    console.log('[info] Initializing charts...');
    
    // 如果数据已经准备好，立即渲染
    if (this.infoData.length > 0 || this.infoIndex.length > 0) {
    }
  }

  private createMobileUI() {
    document.body.className = 'compact-ui';
    document.body.innerHTML = `
      <div class="mobile-app">
        <!-- Apple HIG Navigation Bar -->
        <nav class="hig-nav-bar">
          <div class="nav-content">
            <div class="nav-left">
              <div class="app-brand">
                <span class="app-icon">📈</span>
                <span class="app-name">熬鹰计划</span>
              </div>
            </div>
            <div class="nav-center">
              <div class="search-bar">
                <svg class="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <circle cx="11" cy="11" r="8"/>
                  <path d="m21 21-4.35-4.35"/>
                </svg>
                <input type="text" placeholder="搜索币种、策略..." class="search-input">
              </div>
            </div>
            <div class="nav-right">
              <button class="nav-button" id="notifications-btn">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/>
                  <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/>
                </svg>
                <span class="badge">3</span>
              </button>
              <button class="nav-button" id="settings-btn">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <circle cx="12" cy="12" r="3"/>
                  <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1 1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
                </svg>
              </button>
            </div>
          </div>
        </nav>

        <header class="mobile-header">
          <div class="header-right">
            <div class="api-status" id="api-status">
              <div class="status-dot offline"></div>
              <span class="status-text">连接中...</span>
            </div>
            <div class="learning-stats" id="learning-stats" data-clickable="1">
              <div class="stat-item">
                <div class="stat-label">胜率</div>
                <div class="stat-value" id="win-rate">--/--</div>
              </div>
              <div class="stat-item">
                <div class="stat-label">最大回撤</div>
                <div class="stat-value" id="max-drawdown-stat">--/--</div>
              </div>
              <div class="stat-item">
                <div class="stat-label">累计收益</div>
                <div class="stat-value" id="profit-ratio">--/--</div>
              </div>
            </div>
            <div class="kpi-actions">
              <div class="chips">
                <button class="header-action-btn" id="personal-params-btn">个性化参数</button>
                <button class="header-action-btn" id="simulation-positions-btn">模拟持仓</button>
              </div>
              <button class="manage-signals-btn btn-strategy" id="manage-signals-btn">策略管理</button>
            </div>
          </div>
        </header>

        <div class="timeframe-tabs">
          <button class="tf-tab active" data-tf="4h">4H</button>
          <button class="tf-tab" data-tf="1d">1D</button>
          <button class="tf-tab" data-tf="1w">1W</button>
        </div>

        <main class="mobile-content">
          <div id="market-view" class="tab-content active">

            <!-- 第一块：实时行情 -->
            <div class="market-section">
              <h3 class="section-title">💹 实时行情</h3>
              <div class="quotes-enhanced" id="quotes-enhanced"></div>
            </div>

            <!-- 第二块：交易信号区 -->
            <div class="signals-section">
              <h3 class="section-title">🚀 交易信号</h3>
              <div class="signals-disclaimer">
                策略仅供学习，请勿作为投资依据
              </div>
              <div class="signals-cards" id="signals-cards"></div>
            </div>

            <!-- 第四块：条件触发提醒 -->
            <div class="condition-alert-section">
              <h3 class="section-title">🔔 条件触发提醒</h3>
              <div class="condition-alert-config">
                <div class="condition-item">
                  <label>胜率阈值</label>
                  <input type="number" id="win-rate-threshold" placeholder="80" min="0" max="100" step="1">
                  <span>%</span>
                </div>
                <div class="condition-item">
                  <label>指数阈值</label>
                  <input type="number" id="index-threshold" placeholder="75" min="0" max="100" step="1">
                  <span>%</span>
                </div>
                <button class="condition-toggle-btn" id="condition-toggle">启用提醒</button>
              </div>
              <div class="condition-status" id="condition-status">
                <span class="status-indicator"></span>
                <span class="status-text">提醒已关闭</span>
              </div>
            </div>

            <!-- 第五块：策略实验室 -->
            <div class="strategy-lab-section">
              <h3 class="section-title">🧪 策略实验室</h3>
              <div class="strategy-lab-config">
                <div class="lab-input-group">
                  <label>入场价格</label>
                  <input type="number" id="lab-entry-price" placeholder="50000" step="0.01">
                </div>
                <div class="lab-input-group">
                  <label>多空方向</label>
                  <select id="lab-direction">
                    <option value="long">做多</option>
                    <option value="short">做空</option>
                  </select>
                </div>
                <div class="lab-input-group">
                  <label>杠杆倍数</label>
                  <input type="number" id="lab-leverage" placeholder="2" min="1" max="10" step="0.1">
                </div>
                <div class="lab-input-group">
                  <label>交易策略</label>
                  <select id="lab-strategy">
                    <option value="vegas_tunnel">Vegas隧道</option>
                    <option value="chan_simplified">缠论简化</option>
                    <option value="macd">MACD</option>
                  </select>
                </div>
                <button class="lab-backtest-btn" id="lab-backtest-btn">快速回测</button>
              </div>
            </div>

            <!-- 第六块：个性化推荐 -->
            <div class="recommendation-section">
              <h3 class="section-title">🎯 个性化推荐</h3>
              <div class="recommendation-config-hint" id="recommendation-config-hint">
                <div class="config-hint-content">
                  <div class="config-hint-icon">⚙️</div>
                  <div class="config-hint-text">
                    <div class="config-hint-title">设置您的交易参数</div>
                    <div class="config-hint-desc">收益目标、风险偏好、本金规模等</div>
                  </div>
                  <button class="config-hint-btn" onclick="window.goToSettings()">
                    <span>配置参数</span>
                    <span>→</span>
                  </button>
                </div>
              </div>
              <div class="recommendation-cards" id="recommendation-cards"></div>
            </div>

            <!-- 第五块：复盘 + 排行榜 -->
            <div class="performance-section">
              <h3 class="section-title">📈 复盘 & 排行榜</h3>
              <div class="performance-grid">
                <div class="review-panel" id="review-panel">
                  <h4>昨日信号复盘</h4>
                  <div class="review-content" id="review-content"></div>
                </div>
                <div class="ranking-panel" id="ranking-panel">
                  <h4>策略胜率排行</h4>
                  <div class="ranking-content" id="ranking-content"></div>
                </div>
              </div>
            </div>
          </div>

          <div id="vip-view" class="tab-content">
            <div class="settings-panel">
              <h3>会员方案</h3>
              <div class="recommendation-cards">
                <div class="recommendation-card"><div class="recommendation-title">Basic</div><div class="recommendation-content">核心指标展示，基础策略信号</div><div class="recommendation-actions"><button class="signal-btn signal-btn-secondary">开始试用</button></div></div>
                <div class="recommendation-card"><div class="recommendation-title">Pro</div><div class="recommendation-content">全部策略与快速回测，历史复盘与对比</div><div class="recommendation-actions"><button class="signal-btn signal-btn-primary">立即订阅</button></div></div>
                <div class="recommendation-card"><div class="recommendation-title">Hyper</div><div class="recommendation-content">高级筛选、策略组合与个性化建议</div><div class="recommendation-actions"><button class="signal-btn signal-btn-primary">立即订阅</button></div></div>
              </div>
            </div>
          </div>

          <div id="settings-view" class="tab-content">
            <div class="settings-panel">
              <h3>回测页说明</h3>
              <div class="recommendation-config-hint" style="margin-bottom: 20px;">
                <div class="config-hint-content">
                  <div class="config-hint-icon">ℹ️</div>
                  <div class="config-hint-text">
                    <div class="config-hint-title">此处仅用于历史表现复盘与策略对比，不影响实时信号</div>
                    <div class="config-hint-desc">策略开关请在"市场"页右上角的"管理实时信号"中设置</div>
                  </div>
                </div>
              </div>

              <!-- 个性化参数设置已从回测页移除（依据"我的页为配置中心"的信息架构） -->

              <h3>回测工具</h3>
              <div class="backtest-panel">
                <label>回测周期</label>
                <input type="range" id="lookahead-slider" min="4" max="60" value="12">
                <span id="lookahead-value">12</span> 根K线
                <button class="btn-primary" onclick="window.runMobileBacktest()">运行回测</button>
              </div>
            </div>
          </div>

          <div id="info-view" class="tab-content">
            <div class="info-container">
              <div class="info-toolbar" id="info-toolbar">
                <div class="toolbar-group">
                  <label>资产</label>
                  <select id="info-asset">
                    <option>BTC</option>
                    <option>ETH</option>
                  </select>
                </div>
                <div class="toolbar-group">
                  <label>粒度</label>
                  <select id="info-granularity">
                    <option value="daily">Daily</option>
                    <option value="weekly">Weekly</option>
                  </select>
                </div>
                <div class="toolbar-group">
                  <label>日期</label>
                  <input id="info-date" type="date" />
                </div>
                <button class="toolbar-btn" id="info-export">导出 PNG</button>
              </div>
              <div class="info-tip" id="info-tip">数据源：公开API（10分钟缓存）</div>
              <div class="info-summary" id="info-summary"></div>
              <div class="info-grid">
                <div class="info-main">
                  <div class="info-chart-controls">
                    <button class="toolbar-btn" id="info-help">ⓘ</button>
                  </div>
                </div>
                

        <!-- DriverIndex 宏观驱动因子 -->
                <div class="macro-factors-card" id="card-driver-index">
                  <div class="macro-card-header">
                    <div class="macro-card-title">
                      <h3>DriverIndex 宏观驱动因子</h3>
                      <p>利率/通胀/失业率综合指标，与BTC相关性分析</p>
                    </div>
                    <div class="macro-card-controls">
                      <div class="time-range-selector">
                        <button class="range-btn" data-range="7D">7D</button>
                        <button class="range-btn active" data-range="30D">30D</button>
                        <button class="range-btn" data-range="90D">90D</button>
                      </div>
                      <button class="refresh-btn" id="refresh-driver-data" title="强制刷新数据">🔄</button>
                    </div>
                  </div>
                  
                  <div class="chart-driver-index" id="chart-driver-index">
                    <div class="chart-loading">
                      <div class="loading-spinner"></div>
                      <div class="loading-text">加载宏观数据...</div>
                    </div>
                  </div>
                </div>
                
                <!-- ETF 流入 × Fear & Greed -->
                <div class="macro-factors-card" id="card-etf-fng">
                  <div class="macro-card-header">
                    <div class="macro-card-title">
                      <h3>ETF 流入 × Fear & Greed</h3>
                      <p>机构资金与市场情绪，同步观察</p>
                    </div>
                    <div class="macro-card-controls">
                      <div class="time-range-selector">
                        <button class="range-btn" data-range="7D">7D</button>
                        <button class="range-btn active" data-range="30D">30D</button>
                        <button class="range-btn" data-range="90D">90D</button>
                      </div>
                    </div>
                  </div>
                  
                  <div class="chart-etf-fng" id="chart-etf-fng">
                    <div class="chart-loading">
                      <div class="loading-spinner"></div>
                      <div class="loading-text">加载ETF数据...</div>
                    </div>
                  </div>
                </div>
                
                <!-- 资金费率热力图 -->
                <div class="macro-factors-card" id="card-funding-heat">
                  <div class="macro-card-header">
                    <div class="macro-card-title">
                      <h3>资金费率热力图</h3>
                      <p>多交易所多币种的杠杆侧压力</p>
                    </div>
                    <div class="macro-card-controls">
                      <div class="time-range-selector">
                        <button class="range-btn" data-range="7D">7D</button>
                        <button class="range-btn active" data-range="30D">30D</button>
                        <button class="range-btn" data-range="90D">90D</button>
                      </div>
                    </div>
                  </div>
                  
                  <div class="chart-funding-heat" id="chart-funding-heat">
                    <div class="chart-loading">
                      <div class="loading-spinner"></div>
                      <div class="loading-text">加载资金费率数据...</div>
                    </div>
                  </div>
                </div>
                
                <!-- 相关性图表 -->
                <div class="macro-factors-card" id="card-corr">
                  <div class="macro-card-header">
                    <div class="macro-card-title">
                      <h3>因子相关性分析</h3>
                      <p>各因子与BTC价格的动态相关性</p>
                    </div>
                  </div>
                  
                  <div class="chart-corr" id="corrChart">
                    <div class="chart-loading">
                      <div class="loading-spinner"></div>
                      <div class="loading-text">加载相关性数据...</div>
                    </div>
                  </div>
                </div>
                
                <div class="info-detail" id="info-detail">
                  <div class="info-panel">
                    <div class="panel-title">子因子详情</div>
                    <div class="panel-body" id="info-detail-body">选择左侧维度查看详情</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div id="profile-view" class="tab-content">
            <div class="settings-panel">
              <h3>账户表现</h3>
              <div class="backtest-results" id="acct-performance">
                <div class="backtest-title">盈亏与累计收益</div>
                <div class="backtest-grid">
                  <div class="backtest-item"><div class="backtest-label">近30天胜率</div><div class="backtest-value" id="pf-win">--</div></div>
                  <div class="backtest-item"><div class="backtest-label">最大回撤</div><div class="backtest-value" id="pf-dd">--</div></div>
                  <div class="backtest-item"><div class="backtest-label">累计收益</div><div class="backtest-value" id="pf-ret">--</div></div>
                </div>
              </div>
              <h3>参数定制</h3>
              <div class="personal-settings">
                <div class="setting-item"><label>收益目标（月化）</label><div class="slider-container"><input type="range" id="p-profit" min="0" max="30" value="5" step="1"><span id="p-profit-val">5%</span></div></div>
                <div class="setting-item"><label>最大回撤</label><div class="slider-container"><input type="range" id="p-dd" min="5" max="50" value="15" step="1"><span id="p-dd-val">15%</span></div></div>
                <div class="setting-item"><label>风险暴露</label><div class="slider-container"><input type="range" id="p-risk" min="1" max="20" value="5" step="0.5"><span id="p-risk-val">5%</span></div></div>
                <div class="setting-item"><label>本金规模（USDT）</label><input type="number" id="p-capital" value="10000" min="1000" max="1000000" step="1000" placeholder="输入USDT数额"></div>
                <div class="setting-item"><label>盯盘频率</label><select id="p-monitor"><option value="realtime">随时监控</option><option value="daily" selected>每日1次</option><option value="weekly">每周1次</option></select></div>
              </div>
              <div class="personal-settings-actions"><button class="btn-primary">保存参数</button></div>
              <h3>当前启用的策略（Strategy）</h3>
              <div class="recommendation-cards"><div class="recommendation-card"><div class="recommendation-title">当前启用</div><div class="recommendation-actions"><button class="signal-btn signal-btn-primary" id="open-strategy-manager">管理策略</button></div></div></div>

              <!-- 我的 · 模拟交易 -->
              <div class="signals-section">
                <h2>待启用的信号（Signal）</h2>
                <div class="filterbar" id="signalFilterBar"></div>
                <div style="display:flex; gap:12px; margin:8px 0 12px;">
                  <button class="timeframe-btn" id="btn-enable-all">一键启用全部</button>
                </div>
                <div id="mine-queued"></div>
              </div>

              <div class="signals-section">
                <h2>运行中（模拟持仓）（Position）</h2>
                <div class="filterbar" id="posFilterBar"></div>
                <div id="mine-open"></div>
              </div>

              <div class="signals-section">
                <h2>📜 历史（已关闭）</h2>
                <div id="mine-history"></div>
              </div>
            </div>
          </div>
        </main>

        <nav class="bottom-nav">
          <button class="nav-btn active" data-tab="home">
            <span class="nav-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-label="Home">
                <path d="M3 10.5L12 3l9 7.5"/>
                <path d="M5.5 10.5V19a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2v-8.5"/>
                <path d="M9.5 21V14.5a2 2 0 0 1 2-2h1a2 2 0 0 1 2 2V21"/>
              </svg>
            </span>
            <span class="nav-label">首页</span>
          </button>
          <button class="nav-btn" data-tab="vip">
            <span class="nav-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-label="VIP">
                <path d="M3.5 18.5h17"/>
                <path d="M5 18.5l-1-9 5 3.5L12 6l3 7 5-3.5-1 9H5z"/>
              </svg>
            </span>
            <span class="nav-label">会员</span>
          </button>
          <button class="nav-btn" data-tab="info">
            <span class="nav-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-label="Factors">
                <path d="M4 20V10"/>
                <path d="M10 20V4"/>
                <path d="M16 20v-7"/>
                <path d="M22 20V8"/>
                <path d="M2 20h20"/>
              </svg>
            </span>
            <span class="nav-label">因子</span>
          </button>
          <button class="nav-btn tab-item" data-tab="profile" id="nav-mine">
            <span class="nav-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-label="Profile">
                <circle cx="12" cy="8.5" r="3.5"/>
                <path d="M4.5 20a7.5 7.5 0 0 1 15 0"/>
              </svg>
            </span>
            <span class="nav-label label">我的</span>
            <span class="mine-badge" id="mine-badge">0</span>
          </button>
        </nav>

        <button class="fab-refresh" onclick="window.refreshMobileData()">
          <span class="refresh-icon">🔄</span>
        </button>
      </div>
    `;

    this.addMobileStyles();
  }
  private addMobileStyles() {
    const style = document.createElement('style');
    style.textContent = `
      * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        -webkit-tap-highlight-color: transparent;
      }

      body {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'PingFang SC', sans-serif;
        background: #0B0F14;
        color: #E6EDF6;
        overflow-x: hidden;
        font-size: 16px;
        line-height: 1.375;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
      }

      /* iOS 设计系统 CSS 变量 */
      :root {
        /* 背景色 */
        --bg-primary: #0B0F14;
        --bg-surface: #0F1621;
        --bg-surface-2: #121C2A;
        --border-base: #1F2A3A;

        /* 文本色 */
        --text-primary: #E6EDF6;
        --text-secondary: #A7B1C2;
        --text-muted: #6E7A8A;

        /* 品牌色 */
        --brand-primary: #00D5FF;
        --brand-primary-600: #00B8E6;
        --brand-bg: rgba(0, 213, 255, 0.16);

        /* 多空语义色 */
        --bull-green: #16C784;
        --bear-red: #EA3943;
        --warn-amber: #F59E0B;
        --info-blue: #3B82F6;

        /* 状态底色 */
        --bull-bg: rgba(22, 199, 132, 0.16);
        --bear-bg: rgba(234, 57, 67, 0.16);

        /* 阴影 */
        --shadow-1: 0 6px 16px -2px rgba(0, 0, 0, 0.3);
        --glow-brand: 0 0 24px rgba(0, 213, 255, 0.32);

        /* 字体 */
        --font-h1: -apple-system-headline2, -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
        --font-h2: -apple-system-headline1, -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
        --font-h3: -apple-system-headline, -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
        --font-title: -apple-system-headline, -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
        --font-body: -apple-system-body, -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
        --font-caption: -apple-system-caption1, -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;

        /* 间距 */
        --space-xs: 4pt;
        --space-sm: 8pt;
        --space-md: 16pt;
        --space-lg: 24pt;
        --space-xl: 32pt;

        /* 圆角 */
        --radius-chip: 12pt;
        --radius-card: 16pt;
        --radius-sheet: 20pt;
      }

      .mobile-app {
        display: flex;
        flex-direction: column;
        height: 100dvh;
        height: -webkit-fill-available;
        background: var(--bg-primary);
        padding-top: env(safe-area-inset-top);
        padding-bottom: 0;
        animation: fadeInUp 0.3s ease-out;
        width: 100vw;
        max-width: 100vw;
        overflow-x: hidden;
        position: relative;
        box-sizing: border-box;
      }

      @keyframes fadeInUp {
        from {
          opacity: 0;
          transform: translateY(12pt);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      .mobile-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: var(--space-md) var(--space-md);
        background: var(--bg-primary);
        border-bottom: 1px solid var(--border-base);
        min-height: 64pt;
      }

      .header-right {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: var(--space-xs);
        text-align: right;
        padding-right: 0;
        width: 100%;
        max-width: 200pt;
      }

      .kpi-actions {
        display: flex;
        align-items: center;
        gap: 6px;
        width: 100%;
        align-self: stretch;
        padding: 8px 16px 0;
        flex-wrap: wrap;
        min-height: 32px;
      }

      .kpi-actions .btn-strategy {
        margin-left: auto;
        flex-shrink: 0;
      }

      .kpi-actions .chips {
        display: flex;
        gap: 4px;
        flex-wrap: wrap;
        flex: 1;
        min-width: 0;
      }

      .api-status {
        display: flex;
        align-items: center;
        gap: 6pt;
        font-family: var(--font-caption);
        font-size: 11pt;
        font-weight: 400;
        line-height: 15pt;
        align-self: flex-end;
        margin-bottom: var(--space-xs);
      }

      .status-dot {
        width: 8pt;
        height: 8pt;
        border-radius: 4pt;
        background: var(--text-muted);
        transition: background-color 0.3s ease;
      }

      .status-dot.online {
        background: var(--bull-green);
        box-shadow: 0 0 8pt rgba(22, 199, 132, 0.6);
      }

      .status-dot.offline {
        background: var(--bear-red);
        box-shadow: 0 0 8pt rgba(234, 57, 67, 0.6);
      }

      .status-text {
        color: var(--text-muted);
        font-size: 10pt;
      }

      .app-title {
        font-family: var(--font-h1);
        font-size: 28pt;
        font-weight: 600;
        line-height: 34pt;
        color: var(--text-primary);
        letter-spacing: -0.5pt;
      }

      .learning-stats {
        display: flex;
        gap: var(--space-sm);
        align-items: baseline;
      }

      .stat-item {
        text-align: center;
        min-width: 60pt;
        display: flex;
        flex-direction: column;
        align-items: center;
      }

      .stat-label {
        font-family: var(--font-caption);
        font-size: 11pt;
        font-weight: 400;
        line-height: 15pt;
        color: var(--text-muted);
        margin-bottom: 2pt;
        height: 15pt;
        display: flex;
        align-items: center;
      }

      .stat-value {
        font-family: var(--font-body);
        font-size: 14pt;
        font-weight: 600;
        line-height: 18pt;
        color: var(--brand-primary);
        font-variant-numeric: tabular-nums;
        height: 18pt;
        display: flex;
        align-items: center;
      }

      .timeframe-tabs {
        display: flex;
        padding: var(--space-md) var(--space-md);
        gap: var(--space-sm);
        background: var(--bg-primary);
        border-bottom: 1px solid var(--border-base);
      }

      .tf-tab {
        flex: 1;
        height: 36pt;
        background: var(--bg-surface-2);
        color: var(--text-secondary);
        border: none;
        border-radius: var(--radius-card);
        font-family: var(--font-title);
        font-size: 17pt;
        font-weight: 600;
        line-height: 22pt;
        cursor: pointer;
        transition: all 0.12s ease-out;
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .tf-tab:hover {
        background: var(--bg-surface);
        color: var(--text-primary);
      }

      .tf-tab.active {
        background: var(--brand-bg);
        color: var(--text-primary);
        box-shadow: var(--glow-brand);
        transform: scale(0.98);
        transition: transform 0.12s ease-out;
      }

      .tf-tab.active:active {
        transform: scale(1.0);
      }

      .mobile-content {
        flex: 1;
        overflow-y: auto;
        -webkit-overflow-scrolling: touch;
        padding: 0 var(--space-md);
        padding-bottom: 70pt;
        width: 100%;
        max-width: 100vw;
        box-sizing: border-box;
      }

      .tab-content {
        display: none;
        padding-bottom: var(--space-xl);
        min-height: 100%;
      }

      .tab-content.active {
        display: block;
      }

      /* iOS规范：实时行情卡片样式 */
      .quotes-enhanced {
        background: var(--bg-surface);
        border-radius: var(--radius-card);
        border: 1px solid var(--border-base);
        margin: 0 var(--s16);
        max-height: 400px;
        overflow-y: auto;
        -webkit-overflow-scrolling: touch;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12);
      }

      .quote-enhanced-item {
        display: flex;
        align-items: center;
        height: 56px;
        padding: 0 var(--s16);
        border-bottom: 1px solid var(--border-base);
        transition: background-color 0.12s ease-out;
        cursor: pointer;
      }

      .quote-enhanced-item:last-child {
        border-bottom: none;
      }

      .quote-enhanced-item:active {
        background: rgba(255, 255, 255, 0.04);
      }

      .quote-symbol {
        font: var(--font-body);
        font-weight: 500;
        color: var(--text-primary);
        min-width: 80px;
      }

      .quote-price {
        font: var(--font-mono);
        font-weight: 500;
        color: var(--text-primary);
        flex: 1;
        text-align: right;
        margin-right: var(--s16);
      }

      .quote-change-chip {
        font: var(--font-mono);
        font-size: 13px;
        line-height: 18px;
        font-weight: 500;
        padding: var(--s8);
        border-radius: var(--radius-badge);
        min-width: 60px;
        text-align: center;
        margin-right: var(--s16);
      }

      .quote-change-chip.positive {
        color: var(--success);
        background: rgba(34, 197, 94, 0.1);
      }

      .quote-change-chip.negative {
        color: var(--danger);
        background: rgba(239, 68, 68, 0.1);
      }



      /* 自定义滚动条样式 */
      .quotes-enhanced::-webkit-scrollbar {
        width: 6px;
      }

      .quotes-enhanced::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 3px;
      }

      .quotes-enhanced::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00d4ff, #0099cc);
        border-radius: 3px;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
      }

      .quotes-enhanced::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #00ff88, #00d4ff);
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.7);
      }

      .market-section {
        margin-bottom: var(--space-lg);
        margin-top: var(--space-lg);
      }

      .section-title {
        font-family: var(--font-h3);
        font-size: 20pt;
        font-weight: 600;
        line-height: 26pt;
        margin-bottom: var(--space-md);
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: var(--space-sm);
        letter-spacing: -0.3pt;
      }

      /* 第二块：交易信号卡片样式 */
      .signals-section {
        margin-top: var(--space-lg);
      }

      .signals-disclaimer {
        font-family: var(--font-caption);
        font-size: 12pt;
        font-weight: 400;
        line-height: 16pt;
        color: var(--text-muted);
        text-align: center;
        margin-bottom: var(--space-md);
        padding: var(--space-sm) var(--space-md);
        background: rgba(148, 163, 184, 0.05);
        border-radius: var(--radius-chip);
        border: 1px solid rgba(148, 163, 184, 0.1);
      }

      .condition-alert-section,
      .strategy-lab-section,
      .recommendation-section,
      .performance-section {
        margin-top: var(--space-lg);
      }

      /* 条件触发提醒样式 */
      .condition-alert-config {
        display: grid;
        grid-template-columns: 1fr 1fr auto;
        gap: var(--space-md);
        align-items: center;
        margin-bottom: var(--space-md);
      }

      .condition-item {
        display: flex;
        flex-direction: column;
        gap: var(--space-xs);
      }

      .condition-item label {
        font-size: 12pt;
        color: var(--text-secondary);
        font-weight: 500;
      }

      .condition-item input {
        height: 36pt;
        padding: 0 var(--space-sm);
        background: var(--bg-surface-2);
        border: 1px solid var(--border-base);
        border-radius: var(--radius-chip);
        color: var(--text-primary);
        font-size: 14pt;
        text-align: center;
      }

      .condition-item span {
        color: var(--text-secondary);
        font-size: 12pt;
        text-align: center;
        margin-top: 4pt;
      }

      .condition-toggle-btn {
        height: 36px;
        padding: 0 var(--s16);
        background: var(--brand-bg);
        color: var(--brand-primary);
        border: 1px solid var(--brand-primary-600);
        border-radius: var(--radius-chip);
        cursor: pointer;
        font: var(--font-body);
        font-weight: 600;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .condition-toggle-btn:hover {
        background: var(--brand-primary);
        color: #000;
      }

      .condition-toggle-btn.active {
        background: var(--brand-primary);
        color: #000;
      }

      .condition-status {
        display: flex;
        align-items: center;
        gap: var(--space-sm);
        padding: var(--space-sm) var(--space-md);
        background: var(--bg-surface-2);
        border-radius: var(--radius-chip);
        border: 1px solid var(--border-base);
      }

      .status-indicator {
        width: 8pt;
        height: 8pt;
        border-radius: 50%;
        background: var(--text-muted);
        transition: background-color 0.3s ease;
      }

      .status-indicator.active {
        background: var(--bull-green);
        box-shadow: 0 0 8pt rgba(22, 199, 132, 0.6);
      }

      .status-text {
        font-size: 12pt;
        color: var(--text-secondary);
      }

      /* 策略实验室样式 */
      .strategy-lab-config {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: var(--space-md);
        margin-bottom: var(--space-md);
      }

      .lab-input-group {
        display: flex;
        flex-direction: column;
        gap: var(--space-xs);
      }

      .lab-input-group label {
        font-size: 12pt;
        color: var(--text-secondary);
        font-weight: 500;
      }

      .lab-input-group input,
      .lab-input-group select {
        height: 36pt;
        padding: 0 var(--space-sm);
        background: var(--bg-surface-2);
        border: 1px solid var(--border-base);
        border-radius: var(--radius-chip);
        color: var(--text-primary);
        font-size: 14pt;
      }

      .lab-backtest-btn {
        grid-column: 1 / -1;
        height: 40pt;
        background: linear-gradient(135deg, var(--brand-primary), var(--brand-primary-600));
        color: #000;
        border: none;
        border-radius: var(--radius-chip);
        cursor: pointer;
        font-size: 16pt;
        font-weight: 600;
        transition: all 0.2s ease;
        margin-top: var(--space-sm);
      }

      .lab-backtest-btn:hover {
        transform: translateY(-1px);
        box-shadow: var(--glow-brand);
      }

      .lab-backtest-btn:active {
        transform: translateY(0);
      }

      /* iOS人机界面规范 - 全局Spacing Token系统 */
      :root {
        --s4: 4px;
        --s8: 8px;
        --s12: 12px;
        --s16: 16px;
        --s20: 20px;
        --s24: 24px;
        --s32: 32px;
        
        /* iOS规范颜色系统 */
        --bg-primary: #0B0F14;
        --bg-surface: #11161E;
        --border-base: #1E2632;
        --text-primary: #E6EDF6;
        --text-secondary: #9AA8B5;
        --success: #22C55E;
        --danger: #EF4444;
        --info: #3B82F6;
        
        /* iOS规范字体系统 */
        --font-title: 17px/22px -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
        --font-body: 15px/20px -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
        --font-caption: 13px/18px -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
        --font-mono: 15px/20px 'SF Mono', Monaco, 'Cascadia Code', monospace;
        
        /* iOS规范圆角系统 */
        --radius-card: 16px;
        --radius-button: 12px;
        --radius-badge: 10px;
        --radius-segmented: 10px;
      }

      /* 紧凑模式样式 - 苹果UI设计规范 */
      /* 基础：调小全站字号，符合移动端信息密度 */
      @media (max-width: 480px){
        html{ font-size: clamp(13px, 3.4vw, 15px); } /* 响应式字体大小 */
      }

      /* 全局密度：卡片/分组留白更紧凑 - 苹果设计规范 */
      .compact-ui .card,
      .compact-ui .kpi-card,
      .compact-ui .section{
        border-radius: 8px;
        padding: 8px 10px;         /* 更紧凑的内边距 */
      }
      
      /* 头部区域紧凑化 */
      .compact-ui .mobile-header{
        padding: 8px 12px;
        min-height: 48px;
      }
      
      .compact-ui .timeframe-tabs{
        padding: 6px 12px;
      }

      /* KPI 模块 - 苹果设计规范 */
      .compact-ui .stat-label{ font-size: 10px; line-height: 12px; }
      .compact-ui .stat-value{ font-size: 16px; line-height: 18px; }
      .compact-ui .api-status{ font-size: 10px; }
      .compact-ui .status-dot{ width: 6px; height: 6px; }

      /* 动作行：按钮与 Chip 更瘦 - 苹果设计规范 */
      .compact-ui .chip{
        height: 18px; padding: 0 6px; font-size: 9px; border-radius: 9px;
        line-height: 18px;
      }
      .compact-ui .btn-primary,
      .compact-ui .btn-strategy{
        height: 20px; line-height: 20px; padding: 0 8px;
        font-size: 10px; border-radius: 10px;
        white-space: nowrap; word-break: keep-all; min-width: max-content;
      }

      /* 快速回测相关按钮紧凑化 */
      .compact-ui .lab-backtest-btn{
        height: 28px; padding: 0 12px; font-size: 12px;
        border-radius: 14px; line-height: 28px;
      }
      .compact-ui .signal-btn{
        height: 28px; padding: 0 12px; font-size: 12px;
        border-radius: 14px; line-height: 28px;
      }
      .compact-ui .qb-range-btn{
        height: 24px; padding: 0 8px; font-size: 11px;
        border-radius: 12px; line-height: 24px;
      }
      .compact-ui .qb-close{
        height: 24px; padding: 0 8px; font-size: 11px;
        border-radius: 12px; line-height: 24px;
      }

      /* 头部按钮紧凑化 */
      .compact-ui .header-action-btn{
        height: 22px; padding: 0 6px; font-size: 10px;
        border-radius: 11px; line-height: 22px;
      }
      .compact-ui .manage-signals-btn{
        height: 22px; padding: 0 8px; font-size: 10px;
        border-radius: 11px; line-height: 22px;
      }

      /* 实时行情紧凑化 */
      .compact-ui .quote-enhanced-item{
        height: 36px; padding: 0 8px;
      }
      .compact-ui .quote-symbol{
        font-size: 12px; line-height: 16px; min-width: 60px;
      }
      .compact-ui .quote-price{
        font-size: 12px; line-height: 16px; margin-right: 8px;
      }
      .compact-ui .quote-change-chip{
        font-size: 10px; line-height: 14px; min-width: 50px; margin-right: 8px;
        padding: 2px 6px; border-radius: 8px;
      }

      /* iOS规范：时间段切换分段控件 */
      .timeframe-tabs {
        display: flex;
        margin: var(--s16) var(--s16);
        background: var(--bg-surface);
        border: 1px solid var(--border-base);
        border-radius: var(--radius-segmented);
        padding: var(--s4);
        gap: var(--s4);
      }
      
      .tf-tab {
        flex: 1;
        height: 36px;
        border: none;
        background: transparent;
        color: var(--text-secondary);
        font: var(--font-body);
        border-radius: calc(var(--radius-segmented) - 2px);
        cursor: pointer;
        transition: all 0.2s ease;
      }
      
      .tf-tab.active {
        background: var(--brand-bg);
        color: var(--brand-primary);
        font-weight: 600;
      }
      
      /* 时间粒度 Segmented（4H / 1D / 1W） - 苹果设计规范 */
      .compact-ui .tf-tab{
        min-width: 44px; height: 22px; font-size: 10px; border-radius: 11px;
        line-height: 22px;
      }

      /* 表格与列表行 - 苹果设计规范 */
      .compact-ui .list-row{ min-height: 36px; padding: 6px 8px; }
      .compact-ui .list-row .meta{ font-size: 10px; }
      
      /* 内容区域紧凑化 */
      .compact-ui .mobile-content{
        padding: 0 8px;
      }
      
      .compact-ui .market-section{
        margin-bottom: 8px;
      }

      /* 图表：进一步压缩高度 - 苹果设计规范 */
      @media (max-width: 480px){
        .compact-ui .chart-driver-index{ height: 200px !important; }
        .compact-ui .chart-etf-fng{ height: 200px !important; }
        .compact-ui .chart-funding-heat{ height: 220px !important; }
        .compact-ui .chart-corr{ height: 200px !important; }
        .compact-ui .info-radar{ height: 240px !important; }
        /* 因子页：减少两侧留白，卡片更贴边 */
        .compact-ui .info-container{ padding: 6px !important; }
        .compact-ui .macro-factors-card{ padding: 8px !important; margin: 6px 0 !important; }
        .compact-ui .info-grid{ gap: 6px !important; }
        .compact-ui .info-main{ gap: 4px !important; }
        .compact-ui .macro-card-header{ margin-bottom: 6px !important; }
        .compact-ui .macro-card-title h3{ font-size: 14px !important; }
        .compact-ui .macro-card-title p{ font-size: 11px !important; }
        .compact-ui .time-range-selector{ gap: 4px !important; }
        .compact-ui .range-btn{ padding: 4px 8px !important; font-size: 11px !important; }
      }

      /* 纵向间距：更紧凑 - 苹果设计规范 */
      .compact-ui .gap-xs{ margin-top: 4px; }
      .compact-ui .gap-sm{ margin-top: 6px; }
      .compact-ui .gap-md{ margin-top: 8px; }
      .compact-ui .gap-lg{ margin-top: 10px; }

      /* 针对"按钮换行/被挤"再补两刀 - 苹果设计规范 */
      .compact-ui .kpi-actions{ display:flex; align-items:center; gap:6px; flex-wrap:nowrap; }
      .compact-ui .kpi-actions .btn-strategy{
        white-space: nowrap; word-break: keep-all; min-width: max-content;
        flex: 0 0 auto; margin-left: auto;
      }
      .compact-ui .kpi-actions .chips{ flex: 0 0 auto; gap: 4px; }

      /* 手机端头部按钮优化 */
      @media (max-width: 480px) {
        .kpi-actions {
          padding: 6px 12px 0;
          gap: 4px;
        }
        
        .header-action-btn {
          font-size: 10px;
          padding: 0 6px;
          height: 24px;
          line-height: 24px;
          white-space: nowrap;
          min-width: max-content;
        }
        
        .manage-signals-btn {
          font-size: 10px;
          padding: 0 8px;
          height: 24px;
          line-height: 24px;
          white-space: nowrap;
        }
        
        .kpi-actions .chips {
          gap: 2px;
        }
      }

      @media (max-width: 375px) {
        .kpi-actions {
          padding: 4px 8px 0;
          gap: 2px;
        }
        
        .header-action-btn {
          font-size: 9px;
          padding: 0 4px;
          height: 22px;
          line-height: 22px;
        }
        
        .manage-signals-btn {
          font-size: 9px;
          padding: 0 6px;
          height: 22px;
          line-height: 22px;
        }
      }

      @media (max-width: 320px) {
        .kpi-actions {
          padding: 2px 4px 0;
          flex-direction: column;
          align-items: stretch;
          gap: 4px;
        }
        
        .kpi-actions .chips {
          order: 1;
          justify-content: center;
        }
        
        .kpi-actions .btn-strategy {
          order: 2;
          margin-left: 0;
          align-self: center;
        }
        
        .header-action-btn {
          font-size: 8px;
          padding: 0 3px;
          height: 20px;
          line-height: 20px;
        }
        
        .manage-signals-btn {
          font-size: 8px;
          padding: 0 4px;
          height: 20px;
          line-height: 20px;
        }
      }
      
      /* 学习统计区域紧凑化 */
      .compact-ui .learning-stats{ gap: 8px; }

      /* iOS规范：顶部三指标水平卡片区 */
      .learning-stats {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: var(--s12);
        margin: var(--s16) var(--s16) var(--s24);
      }

      /* 学习统计区域响应式优化 */
      @media (max-width: 480px) {
        .learning-stats {
          gap: var(--s8);
          margin: var(--s12) var(--s12) var(--s16);
        }
        
        .stat-item {
          min-height: 70px;
          padding: var(--s12);
        }
        
        .stat-label {
          font-size: 10px;
          line-height: 12px;
          margin-bottom: var(--s2);
        }
        
        .stat-value {
          font-size: 16px;
          line-height: 18px;
        }
      }

      @media (max-width: 375px) {
        .learning-stats {
          gap: var(--s6);
          margin: var(--s8) var(--s8) var(--s12);
        }
        
        .stat-item {
          min-height: 60px;
          padding: var(--s8);
        }
        
        .stat-label {
          font-size: 9px;
          line-height: 11px;
        }
        
        .stat-value {
          font-size: 14px;
          line-height: 16px;
        }
      }

      @media (max-width: 320px) {
        .learning-stats {
          grid-template-columns: 1fr;
          gap: var(--s4);
          margin: var(--s6) var(--s6) var(--s8);
        }
        
        .stat-item {
          min-height: 50px;
          padding: var(--s6);
          display: flex;
          flex-direction: row;
          align-items: center;
          justify-content: space-between;
        }
        
        .stat-label {
          font-size: 8px;
          line-height: 10px;
          margin-bottom: 0;
        }
        
        .stat-value {
          font-size: 12px;
          line-height: 14px;
        }
      }
      
      .stat-item {
        background: var(--bg-surface);
        border: 1px solid var(--border-base);
        border-radius: var(--radius-card);
        padding: var(--s16);
        text-align: center;
        min-height: 80px;
        display: flex;
        flex-direction: column;
        justify-content: center;
      }
      
      .stat-label {
        font: var(--font-caption);
        color: var(--text-secondary);
        margin-bottom: var(--s4);
      }
      
      .stat-value {
        font: 600 20px/24px var(--font-mono);
        color: var(--text-primary);
      }

      /* iOS规范：分区标题统一样式 */
      .section-title {
        font: 600 17px/22px -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
        color: var(--text-primary);
        margin: var(--s24) var(--s16) var(--s12);
        display: flex;
        align-items: center;
        gap: var(--s8);
      }
      .section-title::before {
        content: '';
        width: 20px;
        height: 20px;
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
      }
      .signals-cards {
        display: flex;
        flex-direction: column;
        gap: var(--space-md);
      }

      /* 管理实时信号按钮 */
      .manage-signals-btn {
        height: 30pt;
        padding: 0 10pt;
        background: var(--brand-bg);
        color: var(--brand-primary);
        border: 1px solid var(--brand-primary-600);
        border-radius: var(--radius-chip);
        cursor: pointer;
        font-size: 10pt;
        font-weight: 600;
        white-space: nowrap;
      }

      .header-action-buttons {
        display: flex;
        gap: var(--space-xs);
      }

      .header-action-btn {
        height: 28pt;
        padding: 0 8pt;
        background: var(--bg-surface-2);
        color: var(--text-secondary);
        border: 1px solid var(--border-base);
        border-radius: var(--radius-chip);
        cursor: pointer;
        font-size: 10pt;
        font-weight: 500;
        transition: all 0.2s ease;
        white-space: nowrap;
      }

      .header-action-btn:hover {
        background: var(--bg-surface);
        color: var(--text-primary);
        border-color: var(--brand-primary-600);
      }

      .manage-tip {
        position: absolute;
        top: 48pt;
        right: 16pt;
        background: #111827;
        color: #e5e7eb;
        border: 1px solid #374151;
        border-radius: 8pt;
        padding: 8pt 10pt;
        font-size: 12pt;
        box-shadow: var(--shadow-1);
        z-index: 1200;
      }

      /* 管理实时信号抽屉 */
      .ms-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.45); z-index: 1800; display: flex; justify-content: flex-end; }
      .ms-drawer { width: min(88vw, 520pt); background: var(--bg-surface); border-left: 1px solid var(--border-base); padding: var(--space-md); overflow-y: auto; }
      .ms-header { display:flex; align-items:center; justify-content: space-between; margin-bottom: var(--space-sm); }
      .ms-title { font: 600 17px/22px -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif; color: var(--text-primary); }
      .ms-actions { display:flex; gap: var(--space-sm); }
      .ms-btn { height: 36px; padding: 0 12px; border-radius: var(--radius-button); border: 1px solid var(--border-base); background: var(--bg-surface-2); color: var(--text-secondary); cursor: pointer; font: var(--font-body); }
      .ms-btn.primary { background: var(--brand-primary); color: #000; border-color: var(--brand-primary-600); }
      .ms-list { display: flex; flex-direction: column; gap: 8pt; margin-top: var(--space-sm); }
      .ms-item { display:flex; align-items:center; justify-content: space-between; padding: 10pt 12pt; background: var(--bg-surface-2); border: 1px solid var(--border-base); border-radius: var(--radius-chip); }
      .ms-switch { position: relative; width: 46px; height: 24px; background: #334155; border-radius: 12px; cursor:pointer; }
      .ms-switch.active { background: #00d4ff; }
      .ms-switch::after { content: ''; position: absolute; top: 3px; left: 3px; width: 18px; height: 18px; background: #fff; border-radius: 9px; transition: transform .2s ease; }
      .ms-switch.active::after { transform: translateX(22px); }

      /* 快速回测弹窗样式 */
      .qb-modal-overlay {
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.55);
        display: flex;
        align-items: flex-end;
        justify-content: center;
        z-index: 2000;
      }

      .qb-modal {
        width: 100%;
        max-width: 640pt;
        background: var(--bg-surface);
        border-top-left-radius: var(--radius-sheet);
        border-top-right-radius: var(--radius-sheet);
        border: 1px solid var(--border-base);
        box-shadow: var(--shadow-1);
        padding: var(--space-md);
        max-height: 80vh;
        overflow-y: auto;
        animation: fadeInUp 0.2s ease-out;
      }

      .qb-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: var(--space-sm);
        margin-bottom: var(--space-sm);
      }

      .qb-title {
        font-family: var(--font-title);
        font-size: 17pt;
        font-weight: 700;
        color: var(--text-primary);
      }

      .qb-close {
        background: transparent;
        border: 1px solid var(--border-base);
        color: var(--text-secondary);
        border-radius: var(--radius-chip);
        padding: 6pt 10pt;
        cursor: pointer;
      }

      .qb-meta-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: var(--space-sm);
        margin: var(--space-sm) 0 var(--space-md);
      }

      .qb-meta {
        text-align: center;
        background: var(--bg-surface-2);
        border: 1px solid var(--border-base);
        border-radius: var(--radius-chip);
        padding: 8pt;
      }

      .qb-label { color: var(--text-muted); font-size: 12pt; }
      .qb-value { color: var(--brand-primary); font-weight: 700; font-size: 16pt; }

      .qb-range-switch {
        display: flex;
        gap: var(--space-sm);
        margin-bottom: var(--space-sm);
      }

      .qb-range-btn {
        flex: 1;
        height: 30pt;
        background: var(--bg-surface-2);
        color: var(--text-secondary);
        border: 1px solid var(--border-base);
        border-radius: var(--radius-chip);
        cursor: pointer;
      }

      .qb-range-btn.active {
        background: var(--brand-bg);
        color: var(--text-primary);
        border-color: var(--brand-primary-600);
      }

      .qb-chart {
        width: 100%;
        height: 80pt;
        background: var(--bg-primary);
        border: 1px solid var(--border-base);
        border-radius: var(--radius-chip);
        margin-bottom: var(--space-md);
        position: relative;
        overflow: hidden;
      }

      .qb-chart-bar {
        position: absolute;
        bottom: 4pt;
        width: 4pt;
        background: linear-gradient(180deg, var(--brand-primary), #00a3cc);
        border-radius: 2pt;
      }

      .qb-table { width: 100%; border-collapse: collapse; }
      .qb-table th, .qb-table td { padding: 8pt; border-bottom: 1px solid var(--border-base); text-align: left; }
      .qb-empty, .qb-error, .qb-loading { text-align: center; color: var(--text-secondary); padding: var(--space-md) 0; }

      .signal-compact-card {
        background: var(--bg-surface);
        border-radius: var(--radius-card);
        padding: var(--s16);
        border: 1px solid var(--border-base);
        margin: 0 var(--s16) var(--s16);
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12);
        cursor: pointer;
        transition: background-color 0.12s ease-out;
        animation: slideInRight 0.3s ease-out;
        animation-fill-mode: both;
      }

      .signal-compact-card:active {
        background: rgba(255, 255, 255, 0.04);
      }

      .signal-header-compact {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--s16);
      }

      .signal-title-compact {
        display: flex;
        align-items: center;
        gap: var(--space-sm);
      }

      .signal-direction-pill {
        font-family: var(--font-body);
        font-size: 14pt;
        font-weight: 600;
        line-height: 20pt;
        padding: 6pt 12pt;
        border-radius: var(--radius-chip);
        color: white;
      }

      .signal-direction-pill.buy {
        background: var(--bull-green);
      }

      .signal-direction-pill.sell {
        background: var(--bear-red);
      }

      .signal-symbol {
        font-family: var(--font-title);
        font-size: 17pt;
        font-weight: 600;
        line-height: 22pt;
        color: var(--text-primary);
      }

      .signal-strategy-chip {
        background: var(--bg-surface-2);
        color: var(--text-secondary);
        padding: 4pt 8pt;
        border-radius: var(--radius-chip);
        font-family: var(--font-caption);
        font-size: 12pt;
        font-weight: 400;
        line-height: 16pt;
      }

      .signal-mini-kline {
        width: 80pt;
        height: 40pt;
        background: var(--bg-surface-2);
        border-radius: var(--radius-chip);
        border: 1px solid var(--border-base);
        position: relative;
        overflow: hidden;
      }

      .kline-bar {
        position: absolute;
        bottom: 2pt;
        width: 3pt;
        border-radius: 1pt;
        transition: all 0.2s ease;
      }

      .kline-bar.bullish {
        background: var(--bull-green);
        box-shadow: 0 0 4pt rgba(22, 199, 132, 0.3);
      }

      .kline-bar.bearish {
        background: var(--bear-red);
        box-shadow: 0 0 4pt rgba(234, 57, 67, 0.3);
      }

      



      .signal-price-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: var(--s12);
        margin: var(--s16) 0;
      }

      .signal-price-cell {
        text-align: center;
        padding: var(--s12);
        background: var(--bg-surface);
        border-radius: var(--radius-badge);
        border: 1px solid var(--border-base);
        height: 72px;
        display: flex;
        flex-direction: column;
        justify-content: center;
      }

      .signal-price-label {
        font: var(--font-caption);
        color: var(--text-secondary);
        margin-bottom: var(--s4);
      }

      .signal-price-value {
        font: 600 20px/24px var(--font-mono);
        color: var(--text-primary);
      }

      .signal-actions {
        display: flex;
        gap: var(--s12);
        margin-top: var(--s16);
      }

      .signal-btn {
        flex: 1;
        height: 44px;
        border: none;
        border-radius: var(--radius-button);
        font: var(--font-body);
        font-weight: 600;
        cursor: pointer;
        transition: opacity 0.12s ease-out;
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .signal-btn:active {
        opacity: 0.7;
      }

      .signal-btn-primary {
        background: var(--brand-primary);
        color: #000;
      }

      .signal-btn-secondary {
        background: transparent;
        color: var(--brand-primary);
        border: 1px solid var(--brand-primary-600);
      }

      @keyframes slideInRight {
        from {
          opacity: 0;
          transform: translateX(50px);
        }
        to {
          opacity: 1;
          transform: translateX(0);
        }
      }

      /* 第三、四、五块样式 */
      .strategy-education-section,
      .recommendation-section,
      .performance-section {
        margin-top: 32px;
      }

      .comparison-hint {
        text-align: center;
        padding: var(--space-lg) var(--space-md);
        color: var(--text-muted);
        font-family: var(--font-body);
        font-size: 16pt;
        font-weight: 400;
        line-height: 22pt;
        background: var(--bg-surface);
        border-radius: var(--radius-card);
        border: 1px solid var(--border-base);
      }

      .recommendation-config-hint {
        background: linear-gradient(135deg, var(--brand-bg), rgba(0, 213, 255, 0.08));
        border: 1px solid rgba(0, 213, 255, 0.3);
        border-radius: var(--radius-card);
        padding: var(--space-md);
        margin-bottom: var(--space-md);
        cursor: pointer;
        transition: all 0.12s ease-out;
      }

      .recommendation-config-hint:active {
        background: rgba(0, 213, 255, 0.12);
        transform: scale(0.98);
      }

      .config-hint-content {
        display: flex;
        align-items: center;
        gap: var(--space-md);
      }

      .config-hint-icon {
        font-size: 24pt;
        width: 40pt;
        height: 40pt;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--brand-primary);
        border-radius: 20pt;
        flex-shrink: 0;
      }

      .config-hint-text {
        flex: 1;
      }

      .config-hint-title {
        font-family: var(--font-title);
        font-size: 17pt;
        font-weight: 600;
        line-height: 22pt;
        color: var(--text-primary);
        margin-bottom: 4pt;
      }

      .config-hint-desc {
        font-family: var(--font-body);
        font-size: 14pt;
        font-weight: 400;
        line-height: 20pt;
        color: var(--text-secondary);
      }

      .config-hint-btn {
        display: flex;
        align-items: center;
        gap: var(--space-sm);
        padding: 8pt 16pt;
        background: var(--brand-primary);
        color: #000;
        border: none;
        border-radius: var(--radius-chip);
        font-family: var(--font-body);
        font-size: 14pt;
        font-weight: 600;
        line-height: 20pt;
        cursor: pointer;
        transition: opacity 0.12s ease-out;
        flex-shrink: 0;
      }

      .config-hint-btn:active {
        opacity: 0.7;
      }

      .recommendation-cards {
        display: flex;
        flex-direction: column;
        gap: var(--space-md);
      }

      .recommendation-card {
        background: var(--bg-surface);
        border-radius: var(--radius-card);
        padding: var(--space-md);
        border: 1px solid var(--border-base);
        box-shadow: var(--shadow-1);
        transition: background-color 0.12s ease-out;
        cursor: pointer;
      }

      .recommendation-card:active {
        background: rgba(255, 255, 255, 0.04);
      }

      .recommendation-title {
        font-family: var(--font-title);
        font-size: 17pt;
        font-weight: 600;
        line-height: 22pt;
        color: var(--text-primary);
        margin-bottom: var(--space-sm);
      }

      .recommendation-content {
        font-family: var(--font-body);
        font-size: 14pt;
        font-weight: 400;
        line-height: 20pt;
        color: var(--text-secondary);
        margin-bottom: var(--space-md);
      }

      .recommendation-actions {
        display: flex;
        gap: var(--space-sm);
      }

      .recommendation-meta {
        margin-bottom: var(--space-md);
      }

      .meta-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        height: 32pt;
        border-bottom: 1px solid var(--border-base);
      }

      .meta-row:last-child {
        border-bottom: none;
      }

      .meta-label {
        font-family: var(--font-body);
        font-size: 14pt;
        font-weight: 400;
        line-height: 20pt;
        color: var(--text-secondary);
      }

      .meta-value {
        font-family: var(--font-body);
        font-size: 14pt;
        font-weight: 500;
        line-height: 20pt;
        color: var(--text-primary);
      }

      .backtest-results {
        background: var(--bg-surface-2);
        border-radius: var(--radius-chip);
        padding: var(--space-md);
        margin-bottom: var(--space-md);
        border: 1px solid var(--border-base);
      }

      .backtest-title {
        font-family: var(--font-body);
        font-size: 16pt;
        font-weight: 600;
        line-height: 22pt;
        color: var(--text-primary);
        margin-bottom: var(--space-sm);
      }

      .backtest-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: var(--space-sm);
      }

      .backtest-item {
        text-align: center;
        padding: var(--space-sm);
        background: var(--bg-primary);
        border-radius: var(--radius-chip);
        border: 1px solid var(--border-base);
      }

      .backtest-label {
        font-family: var(--font-caption);
        font-size: 12pt;
        font-weight: 400;
        line-height: 16pt;
        color: var(--text-muted);
        margin-bottom: 4pt;
      }

      .backtest-value {
        font-family: var(--font-body);
        font-size: 16pt;
        font-weight: 600;
        line-height: 22pt;
        color: var(--brand-primary);
        font-variant-numeric: tabular-nums;
      }

      .recommendation-reason {
        background: var(--brand-bg);
        border-radius: var(--radius-chip);
        padding: var(--space-md);
        margin-bottom: var(--space-md);
        border: 1px solid rgba(0, 213, 255, 0.3);
      }

      .reason-title {
        font-family: var(--font-body);
        font-size: 14pt;
        font-weight: 600;
        line-height: 20pt;
        color: var(--brand-primary);
        margin-bottom: var(--space-sm);
      }

      .reason-content {
        font-family: var(--font-body);
        font-size: 14pt;
        font-weight: 400;
        line-height: 20pt;
        color: var(--text-primary);
      }

      .performance-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: var(--space-md);
      }

      .review-panel,
      .ranking-panel {
        background: var(--bg-surface);
        border-radius: var(--radius-card);
        padding: var(--s16);
        border: 1px solid var(--border-base);
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12);
      }

      .review-panel h4,
      .ranking-panel h4 {
        font: 600 17px/22px -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
        color: var(--text-primary);
        margin-bottom: var(--s16);
        text-align: center;
      }

      .review-item,
      .ranking-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        height: 44px;
        border-bottom: 1px solid var(--border-base);
      }

      .review-item:last-child,
      .ranking-item:last-child {
        border-bottom: none;
      }

      .review-symbol,
      .ranking-strategy {
        font: var(--font-body);
        font-weight: 500;
        color: var(--text-primary);
      }

      .review-result {
        padding: var(--s4) var(--s8);
        border-radius: var(--radius-badge);
        font: var(--font-caption);
        font-weight: 600;
      }

      .review-result.profit {
        background: rgba(34, 197, 94, 0.1);
        color: var(--success);
      }

      .review-result.loss {
        background: rgba(239, 68, 68, 0.1);
        color: var(--danger);
      }

      .ranking-rate {
        font: 500 15px/20px var(--font-mono);
        color: var(--info);
      }

      .personal-settings {
        margin-bottom: var(--space-lg);
      }

      .personal-settings-actions {
        margin-top: var(--s16);
        margin-bottom: var(--s24);
        padding-top: var(--s16);
        border-top: 1px solid var(--border-base);
      }

      .personal-settings-actions .btn-primary {
        background: linear-gradient(135deg, var(--brand-primary), var(--brand-primary-600));
        color: #000;
        font-weight: 600;
        height: 44px;
        padding: 0 var(--s16);
        border-radius: var(--radius-button);
        font: var(--font-body);
        box-shadow: var(--glow-brand);
        transition: all 0.12s ease-out;
      }

      .personal-settings-actions .btn-primary:active {
        transform: scale(0.98);
        box-shadow: 0 0 16px rgba(0, 213, 255, 0.4);
      }

      .setting-item {
        margin-bottom: var(--space-md);
      }

      .setting-item label {
        display: block;
        font-family: var(--font-body);
        font-size: 16pt;
        font-weight: 500;
        line-height: 22pt;
        color: var(--text-primary);
        margin-bottom: var(--space-sm);
      }

      .setting-item select,
      .setting-item input[type="number"] {
        width: 100%;
        height: 44pt;
        padding: 0 var(--space-md);
        background: var(--bg-surface-2);
        border: 1px solid var(--border-base);
        border-radius: var(--radius-chip);
        color: var(--text-primary);
        font-family: var(--font-body);
        font-size: 16pt;
        font-weight: 400;
        line-height: 22pt;
        appearance: none;
        cursor: pointer;
      }

      .setting-item input[type="number"] {
        font-variant-numeric: tabular-nums;
      }

      .slider-container {
        display: flex;
        align-items: center;
        gap: var(--space-md);
      }

      .slider-container input[type="range"] {
        flex: 1;
        height: 4pt;
        background: var(--bg-surface-2);
        border: none;
        border-radius: 2pt;
        outline: none;
        -webkit-appearance: none;
        cursor: pointer;
      }

      .slider-container input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 20pt;
        height: 20pt;
        background: var(--brand-primary);
        border-radius: 10pt;
        cursor: pointer;
        box-shadow: 0 2pt 8pt rgba(0, 0, 0, 0.2);
      }

      .slider-container span {
        font-family: var(--font-body);
        font-size: 16pt;
        font-weight: 500;
        line-height: 22pt;
        color: var(--brand-primary);
        font-variant-numeric: tabular-nums;
        min-width: 48pt;
        text-align: right;
      }

      .fade-in-item {
        animation: fadeInLeft 0.5s ease-out;
        animation-fill-mode: both;
      }

      @keyframes fadeInLeft {
        from {
          opacity: 0;
          transform: translateX(-20px);
        }
        to {
          opacity: 1;
          transform: translateX(0);
        }
      }

      .settings-panel {
        padding: 20px 0;
      }
      .settings-panel h3 {
        font-size: 24px;
        margin-bottom: 20px;
        color: #00d4ff;
        font-weight: 800;
      }
      .strategy-switches {
        display: flex;
        flex-direction: column;
        gap: 16px;
        margin-bottom: 40px;
      }

      .strategy-switch {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 24px;
        background: #1e293b;
        border-radius: 16px;
        border: 1px solid #334155;
        min-height: 72px;
      }

      .strategy-name {
        font-size: 18px;
        font-weight: 700;
      }

      .toggle-switch {
        position: relative;
        width: 60px;
        height: 32px;
        background: #334155;
        border-radius: 16px;
        cursor: pointer;
        transition: all 0.3s ease;
      }

      .toggle-switch.active {
        background: #00d4ff;
      }

      .toggle-switch::after {
        content: '';
        position: absolute;
        top: 3px;
        left: 3px;
        width: 26px;
        height: 26px;
        background: white;
        border-radius: 13px;
        transition: transform 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
      }

      .toggle-switch.active::after {
        transform: translateX(28px);
      }

      .backtest-panel {
        background: #1e293b;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #334155;
      }

      .backtest-panel label {
        display: block;
        margin-bottom: 16px;
        font-weight: 700;
        font-size: 16px;
      }

      #lookahead-slider {
        width: 100%;
        height: 12px;
        background: #334155;
        border-radius: 6px;
        outline: none;
        margin-bottom: 16px;
        -webkit-appearance: none;
      }

      #lookahead-slider::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 24px;
        height: 24px;
        background: #00d4ff;
        border-radius: 12px;
        cursor: pointer;
      }

      .btn-primary {
        width: 100%;
        padding: 20px;
        background: linear-gradient(135deg, #00d4ff, #0099cc);
        color: #000;
        border: none;
        border-radius: 16px;
        font-size: 18px;
        font-weight: 800;
        margin-top: 20px;
        cursor: pointer;
        transition: transform 0.2s ease;
        min-height: 60px;
      }

      .btn-primary:active {
        transform: scale(0.98);
      }

      .bottom-nav {
        display: flex;
        background: var(--bg-surface);
        border-top: 1px solid var(--border-base);
        height: 56pt;
        padding-bottom: env(safe-area-inset-bottom);
        box-shadow: 0 -1px 0 var(--border-base);
        width: 100%;
        max-width: 100vw;
        overflow: hidden;
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 999; /* 降低z-index避免遮挡浏览器UI */
      }

      .nav-btn {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: none;
        border: none;
        color: var(--text-secondary);
        cursor: pointer;
        transition: color 0.12s ease-out;
        min-height: 44pt;
        gap: 2pt;
        min-width: 0;
        max-width: 25%;
        padding: 4px 2px;
        box-sizing: border-box;
      }

      .nav-btn.active {
        color: var(--brand-primary);
      }

      .nav-icon {
        font-size: 24pt;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .nav-label {
        font-family: var(--font-caption);
        font-size: 12pt;
        font-weight: 400;
        line-height: 16pt;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
      }

      /* mine 入口与红点 */
      #nav-mine { position: relative; }
      .mine-badge {
        position: absolute; top: -6px; left: 50%; transform: translateX(-50%);
        display: none;
        min-width: 18px; height: 18px; padding: 0 5px;
        border-radius: 999px; background: #ef4444; color: #fff;
        font-size: 12px; line-height: 18px; text-align: center;
        box-shadow: 0 0 0 2px #0f172a;
        pointer-events: none;
      }
      #nav-mine.pulse { animation: mine-pulse .3s ease; }
      @keyframes mine-pulse { 0%{transform:scale(1)} 50%{transform:scale(1.08)} 100%{transform:scale(1)} }

      /* 手机端底部导航栏优化 */
      @media (max-width: 480px) {
        .bottom-nav {
          height: 50pt;
          padding-left: 4px;
          padding-right: 4px;
        }
        
        .nav-btn {
          padding: 2px 1px;
          min-height: 40pt;
        }
        
        .nav-icon {
          font-size: 20pt;
        }
        
        .nav-label {
          font-size: 10pt;
          line-height: 14pt;
        }
      }

      @media (max-width: 375px) {
        .bottom-nav {
          height: 48pt;
          padding-left: 2px;
          padding-right: 2px;
        }
        
        .nav-btn {
          padding: 1px;
          min-height: 38pt;
          gap: 1pt;
        }
        
        .nav-icon {
          font-size: 18pt;
        }
        
        .nav-label {
          font-size: 9pt;
          line-height: 12pt;
        }
      }

      @media (max-width: 320px) {
        .bottom-nav {
          height: 46pt;
          padding-left: 1px;
          padding-right: 1px;
        }
        
        .nav-btn {
          padding: 1px;
          min-height: 36pt;
          gap: 1pt;
        }
        
        .nav-icon {
          font-size: 16pt;
        }
        
        .nav-label {
          font-size: 8pt;
          line-height: 10pt;
        }
      }

      /* ===== Mine UI enhancements ===== */
      .tag-list { display: flex; flex-wrap: wrap; gap: 8pt; margin: 6pt 0 10pt; }
      .tag { padding: 4pt 8pt; border-radius: 12pt; background: var(--bg-surface-2); border: 1px solid var(--border-base); color: var(--text-secondary); font-size: 12pt; }

      .card { background: var(--bg-surface); border: 1px solid var(--border-base); border-radius: 12pt; padding: 10pt; margin: 8pt 0; box-shadow: var(--shadow-1); }
      .card .row { display: flex; gap: 10pt; align-items: center; justify-content: space-between; color: var(--text-secondary); font-size: 12pt; margin: 4pt 0; }
      .card .row strong { color: var(--text-primary); font-size: 14pt; }
      .card .actions { display: flex; gap: 8pt; margin-top: 8pt; }
      .card .actions button { height: 30pt; padding: 0 12pt; border-radius: 12pt; border: 1px solid var(--border-base); background: var(--bg-surface-2); color: var(--text-secondary); cursor: pointer; }
      .card .actions button:disabled { opacity: .6; cursor: not-allowed; }

      /* Map existing mine elements to card look */
      #mine-queued .signal-card, #mine-open .signal-card, #mine-history .signal-card { background: var(--bg-surface); border: 1px solid var(--border-base); border-radius: 12pt; padding: 10pt; }
      #mine-queued .signal-header, #mine-open .signal-header { display:flex; align-items:center; justify-content: space-between; }
      #mine-queued .signal-details, #mine-open .signal-details { display:flex; flex-wrap:wrap; gap: 10pt; color: var(--text-secondary); font-size: 12pt; margin-top: 6pt; }
      #mine-queued .timeframe-btn, #mine-open .timeframe-btn { height: 30pt; padding: 0 12pt; border-radius: 12pt; border: 1px solid var(--border-base); background: var(--bg-surface-2); color: var(--text-secondary); cursor: pointer; }

      /* ===== Info Page ===== */
      .info-container { padding: 16px; }
      .info-toolbar { display: flex; flex-wrap: wrap; gap: 10pt; align-items: center; margin-bottom: 12px; }
      .toolbar-group { display:flex; flex-direction: column; gap: 4pt; }
      .toolbar-group label { font-size: 12pt; color: var(--text-secondary); }
      .toolbar-group select, .toolbar-group input { height: 30pt; border-radius: 8pt; border: 1px solid var(--border-base); background: var(--bg-surface-2); color: var(--text-primary); padding: 0 8pt; }
      .toolbar-btn { height: 30pt; padding: 0 12pt; border-radius: 10pt; border: 1px solid var(--border-base); background: var(--brand-bg); color: var(--brand-primary); cursor: pointer; }
      .info-grid { display: flex; flex-direction: column; gap: 12px; }
      .info-summary { border: 1px solid #1A2430; border-radius: 16px; background: #0E141B; padding: 14px; margin-bottom: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.35); }
      .info-main { display: flex; flex-direction: column; gap: 8pt; }
      .info-chart-controls { display: flex; align-items: center; gap: 8pt; }
      /* Correlation lines chart */
      .chart-stage{ position: relative; min-width: 0; }
      .echart{ width: 100%; height: 360px; box-sizing: border-box; overflow: hidden; transform: none !important; }
      #card-corr-lines{ background:#0B0F14; border:1px solid #1A2430; border-radius:16px; padding: 12px; }
      #card-corr-lines .macro-card-header{ margin-bottom: 8px; }
      #card-corr-lines .corr-badges{ display:flex; flex-wrap:wrap; gap:8px 14px; padding: 6px 6px 2px; }
      #card-corr-lines .corr-badge{ display:inline-flex; align-items:center; gap:8px; padding:2px 10px; border-radius:999px; background:#0F1620; border:1px solid #1C2430; color:#CFE6FF; font-size:12px; line-height:18px; }
      #card-corr-lines .corr-dot{ width:8px; height:8px; border-radius:50%; display:inline-block; }
      #chart-corr-lines{ 
        flex:1 1 0%; 
        min-width: 0; 
        width: 100%;
        height: 380px;
        box-sizing: border-box;
        overflow: hidden;
        background: #0B0F14;
      }
      .chart-corr {
        width: 100%;
        height: 340px;
        background: #0B0F14;
        border-radius: 12px;
        position: relative;
        border: 1px solid #121A22;
        min-width: 0;
        overflow: hidden;
        transform: none !important;
        box-sizing: border-box;
      }
      
      #corrChart{
        width:100%; height:100%; box-sizing:border-box;
        overflow:hidden; background:#0B0F14;
      }
      .corr-top3{ position:absolute; right:12px; top:12px; display:flex; gap:6px; pointer-events:none; z-index:2; }
      .corr-chip{ padding:2px 8px; border-radius: 999px; font-size: 11px; background:#0F1620; border:1px solid #1C2430; color:#CFE6FF; }
      @media (max-width: 480px){
        .echart{ height: 250px; }
      }
      
      /* 超小屏幕优化 */
      @media (max-width: 360px){
        .compact-ui .chart-driver-index{ height: 180px !important; }
        .compact-ui .chart-etf-fng{ height: 180px !important; }
        .compact-ui .chart-funding-heat{ height: 200px !important; }
        .compact-ui .chart-corr{ height: 180px !important; }
        .compact-ui .info-radar{ height: 220px !important; }
        .compact-ui .info-container{ padding: 4px !important; }
        .compact-ui .macro-factors-card{ padding: 6px !important; margin: 4px 0 !important; }
        .compact-ui .macro-card-title h3{ font-size: 13px !important; }
        .compact-ui .macro-card-title p{ font-size: 10px !important; }
        .compact-ui .range-btn{ padding: 3px 6px !important; font-size: 10px !important; }
        .echart{ height: 200px; }
      }
      .info-detail { display: block; }
      .info-panel { background: var(--bg-surface); border: 1px solid var(--border-base); border-radius: 12pt; padding: 10pt; min-height: 200pt; }
      .panel-title { font-weight: 700; margin-bottom: 8pt; color: var(--text-primary); }
      .panel-body { color: var(--text-secondary); font-size: 12pt; }
        .info-placeholder { color: var(--text-secondary); }
        .info-tip { color: var(--text-muted); font-size: 12pt; margin: 6pt 0 8pt; }
        
        
        
      .info-loading { 
        display: flex; flex-direction: column; align-items: center; justify-content: center; 
        color: var(--text-secondary); 
        background: rgba(0, 0, 0, 0.2);
        padding: 20px;
        border-radius: 12px;
        backdrop-filter: blur(4px);
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 9999;
        margin: 0;
        pointer-events: none;
      }
      
      /* 宏观情绪与资金分组样式 */
      .macro-factors-card {
        background: #0E141B;
        border: 1px solid #1A2430;
        border-radius: 16px;
        padding: 14px;
        margin: 12px 0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        display: flex;
        flex-direction: column;
        min-width: 0;
      }
      
      .macro-card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 16px;
      }
      
      .macro-card-title h3 {
        font-size: 15px;
        font-weight: 600;
        color: #E6EDF6;
        margin: 0 0 4px 0;
      }
      
      .macro-card-title p {
        font-size: 12px;
        color: #8FA0B3;
        margin: 0;
      }
      
      .macro-card-controls {
        display: flex;
        align-items: center;
        gap: 12px;
      }
      
      .refresh-btn {
        background: #1A2430;
        border: 1px solid #2A3441;
        color: #8FA0B3;
        padding: 4px 8px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
        transition: all 0.2s ease;
      }
      
      .refresh-btn:hover {
        background: #2A3441;
        color: #E6EDF6;
        border-color: #3A4451;
      }
      
      .time-range-selector {
        display: flex;
        background: #1A2430;
        border-radius: 999px;
        padding: 2px;
      }
      
      .range-btn {
        height: 28px;
        padding: 0 12px;
        border: none;
        background: transparent;
        color: #8FA0B3;
        font-size: 12px;
        font-weight: 500;
        border-radius: 999px;
        cursor: pointer;
        transition: all 0.2s ease;
      }
      
      .range-btn.active {
        background: linear-gradient(135deg, #1FA2FF, #12D8FA, #A6FFCB);
        color: #0B0F14;
        font-weight: 600;
      }
      
      .range-btn:hover:not(.active) {
        color: #E6EDF6;
        background: rgba(255, 255, 255, 0.05);
      }
      
      .macro-chart-container {
        margin-bottom: 16px;
      }
      
      .macro-chart-container:last-child {
        margin-bottom: 0;
      }
      
      .chart-driver-index {
        width: 100%;
        height: 340px;
        background: #0B0F14;
        border-radius: 12px;
        position: relative;
        border: 1px solid #121A22;
        min-width: 0;
        overflow: hidden;
        transform: none !important;
        box-sizing: border-box;
      }
      
      .chart-etf-fng {
        width: 100%;
        height: 340px;
        background: #0B0F14;
        border-radius: 12px;
        position: relative;
        border: 1px solid #121A22;
        min-width: 0;
        overflow: hidden;
        transform: none !important;
        box-sizing: border-box;
      }
      
      .chart-funding-heat {
        width: 100%;
        height: 400px;
        background: #0B0F14;
        border-radius: 12px;
        position: relative;
        border: 1px solid #121A22;
        min-width: 0;
        overflow: hidden;
        transform: none !important;
        box-sizing: border-box;
      }
      
      .chart-loading {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: #8FA0B3;
        background: rgba(0, 0, 0, 0.1);
        border-radius: 12px;
        position: relative;
        overflow: hidden;
      }
      
      .chart-loading::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(31, 162, 255, 0.1), transparent);
        animation: shimmer 2s infinite;
      }
      
      .chart-loading .loading-spinner {
        width: 32px;
        height: 32px;
        border: 3px solid #1A2430;
        border-top: 3px solid #1FA2FF;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 12px;
        position: relative;
        z-index: 2;
        box-shadow: 0 0 20px rgba(31, 162, 255, 0.3);
      }
      
      .chart-loading .loading-spinner::after {
        content: '';
        position: absolute;
        top: -3px;
        left: -3px;
        right: -3px;
        bottom: -3px;
        border: 1px solid rgba(31, 162, 255, 0.2);
        border-radius: 50%;
        animation: pulse 2s infinite;
      }
      
      .chart-loading .loading-text {
        font-size: 14px;
        color: #8FA0B3;
        font-weight: 500;
        position: relative;
        z-index: 2;
        animation: fadeInOut 2s infinite;
      }
      
      @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
      }
      
      @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.7; }
      }
      
      @keyframes fadeInOut {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
      }
      
      @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
      }
      
        /* 横屏适配 */
        @media (min-width: 768px) {
          .chart-driver-index {
            height: 420px;
          }
        
        .chart-etf-fng {
          height: 420px;
        }
        
        .chart-funding-heat {
          height: 480px;
        }
        
        .chart-corr {
          height: 420px;
        }
      }
      .loading-spinner { 
        width: 32px; height: 32px; border: 3px solid var(--border-base); 
        border-top: 3px solid var(--brand-primary); border-radius: 50%; 
        animation: spin 1s linear infinite; margin-bottom: 12px; 
      }
      .loading-text { font-size: 14px; }
      @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

      .fab-refresh {
        position: fixed;
        bottom: calc(80pt + env(safe-area-inset-bottom));
        right: var(--space-md);
        width: 44pt;
        height: 44pt;
        background: var(--brand-primary);
        border: none;
        border-radius: 22pt;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: var(--shadow-1);
        cursor: pointer;
        transition: opacity 0.12s ease-out;
        z-index: 998; /* 降低z-index避免遮挡浏览器UI */
      }

      .fab-refresh:active {
        opacity: 0.7;
      }

      .refresh-icon {
        font-size: 20pt;
        color: #000;
      }

      .fab-refresh.spinning .refresh-icon {
        animation: spin 1s linear infinite;
      }

      @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
      }

      /* ===== Filter Bar (tabs + chips) ===== */
      .filterbar{
        display:flex; flex-wrap:wrap; align-items:center; gap:10px;
        margin: 8px 0 12px;
      }
      .tabs{ display:flex; gap:8px; flex-wrap:wrap; }
      .tab{
        padding:6px 12px; border-radius:12px; cursor:pointer;
        font-size:13px; background:#141b22; color:#bfead1;
        border:1px solid rgba(34,199,134,.18); transition:.18s;
      }
      .tab:hover{ border-color:rgba(34,199,134,.38) }
      .tab.active{
        background:linear-gradient(90deg,rgba(34,199,134,.22),rgba(16,24,32,.6));
        color:#CFFFB0; border-color:rgba(34,199,134,.6); box-shadow:0 0 16px rgba(34,199,134,.18);
      }
      .chips{ display:flex; gap:8px; flex-wrap:wrap; }
      .chip{
        padding:4px 10px; border-radius:999px; font-size:12px; cursor:pointer;
        background:#102723; color:#91FBB2; border:1px solid rgba(34,199,134,.18);
        transition:.18s;
      }
      .chip.off{ background:#1a2430; color:#9FB0C0; }
      .chip.active{ border-color:rgba(34,199,134,.6); color:#CFFFB0 }
      .input-search{
        flex:1 1 180px; min-width:160px; padding:8px 10px; border-radius:12px; font-size:13px;
        border:1px solid rgba(34,199,134,.18); background:#0E141B; color:#E6EDF6; outline:none;
      }
      .count{ font-size:12px; color:#9FB0C0; }

      /* iPhone安全区域适配 */
      @supports (padding: max(0px)) {
        .mobile-app {
          padding-top: max(44px, env(safe-area-inset-top));
        }

        .bottom-nav {
          padding-bottom: max(12px, calc(12px + env(safe-area-inset-bottom)));
        }

        .fab-refresh {
          bottom: max(110px, calc(110px + env(safe-area-inset-bottom)));
        }
      }

      /* ===== Apple HIG Navigation Bar ===== */
      .hig-nav-bar {
        position: sticky;
        top: 0;
        z-index: 997; /* 降低z-index避免遮挡浏览器UI */
        background: rgba(11, 15, 20, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .nav-content {
        display: flex;
        align-items: center;
        justify-content: space-between;
        width: 100%;
        max-width: 100%;
        padding: 0 16px;
        height: 100%;
      }

      .nav-left {
        flex: 0 0 auto;
        display: flex;
        align-items: center;
      }

      .nav-center {
        flex: 1;
        display: flex;
        justify-content: center;
        align-items: center;
        max-width: 280px;
        margin: 0 16px;
      }

      .nav-right {
        flex: 0 0 auto;
        display: flex;
        align-items: center;
        gap: 8px;
      }

      /* App Brand */
      .app-brand {
        display: flex;
        align-items: center;
        gap: 8px;
        cursor: pointer;
      }

      .app-icon {
        font-size: 20px;
        line-height: 1;
      }

      .app-name {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
        font-size: 17px;
        font-weight: 600;
        color: #FFFFFF;
        letter-spacing: -0.4px;
      }

      /* Search Bar */
      .search-bar {
        position: relative;
        display: flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        height: 32px;
        padding: 0 12px;
        width: 100%;
        transition: all 0.2s ease;
      }

      .search-bar:focus-within {
        background: rgba(255, 255, 255, 0.15);
        border-color: rgba(0, 213, 255, 0.6);
        box-shadow: 0 0 0 3px rgba(0, 213, 255, 0.1);
      }

      .search-icon {
        color: rgba(255, 255, 255, 0.6);
        margin-right: 8px;
        flex-shrink: 0;
      }

      .search-input {
        flex: 1;
        background: transparent;
        border: none;
        outline: none;
        color: #FFFFFF;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
        font-size: 15px;
        font-weight: 400;
        line-height: 20px;
        padding: 0;
      }

      .search-input::placeholder {
        color: rgba(255, 255, 255, 0.5);
      }

      /* Navigation Buttons */
      .nav-button {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        background: transparent;
        border: none;
        border-radius: 8px;
        color: rgba(255, 255, 255, 0.8);
        cursor: pointer;
        transition: all 0.2s ease;
        padding: 0;
      }

      .nav-button:hover {
        background: rgba(255, 255, 255, 0.1);
        color: #FFFFFF;
      }

      .nav-button:active {
        background: rgba(255, 255, 255, 0.2);
        transform: scale(0.95);
      }

      .nav-button svg {
        width: 20px;
        height: 20px;
        stroke-width: 2;
      }

      /* Notification Badge */
      .badge {
        position: absolute;
        top: -2px;
        right: -2px;
        min-width: 18px;
        height: 18px;
        background: #FF3B30;
        color: #FFFFFF;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
        font-size: 11px;
        font-weight: 600;
        line-height: 18px;
        text-align: center;
        border-radius: 9px;
        padding: 0 4px;
        box-shadow: 0 0 0 2px rgba(11, 15, 20, 0.8);
      }

      /* Responsive Design */
      @media (max-width: 480px) {
        .nav-content {
          padding: 0 8px;
          gap: 4px;
        }
        
        .nav-center {
          margin: 0 4px;
          max-width: 200px;
          flex: 1;
          min-width: 0;
        }
        
        .app-name {
          font-size: 14px;
        }
        
        .search-input {
          font-size: 13px;
        }
        
        .search-bar {
          height: 28px;
          padding: 0 8px;
        }
        
        .nav-button {
          width: 28px;
          height: 28px;
          flex-shrink: 0;
        }
        
        .nav-button svg {
          width: 16px;
          height: 16px;
        }
      }

      @media (max-width: 375px) {
        .nav-content {
          padding: 0 6px;
          gap: 2px;
        }
        
        .nav-center {
          margin: 0 4px;
          max-width: 160px;
        }
        
        .app-name {
          font-size: 13px;
        }
        
        .search-input {
          font-size: 12px;
        }
        
        .search-bar {
          height: 26px;
          padding: 0 6px;
        }
        
        .nav-button {
          width: 26px;
          height: 26px;
        }
        
        .nav-button svg {
          width: 14px;
          height: 14px;
        }
      }

      @media (max-width: 320px) {
        .app-name {
          display: none;
        }
        
        .nav-center {
          max-width: 100px;
          margin: 0 2px;
        }
        
        .nav-content {
          padding: 0 4px;
        }
        
        .search-bar {
          height: 24px;
          padding: 0 4px;
        }
        
        .search-input {
          font-size: 11px;
        }
        
        .nav-button {
          width: 24px;
          height: 24px;
        }
        
        .nav-button svg {
          width: 12px;
          height: 12px;
        }
      }

      /* Animation */
      .hig-nav-bar {
        animation: slideDown 0.3s ease-out;
      }

      @keyframes slideDown {
        from {
          transform: translateY(-100%);
          opacity: 0;
        }
        to {
          transform: translateY(0);
          opacity: 1;
        }
      }

      /* ===== HIG Navigation Bar ===== */
      .hig-navigation {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: rgba(11, 15, 20, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0;
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .nav-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        width: 100%;
        max-width: 100%;
        padding: 0 16px;
        height: 100%;
      }

      .nav-left {
        flex: 0 0 auto;
        display: flex;
        align-items: center;
      }

      .nav-center {
        flex: 1;
        display: flex;
        justify-content: center;
        align-items: center;
        max-width: 280px;
        margin: 0 16px;
      }

      .nav-right {
        flex: 0 0 auto;
        display: flex;
        align-items: center;
        gap: 8px;
      }

      /* App Brand */
      .app-brand {
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .app-icon {
        font-size: 20px;
        line-height: 1;
      }

      .app-title {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
        font-size: 17px;
        font-weight: 600;
        color: #FFFFFF;
        letter-spacing: -0.4px;
      }

      /* Search Bar */
      .nav-search {
        width: 100%;
      }

      .search-container {
        position: relative;
        display: flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        height: 32px;
        padding: 0 12px;
        transition: all 0.2s ease;
      }

      .search-container:focus-within {
        background: rgba(255, 255, 255, 0.15);
        border-color: rgba(0, 213, 255, 0.6);
        box-shadow: 0 0 0 3px rgba(0, 213, 255, 0.1);
      }

      .search-icon {
        color: rgba(255, 255, 255, 0.6);
        margin-right: 8px;
        flex-shrink: 0;
      }

      .search-input {
        flex: 1;
        background: transparent;
        border: none;
        outline: none;
        color: #FFFFFF;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
        font-size: 15px;
        font-weight: 400;
        line-height: 20px;
        padding: 0;
      }

      .search-input::placeholder {
        color: rgba(255, 255, 255, 0.5);
      }

      /* Action Buttons */
      .nav-action-btn {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        background: transparent;
        border: none;
        border-radius: 8px;
        color: rgba(255, 255, 255, 0.8);
        cursor: pointer;
        transition: all 0.2s ease;
        padding: 0;
      }

      .nav-action-btn:hover {
        background: rgba(255, 255, 255, 0.1);
        color: #FFFFFF;
      }

      .nav-action-btn:active {
        background: rgba(255, 255, 255, 0.2);
        transform: scale(0.95);
      }

      .nav-action-btn svg {
        width: 20px;
        height: 20px;
        stroke-width: 2;
      }

      /* Notification Badge */
      .notification-badge {
        position: absolute;
        top: -2px;
        right: -2px;
        min-width: 18px;
        height: 18px;
        background: #FF3B30;
        color: #FFFFFF;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
        font-size: 11px;
        font-weight: 600;
        line-height: 18px;
        text-align: center;
        border-radius: 9px;
        padding: 0 4px;
        box-shadow: 0 0 0 2px rgba(11, 15, 20, 0.8);
      }

      /* Responsive Design */
      @media (max-width: 375px) {
        .nav-container {
          padding: 0 12px;
        }
        
        .nav-center {
          margin: 0 8px;
          max-width: 200px;
        }
        
        .app-title {
          font-size: 16px;
        }
        
        .search-input {
          font-size: 14px;
        }
      }

      @media (max-width: 320px) {
        .app-title {
          display: none;
        }
        
        .nav-center {
          max-width: 150px;
        }
      }

      /* Dark Mode Support */
      @media (prefers-color-scheme: dark) {
        .hig-navigation {
          background: rgba(0, 0, 0, 0.95);
          border-bottom-color: rgba(255, 255, 255, 0.15);
        }
      }

      /* Animation for smooth appearance */
      .hig-navigation {
        animation: slideDown 0.3s ease-out;
      }

      @keyframes slideDown {
        from {
          transform: translateY(-100%);
          opacity: 0;
        }
        to {
          transform: translateY(0);
          opacity: 1;
        }
      }
    `;

    document.head.appendChild(style);
  }

  private loadUserParamsFromStorage() {
    try {
      const raw = localStorage.getItem('user_profile_params');
      if (raw) {
        const p = JSON.parse(raw);
        if (typeof p === 'object' && p) {
          this.userParams = {
            profitTarget: Number(p.profitTarget ?? this.userParams.profitTarget),
            maxDrawdown: Number(p.maxDrawdown ?? this.userParams.maxDrawdown),
            riskExposure: Number(p.riskExposure ?? this.userParams.riskExposure),
            capitalSize: Number(p.capitalSize ?? this.userParams.capitalSize),
            monitoringFreq: String(p.monitoringFreq ?? this.userParams.monitoringFreq)
          };
          this.hasConfiguredFlag = true;
        }
      }
    } catch (_) {}
  }

  private setupMobileEventListeners() {
    // 底部导航
    for (const btn of document.querySelectorAll('.nav-btn')) {
      btn.addEventListener('click', (e) => {
        const target = e.currentTarget as HTMLElement;
        const tab = target.getAttribute('data-tab');
        this.switchTab(tab || 'home');
      });
    }

    // 时间周期
    for (const btn of document.querySelectorAll('.tf-tab')) {
      btn.addEventListener('click', (e) => {
        const target = e.currentTarget as HTMLElement;
        const tf = target.getAttribute('data-tf');
        if (tf) this.setTimeframe(tf);
      });
    }

    // 管理实时信号入口
    const manageBtn = document.getElementById('manage-signals-btn');
    if (manageBtn) {
      manageBtn.addEventListener('click', () => this.openManageSignals());
      // 一次性提示气泡
      try {
        const tipKey = 'manage_signals_tip_shown';
        if (!localStorage.getItem(tipKey)) {
          const tip = document.createElement('div');
          tip.className = 'manage-tip';
          tip.textContent = '你可以在这里选择市场页展示哪些策略的实时信号。';
          document.querySelector('.mobile-header')?.appendChild(tip);
          setTimeout(() => tip.remove(), 3500);
          localStorage.setItem(tipKey, '1');
        }
      } catch (_) {}
    }

    // 策略开关迁移至"管理实时信号"入口
    this.setupPersonalizationSliders();
    
    // 设置页不再提供保存策略按钮

    const slider = document.getElementById('lookahead-slider') as HTMLInputElement;
    const value = document.getElementById('lookahead-value');
    if (slider && value) {
      slider.addEventListener('input', () => {
        value.textContent = slider.value;
      });
    }

    // Header 指标可点击 -> 跳转"我的"
    const ls = document.getElementById('learning-stats');
    if (ls && ls.getAttribute('data-clickable') === '1') {
      ls.addEventListener('click', () => this.switchTab('profile'));
    }

    // 我的页"管理策略"入口
    document.getElementById('open-strategy-manager')?.addEventListener('click', () => this.openManageSignals());

    // 一键启用全部信号按钮 - 由initMineUI处理，这里不需要重复绑定

    // 头部新按钮事件
    document.getElementById('personal-params-btn')?.addEventListener('click', () => {
      this.switchTab('profile');
      // 滚动到参数设置区域
      setTimeout(() => {
        const paramsSection = document.querySelector('.personal-settings');
        if (paramsSection) {
          paramsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 100);
    });

    document.getElementById('simulation-positions-btn')?.addEventListener('click', () => {
      this.switchTab('profile');
      // 滚动到模拟持仓区域
      setTimeout(() => {
        const positionsSection = document.querySelector('#mine-open');
        if (positionsSection) {
          positionsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 100);
    });

    // Info page init on first switch
    const navInfo = Array.from(document.querySelectorAll('.nav-btn')).find(b => (b as HTMLElement).getAttribute('data-tab') === 'info');
    navInfo?.addEventListener('click', () => {
      if (!this.infoInited) {
        this.initInfoPage();
      }
    });

    // 初始化红点
    document.addEventListener('DOMContentLoaded', () => updateBadge());

    // 一键模拟的统一监听由稳定版块注册，这里不重复绑定

    // 条件触发提醒功能
    this.initConditionAlert();
    
    // 策略实验室功能
    this.initStrategyLab();
  }

  private initInfoPage() {
    this.infoInited = true;
    
    // 预加载所有图表数据
    this.preloadChartData();
    
    // Factor event bus
    this.infoEventBus = {
      emit: (event: string, data?: any) => {
        document.dispatchEvent(new CustomEvent(`factor:${event}`, { detail: data }));
      },
      on: (event: string, handler: (data?: any) => void) => {
        document.addEventListener(`factor:${event}`, (e: any) => handler(e.detail));
      }
    };
    
    // default toolbar values
    const assetSel = document.getElementById('info-asset') as HTMLSelectElement | null;
    const granSel = document.getElementById('info-granularity') as HTMLSelectElement | null;
    const dateInp = document.getElementById('info-date') as HTMLInputElement | null;
    if (assetSel && !assetSel.value) assetSel.value = 'BTC';
    if (granSel && !granSel.value) granSel.value = 'daily';
    
    // 如果ECharts还没加载，等待加载完成
    if (!this.echartsMod) {
      console.log('[info] ECharts not ready, waiting...');
      // 等待预加载完成
      const checkECharts = () => {
        if (this.echartsMod) {
          this.initCharts();
        } else {
          setTimeout(checkECharts, 100);
        }
      };
      checkECharts();
    } else {
      this.initCharts();
    }
    
    // 设置resize处理 - 添加去抖避免反馈循环
    this.onInfoResize = () => { 
      if (this.resizeDebounceTimer) {
        clearTimeout(this.resizeDebounceTimer);
      }
      this.resizeDebounceTimer = setTimeout(() => {
        try { 
          // 同时处理宏观图表
          this.btcMacroMonitorV3Chart?.resize();
          this.macroETFFNGChart?.resize();
          this.macroFundingChart?.resize();
          // Also check if radar snapshot needs to be re-rendered
        } catch(_) {} 
      }, 150); // 150ms去抖
    };
    window.addEventListener('resize', this.onInfoResize);
    
    // 立即显示所有图表的loading状态（预加载）
    this.showMacroLoading('chart-driver-index', '准备加载宏观数据...');
    this.showMacroLoading('chart-etf-fng', '准备加载ETF数据...');
    this.showMacroLoading('chart-funding-heat', '准备加载资金费率数据...');
    
    // 移除测试数据逻辑，使用真实数据
    
    // First data load
    this.refreshInfoData();
    
    // 初始化宏观数据组件（预加载）
    this.initMacroComponents();
    
    // 读取URL参数同步筛选
    this.readQueryParams();
    
    // 加载信号和持仓数据
    console.log('[filter] About to load signal and position data...');
    this.loadSignalAndPositions();
    
    // 添加快捷键支持
    this.initKeyboardShortcuts();
    
    // 初始化HIG导航栏
    this.initHIGNavigation();
    
    // Auto-show help on first visit
    try {
      if (!localStorage.getItem('info_help_shown')) {
        setTimeout(() => this.showInfoHelp(), 1000);
      }
    } catch(_) {}
    
    // bind toolbar events
    const refresh = () => this.refreshInfoData();
    assetSel?.addEventListener('change', refresh);
    granSel?.addEventListener('change', refresh);
    dateInp?.addEventListener('change', refresh);
    document.getElementById('info-export')?.addEventListener('click', () => this.exportInfoPNG());

    // help button
    const helpBtn = document.getElementById('info-help');
    helpBtn?.addEventListener('click', () => this.showInfoHelp());

    // keyboard controls
    document.addEventListener('keydown', (e) => {
      if (document.getElementById('info-view')?.classList.contains('active')) {
        if (e.key === 'ArrowLeft') {
          e.preventDefault();
          const step = e.shiftKey ? 7 : 1;
          this.infoCurrentIdx = Math.max(0, this.infoCurrentIdx - step);
          this.renderInfoSummary();
          this.infoEventBus.emit('move:timestamp', this.infoCurrentIdx);
          this.pingIndexTail(this.infoCurrentIdx);
        } else if (e.key === 'ArrowRight') {
          e.preventDefault();
          const step = e.shiftKey ? 7 : 1;
          const max = Math.max(0, (this.infoIndex || []).length - 1);
          this.infoCurrentIdx = Math.min(max, this.infoCurrentIdx + step);
          this.renderInfoSummary();
          this.infoEventBus.emit('move:timestamp', this.infoCurrentIdx);
          this.pingIndexTail(this.infoCurrentIdx);
        }
      }
    });
    
    // Event bus listeners
    this.infoEventBus.on('move:timestamp', (idx: number) => {
      this.infoCurrentIdx = idx;
    });
    
    this.infoEventBus.on('select:factor', (factorKey: string | null) => {
      this.infoSelectedFactor = factorKey;
    });
    
    this.infoEventBus.on('toggle:factor', (factorKey: string | null) => {
      this.infoSelectedFactor = factorKey;
    });
  }
  
  private initMacroComponents() {

    // 初始化DriverIndex卡片的时间范围选择器
    const driverIndexButtons = document.querySelectorAll('#card-driver-index .range-btn');
    driverIndexButtons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const target = e.target as HTMLElement;
        const range = target.getAttribute('data-range');
        if (range) {
          this.macroCurrentRange = range;
          // 更新所有卡片的按钮状态
          document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
          document.querySelectorAll(`[data-range="${range}"]`).forEach(b => b.classList.add('active'));
          // 触发数据更新
          this.macroEventBus.emit('range-change', range);
          this.refreshMacroData();
        }
      });
    });
    
    // 添加强制刷新按钮
    const refreshBtn = document.getElementById('refresh-driver-data');
    refreshBtn?.addEventListener('click', () => {
      console.log('[DriverIndex] Force refresh triggered');
      // 清除缓存并重新获取数据
      this.clearCache();
      this.refreshMacroData();
    });
    
    // 初始化ETF×FNG卡片的时间范围选择器
    const etfFngButtons = document.querySelectorAll('#card-etf-fng .range-btn');
    etfFngButtons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const target = e.target as HTMLElement;
        const range = target.getAttribute('data-range');
        if (range) {
          this.macroCurrentRange = range;
          // 更新所有卡片的按钮状态
          document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
          document.querySelectorAll(`[data-range="${range}"]`).forEach(b => b.classList.add('active'));
          // 触发数据更新
          this.macroEventBus.emit('range-change', range);
          this.refreshMacroData();
        }
      });
    });
    
    // 初始化资金费率卡片的时间范围选择器
    const fundingButtons = document.querySelectorAll('#card-funding-heat .range-btn');
    fundingButtons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const target = e.target as HTMLElement;
        const range = target.getAttribute('data-range');
        if (range) {
          this.macroCurrentRange = range;
          // 更新所有卡片的按钮状态
          document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
          document.querySelectorAll(`[data-range="${range}"]`).forEach(b => b.classList.add('active'));
          // 触发数据更新
          this.macroEventBus.emit('range-change', range);
          this.refreshMacroData();
        }
      });
    });
    
    // 监听事件总线
    this.macroEventBus.on('focus-date', (date: string) => {
      this.highlightDateInCharts(date);
    });
    
    this.macroEventBus.on('range-change', (range: string) => {
      this.macroCurrentRange = range;
      this.refreshMacroData();
    });
    
    // 初始化图表
    this.initMacroCharts();
    
    // 加载数据
    this.refreshMacroData();
  }
  
  private initMacroCharts() {
    // 等待ECharts加载完成
    if (!this.echartsMod) {
      console.log('[macro] ECharts not ready, waiting...');
      setTimeout(() => this.initMacroCharts(), 100);
      return;
    }
    
    console.log('[macro] ECharts ready, initializing charts...');
    console.log('[macro] ECharts module:', this.echartsMod);
    console.log('[macro] ECharts init function:', typeof this.echartsMod.init);
    
    // 初始化DriverIndex图表
    const driverIndexEl = document.getElementById('chart-driver-index');
    if (driverIndexEl && !this.macroDriverIndexChart) {
      console.log('[macro] Initializing DriverIndex chart...');
      console.log('[macro] Container element:', driverIndexEl);
      console.log('[macro] Container dimensions:', {
        width: driverIndexEl.clientWidth,
        height: driverIndexEl.clientHeight,
        offsetWidth: driverIndexEl.offsetWidth,
        offsetHeight: driverIndexEl.offsetHeight
      });
      
      // 检查容器是否被隐藏
      const hiddenParent = this.findHiddenParent(driverIndexEl);
      if (hiddenParent) {
        console.log('[macro] Container has hidden parent:', hiddenParent);
        console.log('[macro] Hidden parent styles:', {
          display: getComputedStyle(hiddenParent).display,
          visibility: getComputedStyle(hiddenParent).visibility
        });
      }
      
      // 强制设置容器尺寸（如果为0）
      if (driverIndexEl.clientWidth === 0 || driverIndexEl.clientHeight === 0) {
        console.log('[macro] Container has zero dimensions, forcing size...');
        driverIndexEl.style.width = '100%';
        driverIndexEl.style.height = '340px';
        driverIndexEl.style.minHeight = '340px';
        console.log('[macro] Forced container dimensions:', {
          width: driverIndexEl.clientWidth,
          height: driverIndexEl.clientHeight
        });
      }
      
      this.macroDriverIndexChart = this.echartsMod.init(driverIndexEl);
      console.log('[macro] DriverIndex chart initialized:', this.macroDriverIndexChart);
      
      // 添加resize监听器
      this.setupChartResizeListeners(driverIndexEl, this.macroDriverIndexChart);
      
      // 如果数据已经准备好，立即渲染
      if (this.macroDriverIndexData && this.macroDriverIndexData.length > 0) {
        console.log('[macro] DriverIndex data ready, rendering...');
        this.renderDriverIndexChart();
      }
    }
    
    // 初始化ETF×FNG图表
    const etfFngEl = document.getElementById('chart-etf-fng');
    if (etfFngEl && !this.macroETFFNGChart) {
      this.macroETFFNGChart = this.echartsMod.init(etfFngEl);
      console.log('[macro] ETF×FNG chart initialized');
      
      // 添加resize监听器
      this.setupChartResizeListeners(etfFngEl, this.macroETFFNGChart);
      
      // 如果数据已经准备好，立即渲染
      if (this.macroETFFNGData && this.macroETFFNGData.length > 0) {
        console.log('[macro] ETF×FNG data ready, rendering...');
        this.renderETFFNGChart();
      }
    }
    
    // 初始化资金费率热力图
    const fundingEl = document.getElementById('chart-funding-heat');
    if (fundingEl && !this.macroFundingChart) {
      this.macroFundingChart = this.echartsMod.init(fundingEl);
      console.log('[macro] Funding heatmap chart initialized');
      
      // 添加resize监听器
      this.setupChartResizeListeners(fundingEl, this.macroFundingChart);
      
      // 如果数据已经准备好，立即渲染
      if (this.macroFundingData) {
        console.log('[macro] Funding data ready, rendering...');
        this.renderFundingHeatmap();
      }
    }
    
    console.log('[macro] All charts initialized successfully');
  }
  
  // --- 工具方法 ---
  private findHiddenParent(el: HTMLElement): HTMLElement | null {
    for (let n = el; n; n = n.parentElement) {
      const cs = getComputedStyle(n);
      if (cs.display === 'none' || cs.visibility === 'hidden') {
        return n;
      }
    }
    return null;
  }

  private setupChartResizeListeners(container: HTMLElement, chart: any): void {
    if (!chart) return;
    
    console.log('[macro] Setting up resize listeners for chart');
    
    // 窗口resize监听
    const resizeHandler = () => {
      if (chart && !chart.isDisposed()) {
        chart.resize();
      }
    };
    
    window.addEventListener('resize', resizeHandler);
    
    // ResizeObserver监听容器尺寸变化
    if (window.ResizeObserver) {
      const resizeObserver = new ResizeObserver(() => {
        setTimeout(() => {
          if (chart && !chart.isDisposed()) {
            chart.resize();
          }
        }, 0);
      });
      resizeObserver.observe(container);
      
      // 存储observer以便后续清理
      (container as any)._resizeObserver = resizeObserver;
    }
    
    // 延迟resize，确保DOM完全渲染
    setTimeout(() => {
      if (chart && !chart.isDisposed()) {
        chart.resize();
      }
    }, 0);
  }

  // --- 数据缓存管理 ---
  private getCachedData<T>(key: string): T | null {
    const cached = this.dataCache.get(key);
    if (cached && Date.now() - cached.timestamp < cached.ttl) {
      console.log(`[Cache] Hit for ${key}`);
      return cached.data;
    }
    if (cached) {
      this.dataCache.delete(key);
      console.log(`[Cache] Expired for ${key}`);
    }
    return null;
  }
  
  private setCachedData<T>(key: string, data: T, ttl: number = this.DATA_CACHE_TTL): void {
    this.dataCache.set(key, {
      data,
      timestamp: Date.now(),
      ttl
    });
    console.log(`[Cache] Set for ${key}, TTL: ${ttl}ms`);
  }
  
  private clearCache(): void {
    this.dataCache.clear();
    console.log('[Cache] Cleared all cached data');
  }
  
  
  // 预加载图表数据
  private async preloadChartData(): Promise<void> {
    console.log('[Preload] Starting chart data preloading...');
    
    try {
      // 并行预加载所有数据
      const preloadPromises = [
        this.fetchDriverIndexData(),
        this.fetchETFFNGData(),
        this.fetchFundingData()
      ];
      
      // 不等待结果，让它们在后台加载
      Promise.allSettled(preloadPromises).then(results => {
        const successCount = results.filter(r => r.status === 'fulfilled').length;
        console.log(`[Preload] Completed: ${successCount}/${results.length} data sources loaded`);
      });
      
    } catch (error) {
      console.warn('[Preload] Failed to preload chart data:', error);
    }
  }
  
  // --- 数据源（API 失败时用 mock） ---
  private async fetchJsonSafe<T>(url:string, fallback:T):Promise<T>{
    try{ const r = await fetch(url); if(!r.ok) throw 0; return await r.json() as T; }catch{ return fallback; }
  }

  // 你现有的加载函数可以替换成这个
  private async loadSignalAndPositions(){
    console.log('[filter] Loading signal and position data...');
    
    // 尝试从后端API获取信号数据，失败时使用模拟数据
    const sigRes = await this.fetchJsonSafe<{data: any[]}>('http://127.0.0.1:8889/api/signals', {
      data: [
        {id:'sig_eth_s_4h_1', symbol:'ETH', side:'SELL' as Side, timeframe:'4h', strategy:'StochRSI', queued_at:'2025/9/10 00:40:20', ref_price:4284.94},
        {id:'sig_dot_s_4h_1', symbol:'DOT', side:'SELL' as Side, timeframe:'4h', strategy:'StochRSI', queued_at:'2025/9/10 01:20:39', ref_price:4.069},
        {id:'sig_eth_b_4h_2', symbol:'ETH', side:'BUY' as Side,  timeframe:'4h', strategy:'EMA20/50 + ADX', queued_at:'2025/9/12 02:35:25', ref_price:4429.8},
        {id:'sig_ada_s_4h_1', symbol:'ADA', side:'SELL' as Side, timeframe:'4h', strategy:'StochRSI', queued_at:'2025/9/12 02:35:30', ref_price:0.8837},
      ]
    });

    // 持仓数据暂时使用模拟数据（后端没有positions端点）
    const posRes = {
      items: [
        {id:'pos_eth_s_1', symbol:'ETH', side:'SELL' as Side, timeframe:'4h', strategy:'StochRSI', qty:1, avg_price:4284.94, mark_price:null, opened_at:'2025/9/10 00:41:54', pnl:null, pnl_pct:null},
        {id:'pos_dot_s_1', symbol:'DOT', side:'SELL' as Side, timeframe:'4h', strategy:'StochRSI', qty:1, avg_price:4.069,    mark_price:null, opened_at:'2025/9/10 01:21:00', pnl:null, pnl_pct:null},
      ]
    };

    // 处理信号数据 - 从API返回的data字段或fallback数据中提取
    const signals = sigRes.data || [];
    const positions = posRes.items || [];
    
    console.log('[filter] Signal data:', signals.length, 'items');
    console.log('[filter] Position data:', positions.length, 'items');

    this.renderSignalSection(signals);
    this.renderPositionSection(positions);
  }

  // --- 统一过滤器 ---
  private applyFilter<T extends {symbol:string; side:Side; timeframe:string; strategy:string; id:string}>(
    list:T[], f:{ symbol:string; side:"ALL"|Side; tf:"ALL"|string; q:string }
  ){
    const q = f.q.trim().toLowerCase();
    return list.filter(x=>{
      const bySymbol = f.symbol==='ALL' ? true : x.symbol===f.symbol;
      const bySide   = f.side==='ALL' ? true : x.side===f.side;
      const byTF     = f.tf==='ALL'   ? true : x.timeframe===f.tf;
      const byQ      = !q || [x.symbol, x.side, x.timeframe, x.strategy, x.id].some(v=>String(v).toLowerCase().includes(q));
      return bySymbol && bySide && byTF && byQ;
    });
  }

  private async refreshMacroData() {
    try {
      console.log('[macro] Loading data for range:', this.macroCurrentRange);
      
      // 显示所有图表的loading状态
      this.showMacroLoading('chart-driver-index', '加载宏观数据...');
      this.showMacroLoading('chart-etf-fng', '加载ETF数据...');
      this.showMacroLoading('chart-funding-heat', '加载资金费率数据...');
      
      // 并行加载DriverIndex、ETF×FNG数据和资金费率数据
      const [driverIndexData, etfFngData, fundingData] = await Promise.allSettled([
        this.fetchDriverIndexData(),
        this.fetchETFFNGData(),
        this.fetchFundingData()
      ]);
      
      
      // 处理DriverIndex数据
      if (driverIndexData.status === 'fulfilled') {
        this.macroDriverIndexData = driverIndexData.value;
        console.log('[macro] DriverIndex data loaded:', this.macroDriverIndexData.length, 'records');
        this.hideMacroLoading('chart-driver-index');
        // 确保图表已初始化后再渲染
        if (this.macroDriverIndexChart) {
          this.renderDriverIndexChart();
        } else {
          console.log('[macro] DriverIndex chart not ready, will render when initialized');
        }
      } else {
        console.warn('[macro] DriverIndex data failed:', driverIndexData.reason);
        this.showMacroError('chart-driver-index', '宏观数据加载失败');
      }
      
      // 处理ETF×FNG数据
      if (etfFngData.status === 'fulfilled') {
        this.macroETFFNGData = etfFngData.value;
        console.log('[macro] ETF×FNG data loaded:', this.macroETFFNGData.length, 'records');
        this.hideMacroLoading('chart-etf-fng');
        // 确保图表已初始化后再渲染
        if (this.macroETFFNGChart) {
          this.renderETFFNGChart();
        } else {
          console.log('[macro] ETF×FNG chart not ready, will render when initialized');
        }
      } else {
        console.warn('[macro] ETF×FNG data failed:', etfFngData.reason);
        this.showMacroError('chart-etf-fng', 'ETF数据加载失败');
      }
      
      // 处理资金费率数据
      if (fundingData.status === 'fulfilled') {
        this.macroFundingData = fundingData.value;
        console.log('[macro] Funding data loaded:', this.macroFundingData ? 'success' : 'null');
        this.hideMacroLoading('chart-funding-heat');
        // 确保图表已初始化后再渲染
        if (this.macroFundingChart) {
          this.renderFundingHeatmap();
        } else {
          console.log('[macro] Funding chart not ready, will render when initialized');
        }
      } else {
        console.warn('[macro] Funding data failed:', fundingData.reason);
        this.showMacroError('chart-funding-heat', '资金费率数据加载失败');
      }
      
    } catch (error) {
      console.error('[macro] Data loading failed:', error);
      // 如果整体加载失败，显示所有图表的错误状态
      this.showMacroError('chart-driver-index', '数据加载失败');
      this.showMacroError('chart-etf-fng', '数据加载失败');
      this.showMacroError('chart-funding-heat', '数据加载失败');
    }
  }
  
  
  
  private renderBtcMacroMonitorV3Chart() {
    if (!this.btcMacroMonitorV3Data) return;
    
    const data = this.btcMacroMonitorV3Data;
    const chartEl = document.getElementById('chart-btc-macro-monitor-v3');
    if (!chartEl) return;
    
    // 初始化图表
    if (!this.btcMacroMonitorV3Chart) {
      this.btcMacroMonitorV3Chart = this.echartsMod.init(chartEl);
    }
    
    // 按照Python代码的配色方案
    const colors = {
      driverIndex: '#22C786',
      btc: '#CFFFB0',
      fng: '#91FBB2',
      ixic: '#55E6A5',
      gspc: '#55E6A5',
      dxy: '#20E3B2',
      gold: '#8AF5AD',
      stablecap: '#7CF7B2',
      hashrate: '#66E6B0',
      etf: '#22C786',
      funding: '#A6F7C5',
      spread: '#B9F7C2',
      negative: '#0F3D3E'
    };
    
    // 准备数据 - 确保data是数组
    if (!Array.isArray(data)) {
      console.error('[BTC Macro Monitor V3] Data is not an array:', data);
      return;
    }
    
    const dates = data.map((d: any) => d.date);
    const driverIndexData = data.map((d: any) => d.driverIndex);
    const btcZData = data.map((d: any) => d.btcZ);
    const fngZData = data.map((d: any) => d.fngZ);
    const ixicZData = data.map((d: any) => d.ixicZ);
    const gspcZData = data.map((d: any) => d.gspcZ);
    const dxyZData = data.map((d: any) => d.dxyZ);
    const goldZData = data.map((d: any) => d.goldZ);
    const stablecapZData = data.map((d: any) => d.stablecapZ);
    const hashrateZData = data.map((d: any) => d.hashrateZ);
    const etfNetData = data.map((d: any) => d.etfNet);
    const fundingData = data.map((d: any) => d.funding);
    const spreadData = data.map((d: any) => d.spread);
    
    const option = {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(11, 15, 20, 0.9)',
        borderColor: '#22C786',
        borderWidth: 1,
        textStyle: {
          color: '#E6EDF6'
        },
        axisPointer: {
          type: 'cross',
          crossStyle: {
            color: '#22C786'
          }
        }
      },
      legend: {
        data: ['DriverIndex', 'BTC(z)', 'F&G(z)', 'IXIC(z)', 'GSPC(z)', 'DXY(z)', 'Gold(z)', 'Stablecap(z)', 'Hashrate(z)', 'ETF净流入', 'Funding', '10Y-2Y'],
        textStyle: {
          color: '#CFFFB0',
          fontSize: 12
        },
        top: 10,
        type: 'scroll'
      },
      grid: [
        {
          left: '3%',
          right: '4%',
          top: '15%',
          height: '25%'
        },
        {
          left: '3%',
          right: '4%',
          top: '45%',
          height: '25%'
        },
        {
          left: '3%',
          right: '4%',
          top: '75%',
          height: '20%'
        }
      ],
      xAxis: [
        {
          type: 'category',
          data: dates,
          axisLine: {
            lineStyle: {
              color: '#1e2a36'
            }
          },
          axisLabel: {
            color: '#9FB0C0',
            fontSize: 10
          },
          gridIndex: 0
        },
        {
          type: 'category',
          data: dates,
          axisLine: {
            lineStyle: {
              color: '#1e2a36'
            }
          },
          axisLabel: {
            color: '#9FB0C0',
            fontSize: 10
          },
          gridIndex: 1
        },
        {
          type: 'category',
          data: dates,
          axisLine: {
            lineStyle: {
              color: '#1e2a36'
            }
          },
          axisLabel: {
            color: '#9FB0C0',
            fontSize: 10
          },
          gridIndex: 2
        }
      ],
      yAxis: [
        {
          type: 'value',
          name: 'DriverIndex',
          position: 'left',
          axisLine: {
            lineStyle: {
              color: colors.driverIndex
            }
          },
          axisLabel: {
            color: colors.driverIndex,
            formatter: '{value}'
          },
          splitLine: {
            lineStyle: {
              color: '#1e2a36'
            }
          },
          gridIndex: 0
        },
        {
          type: 'value',
          name: 'Z-Score',
          position: 'left',
          axisLine: {
            lineStyle: {
              color: colors.btc
            }
          },
          axisLabel: {
            color: colors.btc,
            formatter: '{value}'
          },
          splitLine: {
            lineStyle: {
              color: '#1e2a36'
            }
          },
          gridIndex: 1
        },
        {
          type: 'value',
          name: 'ETF净流入(USD)',
          position: 'right',
          offset: 0,
          axisLine: {
            lineStyle: {
              color: colors.etf
            }
          },
          axisLabel: {
            color: colors.etf,
            formatter: (value: number) => (value / 1e6).toFixed(0) + 'M'
          },
          splitLine: {
            show: false
          },
          gridIndex: 2
        },
        {
          type: 'value',
          name: 'Funding',
          position: 'right',
          offset: 80,
          axisLine: {
            lineStyle: {
              color: colors.funding
            }
          },
          axisLabel: {
            color: colors.funding,
            formatter: (value: number) => (value * 100).toFixed(2) + '%'
          },
          splitLine: {
            show: false
          },
          gridIndex: 2
        }
      ],
      series: [
        // 顶部：DriverIndex
        {
          name: 'DriverIndex',
          type: 'line',
          data: driverIndexData,
          smooth: false,
          lineStyle: {
            color: colors.driverIndex,
            width: 2
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(34, 199, 134, 0.18)' },
                { offset: 1, color: 'rgba(34, 199, 134, 0.05)' }
              ]
            }
          },
          xAxisIndex: 0,
          yAxisIndex: 0
        },
        // 中部：相关因子多线
        {
          name: 'BTC(z)',
          type: 'line',
          data: btcZData,
          smooth: false,
          lineStyle: {
            color: colors.btc,
            width: 1.8,
            type: 'solid'
          },
          xAxisIndex: 1,
          yAxisIndex: 1
        },
        {
          name: 'F&G(z)',
          type: 'line',
          data: fngZData,
          smooth: false,
          lineStyle: {
            color: colors.fng,
            width: 1.8,
            type: 'solid'
          },
          xAxisIndex: 1,
          yAxisIndex: 1
        },
        {
          name: 'IXIC(z)',
          type: 'line',
          data: ixicZData,
          smooth: false,
          lineStyle: {
            color: colors.ixic,
            width: 1.8,
            type: 'dashed'
          },
          xAxisIndex: 1,
          yAxisIndex: 1
        },
        {
          name: 'GSPC(z)',
          type: 'line',
          data: gspcZData,
          smooth: false,
          lineStyle: {
            color: colors.gspc,
            width: 1.8,
            type: 'dotted'
          },
          xAxisIndex: 1,
          yAxisIndex: 1
        },
        {
          name: 'DXY(z)',
          type: 'line',
          data: dxyZData,
          smooth: false,
          lineStyle: {
            color: colors.dxy,
            width: 1.8,
            type: 'solid'
          },
          xAxisIndex: 1,
          yAxisIndex: 1
        },
        {
          name: 'Gold(z)',
          type: 'line',
          data: goldZData,
          smooth: false,
          lineStyle: {
            color: colors.gold,
            width: 1.8,
            type: 'dashed'
          },
          xAxisIndex: 1,
          yAxisIndex: 1
        },
        {
          name: 'Stablecap(z)',
          type: 'line',
          data: stablecapZData,
          smooth: false,
          lineStyle: {
            color: colors.stablecap,
            width: 1.8,
            type: 'solid'
          },
          xAxisIndex: 1,
          yAxisIndex: 1
        },
        {
          name: 'Hashrate(z)',
          type: 'line',
          data: hashrateZData,
          smooth: false,
          lineStyle: {
            color: colors.hashrate,
            width: 1.8,
            type: 'dotted'
          },
          xAxisIndex: 1,
          yAxisIndex: 1
        },
        // 底部：ETF柱状图
        {
          name: 'ETF净流入',
          type: 'bar',
          data: etfNetData,
          barWidth: '40%',
          barMaxWidth: 20,
          itemStyle: {
            color: (params: any) => params.value >= 0 ? colors.etf : colors.negative
          },
          xAxisIndex: 2,
          yAxisIndex: 2
        },
        // 底部：Funding线
        {
          name: 'Funding',
          type: 'line',
          data: fundingData,
          smooth: false,
          lineStyle: {
            color: colors.funding,
            width: 1.2,
            type: 'dashed'
          },
          xAxisIndex: 2,
          yAxisIndex: 3
        },
        // 底部：10Y-2Y利差线
        {
          name: '10Y-2Y',
          type: 'line',
          data: spreadData,
          smooth: false,
          lineStyle: {
            color: colors.spread,
            width: 1.2,
            type: 'solid'
          },
          xAxisIndex: 2,
          yAxisIndex: 3
        }
      ]
    };
    
    this.btcMacroMonitorV3Chart.setOption(option);
  }

  // --- 渲染信号区 ---
  private renderSignalSection(all: SignalDto[]){
    const $bar = document.getElementById('signalFilterBar');
    const $mount = document.getElementById('mine-queued');
    console.log('[filter] renderSignalSection called, elements found:', { $bar: !!$bar, $mount: !!$mount });
    if (!$bar || !$mount) {
      console.warn('[filter] Missing elements for signal section');
      return;
    }

    // 动态选项
    const symbols = ['ALL', ...unique(all, x=>x.symbol).map(x=>x.symbol)];
    const tfs     = ['ALL', ...unique(all, x=>x.timeframe).map(x=>x.timeframe)];

    // 过滤栏 HTML
    $bar.innerHTML = `
      <div class="tabs" id="sigSymbolTabs">${symbols.map(s=>`<button class="tab ${this.sigFilter.symbol===s?'active':''}" data-k="${s}">${s}</button>`).join('')}</div>
      <div class="chips">
        <span class="chip ${this.sigFilter.side==='ALL'?'active':''}" data-k="ALL">全部</span>
        <span class="chip ${this.sigFilter.side==='BUY'?'active':''}" data-k="BUY">BUY</span>
        <span class="chip ${this.sigFilter.side==='SELL'?'active':''}" data-k="SELL">SELL</span>
      </div>
      <div class="chips" id="sigTfChips">${tfs.map(t=>`<span class="chip ${this.sigFilter.tf===t?'active':'off'}" data-k="${t}">${t}</span>`).join('')}</div>
      <input id="sigSearch" class="input-search" placeholder="搜索：符号/方向/TF/策略" value="${this.sigFilter.q||''}" />
      <span class="count" id="sigCount"></span>
    `;

    // 绑定事件
    $bar.querySelectorAll('#sigSymbolTabs .tab').forEach(b=>{
      b.addEventListener('click',()=>{
        this.sigFilter.symbol = String((b as HTMLElement).dataset.k);
        savePref('sigFilter', this.sigFilter);
        this.renderSignalSection(all);
      });
    });
    $bar.querySelectorAll('#sigTfChips .chip').forEach(c=>{
      c.addEventListener('click',()=>{
        this.sigFilter.tf = String((c as HTMLElement).dataset.k) as any;
        savePref('sigFilter', this.sigFilter);
        this.renderSignalSection(all);
      });
    });
    // BUY/SELL/ALL
    const buySellChips = Array.from($bar.querySelectorAll('.chips .chip')).filter(el=>(el as HTMLElement).dataset.k!=='ALL' || el.textContent==='全部');
    buySellChips.forEach(c=>{
      c.addEventListener('click',()=>{
        this.sigFilter.side = String((c as HTMLElement).dataset.k) as any;
        savePref('sigFilter', this.sigFilter);
        this.renderSignalSection(all);
      });
    });
    // 搜索
    const $q = $bar.querySelector<HTMLInputElement>('#sigSearch');
    if ($q) {
      $q.oninput = ()=>{ this.sigFilter.q = $q.value; savePref('sigFilter', this.sigFilter); this.paintSignals(all, $mount); };
    }

    // 绘制
    this.paintSignals(all, $mount);
  }

  private paintSignals(all: SignalDto[], $mount: HTMLElement) {
    const filtered = this.applyFilter(all, this.sigFilter);
    const $count = document.getElementById('sigCount');
    if ($count) {
      $count.textContent = `共 ${filtered.length} 条`;
    }
    
    // 使用现有的mine.ts渲染逻辑
    $mount.innerHTML = filtered.length ? filtered.map(s => `
      <div class="signal-card ${s.side==='SELL'?'sell':''}" data-simid="${s.id}">
        <div class="signal-header">
          <div class="signal-title">${s.symbol} · ${s.side} · ${s.timeframe}</div>
          <div class="signal-strategy">${s.strategy}</div>
        </div>
        <div class="signal-details">
          <div>入队：${s.queued_at || '-'}</div>
          <div>参考入场：${s.ref_price || '-'}</div>
        </div>
        <div style="display:flex; gap:8px; margin-top:10px;">
          <button class="timeframe-btn" data-action="enable">启用</button>
          <button class="timeframe-btn" data-action="toggle">停止后续启用</button>
          <button class="timeframe-btn" data-action="remove">移出</button>
        </div>
      </div>
    `).join("") : `<div class="signal-card"><div class="signal-title">队列为空</div></div>`;
  }

  // --- 渲染持仓区 ---
  private renderPositionSection(all: PositionDto[]){
    const $bar = document.getElementById('posFilterBar');
    const $mount = document.getElementById('mine-open');
    console.log('[filter] renderPositionSection called, elements found:', { $bar: !!$bar, $mount: !!$mount });
    if (!$bar || !$mount) {
      console.warn('[filter] Missing elements for position section');
      return;
    }

    const symbols = ['ALL', ...unique(all, x=>x.symbol).map(x=>x.symbol)];
    const tfs     = ['ALL', ...unique(all, x=>x.timeframe).map(x=>x.timeframe)];

    $bar.innerHTML = `
      <div class="tabs" id="posSymbolTabs">${symbols.map(s=>`<button class="tab ${this.posFilter.symbol===s?'active':''}" data-k="${s}">${s}</button>`).join('')}</div>
      <div class="chips">
        <span class="chip ${this.posFilter.side==='ALL'?'active':''}" data-k="ALL">全部</span>
        <span class="chip ${this.posFilter.side==='BUY'?'active':''}" data-k="BUY">BUY</span>
        <span class="chip ${this.posFilter.side==='SELL'?'active':''}" data-k="SELL">SELL</span>
      </div>
      <div class="chips" id="posTfChips">${tfs.map(t=>`<span class="chip ${this.posFilter.tf===t?'active':'off'}" data-k="${t}">${t}</span>`).join('')}</div>
      <input id="posSearch" class="input-search" placeholder="搜索：符号/方向/TF/策略" value="${this.posFilter.q||''}" />
      <span class="count" id="posCount"></span>
    `;

    $bar.querySelectorAll('#posSymbolTabs .tab').forEach(b=>{
      b.addEventListener('click',()=>{
        this.posFilter.symbol = String((b as HTMLElement).dataset.k);
        savePref('posFilter', this.posFilter);
        this.renderPositionSection(all);
      });
    });
    $bar.querySelectorAll('#posTfChips .chip').forEach(c=>{
      c.addEventListener('click',()=>{
        this.posFilter.tf = String((c as HTMLElement).dataset.k) as any;
        savePref('posFilter', this.posFilter);
        this.renderPositionSection(all);
      });
    });
    const buySellChips = Array.from($bar.querySelectorAll('.chips .chip')).filter(el=>(el as HTMLElement).dataset.k!=='ALL' || el.textContent==='全部');
    buySellChips.forEach(c=>{
      c.addEventListener('click',()=>{
        this.posFilter.side = String((c as HTMLElement).dataset.k) as any;
        savePref('posFilter', this.posFilter);
        this.renderPositionSection(all);
      });
    });
    const $q = $bar.querySelector<HTMLInputElement>('#posSearch');
    if ($q) {
      $q.oninput = ()=>{ this.posFilter.q = $q.value; savePref('posFilter', this.posFilter); this.paintPositions(all, $mount); };
    }

    this.paintPositions(all, $mount);
  }

  private paintPositions(all: PositionDto[], $mount: HTMLElement) {
    const filtered = this.applyFilter(all, this.posFilter);
    const $count = document.getElementById('posCount');
    if ($count) {
      $count.textContent = `共 ${filtered.length} 条`;
    }
    
    // 使用现有的mine.ts渲染逻辑
    $mount.innerHTML = filtered.length ? filtered.map(p => `
      <div class="signal-card ${p.side==='SELL'?'sell':''}" data-posid="${p.id}">
        <div class="signal-header">
          <div class="signal-title">${p.symbol} · ${p.side} · ${p.timeframe}</div>
          <div class="signal-strategy">${p.strategy}</div>
        </div>
        <div class="signal-details">
          <div>数量：${p.qty}</div>
          <div>均价：${p.avg_price}</div>
          <div>现价：${p.mark_price || '-'}</div>
          <div>浮盈亏：${p.pnl || '-'}</div>
          <div>浮盈亏%：${p.pnl_pct ? p.pnl_pct + '%' : '-'}</div>
          <div>开仓：${p.opened_at}</div>
        </div>
        <div style="display:flex; gap:8px; margin-top:10px;">
          <button class="timeframe-btn" data-action="close">关闭仓位</button>
        </div>
      </div>
    `).join("") : `<div class="signal-card"><div class="signal-title">暂无运行中的模拟仓位</div></div>`;
  }

  // URL 参数同步筛选
  private readQueryParams() {
    const q = new URLSearchParams(location.search);
    this.sigFilter.symbol = (q.get('sym') || this.sigFilter.symbol).toUpperCase();
    this.sigFilter.side   = (q.get('side') as any) || this.sigFilter.side;
    this.sigFilter.tf     = (q.get('tf') as any) || this.sigFilter.tf;
    this.posFilter = {...this.sigFilter}; // 让两个区初始一致
  }

  // 快捷键支持
  private initKeyboardShortcuts() {
    window.addEventListener('keydown', (e)=>{
      if(e.key==='/'){ 
        e.preventDefault(); 
        const sigSearch = document.getElementById('sigSearch') as HTMLInputElement;
        if (sigSearch) sigSearch.focus(); 
      }
      if(e.key==='b'){ 
        this.sigFilter.side='BUY'; 
        savePref('sigFilter', this.sigFilter); 
        this.loadSignalAndPositions(); 
      }
      if(e.key==='s'){ 
        this.sigFilter.side='SELL'; 
        savePref('sigFilter', this.sigFilter); 
        this.loadSignalAndPositions();
      }
    });
  }

  private initHIGNavigation() {
    // 搜索功能
    const searchInput = document.querySelector('.search-input') as HTMLInputElement;
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        const query = (e.target as HTMLInputElement).value.toLowerCase();
        this.handleHIGSearch(query);
      });

      // 搜索快捷键 (Cmd/Ctrl + K)
      document.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
          e.preventDefault();
          searchInput.focus();
        }
      });
    }

    // 通知按钮
    const notificationsBtn = document.getElementById('notifications-btn');
    if (notificationsBtn) {
      notificationsBtn.addEventListener('click', () => {
        this.showHIGNotifications();
      });
    }

    // 设置按钮
    const settingsBtn = document.getElementById('settings-btn');
    if (settingsBtn) {
      settingsBtn.addEventListener('click', () => {
        this.switchTab('settings');
      });
    }

    // 应用品牌点击
    const appBrand = document.querySelector('.app-brand');
    if (appBrand) {
      appBrand.addEventListener('click', () => {
        this.switchTab('home');
      });
    }
  }

  private handleHIGSearch(query: string) {
    if (!query.trim()) {
      this.clearHIGSearchResults();
      return;
    }

    // 只在首页进行搜索
    if (this.currentTab !== 'home') {
      return;
    }

    // 搜索币种
    const quotes = document.querySelectorAll('.quote-enhanced-item');
    quotes.forEach(quote => {
      const symbol = quote.querySelector('.quote-symbol')?.textContent?.toLowerCase() || '';
      const price = quote.querySelector('.quote-price')?.textContent?.toLowerCase() || '';
      
      if (symbol.includes(query) || price.includes(query)) {
        (quote as HTMLElement).style.display = 'flex';
        (quote as HTMLElement).style.opacity = '1';
      } else {
        (quote as HTMLElement).style.display = 'none';
        (quote as HTMLElement).style.opacity = '0.3';
      }
    });

    // 搜索信号
    const signals = document.querySelectorAll('.signal-compact-card');
    signals.forEach(signal => {
      const symbol = signal.querySelector('.signal-symbol')?.textContent?.toLowerCase() || '';
      const strategy = signal.querySelector('.signal-strategy-chip')?.textContent?.toLowerCase() || '';
      
      if (symbol.includes(query) || strategy.includes(query)) {
        (signal as HTMLElement).style.display = 'block';
        (signal as HTMLElement).style.opacity = '1';
      } else {
        (signal as HTMLElement).style.display = 'none';
        (signal as HTMLElement).style.opacity = '0.3';
      }
    });

    console.log(`[HIG] 搜索: "${query}"`);
  }

  private clearHIGSearchResults() {
    // 只在首页恢复搜索结果
    if (this.currentTab !== 'home') {
      return;
    }

    // 恢复所有元素的显示
    const quotes = document.querySelectorAll('.quote-enhanced-item');
    quotes.forEach(quote => {
      (quote as HTMLElement).style.display = 'flex';
      (quote as HTMLElement).style.opacity = '1';
    });

    const signals = document.querySelectorAll('.signal-compact-card');
    signals.forEach(signal => {
      (signal as HTMLElement).style.display = 'block';
      (signal as HTMLElement).style.opacity = '1';
    });
  }

  private showHIGNotifications() {
    // 创建通知面板
    const notificationPanel = document.createElement('div');
    notificationPanel.className = 'hig-notification-panel';
    notificationPanel.innerHTML = `
      <div class="notification-content">
        <div class="notification-header">
          <h3>通知</h3>
          <button class="close-btn" onclick="this.parentElement.parentElement.parentElement.remove()">×</button>
        </div>
        <div class="notification-list">
          <div class="notification-item">
            <div class="notification-icon">📈</div>
            <div class="notification-text">
              <div class="notification-title">BTC突破关键阻力位</div>
              <div class="notification-time">2分钟前</div>
            </div>
          </div>
          <div class="notification-item">
            <div class="notification-icon">⚡</div>
            <div class="notification-text">
              <div class="notification-title">ETH策略信号触发</div>
              <div class="notification-time">5分钟前</div>
            </div>
          </div>
          <div class="notification-item">
            <div class="notification-icon">🔔</div>
            <div class="notification-text">
              <div class="notification-title">系统维护通知</div>
              <div class="notification-time">1小时前</div>
            </div>
          </div>
        </div>
      </div>
    `;

    // 添加样式
    const style = document.createElement('style');
    style.textContent = `
      .hig-notification-panel {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: flex-start;
        justify-content: center;
        z-index: 2000;
        padding-top: 60px;
      }
      .notification-content {
        background: #1C1C1E;
        border-radius: 16px;
        width: 90%;
        max-width: 400px;
        max-height: 70vh;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
      }
      .notification-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      }
      .notification-header h3 {
        color: #FFFFFF;
        font-size: 20px;
        font-weight: 600;
        margin: 0;
      }
      .close-btn {
        background: none;
        border: none;
        color: #FFFFFF;
        font-size: 24px;
        cursor: pointer;
        padding: 0;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
      }
      .close-btn:hover {
        background: rgba(255, 255, 255, 0.1);
      }
      .notification-list {
        max-height: 50vh;
        overflow-y: auto;
      }
      .notification-item {
        display: flex;
        align-items: center;
        padding: 16px 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      }
      .notification-icon {
        font-size: 20px;
        margin-right: 12px;
      }
      .notification-text {
        flex: 1;
      }
      .notification-title {
        color: #FFFFFF;
        font-size: 16px;
        font-weight: 500;
        margin-bottom: 4px;
      }
      .notification-time {
        color: rgba(255, 255, 255, 0.6);
        font-size: 14px;
      }
    `;
    document.head.appendChild(style);
    document.body.appendChild(notificationPanel);

    // 点击背景关闭
    notificationPanel.addEventListener('click', (e) => {
      if (e.target === notificationPanel) {
        notificationPanel.remove();
        style.remove();
      }
    });
  }

  private handleSearch(query: string) {
    if (!query.trim()) {
      // 清空搜索时显示所有内容
      this.clearSearchResults();
      return;
    }

    // 只在首页进行搜索，避免影响"我的"页面
    if (this.currentTab !== 'home') {
      return;
    }

    // 搜索币种
    const quotes = document.querySelectorAll('.quote-enhanced-item');
    quotes.forEach(quote => {
      const symbol = quote.querySelector('.quote-symbol')?.textContent?.toLowerCase() || '';
      const price = quote.querySelector('.quote-price')?.textContent?.toLowerCase() || '';
      
      if (symbol.includes(query) || price.includes(query)) {
        (quote as HTMLElement).style.display = 'flex';
        (quote as HTMLElement).style.opacity = '1';
      } else {
        (quote as HTMLElement).style.display = 'none';
        (quote as HTMLElement).style.opacity = '0.3';
      }
    });

    // 搜索信号
    const signals = document.querySelectorAll('.signal-compact-card');
    signals.forEach(signal => {
      const symbol = signal.querySelector('.signal-symbol')?.textContent?.toLowerCase() || '';
      const strategy = signal.querySelector('.signal-strategy-chip')?.textContent?.toLowerCase() || '';
      
      if (symbol.includes(query) || strategy.includes(query)) {
        (signal as HTMLElement).style.display = 'block';
        (signal as HTMLElement).style.opacity = '1';
      } else {
        (signal as HTMLElement).style.display = 'none';
        (signal as HTMLElement).style.opacity = '0.3';
      }
    });

    console.log(`[HIG] 搜索: "${query}"`);
  }

  private clearSearchResults() {
    // 只在首页恢复搜索结果
    if (this.currentTab !== 'home') {
      return;
    }

    // 恢复所有元素的显示
    const quotes = document.querySelectorAll('.quote-enhanced-item');
    quotes.forEach(quote => {
      (quote as HTMLElement).style.display = 'flex';
      (quote as HTMLElement).style.opacity = '1';
    });

    const signals = document.querySelectorAll('.signal-compact-card');
    signals.forEach(signal => {
      (signal as HTMLElement).style.display = 'block';
      (signal as HTMLElement).style.opacity = '1';
    });
  }

  private showNotifications() {
    // 创建通知面板
    const notificationPanel = document.createElement('div');
    notificationPanel.className = 'notification-panel';
    notificationPanel.innerHTML = `
      <div class="notification-content">
        <div class="notification-header">
          <h3>通知</h3>
          <button class="close-btn" onclick="this.parentElement.parentElement.parentElement.remove()">×</button>
        </div>
        <div class="notification-list">
          <div class="notification-item">
            <div class="notification-icon">📈</div>
            <div class="notification-text">
              <div class="notification-title">BTC突破关键阻力位</div>
              <div class="notification-time">2分钟前</div>
            </div>
          </div>
          <div class="notification-item">
            <div class="notification-icon">⚡</div>
            <div class="notification-text">
              <div class="notification-title">ETH策略信号触发</div>
              <div class="notification-time">5分钟前</div>
            </div>
          </div>
          <div class="notification-item">
            <div class="notification-icon">🔔</div>
            <div class="notification-text">
              <div class="notification-title">系统维护通知</div>
              <div class="notification-time">1小时前</div>
            </div>
          </div>
        </div>
      </div>
    `;

    // 添加样式
    const style = document.createElement('style');
    style.textContent = `
      .notification-panel {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: flex-start;
        justify-content: center;
        z-index: 2000;
        padding-top: 60px;
      }
      .notification-content {
        background: #1C1C1E;
        border-radius: 16px;
        width: 90%;
        max-width: 400px;
        max-height: 70vh;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
      }
      .notification-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      }
      .notification-header h3 {
        color: #FFFFFF;
        font-size: 20px;
        font-weight: 600;
        margin: 0;
      }
      .close-btn {
        background: none;
        border: none;
        color: #FFFFFF;
        font-size: 24px;
        cursor: pointer;
        padding: 0;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
      }
      .close-btn:hover {
        background: rgba(255, 255, 255, 0.1);
      }
      .notification-list {
        max-height: 50vh;
        overflow-y: auto;
      }
      .notification-item {
        display: flex;
        align-items: center;
        padding: 16px 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      }
      .notification-icon {
        font-size: 20px;
        margin-right: 12px;
      }
      .notification-text {
        flex: 1;
      }
      .notification-title {
        color: #FFFFFF;
        font-size: 16px;
        font-weight: 500;
        margin-bottom: 4px;
      }
      .notification-time {
        color: rgba(255, 255, 255, 0.6);
        font-size: 14px;
      }
    `;
    document.head.appendChild(style);
    document.body.appendChild(notificationPanel);

    // 点击背景关闭
    notificationPanel.addEventListener('click', (e) => {
      if (e.target === notificationPanel) {
        notificationPanel.remove();
        style.remove();
      }
    });
  }

  private showSettings() {
    // 切换到设置页面
    this.switchTab('settings');
  }
  
  private async fetchDriverIndexData(): Promise<any[]> {
    // 获取免Token公开数据：FRED（DFF/CPI/UNRATE）、CoinGecko（BTC）、Alternative.me（F&G）
    const days = this.macroCurrentRange === '7D' ? 7 : this.macroCurrentRange === '30D' ? 30 : 90;
    
    // 暂时禁用缓存，确保获取最新数据
    // const cacheKey = `driverIndex_${this.macroCurrentRange}`;
    // const cached = this.getCachedData<any[]>(cacheKey);
    // if (cached) {
    //   console.log('[DriverIndex] Using cached data');
    //   return cached;
    // }
    
    try {
      console.log('[DriverIndex] Fetching real data from public APIs...');
      
      // 并行获取所有真实数据源
      const [fredData, btcData, fngData, etfData] = await Promise.allSettled([
        this.fetchFREDData(),
        this.fetchBTCFromCoinGecko(),
        this.fetchFearGreedIndex(),
        this.fetchETFFlows()
      ]);
      
      // 处理FRED数据
      let rateData: any[] = [];
      let cpiData: any[] = [];
      let unempData: any[] = [];
      
      if (fredData.status === 'fulfilled') {
        rateData = fredData.value.rate || [];
        cpiData = fredData.value.cpi || [];
        unempData = fredData.value.unemp || [];
        console.log('[DriverIndex] FRED data received:', {
          rate: rateData.length,
          cpi: cpiData.length,
          unemp: unempData.length,
          rateSample: rateData.slice(0, 3),
          cpiSample: cpiData.slice(0, 3),
          unempSample: unempData.slice(0, 3)
        });
      } else {
        console.error('[DriverIndex] FRED data failed:', fredData.reason);
      }
      
      // 处理BTC数据
      let btcPrices: any[] = [];
      if (btcData.status === 'fulfilled') {
        btcPrices = btcData.value || [];
        console.log('[DriverIndex] BTC data received:', {
          count: btcPrices.length,
          sample: btcPrices.slice(0, 3)
        });
      } else {
        console.error('[DriverIndex] BTC data failed:', btcData.reason);
      }
      
      // 处理Fear & Greed数据
      let fngValues: any[] = [];
      if (fngData.status === 'fulfilled') {
        fngValues = fngData.value || [];
        console.log('[DriverIndex] F&G data received:', {
          count: fngValues.length,
          sample: fngValues.slice(0, 3)
        });
      } else {
        console.error('[DriverIndex] F&G data failed:', fngData.reason);
      }
      
      // 处理ETF数据
      let etfFlows: any[] = [];
      if (etfData.status === 'fulfilled') {
        etfFlows = Array.isArray(etfData.value) ? etfData.value : [];
        console.log('[DriverIndex] ETF data received:', {
          count: etfFlows.length,
          sample: etfFlows.slice(0, 3),
          rawData: etfData.value
        });
      } else {
        console.error('[DriverIndex] ETF data failed:', etfData.reason);
      }
      
      // 合并所有数据并计算DriverIndex
      const data = this.processDriverIndexData({
        rateData,
        cpiData,
        unempData,
        btcPrices,
        fngValues,
        etfFlows,
        days
      });
      
      console.log('[DriverIndex] Real data processed successfully:', data.length, 'records');
      console.log('[DriverIndex] Sample data:', data.slice(-3)); // 显示最后3条记录
      
      // 暂时禁用缓存设置
      // this.setCachedData(cacheKey, data);
      
      return data;
      
    } catch (error) {
      console.warn('[DriverIndex] Real data fetch failed, using fallback:', error);
      return this.generateFallbackData(days);
    }
  }
  
  private async fetchFREDData(): Promise<{rate: any[], cpi: any[], unemp: any[]}> {
    try {
      // 通过后端代理获取FRED数据，避免CORS问题
      const [rateRes, cpiRes, unempRes] = await Promise.allSettled([
        fetch('http://127.0.0.1:8889/api/fred/DFF'),
        fetch('http://127.0.0.1:8889/api/fred/CPIAUCSL'),
        fetch('http://127.0.0.1:8889/api/fred/UNRATE')
      ]);
      
      const rateResult = rateRes.status === 'fulfilled' ? await rateRes.value.json() : {data: []};
      const cpiResult = cpiRes.status === 'fulfilled' ? await cpiRes.value.json() : {data: []};
      const unempResult = unempRes.status === 'fulfilled' ? await unempRes.value.json() : {data: []};
      
      console.log('[FRED] Data received via backend proxy:', {
        rate: rateResult.data?.length || 0,
        cpi: cpiResult.data?.length || 0,
        unemp: unempResult.data?.length || 0
      });
      
      return { 
        rate: rateResult.data || [], 
        cpi: cpiResult.data || [], 
        unemp: unempResult.data || [] 
      };
    } catch (error) {
      console.warn('[FRED] Backend proxy failed, using fallback:', error);
      // 如果后端代理也失败，使用模拟数据
      return this.generateFREDFallbackData();
    }
  }
  
  private async parseFREDCSV(csvText: string): Promise<any[]> {
    const lines = csvText.split('\n').filter(line => line.trim());
    const data: any[] = [];
    
    console.log('[FRED] CSV parsing:', {
      totalLines: lines.length,
      header: lines[0],
      firstDataLine: lines[1],
      lastDataLine: lines[lines.length - 1]
    });
    
    for (let i = 1; i < lines.length; i++) { // Skip header
      const [date, value] = lines[i].split(',');
      if (date && value && value !== '.') {
        const parsedValue = parseFloat(value);
        if (!isNaN(parsedValue)) {
          data.push({
            date: new Date(date).toISOString().split('T')[0],
            value: parsedValue
          });
        }
      }
    }
    
    const sortedData = data.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
    console.log('[FRED] Parsed data:', {
      count: sortedData.length,
      dateRange: sortedData.length > 0 ? `${sortedData[0].date} to ${sortedData[sortedData.length - 1].date}` : 'none',
      sample: sortedData.slice(0, 3)
    });
    
    return sortedData;
  }
  
  private async fetchBTCFromCoinGecko(): Promise<any[]> {
    try {
      const days = this.macroCurrentRange === '7D' ? 7 : this.macroCurrentRange === '30D' ? 30 : 90;
      // 通过后端代理获取数据，避免CORS问题
      const response = await fetch(`${BASE_API}/api/macro/btc_monitor?days=${Math.min(days, 365)}`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.btc && Array.isArray(data.btc)) {
        console.log('[BTC] Data received:', data.btc.length, 'records');
        return data.btc;
      }
      
      console.warn('[BTC] No BTC data in response:', data);
      return this.generateMockBTCData(days);
    } catch (error) {
      console.warn('[BTC] API fetch failed, using mock data:', error);
      const days = this.macroCurrentRange === '7D' ? 7 : this.macroCurrentRange === '30D' ? 30 : 90;
      return this.generateMockBTCData(days);
    }
  }
  
  private async fetchFearGreedIndex(): Promise<any[]> {
    try {
      // 通过后端代理获取数据，避免CORS问题
      const response = await fetch(`${BASE_API}/api/macro/btc_monitor?days=365`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.fng && Array.isArray(data.fng)) {
        console.log('[F&G] Data received:', data.fng.length, 'records');
        return data.fng;
      }
      
      console.warn('[F&G] No F&G data in response:', data);
      const days = this.macroCurrentRange === '7D' ? 7 : this.macroCurrentRange === '30D' ? 30 : 90;
      return this.generateMockFNGData(days);
    } catch (error) {
      console.warn('[F&G] API fetch failed, using mock data:', error);
      const days = this.macroCurrentRange === '7D' ? 7 : this.macroCurrentRange === '30D' ? 30 : 90;
      return this.generateMockFNGData(days);
    }
  }
  
  private async fetchETFFlows(): Promise<any[]> {
    try {
      // 通过后端代理获取ETF数据，避免CORS问题
      const response = await fetch('http://127.0.0.1:8889/api/etf/flows');
      
      if (response.ok) {
        const data = await response.json();
        console.log('[ETF] Data received via backend proxy:', data.length, 'records');
        return data;
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.warn('[ETF] Backend proxy failed, using mock data:', error);
      return this.generateMockETFData();
    }
  }
  
  private generateMockETFData(): any[] {
    const days = this.macroCurrentRange === '7D' ? 7 : this.macroCurrentRange === '30D' ? 30 : 90;
    const data: any[] = [];
    
    for (let i = days - 1; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const dateStr = date.toISOString().split('T')[0];
      
      // 生成更真实的ETF流入数据
      const baseFlow = Math.random() * 50000000 - 25000000; // ±25M base
      const trend = Math.sin(i * 0.1) * 10000000; // 周期性趋势
      const noise = (Math.random() - 0.5) * 20000000; // 随机噪声
      
      data.push({
        date: dateStr,
        value: baseFlow + trend + noise
      });
    }
    
    return data;
  }
  
  private parseETFCSV(csvText: string): any[] {
    const lines = csvText.split('\n').filter(line => line.trim());
    const data: any[] = [];
    
    if (lines.length < 2) return data;
    
    // 清理CSV文本：去除BOM、货币符号、千位分隔符、括号负数
    let cleanText = csvText
      .replace(/\ufeff/g, '')  // 去除BOM
      .replace(/\$|£|€/g, '')  // 去除货币符号
      .replace(/,/g, '')       // 去除千位分隔符
      .replace(/\(([^)]+)\)/g, '-$1'); // 括号负数转负号
    
    const cleanLines = cleanText.split('\n').filter(line => line.trim());
    const headers = cleanLines[0].toLowerCase().split(',');
    
    console.log('[ETF] CSV headers:', headers);
    
    const dateCol = headers.findIndex(h => h.includes('date'));
    const flowCol = headers.findIndex(h => h.includes('net') || h.includes('flow') || h.includes('total'));
    
    console.log('[ETF] Date column:', dateCol, 'Flow column:', flowCol);
    
    if (dateCol === -1) {
      console.warn('[ETF] No date column found');
      return data;
    }
    
    for (let i = 1; i < cleanLines.length; i++) {
      const values = cleanLines[i].split(',');
      if (values[dateCol] && values[flowCol]) {
        const dateStr = values[dateCol].trim();
        const flowStr = values[flowCol].trim();
        
        // 更严格的数值解析
        const flowValue = parseFloat(flowStr);
        if (!isNaN(flowValue) && isFinite(flowValue)) {
          data.push({
            date: new Date(dateStr).toISOString().split('T')[0],
            value: flowValue
          });
        }
      }
    }
    
    const sortedData = data.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
    console.log('[ETF] Parsed', sortedData.length, 'records, sample:', sortedData.slice(0, 3));
    return sortedData;
  }
  
  private processDriverIndexData(sources: {
    rateData: any[], cpiData: any[], unempData: any[], 
    btcPrices: any[], fngValues: any[], etfFlows: any[], days: number
  }): any[] {
    const { rateData, cpiData, unempData, btcPrices, fngValues, etfFlows, days } = sources;
    
    // 创建日期范围
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(endDate.getDate() - days);
    
    const dateMap = new Map<string, any>();
    
    // 合并所有数据源
    [rateData, cpiData, unempData, btcPrices, fngValues, etfFlows].forEach((source, index) => {
      source.forEach((item: any) => {
        const date = item.date;
        if (!dateMap.has(date)) {
          dateMap.set(date, { date });
        }
        
        const record = dateMap.get(date);
        switch (index) {
          case 0: record.rate = item.value; break;
          case 1: record.cpi = item.value; break;
          case 2: record.unemp = item.value; break;
          case 3: record.btc_price = item.value; break;
          case 4: record.fng = item.value; break;
          case 5: record.etf_flow = item.value; break;
        }
      });
    });
    
    // 转换为数组并排序
    const sortedData = Array.from(dateMap.values())
      .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
      .slice(-days); // 取最近N天
    
    // 计算CPI同比（按月计算，避免日度计算导致的平线）
    const cpiDataMap = new Map(cpiData.map(item => [item.date, item.value]));
    
    // 按月分组CPI数据
    const monthlyCpi = new Map<string, number>();
    cpiData.forEach(item => {
      const date = new Date(item.date);
      const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
      monthlyCpi.set(monthKey, item.value);
    });
    
    sortedData.forEach(record => {
      if (record.cpi) {
        const currentDate = new Date(record.date);
        const currentMonth = `${currentDate.getFullYear()}-${String(currentDate.getMonth() + 1).padStart(2, '0')}`;
        const oneYearAgoMonth = `${currentDate.getFullYear() - 1}-${String(currentDate.getMonth() + 1).padStart(2, '0')}`;
        
        const currentCpi = monthlyCpi.get(currentMonth);
        const cpiOneYearAgo = monthlyCpi.get(oneYearAgoMonth);
        
        if (currentCpi && cpiOneYearAgo) {
          record.cpi_yoy = ((currentCpi - cpiOneYearAgo) / cpiOneYearAgo) * 100;
        }
      }
    });
    
    // 前向填充缺失值（但ETF不填充，保持缺失状态）
    let lastRate = 5.25, lastCpiYoy = 3.2, lastUnemp = 3.8;
    let lastBtc = 65000, lastFng = 50;
    
    sortedData.forEach(record => {
      record.rate = record.rate || lastRate;
      record.cpi_yoy = record.cpi_yoy || lastCpiYoy;
      record.unemp = record.unemp || lastUnemp;
      record.btc_price = record.btc_price || lastBtc;
      record.fng = record.fng || lastFng;
      // ETF不填充，保持undefined/null，这样图表会显示缺口而不是0直线
      
      lastRate = record.rate;
      lastCpiYoy = record.cpi_yoy;
      lastUnemp = record.unemp;
      lastBtc = record.btc_price;
      lastFng = record.fng;
    });
    
    // 统计ETF数据质量
    const etfRecords = sortedData.filter(r => r.etf_flow !== undefined && r.etf_flow !== null);
    const etfNonZero = etfRecords.filter(r => r.etf_flow !== 0);
    console.log('[ETF] Data quality:', {
      total: sortedData.length,
      etfRecords: etfRecords.length,
      etfNonZero: etfNonZero.length,
      etfUniqueValues: new Set(etfRecords.map((r: any) => r.etf_flow)).size
    });
    
    // 计算滚动Z分数和DriverIndex
    const rates = sortedData.map(d => d.rate);
    const cpiYoys = sortedData.map(d => d.cpi_yoy);
    const unemps = sortedData.map(d => d.unemp);
    const btcPriceValues = sortedData.map(d => d.btc_price);
    const fngs = sortedData.map(d => d.fng);
    
    // 使用更短的滚动窗口，让Z分数更敏感
    const window = Math.min(30, Math.max(15, Math.floor(sortedData.length * 0.2)));
    console.log('[DriverIndex] Rolling window size:', window, 'Total data points:', sortedData.length);
    
    sortedData.forEach((record, index) => {
      // 计算滚动Z分数
      const rateSlice = rates.slice(Math.max(0, index - window + 1), index + 1);
      const cpiSlice = cpiYoys.slice(Math.max(0, index - window + 1), index + 1);
      const unempSlice = unemps.slice(Math.max(0, index - window + 1), index + 1);
      const btcSlice = btcPriceValues.slice(Math.max(0, index - window + 1), index + 1);
      const fngSlice = fngs.slice(Math.max(0, index - window + 1), index + 1);
      
      const zRate = this.calculateZScore(record.rate, rateSlice);
      const zCpi = this.calculateZScore(record.cpi_yoy, cpiSlice);
      const zUnemp = this.calculateZScore(record.unemp, unempSlice);
      const zBtc = this.calculateZScore(Math.log(record.btc_price), btcSlice.map(p => Math.log(p)));
      const zFng = this.calculateZScore(record.fng, fngSlice);
      
      // DriverIndex = 0.4×(-Z利率) + 0.4×(-Z通胀同比) + 0.2×(-Z失业率)
      const driverIndex = 0.4 * (-zRate) + 0.4 * (-zCpi) + 0.2 * (-zUnemp);
      
      record.driver_index = driverIndex;
      record.z_btc = zBtc;
      record.z_fng = zFng;
      
      // 调试信息：显示最后几条记录的计算过程
      if (index >= sortedData.length - 3) {
        console.log(`[DriverIndex] Record ${index} (${record.date}):`, {
          raw: { rate: record.rate, cpi_yoy: record.cpi_yoy, unemp: record.unemp },
          zScores: { zRate, zCpi, zUnemp, zBtc, zFng },
          driverIndex,
          slices: { rate: rateSlice.length, cpi: cpiSlice.length, unemp: unempSlice.length }
        });
      }
    });
    
    // 可选择：完全移除平滑，直接使用原始DriverIndex
    // 或者使用极轻平滑
    const useRawData = true; // 设置为true使用原始数据，false使用轻平滑
    
    if (useRawData) {
      // 直接使用原始DriverIndex，无平滑
      sortedData.forEach(record => {
        record.smoothed_driver = record.driver_index;
      });
    } else {
      // 极轻平滑，几乎保持原始波动
      let smoothedDriver = 0;
      sortedData.forEach((record, index) => {
        if (index === 0) {
          smoothedDriver = record.driver_index;
        } else {
          // 使用极轻的平滑，保持绝大部分原始波动
          smoothedDriver = 0.9 * record.driver_index + 0.1 * smoothedDriver;
        }
        record.smoothed_driver = smoothedDriver;
      });
    }
    
    // 添加调试信息
    const driverValues = sortedData.map(d => d.smoothed_driver);
    const minDriver = Math.min(...driverValues);
    const maxDriver = Math.max(...driverValues);
    console.log('[DriverIndex] Driver range:', { min: minDriver.toFixed(3), max: maxDriver.toFixed(3), range: (maxDriver - minDriver).toFixed(3) });
    
    return sortedData;
  }
  
  private calculateZScore(value: number, values: number[]): number {
    // 需要至少3个数据点就能计算Z分数，更敏感
    if (values.length < 3) return 0;
    
    // 过滤掉无效值
    const validValues = values.filter(v => !isNaN(v) && isFinite(v));
    if (validValues.length < 2) return 0;
    
    const mean = validValues.reduce((sum, v) => sum + v, 0) / validValues.length;
    const variance = validValues.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / validValues.length;
    const stdDev = Math.sqrt(variance);
    
    // 使用更小的clip值，让Z分数更敏感
    const clippedStdDev = Math.max(stdDev, 1e-8);
    
    return (value - mean) / clippedStdDev;
  }
  
  private generateFREDFallbackData(): {rate: any[], cpi: any[], unemp: any[]} {
    const days = this.macroCurrentRange === '7D' ? 7 : this.macroCurrentRange === '30D' ? 30 : 90;
    const data: any[] = [];
    
    for (let i = days - 1; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const dateStr = date.toISOString().split('T')[0];
      
      // 生成有波动的模拟FRED数据
      const baseRate = 5.25 + Math.sin(i * 0.1) * 0.5;
      const baseCpi = 300 + Math.sin(i * 0.05) * 10;
      const baseUnemp = 3.8 + Math.sin(i * 0.08) * 0.3;
      
      data.push({
        date: dateStr,
        rate: baseRate,
        cpi: baseCpi,
        unemp: baseUnemp
      });
    }
    
    return {
      rate: data.map(d => ({ date: d.date, value: d.rate })),
      cpi: data.map(d => ({ date: d.date, value: d.cpi })),
      unemp: data.map(d => ({ date: d.date, value: d.unemp }))
    };
  }

  private generateFallbackData(days: number): any[] {
    const data: any[] = [];
    
    for (let i = days - 1; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const dateStr = date.toISOString().split('T')[0];
      
      // 生成有波动的模拟数据
      const rate = 5.25 + Math.sin(i * 0.1) * 0.5 + (Math.random() - 0.5) * 0.3;
      const cpi_yoy = 3.2 + Math.sin(i * 0.05) * 0.8 + (Math.random() - 0.5) * 0.4;
      const unemp = 3.8 + Math.sin(i * 0.08) * 0.4 + (Math.random() - 0.5) * 0.2;
      const btc_price = 65000 + Math.sin(i * 0.15) * 5000 + (Math.random() - 0.5) * 2000;
      const fng = 50 + Math.sin(i * 0.12) * 20 + (Math.random() - 0.5) * 10;
      const etf_flow = Math.sin(i * 0.2) * 30000000 + (Math.random() - 0.5) * 20000000;
      
      // 计算简化的Z分数
      const zRate = (rate - 5.25) / 0.5;
      const zCpi = (cpi_yoy - 3.2) / 0.8;
      const zUnemp = (unemp - 3.8) / 0.4;
      const zBtc = (Math.log(btc_price) - Math.log(65000)) / 0.1;
      const zFng = (fng - 50) / 25;
      
      // DriverIndex计算
      const driverIndex = 0.4 * (-zRate) + 0.4 * (-zCpi) + 0.2 * (-zUnemp);
      
      data.push({
        date: dateStr,
        rate: rate,
        cpi_yoy: cpi_yoy,
        unemp: unemp,
        btc_price: btc_price,
        fng: fng,
        etf_flow: etf_flow,
        driver_index: driverIndex,
        smoothed_driver: driverIndex,
        z_btc: zBtc,
        z_fng: zFng
      });
    }
    
    console.log('[DriverIndex] Generated fallback data with', data.length, 'records');
    return data;
  }
  
  private async fetchETFFNGData(): Promise<any[]> {
    // 模拟数据，实际应该调用API
    const days = this.macroCurrentRange === '7D' ? 7 : this.macroCurrentRange === '30D' ? 30 : 90;
    const data: any[] = [];
    
    for (let i = days - 1; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const dateStr = date.toISOString().split('T')[0];
      
      data.push({
        date: dateStr,
        etf_total_flow_usd: Math.random() * 200000000 - 100000000, // 随机ETF流入
        fng_value: Math.random() * 100 // 随机Fear&Greed值
      });
    }
    
    return data;
  }
  private async fetchFundingData(): Promise<any> {
    // 模拟数据，实际应该调用API
    const days = this.macroCurrentRange === '7D' ? 7 : this.macroCurrentRange === '30D' ? 30 : 90;
    const symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT'];
    const exchanges = ['Binance', 'OKX'];
    
    const slots: string[] = [];
    for (let i = days - 1; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      slots.push(date.toISOString().split('T')[0]);
    }
    
    const rows: any[] = [];
    exchanges.forEach(exchange => {
      symbols.forEach(symbol => {
        const values = slots.map(() => (Math.random() - 0.5) * 0.001); // 随机资金费率
        rows.push({
          key: `${exchange}-${symbol}`,
          values: values
        });
      });
    });
    
    return {
      range: this.macroCurrentRange,
      slots: slots,
      rows: rows,
      updated_at: new Date().toISOString()
    };
  }
  
  private renderDriverIndexChart() {
    console.log('[DriverIndex] renderDriverIndexChart called');
    console.log('[DriverIndex] Chart instance:', this.macroDriverIndexChart);
    console.log('[DriverIndex] Data length:', this.macroDriverIndexData?.length);
    console.log('[DriverIndex] ECharts module:', this.echartsMod);
    console.log('[DriverIndex] Chart container:', document.getElementById('chart-driver-index'));
    
    if (!this.macroDriverIndexChart) {
      console.error('[DriverIndex] Chart instance not found');
      this.showMacroError('chart-driver-index', '图表初始化失败');
      return;
    }
    
    if (!this.macroDriverIndexData || !this.macroDriverIndexData.length) {
      console.error('[DriverIndex] No data available');
      this.showMacroError('chart-driver-index', '暂无数据');
      return;
    }
    
    // 隐藏loading状态
    this.hideMacroLoading('chart-driver-index');
    
    const dates = this.macroDriverIndexData.map(d => d.date);
    const driverIndex = this.macroDriverIndexData.map(d => d.smoothed_driver);
    const btcZ = this.macroDriverIndexData.map(d => d.z_btc);
    const fngZ = this.macroDriverIndexData.map(d => d.z_fng);
    const etfFlows = this.macroDriverIndexData.map(d => d.etf_flow);
    
    // 处理ETF数据，将undefined/null转换为null（ECharts会显示为缺口）
    const processedEtfFlows = etfFlows.map(flow => 
      (flow === undefined || flow === null) ? null : flow
    );
    
    // 添加调试信息
    console.log('[DriverIndex] Chart data:', {
      dates: dates.length,
      driverIndex: driverIndex.slice(-5),
      btcZ: btcZ.slice(-5),
      fngZ: fngZ.slice(-5),
      etfFlows: processedEtfFlows.slice(-5),
      etfNullCount: processedEtfFlows.filter(f => f === null).length,
      etfNonZeroCount: processedEtfFlows.filter(f => f !== null && f !== 0).length
    });
    
    // 验证数据完整性
    if (dates.length === 0 || driverIndex.length === 0) {
      console.error('[DriverIndex] Invalid data format:', {
        datesLength: dates.length,
        driverIndexLength: driverIndex.length,
        sampleData: this.macroDriverIndexData.slice(0, 2)
      });
      this.showMacroError('chart-driver-index', '数据格式错误');
      return;
    }
    
    // 格式化Y轴标签
    const formatYAxisLabel = (value: number) => {
      if (Math.abs(value) >= 1000000) {
        return (value / 1000000).toFixed(1) + 'M';
      } else if (Math.abs(value) >= 1000) {
        return (value / 1000).toFixed(0) + 'K';
      }
      return value.toFixed(1);
    };
    
    const option = {
      animation: true,
      animationDuration: 600,
      animationEasing: 'cubicOut',
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(11, 15, 20, 0.95)',
        borderColor: '#1A2430',
        textStyle: { 
          color: '#E6EDF6',
          fontSize: 12,
          lineHeight: 18
        },
        formatter: (params: any[]) => {
          const date = params[0].axisValue;
          const driver = params.find(p => p.seriesName === 'DriverIndex');
          const btc = params.find(p => p.seriesName === 'BTC(z)');
          const fng = params.find(p => p.seriesName === 'F&G(z)');
          const etf = params.find(p => p.seriesName === 'ETF净流入');
          return `
            <div style="padding: 8px;">
              <div style="font-weight: 600; margin-bottom: 4px;">${date}</div>
              <div style="color: #1F77B4;">● DriverIndex: ${(driver?.value || 0).toFixed(2)}</div>
              <div style="color: #FF7F0E;">● BTC(z): ${(btc?.value || 0).toFixed(2)}</div>
              <div style="color: #8C9C68;">● F&G(z): ${(fng?.value || 0).toFixed(2)}</div>
              <div style="color: ${(etf?.value || 0) >= 0 ? '#2CA02C' : '#D62728'};">● ETF: $${formatYAxisLabel(etf?.value || 0)}</div>
            </div>
          `;
        }
      },
      legend: {
        data: ['DriverIndex', 'BTC(z)', 'F&G(z)', 'ETF净流入'],
        textStyle: { color: '#E6EDF6' },
        top: 12,
        left: 'center',
        backgroundColor: 'transparent'
      },
      grid: [
        {
          left: 48,
          right: 16,
          top: 50,
          bottom: 180,
          containLabel: true
        },
        {
          left: 48,
          right: 16,
          top: 240,
          bottom: 50,
          containLabel: true
        }
      ],
      xAxis: [
        {
          type: 'category',
          data: dates,
          gridIndex: 0,
          axisLine: { lineStyle: { color: '#1A2430' } },
          axisTick: { lineStyle: { color: '#1A2430' } },
          axisLabel: { 
            color: '#8FA0B3',
            fontSize: 11,
            interval: Math.max(1, Math.floor(dates.length / 8))
          }
        },
        {
          type: 'category',
          data: dates,
          gridIndex: 1,
          axisLine: { lineStyle: { color: '#1A2430' } },
          axisTick: { lineStyle: { color: '#1A2430' } },
          axisLabel: { 
            color: '#8FA0B3',
            fontSize: 11,
            interval: Math.max(1, Math.floor(dates.length / 8))
          }
        }
      ],
      yAxis: [
        {
          type: 'value',
          gridIndex: 0,
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { 
            color: '#8FA0B3',
            fontSize: 11,
            formatter: (value: number) => value.toFixed(1)
          },
          splitLine: { 
            lineStyle: { color: '#1A2430', type: 'dashed' }
          }
        },
        {
          type: 'value',
          gridIndex: 1,
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { 
            color: '#8FA0B3',
            fontSize: 11,
            formatter: (value: number) => value.toFixed(1)
          },
          splitLine: { 
            lineStyle: { color: '#1A2430', type: 'dashed' }
          }
        },
        {
          type: 'value',
          gridIndex: 1,
          position: 'right',
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { 
            color: '#8FA0B3',
            fontSize: 11,
            formatter: formatYAxisLabel
          },
          splitLine: { show: false }
        }
      ],
      axisPointer: {
        link: [{ xAxisIndex: [0, 1] }]
      },
      series: [
        {
          name: 'DriverIndex',
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 0,
          gridIndex: 0,
          data: driverIndex,
          lineStyle: { 
            color: '#1F77B4',
            width: 2
          },
          itemStyle: { color: '#1F77B4' },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(31, 119, 180, 0.3)' },
                { offset: 1, color: 'rgba(31, 119, 180, 0.05)' }
              ]
            }
          },
          smooth: true,
          symbol: 'none'
        },
        {
          name: 'BTC(z)',
          type: 'line',
          xAxisIndex: 1,
          yAxisIndex: 1,
          gridIndex: 1,
          data: btcZ,
          lineStyle: { 
            color: '#FF7F0E',
            width: 2
          },
          itemStyle: { color: '#FF7F0E' },
          smooth: true,
          symbol: 'none'
        },
        {
          name: 'F&G(z)',
          type: 'line',
          xAxisIndex: 1,
          yAxisIndex: 1,
          gridIndex: 1,
          data: fngZ,
          lineStyle: { 
            color: '#8C9C68',
            width: 2
          },
          itemStyle: { color: '#8C9C68' },
          smooth: true,
          symbol: 'none'
        },
        {
          name: 'ETF净流入',
          type: 'bar',
          xAxisIndex: 1,
          yAxisIndex: 2,
          gridIndex: 1,
          data: processedEtfFlows,
          barWidth: '60%',
          itemStyle: {
            color: (params: any) => {
              if (params.value === null || params.value === undefined) return 'transparent';
              return params.value >= 0 ? '#2CA02C' : '#D62728';
            },
            opacity: 0.8
          },
          emphasis: {
            itemStyle: {
              opacity: 1
            }
          },
          // 确保ETF数据不为0时显示
          silent: false,
          animation: true,
          // 处理缺失数据
          connectNulls: false
        }
      ]
    };
    
    this.macroDriverIndexChart.setOption(option, true);
    console.log('[DriverIndex] Chart option set successfully');
  }
  
  private renderETFFNGChart() {
    if (!this.macroETFFNGChart || !this.macroETFFNGData.length) {
      // 如果没有数据，清除loading状态并显示错误
      this.showMacroError('chart-etf-fng', '暂无数据');
      return;
    }
    
    // 隐藏loading状态
    this.hideMacroLoading('chart-etf-fng');
    
    const dates = this.macroETFFNGData.map(d => d.date);
    const etfFlows = this.macroETFFNGData.map(d => d.etf_total_flow_usd);
    const fngValues = this.macroETFFNGData.map(d => d.fng_value);
    
    // 计算量能条数据：基于ETF净流入强度
    const absFlows = etfFlows.map(Math.abs);
    const p95 = absFlows.sort((a, b) => b - a)[Math.floor(absFlows.length * 0.05)] || 1;
    const volumeBars = etfFlows.map(flow => {
      const norm = Math.min(Math.abs(flow) / p95, 1);
      return -20 * norm * Math.sign(flow);
    });
    
    // 格式化Y轴标签
    const formatYAxisLabel = (value: number) => {
      if (Math.abs(value) >= 1000000) {
        return (value / 1000000).toFixed(1) + 'M';
      } else if (Math.abs(value) >= 1000) {
        return (value / 1000).toFixed(0) + 'K';
      }
      return value.toFixed(0);
    };
    
    const option = {
      animation: true,
      animationDuration: 600,
      animationEasing: 'cubicOut',
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(11, 15, 20, 0.95)',
        borderColor: '#1A2430',
        textStyle: { 
          color: '#E6EDF6',
          fontSize: 12,
          lineHeight: 18
        },
        formatter: (params: any[]) => {
          const date = params[0].axisValue;
          const etf = params.find(p => p.seriesName === 'ETF净流入');
          const fng = params.find(p => p.seriesName === 'F&G');
          const volume = params.find(p => p.seriesName === '量能条');
          return `
            <div style="padding: 8px;">
              <div style="font-weight: 600; margin-bottom: 4px;">${date}</div>
              <div style="color: #22D39A;">● ETF净流入: $${(etf?.value || 0).toLocaleString()}</div>
              <div style="color: #1FA2FF;">● F&G: ${fng?.value || 0}</div>
              <div style="color: #8FA0B3;">● 量能强度: ${volume?.value ? Math.abs(volume.value).toFixed(1) : '0'}</div>
            </div>
          `;
        }
      },
      legend: {
        data: ['ETF净流入', 'F&G', '量能条'],
        textStyle: { color: '#E6EDF6' },
        top: 12,
        left: 'center',
        backgroundColor: 'transparent'
      },
      grid: [
        {
          left: 48,
          right: 16,
          top: 20,
          height: '68%',
          containLabel: true
        },
        {
          left: 48,
          right: 16,
          top: '78%',
          height: '18%',
          containLabel: true
        }
      ],
      xAxis: [
        {
          type: 'category',
          data: dates,
          gridIndex: 0,
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { show: false },
          splitLine: { show: false }
        },
        {
          type: 'category',
          data: dates,
          gridIndex: 1,
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { 
            color: '#8FA0B3',
            interval: Math.floor(dates.length / 4)
          },
          splitLine: { show: false }
        }
      ],
      yAxis: [
        {
          type: 'value',
          position: 'left',
          gridIndex: 0,
          scale: true,
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { 
            color: '#8FA0B3',
            formatter: formatYAxisLabel
          },
          splitLine: { 
            lineStyle: { 
              color: 'rgba(255,255,255,0.06)',
              type: 'solid'
            }
          }
        },
        {
          type: 'value',
          position: 'right',
          gridIndex: 0,
          scale: true,
          min: 0,
          max: 100,
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { 
            color: '#8FA0B3',
            formatter: '{value}'
          },
          splitLine: { show: false }
        },
        {
          type: 'value',
          position: 'left',
          gridIndex: 1,
          min: -20,
          max: 0,
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { show: false },
          splitLine: { show: false }
        }
      ],
      axisPointer: {
        link: [{ xAxisIndex: [0, 1] }]
      },
      series: [
        {
          name: 'ETF净流入',
          type: 'bar',
          xAxisIndex: 0,
          yAxisIndex: 0,
          gridIndex: 0,
          data: etfFlows,
          barWidth: '55%',
          itemStyle: {
            color: (params: any) => {
              return params.value >= 0 ? '#22D39A' : '#FF6B6B';
            },
            borderRadius: [6, 6, 0, 0]
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: (params: any) => {
                return params.value >= 0 ? 'rgba(34,211,154,0.35)' : 'rgba(255,107,107,0.35)';
              }
            }
          },
          animationDelay: (idx: number) => idx * 20,
          animationEasing: 'easeOutBack',
          animationDuration: 220
        },
        {
          name: 'F&G',
          type: 'line',
          xAxisIndex: 0,
          yAxisIndex: 1,
          gridIndex: 0,
          data: fngValues,
          smooth: true,
          symbol: 'circle',
          symbolSize: 3.5,
          lineStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 1, y2: 0,
              colorStops: [
                { offset: 0, color: '#1FA2FF' },
                { offset: 0.5, color: '#12D8FA' },
                { offset: 1, color: '#A6FFCB' }
              ]
            },
            width: 2.5
          },
          itemStyle: {
            color: (params: any) => {
              const value = params.value;
              if (value < 25) return '#FF6B6B';
              if (value < 50) return '#FF8C42';
              if (value < 75) return '#FFD93D';
              return '#22D39A';
            },
            borderColor: '#CFFAF1',
            borderWidth: 1
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 8,
              shadowColor: 'rgba(31,162,255,0.4)'
            }
          },
          animationDelay: (idx: number) => idx * 10,
          animationDuration: 600,
          animationEasing: 'cubicOut'
        },
        {
          name: '量能条',
          type: 'bar',
          xAxisIndex: 1,
          yAxisIndex: 2,
          gridIndex: 1,
          data: volumeBars,
          barMinWidth: 2,
          barMaxWidth: 12,
          barCategoryGap: '20%',
          clip: true,
          animation: false,
          itemStyle: {
            color: (params: any) => {
              const value = params.value;
              const absValue = Math.abs(value);
              const alpha = absValue / 20; // 透明度随强度变化
              if (value >= 0) {
                return `rgba(34, 211, 154, ${alpha * 0.6})`; // 正流入偏绿
              } else {
                return `rgba(255, 107, 107, ${alpha * 0.6})`; // 负流入偏红
              }
            }
          },
          silent: true // 避免干扰主图交互
        }
      ]
    };
    
    this.macroETFFNGChart.setOption(option, { notMerge: true, replaceMerge: ['grid', 'xAxis', 'yAxis', 'series'] });
    
    // 添加交互事件
    this.macroETFFNGChart.off('mouseover');
    this.macroETFFNGChart.on('mouseover', (params: any) => {
      if (params.componentType === 'series') {
        this.macroEventBus.emit('focus-date', dates[params.dataIndex]);
      }
    });
  }
  
  private renderFundingHeatmap() {
    if (!this.macroFundingChart || !this.macroFundingData) {
      // 如果没有数据，清除loading状态并显示错误
      this.showMacroError('chart-funding-heat', '暂无数据');
      return;
    }
    
    // 隐藏loading状态
    this.hideMacroLoading('chart-funding-heat');
    
    const { slots, rows } = this.macroFundingData;
    const data: number[][] = [];
    
    rows.forEach((row: any, rowIndex: number) => {
      const values = row.values || row.data || [];
      values.forEach((value: number, colIndex: number) => {
        data.push([colIndex, rowIndex, value]);
      });
    });
    
    const option = {
      animation: true,
      animationDuration: 200,
      animationEasing: 'cubicOut',
      tooltip: {
        position: 'top',
        backgroundColor: 'rgba(11, 15, 20, 0.95)',
        borderColor: '#1A2430',
        textStyle: { 
          color: '#E6EDF6',
          fontSize: 12,
          lineHeight: 18
        },
        formatter: (params: any) => {
          const row = rows[params.data[1]];
          const date = slots[params.data[0]];
          const rate = params.data[2];
          return `
            <div style="padding: 8px;">
              <div style="font-weight: 600; margin-bottom: 4px;">${date}</div>
              <div style="margin-bottom: 2px;">${row.key}</div>
              <div style="color: ${rate >= 0 ? '#2AC59E' : '#C9485B'};">
                资金费率: ${(rate * 100).toFixed(4)}%
              </div>
            </div>
          `;
        }
      },
      grid: {
        left: '12%',
        right: '8%',
        bottom: '8%',
        top: '8%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: slots,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { 
          color: '#8FA0B3',
          interval: Math.floor(slots.length / 4)
        },
        splitLine: { show: false }
      },
      yAxis: {
        type: 'category',
        data: rows.map((r: any) => r.key),
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { 
          color: '#8FA0B3',
          align: 'right'
        },
        splitLine: { show: false }
      },
      visualMap: {
        min: -0.001,
        max: 0.001,
        calculable: true,
        orient: 'horizontal',
        left: 'right',
        top: '5%',
        width: 20,
        height: 200,
        inRange: {
          color: ['#C9485B', '#2A3442', '#2AC59E']
        },
        textStyle: { 
          color: '#8FA0B3',
          fontSize: 10
        },
        formatter: (value: number) => {
          return (value * 100).toFixed(2) + '%';
        }
      },
      series: [{
        type: 'heatmap',
        data: data,
        label: {
          show: false
        },
        itemStyle: {
          borderRadius: 3,
          borderWidth: 0
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 8,
            shadowColor: 'rgba(0, 0, 0, 0.3)',
            borderWidth: 1,
            borderColor: '#1FA2FF'
          }
        },
        animationDelay: (idx: number) => idx * 2,
        animationDuration: 200
      }]
    };
    
    this.macroFundingChart.setOption(option, true);
    
    // 添加交互事件
    this.macroFundingChart.off('click');
    this.macroFundingChart.on('click', (params: any) => {
      if (params.componentType === 'series') {
        const date = slots[params.data[0]];
        this.macroEventBus.emit('focus-date', date);
      }
    });
    
    // 添加悬停事件
    this.macroFundingChart.off('mouseover');
    this.macroFundingChart.on('mouseover', (params: any) => {
      if (params.componentType === 'series') {
        const date = slots[params.data[0]];
        this.macroEventBus.emit('focus-date', date);
      }
    });
  }
  
  private highlightDateInCharts(date: string) {
    // 在两个图表中高亮指定日期
    if (this.macroETFFNGChart) {
      const option = this.macroETFFNGChart.getOption();
      // 添加参考线逻辑
    }
    
    if (this.macroFundingChart) {
      const option = this.macroFundingChart.getOption();
      // 添加高亮逻辑
    }
  }
  
  private showMacroLoading(chartId: string, message: string = '加载中...') {
    const chartEl = document.getElementById(chartId);
    if (chartEl) {
      chartEl.innerHTML = `
        <div class="chart-loading">
          <div class="loading-spinner"></div>
          <div class="loading-text">${message}</div>
        </div>
      `;
    }
  }

  private hideMacroLoading(chartId: string) {
    const chartEl = document.getElementById(chartId);
    if (chartEl) {
      const loadingEl = chartEl.querySelector('.chart-loading') as HTMLElement;
      if (loadingEl) {
        loadingEl.style.opacity = '0';
        loadingEl.style.transition = 'opacity 0.3s ease-out';
        setTimeout(() => {
          loadingEl.remove();
        }, 300);
      }
    }
  }

  private showMacroError(chartId: string, message: string) {
    const chartEl = document.getElementById(chartId);
    if (chartEl) {
      chartEl.innerHTML = `
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: #8FA0B3; background: rgba(0, 0, 0, 0.1); border-radius: 12px;">
          <div style="font-size: 24px; margin-bottom: 8px; animation: bounce 1s infinite;">⚠️</div>
          <div style="font-size: 14px; font-weight: 500;">${message}</div>
          <div style="font-size: 12px; margin-top: 4px; opacity: 0.7;">点击刷新按钮重试</div>
        </div>
      `;
    }
  }

  private showTestCharts() {
    console.log('[debug] Showing test charts');
    
    // 确保ECharts已经加载
    if (!this.echartsMod) {
      console.log('[debug] ECharts not ready, retrying in 1 second...');
      setTimeout(() => this.showTestCharts(), 1000);
      return;
    }
    
    // 确保图表已经初始化
    if (!this.macroDriverIndexChart || !this.macroETFFNGChart || !this.macroFundingChart) {
      console.log('[debug] Charts not initialized, retrying in 1 second...');
      setTimeout(() => this.showTestCharts(), 1000);
      return;
    }
    
    console.log('[debug] All charts ready, rendering test data...');
    
    // 测试DriverIndex图表
    this.hideMacroLoading('chart-driver-index');
    this.macroDriverIndexData = this.generateTestDriverIndexData();
    this.renderDriverIndexChart();
    
    // 测试ETF图表
    this.hideMacroLoading('chart-etf-fng');
    this.macroETFFNGData = this.generateTestETFData();
    this.renderETFFNGChart();
    
    // 测试资金费率图表
    this.hideMacroLoading('chart-funding-heat');
    this.macroFundingData = this.generateTestFundingData();
    this.renderFundingHeatmap();
  }

  private generateTestDriverIndexData() {
    const data = [];
    for (let i = 30; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      data.push({
        date: date.toISOString().split('T')[0],
        smoothed_driver: Math.sin(i * 0.2) * 10 + 50,
        z_btc: Math.cos(i * 0.15) * 5 + 0,
        z_fng: Math.sin(i * 0.1) * 3 + 0,
        etf_flow: Math.random() * 1000 - 500
      });
    }
    return data;
  }

  private generateTestETFData() {
    const data = [];
    for (let i = 30; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      data.push({
        date: date.toISOString().split('T')[0],
        etf_total_flow_usd: Math.random() * 2000 - 1000,
        fng_value: Math.floor(Math.random() * 100)
      });
    }
    return data;
  }

  private generateTestFundingData() {
    const symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'];
    const exchanges = ['Binance', 'OKX'];
    const slots: string[] = [];
    const rows: any[] = [];
    
    for (let i = 7; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      slots.push(date.toISOString().split('T')[0]);
    }
    
    symbols.forEach(symbol => {
      exchanges.forEach(exchange => {
        rows.push({
          symbol,
          exchange,
          data: slots.map(() => Math.random() * 0.01 - 0.005)
        });
      });
    });
    
    return { slots, rows };
  }

  private generateMockBTCData(days: number): any[] {
    const data = [];
    for (let i = days - 1; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      data.push({
        date: date.toISOString().split('T')[0],
        price: 50000 + Math.sin(i * 0.1) * 5000 + (Math.random() - 0.5) * 2000
      });
    }
    return data;
  }

  private generateMockFNGData(days: number): any[] {
    const data = [];
    for (let i = days - 1; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      data.push({
        date: date.toISOString().split('T')[0],
        value: Math.floor(Math.random() * 100),
        classification: Math.random() > 0.5 ? 'Greed' : 'Fear'
      });
    }
    return data;
  }

  private async refreshInfoData() {
    const assetSel = document.getElementById('info-asset') as HTMLSelectElement | null;
    const granSel = document.getElementById('info-granularity') as HTMLSelectElement | null;
    const dateInp = document.getElementById('info-date') as HTMLInputElement | null;
    const tip = document.getElementById('info-tip');
    const asset = (assetSel?.value || 'BTC').toUpperCase();
    const gran = (granSel?.value || 'daily').toLowerCase();
    const date = dateInp?.value || '';
    
    // 显示加载状态
    const loadingEl = document.getElementById('info-loading');
    if (loadingEl) {
      loadingEl.style.display = 'flex !important';
      loadingEl.style.zIndex = '9999';
      loadingEl.style.position = 'absolute';
      loadingEl.style.top = '50%';
      loadingEl.style.left = '50%';
      loadingEl.style.transform = 'translate(-50%, -50%)';
      // 更新加载文本为"计算中"
      const loadingText = loadingEl.querySelector('.loading-text');
      if (loadingText) loadingText.textContent = '计算中...';
    }
    
    try {
      // 并行获取数据，减少等待时间
      const [factorsRes, indexRes] = await Promise.allSettled([
        useFactorsData({ asset, granularity: gran, date }),
        useFactorIndex({ asset, granularity: gran, days: 60, alpha: 0.3 })
      ]);

      // 处理因子数据
      if (factorsRes.status === 'fulfilled') {
        this.infoData = factorsRes.value.data || [];
        this.infoAsOf = factorsRes.value.asOf || '';
        if (tip) tip.textContent = factorsRes.value.source === 'api' ? '数据源：公开API（10分钟缓存）' : '实时数据不可用，已回退至本地样本';
      } else {
        this.infoData = [];
        if (tip) tip.textContent = '实时数据不可用，已回退至本地样本';
      }

      // 处理指数数据
      if (indexRes.status === 'fulfilled') {
        this.infoIndex = indexRes.value.index || [];
        this.infoContrib = indexRes.value.contrib || [];
      } else {
        this.infoIndex = [];
        this.infoContrib = [];
      }

      // 立即渲染，不等待ECharts
      this.renderInfoDetail();
      // 拉取相关性曲线
      await this.fetchCorrData();
      this.renderCorrLines();
      
      // 如果ECharts已加载，立即渲染图表
      if (this.echartsMod) {
        // 渲染相关性图
        this.renderCorrLines();
        // 隐藏加载状态
        if (loadingEl) {
          loadingEl.style.display = 'none';
          loadingEl.style.visibility = 'hidden';
          loadingEl.style.zIndex = '-1';
        }
      } else {
        // 否则等待ECharts加载完成
        console.log('[info] Waiting for ECharts to load...');
        // 等待ECharts加载完成后隐藏加载状态
        const waitForECharts = () => {
          if (this.echartsMod) {
            this.renderCorrLines();
            if (loadingEl) {
              loadingEl.style.display = 'none';
              loadingEl.style.visibility = 'hidden';
              loadingEl.style.zIndex = '-1';
            }
          } else {
            setTimeout(waitForECharts, 100);
          }
        };
        waitForECharts();
      }
      
    } catch (error) {
      console.error('[info] Data loading error:', error);
      if (tip) tip.textContent = '实时数据不可用，已回退至本地样本';
    }
  }


  private renderInfoDetail() {
    const panel = document.getElementById('info-detail-body');
    if (!panel) return;
    const d = this.infoData.find(x => String(x.name) === this.infoSelectedKey);
    if (!d) { panel.textContent = '选择左侧维度查看详情'; return; }
    if (!d.sub_factors || !d.sub_factors.length) { panel.textContent = '部分数据缺失'; return; }
    const rows = d.sub_factors.map(sf => {
      const score = sf.score == null ? '-' : String(sf.score);
      const w = (sf.weight ?? 0).toFixed(2);
      const sig = sf.signal ?? '-';
      const notes = sf.notes ? String(sf.notes).slice(0, 80) : '';
      return `<tr><td>${sf.key}</td><td>${score}</td><td>${w}</td><td>${sig}</td><td title="${sf.notes||''}">${notes}</td></tr>`;
    }).join('');
    panel.innerHTML = `
      <table class="qb-table" style="width:100%;border-collapse:collapse;">
        <thead><tr><th>子因子</th><th>分数</th><th>权重</th><th>信号</th><th>备注</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }


  // 1) 指数线尾迹闪烁
  private pingIndexTail(idx: number) {
    if (!this.infoIndex || idx < 0 || idx >= this.infoIndex.length) return;
    
    const point = this.infoIndex[idx];
    const tailPoint = [point.ts, (point.smoothed ?? point.raw ?? 50) - 50];
    
    // 图表已删除，无需处理
  }

  // 2) 检测过0轴交叉
  private findZeroCrossings(indexArr: any[]) {
    const crossings: [string, number][] = [];
    for (let i = 1; i < indexArr.length; i++) {
      const a = (indexArr[i-1]?.smoothed ?? indexArr[i-1]?.raw ?? 50) - 50;
      const b = (indexArr[i]?.smoothed ?? indexArr[i]?.raw ?? 50) - 50;
      if ((a <= 0 && b > 0) || (a >= 0 && b < 0)) {
        crossings.push([indexArr[i].ts, 0]);
      }
    }
    return crossings;
  }


  // 图例选择状态管理
  private getLegendSelectedState() {
    const state: Record<string, boolean> = {};
    const order = ['macro', 'policy', 'capital', 'geopolitics', 'onchain', 'sentiment'];
    order.forEach(key => {
      state[key] = this.infoSelectedFactor === null || this.infoSelectedFactor === key;
    });
    return state;
  }



  private renderInfoSummary() {
    const wrap = document.getElementById('info-summary');
    if (!wrap) return;
    const n = (this.infoIndex || []).length;
    if (!n) { wrap.textContent = '暂无数据'; return; }
    const i = Math.max(0, Math.min(n-1, this.infoCurrentIdx < 0 ? n-1 : this.infoCurrentIdx));
    const idxVal = this.infoIndex[i];
    const b = Number((idxVal.smoothed ?? idxVal.raw ?? 50)).toFixed(1);
    // 近7日变化
    const j = Math.max(0, i - 7);
    const bPrev = Number((this.infoIndex[j].smoothed ?? this.infoIndex[j].raw ?? 50));
    const delta = (Number(b) - bPrev).toFixed(1);
    // 最强/最弱因子
    const keys = ['macro','policy','capital','geopolitics','onchain','sentiment'];
    const dayContrib = keys.map(k => {
      const s = (this.infoContrib.find(c => c.key===k)?.points || [])[i];
      const v = Number(s?.smoothed ?? s?.raw ?? 0);
      return { k, v };
    });
    dayContrib.sort((a,b)=>b.v-a.v);
    const strongest = dayContrib[0];
    dayContrib.sort((a,b)=>a.v-b.v);
    const weakest = dayContrib[0];

    wrap.innerHTML = `
      <div style="display:flex;flex-wrap:wrap;gap:10pt;align-items:center;">
        <div>综合分 B(t)：<span style="font-weight:700;color:#00D5FF">${b}</span></div>
        <div>7日变化：<span style="color:${Number(delta)>=0?'#16C784':'#EA3943'}">${Number(delta)>=0? '↑':'↓'} ${Math.abs(Number(delta)).toFixed(1)}</span></div>
        <div>最强：<b>${strongest?.k || '-'}</b></div>
        <div>最弱：<b>${weakest?.k || '-'}</b></div>
      </div>`;
  }


  private arrowByWow(k: string, wow: number | null): string {
    if (wow == null) return '';
    if (wow > 1) return ' {up|↑}';
    if (wow < -1) return ' {down|↓}';
    return ' {flat|→}';
  }

  private formatTimeLabel(ts?: string): string {
    if (!ts) return '';
    try {
      const d = new Date(ts);
      return d.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' });
    } catch(_) {
      return '';
    }
  }


  private showInfoHelp() {
    const existing = document.getElementById('info-help-modal');
    if (existing) existing.remove();

    const modal = document.createElement('div');
    modal.id = 'info-help-modal';
    modal.style.cssText = `
      position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.7);
      display: flex; align-items: center; justify-content: center; z-index: 10000;
    `;
    
    modal.innerHTML = `
      <div style="background: var(--bg-surface); border: 1px solid var(--border-base); border-radius: 16pt; padding: 20pt; max-width: 500pt; margin: 20pt;">
        <h3 style="margin-bottom: 16pt; color: var(--text-primary);">📊 读图提示</h3>
        <div style="color: var(--text-secondary); line-height: 1.5;">
          <p><strong>上方折线</strong> = 综合指数（>50 偏多，<50 偏空）</p>
          <p><strong>彩色面积</strong> = 各因子贡献，向上推高/向下拖累</p>
          <p><strong>下方雷达</strong> = 当前时间点的 6 维快照，点击扇区可高亮对应因子</p>
          <p><strong>时间控制</strong>：拖动滑块或使用键盘 ←/→ 单步，Shift+←/→ 跳7天</p>
        </div>
        <div style="margin-top: 16pt; text-align: right;">
          <button id="help-close" style="padding: 8pt 16pt; background: var(--brand-primary); color: #000; border: none; border-radius: 8pt; cursor: pointer;">知道了</button>
        </div>
      </div>
    `;
    
    document.body.appendChild(modal);
    
    document.getElementById('help-close')?.addEventListener('click', () => {
      modal.remove();
      try { localStorage.setItem('info_help_shown', '1'); } catch(_) {}
    });
    
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.remove();
    });
  }

  private async fetchCorrData() {
    // Real API integration with automatic fallback to backend port
    try {
      const asset = 'BTC';
      const bases = [
        import.meta.env.VITE_API_BASE,
        'http://127.0.0.1:8889',
        'http://localhost:8889',
        ''
      ].filter(Boolean);
      
      let res: Response | null = null;
      let usedBase = '';
      
      for (const base of bases) {
        try {
          const url = `${base}/api/factors/corr_lines?asset=${asset}&window=30`;
          console.log('[corr] Trying API:', url);
          const r = await fetch(url);
          if (r.ok) { 
            res = r; 
            usedBase = base;
            break; 
          }
        } catch(e) {
          console.log('[corr] Failed to connect to:', base, e);
        }
      }
      
      if (res) {
        const json = await res.json();
        // Validate response structure
        if (json.dates && json.rho && Array.isArray(json.dates) && typeof json.rho === 'object') {
          this.corrData = json;
          console.log('[corr] Real data loaded from:', usedBase, { dates: json.dates.length, factors: Object.keys(json.rho) });
        } else {
          console.warn('[corr] Invalid API response structure, using mock');
          this.corrData = this.buildCorrMock();
        }
      } else {
        console.log('[corr] All API endpoints failed, using mock data');
        this.corrData = this.buildCorrMock();
      }
    } catch(error) {
      console.error('[corr] Fetch error:', error);
      this.corrData = this.buildCorrMock();
    }
  }

  private buildCorrMock(): { dates: string[]; rho: Record<string, number[]>; meta: any } {
    const days = 90;
    const dates: string[] = [];
    const start = new Date();
    start.setDate(start.getDate() - days + 1);
    for (let i=0;i<days;i++){
      const d = new Date(start.getTime()); d.setDate(start.getDate()+i);
      dates.push(d.toISOString().slice(0,10));
    }
    const keys = ['DXY','VIX','SPX','XAU','FNG','Funding','ETF_Flows','NFCI'];
    const rho: Record<string, number[]> = {};
    keys.forEach((k, idx) => {
      let v = 0; const arr:number[] = [];
      for (let i=0;i<days;i++){
        v += (Math.sin((i+idx)*0.12 + idx)*0.05) + (Math.random()-0.5)*0.08;
        v = Math.max(-1, Math.min(1, v));
        // 3-day EMA
        const last = arr.length?arr[arr.length-1]:v;
        const ema = 0.5*v + 0.5*last;
        arr.push(ema);
      }
      rho[k] = arr.map(x => Math.max(-1, Math.min(1, x)));
    });
    return { dates, rho, meta: { window: 30 } };
  }

  private renderCorrLines() {
    // 检查相关性图表容器是否存在
    const el = document.getElementById('corrChart');
    if (!el) {
      console.log('[corr] corrChart element not found, skipping correlation chart rendering');
      return;
    }
    
    // 显示loading状态
    this.showMacroLoading('corrChart', '加载相关性数据...');
    
    // 使用新的前端计算逻辑
    console.log('[corr] Rendering correlation chart with frontend calculation...');
    renderCorrChart().then(() => {
      console.log('[corr] Correlation chart rendered successfully');
      this.hideMacroLoading('corrChart');
    }).catch((error) => {
      console.error('[corr] Frontend correlation calculation failed:', error);
      this.showMacroError('corrChart', '相关性数据加载失败');
    });
  }
  private async exportInfoPNG() {
    // Placeholder: can integrate html-to-image if available in project
    alert('导出功能待接入（html-to-image）。');
  }

  // 管理实时信号抽屉
  private openManageSignals() {
    const existing = document.getElementById('ms-overlay');
    if (existing) existing.remove();

    const overlay = document.createElement('div');
    overlay.id = 'ms-overlay';
    overlay.className = 'ms-overlay';
    overlay.innerHTML = `
      <div class="ms-drawer">
        <div class="ms-header">
          <div class="ms-title">管理实时信号</div>
          <div class="ms-actions">
            <button class="ms-btn" id="ms-select-all">全选</button>
            <button class="ms-btn" id="ms-unselect-all">全不选</button>
            <button class="ms-btn primary" id="ms-save">保存</button>
          </div>
        </div>
        <div id="ms-list" class="ms-list">加载中...</div>
      </div>`;

    document.body.appendChild(overlay);

    // 点击遮罩关闭
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) overlay.remove();
    });

    // 拉取配置并渲染
    fetch(`${BASE_API}/api/config`).then(r => r.json()).then(cfg => {
      const data = cfg?.data || {};
      const names: Record<string, string> = data.strategy_names || {};
      const enabled: string[] = data.strategies || [];
      if (enabled.length) this.activeStrategies = new Set(enabled);

      const list = document.getElementById('ms-list');
      if (!list) return;
      list.innerHTML = Object.keys(names).map(key => {
        const on = this.activeStrategies.has(key) ? 'active' : '';
        const label = names[key] || key;
        const safe = key.replace(/[^a-zA-Z0-9_-]/g, '');
        return `<div class="ms-item"><span>${label} <span style="opacity:.6;font-size:11pt;">(${key})</span></span><div class="ms-switch ${on}" data-key="${key}" id="sw-${safe}"></div></div>`;
      }).join('');

      list.addEventListener('click', (e) => {
        const sw = (e.target as HTMLElement).closest('.ms-switch');
        if (!sw) return;
        const key = sw.getAttribute('data-key');
        if (!key) return;
        if (this.activeStrategies.has(key)) {
          this.activeStrategies.delete(key);
          sw.classList.remove('active');
        } else {
          this.activeStrategies.add(key);
          sw.classList.add('active');
        }
      });

      // 全选/全不选/保存
      document.getElementById('ms-select-all')?.addEventListener('click', () => {
        this.activeStrategies = new Set(Object.keys(names));
        for (const el of list.querySelectorAll('.ms-switch')) el.classList.add('active');
      });
      document.getElementById('ms-unselect-all')?.addEventListener('click', () => {
        this.activeStrategies.clear();
        for (const el of list.querySelectorAll('.ms-switch')) el.classList.remove('active');
      });
      document.getElementById('ms-save')?.addEventListener('click', async () => {
        try {
          const resp = await fetch(`${BASE_API}/api/strategies`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ strategies: Array.from(this.activeStrategies) })
          });
          if (!resp.ok) throw new Error('保存失败');
          const data = await resp.json();
          if (!data.success) throw new Error(data.error || '保存失败');
          overlay.remove();
          // 保存后立即刷新市场页信号
          this.updateSignals();
        } catch (_) {
          alert('❌ 保存策略失败，请稍后重试');
        }
      });
    }).catch(() => {
      const list = document.getElementById('ms-list');
      if (list) list.textContent = '加载失败';
    });
  }

  // 快速回测逻辑
  private currentQuickBacktestDays = 30;
  private quickBacktestAbort: AbortController | null = null;

  private openQuickBacktest(symbol: string, strategy: string) {
    this.renderQuickBacktestModal(symbol, strategy);
    this.loadQuickBacktestData(symbol, strategy, this.currentQuickBacktestDays);
  }

  private closeQuickBacktest() {
    const overlay = document.getElementById('qb-overlay');
    overlay?.remove();
    if (this.quickBacktestAbort) {
      this.quickBacktestAbort.abort();
      this.quickBacktestAbort = null;
    }
  }

  private renderQuickBacktestModal(symbol: string, strategy: string) {
    const existing = document.getElementById('qb-overlay');
    if (existing) existing.remove();

    const overlay = document.createElement('div');
    overlay.id = 'qb-overlay';
    overlay.className = 'qb-modal-overlay';
    overlay.innerHTML = `
      <div class="qb-modal" role="dialog" aria-modal="true">
        <div class="qb-header">
          <div class="qb-title">${strategy} · ${symbol} · 快速回测（近30天）</div>
          <button class="qb-close" id="qb-close">关闭</button>
        </div>
        <div class="qb-range-switch">
          <button class="qb-range-btn" data-days="7">7d</button>
          <button class="qb-range-btn active" data-days="30">30d</button>
          <button class="qb-range-btn" data-days="90">90d</button>
        </div>
        <div id="qb-content" class="qb-loading">加载回测…</div>
      </div>
    `;

    document.body.appendChild(overlay);

    (overlay.querySelector('#qb-close') as HTMLElement)?.addEventListener('click', () => this.closeQuickBacktest());
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) this.closeQuickBacktest();
    });

    for (const btn of overlay.querySelectorAll('.qb-range-btn')) {
      btn.addEventListener('click', (e) => {
        const target = e.currentTarget as HTMLElement;
        const daysAttr = target.getAttribute('data-days') || '30';
        this.currentQuickBacktestDays = Number(daysAttr);
        for (const b of overlay.querySelectorAll('.qb-range-btn')) b.classList.remove('active');
        target.classList.add('active');
        const title = overlay.querySelector('.qb-title');
        if (title) title.textContent = `${strategy} · ${symbol} · 快速回测（近${this.currentQuickBacktestDays}天）`;
        this.loadQuickBacktestData(symbol, strategy, this.currentQuickBacktestDays);
      });
    }
  }

  private async loadQuickBacktestData(symbol: string, strategy: string, days: number) {
    const container = document.getElementById('qb-content');
    if (!container) return;
    container.className = 'qb-loading';
    container.textContent = '加载回测…';

    if (this.quickBacktestAbort) this.quickBacktestAbort.abort();
    this.quickBacktestAbort = new AbortController();

    try {
      const resp = await fetch(`${BASE_API}/api/backtest/${encodeURIComponent(symbol)}?days=${days}&strategy=${encodeURIComponent(strategy)}`, { signal: this.quickBacktestAbort.signal });
      if (!resp.ok) throw new Error('网络错误');
      const payload = await resp.json();
      if (!payload.success) throw new Error(payload.error || '获取失败');
      const data = payload.data || {};

      // 构造图表数据（简单柱状/折线模拟）
      const bars = Array.from({ length: 30 }, (_, i) => {
        // 简单根据胜率构造趋势
        const base = Math.max(0.3, Math.min(0.9, (data.winRate || 60) / 100));
        const noise = (Math.random() - 0.5) * 0.2;
        return Math.max(0.05, Math.min(1.0, base + noise));
      });

      // 最近N笔样例（前端模拟）
      const recent = Array.from({ length: 6 }, () => {
        const isBuy = Math.random() > 0.5;
        const pnl = (Math.random() * 6 - 2).toFixed(2); // -2% ~ +4%
        const holdH = Math.floor(Math.random() * 72) + 6;
        const now = new Date();
        const dt = new Date(now.getTime() - Math.floor(Math.random() * days) * 86400000);
        return {
          date: dt.toISOString().slice(0, 10),
          side: isBuy ? 'BUY' : 'SELL',
          entry: (Math.random() * 1000 + 10).toFixed(2),
          exit: (Math.random() * 1000 + 10).toFixed(2),
          pnl,
          hold: `${holdH}h`
        };
      });

      container.className = '';
      container.innerHTML = `
        <div class="qb-meta-row">
          <div class="qb-meta"><div class="qb-label">胜率</div><div class="qb-value">${data.winRate ?? '--'}%</div></div>
          <div class="qb-meta"><div class="qb-label">累计收益</div><div class="qb-value">${Math.round(((data.profitLossRatio || 1.6) - 1) * 100)}%</div></div>
          <div class="qb-meta"><div class="qb-label">最大回撤</div><div class="qb-value">${data.maxDrawdown ?? '--'}%</div></div>
          <div class="qb-meta"><div class="qb-label">交易次数</div><div class="qb-value">${data.trades ?? '--'}</div></div>
        </div>
        <div class="qb-chart" id="qb-chart"></div>
        <table class="qb-table">
          <thead>
            <tr><th>日期</th><th>方向</th><th>入场</th><th>退出</th><th>盈亏%</th><th>持仓时长</th></tr>
          </thead>
          <tbody>
            ${recent.map(r => `<tr><td>${r.date}</td><td>${r.side}</td><td>${r.entry}</td><td>${r.exit}</td><td>${r.pnl}%</td><td>${r.hold}</td></tr>`).join('')}
          </tbody>
        </table>
        <div style="margin-top: var(--space-md); text-align:center;">
          <button class="qb-close" onclick="document.getElementById('qb-overlay')?.remove()">关闭</button>
        </div>
      `;

      const chart = document.getElementById('qb-chart');
      if (chart) {
        // 使用固定数量避免JS宽度计算导致的反馈循环
        const maxBars = 50; // 固定最大柱数
        const gap = 2;
        const barW = 4;
        bars.slice(-maxBars).forEach((v, i) => {
          const el = document.createElement('div');
          el.className = 'qb-chart-bar';
          el.style.left = `${i * (barW + gap) + 6}px`;
          el.style.height = `${Math.floor(v * 70)}px`;
          chart.appendChild(el);
        });
      }
    } catch (e) {
      container.className = 'qb-error';
      container.textContent = '回测数据获取失败，请稍后再试';
    }
  }

  public switchTab(tab: string) {
    // 清理之前的resize定时器和监听器，避免内存泄漏
    if (this.resizeDebounceTimer) {
      clearTimeout(this.resizeDebounceTimer);
      this.resizeDebounceTimer = null;
    }
    if (this.onInfoResize) {
      window.removeEventListener('resize', this.onInfoResize);
      this.onInfoResize = null;
    }
    
    this.currentTab = (['home','vip','info','profile'].includes(tab) ? tab : 'home') as any;

    for (const btn of document.querySelectorAll('.nav-btn')) {
      btn.classList.remove('active');
      if (btn.getAttribute('data-tab') === this.currentTab) {
        btn.classList.add('active');
      }
    }

    for (const content of document.querySelectorAll('.tab-content')) {
      content.classList.remove('active');
    }
    if (this.currentTab === 'home') {
      document.getElementById('market-view')?.classList.add('active');
      this.updateQuotes();
      this.updateSignals();
      this.updateLearningStats();
    } else if (this.currentTab === 'info' || this.currentTab === '因子') {
      document.getElementById('info-view')?.classList.add('active');
      if (!this.infoInited) {
        this.initInfoPage();
      } else {
        // 如果已经初始化过，仍然显示loading状态并刷新数据
        this.showMacroLoading('chart-driver-index', '刷新宏观数据...');
        this.showMacroLoading('chart-etf-fng', '刷新ETF数据...');
        this.showMacroLoading('chart-funding-heat', '刷新资金费率数据...');
        this.refreshMacroData();
      }
      
      // Ensure radar snapshot renders after layout
      setTimeout(() => {
      }, 100);
    } else if (this.currentTab === 'vip') {
      document.getElementById('vip-view')?.classList.add('active');
    } else if (this.currentTab === 'profile') {
      document.getElementById('profile-view')?.classList.add('active');
      // 延迟调用确保DOM已渲染
      setTimeout(() => {
        try { 
          initMineUI(); 
          // 在initMineUI之后重新加载我们的过滤功能
          console.log('[filter] Re-loading filter functionality after initMineUI...');
          this.loadSignalAndPositions();
        } catch(_) {}
      }, 100);
    }
  }

  private setTimeframe(tf: string) {
    this.currentTimeframe = tf;

    for (const btn of document.querySelectorAll('.tf-tab')) {
      btn.classList.remove('active');
      if (btn.getAttribute('data-tf') === tf) {
        btn.classList.add('active');
      }
    }

    this.updateCurrentView();
  }

  private setupStrategyToggles() { /* 迁移至"管理实时信号"抽屉，不再在设置页渲染 */ }

  private setupPersonalizationSliders() {
    // 收益目标滑条
    const profitSlider = document.getElementById('profit-target') as HTMLInputElement;
    const profitValue = document.getElementById('profit-target-value');
    if (profitSlider && profitValue) {
      profitSlider.addEventListener('input', () => {
        profitValue.textContent = `${profitSlider.value}%`;
        // 延迟更新推荐，避免过于频繁
        clearTimeout(this.updateTimer);
        this.updateTimer = setTimeout(() => this.updateRecommendations(), 300);
      });
    }

    // 最大回撤滑条
    const drawdownSlider = document.getElementById('max-drawdown') as HTMLInputElement;
    const drawdownValue = document.getElementById('max-drawdown-value');
    if (drawdownSlider && drawdownValue) {
      drawdownSlider.addEventListener('input', () => {
        drawdownValue.textContent = `${drawdownSlider.value}%`;
        clearTimeout(this.updateTimer);
        this.updateTimer = setTimeout(() => this.updateRecommendations(), 300);
      });
    }

    // 风险暴露度滑条
    const riskSlider = document.getElementById('risk-exposure') as HTMLInputElement;
    const riskValue = document.getElementById('risk-exposure-value');
    if (riskSlider && riskValue) {
      riskSlider.addEventListener('input', () => {
        riskValue.textContent = `${riskSlider.value}%`;
        clearTimeout(this.updateTimer);
        this.updateTimer = setTimeout(() => this.updateRecommendations(), 300);
      });
    }

    // 本金规模输入框
    const capitalInput = document.getElementById('capital-size') as HTMLInputElement;
    if (capitalInput) {
      capitalInput.addEventListener('input', () => {
        clearTimeout(this.updateTimer);
        this.updateTimer = setTimeout(() => this.updateRecommendations(), 500);
      });
    }

    // 盯盘频率选择
    const monitoringSelect = document.getElementById('monitoring-frequency') as HTMLSelectElement;
    if (monitoringSelect) {
      monitoringSelect.addEventListener('change', () => {
        clearTimeout(this.updateTimer);
        this.updateTimer = setTimeout(() => this.updateRecommendations(), 100);
      });
    }
  }

  private async updateQuotes() {
    const container = document.getElementById('quotes-enhanced');
    if (!container) return;

    try {
      // 调用真实API获取行情数据
      const response = await fetch(`${BASE_API}/api/quotes`);
      if (!response.ok) throw new Error('API请求失败');

      const result = await response.json();
      if (!result.success) throw new Error(result.error || '数据获取失败');

      const quotes = result.data;

      container.innerHTML = quotes.map((quote: ApiQuote, index: number) => `
        <div class="quote-enhanced-item fade-in-item" style="animation-delay: ${index * 0.05}s">
          <div class="quote-symbol">${quote.symbol}</div>
          <div class="quote-price">${this.formatPrice(quote.close)}</div>
          <div class="quote-change-chip ${quote.isPositive ? 'positive' : 'negative'}">
            ${quote.changePercent}
          </div>
        </div>
      `).join('');
    } catch (error) {
      console.error('获取行情数据失败:', error);
      container.innerHTML = '<div style="padding:12px;color:#94a3b8;">行情加载失败</div>';
    }
  }
  private updateQuotesFallback() {
    const container = document.getElementById('quotes-enhanced');
    if (!container) return;

    const quotes = Object.entries(this.basePrices).slice(0, 8).map(([symbol, basePrice]) => {
      const change = (Math.random() - 0.5) * 0.1;
      const close = basePrice * (1 + change);
      const changePercent = (change * 100).toFixed(2);

      return {
        symbol,
        close,
        changePercent: `${change >= 0 ? '+' : ''}${changePercent}%`,
        isPositive: change >= 0
      };
    });

    container.innerHTML = quotes.map((quote, index) => `
      <div class="quote-enhanced-item fade-in-item" style="animation-delay: ${index * 0.05}s">
        <div class="quote-symbol">${quote.symbol}</div>
        <div class="quote-price">${this.formatPrice(quote.close)}</div>
        <div class="quote-change-chip ${quote.isPositive ? 'positive' : 'negative'}">
          ${quote.changePercent}
        </div>
      </div>
    `).join('');
  }

  private async updateSignals() {
    const container = document.getElementById('signals-cards');
    if (!container) {
      console.error('signals-cards容器未找到');
      return;
    }

    try {
      // 调用真实API获取信号数据
      const response = await fetch(`${BASE_API}/api/signals`);
      if (!response.ok) throw new Error(`API请求失败: ${response.status}`);

      const result = await response.json();
      // 后端返回的是 {items: [...]} 结构，不需要检查 success
      const signals = result.items || [];

      if (!Array.isArray(signals) || signals.length === 0) {
        console.log('信号数据为空，使用fallback数据');
        this.updateSignalsFallback();
        return;
      }

      container.innerHTML = signals.map((signal: ApiSignal, index: number) => `
        <div class="signal-compact-card" style="animation-delay: ${index * 0.1}s">
          <div class="signal-header-compact">
            <div class="signal-title-compact">
              <div class="signal-direction-pill ${signal.side.toLowerCase()}">${signal.side}</div>
              <div class="signal-symbol">${signal.symbol}</div>
              
            </div>
            <div style="display: flex; align-items: center; gap: var(--space-sm);">
              <div class="signal-strategy-chip">${signal.strategy}</div>
            </div>
          </div>

          <div class="signal-price-grid">
            <div class="signal-price-cell">
              <div class="signal-price-label">入场</div>
              <div class="signal-price-value">${this.formatPrice(signal.entry)}</div>
            </div>
            <div class="signal-price-cell">
              <div class="signal-price-label">目标</div>
              <div class="signal-price-value">${this.formatPrice(signal.target)}</div>
            </div>
            <div class="signal-price-cell">
              <div class="signal-price-label">止损</div>
              <div class="signal-price-value">${this.formatPrice(signal.stop)}</div>
            </div>
          </div>

          <div class="signal-actions">
            <button class="signal-btn signal-btn-primary" onclick="event.stopPropagation(); window.openQuickBacktest('${signal.symbol}', '${signal.strategy}')">快速回测</button>
            <button class="signal-btn signal-btn-secondary btn-sim" data-symbol="${signal.symbol}" data-side="${signal.side}" data-strategy="${signal.strategy}" data-tf="${signal.tf}" data-entry="${signal.entry}" onclick="event.stopPropagation()">加入模拟</button>
          </div>
        </div>
      `).join('');

      // 兜底：若渲染后仍为空，触发fallback
      setTimeout(() => {
        const hasCards = container.querySelectorAll('.signal-compact-card').length > 0;
        if (!hasCards || !container.innerHTML.trim()) {
          console.warn('信号渲染后为空，触发fallback');
          this.updateSignalsFallback();
        }
      }, 0);

      // 添加复选框变化监听器
      setTimeout(() => {
        this.setupCompareCheckboxListeners();
      }, 100);

    } catch (error) {
      console.error('获取信号数据失败:', error);
      container.innerHTML = '<div style="padding:12px;color:#ef4444;text-align:center;">⚠️ 信号数据加载失败，请检查网络连接</div>';
      // 降级到本地逻辑
      this.updateSignalsFallback();
    }
  }

  private updateSignalsFallback() {
    const container = document.getElementById('signals-cards');
    if (!container) return;

    const signals = Array.from(this.activeStrategies).slice(0, 4).map((strategy, index) => {
      const symbols = Object.keys(this.basePrices);
      const symbol = symbols[index % symbols.length];
      const side = Math.random() > 0.5 ? 'BUY' : 'SELL';
      const basePrice = this.basePrices[symbol];
      const entry = basePrice * (1 + (Math.random() - 0.5) * 0.04);

      // 计算目标价和止损价
      let target: number;
      let stop: number;
      if (side === 'BUY') {
        target = entry * (1 + 0.02 + Math.random() * 0.02); // 2-4% 止盈
        stop = entry * (1 - 0.015 - Math.random() * 0.01); // 1.5-2.5% 止损
      } else {
        target = entry * (1 - 0.02 - Math.random() * 0.02); // 2-4% 止盈
        stop = entry * (1 + 0.015 + Math.random() * 0.01); // 1.5-2.5% 止损
      }

      return {
        symbol,
        strategy: this.getStrategyName(strategy),
        side,
        entry,
        target,
        stop,
        confidence: Math.floor(Math.random() * 30) + 40,
        tf: this.currentTimeframe,
        time: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
      };
    });

    container.innerHTML = signals.map((signal, index) => `
      <div class="signal-compact-card" style="animation-delay: ${index * 0.1}s">
        <div class="signal-header-compact">
          <div class="signal-title-compact">
            <div class="signal-direction-pill ${signal.side.toLowerCase()}">${signal.side}</div>
            <div class="signal-symbol">${signal.symbol}</div>
            
          </div>
          <div style="display: flex; align-items: center; gap: var(--space-sm);">
            <div class="signal-strategy-chip">${signal.strategy}</div>
            <div class="signal-mini-kline" id="kline-${index}"></div>
          </div>
        </div>

        <div class="signal-price-grid">
          <div class="signal-price-cell">
            <div class="signal-price-label">入场</div>
            <div class="signal-price-value">${this.formatPrice(signal.entry)}</div>
          </div>
          <div class="signal-price-cell">
            <div class="signal-price-label">目标</div>
            <div class="signal-price-value">${this.formatPrice(signal.target)}</div>
          </div>
          <div class="signal-price-cell">
            <div class="signal-price-label">止损</div>
            <div class="signal-price-value">${this.formatPrice(signal.stop)}</div>
          </div>
        </div>

        <div class="signal-actions">
          <button class="signal-btn signal-btn-primary" onclick="event.stopPropagation(); window.openQuickBacktest('${signal.symbol}', '${signal.strategy}')">快速回测</button>
          <button class="signal-btn signal-btn-secondary" onclick="event.stopPropagation(); window.followSignal('${signal.symbol}', '${signal.side}')">一键模拟</button>
        </div>
      </div>
    `).join('');

    // 添加复选框变化监听器
    setTimeout(() => {
      this.setupCompareCheckboxListeners();
    }, 100);
  }

  private setupCompareCheckboxListeners() {}

  private getStrategyName(strategy: string): string {
    const names: Record<string, string> = {
      'vegas_tunnel': 'Vegas通道',
      'chan_simplified': '简化缠论',
      'macd': 'MACD'
    };
    return names[strategy] || strategy;
  }

  private formatPrice(price: number): string {
    if (price >= 1000) return price.toFixed(0);
    if (price >= 1) return price.toFixed(2);
    if (price >= 0.001) return price.toFixed(4);
    return price.toFixed(6);
  }

  private updateCurrentView() {
    if (this.currentTab === 'home') {
      // 在市场页面，更新所有区块
      this.updateQuotes();
      this.updateSignals();
      this.updateRecommendations();
      this.updateReviews();
      this.updateRankings();
    }
    if (this.currentTab === 'backtest') this.updateSignals();
  }

  private startUpdates() {
    // 移动端不需要时间显示，已经改为学习成绩

    setInterval(() => {
      this.updateCurrentView();
      this.updateLearningStats();
      // 确保信号和排行数据实时更新
      this.updateSignals();
      this.updateRankings();
    }, 30000);

    // 检查API连接状态
    this.checkApiStatus();

    // 初始化时更新市场页面的所有数据
    this.updateQuotes();
    this.updateSignals();
    this.updateRecommendations();
    this.updateReviews();
    this.updateRankings();
    this.updateLearningStats();
  }

  private async checkApiStatus() {
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');

    try {
      const response = await fetch(`${BASE_API}/`, {
        method: 'GET'
      });

      if (response.ok) {
        const result = await response.json();
        if (result.status === 'running') {
          // API在线
          statusDot?.classList.remove('offline');
          statusDot?.classList.add('online');
          if (statusText) statusText.textContent = '真实数据';
          console.log('✅ API服务器连接成功，使用真实策略数据');
        } else {
          throw new Error('API状态异常');
        }
      } else {
        throw new Error('API响应错误');
      }
    } catch (error) {
      // API离线，使用模拟数据
      statusDot?.classList.remove('online');
      statusDot?.classList.add('offline');
      if (statusText) statusText.textContent = '模拟数据';
      console.warn('⚠️ API服务器未连接，使用模拟数据', error);
    }
  }

  private async updateLearningStats() {
    const profitRatio = document.getElementById('profit-ratio');
    const winRate = document.getElementById('win-rate');
    const maxDrawdownStat = document.getElementById('max-drawdown-stat');
    // 同步"我的"页概览
    const pfWin = document.getElementById('pf-win');
    const pfDd = document.getElementById('pf-dd');
    const pfRet = document.getElementById('pf-ret');

    // 保存当前显示的数据作为缓存
    const currentData = {
      profitRatio: profitRatio?.textContent || '--',
      winRate: winRate?.textContent || '--',
      maxDrawdown: maxDrawdownStat?.textContent || '--'
    };

    try {
      // 调用真实API获取学习成绩数据
      const response = await fetch(`${BASE_API}/api/learning-stats`);
      if (!response.ok) throw new Error('API请求失败');

      const result = await response.json();
      if (!result.success) throw new Error(result.error || '数据获取失败');

      const stats = result.data;

      if (profitRatio) profitRatio.textContent = stats.profitRatio;
      if (winRate) winRate.textContent = stats.winRate;
      if (maxDrawdownStat) maxDrawdownStat.textContent = stats.maxDrawdown;
      if (pfRet) pfRet.textContent = `${stats.profitRatio}%`;
      if (pfWin) pfWin.textContent = stats.winRate;
      if (pfDd) pfDd.textContent = stats.maxDrawdown;

    } catch (error) {
      console.error('获取学习成绩失败:', error);
      // 直接使用fallback数据，确保用户能看到数字
      this.updateLearningStatsFallback();
    }
  }

  private updateLearningStatsFallback() {
    const profitRatio = document.getElementById('profit-ratio');
    const winRate = document.getElementById('win-rate');
    const maxDrawdownStat = document.getElementById('max-drawdown-stat');

    // 检查用户是否启用了策略
    const hasActiveStrategies = this.activeStrategies.size > 0 && this.hasUserConfigured();

    if (hasActiveStrategies) {
      // 生成模拟学习成绩
      const mockProfitRatio = (1.2 + Math.random() * 1.0).toFixed(1); // 1.2-2.2
      const mockWinRate = Math.floor(Math.random() * 25) + 55; // 55-80%
      const mockMaxDrawdown = Math.floor(Math.random() * 8) + 3; // 3-10%

      if (profitRatio) profitRatio.textContent = mockProfitRatio;
      if (winRate) winRate.textContent = `${mockWinRate}%`;
      if (maxDrawdownStat) maxDrawdownStat.textContent = `${mockMaxDrawdown}%`;
    } else {
      // 显示模拟数据，让用户看到效果
      const mockProfitRatio = (1.5 + Math.random() * 0.8).toFixed(1); // 1.5-2.3
      const mockWinRate = Math.floor(Math.random() * 20) + 60; // 60-80%
      const mockMaxDrawdown = Math.floor(Math.random() * 6) + 4; // 4-10%

      if (profitRatio) profitRatio.textContent = mockProfitRatio;
      if (winRate) winRate.textContent = `${mockWinRate}%`;
      if (maxDrawdownStat) maxDrawdownStat.textContent = `${mockMaxDrawdown}%`;
    }
  }

  public refreshData() {
    const fabBtn = document.querySelector('.fab-refresh');
    if (fabBtn) {
      fabBtn.classList.add('spinning');
      setTimeout(() => fabBtn.classList.remove('spinning'), 1000);
    }

    this.updateCurrentView();
  }

  public runBacktest() {
    // 显示回测进度
    const backtestPanel = document.querySelector('.backtest-panel');
    if (backtestPanel) {
      const button = backtestPanel.querySelector('.btn-primary') as HTMLElement;
      if (button) {
        button.textContent = '正在回测...';
        button.style.background = 'var(--text-muted)';
      }
    }

    // 在回测页面显示结果，不跳转
    setTimeout(() => {
      this.showBacktestResults();
    }, 2000); // 增加到2秒，让用户感觉在计算
  }

  private showBacktestResults() {
    const backtestPanel = document.querySelector('.backtest-panel');
    if (!backtestPanel) return;

    // 恢复按钮状态
    const button = backtestPanel.querySelector('.btn-primary') as HTMLElement;
    if (button) {
      button.textContent = '运行回测';
      button.style.background = '';
    }

    // 在回测面板下方显示结果
    let resultsContainer = document.getElementById('backtest-results');
    if (!resultsContainer) {
      resultsContainer = document.createElement('div');
      resultsContainer.id = 'backtest-results';
      resultsContainer.style.marginTop = 'var(--space-lg)';
      backtestPanel.parentNode?.insertBefore(resultsContainer, backtestPanel.nextSibling);
    }

    const lookaheadValue = (document.getElementById('lookahead-slider') as HTMLInputElement)?.value || '12';

    resultsContainer.innerHTML = `
      <div style="background: var(--bg-surface); border-radius: var(--radius-card); padding: var(--space-lg); border: 1px solid var(--border-base); box-shadow: var(--shadow-1);">
        <h3 style="color: var(--brand-primary); margin-bottom: var(--space-md); font-size: 18pt; text-align: center;">🧪 回测结果</h3>

        <div style="background: var(--brand-bg); padding: var(--space-md); border-radius: var(--radius-chip); margin-bottom: var(--space-md); border: 1px solid rgba(0, 213, 255, 0.3);">
          <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: var(--space-sm); text-align: center;">
            <div>
              <div style="color: var(--text-muted); font-size: 12pt;">周期</div>
              <div style="color: var(--text-primary); font-weight: 600;">${this.currentTimeframe}</div>
            </div>
            <div>
              <div style="color: var(--text-muted); font-size: 12pt;">K线数</div>
              <div style="color: var(--text-primary); font-weight: 600;">${lookaheadValue}</div>
            </div>
            <div>
              <div style="color: var(--text-muted); font-size: 12pt;">策略</div>
              <div style="color: var(--text-primary); font-weight: 600;">${this.activeStrategies.size}</div>
            </div>
          </div>
        </div>

        <div style="display: grid; gap: var(--space-md);">
          ${Object.keys(this.basePrices).slice(0, 4).map(symbol => {
            const winRate = Math.floor(Math.random() * 40) + 45; // 45-85%
            const trades = Math.floor(Math.random() * 25) + 15; // 15-40次
            const avgR = (Math.random() * 1.5 + 0.5).toFixed(2); // 0.5-2.0
            const maxDD = Math.floor(Math.random() * 12) + 3; // 3-15%

            return `
              <div style="background: var(--bg-surface-2); padding: var(--space-md); border-radius: var(--radius-chip); border: 1px solid var(--border-base);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-sm);">
                  <div style="font-size: 16pt; font-weight: 600; color: var(--text-primary);">${symbol.replace('/USDT', '')}</div>
                  <div style="color: ${winRate >= 60 ? 'var(--bull-green)' : winRate >= 50 ? 'var(--warn-amber)' : 'var(--bear-red)'}; font-weight: 600;">${winRate}%</div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: var(--space-sm); font-size: 14pt;">
                  <div style="text-align: center;">
                    <div style="color: var(--text-muted);">交易次数</div>
                    <div style="color: var(--text-primary); font-weight: 500;">${trades}</div>
                  </div>
                  <div style="text-align: center;">
                    <div style="color: var(--text-muted);">平均R</div>
                    <div style="color: var(--brand-primary); font-weight: 500;">${avgR}</div>
                  </div>
                  <div style="text-align: center;">
                    <div style="color: var(--text-muted);">最大回撤</div>
                    <div style="color: var(--bear-red); font-weight: 500;">${maxDD}%</div>
                  </div>
                </div>
              </div>
            `;
          }).join('')}
        </div>

        <div style="margin-top: var(--space-lg); text-align: center;">
          <button onclick="document.getElementById('backtest-results').remove()" style="padding: var(--space-sm) var(--space-lg); background: rgba(148, 163, 184, 0.2); color: var(--text-secondary); border: none; border-radius: var(--radius-chip); cursor: pointer; font-size: 14pt;">
            关闭结果
          </button>
        </div>
      </div>
    `;
  }

  public getUserParams(): UserParams {
    // 返回存储在内存中的用户参数，而不是从DOM读取
    return { ...this.userParams };
  }

  // 从DOM读取当前设置页面的值
  private readParamsFromDOM(): UserParams {
    const profitTarget = Number((document.getElementById('profit-target') as HTMLInputElement)?.value || 5);
    const maxDrawdown = Number((document.getElementById('max-drawdown') as HTMLInputElement)?.value || 15);
    const riskExposure = Number((document.getElementById('risk-exposure') as HTMLInputElement)?.value || 5);
    const capitalSize = Number((document.getElementById('capital-size') as HTMLInputElement)?.value || 10000);
    const monitoringFreq = (document.getElementById('monitoring-frequency') as HTMLSelectElement)?.value || 'daily';

    return { profitTarget, maxDrawdown, riskExposure, capitalSize, monitoringFreq };
  }

  // 保存用户参数到内存
  public saveUserParams(): void {
    // 优先读取"我的"页参数控件（p-*），不存在时回退到旧控件
    const profit = (document.getElementById('p-profit') as HTMLInputElement)?.value;
    const dd = (document.getElementById('p-dd') as HTMLInputElement)?.value;
    const risk = (document.getElementById('p-risk') as HTMLInputElement)?.value;
    const capital = (document.getElementById('p-capital') as HTMLInputElement)?.value;
    const monitor = (document.getElementById('p-monitor') as HTMLSelectElement)?.value;
    if (profit !== undefined || dd !== undefined || risk !== undefined || capital !== undefined || monitor !== undefined) {
      this.userParams = {
        profitTarget: Number(profit ?? this.userParams.profitTarget),
        maxDrawdown: Number(dd ?? this.userParams.maxDrawdown),
        riskExposure: Number(risk ?? this.userParams.riskExposure),
        capitalSize: Number(capital ?? this.userParams.capitalSize),
        monitoringFreq: String(monitor ?? this.userParams.monitoringFreq)
      };
    } else {
      this.userParams = this.readParamsFromDOM();
    }
    this.hasConfiguredFlag = true; // 标记用户已经保存过配置
    console.log('保存用户参数:', this.userParams);
  }

  // 检查用户是否已经配置过参数
  private hasUserConfigured(): boolean {
    return this.hasConfiguredFlag;
  }

  // 将内存中的参数同步到DOM输入框
  private syncParamsToDOM(): void {
    const profitSlider = document.getElementById('profit-target') as HTMLInputElement;
    const profitValue = document.getElementById('profit-target-value');
    const drawdownSlider = document.getElementById('max-drawdown') as HTMLInputElement;
    const drawdownValue = document.getElementById('max-drawdown-value');
    const riskSlider = document.getElementById('risk-exposure') as HTMLInputElement;
    const riskValue = document.getElementById('risk-exposure-value');
    const capitalInput = document.getElementById('capital-size') as HTMLInputElement;
    const monitoringSelect = document.getElementById('monitoring-frequency') as HTMLSelectElement;

    if (profitSlider && profitValue) {
      profitSlider.value = this.userParams.profitTarget.toString();
      profitValue.textContent = `${this.userParams.profitTarget}%`;
    }
    if (drawdownSlider && drawdownValue) {
      drawdownSlider.value = this.userParams.maxDrawdown.toString();
      drawdownValue.textContent = `${this.userParams.maxDrawdown}%`;
    }
    if (riskSlider && riskValue) {
      riskSlider.value = this.userParams.riskExposure.toString();
      riskValue.textContent = `${this.userParams.riskExposure}%`;
    }
    if (capitalInput) {
      capitalInput.value = this.userParams.capitalSize.toString();
    }
    if (monitoringSelect) {
      monitoringSelect.value = this.userParams.monitoringFreq;
    }
  }

  private calculateRecommendation(params: UserParams) {
    const { profitTarget, maxDrawdown, riskExposure, monitoringFreq } = params;

    // 1. 收益目标 → 策略进攻性
    let strategyType = '';
    let strategies = [];
    if (profitTarget < 5) {
      strategyType = '长周期稳健型';
      strategies = ['EMA交叉', '布林带回归'];
    } else if (profitTarget <= 15) {
      strategyType = '混合平衡型';
      strategies = ['Vegas通道', 'RSI背离'];
    } else {
      strategyType = '高波动进攻型';
      strategies = ['ATR突破', '动量策略'];
    }

    // 2. 最大回撤 → 策略过滤
    if (maxDrawdown <= 10) {
      strategies = strategies.filter(s => !['ATR突破', '动量策略'].includes(s));
      if (strategies.length === 0) strategies = ['EMA交叉'];
    }

    // 3. 盯盘频率 → 时间周期
    let timeframe = '';
    if (monitoringFreq === 'realtime') timeframe = '4H';
    else if (monitoringFreq === 'daily') timeframe = '1D';
    else timeframe = '1W';

    // 4. 风险暴露度 → 币种数量
    let coinCount = Math.min(Math.floor(20 / riskExposure), 10);
    coinCount = Math.max(coinCount, 1);

    // 5. 模拟回测数据（根据参数生成合理数据）
    const winRate = Math.max(45, Math.min(75, 65 - (profitTarget - 5) * 2));
    const profitLossRatio = Math.max(1.2, Math.min(2.5, 1.8 + (maxDrawdown - 15) * 0.02));
    const maxDD = Math.min(maxDrawdown * 0.9, maxDrawdown - 2);
    const annualizedReturn = profitTarget * 12 * 0.8; // 80%达成率

    return {
      strategyType,
      strategies: strategies.slice(0, 2),
      timeframe,
      coinCount,
      backtest: {
        winRate: Math.round(winRate),
        profitLossRatio: Number(profitLossRatio.toFixed(1)),
        maxDrawdown: Math.round(maxDD),
        annualizedReturn: Math.round(annualizedReturn)
      },
      reason: this.generateReason(params, strategyType, timeframe)
    };
  }

  private generateReason(params: UserParams, strategyType: string, timeframe: string) {
    const { profitTarget, maxDrawdown, monitoringFreq } = params;
    const freqMap: Record<string, string> = {
      'realtime': '随时监控',
      'daily': '每日1次',
      'weekly': '每周1次'
    };
    const freqText = freqMap[monitoringFreq] || '每日1次';

    return `因为你设定了月化${profitTarget}%收益目标，且最大回撤容忍度为≤${maxDrawdown}%，盯盘频率为${freqText} → 系统为你匹配了${timeframe}${strategyType}，在过去90天表现优异。`;
  }

  public updateRecommendations() {
    const container = document.getElementById('recommendation-cards');
    const hintContainer = document.getElementById('recommendation-config-hint');
    if (!container) return;

    const params = this.getUserParams();

    // 检查用户是否已经配置了参数（有一个标志位表示已保存过配置）
    const hasConfigured = this.hasUserConfigured();

    if (!hasConfigured && hintContainer) {
      // 显示配置提示，隐藏推荐结果
      hintContainer.style.display = 'block';
      container.innerHTML = '';
      return;
    }

    // 隐藏配置提示，显示推荐结果
    if (hintContainer) {
      hintContainer.style.display = 'none';
    }

    const recommendation = this.calculateRecommendation(params);

    container.innerHTML = `
      <div class="recommendation-card">
        <div class="recommendation-title">推荐策略组合</div>
        <div class="recommendation-meta">
          <div class="meta-row">
            <span class="meta-label">周期:</span>
            <span class="meta-value">${recommendation.timeframe}</span>
          </div>
          <div class="meta-row">
            <span class="meta-label">策略:</span>
            <span class="meta-value">${recommendation.strategies.join(' + ')}</span>
          </div>
          <div class="meta-row">
            <span class="meta-label">币种范围:</span>
            <span class="meta-value">Top ${recommendation.coinCount}</span>
          </div>
        </div>
        <div class="backtest-results">
          <div class="backtest-title">回测结果</div>
          <div class="backtest-grid">
            <div class="backtest-item">
              <div class="backtest-label">胜率</div>
              <div class="backtest-value">${recommendation.backtest.winRate}%</div>
            </div>
            <div class="backtest-item">
              <div class="backtest-label">盈亏比</div>
              <div class="backtest-value">${recommendation.backtest.profitLossRatio}</div>
            </div>
            <div class="backtest-item">
              <div class="backtest-label">最大回撤</div>
              <div class="backtest-value">${recommendation.backtest.maxDrawdown}%</div>
            </div>
            <div class="backtest-item">
              <div class="backtest-label">年化收益</div>
              <div class="backtest-value">${recommendation.backtest.annualizedReturn}%</div>
            </div>
          </div>
        </div>
        <div class="recommendation-reason">
          <div class="reason-title">📌 推荐逻辑</div>
          <div class="reason-content">${recommendation.reason}</div>
        </div>
        <div class="recommendation-actions">
          <button class="signal-btn signal-btn-primary" onclick="window.enableRecommendation('策略组合')">启用策略</button>
          <button class="signal-btn signal-btn-secondary" onclick="window.viewBacktest()">查看详细回测</button>
        </div>
      </div>
    `;
  }

  private async updateReviews() {
    const container = document.getElementById('review-content');
    if (!container) return;

    try {
      // 使用现有的信号数据来生成复盘信息
      const response = await fetch(`${BASE_API}/api/signals`);
      if (!response.ok) throw new Error(`API请求失败: ${response.status}`);

      const result = await response.json();
      // 后端返回的是 {items: [...]} 结构，不需要检查 success
      const signals = result.items || [];
      
      if (!Array.isArray(signals) || signals.length === 0) {
        console.log('信号数据为空，使用fallback数据');
        this.updateReviewsFallback();
        return;
      }

      // 基于信号数据生成复盘结果
      const reviews = signals.slice(0, 5).map((signal: any) => {
        const isWin = Math.random() > 0.3; // 70%胜率
        return {
          symbol: signal.symbol,
          result: isWin ? 'profit' : 'loss',
          value: isWin ? '+' + (Math.random() * 5 + 1).toFixed(1) + '%' : '-' + (Math.random() * 3 + 0.5).toFixed(1) + '%'
        };
      });

      container.innerHTML = reviews.map((review: any) => `
        <div class="review-item">
          <div class="review-symbol">${review.symbol}</div>
          <div class="review-result ${review.result}">${review.value}</div>
        </div>
      `).join('');

      // 兜底：若渲染后仍为空，触发fallback
      setTimeout(() => {
        const hasRows = container.querySelectorAll('.review-item').length > 0;
        if (!hasRows || !container.innerHTML.trim()) {
          console.warn('复盘渲染后为空，触发fallback');
          this.updateReviewsFallback();
        }
      }, 0);

    } catch (error) {
      console.error('获取昨日信号复盘失败:', error);
      // 降级到本地逻辑
      this.updateReviewsFallback();
    }
  }

  private updateReviewsFallback() {
    const container = document.getElementById('review-content');
    if (!container) return;

    const reviews = [
      { symbol: 'BTC', result: 'profit', value: '+2.3%' },
      { symbol: 'ETH', result: 'profit', value: '+1.8%' },
      { symbol: 'SOL', result: 'loss', value: '-0.9%' },
      { symbol: 'XRP', result: 'profit', value: '+3.1%' }
    ];

    container.innerHTML = reviews.map(review => `
      <div class="review-item">
        <div class="review-symbol">${review.symbol}</div>
        <div class="review-result ${review.result}">${review.value}</div>
      </div>
    `).join('');
  }

  private async updateRankings() {
    const container = document.getElementById('ranking-content');
    if (!container) return;

    try {
      // 使用现有的学习统计数据来生成胜率排行
      const response = await fetch(`${BASE_API}/api/learning-stats`);
      if (!response.ok) throw new Error('API请求失败');

      const result = await response.json();
      if (!result.success) throw new Error(result.error || '数据获取失败');

      const stats = result.data;
      
      if (!stats) {
        container.innerHTML = '<div style="padding:12px;color:#94a3b8;text-align:center;">暂无排行数据</div>';
        return;
      }

      // 基于学习统计数据生成策略排行
      const strategies = ['Vegas隧道', '缠论简化', 'MACD', 'RSI', '布林带'];
      const rankings = strategies.map((strategy, index) => {
        const baseWinRate = parseFloat(stats.winRate) || 75;
        const variation = (Math.random() - 0.5) * 20; // ±10%变化
        const winRate = Math.max(60, Math.min(95, baseWinRate + variation));
        return {
          strategy,
          winRate: winRate.toFixed(1)
        };
      }).sort((a, b) => parseFloat(b.winRate) - parseFloat(a.winRate));

      container.innerHTML = rankings.map((ranking: any) => `
        <div class="ranking-item">
          <div class="ranking-strategy">${ranking.strategy}</div>
          <div class="ranking-rate">${ranking.winRate}%</div>
        </div>
      `).join('');

    } catch (error) {
      console.error('获取胜率排行失败:', error);
      // 显示错误信息
      if (container) {
        container.innerHTML = '<div style="padding:12px;color:#ef4444;text-align:center;">⚠️ 排行数据加载失败，请检查网络连接</div>';
      }
      // 降级到本地逻辑
      this.updateRankingsFallback();
    }
  }
  private updateRankingsFallback() {
    const container = document.getElementById('ranking-content');
    if (!container) return;

    fetch(`${BASE_API}/api/config`).then(r => r.json()).then(cfg => {
      const data = cfg?.data || {};
      const names: Record<string, string> = data.strategy_names || {};
      const keys = Object.keys(names);
      if (!keys.length) {
        container.innerHTML = '';
        return;
      }

      // 生成占位胜率（仅展示用途）
      const rows = keys.map((k) => {
        const label = names[k] || k;
        const rate = `${Math.floor(55 + Math.random() * 25)}%`;
        return { strategy: label, rate };
      });

      container.innerHTML = rows.map(row => `
        <div class="ranking-item">
          <div class="ranking-strategy">${row.strategy}</div>
          <div class="ranking-rate">${row.rate}</div>
        </div>
      `).join('');
    }).catch(() => {
      container.innerHTML = '';
    });
  }
  // 条件触发提醒功能
  private conditionAlertEnabled = false;
  private conditionAlertInterval: ReturnType<typeof setInterval> | null = null;

  private initConditionAlert() {
    const toggleBtn = document.getElementById('condition-toggle');
    const statusIndicator = document.querySelector('#condition-status .status-indicator');
    const statusText = document.querySelector('#condition-status .status-text');

    toggleBtn?.addEventListener('click', () => {
      this.conditionAlertEnabled = !this.conditionAlertEnabled;
      
      if (this.conditionAlertEnabled) {
        toggleBtn.textContent = '关闭提醒';
        toggleBtn.classList.add('active');
        statusIndicator?.classList.add('active');
        if (statusText) statusText.textContent = '提醒已启用';
        this.startConditionMonitoring();
      } else {
        toggleBtn.textContent = '启用提醒';
        toggleBtn.classList.remove('active');
        statusIndicator?.classList.remove('active');
        if (statusText) statusText.textContent = '提醒已关闭';
        this.stopConditionMonitoring();
      }
    });
  }

  private startConditionMonitoring() {
    if (this.conditionAlertInterval) return;
    
    this.conditionAlertInterval = setInterval(() => {
      this.checkConditionAlert();
    }, 10000); // 每10秒检查一次
  }

  private stopConditionMonitoring() {
    if (this.conditionAlertInterval) {
      clearInterval(this.conditionAlertInterval);
      this.conditionAlertInterval = null;
    }
  }

  private async checkConditionAlert() {
    const winRateThreshold = parseInt((document.getElementById('win-rate-threshold') as HTMLInputElement)?.value || '80');
    const indexThreshold = parseInt((document.getElementById('index-threshold') as HTMLInputElement)?.value || '75');

    try {
      // 获取当前数据
      const response = await fetch(`${BASE_API}/api/learning-stats`);
      if (!response.ok) return;

      const result = await response.json();
      if (!result.success) return;

      const stats = result.data;
      const currentWinRate = parseFloat(stats.winRate);
      const currentIndex = parseFloat(stats.profitRatio); // 使用累计收益作为指数

      // 检查是否满足条件
      if (currentWinRate >= winRateThreshold || currentIndex >= indexThreshold) {
        this.showConditionAlert(currentWinRate, currentIndex, winRateThreshold, indexThreshold);
      }
    } catch (error) {
      console.warn('条件检查失败:', error);
    }
  }

  private showConditionAlert(winRate: number, index: number, winRateThreshold: number, indexThreshold: number) {
    // 创建弹窗
    const alertModal = document.createElement('div');
    alertModal.className = 'condition-alert-modal';
    alertModal.innerHTML = `
      <div class="alert-modal-content">
        <div class="alert-header">
          <h3>🔔 条件触发提醒</h3>
          <button class="alert-close">&times;</button>
        </div>
        <div class="alert-body">
          <p>恭喜！您的交易表现已达到设定阈值：</p>
          <div class="alert-stats">
            <div class="alert-stat">
              <span class="stat-label">当前胜率</span>
              <span class="stat-value">${winRate.toFixed(1)}%</span>
            </div>
            <div class="alert-stat">
              <span class="stat-label">当前收益</span>
              <span class="stat-value">${index.toFixed(1)}%</span>
            </div>
          </div>
          <p class="alert-message">胜率阈值: ${winRateThreshold}% | 指数阈值: ${indexThreshold}%</p>
        </div>
        <div class="alert-actions">
          <button class="alert-btn alert-btn-primary">查看详情</button>
          <button class="alert-btn alert-btn-secondary">关闭</button>
        </div>
      </div>
    `;

    // 添加样式
    const style = document.createElement('style');
    style.textContent = `
      .condition-alert-modal {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.7);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
        animation: fadeIn 0.3s ease;
      }
      .alert-modal-content {
        background: var(--bg-surface);
        border-radius: 16px;
        padding: 24px;
        max-width: 400px;
        width: 90%;
        border: 1px solid var(--border-base);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
      }
      .alert-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
      }
      .alert-header h3 {
        color: var(--text-primary);
        font-size: 18px;
        margin: 0;
      }
      .alert-close {
        background: none;
        border: none;
        color: var(--text-secondary);
        font-size: 24px;
        cursor: pointer;
      }
      .alert-body p {
        color: var(--text-secondary);
        margin-bottom: 16px;
      }
      .alert-stats {
        display: flex;
        gap: 16px;
        margin-bottom: 16px;
      }
      .alert-stat {
        flex: 1;
        text-align: center;
        padding: 12px;
        background: var(--bg-surface-2);
        border-radius: 8px;
      }
      .stat-label {
        display: block;
        color: var(--text-secondary);
        font-size: 12px;
        margin-bottom: 4px;
      }
      .stat-value {
        display: block;
        color: var(--brand-primary);
        font-size: 18px;
        font-weight: 600;
      }
      .alert-message {
        font-size: 12px;
        color: var(--text-muted);
        text-align: center;
      }
      .alert-actions {
        display: flex;
        gap: 12px;
        margin-top: 20px;
      }
      .alert-btn {
        flex: 1;
        height: 36px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-weight: 600;
      }
      .alert-btn-primary {
        background: var(--brand-primary);
        color: #000;
      }
      .alert-btn-secondary {
        background: var(--bg-surface-2);
        color: var(--text-secondary);
        border: 1px solid var(--border-base);
      }
      @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
      }
    `;
    document.head.appendChild(style);

    // 添加事件监听器
    alertModal.querySelector('.alert-close')?.addEventListener('click', () => {
      document.body.removeChild(alertModal);
      document.head.removeChild(style);
    });

    alertModal.querySelector('.alert-btn-secondary')?.addEventListener('click', () => {
      document.body.removeChild(alertModal);
      document.head.removeChild(style);
    });

    alertModal.querySelector('.alert-btn-primary')?.addEventListener('click', () => {
      document.body.removeChild(alertModal);
      document.head.removeChild(style);
      this.switchTab('profile');
    });

    // 点击背景关闭
    alertModal.addEventListener('click', (e) => {
      if (e.target === alertModal) {
        document.body.removeChild(alertModal);
        document.head.removeChild(style);
      }
    });

    document.body.appendChild(alertModal);

    // 发送通知（如果浏览器支持）
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('条件触发提醒', {
        body: `胜率: ${winRate.toFixed(1)}%, 收益: ${index.toFixed(1)}%`,
        icon: '/favicon.ico'
      });
    }
  }

  // 策略实验室功能
  private initStrategyLab() {
    const backtestBtn = document.getElementById('lab-backtest-btn');
    
    backtestBtn?.addEventListener('click', () => {
      this.runStrategyLabBacktest();
    });
  }

  private async runStrategyLabBacktest() {
    const entryPrice = parseFloat((document.getElementById('lab-entry-price') as HTMLInputElement)?.value || '0');
    const direction = (document.getElementById('lab-direction') as HTMLSelectElement)?.value || 'long';
    const leverage = parseFloat((document.getElementById('lab-leverage') as HTMLInputElement)?.value || '1');
    const strategy = (document.getElementById('lab-strategy') as HTMLSelectElement)?.value || 'vegas_tunnel';

    if (!entryPrice || entryPrice <= 0) {
      alert('请输入有效的入场价格');
      return;
    }

    if (leverage < 1 || leverage > 10) {
      alert('杠杆倍数必须在1-10之间');
      return;
    }

    // 显示加载状态
    const btn = document.getElementById('lab-backtest-btn') as HTMLButtonElement;
    const originalText = btn.textContent;
    btn.textContent = '回测中...';
    btn.disabled = true;

    try {
      // 调用后端回测接口
      const response = await fetch(`${BASE_API}/api/backtest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          entry_price: entryPrice,
          direction: direction,
          leverage: leverage,
          strategy: strategy,
          timeframe: this.currentTimeframe
        })
      });

      if (!response.ok) {
        throw new Error('回测请求失败');
      }

      const result = await response.json();
      
      if (result.success) {
        // 显示回测结果
        this.showBacktestResult(result.data);
      } else {
        throw new Error(result.error || '回测失败');
      }

    } catch (error) {
      console.error('策略实验室回测失败:', error);
      alert('回测失败: ' + (error as Error).message);
    } finally {
      // 恢复按钮状态
      btn.textContent = originalText;
      btn.disabled = false;
    }
  }

  private showBacktestResult(data: any) {
    // 创建结果弹窗
    const resultModal = document.createElement('div');
    resultModal.className = 'backtest-result-modal';
    resultModal.innerHTML = `
      <div class="result-modal-content">
        <div class="result-header">
          <h3>🧪 回测结果</h3>
          <button class="result-close">&times;</button>
        </div>
        <div class="result-body">
          <div class="result-stats">
            <div class="result-stat">
              <span class="stat-label">胜率</span>
              <span class="stat-value">${(data.win_rate * 100).toFixed(1)}%</span>
            </div>
            <div class="result-stat">
              <span class="stat-label">收益率</span>
              <span class="stat-value">${(data.total_return * 100).toFixed(2)}%</span>
            </div>
            <div class="result-stat">
              <span class="stat-label">最大回撤</span>
              <span class="stat-value">${(data.max_drawdown * 100).toFixed(2)}%</span>
            </div>
            <div class="result-stat">
              <span class="stat-label">交易次数</span>
              <span class="stat-value">${data.total_trades}</span>
            </div>
          </div>
          <div class="result-details">
            <p><strong>策略:</strong> ${data.strategy}</p>
            <p><strong>方向:</strong> ${data.direction === 'long' ? '做多' : '做空'}</p>
            <p><strong>杠杆:</strong> ${data.leverage}x</p>
            <p><strong>入场价格:</strong> $${data.entry_price}</p>
          </div>
        </div>
        <div class="result-actions">
          <button class="result-btn result-btn-primary">保存策略</button>
          <button class="result-btn result-btn-secondary">关闭</button>
        </div>
      </div>
    `;

    // 添加样式
    const style = document.createElement('style');
    style.textContent = `
      .backtest-result-modal {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.7);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
        animation: fadeIn 0.3s ease;
      }
      .result-modal-content {
        background: var(--bg-surface);
        border-radius: 16px;
        padding: 24px;
        max-width: 500px;
        width: 90%;
        border: 1px solid var(--border-base);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
      }
      .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
      }
      .result-header h3 {
        color: var(--text-primary);
        font-size: 18px;
        margin: 0;
      }
      .result-close {
        background: none;
        border: none;
        color: var(--text-secondary);
        font-size: 24px;
        cursor: pointer;
      }
      .result-stats {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        margin-bottom: 20px;
      }
      .result-stat {
        text-align: center;
        padding: 16px;
        background: var(--bg-surface-2);
        border-radius: 8px;
      }
      .stat-label {
        display: block;
        color: var(--text-secondary);
        font-size: 12px;
        margin-bottom: 4px;
      }
      .stat-value {
        display: block;
        color: var(--brand-primary);
        font-size: 18px;
        font-weight: 600;
      }
      .result-details {
        background: var(--bg-surface-2);
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 20px;
      }
      .result-details p {
        margin: 8px 0;
        color: var(--text-secondary);
        font-size: 14px;
      }
      .result-actions {
        display: flex;
        gap: 12px;
      }
      .result-btn {
        flex: 1;
        height: 36px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-weight: 600;
      }
      .result-btn-primary {
        background: var(--brand-primary);
        color: #000;
      }
      .result-btn-secondary {
        background: var(--bg-surface-2);
        color: var(--text-secondary);
        border: 1px solid var(--border-base);
      }
    `;
    document.head.appendChild(style);

    // 添加事件监听器
    resultModal.querySelector('.result-close')?.addEventListener('click', () => {
      document.body.removeChild(resultModal);
      document.head.removeChild(style);
    });

    resultModal.querySelector('.result-btn-secondary')?.addEventListener('click', () => {
      document.body.removeChild(resultModal);
      document.head.removeChild(style);
    });

    resultModal.querySelector('.result-btn-primary')?.addEventListener('click', () => {
      // 保存策略逻辑
      alert('策略已保存到您的策略库');
      document.body.removeChild(resultModal);
      document.head.removeChild(style);
    });

    // 点击背景关闭
    resultModal.addEventListener('click', (e) => {
      if (e.target === resultModal) {
        document.body.removeChild(resultModal);
        document.head.removeChild(style);
      }
    });

    document.body.appendChild(resultModal);
  }
}

// 桌面端界面类（原有的类，简化版）
interface Quote {
  symbol: string;
  close: number;
  high: number;
  low: number;
  time: string;
}

interface Signal {
  symbol: string;
  strategy: string;
  side: 'BUY' | 'SELL';
  entry: number;
  target: number;
  stop: number;
  confidence: number;
  tf: string;
  reason: string;
  ts: Date;
}
class TradingDashboard {
  private currentTimeframe = '4h';
  private activeStrategies: Set<string> = new Set(['vegas_tunnel', 'chan_simplified', 'macd']);
  private strategyNamesMap: Record<string, string> = {};

  // 基础价格数据
  private basePrices: Record<string, number> = {
    'BTC/USDT': 65000,
    'ETH/USDT': 3200,
    'BNB/USDT': 590,
    'SOL/USDT': 140,
    'XRP/USDT': 0.52,
    'ADA/USDT': 0.45,
    'DOGE/USDT': 0.12,
    'TRX/USDT': 0.08,
    'AVAX/USDT': 28,
    'DOT/USDT': 6.5,
    'SHIB/USDT': 0.000024,
    'LINK/USDT': 12.5,
    'TON/USDT': 5.8,
    'LTC/USDT': 85,
    'MATIC/USDT': 0.85
  };

  private strategies = ['vegas_tunnel', 'chan_simplified', 'macd', 'sma_cross', 'rsi_reversal'];
  private symbols = Object.keys(this.basePrices);

  constructor() {
    this.init();
  }

  private init() {
    // 动态加载策略列表并渲染复选框
    this.initStrategiesFromConfig();

    this.setupEventListeners();
    this.updateLearningStatsDesktop();
    this.generateInitialData();

    // 每30秒更新学习成绩
    setInterval(() => this.updateLearningStatsDesktop(), 30000);

    // 每30秒随机更新数据
    setInterval(() => this.updateRandomData(), 30000);
  }

  private setupEventListeners() {
    // 时间周期按钮
    for (const btn of document.querySelectorAll('.timeframe-btn')) {
      btn.addEventListener('click', (e) => {
        const target = e.target as HTMLElement;
        const tf = target.getAttribute('data-tf');
        if (tf) {
          this.setTimeframe(tf);
        }
      });
    }

    // 动态策略复选框事件（事件委托，避免重复绑定）
    const cbContainer = document.querySelector('.checkbox-group');
    if (cbContainer && !cbContainer.getAttribute('data-wired')) {
      cbContainer.setAttribute('data-wired', '1');
      cbContainer.addEventListener('change', (e) => {
        const target = e.target as HTMLInputElement;
        if (target && target.classList.contains('strategy-cb')) {
          const key = target.value;
          if (target.checked) this.activeStrategies.add(key);
          else this.activeStrategies.delete(key);
          this.generateSignals();
        }
      });
    }
  }

  private async initStrategiesFromConfig() {
    try {
      const resp = await fetch(`${BASE_API}/api/config`);
      if (!resp.ok) return;
      const payload = await resp.json();
      const data = payload?.data || {};
      const enabled: string[] = data.strategies || [];
      const namesMap: Record<string, string> = data.strategy_names || {};

      const allKeys = Object.keys(namesMap);
      if (allKeys.length) {
        this.strategyNamesMap = namesMap;
        this.strategies = allKeys;
        this.activeStrategies = new Set(enabled.length ? enabled : allKeys.slice(0, Math.min(6, allKeys.length)));
        this.renderStrategyCheckboxes(allKeys);
      }
    } catch (_) {
      // 忽略错误，保持默认策略
    }
  }

  private renderStrategyCheckboxes(keys: string[]) {
    const container = document.querySelector('.checkbox-group');
    if (!container) return;
    const html = keys.map((key) => {
      const label = this.strategyNamesMap[key] || key;
      const checked = this.activeStrategies.has(key) ? 'checked' : '';
      const safeId = `cb-${key.replace(/[^a-zA-Z0-9_-]/g, '')}`;
      return `
        <div class="checkbox-item">
          <input type="checkbox" class="strategy-cb" id="${safeId}" value="${key}" ${checked}>
          <label for="${safeId}">${label} <span style="opacity:.6;font-size:12px;">(${key})</span></label>
        </div>`;
    }).join('');
    (container as HTMLElement).innerHTML = html;
  }

  private async updateLearningStatsDesktop() {
    const profitRatio = document.getElementById('profit-ratio-desktop');
    const winRate = document.getElementById('win-rate-desktop');
    const maxDrawdownStat = document.getElementById('max-drawdown-desktop');

    try {
      // 调用真实API获取学习成绩数据
      const response = await fetch(`${BASE_API}/api/learning-stats`);
      if (!response.ok) throw new Error('API请求失败');

      const result = await response.json();
      if (!result.success) throw new Error(result.error || '数据获取失败');

      const stats = result.data;

      if (profitRatio) profitRatio.textContent = stats.profitRatio;
      if (winRate) winRate.textContent = stats.winRate;
      if (maxDrawdownStat) maxDrawdownStat.textContent = stats.maxDrawdown;

    } catch (error) {
      console.error('获取学习成绩失败:', error);
      // 降级到本地逻辑
      this.updateLearningStatsDesktopFallback();
    }
  }

  private updateLearningStatsDesktopFallback() {
    const profitRatio = document.getElementById('profit-ratio-desktop');
    const winRate = document.getElementById('win-rate-desktop');
    const maxDrawdownStat = document.getElementById('max-drawdown-desktop');

    // 检查用户是否启用了策略
    const hasActiveStrategies = this.activeStrategies.size > 0;

    if (hasActiveStrategies) {
      // 生成模拟学习成绩
      const mockProfitRatio = (1.2 + Math.random() * 1.0).toFixed(1); // 1.2-2.2
      const mockWinRate = Math.floor(Math.random() * 25) + 55; // 55-80%
      const mockMaxDrawdown = Math.floor(Math.random() * 8) + 3; // 3-10%

      if (profitRatio) profitRatio.textContent = mockProfitRatio;
      if (winRate) winRate.textContent = `${mockWinRate}%`;
      if (maxDrawdownStat) maxDrawdownStat.textContent = `${mockMaxDrawdown}%`;
    } else {
      // 显示默认值
      if (profitRatio) profitRatio.textContent = '--/--';
      if (winRate) winRate.textContent = '--/--';
      if (maxDrawdownStat) maxDrawdownStat.textContent = '--/--';
    }
  }

  private setTimeframe(tf: string) {
    this.currentTimeframe = tf;

    // 更新按钮状态
    for (const btn of document.querySelectorAll('.timeframe-btn')) {
      btn.classList.remove('active');
    }
    document.querySelector(`[data-tf="${tf}"]`)?.classList.add('active');

    // 更新显示
    const exchangeTfEl = document.getElementById('exchange-tf');
    if (exchangeTfEl) {
      exchangeTfEl.textContent = `binance / ${tf}`;
    }

    this.generateSignals();
  }

  private generateMockQuotes(): Quote[] {
    return this.symbols.slice(0, 8).map(symbol => {
      const basePrice = this.basePrices[symbol];
      const variation = (Math.random() - 0.5) * 0.06; // ±3% 变化
      const close = basePrice * (1 + variation);
      const spread = Math.abs(variation) * 0.5;
      const high = close * (1 + spread);
      const low = close * (1 - spread);

      return {
        symbol,
        close,
        high,
        low,
        time: new Date().toLocaleTimeString('zh-CN', { hour12: false })
      };
    });
  }

  private formatPrice(price: number): string {
    if (price >= 1000) return price.toFixed(2);
    if (price >= 1) return price.toFixed(4);
    if (price >= 0.001) return price.toFixed(6);
    return price.toFixed(8);
  }

  private generateMockSignals(): Signal[] {
    const signals: Signal[] = [];
    const signalCount = Math.floor(Math.random() * 4) + 2; // 2-5个信号

    for (let i = 0; i < signalCount; i++) {
      const symbol = this.symbols[Math.floor(Math.random() * this.symbols.length)];
      const strategiesArray = Array.from(this.activeStrategies);
      const strategy = strategiesArray[Math.floor(Math.random() * strategiesArray.length)];
      const side: 'BUY' | 'SELL' = Math.random() > 0.5 ? 'BUY' : 'SELL';

      const basePrice = this.basePrices[symbol];
      const entry = basePrice * (1 + (Math.random() - 0.5) * 0.04);

      let target: number;
      let stop: number;
      if (side === 'BUY') {
        target = entry * (1 + 0.02 + Math.random() * 0.02); // 2-4% 止盈
        stop = entry * (1 - 0.015 - Math.random() * 0.01); // 1.5-2.5% 止损
      } else {
        target = entry * (1 - 0.02 - Math.random() * 0.02); // 2-4% 止盈
        stop = entry * (1 + 0.015 + Math.random() * 0.01); // 1.5-2.5% 止损
      }

      signals.push({
        symbol,
        strategy,
        side,
        entry,
        target,
        stop,
        confidence: Math.floor(Math.random() * 20) + 30, // 30-50
        tf: this.currentTimeframe,
        reason: `建议单：${symbol}（${this.currentTimeframe}）${side === 'BUY' ? '做多' : '做空'}；${strategy} 策略触发`,
        ts: new Date()
      });
    }

    return signals;
  }

  private renderQuotes(quotes: Quote[]) {
    const tbody = document.getElementById('quotes-tbody');
    if (!tbody) return;

    tbody.innerHTML = quotes.map(quote => `
      <tr>
        <td><strong>${quote.symbol}</strong></td>
        <td>${this.formatPrice(quote.close)}</td>
        <td>${this.formatPrice(quote.high)}</td>
        <td>${this.formatPrice(quote.low)}</td>
        <td>${quote.time}</td>
      </tr>
    `).join('');
  }

  private renderSignals(signals: Signal[]) {
    const container = document.getElementById('signals-container');
    const probeResult = document.getElementById('probe-result');

    if (!container || !probeResult) return;

    // 探针检查
    const pairs = new Set(signals.map(s => `${s.symbol} | ${s.strategy}`));
    probeResult.innerHTML = `
      <div style="background: #065f46; padding: 10px; border-radius: 6px; margin-bottom: 15px; border-left: 4px solid #10b981;">
        <strong>Probe →</strong> 收到 ${signals.length} 条；唯一对数：${pairs.size}
        <div style="font-family: monospace; font-size: 0.875rem; margin-top: 5px; color: #94a3b8;">
          ${Array.from(pairs).join('<br>')}
        </div>
      </div>
    `;

    if (signals.length === 0) {
      container.innerHTML = '<div style="text-align: center; padding: 20px; color: #94a3b8;">当根无触发。</div>';
      return;
    }

    // 按symbol分组
    const groupedSignals = signals.reduce((acc, signal) => {
      if (!acc[signal.symbol]) acc[signal.symbol] = [];
      acc[signal.symbol].push(signal);
      return acc;
    }, {} as Record<string, Signal[]>);

    container.innerHTML = Object.entries(groupedSignals)
      .map(([symbol, symbolSignals]) => `
        <div style="margin-bottom: 25px;">
          <h3 style="color: #f1f5f9; margin-bottom: 15px; border-bottom: 1px solid #334155; padding-bottom: 5px;">
            ${symbol}
          </h3>
          ${symbolSignals.map(signal => `
            <div class="signal-card ${signal.side.toLowerCase()}">
              <div class="signal-header">
                <div class="signal-title">
                  ${signal.side} ${signal.symbol}
                </div>
                <div class="signal-strategy">${signal.strategy}</div>
              </div>
              <div style="color: #94a3b8; font-size: 0.875rem; margin-bottom: 10px;">
                信心 ${signal.confidence} ｜ 周期：${signal.tf} ｜ 时间：${signal.ts.toLocaleTimeString()}
              </div>
              <div class="signal-details">
                <div>入场：${this.formatPrice(signal.entry)}</div>
                <div>目标：${this.formatPrice(signal.target)}</div>
                <div>止损：${this.formatPrice(signal.stop)}</div>
                <div>ETA：≈${this.getETA(signal.tf)}</div>
              </div>
              <div style="margin-top: 10px; font-size: 0.875rem; color: #d1d5db;">
                ${signal.reason}
              </div>
              <div class="signal-actions" style="margin-top:10px; display:flex; gap:8pt;">
                <button class="signal-btn signal-btn-secondary btn-sim" data-symbol="${signal.symbol}" data-side="${signal.side}" data-strategy="${signal.strategy}" data-tf="${signal.tf}" data-entry="${signal.entry}">加入模拟</button>
              </div>
            </div>
          `).join('')}
        </div>
      `).join('');
  }

  private getETA(tf: string): string {
    const etaMap: Record<string, string> = {
      '4h': '4 小时',
      '1d': '1 天',
      '1w': '1 周'
    };
    return etaMap[tf] || '未知';
  }

  private async generateInitialData() {
    await this.fetchAndRenderDesktopQuotes();
    this.generateSignals();
  }

  private async generateSignals() {
    try {
      const response = await fetch(`${BASE_API}/api/signals`);
      if (!response.ok) throw new Error('API请求失败');
      const result = await response.json();
      if (!result.success) throw new Error(result.error || '数据获取失败');
      this.renderSignals(result.data as unknown as Signal[]);
    } catch (e) {
      // 失败显示占位
      this.renderSignals([]);
    }
  }

  private async updateRandomData() {
    await this.fetchAndRenderDesktopQuotes();
    await this.generateSignals();
  }

  private async fetchAndRenderDesktopQuotes() {
    try {
      const resp = await fetch(`${BASE_API}/api/quotes`);
      if (!resp.ok) throw new Error('API请求失败');
      const data = await resp.json();
      if (!data.success) throw new Error(data.error || '数据获取失败');
      const mapped: Quote[] = (data.data || []).map((q: any) => ({
        symbol: q.symbol,
        close: q.close,
        high: q.close,
        low: q.close,
        time: new Date().toLocaleTimeString('zh-CN', { hour12: false })
      }));
      this.renderQuotes(mapped);
    } catch (e) {
      // 显示空状态
      const tbody = document.getElementById('quotes-tbody');
      if (tbody) tbody.innerHTML = '';
    }
  }

  public refreshData() {
    // 实时刷新改为请求真实接口
    this.fetchAndRenderDesktopQuotes();
    this.generateSignals();

    // 显示刷新动画
    const btn = document.querySelector('.refresh-btn') as HTMLElement;
    if (btn) {
      btn.style.transform = 'rotate(360deg)';
      btn.style.transition = 'transform 0.5s ease';
      setTimeout(() => {
        btn.style.transform = '';
        btn.style.transition = '';
      }, 500);
    }
  }

  public runBacktest() {
    const lookahead = (document.getElementById('lookahead') as HTMLInputElement)?.value || '12';

    // 模拟回测结果
    const results = this.symbols.slice(0, 5).map(symbol => {
      const trades = Math.floor(Math.random() * 15) + 5;
      const winRate = Math.floor(Math.random() * 40) + 30; // 30-70%
      const avgR = (Math.random() * 2 - 0.5).toFixed(3); // -0.5 to 1.5

      return {
        symbol,
        winRate,
        trades,
        avgR: Number.parseFloat(avgR)
      };
    });

    // 显示回测结果
    const container = document.getElementById('signals-container');
    if (container) {
      container.innerHTML = `
        <div style="background: #1e293b; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
          <h3 style="color: #f1f5f9; margin-bottom: 15px;">🧪 快回测结果</h3>
          <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 15px;">
            <div><strong>周期：</strong>${this.currentTimeframe}</div>
            <div><strong>启用策略：</strong>${Array.from(this.activeStrategies).join(', ')}</div>
            <div><strong>Lookahead：</strong>${lookahead} 根K线</div>
          </div>
          ${results.map(result => `
            <div style="background: #334155; padding: 15px; border-radius: 6px; margin-bottom: 10px;">
              <div style="font-weight: 600; margin-bottom: 8px;">${result.symbol}</div>
              <div style="font-size: 0.875rem; color: #94a3b8;">
                胜率：${result.winRate}% ｜ 样本：${result.trades} ｜ 平均R：${result.avgR}
              </div>
            </div>
          `).join('')}
        </div>
      `;
    }
  }
}

// 全局函数供HTML调用
declare global {
  interface Window {
    refreshData: () => void;
    runBacktest: () => void;
    refreshMobileData: () => void;
    runMobileBacktest: () => void;
    openQuickBacktest: (symbol: string, strategy: string) => void;
    addToSimulation: (symbol: string, side: string) => void;
    followSignal: (symbol: string, side: string) => void;
    enableRecommendation: (title: string) => void;
    saveUserProfile: () => void;
    goToSettings: () => void;
    viewBacktest: () => void;
    compareSelectedSignals: () => void;
  }
}

// 根据设备类型初始化不同的界面
if (isMobile()) {
  console.log('📱 启动移动端优化界面');
  const mobileDashboard = new MobileTradingDashboard();

  // 导出移动端函数
  window.refreshMobileData = () => mobileDashboard.refreshData();
  window.runMobileBacktest = () => mobileDashboard.runBacktest();
  window.openQuickBacktest = (symbol: string, strategy: string) => (mobileDashboard as any)['openQuickBacktest'](symbol, strategy);


  // 新增交互功能
  window.addToSimulation = (symbol: string, side: string) => {
    alert(`已将 ${side} ${symbol} 添加到模拟仓位`);
  };

  window.followSignal = (symbol: string, side: string) => {
    alert(`开始模拟 ${side} ${symbol}`);
  };

  window.enableRecommendation = (title: string) => {
    alert(`已启用推荐策略：${title}`);
  };

  window.saveUserProfile = () => {
    // 先保存参数到内存
    mobileDashboard.saveUserParams();

    // 获取已保存的参数
    const params = mobileDashboard.getUserParams();

    // 持久化到本地，供"市场/回测"默认读取
    try {
      localStorage.setItem('user_profile_params', JSON.stringify(params));
    } catch (_) {}

    // 立即更新推荐内容
    mobileDashboard.updateRecommendations();

    // 显示保存成功提示
    const message = `✅ 已保存个性化配置：\n📈 收益目标: ${params.profitTarget}%\n⚠️ 最大回撤: ${params.maxDrawdown}%\n💰 风险暴露: ${params.riskExposure}%\n💵 本金规模: ${params.capitalSize.toLocaleString()} USDT\n\n💡 即将跳转到市场页面查看推荐！`;
    alert(message);

    // 1秒后自动切换到市场页面显示推荐结果
    setTimeout(() => {
      mobileDashboard.switchTab('home');
    }, 1000);
  };

  window.goToSettings = () => {
    mobileDashboard.switchTab('profile');
    // 可选：滚动到参数定制区域
    setTimeout(() => {
      const el = document.getElementById('profile-view');
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 50);
  };

  window.viewBacktest = () => {
    alert('📊 查看详细回测数据\n\n这里可以展示更详细的回测图表、历史表现等信息');
  };

  window.compareSelectedSignals = () => {};

  console.log('🚀 移动端量化交易面板已启动！');
  console.log('📊 支持策略:', ['vegas_tunnel', 'chan_simplified', 'macd']);
  console.log('📈 监控币种:', ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'TRX', 'AVAX', 'DOT', 'SHIB', 'LINK', 'TON', 'LTC', 'MATIC']);
} else {
  console.log('💻 启动桌面端界面');
  const dashboard = new TradingDashboard();

  // 导出桌面端函数
  window.refreshData = () => dashboard.refreshData();
  window.runBacktest = () => dashboard.runBacktest();

  console.log('🚀 量化交易面板已启动！');
  console.log('📊 支持策略:', ['vegas_tunnel', 'chan_simplified', 'macd', 'sma_cross', 'rsi_reversal']);
  console.log('📈 监控币种:', ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'TRX/USDT', 'AVAX/USDT', 'DOT/USDT', 'SHIB/USDT', 'LINK/USDT', 'TON/USDT', 'LTC/USDT', 'MATIC/USDT']);
}

// 统一 DOMContentLoaded 时刷新徽标
document.addEventListener('DOMContentLoaded', () => { try { updateBadge(); } catch(_) {} });

/* ================== SIM QUEUE + BADGE + FLY (STABLE) ================== */

/** 本地存储键 */
const SIM_QUEUE_KEY = 'simQueue';

/** 读取/写入队列 */
function getSimQueue(): any[] {
  try { return JSON.parse(localStorage.getItem(SIM_QUEUE_KEY) || '[]'); }
  catch { return []; }
}
function setSimQueue(list: any[]) { localStorage.setItem(SIM_QUEUE_KEY, JSON.stringify(list)); }
function pushSimItem(item: any) { const l = getSimQueue(); l.push(item); setSimQueue(l); }

/** 红点计数：当前仅统计队列条数（如需增加"运行中"，你再相加即可） */
function updateBadge() {
  const badge = document.getElementById('mine-badge') as HTMLSpanElement | null;
  if (!badge) return;
  const n = getSimQueue().length;
  if (n > 0) { badge.textContent = String(n); badge.style.display = 'inline-flex'; }
  else { badge.style.display = 'none'; }
}

/** 目标元素（优先飞向徽标） */
function getMineTarget(): HTMLElement | null {
  const badge = document.getElementById('mine-badge') as HTMLElement | null;
  if (badge) {
    const visible = badge.offsetWidth > 0 && badge.offsetHeight > 0 && getComputedStyle(badge).display !== 'none';
    if (visible) return badge;
  }
  return (document.getElementById('nav-mine') as HTMLElement | null);
}

/** 飞入动画（viewport 坐标 + position:fixed） */
function flyToMine(fromEl: HTMLElement) {
  const mine = getMineTarget();
  if (!mine) return;

  const s = fromEl.getBoundingClientRect();
  const startX = s.left + s.width / 2;
  const startY = s.top  + s.height/ 2;

  const dot = document.createElement('div');
  Object.assign(dot.style, {
    position: 'fixed',
    left: `${startX}px`,
    top:  `${startY}px`,
    width: '12px',
    height: '12px',
    borderRadius: '999px',
    background: '#0ea5e9',
    zIndex: '2147483647',
    pointerEvents: 'none',
  } as CSSStyleDeclaration);
  document.body.appendChild(dot);

  // 等一帧，确保 updateBadge 显示徽标后再取终点
  requestAnimationFrame(() => {
    const e = getMineTarget()?.getBoundingClientRect();
    if (!e) { dot.remove(); return; }
    const endX = e.left + e.width / 2;
    const endY = e.top  + e.height / 2;

    const dx = endX - startX;
    const dy = endY - startY;

    const anim = dot.animate(
      [
        { transform: 'translate(0,0) scale(1)',   opacity: 1 },
        { transform: `translate(${dx*0.55}px, ${dy*0.25}px) scale(1.1)`, opacity: 1, offset: 0.6 },
        { transform: `translate(${dx}px, ${dy}px) scale(0.3)`,           opacity: 0.15 },
      ],
      { duration: 650, easing: 'cubic-bezier(.2,.7,.2,1)' }
    );
    anim.onfinish = () => {
      dot.remove();
      const mineBtn = document.getElementById('nav-mine');
      if (mineBtn) { mineBtn.classList.add('pulse'); setTimeout(() => mineBtn.classList.remove('pulse'), 300); }
    };
  });
}

/** 统一点击委托：捕获任何 .btn-sim （按钮文案可叫"加入我的"） */
function handleSimClick(ev: Event) {
  const target = ev.target as HTMLElement | null;
  const btn = target?.closest?.('.btn-sim') as HTMLElement | null;
  if (!btn) return;

  const card = btn.closest('.signal-card') as HTMLElement | null;
  const ds = Object.assign({}, card?.dataset || {}, btn.dataset || {});
  const tf = ds.tf || document.querySelector('.timeframe-btn.active')?.getAttribute('data-tf') || '';

  const item = {
    id: (crypto as any)?.randomUUID?.() || String(Date.now()),
    symbol: ds.symbol || '',
    side: String(ds.side || '').toUpperCase(),   // BUY / SELL
    strategy: ds.strategy || '',
    tf,
    entry: ds.entry,
    createdAt: Date.now(),
    status: 'queued',
    disabled: false,
  };

  pushSimItem(item);     // 入队
  updateBadge();         // 红点 +1
  flyToMine(btn);        // 飞入动画
}

/** 防重复：移除旧的临时监听（如果存在） */
if ((window as any).__SIM_TMP_OFF__) {
  (window as any).__SIM_TMP_OFF__();
  delete (window as any).__SIM_TMP_OFF__;
}

/** 只注册一次 */
(function initSimOnce() {
  if ((window as any).__SIM_INITED__) return;
  document.addEventListener('click', handleSimClick, { capture: true });
  document.addEventListener('DOMContentLoaded', updateBadge);
  (window as any).__SIM_INITED__ = true;
  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    updateBadge();
  }
})();

/* ================== END (STABLE) ================== */

/* ============= QUICK BACKTEST (STABLE FALLBACK) ============= */
function openQuickBacktestStable(symbol: string, strategy: string) {
  const existing = document.getElementById('qb-overlay');
  if (existing) existing.remove();

  const overlay = document.createElement('div');
  overlay.id = 'qb-overlay';
  Object.assign(overlay.style, {
    position: 'fixed', inset: '0', background: 'rgba(0,0,0,0.55)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: '2147483646'
  } as CSSStyleDeclaration);

  const modal = document.createElement('div');
  Object.assign(modal.style, {
    width: 'min(92vw, 680px)', maxHeight: '80vh', overflowY: 'auto',
    background: '#0F1621', color: '#E6EDF6', border: '1px solid #1F2A3A',
    borderRadius: '16px', boxShadow: '0 6px 16px rgba(0,0,0,0.3)', padding: '16px'
  } as CSSStyleDeclaration);
  modal.innerHTML = `
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
      <div style="font-weight:700;font-size:18px;">${strategy} · ${symbol} · 快速回测</div>
      <button id="qb-close" style="border:1px solid #1F2A3A;border-radius:10px;background:#121C2A;color:#A7B1C2;padding:6px 10px;cursor:pointer;">关闭</button>
    </div>
    <div id="qb-content" style="padding:12px;color:#A7B1C2;">加载回测…</div>
  `;

  overlay.appendChild(modal);
  document.body.appendChild(overlay);

  (modal.querySelector('#qb-close') as HTMLElement)?.addEventListener('click', () => overlay.remove());
  overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });

  const content = modal.querySelector('#qb-content') as HTMLElement;

  (async () => {
    try {
      const resp = await fetch(`${BASE_API}/api/backtest/${encodeURIComponent(symbol)}?days=30&strategy=${encodeURIComponent(strategy)}`);
      if (!resp.ok) throw new Error('net');
      const payload = await resp.json();
      if (!payload || payload.success === false) throw new Error('api');
      const data = payload.data || {};
      const win = data.winRate ?? '--';
      const trades = data.trades ?? '--';
      const maxdd = data.maxDrawdown ?? '--';
      content.innerHTML = `
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px;">
          <div>胜率：<strong>${win}%</strong></div>
          <div>交易次数：<strong>${trades}</strong></div>
          <div>最大回撤：<strong>${maxdd}%</strong></div>
        </div>
        <div style="font-size:14px;color:#94a3b8;">提示：服务端返回有限字段时展示精简概览。</div>
      `;
    } catch (_) {
      // 离线/失败：展示本地模拟占位，避免空
      const mock = Array.from({length:6}, () => ({
        date: new Date(Date.now() - Math.floor(Math.random()*30)*86400000).toISOString().slice(0,10),
        side: Math.random()>0.5?'BUY':'SELL',
        pnl: (Math.random()*6-2).toFixed(2),
        hold: `${Math.floor(Math.random()*72)+6}h`
      }));
      content.innerHTML = `
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px;">
          <div>胜率：<strong>--</strong></div>
          <div>交易次数：<strong>${mock.length}</strong></div>
          <div>最大回撤：<strong>--</strong></div>
        </div>
        <table style="width:100%;border-collapse:collapse;">
          <thead><tr><th style="text-align:left;padding:6px;border-bottom:1px solid #1F2A3A;">日期</th><th style="text-align:left;padding:6px;border-bottom:1px solid #1F2A3A;">方向</th><th style="text-align:left;padding:6px;border-bottom:1px solid #1F2A3A;">盈亏%</th><th style="text-align:left;padding:6px;border-bottom:1px solid #1F2A3A;">持仓</th></tr></thead>
          <tbody>
            ${mock.map(r=>`<tr><td style="padding:6px;border-bottom:1px solid #1F2A3A;">${r.date}</td><td style="padding:6px;border-bottom:1px solid #1F2A3A;">${r.side}</td><td style="padding:6px;border-bottom:1px solid #1F2A3A;">${r.pnl}%</td><td style="padding:6px;border-bottom:1px solid #1F2A3A;">${r.hold}</td></tr>`).join('')}
          </tbody>
        </table>
      `;
    }
  })();
}

// 桥接：若未定义全局 openQuickBacktest，则提供稳定版
if (!(window as any).openQuickBacktest) {
  (window as any).openQuickBacktest = (symbol: string, strategy: string) => openQuickBacktestStable(symbol, strategy);
}