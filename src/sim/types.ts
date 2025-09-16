export type Side = "BUY" | "SELL";
export type TF = "4h" | "1d" | "1w" | string;

export interface SimItem {
  id: string;
  symbol: string;
  side: Side;
  strategy: string;
  tf: TF;
  entry?: string;
  createdAt: number;
  disabled?: boolean;   // 停止后续启用
  status: "queued";
}

export interface SimPosition {
  posId: string;
  fromSimId: string;
  symbol: string;
  side: Side;
  strategy: string;
  tf: TF;

  qty: number;          // 手数（默认 1）
  avgEntry: number;     // 开仓均价
  openTime: number;

  lastPrice?: number;   // 实时更新用
  pnl?: number;         // 浮盈亏
  pnlPct?: number;

  status: "open" | "closed";
  closeTime?: number;
  closePrice?: number;
  pnlFinal?: number;
  notes?: string;
}

export interface SimPrefs {
  defaultQty: number;
  autoOpenOnEnableAll: boolean;
  priceSource: "quotesBus" | "polling" | "none";
}

