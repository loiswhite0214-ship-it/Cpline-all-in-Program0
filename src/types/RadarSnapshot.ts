// RadarSnapshot 组件接口定义

// 维度枚举
export type DimensionKey =
  'macro' | 'policy' | 'capital' | 'geopolitics' | 'onchain' | 'sentiment';

export type Subfactor = {
  key: string; 
  score: number | null; 
  weight: number;
  signal?: string; 
  notes?: string;
};

export type Dimension = {
  key: DimensionKey;
  score: number | null;     // 0..100 | null
  wow: number | null;       // 与上期差值（分）
  subfactors: Subfactor[];
};

export interface RadarSnapshotProps {
  dims: Dimension[];                // 当点 6 维
  timeISO: string;                  // 当前时间点
  selected?: DimensionKey | null;   // 高亮维度
  onToggleFactor?: (k: DimensionKey) => void;
  onHoverFactor?: (k: DimensionKey | null) => void;
}

// 事件总线类型定义
export interface FactorEventBus {
  emit: (event: string, data?: any) => void;
  on: (event: string, handler: (data?: any) => void) => void;
}

// 事件类型
export type FactorEventType = 
  | 'move:timestamp'    // 时间变动: { tsISO: string }
  | 'select:factor'     // 悬停/选中: { key: DimensionKey | null }
  | 'toggle:factor';    // 显隐/高亮: { key: DimensionKey }

// ECharts 雷达配置类型
export interface RadarConfig {
  indicator: Array<{
    name: string;
    max: number;
  }>;
  splitNumber: number;
  axisName: {
    formatter: (name: string) => string;
    rich: {
      up: { color: string; fontWeight: string };
      down: { color: string; fontWeight: string };
      flat: { color: string };
    };
  };
  splitLine: { lineStyle: { opacity: number } };
  splitArea: { show: boolean };
  axisLine: { lineStyle: { color: string } };
}

// 雷达系列配置
export interface RadarSeries {
  type: 'radar';
  name: string;
  data: Array<{
    value: number[];
    name: string;
    areaStyle?: { opacity: number };
    lineStyle?: { width: number; color?: string };
    symbol?: string;
    symbolSize?: number;
    itemStyle?: any;
    emphasis?: any;
  }>;
  animationDuration?: number;
  animationDurationUpdate?: number;
  silent?: boolean;
  tooltip?: { show: boolean };
  z?: number;
}
