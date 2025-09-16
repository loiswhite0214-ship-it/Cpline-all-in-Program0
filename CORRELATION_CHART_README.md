# 前端计算30D秩相关图表

## 功能概述

这个功能实现了从前端拉取 `/api/factors/history` 数据，然后在前端计算30日滚动秩相关（Spearman），并生成同轴多曲线的ECharts配置。

## 主要特性

### 1. 前端计算
- **完全前端计算**：不修改后端，所有相关性计算在前端完成
- **30日滚动窗口**：使用30天滚动窗口计算Spearman秩相关
- **平滑处理**：3日EMA轻平滑，7日EMA去台阶
- **限幅保护**：±95%限幅，消除"满屏针"现象

### 2. 数据处理
- **低频插值**：自动检测低频数据并进行日频插值
- **统一口径**：价格类走对数收益，其他走一阶差分
- **低方差保护**：避免低方差情况下的计算错误

### 3. 视觉效果
- **Web3风格**：3色相配色方案（青色、紫色、薄荷绿）
- **发光效果**：线条带有发光阴影效果
- **能量带**：底部显示负相关能量带
- **扫描线**：动态扫描线效果

## 文件结构

```
src/
├── corr_frontonly.ts    # 前端计算逻辑
└── main.ts             # 主界面集成
```

## 使用方法

### 1. 在页面中添加容器

```html
<div class="chart-stage">
  <div id="corrChart"></div>
</div>
```

### 2. 在组件中调用

```typescript
import { renderCorrChart } from './corr_frontonly';

// 在组件mounted/useEffect中调用
useEffect(() => {
  renderCorrChart();
}, []);
```

### 3. CSS样式

```css
.chart-stage{ 
  position: relative; 
  min-width: 0; 
}

#corrChart{
  width: 100%; 
  height: 380px; 
  box-sizing: border-box;
  overflow: hidden; 
  background: #0B0F14;
}
```

## 配置参数

```typescript
const WIN = 30;                    // 滚动窗口天数
const LOW_VAR_EPS = 1e-8;          // 低方差保护阈值
const EMA_SOFT = 3;                // 轻平滑
const INTERP_SPAN = 7;             // 低频插值 & EMA 去台阶
```

## 支持的因子

- **macro**: 宏观因子
- **policy**: 政策因子  
- **capital**: 资本因子
- **geopolitics**: 地缘政治因子
- **onchain**: 链上因子
- **sentiment**: 情绪因子

## 技术实现

### 1. 数据获取
- 使用 `/api/factors/history` 端点获取历史数据
- 支持90天历史数据
- 自动处理API响应格式

### 2. 相关性计算
- **Spearman秩相关**：对台阶数据更稳定
- **滚动窗口**：30天滚动计算
- **平滑处理**：EMA平滑消除噪声

### 3. 图表渲染
- **ECharts配置**：完整的Web3风格配置
- **响应式设计**：自动适应容器尺寸
- **交互功能**：缩放、图例、工具提示

## 错误处理

- API请求失败时显示错误信息
- 数据格式错误时自动降级
- 容器尺寸检查，避免0宽高问题

## 性能优化

- **去抖处理**：resize事件去抖
- **渐进渲染**：大数据集渐进加载
- **内存管理**：自动清理事件监听器

## 注意事项

1. 确保ECharts已正确加载
2. 容器必须有明确的尺寸
3. API端点必须可访问
4. 建议在组件卸载时清理资源

## 扩展性

- 可以轻松添加新的因子
- 支持自定义颜色方案
- 可以调整计算参数
- 支持不同的时间窗口
