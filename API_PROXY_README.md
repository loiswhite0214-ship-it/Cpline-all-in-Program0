# API代理服务器使用说明

## 问题解决

您遇到的CORS（跨域资源共享）问题已经通过创建API代理服务器来解决。前端现在通过本地后端代理来获取外部API数据，避免了跨域限制。

## 启动步骤

### 1. 启动API代理服务器

```bash
# 方法1：使用批处理文件（推荐）
start_api_proxy.bat

# 方法2：直接运行Python脚本
python api_proxy_server.py
```

### 2. 启动前端开发服务器

```bash
# 在另一个终端窗口中
npm run dev
# 或者
yarn dev
```

## API端点说明

代理服务器运行在 `http://localhost:5000`，提供以下端点：

### 数据获取端点

| 端点 | 说明 | 参数 |
|------|------|------|
| `/api/coingecko/bitcoin/market_chart` | BTC价格数据 | `vs_currency`, `days`, `interval` |
| `/api/alternative/fng` | Fear & Greed指数 | `limit`, `format` |
| `/api/etf/flows` | ETF资金流数据 | `days` |
| `/api/binance/funding-rate` | 资金费率数据 | `symbol`, `days`, `limit` |
| `/api/fred/data` | FRED宏观数据 | `series_id`, `days` |
| `/api/yahoo/<symbol>` | Yahoo Finance数据 | `days` |

### 综合端点

| 端点 | 说明 | 参数 |
|------|------|------|
| `/api/macro/btc_monitor` | BTC宏观监控综合数据 | `days` |
| `/api/health` | 健康检查 | 无 |

## 数据源说明

### 支持的数据源

1. **CoinGecko API** - BTC价格数据
2. **Alternative.me API** - Fear & Greed指数
3. **Farside.co.uk** - ETF资金流数据
4. **Binance API** - 资金费率数据
5. **FRED (Federal Reserve)** - 宏观数据
   - DFF: 联邦基金利率
   - CPIAUCSL: CPI数据
   - UNRATE: 失业率
6. **Yahoo Finance** - 股票和商品数据

### 数据获取特点

- **并行获取**：多个数据源同时获取，提高效率
- **容错处理**：单个数据源失败不影响整体
- **缓存机制**：减少重复请求
- **超时控制**：30秒超时保护

## 故障排除

### 常见问题

1. **端口冲突**
   ```
   Error: Port 5000 is already in use
   ```
   解决方案：修改 `api_proxy_server.py` 中的端口号

2. **Python依赖缺失**
   ```
   ModuleNotFoundError: No module named 'flask'
   ```
   解决方案：运行 `pip install flask flask-cors requests`

3. **网络连接问题**
   ```
   ConnectionError: Failed to establish connection
   ```
   解决方案：检查网络连接和防火墙设置

### 调试模式

启用调试模式：
```bash
set DEBUG=True
python api_proxy_server.py
```

### 日志查看

服务器会输出详细的日志信息，包括：
- 请求URL和参数
- 响应状态和数据量
- 错误信息和堆栈跟踪

## 性能优化

### 建议配置

1. **并发限制**：默认最大6个并发请求
2. **超时设置**：30秒请求超时
3. **重试机制**：ETF数据源自动重试
4. **数据缓存**：前端可添加本地缓存

### 监控指标

- 请求响应时间
- 成功率统计
- 数据源可用性
- 内存使用情况

## 安全考虑

1. **CORS配置**：仅允许本地开发环境
2. **请求头伪装**：使用标准浏览器User-Agent
3. **错误信息**：不暴露敏感信息
4. **速率限制**：避免过度请求外部API

## 扩展功能

### 添加新的数据源

1. 在 `api_proxy_server.py` 中添加新的路由
2. 实现数据获取逻辑
3. 更新前端调用代码
4. 添加错误处理

### 数据预处理

可以在代理服务器中添加数据预处理逻辑：
- 数据清洗
- 格式转换
- 计算衍生指标
- 数据验证

## 联系支持

如果遇到问题，请检查：
1. Python版本（需要3.8+）
2. 网络连接状态
3. 防火墙设置
4. 端口占用情况

---

**注意**：此代理服务器仅用于开发环境，生产环境需要额外的安全配置。

