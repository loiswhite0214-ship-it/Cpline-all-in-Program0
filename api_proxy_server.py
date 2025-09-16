#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API代理服务器 - 解决前端CORS问题
为前端提供数据代理服务，避免跨域限制
"""

import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 请求头配置
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# 代理配置（支持Clash等代理）
PROXY_CONFIG = {
    'http': 'http://127.0.0.1:7890',  # Clash默认端口
    'https': 'http://127.0.0.1:7890'
}

# 检查代理是否可用
def check_proxy():
    try:
        response = requests.get('http://httpbin.org/ip', proxies=PROXY_CONFIG, timeout=5)
        logger.info(f"Proxy working, external IP: {response.json().get('origin', 'unknown')}")
        return True
    except Exception as e:
        logger.warning(f"Proxy not working: {e}, using direct connection")
        return False

# 获取请求会话
def get_session():
    session = requests.Session()
    if check_proxy():
        session.proxies.update(PROXY_CONFIG)
    return session

@app.route('/api/coingecko/bitcoin/market_chart', methods=['GET'])
def get_bitcoin_market_chart():
    """获取BTC价格数据"""
    try:
        vs_currency = request.args.get('vs_currency', 'usd')
        days = request.args.get('days', '365')
        interval = request.args.get('interval', 'daily')
        
        url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {
            'vs_currency': vs_currency,
            'days': days,
            'interval': interval
        }
        
        logger.info(f"Fetching BTC data: {url} with params {params}")
        session = get_session()
        response = session.get(url, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"BTC data fetched successfully: {len(data.get('prices', []))} price points")
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error fetching BTC data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alternative/fng', methods=['GET'])
def get_fear_greed_index():
    """获取Fear & Greed指数"""
    try:
        limit = request.args.get('limit', '365')
        format_type = request.args.get('format', 'json')
        
        url = "https://api.alternative.me/fng/"
        params = {
            'limit': limit,
            'format': format_type
        }
        
        logger.info(f"Fetching Fear & Greed data: {url} with params {params}")
        response = requests.get(url, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"Fear & Greed data fetched successfully: {len(data.get('data', []))} records")
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error fetching Fear & Greed data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/etf/flows', methods=['GET'])
def get_etf_flows():
    """获取ETF资金流数据"""
    try:
        days = int(request.args.get('days', '365'))
        
        # 尝试多个ETF数据源
        urls = [
            'https://farside.co.uk/wp-content/uploads/bitcoin_etf_flows.csv',
            'https://www.farside.co.uk/wp-content/uploads/bitcoin_etf_flows.csv'
        ]
        
        for url in urls:
            try:
                logger.info(f"Fetching ETF flows from: {url}")
                response = requests.get(url, headers=HEADERS, timeout=30)
                response.raise_for_status()
                
                csv_data = response.text
                logger.info(f"ETF flows data fetched successfully: {len(csv_data)} characters")
                
                return jsonify({
                    'csvData': csv_data,
                    'source': url,
                    'days': days
                })
                
            except Exception as e:
                logger.warning(f"Failed to fetch from {url}: {e}")
                continue
        
        logger.error("All ETF data sources failed")
        return jsonify({'error': 'All ETF data sources failed'}), 500
        
    except Exception as e:
        logger.error(f"Error fetching ETF flows: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/binance/funding-rate', methods=['GET'])
def get_funding_rate():
    """获取Binance资金费率数据"""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        days = int(request.args.get('days', '365'))
        limit = int(request.args.get('limit', '1000'))
        
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        params = {
            'symbol': symbol,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        
        logger.info(f"Fetching funding rate data: {url} with params {params}")
        response = requests.get(url, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"Funding rate data fetched successfully: {len(data)} records")
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error fetching funding rate data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/fred/data', methods=['GET'])
def get_fred_data():
    """获取FRED宏观数据"""
    try:
        series_id = request.args.get('series_id')
        days = int(request.args.get('days', '365'))
        
        if not series_id:
            return jsonify({'error': 'series_id parameter is required'}), 400
        
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
        params = {'id': series_id}
        
        logger.info(f"Fetching FRED data for {series_id}: {url} with params {params}")
        response = requests.get(url, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        csv_data = response.text
        logger.info(f"FRED data for {series_id} fetched successfully: {len(csv_data)} characters")
        
        return jsonify({
            'csvData': csv_data,
            'seriesId': series_id,
            'days': days
        })
        
    except Exception as e:
        logger.error(f"Error fetching FRED data for {series_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/yahoo/<symbol>', methods=['GET'])
def get_yahoo_data(symbol):
    """获取Yahoo Finance数据"""
    try:
        days = int(request.args.get('days', '365'))
        
        # 计算时间范围
        end_time = int(time.time())
        start_time = end_time - (days * 24 * 60 * 60)
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            'period1': start_time,
            'period2': end_time,
            'interval': '1d',
            'includePrePost': 'false'
        }
        
        logger.info(f"Fetching Yahoo Finance data for {symbol}: {url} with params {params}")
        response = requests.get(url, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"Yahoo Finance data for {symbol} fetched successfully")
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/macro/btc_monitor', methods=['GET'])
def get_btc_macro_monitor():
    """获取BTC宏观监控数据（综合接口）"""
    try:
        days = int(request.args.get('days', '365'))
        
        # 并行获取多个数据源
        import concurrent.futures
        
        def fetch_btc_data():
            try:
                response = requests.get(f"http://localhost:5000/api/coingecko/bitcoin/market_chart?days={days}", timeout=30)
                return response.json() if response.status_code == 200 else None
            except:
                return None
        
        def fetch_fng_data():
            try:
                response = requests.get(f"http://localhost:5000/api/alternative/fng?limit={days}", timeout=30)
                return response.json() if response.status_code == 200 else None
            except:
                return None
        
        def fetch_etf_data():
            try:
                response = requests.get(f"http://localhost:5000/api/etf/flows?days={days}", timeout=30)
                return response.json() if response.status_code == 200 else None
            except:
                return None
        
        def fetch_funding_data():
            try:
                response = requests.get(f"http://localhost:5000/api/binance/funding-rate?days={days}", timeout=30)
                return response.json() if response.status_code == 200 else None
            except:
                return None
        
        def fetch_fred_data(series_id):
            try:
                response = requests.get(f"http://localhost:5000/api/fred/data?series_id={series_id}&days={days}", timeout=30)
                return response.json() if response.status_code == 200 else None
            except:
                return None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            btc_future = executor.submit(fetch_btc_data)
            fng_future = executor.submit(fetch_fng_data)
            etf_future = executor.submit(fetch_etf_data)
            funding_future = executor.submit(fetch_funding_data)
            rate_future = executor.submit(fetch_fred_data, 'DFF')
            cpi_future = executor.submit(fetch_fred_data, 'CPIAUCSL')
            unemp_future = executor.submit(fetch_fred_data, 'UNRATE')
            
            btc_data = btc_future.result()
            fng_data = fng_future.result()
            etf_data = etf_future.result()
            funding_data = funding_future.result()
            rate_data = rate_future.result()
            cpi_data = cpi_future.result()
            unemp_data = unemp_future.result()
        
        result = {
            'btc': btc_data,
            'fng': fng_data,
            'etf': etf_data,
            'funding': funding_data,
            'rate': rate_data,
            'cpi': cpi_data,
            'unemp': unemp_data,
            'days': days,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"BTC macro monitor data aggregated successfully for {days} days")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error aggregating BTC macro monitor data: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting API proxy server on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
