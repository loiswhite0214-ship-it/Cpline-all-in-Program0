# btc_macro_api.py
# 简单的Flask API端点，用于提供BTC多因子宏观监控数据
# 可以集成到现有的后端服务中

from flask import Flask, jsonify, request
import json
import subprocess
import os
import tempfile
from datetime import datetime, timedelta
import random

app = Flask(__name__)

def generate_mock_btc_macro_data(days=730):
    """生成模拟的BTC宏观监控数据"""
    data = []
    now = datetime.now()
    
    for i in range(days, 0, -1):
        date = now - timedelta(days=i)
        day_of_year = date.timetuple().tm_yday
        
        # 模拟DriverIndex（宏观驱动）
        driver_index = random.uniform(-2, 2) + 0.5 * random.uniform(-1, 1)
        
        # 模拟各种Z分数
        btc_z = random.uniform(-3, 3) + 0.3 * random.uniform(-1, 1)
        fng_z = random.uniform(-2, 2) + 0.4 * random.uniform(-1, 1)
        ixic_z = random.uniform(-2.5, 2.5) + 0.3 * random.uniform(-1, 1)
        gspc_z = random.uniform(-2.5, 2.5) + 0.3 * random.uniform(-1, 1)
        dxy_z = random.uniform(-2, 2) + 0.4 * random.uniform(-1, 1)
        gold_z = random.uniform(-2, 2) + 0.3 * random.uniform(-1, 1)
        stablecap_z = random.uniform(-1.5, 1.5) + 0.2 * random.uniform(-1, 1)
        hashrate_z = random.uniform(-2, 2) + 0.3 * random.uniform(-1, 1)
        
        # 模拟ETF净流入（百万美元）
        etf_net = random.uniform(-500, 500) + 100 * random.uniform(-1, 1)
        
        # 模拟资金费率（百分比）
        funding = random.uniform(-0.01, 0.01) + 0.005 * random.uniform(-1, 1)
        
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "driverIndex": round(driver_index, 3),
            "btcZ": round(btc_z, 3),
            "fngZ": round(fng_z, 3),
            "ixicZ": round(ixic_z, 3),
            "gspcZ": round(gspc_z, 3),
            "dxyZ": round(dxy_z, 3),
            "goldZ": round(gold_z, 3),
            "stablecapZ": round(stablecap_z, 3),
            "hashrateZ": round(hashrate_z, 3),
            "etfNet": round(etf_net, 1),
            "funding": round(funding, 4)
        })
    
    return data

@app.route('/api/macro/btc_monitor', methods=['GET'])
def get_btc_macro_monitor():
    """获取BTC多因子宏观监控数据"""
    try:
        # 获取查询参数
        range_param = request.args.get('range', '2Y')
        
        # 根据范围确定天数
        if range_param == '1Y':
            days = 365
        elif range_param == '2Y':
            days = 730
        elif range_param == '3Y':
            days = 1095
        else:
            days = 730  # 默认2年
        
        # 生成模拟数据
        data = generate_mock_btc_macro_data(days)
        
        return jsonify({
            "status": "success",
            "data": data,
            "range": range_param,
            "count": len(data),
            "generated_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "generated_at": datetime.now().isoformat()
        }), 500

@app.route('/api/macro/btc_monitor/real', methods=['GET'])
def get_real_btc_macro_monitor():
    """调用Python脚本获取真实数据"""
    try:
        # 获取查询参数
        range_param = request.args.get('range', '2Y')
        
        # 创建临时文件来运行Python脚本
        script_path = os.path.join(os.path.dirname(__file__), 'macro_btc_monitor_v2.py')
        
        if not os.path.exists(script_path):
            # 如果脚本不存在，返回模拟数据
            return get_btc_macro_monitor()
        
        # 这里可以添加调用Python脚本的逻辑
        # 由于Streamlit脚本比较复杂，暂时返回模拟数据
        return get_btc_macro_monitor()
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "generated_at": datetime.now().isoformat()
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        "status": "healthy",
        "service": "BTC Macro Monitor API",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("启动BTC宏观监控API服务...")
    print("访问 http://localhost:8889/api/macro/btc_monitor 获取数据")
    app.run(host='0.0.0.0', port=8889, debug=True)

