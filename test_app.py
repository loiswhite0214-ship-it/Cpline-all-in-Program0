import streamlit as st
st.caption("THIS IS test_app.py")

import streamlit as st
import pandas as pd
import numpy as np
import time

# 页面配置
st.set_page_config(page_title="测试应用", layout="wide")

# 标题
st.title("🚀 Streamlit 测试应用")
st.write("如果您能看到这个页面，说明应用运行正常！")

# 当前时间
st.subheader("⏰ 当前时间")
st.write(time.strftime("%Y-%m-%d %H:%M:%S"))

# 简单数据表
st.subheader("📊 测试数据")
data = pd.DataFrame({
    'Symbol': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
    'Price': [65000, 3200, 590],
    'Change': ['+2.5%', '-1.2%', '+0.8%']
})
st.dataframe(data, use_container_width=True)

# 交互按钮
if st.button("🔄 点击测试"):
    st.success("✅ 按钮功能正常！")
    st.balloons()

# 侧边栏
with st.sidebar:
    st.header("控制面板")
    test_option = st.selectbox("选择测试项", ["基础功能", "数据展示", "交互测试"])
    st.write(f"当前选择：{test_option}")

st.markdown("---")
st.caption("如果您看到这个页面，说明 Streamlit 应用已成功运行！")
