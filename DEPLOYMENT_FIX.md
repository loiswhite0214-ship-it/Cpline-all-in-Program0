# 部署修复说明

## 问题描述
- Python 3.13 环境 + 老版本依赖导致兼容性问题
- pandas-ta==0.3.14b0 安装失败
- numpy/pandas 需要从源码编译，但 Python 3.13 移除了 distutils

## 解决方案

### 1. Python 版本降级
- 创建了 `runtime.txt` 指定 Python 3.11.9
- Streamlit Cloud 官方支持，有现成的 numpy/pandas 轮子

### 2. 依赖版本更新
- 更新 `requirements.txt` 使用兼容版本：
  - numpy==1.26.4 (有 Py3.11 轮子)
  - pandas==2.1.4 (有 Py3.11 轮子)
  - pandas-ta==0.3.14b0 (优先尝试 PyPI)

### 3. 备用方案
- 创建了 `requirements-git.txt` 作为备用
- 如果 PyPI 的 pandas-ta 不可用，使用 Git 源：
  ```
  git+https://github.com/twopirllc/pandas-ta@main#egg=pandas-ta
  ```

### 4. Streamlit 配置
- 添加了 `.streamlit/config.toml` 优化部署配置
- 启用 headless 模式，禁用 CORS/XSRF 保护

## 使用方法

### 方案A（推荐）
直接使用更新后的 `requirements.txt`

### 方案B（备用）
如果 pandas-ta 安装失败，重命名文件：
```bash
mv requirements.txt requirements-pypi.txt
mv requirements-git.txt requirements.txt
```

## 验证
部署成功后，pandas-ta 的导入方式不变：
```python
import pandas_ta as ta
```

