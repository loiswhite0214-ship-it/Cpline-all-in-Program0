#!/bin/bash

# 熬鹰计划 iOS App 构建脚本
echo "🚀 开始构建熬鹰计划 iOS App..."

# 检查必要工具
echo "📋 检查构建环境..."
if ! command -v npm &> /dev/null; then
    echo "❌ 错误: npm 未安装"
    exit 1
fi

if ! command -v npx &> /dev/null; then
    echo "❌ 错误: npx 未安装"
    exit 1
fi

# 安装依赖
echo "📦 安装依赖包..."
npm install

# 构建Web版本
echo "🔨 构建Web应用..."
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Web构建失败"
    exit 1
fi

# 同步到iOS
echo "📱 同步到iOS平台..."
npx cap sync ios

if [ $? -ne 0 ]; then
    echo "❌ iOS同步失败"
    exit 1
fi

# 检查iOS项目
if [ ! -d "ios/App" ]; then
    echo "❌ iOS项目不存在，正在创建..."
    npx cap add ios
fi

echo "✅ 构建完成！"
echo ""
echo "📖 下一步操作："
echo "1. 打开Xcode项目: npx cap open ios"
echo "2. 在Xcode中配置签名和Bundle ID"
echo "3. 连接设备并测试应用"
echo "4. 创建Archive用于App Store上架"
echo ""
echo "📁 项目文件位置:"
echo "   - iOS项目: ./ios/App/App.xcworkspace"
echo "   - Web构建: ./dist/"
echo "   - 配置文件: ./capacitor.config.ts"
echo ""
echo "🔗 有用链接:"
echo "   - Capacitor文档: https://capacitorjs.com/docs/ios"
echo "   - App Store指南: https://developer.apple.com/app-store/review/guidelines/"
echo "   - Xcode发布指南: https://developer.apple.com/documentation/xcode/distributing-your-app-for-beta-testing-and-releases"

# 可选：自动打开Xcode（如果在macOS上）
if [[ "$OSTYPE" == "darwin"* ]]; then
    read -p "🍎 是否现在打开Xcode项目? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔧 正在打开Xcode..."
        npx cap open ios
    fi
fi

echo "🎉 熬鹰计划 iOS App 准备就绪！"
