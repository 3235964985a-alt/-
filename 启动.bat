@echo off
chcp 65001 >nul
title 金融智能助手

echo.
echo  ========================================
echo    金融智能助手 - Financial AI Assistant
echo  ========================================
echo.

:: 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

:: 检查 .env 配置
if not exist ".env" (
    echo [提示] 未找到 .env 文件，正在从模板创建...
    copy .env.example .env >nul
    echo [提示] 请编辑 .env 填入你的 API Key 后重新运行
    pause
    exit /b 1
)

:: 安装依赖（如需要）
echo [1/2] 检查依赖...
pip install -r requirements.txt -q 2>nul
echo       依赖检查完成

:: 启动服务
echo [2/2] 启动服务...
echo.
echo   打开浏览器访问: http://localhost:8501
echo   按 Ctrl+C 停止服务
echo.
start http://localhost:8501
streamlit run app.py

pause
