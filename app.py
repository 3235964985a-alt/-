"""
金融智能助手 - Streamlit Web 界面

基于 LangGraph 多智能体系统的对话应用。
"""

import streamlit as st
import sys
import os
import uuid
import traceback
from datetime import datetime

# 添加src到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent import chat, chat_stream, analyze_watchlist, get_market_overview


# ---------- 页面配置 ----------
st.set_page_config(
    page_title="金融智能助手",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------- CSS 样式 ----------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .main-header p {
        color: #666;
        font-size: 1rem;
    }
    .agent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-supervisor { background: #e0e7ff; color: #3730a3; }
    .badge-stock { background: #d1fae5; color: #065f46; }
    .badge-analysis { background: #fef3c7; color: #92400e; }
    .badge-esg { background: #ede9fe; color: #5b21b6; }
    .badge-news { background: #fce7f3; color: #9d174d; }
    .badge-general { background: #e5e7eb; color: #374151; }
    .badge-tool { background: #fce7f3; color: #9d174d; }
    .chat-message {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
    }
    .user-message {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
    }
    .assistant-message {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
    }
    .info-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.8rem 0;
        font-size: 0.85rem;
    }
    .stTextInput > div > div > input {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------- 侧边栏 ----------
with st.sidebar:
    st.markdown("##  金融智能助手")
    st.markdown("---")

    st.markdown("###  系统概况")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("  Agent", "5")
    with col2:
        st.metric("  Tools", "23")
    with col3:
        st.metric("  数据源", "4")

    st.markdown("---")

    st.markdown("###  多智能体架构")
    st.markdown("""
| Agent | 职能 | 工具数 |
|---|---|---|
| ** 股票数据** | 个股市值、收盘价、调研 | 4 |
| ** 财务分析** | DCF估值、ROE/ROIC/毛利率/净利率/股息率筛选 | 10 |
| ** ESG评级** | 妙盈科技、华证指数、商道融绿三大机构评级 | 3 |
| ** 财经新闻** | 财联社/雪球/华尔街见闻/知乎/微博等14+源，含情绪评分 | 5 |
| ** 通用助手** | 金融知识问答 + RAG知识库 | 1 |
    """)

    st.markdown("---")

    st.markdown("###  数据源")
    st.markdown("""
| 来源 | 协议 | 内容 |
|---|---|---|
| **证券之星** | MCP (SSE) | 股票市值、DCF估值、ESG评级、筛选指标 |
| **东方财富** | AKShare HTTP | 概念板块(494) + 行业板块(496) 实时行情 |
| **NewsNow** | HTTP API | 14+新闻源实时热点 + LLM情绪评分 |
| **ChromaDB** | 本地 | RAG金融知识库 |
    """)

    st.markdown("---")

    st.markdown("###  特色功能")
    st.markdown("""
-   大盘综合报告（龙头股 + 板块 + 新闻三路聚合）
-   板块轮动分析（概念/行业涨跌TOP5）
-   新闻情绪评分（三源LLM打分，-100~100）
-   持仓截图OCR识别（EasyOCR + LLM解析）
-   自选股多Agent协作（news→stock→analysis→申论）
-   流式逐字输出
    """)

    st.markdown("---")

    st.markdown("### ✨ 技术栈")
    st.markdown("""
    - **LLM**: GPT-4o
    - **框架**: LangChain + LangGraph
    - **协议**: MCP (Model Context Protocol)
    - **知识库**: RAG + ChromaDB
    - **识别**: EasyOCR 持股截图解析
    - **架构**: Supervisor-Worker 多智能体
    """)

    st.markdown("---")

    # 清空对话按钮
    if st.button(" 清空对话历史", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    # 大盘概览快捷按钮
    st.markdown("---")
    if st.button("  大盘概览", type="primary", use_container_width=True, key="market_overview_btn",
                 help="一键获取今日A股核心龙头行情概览"):
        st.session_state.trigger_market_overview = True
        st.rerun()

    st.markdown("---")
    st.caption("© 2025 课程设计项目 | 金融智能对话系统")


# ---------- 主体 ----------
st.markdown("""
<div class="main-header">
    <h1>  金融智能助手</h1>
    <p>基于 LangGraph 多智能体协同的金融对话系统</p>
</div>
""", unsafe_allow_html=True)

# 会话初始化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# 欢迎消息
if not st.session_state.messages:
    st.markdown("""
    <div class="info-box">
        <strong> 欢迎使用金融智能助手！</strong><br>
        我可以帮你：<br>
        • 查询龙头行情大盘概览<br>
        • 查询股票市值和机构调研信息<br>
        • 进行DCF估值诊断和综合评估<br>
        • 查询ESG评级（妙盈科技/华证指数/商道融绿）<br>
        • 按ROE/ROIC/毛利率/净利率/股息率筛选优质股票<br>
        • 回答金融投资知识问题<br><br>
        <em> 试试输入：<code>帮我分析一下600519的估值</code> 或 <code>查询600519的ESG评级</code></em>
    </div>
    """, unsafe_allow_html=True)

# 快捷问题
if len(st.session_state.messages) == 0:
    st.markdown("#####  快捷提问")
    cols = st.columns(3)
    with cols[0]:
        if st.button("  分析茅台估值", use_container_width=True):
            st.session_state.quick_msg = "帮我分析一下600519（贵州茅台）的估值"
            st.rerun()
    with cols[1]:
        if st.button("  查看ESG评级", use_container_width=True):
            st.session_state.quick_msg = "查询600036（招商银行）的ESG评级"
            st.rerun()
    with cols[2]:
        if st.button("  筛选高ROE股", use_container_width=True):
            st.session_state.quick_msg = "帮我筛选ROE较高的股票"
            st.rerun()

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Agent标签
        agent_name = msg.get("agent", "")
        if agent_name and msg["role"] == "assistant":
            badge_class = {
                "supervisor": "badge-supervisor",
                "stock_agent": "badge-stock",
                "analysis_agent": "badge-analysis",
                "esg_agent": "badge-esg",
                "news_agent": "badge-news",
                "general_agent": "badge-general",
            }.get(agent_name, "badge-general")

            agent_label = {
                "supervisor": "主管",
                "stock_agent": "股票数据",
                "analysis_agent": "财务分析",
                "esg_agent": "ESG评级",
                "news_agent": "财经新闻",
                "general_agent": "通用助手",
            }.get(agent_name, agent_name)

            st.markdown(
                f'<span class="agent-badge {badge_class}">  {agent_label}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])

# 输入框
user_input = st.chat_input("请输入您的问题...", key="user_input")

# 处理快捷消息
if "quick_msg" in st.session_state and st.session_state.quick_msg:
    user_input = st.session_state.quick_msg
    st.session_state.quick_msg = ""

# 处理自选股分析
if st.session_state.get("trigger_analysis"):
    st.session_state.trigger_analysis = False
    all_stocks = []
    for entries in st.session_state.watchlist.values():
        for e in entries:
            all_stocks.append(e[0] if isinstance(e, list) else e)

    if all_stocks:
        with st.chat_message("user"):
            st.markdown(f"  分析自选股（{len(all_stocks)}只）：{', '.join(all_stocks)}")
        st.session_state.messages.append({
            "role": "user",
            "content": f"分析自选股：{', '.join(all_stocks)}",
        })

        with st.chat_message("assistant"):
            with st.spinner(f"  正在分析 {len(all_stocks)} 只自选股（市值+估值+ESG）..."):
                try:
                    report = analyze_watchlist(all_stocks)
                    st.markdown('<span class="agent-badge badge-analysis">  自选股报告</span>', unsafe_allow_html=True)
                    st.markdown(report)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": report,
                        "agent": "analysis_agent",
                    })
                except Exception as e:
                    error_msg = f"抱歉，分析出错：{str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "agent": "error",
                    })

# 处理 OCR 持仓分析
if st.session_state.get("trigger_ocr_analysis"):
    st.session_state.trigger_ocr_analysis = False
    codes = st.session_state.get("ocr_codes", [])
    if codes:
        code_list = ", ".join(codes)
        with st.chat_message("user"):
            st.markdown(f"  持仓截图识别（{len(codes)}只）：{code_list}")
        st.session_state.messages.append({
            "role": "user",
            "content": f"分析持仓：{code_list}",
        })

        with st.chat_message("assistant"):
            with st.spinner(f"  正在分析 {len(codes)} 只持仓股（市值+估值+ESG）..."):
                try:
                    report = analyze_watchlist(codes)
                    st.markdown('<span class="agent-badge badge-analysis">  持仓分析报告</span>', unsafe_allow_html=True)
                    st.markdown(report)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": report,
                        "agent": "analysis_agent",
                    })
                except Exception as e:
                    error_msg = f"抱歉，持仓分析出错：{str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "agent": "error",
                    })

# 处理大盘概览
if st.session_state.get("trigger_market_overview"):
    st.session_state.trigger_market_overview = False
    with st.chat_message("user"):
        st.markdown("  今日大盘概览")
    st.session_state.messages.append({"role": "user", "content": "大盘概览"})

    with st.chat_message("assistant"):
        with st.spinner("  查询核心龙头行情..."):
            try:
                overview = get_market_overview()
                st.markdown('<span class="agent-badge badge-stock">  大盘概览</span>', unsafe_allow_html=True)
                st.markdown(overview)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": overview,
                    "agent": "stock_agent",
                })
            except Exception as e:
                error_msg = f"抱歉，获取大盘数据失败：{str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "agent": "error",
                })

if user_input:
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 调用后端Agent（流式）
    with st.chat_message("assistant"):
        with st.spinner("  思考中..."):
            try:
                # 流式输出
                response_placeholder = st.empty()
                current_agent = "general_agent"
                streamed_text = ""

                for chunk in chat_stream(user_input, thread_id=st.session_state.thread_id):
                    if chunk["type"] == "agent":
                        current_agent = chunk["name"]

                    elif chunk["type"] == "content":
                        streamed_text += chunk["text"]
                        badge_class = {
                            "supervisor": "badge-supervisor",
                            "stock_agent": "badge-stock",
                            "analysis_agent": "badge-analysis",
                            "esg_agent": "badge-esg",
                            "news_agent": "badge-news",
                            "finish_agent": "badge-general",
                            "general_agent": "badge-general",
                        }.get(current_agent, "badge-general")

                        agent_label = {
                            "supervisor": "主管",
                            "stock_agent": "股票数据",
                            "analysis_agent": "财务分析",
                            "esg_agent": "ESG评级",
                            "finish_agent": "综合报告",
                            "news_agent": "财经新闻",
                            "general_agent": "通用助手",
                        }.get(current_agent, current_agent)

                        response_placeholder.markdown(
                            f'<span class="agent-badge {badge_class}">  {agent_label}</span>\n\n{streamed_text}',
                            unsafe_allow_html=True,
                        )

                    elif chunk["type"] == "done":
                        response_text = chunk.get("response", streamed_text)
                        final_agent = chunk.get("agent", current_agent)

                        badge_class = {
                            "supervisor": "badge-supervisor",
                            "stock_agent": "badge-stock",
                            "analysis_agent": "badge-analysis",
                            "esg_agent": "badge-esg",
                            "news_agent": "badge-news",
                            "finish_agent": "badge-general",
                            "general_agent": "badge-general",
                        }.get(final_agent, "badge-general")

                        agent_label = {
                            "supervisor": "主管",
                            "stock_agent": "股票数据",
                            "analysis_agent": "财务分析",
                            "esg_agent": "ESG评级",
                            "finish_agent": "综合报告",
                            "news_agent": "财经新闻",
                            "general_agent": "通用助手",
                        }.get(final_agent, final_agent)

                        response_placeholder.markdown(
                            f'<span class="agent-badge {badge_class}">  {agent_label}</span>\n\n{response_text}',
                            unsafe_allow_html=True,
                        )

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                            "agent": final_agent,
                        })

            except Exception as e:
                error_msg = f" 抱歉，系统出错了：{str(e)}"
                st.error(error_msg)
                traceback.print_exc()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "agent": "error",
                })
