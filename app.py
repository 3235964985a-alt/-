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

from src.agent import chat


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

    st.markdown("###  功能模块")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("  Agent", "4")
    with col2:
        st.metric("  Tools", "16")

    st.markdown("---")

    st.markdown("###  专业Agent")
    st.markdown("""
    - ** 股票数据专家** — 市值、调研信息
    - ** 财务分析师** — DCF估值、指标筛选
    - ** ESG评级专家** — 三大机构ESG评级
    - ** 通用助手** — 金融知识问答 + RAG
    """)

    st.markdown("---")

    st.markdown("### ✨ 技术栈")
    st.markdown("""
    - **LLM**: GPT-4o
    - **框架**: LangChain + LangGraph
    - **协议**: MCP (Model Context Protocol)
    - **知识库**: RAG + ChromaDB
    - **架构**: Supervisor-Worker 多智能体
    """)

    st.markdown("---")

    # 清空对话按钮
    if st.button(" 清空对话历史", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
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
                "general_agent": "badge-general",
            }.get(agent_name, "badge-general")

            agent_label = {
                "supervisor": "主管",
                "stock_agent": "股票数据",
                "analysis_agent": "财务分析",
                "esg_agent": "ESG评级",
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

if user_input:
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 调用后端Agent
    with st.chat_message("assistant"):
        with st.spinner("  思考中..."):
            try:
                result = chat(user_input, thread_id=st.session_state.thread_id)
                response_text = result.get("response", "抱歉，无法处理您的请求。")
                agent = result.get("agent", "general_agent")

                # Agent标签
                badge_class = {
                    "supervisor": "badge-supervisor",
                    "stock_agent": "badge-stock",
                    "analysis_agent": "badge-analysis",
                    "esg_agent": "badge-esg",
                    "general_agent": "badge-general",
                }.get(agent, "badge-general")

                agent_label = {
                    "supervisor": "主管",
                    "stock_agent": "股票数据",
                    "analysis_agent": "财务分析",
                    "esg_agent": "ESG评级",
                    "general_agent": "通用助手",
                }.get(agent, agent)

                st.markdown(
                    f'<span class="agent-badge {badge_class}">  {agent_label}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(response_text)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "agent": agent,
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
