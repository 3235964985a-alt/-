"""
LangGraph 多智能体系统

架构：Supervisor-Worker 模式
    Supervisor（主管）→ 分析用户意图，分派给专业Agent
        ├── StockAgent（股票数据专家）
        ├── AnalysisAgent（财务分析师）
        ├── ESGAgent（ESG评级专家）
        └── GeneralAgent（通用助手 + RAG）

使用 LangGraph 的 StateGraph 构建，支持流式输出。
"""

import logging
import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from .mcp_tools import STOCK_TOOLS, ANALYSIS_TOOLS, ESG_TOOLS, TOOL_MAP, _call_mcp_tool_sync
from .news_mcp import NEWS_TOOLS, NEWS_TOOL_MAP
from .sector_data import get_sector_overview_text
from .data_layer import safe_json, fmt_cap
from .prompts import (
    SUPERVISOR_PROMPT,
    STOCK_AGENT_PROMPT,
    ANALYSIS_AGENT_PROMPT,
    ESG_AGENT_PROMPT,
    GENERAL_AGENT_PROMPT,
    NEWS_AGENT_PROMPT,
)
from .rag import retrieve_knowledge_as_context

# 合并工具映射
_ALL_TOOL_MAP = {**TOOL_MAP, **NEWS_TOOL_MAP}

logger = logging.getLogger(__name__)

# ---------- Agent节点标识 ----------
AGENT_NODES = {
    "supervisor": "supervisor",
    "stock_agent": "stock_agent",
    "analysis_agent": "analysis_agent",
    "esg_agent": "esg_agent",
    "news_agent": "news_agent",
    "finish_agent": "finish_agent",
    "general_agent": "general_agent",
}

# ---------- State 定义 ----------

class AgentState(TypedDict):
    """多智能体系统的全局状态"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    round_count: int
    visited_agents: List[str]
    analysis_chain: str  # "active"=正在执行综合分析链, ""=普通模式


# ---------- LLM 工厂 ----------

def _create_llm(temperature: float = 0.3) -> ChatOpenAI:
    kwargs = {
        "model": OPENAI_MODEL,
        "temperature": temperature,
    }
    if OPENAI_API_KEY:
        kwargs["openai_api_key"] = OPENAI_API_KEY
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return ChatOpenAI(**kwargs)


def _is_comprehensive_intent(decision: str) -> bool:
    """检测 Supervisor 决策文本中是否包含综合分析意图"""
    keywords = ["分析", "报告", "诊断", "怎么看", "全面", "综合", "buy", "sell", "买入", "卖出"]
    has_keyword = any(k in decision for k in keywords)
    # 如果 Supervisor 提到多个 agent → 综合分析
    agent_mentions = sum(1 for a in ["stock_agent", "analysis_agent", "esg_agent", "news_agent"] if a in decision)
    return has_keyword or agent_mentions >= 2


# ---------- Supervisor ----------

def create_supervisor_node():
    """Supervisor节点：分析用户意图，决定下一个Agent"""

    llm = _create_llm(temperature=0.3)

    def supervisor_node(state: AgentState) -> Dict[str, Any]:
        messages = list(state.get("messages", []))
        round_count = state.get("round_count", 0)
        visited = list(state.get("visited_agents", []))
        analysis_chain = state.get("analysis_chain", "")

        # 综合分析链在跑 → 强制按序路由
        CHAIN_ORDER = ["stock_agent", "analysis_agent", "esg_agent", "news_agent"]
        if analysis_chain == "active":
            for next_agent in CHAIN_ORDER:
                if next_agent not in visited:
                    logger.info(f"综合分析链 → {next_agent}")
                    return {"next_agent": next_agent, "round_count": round_count + 1, "visited_agents": visited, "analysis_chain": "active"}
            # 所有Agent都跑完了
            logger.info("综合分析链完成 → finish_agent")
            return {"next_agent": "finish_agent", "round_count": round_count + 1, "visited_agents": visited, "analysis_chain": ""}

        # 构造路由决策消息——保留最近20条让 Supervisor 有足够上下文
        system_msg = SystemMessage(content=SUPERVISOR_PROMPT)
        recent = messages[-20:] if len(messages) > 20 else messages
        decision_msgs = [system_msg] + recent

        response = llm.invoke(decision_msgs)
        decision = (response.content or "").strip().lower()

        # 防死循环：最多3轮
        if round_count >= 3:
            logger.info(f"Supervisor 达到最大轮次 → finish_agent")
            return {"next_agent": "finish_agent", "round_count": round_count + 1, "visited_agents": visited, "analysis_chain": ""}

        # 第1轮之后，Supervisor 可以决定 FINISH
        if round_count >= 1:
            if "finish" in decision or "summary" in decision or "总结" in decision:
                logger.info(f"Supervisor 决定汇总 → finish_agent")
                return {"next_agent": "finish_agent", "round_count": round_count + 1, "visited_agents": visited, "analysis_chain": ""}

        # 检测综合分析意图 → 启动分析链
        if _is_comprehensive_intent(decision):
            logger.info(f"Supervisor 检测到综合分析意图 → 启动分析链 stock_agent")
            return {"next_agent": "stock_agent", "round_count": round_count + 1, "visited_agents": visited, "analysis_chain": "active"}

        # 解析并规范化决策
        for agent_name in ["stock_agent", "analysis_agent", "esg_agent", "news_agent", "general_agent"]:
            if agent_name in decision:
                logger.info(f"Supervisor路由 → {agent_name}")
                return {"next_agent": agent_name, "round_count": round_count + 1, "visited_agents": visited, "analysis_chain": ""}

        # 默认走通用Agent
        logger.info(f"Supervisor路由 → general_agent (fallback, decision={decision})")
        return {"next_agent": "general_agent", "round_count": round_count + 1, "visited_agents": visited, "analysis_chain": ""}

    return supervisor_node


# ---------- Worker Agent 工厂 ----------

def _create_worker_node(
    agent_name: str,
    system_prompt: str,
    tools: List,
    use_rag: bool = False,
):
    """创建Worker Agent节点

    Args:
        agent_name: Agent名称标识
        system_prompt: 系统提示词
        tools: 该Agent可用的工具列表
        use_rag: 是否启用RAG知识检索
    """

    llm = _create_llm(temperature=0.3)
    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    def worker_node(state: AgentState) -> Dict[str, Any]:
        messages = list(state.get("messages", []))

        # 构建上下文：系统提示 + 可选RAG + 历史消息
        system_content = system_prompt

        if use_rag:
            # 提取用户问题用于RAG检索
            user_query = ""
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    user_query = msg.content
                    break

            if user_query:
                rag_context = retrieve_knowledge_as_context(user_query, k=3)
                if rag_context:
                    system_content += f"\n\n【知识库参考资料】\n{rag_context}"

        system_msg = SystemMessage(content=system_content)

        # 保留最近30条（含工具消息），确保多轮对话上下文完整
        filtered = []
        for msg in messages[-30:]:
            if isinstance(msg, SystemMessage):
                continue
            filtered.append(msg)

        invoke_msgs = [system_msg] + filtered

        response = llm_with_tools.invoke(invoke_msgs)

        # 如果有工具调用，执行工具
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_messages = []
            for tc in response.tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
                tool_id = tc.get("id", "")

                tool = _ALL_TOOL_MAP.get(tool_name)
                if tool:
                    try:
                        result = tool.invoke(tool_args)
                        logger.info(f"工具调用: {tool_name}({tool_args}) → 成功")
                    except Exception as e:
                        result = f"工具调用失败: {e}"
                        logger.error(f"工具调用失败: {tool_name}, 错误: {e}")
                else:
                    result = f"未知工具: {tool_name}"

                tool_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_id)
                )

            # 用工具结果再次调用LLM生成最终回复
            final_msgs = [system_msg] + filtered + [response] + tool_messages
            final_response = llm.invoke(final_msgs)
            logger.info(f"{agent_name} 生成回复 ({'含' if tool_messages else '无'}工具调用)")
            visited = list(state.get("visited_agents", []))
            if agent_name not in visited:
                visited.append(agent_name)
            return {"messages": [final_response], "next_agent": END, "visited_agents": visited}

        logger.info(f"{agent_name} 直接回复")
        visited = list(state.get("visited_agents", []))
        if agent_name not in visited:
            visited.append(agent_name)
        return {"messages": [response], "next_agent": END, "visited_agents": visited}

    return worker_node


# ---------- 构建 Graph ----------

def build_graph() -> StateGraph:
    """构建 LangGraph 多智能体图"""
    workflow = StateGraph(AgentState)

    # 创建节点
    workflow.add_node(AGENT_NODES["supervisor"], create_supervisor_node())
    workflow.add_node(
        AGENT_NODES["stock_agent"],
        _create_worker_node("stock_agent", STOCK_AGENT_PROMPT, STOCK_TOOLS + [market_overview_tool, sector_overview_tool], use_rag=False),
    )
    workflow.add_node(
        AGENT_NODES["analysis_agent"],
        _create_worker_node("analysis_agent", ANALYSIS_AGENT_PROMPT, ANALYSIS_TOOLS, use_rag=False),
    )
    workflow.add_node(
        AGENT_NODES["esg_agent"],
        _create_worker_node("esg_agent", ESG_AGENT_PROMPT, ESG_TOOLS, use_rag=False),
    )
    workflow.add_node(
        AGENT_NODES["general_agent"],
        _create_worker_node("general_agent", GENERAL_AGENT_PROMPT, [], use_rag=True),
    )
    workflow.add_node(
        AGENT_NODES["news_agent"],
        _create_worker_node("news_agent", NEWS_AGENT_PROMPT, NEWS_TOOLS, use_rag=False),
    )

    # finish_agent：汇总各 Agent 输出，生成综合报告
    workflow.add_node(
        AGENT_NODES["finish_agent"],
        _create_worker_node("finish_agent", GENERAL_AGENT_PROMPT, [], use_rag=False),
    )

    # 设置入口
    workflow.set_entry_point(AGENT_NODES["supervisor"])

    # Supervisor → 各Worker的条件路由
    def route_to_worker(state: AgentState) -> Literal["stock_agent", "analysis_agent", "esg_agent", "news_agent", "finish_agent", "general_agent"]:
        return state["next_agent"]

    workflow.add_conditional_edges(
        AGENT_NODES["supervisor"],
        route_to_worker,
        {
            "stock_agent": AGENT_NODES["stock_agent"],
            "analysis_agent": AGENT_NODES["analysis_agent"],
            "esg_agent": AGENT_NODES["esg_agent"],
            "news_agent": AGENT_NODES["news_agent"],
            "finish_agent": AGENT_NODES["finish_agent"],
            "general_agent": AGENT_NODES["general_agent"],
        },
    )

    # 各Worker → Supervisor（多轮协作：干完活回来，让 Supervisor 判断是否继续）
    workflow.add_edge(AGENT_NODES["stock_agent"], AGENT_NODES["supervisor"])
    workflow.add_edge(AGENT_NODES["analysis_agent"], AGENT_NODES["supervisor"])
    workflow.add_edge(AGENT_NODES["esg_agent"], AGENT_NODES["supervisor"])
    workflow.add_edge(AGENT_NODES["general_agent"], AGENT_NODES["supervisor"])
    workflow.add_edge(AGENT_NODES["news_agent"], AGENT_NODES["supervisor"])

    # finish_agent 直接结束
    workflow.add_edge(AGENT_NODES["finish_agent"], END)

    # 编译图（带内存检查点）
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    logger.info("LangGraph多智能体图构建完成")
    return graph


# ---------- 全局实例 ----------
_graph = None


def get_graph() -> StateGraph:
    """获取全局Graph实例（懒加载）"""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ---------- 对话接口 ----------

def analyze_watchlist(stock_codes: List[str]) -> str:
    """批量分析自选股，生成含股价、数据、ESG、舆情的综合报告

    使用统一数据层 data_layer.fetch_stock_data() 获取 12 维数据。

    Args:
        stock_codes: 股票代码列表

    Returns:
        格式化的综合分析报告（Markdown）
    """
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .data_layer import fetch_stock_data, fetch_sentiment

    llm = _create_llm(temperature=0.3)
    stock_codes = stock_codes[:10]

    # Step 1: 并行获取 舆情 + 逐只股票全维度数据
    with ThreadPoolExecutor(max_workers=6) as outer:
        f_sentiment = outer.submit(fetch_sentiment)

        stock_data = {}
        with ThreadPoolExecutor(max_workers=min(len(stock_codes), 5)) as pool:
            futures = {pool.submit(fetch_stock_data, c): c for c in stock_codes}
            for f in as_completed(futures):
                code = futures[f]
                stock_data[code] = f.result()

        try:
            sentiment = f_sentiment.result(timeout=20)
        except Exception:
            sentiment = {"error": "舆情获取失败"}

    # Step 2: 构建完整数据摘要（MCP + AKShare 双源）
    data_blocks = []
    for code, info in stock_data.items():
        mv = info.get("market", {})
        dcf = info.get("dcf", {})
        ev = info.get("eval", {})
        esg_m = info.get("esg_m", {})
        esg_c = info.get("esg_c", {})
        esg_s = info.get("esg_s", {})
        ak = info.get("akshare", {})

        # AKShare 财务指标
        fin = ak.get("financials", {}).get("latest", {})
        fin_parts = []
        if fin.get("roe") is not None:
            fin_parts.append(f"ROE: {fin['roe']}%")
        if fin.get("npm") is not None:
            fin_parts.append(f"净利率: {fin['npm']}%")
        if fin.get("gpm") is not None:
            fin_parts.append(f"毛利率: {fin['gpm']}%")
        if fin.get("revenue_growth") is not None:
            fin_parts.append(f"营收增长: {fin['revenue_growth']}%")
        if fin.get("profit_growth") is not None:
            fin_parts.append(f"净利增长: {fin['profit_growth']}%")
        if fin.get("debt_ratio") is not None:
            fin_parts.append(f"负债率: {fin['debt_ratio']}%")
        fin_text = " | ".join(fin_parts) if fin_parts else "暂无"

        # AKShare 估值
        val = ak.get("valuation", {})
        val_parts = []
        if val.get("pe"):
            val_parts.append(f"PE: {val['pe']}")
        if val.get("pb"):
            val_parts.append(f"PB: {val['pb']}")
        val_text = " | ".join(val_parts) if val_parts else "暂无"

        # AKShare 行业
        stock_info = ak.get("stock_info", {})
        industry = stock_info.get("industry", "未知")

        block = f"""### {info['name']}（{code}）| 行业：{industry}

**行情**：收盘 {mv.get('close_price','?')} 元 | 市值 {fmt_cap(mv.get('total_market_cap','?'))} | 总股本 {mv.get('total_shares','?')}

**估值**：{val_text}

**DCF估值**：{json.dumps(dcf, ensure_ascii=False)[:300] if dcf else '无数据'}

**综合评估（MCP）**：{json.dumps(ev, ensure_ascii=False)[:1200] if ev else '无数据'}

**最新财务指标（AKShare 实时）**：{fin_text}

**ESG评级**：妙盈 {esg_m.get('esg_rate','?')} | 华证 {esg_c.get('esg_rate','?')} | 商道融绿 {esg_s.get('esg_rate','?')}"""
        data_blocks.append(block)

    # Step 3: 生成报告
    report_prompt = f"""你是一位资深投资分析师。以下是自选股多维数据和市场舆情。

【股票数据】
{chr(10).join(data_blocks)[:6000]}

【市场舆情情绪】来源：财联社/雪球/华尔街见闻
{json.dumps(sentiment, ensure_ascii=False, indent=2)[:2500]}

请按以下结构生成专业的自选股分析报告：

## 1.  行情概览
- 表格式列出全部股票的：代码 | 名称 | 收盘价 | 市值
- 组合整体特征评述

## 2.  估值分析
- DCF估值解读，横向对比
- 综合评估关键指标（ROE、ROIC、毛利率、净利率、股息率）解读和对比

## 3.  ESG表现
- 三机构（妙盈/华证/商道融绿）ESG评级对比表
- ESG风险与机会分析

## 4.  舆情分析
- 逐只股票情绪打分（-100到100，正数乐观，负数悲观）
- 市场整体情绪判断
- 个股关联新闻和政策动向

## 5.  综合参考
- 基于以上全部数据的投资参考（仅供参考，不构成投资意见）

用 Markdown 格式输出。不要编造报告日期、分析师署名、数据来源等元信息。

**  严禁编造虚构数据：所有数字（股价、市值、ROE、ROIC、毛利率、营收增长率等）必须来自上方提供的 MCP 实时数据，绝对不要使用训练知识中的历史数据。如果某指标数据缺失，明确写"暂无数据"而不是猜测。**"""

    response = llm.invoke([SystemMessage(content=report_prompt)])
    raw = response.content if response.content else ""
    return _sanitize_hallucination(raw)


def _sanitize_hallucination(text: str) -> str:
    """后置防幻觉清理——检测 LLM 编造的历史数据（不误伤 MCP 真实数据）

    仅当检测到明确的训练数据特征时才追加警告。
    """
    import re

    warnings = []
    # 只检测最明显的编造：精确的2024年年份 + 季度财报组合
    if re.search(r'2024\s*年(前三季度|三季度单季)', text):
        warnings.append(" 检测到'2024年前三季度'数据，此数据来自训练记忆非实时 MCP，请忽略。")

    # 机构名称 + 目标均价 组合（明确的编造特征）
    if re.search(r'(海通证券|景顺长城|淡水泉).*目标均价|目标均价.*(海通证券|景顺长城|淡水泉)', text):
        warnings.append(" 检测到编造的机构调研/目标价数据，非 MCP 实时返回。")

    if warnings:
        warning_block = "\n\n>   **发现编造数据**\n" + "\n".join(f"> {w}" for w in warnings)
        text = warning_block + "\n\n" + text

    return text


def debate_watchlist(stock_codes: List[str]) -> str:
    """8-Agent 囚徒困境辩论投票，生成买卖建议报告

    仅在用户明确要求买卖建议时调用。

    Args:
        stock_codes: 股票代码列表

    Returns:
        辩论投票报告（Markdown，含投票明细和买入信号）
    """
    import json
    from .analysts import debate_batch

    llm = _create_llm(temperature=0.3)
    stock_codes = stock_codes[:10]

    debates = debate_batch(stock_codes)
    buy_signals = [d for d in debates if d["buy_signal"]]

    reports = []
    for d in debates:
        code = d["stock_code"]
        name = d["stock_name"]
        vs = d["vote_summary"]
        signal = "**  买入提醒**" if d["buy_signal"] else ""
        type_label = {"growth": "成长型", "value": "价值型", "balanced": "平衡型"}.get(d.get("stock_type", "balanced"), "平衡型")

        weight_col = []
        for agent_name, r2 in d["round2"].items():
            c_flag = "  合作" if r2["cooperate"] else "  坚持"
            wb = d.get("weight_breakdown", {}).get(agent_name, {})
            fw = wb.get("final_weight", 1.0)
            weight_col.append(
                f"| {agent_name} | {c_flag} | {r2['vote']} | {r2['score']} | ×{fw:.1f} | {r2['reason']} |"
            )

        debate_block = f"""### {name}（{code}） {signal}

**  股票类型：{type_label}** | 最终得分: {d['final_score']} | 加权: buy {vs['buy_weight']}  hold {vs['hold_weight']}  sell {vs['sell_weight']}

【Bull/Bear 对抗辩论】
{d.get('bull_bear_debate', '')[:500]}

【投票明细】
| 分析师 | 立场 | 投票 | 评分 | 权重 | 理由 |
|---|---|---|---|---|---|
{chr(10).join(weight_col)}

"""

        single_report_prompt = f"""你是一位资深分析师。以下是 8 位分析师对 {name}({code}) 的辩论投票结果：

{debate_block}

请仅针对 {name}({code}) 一只股票，生成买卖建议：
1. 投票结论 — buy/hold/sell 分布和信号
2. 关键分歧 — 多方和空方核心观点
3. 风险提示 — 潜在风险
4. 综合建议 — 一句话建议（仅供参考）

用 Markdown 输出，语言简洁。不要编造日期和分析师署名。
**  严禁编造数据：所有统计数字只可引用上方辩论明细中已出现的评分和票数。**"""

        resp = llm.invoke([SystemMessage(content=single_report_prompt)])
        text = resp.content if hasattr(resp, "content") else str(resp)
        reports.append(f"##   {name}（{code}）{signal}\n\n{text}")

    final_report = "\n\n---\n\n".join(reports)

    if buy_signals:
        buy_list = ", ".join(f"**{d['stock_name']}**（{d['stock_code']}）" for d in buy_signals)
        final_report = (
            f">   买入提醒：以下股票获 8 Agent 辩论多数 buy 票：{buy_list}\n"
            f"> 以上仅供参考，不构成投资建议。\n\n---\n\n{final_report}"
        )

    return final_report


def get_market_overview() -> str:
    """获取每日大盘综合报告

    采集三路数据（并行加速）：
    1. 核心龙头股（MCP 实时行情）
    2. 概念/行业板块涨跌 TOP5（AKShare）
    3. 财联社/雪球/华尔街见闻热点新闻（NewsNow）
    由 LLM 汇总生成「指数·板块·新闻」三位一体的综合报告。

    Returns:
        格式化的综合市场报告
    """
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    llm = _create_llm(temperature=0.3)

    # ── 并行获取三路数据 ──
    BENCHMARK_STOCKS = {
        # 银行
        "600036": "银行·招商银行",
        "601398": "银行·工商银行",
        # 金融（证券/保险）
        "601318": "金融·中国平安",
        "600030": "金融·中信证券",
        # 科技
        "002415": "科技·海康威视",
        "688981": "科技·中芯国际",
        # 消费
        "600519": "消费·贵州茅台",
        "000858": "消费·五粮液",
    }

    def _fetch_one_stock(code_label):
        code, label = code_label
        try:
            raw = _call_mcp_tool_sync("stk_market_value", {"security_code": code})
            d = safe_json(raw)
            if d.get("security_name"):
                return {
                    "name": d.get("security_name", code),
                    "label": label,
                    "price": d.get("close_price", "--"),
                    "cap": fmt_cap(d.get("total_market_cap", 0)),
                }
        except Exception as e:
            logger.warning(f"获取 {code} 数据失败: {e}")
        return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        # 同时提交：7只个股 + 板块 + 新闻 + 市场估值
        stock_futures = {pool.submit(_fetch_one_stock, item): item for item in BENCHMARK_STOCKS.items()}
        sector_future = pool.submit(get_sector_overview_text)
        news_future = pool.submit(_fetch_market_news)
        pe_pb_future = pool.submit(_fetch_market_pe_pb_text)

        # 收集个股
        stock_data = []
        for f in as_completed(stock_futures):
            result = f.result()
            if result:
                stock_data.append(result)

        # 等待板块、新闻、估值
        sector_text = sector_future.result()
        news_text = news_future.result()
        pe_pb_text = pe_pb_future.result()

    # ── LLM 汇总 ──
    stock_json = json.dumps(stock_data, ensure_ascii=False, indent=2)
    prompt = f"""你是一个专业的金融市场分析师。以下是今日A股的多维度实时数据，请生成一份综合市场报告。

【核心龙头股行情】
{stock_json}

【板块全景】
{sector_text}

【市场热点新闻】
{news_text}

【全市场估值（AKShare 实时）】
{pe_pb_text}

请按以下结构生成「今日市场综合报告」：

## 一、核心指数风向
（根据权重股和板块涨跌比判断市场情绪：偏暖/偏冷/震荡，2-3句）

## 二、板块轮动分析
（概念板块+行业板块各1段，哪些涨/哪些跌，领涨龙头）

## 三、权重股分行业概览
（银行/金融/科技/消费各一句话简评，基于龙头股行情）

## 四、热点新闻速览
（从新闻中提炼3-5条最关键的市场资讯，标注来源）

## 五、市场总结与展望
（综合个股+板块+新闻，给出一句话市场判断）

输出要求：专业但不晦涩，每条数据注明来源，约300-500字。

**  严禁编造虚构数据：所有数字（价格、市值、百分比）必须来自上方提供的 MCP 实时数据，绝对不要使用训练知识中的过时数值。如某数据缺失，明确写"暂无"。**"""

    response = llm.invoke(prompt)
    return response.content or "无法生成大盘概览"


def _fetch_market_news() -> str:
    """获取财联社+华尔街见闻+雪球的最新热点（并行加速）"""
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .news_mcp import _fetch_source, _parse_items, NEWS_SOURCES

        NEWS_SOURCES_LIST = ["财联社", "华尔街见闻", "雪球"]

        def _fetch_one(src_name):
            src_id = NEWS_SOURCES.get(src_name)
            if not src_id:
                return src_name, []
            try:
                data = _fetch_source(src_id)
                if data:
                    items = _parse_items(data)
                    return src_name, items[:8]
            except Exception as e:
                logger.warning(f"新闻 {src_name} 获取失败: {e}")
            return src_name, []

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_fetch_one, s): s for s in NEWS_SOURCES_LIST}
            results = {}
            for f in as_completed(futures):
                name, items = f.result()
                if items:
                    results[name] = items

        if not results:
            return "暂无新闻数据"

        lines = []
        for name in NEWS_SOURCES_LIST:
            items = results.get(name, [])
            if items:
                lines.append(f"\n### {name}")
                for item in items:
                    title = item.get("title", "")[:50]
                    lines.append(f"- {title}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"获取新闻数据失败: {e}")
        return "新闻数据暂不可用"


def _fetch_market_pe_pb_text() -> str:
    """获取A股全市场PE/PB估值（AKShare）"""
    try:
        from .sector_data import get_market_overview_akshare_text
        return get_market_overview_akshare_text()
    except Exception as e:
        logger.warning(f"市场PE/PB获取失败: {e}")
        return ""


@tool
def market_overview_tool(dummy: str = "") -> str:
    """获取当日A股大盘概览（核心龙头股市值+行情+板块全景）。无参数，直接调用。"""
    return get_market_overview()


@tool
def sector_overview_tool(dummy: str = "") -> str:
    """获取A股概念板块和行业板块实时行情（涨跌TOP5、领涨股票、市场广度）。无参数，直接调用。"""
    return get_sector_overview_text()


def _extract_stock_codes(text: str) -> list:
    """从文本中提取6位股票代码"""
    import re
    codes = re.findall(r'\b(\d{6})\b', text)
    return list(set(codes))


def _is_buy_sell_intent(text: str) -> bool:
    """判断用户是否在请求买卖建议"""
    keywords = [
        "能不能买", "该不该买", "该不该入", "能不能入", "是否买入",
        "是否卖出", "买入建议", "卖出建议", "可以买", "可以入",
        "值得买", "值得入", "要不要买", "要不要卖", "该卖吗",
        "投资建议", "操作建议", "仓位建议", "买卖建议",
        "要不要入", "建仓", "减仓", "加仓", "清仓", "能买吗", "能卖吗",
        "卖不卖", "该不该卖", "能不能卖", "可以卖吗", "值得卖",
        "止盈", "止损", "离场", "出场", "该跑吗", "是不是该卖",
        "要不要出", "该不该出", "买不买",
    ]
    return any(kw in text for kw in keywords)


def _maybe_debate(message: str, thread_id: str = "", messages: list = None) -> str:
    """若用户消息含买卖意图 + 股票代码 → 返回辩论文本，否则 ''。

    优先从当前消息提取代码，其次从传入的 messages，最后从 graph state 查找。
    """
    codes = _extract_stock_codes(message)

    # 当前消息无代码，尝试从传入的消息历史中提取
    if not codes and messages:
        history = ""
        for m in messages[-10:]:
            if hasattr(m, "content"):
                history += str(m.content) + " "
        codes = _extract_stock_codes(history)

    # 仍未找到，最后尝试从 graph.checkpointer 查找
    if not codes and thread_id:
        try:
            graph = get_graph()
            config = {"configurable": {"thread_id": thread_id}}
            state = graph.get_state(config)
            if state and state.values:
                history = ""
                for m in state.values.get("messages", [])[-10:]:
                    if hasattr(m, "content"):
                        history += str(m.content) + " "
                codes = _extract_stock_codes(history)
        except Exception as e:
            logger.warning(f"从 graph state 提取股票代码失败: {e}")

    if not codes or not _is_buy_sell_intent(message):
        return ""
    try:
        debate = debate_watchlist(codes)
        return f"\n\n---\n\n##  8-Agent 辩论投票\n\n{debate}"
    except Exception as e:
        logger.warning(f"辩论触发失败: {e}")
        return ""


def chat(message: str, thread_id: str = "default") -> Dict[str, Any]:
    """同步对话接口（带上下文记忆）。若含买卖意图则自动触发辩论。

    Returns:
        {"response": str, "agent": str}
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke({"messages": [HumanMessage(content=message)]}, config)
    final_msg = result["messages"][-1] if result.get("messages") else None
    response_text = final_msg.content if final_msg else "抱歉，无法处理您的请求。"

    # 买卖建议 → 自动触发辩论（传入 messages 从历史提取代码）
    all_msgs = result.get("messages", [])
    debate_text = _maybe_debate(message, thread_id, messages=all_msgs)
    if debate_text:
        response_text += debate_text

    return {"response": response_text, "agent": result.get("next_agent", "unknown")}


async def chat_async(message: str, thread_id: str = "default") -> Dict[str, Any]:
    """异步对话接口"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, chat, message, thread_id)


def chat_stream(message: str, thread_id: str = "default"):
    """流式对话接口（带上下文记忆）。若含买卖意图则流式结束后追加辩论。

    Yields:
        {"type": "agent"|"content"|"done"|"debate_signal", ...}
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    last_agent = "supervisor"
    last_event = None

    for event in graph.stream({"messages": [HumanMessage(content=message)]}, config, stream_mode="values"):
        last_event = event
        agent_name = event.get("next_agent", "")
        if agent_name and agent_name != last_agent:
            yield {"type": "agent", "name": agent_name}
            last_agent = agent_name

    # 最终回复
    final_text = ""
    if last_event:
        all_messages = last_event.get("messages", [])
        for m in reversed(all_messages):
            if isinstance(m, AIMessage):
                final_text = m.content
                break
    if not final_text:
        final_text = "抱歉，无法处理您的请求。"

    import time
    chunk_size = 15
    for i in range(0, len(final_text), chunk_size):
        yield {"type": "content", "text": final_text[i:i + chunk_size]}
        time.sleep(0.02)

    yield {"type": "done", "agent": last_agent, "response": final_text}

    # 买卖建议 → 辩论（传入 messages 从历史提取代码）
    all_msgs = last_event.get("messages", []) if last_event else []
    debate_text = _maybe_debate(message, thread_id, messages=all_msgs)
    if debate_text:
        yield {"type": "debate_signal", "text": debate_text}
