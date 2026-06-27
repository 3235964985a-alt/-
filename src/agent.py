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


# ---------- Supervisor ----------

def create_supervisor_node():
    """Supervisor节点：分析用户意图，决定下一个Agent"""

    llm = _create_llm(temperature=0.3)

    def supervisor_node(state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        round_count = state.get("round_count", 0)

        # 构造路由决策消息
        system_msg = SystemMessage(content=SUPERVISOR_PROMPT)
        # 只取最近的用户消息 + 历史摘要
        recent = list(messages[-6:])  # 最近6条
        decision_msgs = [system_msg] + recent

        response = llm.invoke(decision_msgs)
        decision = (response.content or "").strip().lower()

        # 防死循环：最多3轮
        if round_count >= 3:
            logger.info(f"Supervisor 达到最大轮次 → finish_agent")
            return {"next_agent": "finish_agent", "round_count": round_count + 1}

        # 第1轮之后，Supervisor 可以决定 FINISH
        if round_count >= 1:
            if "finish" in decision or "summary" in decision or "总结" in decision:
                logger.info(f"Supervisor 决定汇总 → finish_agent")
                return {"next_agent": "finish_agent", "round_count": round_count + 1}

        # 解析并规范化决策
        for agent_name in ["stock_agent", "analysis_agent", "esg_agent", "news_agent", "general_agent"]:
            if agent_name in decision:
                logger.info(f"Supervisor路由 → {agent_name}")
                return {
                    "next_agent": agent_name,
                    "round_count": round_count + 1,
                }

        # 默认走通用Agent
        logger.info(f"Supervisor路由 → general_agent (fallback, decision={decision})")
        return {
            "next_agent": "general_agent",
            "round_count": round_count + 1,
        }

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

        # 过滤掉工具消息和旧的系统消息，保留最近的对话
        filtered = []
        for msg in messages[-10:]:
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
            return {"messages": [final_response], "next_agent": END}

        logger.info(f"{agent_name} 直接回复")
        return {"messages": [response], "next_agent": END}

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
    """批量分析自选股，生成对比报告（并行加速）

    每只股票并行调用：市值、综合评估、ESG评级，
    然后由 LLM 汇总生成对比分析报告。

    Args:
        stock_codes: 股票代码列表

    Returns:
        格式化的分析报告字符串
    """
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    llm = _create_llm(temperature=0.3)

    # 并行收集每只股票的数据
    def _fetch_one_stock(code):
        """并行：同时取市值+评估+ESG"""
        info = {"code": code, "name": code}
        with ThreadPoolExecutor(max_workers=3) as inner:
            f_mv = inner.submit(_call_mcp_tool_sync, "stk_market_value", {"security_code": code})
            f_ev = inner.submit(_call_mcp_tool_sync, "stk_eval", {"security_code": code})
            f_esg = inner.submit(_call_mcp_tool_sync, "miotech_esg_rating", {"security_code": code})

            try:
                info["market"] = _safe_json(f_mv.result(timeout=10))
                info["name"] = info["market"].get("security_name", code)
            except Exception:
                info["market"] = {}
            try:
                info["eval"] = _safe_json(f_ev.result(timeout=10))
            except Exception:
                info["eval"] = {}
            try:
                info["esg"] = _safe_json(f_esg.result(timeout=10))
            except Exception:
                info["esg"] = {}
        return code, info

    stock_data = {}
    max_stocks = stock_codes[:10]

    with ThreadPoolExecutor(max_workers=min(len(max_stocks), 5)) as pool:
        futures = {pool.submit(_fetch_one_stock, c): c for c in max_stocks}
        for f in as_completed(futures):
            code, info = f.result()
            stock_data[code] = info

    # 构建 LLM 提示词
    summary_lines = []
    for code, info in stock_data.items():
        mv = info.get("market", {})
        ev = info.get("eval", {})
        esg = info.get("esg", {})
        summary_lines.append(
            f"| {info['name']}({code}) "
            f"| 收盘 {mv.get('close_price','?')} "
            f"| 市值 {_fmt_cap(mv.get('total_market_cap','?'))} "
            f"| ESG {esg.get('esg_rate','?')} "
            f"| 评估: {str(ev)[:100]} |"
        )

    report_prompt = f"""你是一位资深投资分析师。以下是自选股数据：

{chr(10).join(summary_lines)}

请生成一份专业的自选股分析报告，包含：
1. **概览** — 组合整体特征（行业分布、市值规模等）
2. **估值对比** — 各股票估值水平横向对比
3. **ESG 表现** — ESG评级对比
4. **综合建议** — 基于以上数据给出投资建议（仅供参考，不构成投资意见）

用 Markdown 格式输出，语言专业但不晦涩。

**重要**：不要编造报告日期、分析师署名、数据来源等元信息。直接输出分析内容即可。"""

    response = llm.invoke([SystemMessage(content=report_prompt)])
    return response.content


def _safe_json(data):
    """安全解析 JSON"""
    import json
    if isinstance(data, dict):
        return data
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return {"text": str(data)[:500]}


def _fmt_cap(value):
    """格式化市值"""
    if isinstance(value, (int, float)):
        if value > 1e12:
            return f"{value/1e12:.2f}万亿"
        elif value > 1e8:
            return f"{value/1e8:.2f}亿"
    return str(value)


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
            d = _safe_json(raw)
            if d.get("security_name"):
                return {
                    "name": d.get("security_name", code),
                    "label": label,
                    "price": d.get("close_price", "--"),
                    "cap": _fmt_cap(d.get("total_market_cap", 0)),
                }
        except Exception as e:
            logger.warning(f"获取 {code} 数据失败: {e}")
        return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        # 同时提交：7只个股 + 板块 + 新闻
        stock_futures = {pool.submit(_fetch_one_stock, item): item for item in BENCHMARK_STOCKS.items()}
        sector_future = pool.submit(get_sector_overview_text)
        news_future = pool.submit(_fetch_market_news)

        # 收集个股
        stock_data = []
        for f in as_completed(stock_futures):
            result = f.result()
            if result:
                stock_data.append(result)

        # 等待板块和新闻
        sector_text = sector_future.result()
        news_text = news_future.result()

    # ── LLM 汇总 ──
    stock_json = json.dumps(stock_data, ensure_ascii=False, indent=2)
    prompt = f"""你是一个专业的金融市场分析师。以下是今日A股的多维度实时数据，请生成一份综合市场报告。

【核心龙头股行情】
{stock_json}

【板块全景】
{sector_text}

【市场热点新闻】
{news_text}

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

输出要求：专业但不晦涩，每条数据注明来源，约300-500字。"""

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


@tool
def market_overview_tool(dummy: str = "") -> str:
    """获取当日A股大盘概览（核心龙头股市值+行情+板块全景）。无参数，直接调用。"""
    return get_market_overview()


@tool
def sector_overview_tool(dummy: str = "") -> str:
    """获取A股概念板块和行业板块实时行情（涨跌TOP5、领涨股票、市场广度）。无参数，直接调用。"""
    return get_sector_overview_text()


def chat(message: str, thread_id: str = "default") -> Dict[str, Any]:
    """同步对话接口

    Args:
        message: 用户消息
        thread_id: 会话线程ID（用于多轮对话记忆）

    Returns:
        {"response": str, "agent": str, "tool_calls": list}
    """
    graph = get_graph()

    config = {"configurable": {"thread_id": thread_id}}
    state = {
        "messages": [HumanMessage(content=message)],
        "next_agent": "supervisor",
        "round_count": 0,
    }

    result = graph.invoke(state, config)

    # 提取最终回复
    final_msg = result["messages"][-1] if result.get("messages") else None
    response_text = final_msg.content if final_msg else "抱歉，无法处理您的请求。"

    return {
        "response": response_text,
        "agent": result.get("next_agent", "unknown"),
    }


async def chat_async(message: str, thread_id: str = "default") -> Dict[str, Any]:
    """异步对话接口"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, chat, message, thread_id)


def chat_stream(message: str, thread_id: str = "default"):
    """流式对话接口

    使用 LangGraph stream 模式，每个节点完成后产出增量状态，
    最终 Agent 的输出会逐段流式返回。

    Yields:
        {"type": "agent", "name": str}  — Agent 切换
        {"type": "content", "text": str} — 最终回复内容（逐字模拟流式）
        {"type": "done", "agent": str, "response": str} — 完成信号
    """
    graph = get_graph()

    config = {"configurable": {"thread_id": thread_id}}
    state = {
        "messages": [HumanMessage(content=message)],
        "next_agent": "supervisor",
        "round_count": 0,
    }

    last_agent = "supervisor"
    last_event = None

    for event in graph.stream(state, config, stream_mode="values"):
        last_event = event
        agent_name = event.get("next_agent", "")

        if agent_name and agent_name != last_agent:
            yield {"type": "agent", "name": agent_name}
            last_agent = agent_name

    # 拿到最终回复
    final_text = ""
    if last_event:
        all_messages = last_event.get("messages", [])
        for m in reversed(all_messages):
            if isinstance(m, AIMessage):
                final_text = m.content
                break

    if not final_text:
        final_text = "抱歉，无法处理您的请求。"

    # 模拟流式逐字输出（实际生产中可用 LLM stream 替换）
    import time
    chunk_size = 15
    for i in range(0, len(final_text), chunk_size):
        chunk = final_text[i:i+chunk_size]
        yield {"type": "content", "text": chunk}
        time.sleep(0.02)

    yield {"type": "done", "agent": last_agent, "response": final_text}
