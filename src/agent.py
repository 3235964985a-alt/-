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
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from .mcp_tools import STOCK_TOOLS, ANALYSIS_TOOLS, ESG_TOOLS, TOOL_MAP
from .prompts import (
    SUPERVISOR_PROMPT,
    STOCK_AGENT_PROMPT,
    ANALYSIS_AGENT_PROMPT,
    ESG_AGENT_PROMPT,
    GENERAL_AGENT_PROMPT,
)
from .rag import retrieve_knowledge_as_context

logger = logging.getLogger(__name__)

# ---------- Agent节点标识 ----------
AGENT_NODES = {
    "supervisor": "supervisor",
    "stock_agent": "stock_agent",
    "analysis_agent": "analysis_agent",
    "esg_agent": "esg_agent",
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

        # 解析并规范化决策
        for agent_name in ["stock_agent", "analysis_agent", "esg_agent", "general_agent"]:
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

                tool = TOOL_MAP.get(tool_name)
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
        _create_worker_node("stock_agent", STOCK_AGENT_PROMPT, STOCK_TOOLS, use_rag=False),
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

    # 设置入口
    workflow.set_entry_point(AGENT_NODES["supervisor"])

    # Supervisor → 各Worker的条件路由
    def route_to_worker(state: AgentState) -> Literal["stock_agent", "analysis_agent", "esg_agent", "general_agent"]:
        return state["next_agent"]

    workflow.add_conditional_edges(
        AGENT_NODES["supervisor"],
        route_to_worker,
        {
            "stock_agent": AGENT_NODES["stock_agent"],
            "analysis_agent": AGENT_NODES["analysis_agent"],
            "esg_agent": AGENT_NODES["esg_agent"],
            "general_agent": AGENT_NODES["general_agent"],
        },
    )

    # 各Worker → END
    workflow.add_edge(AGENT_NODES["stock_agent"], END)
    workflow.add_edge(AGENT_NODES["analysis_agent"], END)
    workflow.add_edge(AGENT_NODES["esg_agent"], END)
    workflow.add_edge(AGENT_NODES["general_agent"], END)

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

    Yields:
        状态更新dict，包含当前输出
    """
    graph = get_graph()

    config = {"configurable": {"thread_id": thread_id}}
    state = {
        "messages": [HumanMessage(content=message)],
        "next_agent": "supervisor",
        "round_count": 0,
    }

    for event in graph.stream(state, config):
        yield event
