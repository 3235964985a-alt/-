"""
MCP 工具封装层 - 将 MCP 金融数据工具适配为 LangChain Tool

支持两种模式：
1. MCP 模式：通过 MCP SDK 连接远程 MCP Server
2. 模拟模式：当 MCP Server 不可用时使用模拟数据
"""

import json
import logging
from typing import Any, Dict, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .config import MCP_TOOLS_META

logger = logging.getLogger(__name__)

# ---------- MCP 客户端（每次调用都新建短连接，避免嵌套事件循环） ----------
_mcp_client = None
_mcp_lock = None

try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("mcp 包未安装，将使用模拟模式。安装: pip install mcp")


def _extract_text(result) -> str:
    """从 MCP 调用结果中提取文本内容"""
    content = getattr(result, "content", None) or []
    parts = []
    for item in content:
        text = getattr(item, "text", None)
        if text is None and isinstance(item, dict):
            text = item.get("text")
        if text is not None:
            parts.append(text)
    return "\n".join(parts) if parts else json.dumps(content, ensure_ascii=False, default=str)


async def _call_mcp_tool(tool_name: str, arguments: dict) -> str:
    """通过 MCP 客户端调用远程工具（每次新建连接，避免会话缓存问题）"""
    from .config import MCP_SERVER_URL

    if not MCP_AVAILABLE:
        return _mock_tool_response(tool_name, arguments)

    try:
        async with sse_client(MCP_SERVER_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                return _extract_text(result)
    except Exception as e:
        logger.warning(f"MCP调用失败: {e}, 回退到模拟模式")
        return _mock_tool_response(tool_name, arguments)


def _call_mcp_tool_sync(tool_name: str, arguments: dict) -> str:
    """同步版本的MCP工具调用（在新线程中跑 asyncio.run，避免事件循环嵌套）"""
    import asyncio
    import concurrent.futures

    def _runner():
        return asyncio.run(_call_mcp_tool(tool_name, arguments))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_runner)
            return future.result(timeout=30)
    except Exception as e:
        logger.error(f"MCP同步调用失败: {e}")
        return _mock_tool_response(tool_name, arguments)


def _mock_tool_response(tool_name: str, arguments: dict) -> str:
    """模拟工具响应（当MCP不可用时）"""
    security_code = arguments.get("security_code", "未知")

    mock_data = {
        "stk_market_value": {
            "security_id": f"{security_code}.SH",
            "security_code": security_code,
            "security_name": "示例股票",
            "close_price": 50.00,
            "total_market_cap": "5000亿",
            "total_shares": "100亿股"
        },
        "stk_survey": {
            "security_name": "示例股票",
            "declare_date": "2024-06-15",
            "org_nums": 25,
            "title": "机构调研纪要",
            "content": "多家机构对公司业务前景持乐观态度..."
        },
        "miotech_esg_rating": {
            "security_name": "示例股票",
            "esg_rate": "A",
            "enddate": "2024-12-31"
        },
        "chindices_esg_rating": {
            "security_name": "示例股票",
            "esg_rate": "BBB",
            "enddate": "2024-12-31"
        },
        "syntaogf_esg_rating": {
            "security_name": "示例股票",
            "esg_rate": "A-",
            "enddate": "2024-12-31"
        },
        "stk_dcf": {
            "security_name": "示例股票",
            "text": "基于DCF模型，该股票当前估值处于合理区间..."
        },
        "stk_eval": {
            "security_name": "示例股票",
            "eval": "综合评估：公司基本面良好，ROE稳定，建议关注..."
        },
    }

    filter_mock = {
        "result": [
            {"security_code": "600519", "security_name": "贵州茅台", "value": "35.2%"},
            {"security_code": "000858", "security_name": "五粮液", "value": "28.1%"},
            {"security_code": "600809", "security_name": "山西汾酒", "value": "42.5%"},
        ]
    }

    if "filter" in tool_name:
        return json.dumps(filter_mock, ensure_ascii=False)
    if tool_name in mock_data:
        return json.dumps(mock_data[tool_name], ensure_ascii=False)
    return json.dumps({"info": f"工具 {tool_name} 的模拟数据"}, ensure_ascii=False)


# ---------- LangChain Tool 封装 ----------

class StockToolInput(BaseModel):
    """需要证券代码的工具输入"""
    security_code: str = Field(description="证券代码，6位数字，如 600519")


class NoArgInput(BaseModel):
    """无参数工具的输入"""
    pass


class MCPTool(BaseTool):
    """通用MCP工具封装 - 适配所有MCP金融数据工具"""

    def __init__(self, tool_meta: dict, **kwargs):
        t_name = tool_meta["name"]
        t_desc = tool_meta["description"]
        has_args = len(tool_meta.get("parameters", {}).get("required", [])) > 0

        super().__init__(
            name=t_name,
            description=t_desc,
            args_schema=StockToolInput if has_args else NoArgInput,
            **kwargs,
        )
        self._tool_name = t_name

    def _run(self, **kwargs: Any) -> str:
        return _call_mcp_tool_sync(self._tool_name, kwargs)

    async def _arun(self, **kwargs: Any) -> str:
        return await _call_mcp_tool(self._tool_name, kwargs)


def _create_langchain_tool(tool_meta: dict) -> BaseTool:
    """根据工具元数据创建 LangChain Tool 实例"""
    return MCPTool(tool_meta)


# 创建所有工具实例
ALL_TOOLS = [_create_langchain_tool(meta) for meta in MCP_TOOLS_META]

# 按类别分组
STOCK_TOOLS = [t for t in ALL_TOOLS if t.name in ["stk_market_value", "stk_survey"]]
ANALYSIS_TOOLS = [t for t in ALL_TOOLS if t.name in [
    "stk_dcf", "stk_eval",
    "stk_eval_filter_by_roe_1y", "stk_eval_filter_by_roe_3y",
    "stk_eval_filter_by_roic_1y", "stk_eval_filter_by_roic_3y",
    "stk_eval_filter_by_gpm_1y", "stk_eval_filter_by_gpm_3y",
    "stk_eval_filter_by_npm_1y", "stk_eval_filter_by_npm_3y",
    "stk_eval_filter_by_div_rate",
]]
ESG_TOOLS = [t for t in ALL_TOOLS if t.name in [
    "miotech_esg_rating", "chindices_esg_rating", "syntaogf_esg_rating"
]]

# 工具名称 → 工具对象映射
TOOL_MAP = {t.name: t for t in ALL_TOOLS}
