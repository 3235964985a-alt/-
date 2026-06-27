"""
统一数据接入层 —— 所有模块共享的 MCP 数据获取函数

消除 agent.py / debate_agent.py / analysts.py 中三套重复的 MCP 调用逻辑。
"""
import json
import logging
from typing import Any, Dict
from concurrent.futures import ThreadPoolExecutor

from .mcp_tools import _call_mcp_tool_sync

logger = logging.getLogger(__name__)


# ---------- 工具函数 ----------

def safe_json(data: Any) -> Dict[str, Any]:
    """安全解析 JSON — 已是 dict 直接返回，字符串则解析，否则包裹"""
    if isinstance(data, dict):
        return data
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return {"text": str(data)[:500]}


def fmt_cap(value) -> str:
    """格式化市值：亿 / 万亿"""
    if isinstance(value, (int, float)):
        if value > 1e12:
            return f"{value / 1e12:.2f}万亿"
        elif value > 1e8:
            return f"{value / 1e8:.2f}亿"
    return str(value)


# ---------- 单只股票全维度数据 ----------

def fetch_stock_data(code: str) -> Dict[str, Any]:
    """并行获取一只股票的全维度数据（7 MCP + 4 AKShare）

    MCP 数据：市值、DCF、综合评估、调研、3机构ESG
    AKShare 补充：个股信息、财务指标、财报摘要、估值概要

    注意：stk_eval_filter_by_* 系列是全局筛选器，不支持单只股票查询。
    单只股票的 ROE/ROIC/毛利率/净利率/股息率 从 stk_eval 文本和 akshare 财务指标双源获取。

    返回:
        {code, name, market, eval, survey, dcf, esg_m, esg_c, esg_s,
         akshare: {stock_info, financials, abstract, valuation}}
    """
    from .akshare_data import fetch_stock_akshare

    data: Dict[str, Any] = {"code": code, "name": code}

    # MCP + AKShare 双管齐下
    with ThreadPoolExecutor(max_workers=2) as dual:
        f_mcp = dual.submit(_fetch_mcp_batch, code)
        f_ak = dual.submit(fetch_stock_akshare, code)

        # MCP 数据
        mcp_data = f_mcp.result()
        for k, v in mcp_data.items():
            data[k] = v

        # AKShare 补充
        data["akshare"] = f_ak.result()

    return data


def _fetch_mcp_batch(code: str) -> Dict[str, Any]:
    """仅 MCP 数据获取（7 个工具并行）"""
    data: Dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=7) as pool:
        tasks = {
            "market": pool.submit(_call_mcp_tool_sync, "stk_market_value", {"security_code": code}),
            "eval":   pool.submit(_call_mcp_tool_sync, "stk_eval", {"security_code": code}),
            "survey": pool.submit(_call_mcp_tool_sync, "stk_survey", {"security_code": code}),
            "dcf":    pool.submit(_call_mcp_tool_sync, "stk_dcf", {"security_code": code}),
            "esg_m":  pool.submit(_call_mcp_tool_sync, "miotech_esg_rating", {"security_code": code}),
            "esg_c":  pool.submit(_call_mcp_tool_sync, "chindices_esg_rating", {"security_code": code}),
            "esg_s":  pool.submit(_call_mcp_tool_sync, "syntaogf_esg_rating", {"security_code": code}),
        }

        for key, fut in tasks.items():
            try:
                data[key] = safe_json(fut.result(timeout=15))
            except Exception:
                data[key] = {}

    # 取股票名称（优先 MCP，其次 akshare）
    if data.get("market"):
        data["name"] = data["market"].get("security_name", code)

    return data


# ---------- 市场舆情 ----------

def fetch_sentiment() -> Dict[str, Any]:
    """获取财联社/雪球/华尔街见闻三源舆情情绪评分"""
    try:
        from .news_mcp import get_news_sentiment
        raw = get_news_sentiment()
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return {"overall_score": 0, "overall_label": "无数据"}


# ---------- 板块趋势 ----------

def fetch_sector_trend() -> Dict[str, Any]:
    """获取概念板块和行业板块涨跌 TOP5（供技术派使用）"""
    try:
        from .sector_data import get_concept_sectors, get_industry_sectors
        concepts = get_concept_sectors(top_n=5)
        industries = get_industry_sectors(top_n=5)
        return {
            "concept_top5": [
                {"name": c.get("name", "?"), "pct": c.get("pct_change", 0)}
                for c in (concepts or [])[:5]
            ],
            "industry_top5": [
                {"name": i.get("name", "?"), "pct": i.get("pct_change", 0)}
                for i in (industries or [])[:5]
            ],
        }
    except Exception as e:
        logger.debug(f"板块数据获取失败: {e}")
        return {}
