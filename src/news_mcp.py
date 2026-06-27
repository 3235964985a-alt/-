"""
NewsNow 新闻客户端 — 直接通过 HTTP API 获取热点新闻
API: https://newsnow.busiyi.world/api/
"""

import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

NEWS_API_BASE = "https://newsnow.busiyi.world/api"

NEWS_SOURCES = {
    "酷安": "coolapk", "b站": "bilibili-hot-search", "知乎": "zhihu",
    "微博": "weibo", "头条": "toutiao", "抖音": "douyin",
    "github热榜": "github-trending-today", "贴吧": "tieba",
    "华尔街见闻": "wallstreetcn", "澎湃": "thepaper",
    "财联社": "cls-hot", "雪球": "xueqiu", "快手": "kuaishou",
    "linux热榜": "linuxdo-hot",
}


def _fetch_source(source_id: str) -> Optional[Dict]:
    """从 NewsNow API 获取单个源的热点"""
    try:
        with httpx.Client(timeout=15, headers={"User-Agent": "Mozilla/5.0"}) as client:
            resp = client.get(f"{NEWS_API_BASE}/s", params={"id": source_id, "latest": ""})
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning(f"NewsNow {source_id} 获取失败: {e}")
        return None


def _call_newsnow(tool_name: str, arguments: dict) -> str:
    """统一入口"""
    if tool_name == "list_sources":
        return json.dumps(NEWS_SOURCES, ensure_ascii=False, indent=2)

    if tool_name == "get_newsnow":
        source = arguments.get("source", "")
        source_id = _resolve_source(source)
        if not source_id:
            return json.dumps({"error": f"未知新闻源: {source}，可用源: {list(NEWS_SOURCES.keys())}"}, ensure_ascii=False)
        data = _fetch_source(source_id)
        if data is None:
            return json.dumps({"error": f"无法获取 {source} 的新闻"}, ensure_ascii=False)
        return _format_source(data)

    if tool_name == "get_multi_news":
        sources = arguments.get("sources", [])
        if isinstance(sources, str):
            import re
            sources = [s.strip() for s in re.split(r'[,，\s]+', sources) if s.strip()]
        results = {}
        for src in sources[:5]:
            src_id = _resolve_source(src)
            if src_id:
                data = _fetch_source(src_id)
                results[src] = _parse_items(data) if data else []
        if not results:
            return json.dumps({"error": "未成功获取任何新闻"}, ensure_ascii=False)
        return _format_multi(results)

    if tool_name == "get_all_news":
        results = {}
        for name, src_id in list(NEWS_SOURCES.items())[:10]:  # 限10个避免超时
            data = _fetch_source(src_id)
            if data:
                results[name] = _parse_items(data)[:5]
        return _format_multi(results)

    return json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False)


def _resolve_source(source: str) -> Optional[str]:
    """解析新闻源名称到 ID"""
    if source in NEWS_SOURCES:
        return NEWS_SOURCES[source]
    for name, sid in NEWS_SOURCES.items():
        if source.lower() in name.lower() or source.lower() in sid.lower():
            return sid
    return None


def _parse_items(data: Dict) -> List[Dict]:
    """解析新闻条目"""
    items = []
    for item in data.get("items", [])[:10]:
        items.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "info": item.get("extra", {}).get("info", ""),
        })
    return items


def _format_source(data: Dict) -> str:
    """格式化单个源输出"""
    items = _parse_items(data)
    lines = [f"【{data.get('id', '')}】更新时间: {data.get('updatedTime', '未知')}"]
    for i, item in enumerate(items, 1):
        lines.append(f"{i}. {item['title']}")
        if item.get("url"):
            lines.append(f"   {item['url']}")
    return "\n".join(lines)


def _format_multi(results: Dict[str, List]) -> str:
    """格式化多源输出"""
    parts = []
    for name, items in results.items():
        if not items:
            continue
        parts.append(f"\n### {name}")
        for i, item in enumerate(items, 1):
            parts.append(f"{i}. {item['title']}")
    return "\n".join(parts) if parts else "暂无新闻数据"


def _call_newsnow_sync(tool_name: str, arguments: dict) -> str:
    """同步调用（供 LangChain Tool 使用）"""
    return _call_newsnow(tool_name, arguments)


# ---------- LangChain Tool 封装 ----------

class GetSingleNewsInput(BaseModel):
    source: str = Field(description="新闻源名称，如：知乎、微博、财联社、雪球、华尔街见闻等")


class GetMultiNewsInput(BaseModel):
    sources: str = Field(description="新闻源名称列表，逗号分隔，最多5个。如：财联社,雪球,知乎")


class _GetSingleNewsTool(BaseTool):
    name: str = "get_newsnow"
    description: str = "从指定新闻源获取最新热点新闻。支持的源：知乎、微博、财联社、雪球、华尔街见闻、B站、抖音、今日头条等14+平台。"
    args_schema: type = GetSingleNewsInput

    def _run(self, source: str) -> str:
        return _call_newsnow_sync("get_newsnow", {"source": source})


class _GetMultiNewsTool(BaseTool):
    name: str = "get_multi_newsnow"
    description: str = "从多个新闻源获取最新热点新闻（最多5个）。可同时获取财经+社交媒体的新闻。参数示例：财联社,雪球,知乎"
    args_schema: type = GetMultiNewsInput

    def _run(self, sources: str) -> str:
        return _call_newsnow_sync("get_multi_news", {"sources": sources})


class _GetAllNewsTool(BaseTool):
    name: str = "get_all_newsnow"
    description: str = "获取所有主要新闻源的最新热点新闻，适用于宏观市场情绪分析。"

    def _run(self, _: str = "") -> str:
        return _call_newsnow_sync("get_all_news", {})


class _ListSourcesTool(BaseTool):
    name: str = "list_news_sources"
    description: str = "列出所有可用的新闻源及其中文名称（酷安、B站、知乎、微博、财联社、雪球等14+平台）。"

    def _run(self, _: str = "") -> str:
        return _call_newsnow_sync("list_sources", {})


NEWS_TOOLS = [_GetSingleNewsTool(), _GetMultiNewsTool(), _GetAllNewsTool(), _ListSourcesTool()]
NEWS_TOOL_MAP = {t.name: t for t in NEWS_TOOLS}
