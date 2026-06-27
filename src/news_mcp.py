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

_FINANCE_SOURCES = {"cls-hot", "xueqiu", "wallstreetcn"}  # 核心财经源，全量获取

# 仅返回有实际新闻内容的源，过滤名字列表型源
_SKIP_SOURCES = {
    "bilibili-hot-search", "coolapk", "douyin", "kuaishou",  # 非财经内容
    "github-trending-today", "linuxdo-hot",  # 非中文/非财经
}

# 雪球只返回股票代码名，取更多条才有内容
_XUEQIU_LIMIT = 15


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
        for name, src_id in NEWS_SOURCES.items():
            if src_id in _SKIP_SOURCES:
                continue
            data = _fetch_source(src_id)
            if data:
                items = _parse_items(data)
                if src_id == "xueqiu":
                    items = items[:_XUEQIU_LIMIT]
                elif src_id in _FINANCE_SOURCES:
                    pass  # 全取
                else:
                    items = items[:5]
                if items:
                    results[name] = items
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
    source_name = _id_to_name(data.get('id', ''))
    items = _parse_items(data)
    lines = [f"来源：{source_name} | 更新时间: {data.get('updatedTime', '未知')}"]
    for i, item in enumerate(items, 1):
        lines.append(f"{i}. {item['title']} 【来源：{source_name}】")
        if item.get("url"):
            lines.append(f"   {item['url']}")
    return "\n".join(lines)


def _format_multi(results: Dict[str, List]) -> str:
    """格式化多源输出"""
    parts = []
    for name, items in results.items():
        if not items:
            continue
        parts.append(f"\n### 【来源：{name}】")
        for i, item in enumerate(items, 1):
            parts.append(f"{i}. {item['title']} 【来源：{name}】")
    return "\n".join(parts) if parts else "暂无新闻数据"


def _id_to_name(source_id: str) -> str:
    """根据 source_id 反查中文名"""
    for name, sid in NEWS_SOURCES.items():
        if sid == source_id:
            return name
    return source_id


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


# ---------- 新闻情绪评分 ----------

def _get_sentiment(llm, titles: List[str], source_name: str) -> Dict:
    """调用 LLM 对一批新闻标题做情绪评分"""
    if not titles:
        return {"score": 0, "label": "中性", "summary": "无新闻数据"}

    titles_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles[:15]))
    prompt = f"""你是金融市场情绪分析专家。请对以下【{source_name}】的新闻标题做情绪评分。

新闻标题：
{titles_text}

请以 JSON 格式返回：
{{
    "score": -100到100的整数（-100极度悲观，0中性，100极度乐观），
    "label": "乐观/偏乐观/中性/偏悲观/悲观",
    "summary": "一句话概括该源的整体情绪倾向和关键信号"
}}

仅输出 JSON，不要多余文字。"""

    try:
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        # 提取 JSON
        text = text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        logger.warning(f"情绪评分失败 ({source_name}): {e}")
        return {"score": 0, "label": "中性", "summary": f"评分异常: {e}"}


def get_news_sentiment(force_refresh: bool = False) -> str:
    """获取新闻情绪综合评分

    从财联社、华尔街见闻、雪球获取最新新闻标题，
    由 LLM 分别打分（-100到100），汇总情绪全景。

    Returns:
        格式化的情绪分析报告
    """
    try:
        from .config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0.1,
        )
    except Exception as e:
        logger.warning(f"创建 LLM 失败: {e}")
        return json.dumps({"error": "LLM 不可用"}, ensure_ascii=False)

    FINANCE_SOURCES = ["财联社", "华尔街见闻", "雪球"]
    results = {}

    for src_name in FINANCE_SOURCES:
        src_id = NEWS_SOURCES.get(src_name)
        if not src_id:
            continue
        try:
            data = _fetch_source(src_id)
            if data:
                items = _parse_items(data)
                titles = [item.get("title", "") for item in items[:12]]
            else:
                titles = []
        except Exception as e:
            logger.warning(f"获取 {src_name} 失败: {e}")
            titles = []

        sent = _get_sentiment(llm, titles, src_name)
        results[src_name] = {
            "score": sent.get("score", 0),
            "label": sent.get("label", "中性"),
            "titles": titles[:3],  # 前端展示前3条
        }

    # 汇总
    scores = [r["score"] for r in results.values()]
    avg_score = sum(scores) / len(scores) if scores else 0
    if avg_score > 30:
        overall = "市场情绪偏乐观"
    elif avg_score < -30:
        overall = "市场情绪偏悲观"
    else:
        overall = "市场情绪中性"

    report = {
        "overall_score": round(avg_score, 1),
        "overall_label": overall,
        "sources": results,
    }
    return json.dumps(report, ensure_ascii=False, indent=2)


class _GetNewsSentimentTool(BaseTool):
    name: str = "get_news_sentiment"
    description: str = "获取新闻情绪综合评分。从财联社、华尔街见闻、雪球获取最新新闻，由LLM评分（-100到100），返回市场情绪全景。无参数。"

    def _run(self, _: str = "") -> str:
        return get_news_sentiment()


NEWS_TOOLS = [_GetSingleNewsTool(), _GetMultiNewsTool(), _GetAllNewsTool(), _ListSourcesTool(), _GetNewsSentimentTool()]
NEWS_TOOL_MAP = {t.name: t for t in NEWS_TOOLS}
