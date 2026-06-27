"""
持仓截图识别模块
上传同花顺/东方财富等券商持仓截图 → EasyOCR 提取文字 → LLM 自由识别 → 补全代码 → 返回结构化数据
"""
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ocr_reader = None


def _get_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        logger.info("EasyOCR 中文模型加载完成")
    return _ocr_reader


def ocr_image(image_path: str) -> str:
    """对图片做 OCR，返回提取的纯文本"""
    reader = _get_reader()
    results = reader.readtext(image_path, detail=0)
    # 按行合并，避免一行被切碎
    return "\n".join(results)


def _create_llm():
    from .config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, temperature=0)


def analyze_screenshot_full(ocr_text: str) -> Dict[str, Any]:
    """第一步：让 LLM 自由识别截图中的所有金融信息

    Returns:
        {
          "holdings": [{name, code, shares, cost, profit, ...}, ...],
          "summary": {total_asset, total_profit, ...},
          "raw_thoughts": "LLM的原始分析思路"
        }
    """
    try:
        llm = _create_llm()
    except Exception as e:
        logger.warning(f"创建 LLM 失败: {e}")
        return {"holdings": _fallback_parse(ocr_text), "summary": {}, "raw_thoughts": str(e)}

    prompt = f"""你是一个金融截图解析专家。以下是用户上传的券商持仓截图中 OCR 提取的文字。

【OCR 识别文字】
{ocr_text[:4000]}

请仔细阅读上述文字，从中提取**所有你能识别到的信息**，不限于股票代码。包括但不限于：
- 股票名称（如"贵州茅台""宁德时代"）
- 股票代码（6位数字，可能多个）
- 持仓数量/股数
- 持仓成本/成本价
- 最新价/现价
- 持仓市值
- 盈亏金额/盈亏比例
- 总资产/总市值/总盈亏
- 现金余额
- 还有其他任何你能识别的财务数字

返回 JSON 格式（不要```包裹）：
{{
  "holdings": [
    {{"name": "股票名称", "code": "代码或空字符串", "shares": "股数", "cost": "成本价", "price": "现价", "profit": "盈亏", "profit_pct": "盈亏比例"}}
  ],
  "summary": {{
    "total_asset": "总资产", "total_profit": "总盈亏", "cash": "现金余额", "stock_count": 持仓股票数
  }},
  "notes": ["任何你注意到的额外信息"]
}}

重要规则：
1. name 用中文全称，如 OCR 出现"贵州茅"也尽量还原为"贵州茅台"
2. code 如果有6位数字代码就填，没有就填 ""（后续我们会补全）
3. 数字保留原始格式，不要四舍五入
4. 宁可多提取也不要漏掉 — 不确定的字段填 """""

    try:
        resp = llm.invoke(prompt)
        text = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        logger.info(f"LLM 识别: {len(result.get('holdings', []))} 只股票, summary={result.get('summary', {})}")
        return result
    except Exception as e:
        logger.warning(f"LLM 自由解析失败: {e}")
        return {"holdings": _fallback_parse(ocr_text), "summary": {}, "notes": [str(e)]}


def _resolve_code_by_name(name: str) -> str:
    """通过股票名称查 MCP 获取代码"""
    if not name or len(name) < 2:
        return ""
    try:
        from .mcp_tools import _call_mcp_tool_sync
        raw = _call_mcp_tool_sync("stk_market_value", {"security_code": name})
        # MCP 可能通过 name 直接查到，试试看
        d = json.loads(raw)
        code = d.get("security_code", "")
        if code:
            return code
    except Exception:
        pass
    return ""


def _fill_missing_codes(holdings: List[Dict]) -> List[Dict]:
    """为没代码的股票尝试 MCP 补全代码 — 用 LLM 推断然后 MCP 验"""
    import re

    # 先检查哪些缺 code
    missing = [(i, h) for i, h in enumerate(holdings) if not h.get("code") or not re.match(r'^\d{6}$', str(h.get("code", "")))]

    if not missing:
        return holdings

    # 用 LLM 根据名称推断代码
    names_map = {h["name"]: "" for _, h in missing if h.get("name")}
    if names_map:
        try:
            llm = _create_llm()
            prompt = f"""根据股票中文名称推断6位数字代码。

股票名称列表：{list(names_map.keys())}

返回 JSON 对象：{{"股票名称": "6位代码", ...}}，不确定的填空字符串。
仅输出 JSON。"""

            resp = llm.invoke(prompt)
            text = (resp.content if hasattr(resp, "content") else str(resp)).strip()
            if text.startswith("```"):
                text = text.split("```", 2)[1]
                if text.startswith("json"):
                    text = text[4:]
            inferred = json.loads(text)
            for name, code in inferred.items():
                if isinstance(code, str) and re.match(r'^\d{6}$', code):
                    names_map[name] = code
        except Exception as e:
            logger.warning(f"LLM 推断代码失败: {e}")

    # 写入代码
    for i, h in enumerate(holdings):
        if not h.get("code") or not re.match(r'^\d{6}$', str(h.get("code", ""))):
            name = h.get("name", "")
            h["code"] = names_map.get(name, "")

    return holdings


def parse_portfolio_from_image(image_path: str) -> Dict[str, Any]:
    """完整流程：OCR → LLM 自由解析 → 补全代码 → 结构化数据

    Returns:
        {holdings: [...], summary: {...}, notes: [...]}
    """
    ocr_text = ocr_image(image_path)
    logger.info(f"OCR 完成，提取字符数: {len(ocr_text)}")

    # 第一步：LLM 自由识别
    result = analyze_screenshot_full(ocr_text)

    # 第二步：补全缺失代码
    holdings = result.get("holdings", [])
    if holdings:
        holdings = _fill_missing_codes(holdings)
        result["holdings"] = holdings
        logger.info(f"最终持仓 {len(holdings)} 只: {[(h.get('name'), h.get('code')) for h in holdings]}")

    return result


# 保留降级方案
def _fallback_parse(text: str) -> List[Dict]:
    import re
    codes = re.findall(r'\b(\d{6})\b', text)
    seen = set()
    result = []
    for c in codes:
        if c not in seen and not c.startswith('0'):
            seen.add(c)
            result.append({"code": c, "name": "", "shares": "", "cost": "", "profit": ""})
    return result
