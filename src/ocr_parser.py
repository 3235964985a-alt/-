"""
持仓截图识别模块
上传同花顺/东方财富等券商持仓截图 → OCR 提取文字 → LLM 解析代码/名称 → 返回结构化数据
"""
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# 延迟加载 easyocr（首次运行会自动下载模型，约 100MB）
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
    return "\n".join(results)


def parse_portfolio_text(ocr_text: str) -> List[Dict]:
    """用 LLM 从 OCR 文字中解析出持仓结构

    识别：股票代码(6位数字)、股票名称、持仓数量/市值/盈亏
    返回：[{"code": "600519", "name": "贵州茅台", "shares": "100"}, ...]
    """
    try:
        from .config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0,
        )
    except Exception as e:
        logger.warning(f"创建 LLM 失败: {e}")
        return _fallback_parse(ocr_text)

    prompt = f"""你是金融数据解析专家。以下是从券商持仓截图中 OCR 提取的文字，请从中识别出每只持仓股票的信息。

OCR 文字：
{ocr_text[:3000]}

请以 JSON 数组格式返回，每只股票一个对象：
[
  {{"code": "6位数字代码", "name": "股票名称", "shares": "持股数量(如有)", "profit": "盈亏(如有)"}},
  ...
]

规则：
- code 必须是6位纯数字（如 600519），不要包含.SH/.SZ
- name 是股票中文名称
- 如果某个字段无法识别，填 ""
- 只提取股票持仓，忽略基金、债券等
- 如果 OCR 文字中没有股票代码，尝试根据股票名称推断（留空也可以）

仅输出 JSON 数组，不要多余文字。"""

    try:
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        text = text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        if isinstance(result, list):
            return [r for r in result if r.get("code", "").isdigit() and len(r.get("code", "")) == 6]
    except Exception as e:
        logger.warning(f"LLM 解析持仓失败: {e}")

    return _fallback_parse(ocr_text)


def _fallback_parse(text: str) -> List[Dict]:
    """降级方案：正则提取6位数字代码"""
    import re
    codes = re.findall(r'\b(\d{6})\b', text)
    seen = set()
    result = []
    for c in codes:
        if c not in seen and not c.startswith('0'):
            seen.add(c)
            result.append({"code": c, "name": "", "shares": "", "profit": ""})
    return result


def parse_portfolio_from_image(image_path: str) -> List[Dict]:
    """从持仓截图完整流程：OCR → LLM 解析 → 结构化数据"""
    ocr_text = ocr_image(image_path)
    logger.info(f"OCR 完成，提取字符数: {len(ocr_text)}")
    result = parse_portfolio_text(ocr_text)
    logger.info(f"解析出 {len(result)} 只股票: {[r.get('code') for r in result]}")
    return result
