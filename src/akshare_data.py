"""
AKShare 个股数据补充层 —— 提供 MCP 未覆盖的最新财务/估值数据

接口基于 AKShare 免费数据源，优先使用 akshare 最新数据。

覆盖维度：
  - 个股基本信息（行业、上市日期等）
  - 财务分析指标（ROE、ROA、净利率、毛利率等）
  - 最新财报摘要
  - 估值数据（PE、PB历史分位等）
"""
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _safe_float(val) -> Optional[float]:
    """安全转浮点数"""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def get_stock_info(code: str) -> Dict[str, Any]:
    """获取个股基本信息（AKShare）

    Returns:
        {industry, listing_date, full_name, ...}
    """
    try:
        import akshare as ak
        # 加市场后缀：6开头=SH，0/3开头=SZ
        symbol = f"SH{code}" if code.startswith("6") else f"SZ{code}"
        df = ak.stock_individual_info_em(symbol=symbol)
        if df is None or len(df) == 0:
            return {}

        info = {}
        for _, row in df.iterrows():
            key = str(row.get("item", ""))
            val = str(row.get("value", ""))
            if key and val:
                info[key] = val

        result = {
            "company_name": info.get("股票简称", info.get("公司名称", "")),
            "industry": info.get("行业", ""),
            "listing_date": info.get("上市时间", ""),
            "total_shares": info.get("总股本", ""),
            "circulating_shares": info.get("流通股", ""),
        }
        return result
    except Exception as e:
        logger.debug(f"AKShare 个股信息获取失败 {code}: {e}")
        return {}


def get_financial_indicators(code: str) -> Dict[str, Any]:
    """获取个股财务分析指标（AKShare）

    包含最新季度的 ROE、ROA、净利率、毛利率、营收增长率、净利增长率等。

    Returns:
        {latest_quarter: {roe, roa, npm, gpm, ...}, ...}
    """
    try:
        import akshare as ak
        symbol = f"SH{code}" if code.startswith("6") else f"SZ{code}"
        df = ak.stock_financial_analysis_indicator(symbol=symbol)
        if df is None or len(df) == 0:
            return {}

        # 取最近 4 个季度
        recent = df.tail(4)
        indicators = []
        for _, row in recent.iterrows():
            indicators.append({
                "date": str(row.get("日期", row.get("季度", ""))),
                "roe": _safe_float(row.get("净资产收益率(%)", row.get("净资产收益率", None))),
                "roa": _safe_float(row.get("总资产收益率(%)", row.get("总资产收益率", None))),
                "npm": _safe_float(row.get("净利率(%)", row.get("净利率", None))),
                "gpm": _safe_float(row.get("毛利率(%)", row.get("毛利率", None))),
                "profit_growth": _safe_float(row.get("净利润增长率(%)", row.get("净利润同比增长率", None))),
                "revenue_growth": _safe_float(row.get("营业收入增长率(%)", row.get("营收同比增长率", None))),
                "debt_ratio": _safe_float(row.get("资产负债率(%)", row.get("资产负债率", None))),
            })

        return {
            "latest": indicators[-1] if indicators else {},
            "recent_quarters": indicators,
        }
    except Exception as e:
        logger.debug(f"AKShare 财务指标获取失败 {code}: {e}")
        return {}


def get_financial_abstract(code: str) -> Dict[str, Any]:
    """获取个股最新财报摘要（AKShare）

    Returns:
        {report_date, revenue, net_profit, eps, ...}
    """
    try:
        import akshare as ak
        symbol = f"SH{code}" if code.startswith("6") else f"SZ{code}"
        df = ak.stock_financial_abstract(symbol=symbol)
        if df is None or len(df) == 0:
            return {}

        # 取最新一条
        latest = df.iloc[-1].to_dict() if len(df) > 0 else {}
        result = {}
        # 常见字段映射
        field_map = {
            "营业收入": "revenue",
            "营业利润": "operating_profit",
            "利润总额": "total_profit",
            "净利润": "net_profit",
            "基本每股收益": "eps",
            "报告日期": "report_date",
        }
        for cn_key, en_key in field_map.items():
            val = latest.get(cn_key)
            if val is not None:
                result[en_key] = _safe_float(val) or str(val)
        return result
    except Exception as e:
        logger.debug(f"AKShare 财报摘要获取失败 {code}: {e}")
        return {}


def get_valuation_summary(code: str) -> Dict[str, Any]:
    """获取个股估值概要（AKShare 百度接口）

    返回 PE、PB 等估值指标。

    Returns:
        {pe, pb, market_cap, ...}
    """
    try:
        import akshare as ak
        symbol = f"sh{code}" if code.startswith("6") else f"sz{code}"
        df = ak.stock_zh_valuation_baidu(symbol=symbol, indicator="全部")
        if df is None or len(df) == 0:
            return {}

        latest = df.iloc[-1].to_dict()
        result = {}
        field_map = {
            "pe": "市盈率",
            "pb": "市净率",
            "market_cap": "总市值",
            "date": "日期",
        }
        for en_key, cn_key in field_map.items():
            val = latest.get(cn_key)
            if val is not None:
                result[en_key] = _safe_float(val) or str(val)
        return result
    except Exception as e:
        logger.debug(f"AKShare 估值数据获取失败 {code}: {e}")
        return {}


def fetch_stock_akshare(code: str) -> Dict[str, Any]:
    """并行获取 akshare 所有个股数据维度

    返回所有 akshare 数据的聚合 dict，供 data_layer 合并使用。
    """
    from concurrent.futures import ThreadPoolExecutor

    result = {"code": code}
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_info = pool.submit(get_stock_info, code)
        f_indicators = pool.submit(get_financial_indicators, code)
        f_abstract = pool.submit(get_financial_abstract, code)
        f_valuation = pool.submit(get_valuation_summary, code)

        for key, fut in [("stock_info", f_info), ("financials", f_indicators),
                          ("abstract", f_abstract), ("valuation", f_valuation)]:
            try:
                result[key] = fut.result(timeout=10)
            except Exception:
                result[key] = {}

    return result
