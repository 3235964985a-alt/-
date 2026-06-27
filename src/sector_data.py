"""
板块行情数据模块 — 基于 AKShare 免费接口
覆盖概念板块 + 行业板块的实时行情
"""
import logging
import json
from typing import Dict, List, Optional

import akshare as ak

logger = logging.getLogger(__name__)


def get_concept_sectors(top_n: int = 10) -> List[Dict]:
    """获取概念板块实时行情TOP N（按涨跌幅排序）

    Returns:
        [{"name": "半导体", "change_pct": 3.2, "leading_stock": "中芯国际", ...}, ...]
    """
    try:
        df = ak.stock_board_concept_name_em()
        if df is None or len(df) == 0:
            return []

        # 按涨跌幅绝对值排序，取前 top_n 和后 top_n
        df_sorted = df.sort_values("涨跌幅", ascending=False, key=abs)
        top = df_sorted.head(top_n)
        bottom = df_sorted.tail(top_n)

        result = []
        for _, row in pd.concat([top, bottom]).iterrows():
            result.append({
                "name": str(row.get("板块名称", "")),
                "change_pct": float(row.get("涨跌幅", 0)),
                "change_amount": float(row.get("涨跌额", 0)),
                "latest_price": float(row.get("最新价", 0)),
                "leading_stock": str(row.get("领涨股票", "")),
                "leading_change": str(row.get("领涨股票-涨跌幅", "")),
                "up_count": int(row.get("上涨家数", 0)),
                "down_count": int(row.get("下跌家数", 0)),
            })
        return result
    except Exception as e:
        logger.warning(f"获取概念板块失败: {e}")
        return []


def get_industry_sectors(top_n: int = 10) -> List[Dict]:
    """获取行业板块实时行情TOP N（按涨跌幅排序）

    Returns:
        [{"name": "半导体", "change_pct": 3.2, ...}, ...]
    """
    try:
        df = ak.stock_board_industry_name_em()
        if df is None or len(df) == 0:
            return []

        df_sorted = df.sort_values("涨跌幅", ascending=False, key=abs)
        top = df_sorted.head(top_n)
        bottom = df_sorted.tail(top_n)

        result = []
        for _, row in pd.concat([top, bottom]).iterrows():
            result.append({
                "name": str(row.get("板块名称", "")),
                "change_pct": float(row.get("涨跌幅", 0)),
                "change_amount": float(row.get("涨跌额", 0)),
                "latest_price": float(row.get("最新价", 0)),
                "leading_stock": str(row.get("领涨股票", "")),
                "leading_change": str(row.get("领涨股票-涨跌幅", "")),
                "up_count": int(row.get("上涨家数", 0)),
                "down_count": int(row.get("下跌家数", 0)),
            })
        return result
    except Exception as e:
        logger.warning(f"获取行业板块失败: {e}")
        return []


def get_sector_overview() -> str:
    """获取板块全景概览（概念+行业，并行加速）"""
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor

    def _fetch_concept():
        try:
            df = ak.stock_board_concept_name_em()
            if df is None or len(df) == 0:
                return 0, 0, 0, [], []
            total = len(df)
            up = int((df["涨跌幅"] > 0).sum())
            down = int((df["涨跌幅"] < 0).sum())
            top = df.nlargest(5, "涨跌幅")[["板块名称", "涨跌幅", "领涨股票"]].to_dict("records")
            bot = df.nsmallest(5, "涨跌幅")[["板块名称", "涨跌幅", "领涨股票"]].to_dict("records")
            return total, up, down, top, bot
        except Exception as e:
            logger.warning(f"概念板块数据获取失败: {e}")
            return 0, 0, 0, [], []

    def _fetch_industry():
        try:
            df = ak.stock_board_industry_name_em()
            if df is None or len(df) == 0:
                return 0, 0, 0, [], []
            total = len(df)
            up = int((df["涨跌幅"] > 0).sum())
            down = int((df["涨跌幅"] < 0).sum())
            top = df.nlargest(5, "涨跌幅")[["板块名称", "涨跌幅", "领涨股票"]].to_dict("records")
            bot = df.nsmallest(5, "涨跌幅")[["板块名称", "涨跌幅", "领涨股票"]].to_dict("records")
            return total, up, down, top, bot
        except Exception as e:
            logger.warning(f"行业板块数据获取失败: {e}")
            return 0, 0, 0, [], []

    with ThreadPoolExecutor(max_workers=2) as pool:
        concept_future = pool.submit(_fetch_concept)
        industry_future = pool.submit(_fetch_industry)
        concept_total, concept_up, concept_down, concept_top, concept_bot = concept_future.result()
        industry_total, industry_up, industry_down, industry_top, industry_bot = industry_future.result()

    result = {
        "concept": {
            "total": concept_total,
            "up": concept_up,
            "down": concept_down,
            "top5": concept_top,
            "bottom5": concept_bot,
        },
        "industry": {
            "total": industry_total,
            "up": industry_up,
            "down": industry_down,
            "top5": industry_top,
            "bottom5": industry_bot,
        },
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


def get_sector_overview_text() -> str:
    """板块概览的人类可读文本（供 tool 使用）"""
    raw = get_sector_overview()
    try:
        data = json.loads(raw)
    except Exception:
        return raw

    lines = []
    c = data.get("concept", {})
    i = data.get("industry", {})

    lines.append(f"## 概念板块（共{c.get('total',0)}个，涨{c.get('up',0)}/跌{c.get('down',0)}）")
    lines.append("###  领涨概念 TOP5")
    for item in c.get("top5", []):
        lines.append(f"- {item.get('板块名称','?')}: {item.get('涨跌幅','')}%（领涨: {item.get('领涨股票','')}）")
    lines.append("###  领跌概念 TOP5")
    for item in c.get("bottom5", []):
        lines.append(f"- {item.get('板块名称','?')}: {item.get('涨跌幅','')}%")

    lines.append(f"\n## 行业板块（共{i.get('total',0)}个，涨{i.get('up',0)}/跌{i.get('down',0)}）")
    lines.append("###  领涨行业 TOP5")
    for item in i.get("top5", []):
        lines.append(f"- {item.get('板块名称','?')}: {item.get('涨跌幅','')}%（领涨: {item.get('领涨股票','')}）")
    lines.append("###  领跌行业 TOP5")
    for item in i.get("bottom5", []):
        lines.append(f"- {item.get('板块名称','?')}: {item.get('涨跌幅','')}%")

    return "\n".join(lines)
