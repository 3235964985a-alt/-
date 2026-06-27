"""
5-Agent 辩论投票系统 —— Bull/Bear 对抗辩论 + 独立打分

使用统一的 data_layer 获取 MCP 数据。

架构:
  Round 0: Bull vs Bear 单轮对抗辩论（2 LLM 并行）
  Round 1: 5 Agent 并行独立打分（5 LLM 并行）
  最终裁决: 角色权重加成，buy 权重 > 50% → 买入提醒

LLM 调用总数：2 + 5 + 1（最终摘要）= 8 次（原 21 次）
"""
import json
import logging
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from .config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from .data_layer import safe_json, fetch_stock_data, fetch_sentiment, fetch_sector_trend

logger = logging.getLogger(__name__)


# ---------- 分析师数据类 ----------

class AnalystAgent:
    def __init__(self, name: str, role: str, tools: Dict[str, Tuple[str, dict]], philosophy: str):
        self.name = name
        self.role = role
        self.tools = tools
        self.philosophy = philosophy
        self.score = 0
        self.vote = "hold"
        self.reason = ""


# ---------- LLM 创建 ----------

def _create_debate_llm() -> ChatOpenAI:
    kwargs = {"model": OPENAI_MODEL, "temperature": 0}
    if OPENAI_API_KEY:
        kwargs["openai_api_key"] = OPENAI_API_KEY
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return ChatOpenAI(**kwargs)


# ---------- 5 位分析师定义 ----------

def _build_analysts() -> List[AnalystAgent]:
    """5 位核心分析师（精简版，加速辩论）"""
    return [
        AnalystAgent(
            name="价值派·格雷厄姆",
            role="价值投资者",
            tools={"stk_dcf": "DCF现金流折现估值"},
            philosophy="买入价格低于内在价值的股票，安全边际是第一原则。PE越低越好，DCF估值折价越大越安全。"
        ),
        AnalystAgent(
            name="成长派·费雪",
            role="成长投资者",
            tools={"stk_eval_filter_by_roe_1y": "ROE筛选", "stk_eval_filter_by_roic_1y": "ROIC筛选"},
            philosophy="寻找高ROE、高ROIC的优质成长股。盈利能力强的公司值得溢价买入，成长性比当前估值更重要。"
        ),
        AnalystAgent(
            name="质量派·芒格",
            role="质量投资者",
            tools={"stk_eval_filter_by_gpm_1y": "毛利率", "stk_eval_filter_by_npm_1y": "净利率", "stk_eval_filter_by_div_rate": "股息率"},
            philosophy="以合理价格买入优质企业。高毛利、高净利、稳定分红是好公司的标志。宁可贵一点买好公司，不买便宜的烂公司。"
        ),
        AnalystAgent(
            name="责任派·ESG分析师",
            role="ESG分析师",
            tools={"miotech_esg_rating": "妙盈ESG", "chindices_esg_rating": "华证ESG", "syntaogf_esg_rating": "商道融绿ESG"},
            philosophy="环境社会和治理表现优秀的企业长期更有韧性。ESG评级高的企业风险更低，值得长期持有。"
        ),
        AnalystAgent(
            name="风控官·塔勒布",
            role="风险管理官",
            tools={},
            philosophy="黑天鹅永远存在。不依赖任何单一维度的判断。综合所有分析师的结论，识别尾部风险。宁可错过，不要做错。"
        ),
    ]


# ---------- 股票类型识别 ----------

def _classify_stock_type(data: Dict[str, Any]) -> str:
    """从 stk_eval 综合评估文本中提取 ROE 和股息率，判断 growth / value / balanced

    注意：stk_eval_filter_by_* 系列是全局筛选器，不支持单股查询。
    ROE/股息率 数据从 stk_eval 的综合评估文本中解析。
    """
    import re

    roe_score = 0
    div_score = 0

    # 优先尝试结构化字段（如果有其他来源），否则从 eval 文本解析
    eval_raw = data.get("eval", {})
    eval_text = ""
    if isinstance(eval_raw, dict):
        eval_text = eval_raw.get("eval", "") or eval_raw.get("text", "") or json.dumps(eval_raw, ensure_ascii=False)
    elif isinstance(eval_raw, str):
        eval_text = eval_raw

    # 从文本中匹配 ROE 数值
    roe_matches = re.findall(r'ROE[：:=\s]*(\d+\.?\d*)\s*%', eval_text, re.IGNORECASE)
    if roe_matches:
        try:
            roe_val = float(roe_matches[0])
            if roe_val > 20:
                roe_score = 2
            elif roe_val > 10:
                roe_score = 1
        except ValueError:
            pass

    # 从文本中匹配股息率
    div_matches = re.findall(r'(?:股息率|分红率|dividend)[：:=\s]*(\d+\.?\d*)\s*%', eval_text, re.IGNORECASE)
    if div_matches:
        try:
            div_val = float(div_matches[0])
            if div_val > 3:
                div_score = 2
            elif div_val > 1:
                div_score = 1
        except ValueError:
            pass

    # 优先从 akshare 财务指标取结构化数据（更精确）
    fin = data.get("akshare", {}).get("financials", {}).get("latest", {})
    if fin.get("roe") is not None:
        try:
            roe_val = float(fin["roe"])
            if roe_val > 20:
                roe_score = 2
            elif roe_val > 10:
                roe_score = 1
        except (ValueError, TypeError):
            pass
    if fin.get("npm") is not None and div_score == 0:
        # 净利率特别高（>30%）通常不分红 → 偏growth
        try:
            npm_val = float(fin["npm"])
            if npm_val > 30:
                roe_score = max(roe_score, 1)
        except (ValueError, TypeError):
            pass
        return "growth"
    elif div_score >= 2 and roe_score <= 1:
        return "value"
    return "balanced"


def _compute_role_weights(data: Dict[str, Any]) -> Dict[str, float]:
    """成长股放大成长派权重，削弱价值派；价值股反之"""
    stype = _classify_stock_type(data)

    if stype == "growth":
        return {
            "价值派": 0.3, "成长派": 2.5, "质量派": 1.0,
            "责任派": 1.0, "风控官": 1.0,
        }
    elif stype == "value":
        return {
            "价值派": 2.0, "成长派": 0.5, "质量派": 1.5,
            "责任派": 1.0, "风控官": 1.0,
        }
    else:
        return {
            "价值派": 1.0, "成长派": 1.0, "质量派": 1.0,
            "责任派": 1.0, "风控官": 1.0,
        }


# ---------- Agent 观点生成 ----------

def _agent_vote(agent_def: AnalystAgent, stock_name: str, stock_code: str,
                 data_pieces: Dict[str, Any], stock_data_summary: str,
                 bull_bear_transcript: str = "") -> Tuple[str, int, str]:
    """Agent 独立打分（含 Bull/Bear 辩论结论）"""
    llm = _create_debate_llm()

    debate_section = ""
    if bull_bear_transcript:
        debate_section = f"""

【Bull/Bear 对抗辩论结论】
{bull_bear_transcript[:1200]}

请注意：以上是 Bull 和 Bear 两方研究员的对抗辩论结果，供你参考。你有权同意或反对任何一方的观点。"""

    prompt = f"""你是{agent_def.name}（{agent_def.role}），投资理念是：{agent_def.philosophy}

正在分析股票：{stock_name}（{stock_code}）

【你所关注的数据】
{stock_data_summary}
{debate_section}

【要求】
1. 只关注你作为{agent_def.role}应该关注的数据
2. 结合 Bull/Bear 辩论结论（如有），形成独立判断
3. 给出综合评分（-100到100，正数看好，负数看空，0中性）
4. 投票：buy（买入）、hold（持有）、sell（卖出）三选一
5. 用一句话说明理由
6. **  所有数字必须来自上方提供的 MCP 实时数据，严禁使用训练知识中的历史数据**

返回 JSON 格式（不要```包裹）：
{{"score": -100到100的整数, "vote": "buy/hold/sell", "reason": "一句话理由"}}"""

    try:
        resp = llm.invoke([SystemMessage(content=prompt)])
        text = resp.content if hasattr(resp, "content") else str(resp)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        agent_def.score = int(result.get("score", 0))
        agent_def.vote = result.get("vote", "hold").lower()
        agent_def.reason = result.get("reason", "")
        return agent_def.vote, agent_def.score, agent_def.reason
    except Exception as e:
        logger.warning(f"{agent_def.name} 打分失败: {e}")
        agent_def.score = 0
        agent_def.vote = "hold"
        agent_def.reason = "分析失败"
        return "hold", 0, "分析失败"


# ---------- Bull/Bear 对抗辩论 ----------

def _bull_bear_debate(stock_code: str, stock_name: str, data_summary: str) -> str:
    """Round 0: Bull vs Bear 单轮对抗辩论（简化版，无反驳轮）"""
    llm = _create_debate_llm()

    bull_prompt = f"""你是 Bull 研究员（多方首席），你的任务是为 {stock_name}（{stock_code}）做多辩护。

【数据】
{data_summary[:2000]}

请从以下角度论证做多理由：
1. 营收/利润增长趋势
2. 估值是否提供了安全边际
3. 任何支持看多的数据信号

输出 Markdown 格式的做多报告，字数 200 字以内。
**  所有数据必须来自上方提供的 MCP 实时数据，严禁使用训练知识中的历史数字。**"""

    bear_prompt = f"""你是 Bear 研究员（空方首席），你的任务是为 {stock_name}（{stock_code}）做空辩护。

【数据】
{data_summary[:2000]}

请从以下角度论证做空理由：
1. 估值泡沫风险
2. 盈利能力下滑信号
3. 任何支持看空的数据信号

输出 Markdown 格式的做空报告，字数 200 字以内。
**  所有数据必须来自上方提供的 MCP 实时数据，严禁使用训练知识中的历史数字。**"""

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_bull = pool.submit(lambda: llm.invoke([SystemMessage(content=bull_prompt)]))
        f_bear = pool.submit(lambda: llm.invoke([SystemMessage(content=bear_prompt)]))
        bull_report = f_bull.result().content
        bear_report = f_bear.result().content

    debate_transcript = f"""## Bull/Bear 对抗辩论

###   多方（Bull）观点
{bull_report[:500]}

###   空方（Bear）观点
{bear_report[:500]}"""

    logger.info(f"  Bull/Bear 辩论完成: {stock_code}")
    return debate_transcript


# ---------- 构建数据摘要 ----------

def _build_data_summary(stock_data: Dict[str, Any], sentiment: Dict[str, Any]) -> str:
    """将股票数据 + 舆情 + 板块趋势拼接为文本摘要

    ROE/ROIC/毛利率/净利率/股息率 全部包含在 stk_eval 的综合评估文本中。
    """
    parts = []
    d = stock_data

    if d.get("market"):
        m = d["market"]
        parts.append(f"市值: {m.get('total_market_cap', m.get('close_price', '?'))}, 收盘: {m.get('close_price', '?')}")
    if d.get("eval"):
        parts.append(f"综合评估（含ROE/ROIC/GPM/NPM/股息率等财务指标）: {json.dumps(d['eval'], ensure_ascii=False)[:1500]}")
    if d.get("dcf"):
        parts.append(f"DCF估值: {json.dumps(d['dcf'], ensure_ascii=False)[:300]}")
    if d.get("esg_m"):
        parts.append(f"妙盈ESG: {json.dumps(d['esg_m'], ensure_ascii=False)[:150]}")
    if d.get("esg_c"):
        parts.append(f"华证ESG: {json.dumps(d['esg_c'], ensure_ascii=False)[:150]}")
    if d.get("esg_s"):
        parts.append(f"商道融绿ESG: {json.dumps(d['esg_s'], ensure_ascii=False)[:150]}")
    parts.append(f"市场舆情: {json.dumps(sentiment, ensure_ascii=False)[:300]}")
    if d.get("survey"):
        parts.append(f"机构调研: {json.dumps(d['survey'], ensure_ascii=False)[:150]}")

    # AKShare 补充数据
    ak = d.get("akshare", {})
    fin = ak.get("financials", {}).get("latest", {})
    if fin:
        parts.append(f"最新财务指标（AKShare）: {json.dumps(fin, ensure_ascii=False)[:400]}")
    val = ak.get("valuation", {})
    if val:
        parts.append(f"估值数据（AKShare）: {json.dumps(val, ensure_ascii=False)[:200]}")
    stock_info = ak.get("stock_info", {})
    if stock_info.get("industry"):
        parts.append(f"行业: {stock_info['industry']}")

    sector_info = fetch_sector_trend()
    if sector_info:
        parts.append(f"板块趋势: {json.dumps(sector_info, ensure_ascii=False)[:300]}")

    return "\n".join(parts)


# ---------- 单只股票辩论主流程 ----------

def debate_stock(stock_code: str) -> Dict[str, Any]:
    """对单只股票执行辩论投票

    简化流程（5 Agent + 单轮投票 + Bull/Bear 对抗）：
      1. 获取数据
      2. Bull/Bear 单轮对抗（2 LLM 并行）
      3. 5 Agent 独立打分（5 LLM 并行）
      4. 角色权重加权 → 最终裁决

    Returns:
        {stock_code, stock_name, stock_type, buy_signal, final_score,
         bull_bear_debate, agent_votes, role_weights, weight_breakdown, vote_summary, sentiment}
    """
    analysts = _build_analysts()
    logger.info(f"开始 5 Agent 辩论: {stock_code}")

    # 获取数据（统一从 data_layer）
    stock_data = fetch_stock_data(stock_code)
    stock_name = stock_data.get("name", stock_code)
    sentiment = fetch_sentiment()
    data_summary = _build_data_summary(stock_data, sentiment)

    # Bull/Bear 单轮对抗（2 LLM 并行）
    logger.info(f"  Bull vs Bear 对抗辩论...")
    bull_bear_transcript = _bull_bear_debate(stock_code, stock_name, data_summary)

    # 5 Agent 独立打分（并行）
    logger.info(f"  5 Agent 独立打分...")
    agent_votes: Dict[str, Dict] = {}

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {}
        for a in analysts:
            futures[pool.submit(_agent_vote, a, stock_name, stock_code, {},
                                data_summary, bull_bear_transcript)] = a.name

        for f in as_completed(futures):
            name = futures[f]
            vote, score, reason = f.result()
            agent_votes[name] = {"score": score, "vote": vote, "reason": reason}

    # 动态角色权重
    role_weights = _compute_role_weights(stock_data)

    # 最终加权裁决
    weighted_votes = {"buy": 0, "hold": 0, "sell": 0}
    total_weight = 0
    total_score = 0
    weight_breakdown = {}

    for name, v in agent_votes.items():
        rw = 1.0
        for role_prefix, w in role_weights.items():
            if name.startswith(role_prefix):
                rw = w
                break
        final_w = round(rw, 2)
        weight_breakdown[name] = {"role_weight": rw, "final_weight": final_w}

        weighted_votes[v["vote"]] += final_w
        total_weight += final_w
        total_score += v["score"] * final_w

    final_score = round(total_score / total_weight, 1) if total_weight > 0 else 0
    buy_signal = weighted_votes["buy"] > (total_weight / 2)

    result = {
        "stock_code": stock_code,
        "stock_name": stock_name,
        "stock_type": _classify_stock_type(stock_data),
        "buy_signal": buy_signal,
        "final_score": final_score,
        "bull_bear_debate": bull_bear_transcript,
        "agent_votes": agent_votes,
        "role_weights": {k: round(v, 1) for k, v in role_weights.items()},
        "weight_breakdown": weight_breakdown,
        "vote_summary": {
            "buy_weight": round(weighted_votes["buy"], 1),
            "hold_weight": round(weighted_votes["hold"], 1),
            "sell_weight": round(weighted_votes["sell"], 1),
            "total": round(total_weight, 1),
            "buy_ratio": round(weighted_votes["buy"] / total_weight * 100, 1) if total_weight > 0 else 0,
        },
        "sentiment": sentiment,
    }

    signal = "买入提醒" if buy_signal else "无买入信号"
    logger.info(f"  辩论完成: {stock_name}({stock_code}) [{result['stock_type']}] → {signal}, "
                f"最终得分{final_score}, buy{weighted_votes['buy']}, hold{weighted_votes['hold']}, "
                f"sell{weighted_votes['sell']}")

    return result


def debate_batch(stock_codes: List[str]) -> List[Dict[str, Any]]:
    """并行分析多只股票，返回按 buy_signal 排序的辩论结果列表"""
    results = []
    with ThreadPoolExecutor(max_workers=min(len(stock_codes), 2)) as pool:
        futures = {pool.submit(debate_stock, c): c for c in stock_codes[:10]}
        for f in as_completed(futures):
            results.append(f.result())

    results.sort(key=lambda r: (r["buy_signal"], r["final_score"]), reverse=True)
    return results
