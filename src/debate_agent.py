"""
7-Agent 辩论投票系统 —— 囚徒困境博弈

架构：
  Round 1: 7 Agent 并行独立打分（-100~100），各投 buy/hold/sell
  Round 2: 公布所有人打分 → 囚徒困境博弈（合作 vs 背叛）
  最终裁决：合作者 1.5x 权重，buy > 50%（≥4/7）→ 买入提醒

Agent:
  1. 价值 → DCF 估值
  2. 成长 → ROE/ROIC 筛选
  3. 质量 → GPM/NPM/股息率
  4. 市值 → 市值+调研
  5. ESG  → 三机构评级
  6. 情绪 → 市场舆情
  7. 风险 → 纯推理综合
"""
import json
import logging
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from .config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from .mcp_tools import _call_mcp_tool_sync

logger = logging.getLogger(__name__)


# ---------- 7 位分析师定义 ----------

class AnalystAgent:
    def __init__(self, name: str, role: str, tools: Dict[str, Tuple[str, dict]], philosophy: str):
        self.name = name
        self.role = role
        self.tools = tools  # {tool_name: (description, args)}
        self.philosophy = philosophy
        self.round1_score = 0
        self.round1_vote = "hold"
        self.round1_reason = ""
        self.round2_score = 0
        self.round2_vote = "hold"
        self.round2_cooperate = False  # 合作=向多数派靠拢
        self.round2_reason = ""


# ---------- 创建 LLM ----------

def _create_debate_llm() -> ChatOpenAI:
    kwargs = {"model": OPENAI_MODEL, "temperature": 0}
    if OPENAI_API_KEY:
        kwargs["openai_api_key"] = OPENAI_API_KEY
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return ChatOpenAI(**kwargs)


# ---------- 工具调用 ----------

def _safe_json(data):
    if isinstance(data, dict):
        return data
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return {"text": str(data)[:500]}


def _fetch_stock_data(code: str) -> Dict[str, Any]:
    """并行获取一只股票的所有维度数据"""
    data = {"code": code, "name": code}

    def _try_fetch(tool_name, args):
        try:
            return _safe_json(_call_mcp_tool_sync(tool_name, args))
        except Exception:
            return {}

    with ThreadPoolExecutor(max_workers=8) as pool:
        tasks = {
            "market": pool.submit(_try_fetch, "stk_market_value", {"security_code": code}),
            "eval": pool.submit(_try_fetch, "stk_eval", {"security_code": code}),
            "survey": pool.submit(_try_fetch, "stk_survey", {"security_code": code}),
            "dcf": pool.submit(_try_fetch, "stk_dcf", {"security_code": code}),
            "roe": pool.submit(_try_fetch, "stk_eval_filter_by_roe_1y", {"security_code": code}),
            "roic": pool.submit(_try_fetch, "stk_eval_filter_by_roic_1y", {"security_code": code}),
            "gpm": pool.submit(_try_fetch, "stk_eval_filter_by_gpm_1y", {"security_code": code}),
            "npm": pool.submit(_try_fetch, "stk_eval_filter_by_npm_1y", {"security_code": code}),
            "div": pool.submit(_try_fetch, "stk_eval_filter_by_div_rate", {"security_code": code}),
            "esg_m": pool.submit(_try_fetch, "miotech_esg_rating", {"security_code": code}),
            "esg_c": pool.submit(_try_fetch, "chindices_esg_rating", {"security_code": code}),
            "esg_s": pool.submit(_try_fetch, "syntaogf_esg_rating", {"security_code": code}),
        }

        for key, future in tasks.items():
            data[key] = future.result()

    # 取股票名称
    if data["market"]:
        data["name"] = data["market"].get("security_name", code)

    return data


def _fetch_sentiment() -> Dict[str, Any]:
    """获取市场舆情情绪"""
    try:
        from .news_mcp import get_news_sentiment
        raw = get_news_sentiment()
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return {"overall_score": 0, "overall_label": "无数据"}


# ---------- Agent 观点生成 ----------

def _agent_round1(agent_def: AnalystAgent, stock_name: str, stock_code: str,
                   data_pieces: Dict[str, Any], stock_data_summary: str) -> Tuple[str, int, str]:
    """Round 1: 独立打分"""
    llm = _create_debate_llm()

    prompt = f"""你是{agent_def.name}（{agent_def.role}），投资理念是：{agent_def.philosophy}

正在分析股票：{stock_name}（{stock_code}）

【你所关注的数据】
{stock_data_summary}

【要求】
1. 只关注你作为{agent_def.role}应该关注的数据
2. 给出综合评分（-100到100，正数看好，负数看空，0中性）
3. 投票：buy（买入）、hold（持有）、sell（卖出）三选一
4. 用一句话说明理由

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
        agent_def.round1_score = int(result.get("score", 0))
        agent_def.round1_vote = result.get("vote", "hold").lower()
        agent_def.round1_reason = result.get("reason", "")
        return agent_def.round1_vote, agent_def.round1_score, agent_def.round1_reason
    except Exception as e:
        logger.warning(f"{agent_def.name} Round1 失败: {e}")
        agent_def.round1_score = 0
        agent_def.round1_vote = "hold"
        agent_def.round1_reason = "分析失败"
        return "hold", 0, "分析失败"


def _agent_round2(agent_def: AnalystAgent, stock_name: str, stock_code: str,
                   all_r1_scores: Dict[str, Tuple[str, int, str]],
                   stock_data_summary: str) -> Tuple[str, int, str, bool]:
    """Round 2: 囚徒困境 — 看到别人的分后，决定合作还是背叛"""
    llm = _create_debate_llm()

    # 计算多数派
    votes = [v for v, _, _ in all_r1_scores.values()]
    buy_count = votes.count("buy")
    sell_count = votes.count("sell")
    hold_count = votes.count("hold")
    total = len(votes) or 1

    # 多数派方向
    if buy_count >= sell_count and buy_count >= hold_count:
        majority = "buy"
    elif sell_count >= buy_count and sell_count >= hold_count:
        majority = "sell"
    else:
        majority = "hold"

    others_summary = "\n".join(
        f"  {name}: 评分{score}, 投票{vote}, 理由:{reason}"
        for name, (vote, score, reason) in all_r1_scores.items()
        if name != agent_def.name
    )

    prompt = f"""你是{agent_def.name}（{agent_def.role}）。

【股票】{stock_name}（{stock_code}）

【你 Round 1 的观点】
评分: {agent_def.round1_score}
投票: {agent_def.round1_vote}
理由: {agent_def.round1_reason}

【其他分析师的 Round 1 观点】
{others_summary}

【多数派方向】{majority}（buy {buy_count}票, sell {sell_count}票, hold {hold_count}票）

【囚徒困境规则】
现在你可以看到所有人的打分。
- **合作**：向多数派方向靠拢（修正你的评分），获得 1.5x 投票权重
- **背叛**：坚持你 Round 1 的判断不变，保持 1.0x 权重

合作的好处是你的投票更有影响力；风险是如果多数派错了，你也要跟着错。
背叛的好处是你保持独立判断；风险是你的投票权重低，影响力小。

【再次审视数据】
{stock_data_summary}

返回 JSON：
{{"score": 最终评分(-100~100), "vote": "buy/hold/sell", "cooperate": true/false, "reason": "一句话说明为何合作或背叛"}}"""

    try:
        resp = llm.invoke([SystemMessage(content=prompt)])
        text = resp.content if hasattr(resp, "content") else str(resp)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        agent_def.round2_score = int(result.get("score", agent_def.round1_score))
        agent_def.round2_vote = result.get("vote", agent_def.round1_vote).lower()
        agent_def.round2_cooperate = bool(result.get("cooperate", False))
        agent_def.round2_reason = result.get("reason", "")
        return agent_def.round2_vote, agent_def.round2_score, agent_def.round2_reason, agent_def.round2_cooperate
    except Exception as e:
        logger.warning(f"{agent_def.name} Round2 失败: {e}")
        agent_def.round2_score = agent_def.round1_score
        agent_def.round2_vote = agent_def.round1_vote
        agent_def.round2_cooperate = False
        agent_def.round2_reason = "博弈分析失败，坚持原判"
        return agent_def.round1_vote, agent_def.round1_score, agent_def.round1_reason, False


# ---------- 7 位分析师定义 ----------

def _build_analysts() -> List[AnalystAgent]:
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
            name="市场派·林奇",
            role="市场分析师",
            tools={"stk_market_value": "市值信息", "stk_survey": "机构调研"},
            philosophy="市值反映市场共识。被机构频繁调研的公司往往有催化剂。关注市值与行业地位的匹配度。"
        ),
        AnalystAgent(
            name="责任派·ESG分析师",
            role="ESG分析师",
            tools={"miotech_esg_rating": "妙盈ESG", "chindices_esg_rating": "华证ESG", "syntaogf_esg_rating": "商道融绿ESG"},
            philosophy="环境社会和治理表现优秀的企业长期更有韧性。ESG评级高的企业风险更低，值得长期持有。"
        ),
        AnalystAgent(
            name="情绪派·新闻猎手",
            role="舆情分析师",
            tools={"get_news_sentiment": "市场情绪评分"},
            philosophy="市场情绪是短期风向标。恐慌时贪婪，狂热时恐惧。情绪极度负面的优质股可能是抄底机会。"
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
    """根据财务数据判断股票类型：growth / value / balanced"""
    roe_score = 0
    div_score = 0

    # ROE 高 → 成长型
    roe_raw = data.get("roe", {})
    if isinstance(roe_raw, dict):
        roe_val = roe_raw.get("roe_1y", roe_raw.get("roe", 0)) or 0
        try:
            if float(roe_val) > 20:
                roe_score = 2
            elif float(roe_val) > 10:
                roe_score = 1
        except (ValueError, TypeError):
            pass

    # 股息率高 → 价值型
    div_raw = data.get("div", {})
    if isinstance(div_raw, dict):
        div_val = div_raw.get("div_rate", div_raw.get("dividend_yield", 0)) or 0
        try:
            if float(div_val) > 3:
                div_score = 2
            elif float(div_val) > 1:
                div_score = 1
        except (ValueError, TypeError):
            pass

    if roe_score >= 2 and div_score <= 1:
        return "growth"
    elif div_score >= 2 and roe_score <= 1:
        return "value"
    return "balanced"


def _compute_role_weights(data: Dict[str, Any]) -> Dict[str, float]:
    """根据股票类型动态分配各角色权重

    成长股：放大成长派/市场派，削弱价值派/质量派
    价值股：放大价值派/质量派，削弱成长派
    平衡型：等权
    """
    stype = _classify_stock_type(data)

    if stype == "growth":
        return {
            "价值派": 0.5,   # DCF对高增长股天然不利
            "成长派": 2.0,   # ROE/ROIC 是关键
            "质量派": 0.5,   # 成长股通常不分红
            "市场派": 1.5,   # 市场共识对成长股重要
            "责任派": 1.0,
            "情绪派": 1.0,
            "风控官": 1.0,
        }
    elif stype == "value":
        return {
            "价值派": 2.0,
            "成长派": 0.5,
            "质量派": 1.5,
            "市场派": 1.0,
            "责任派": 1.0,
            "情绪派": 1.0,
            "风控官": 1.0,
        }
    else:  # balanced
        return {
            "价值派": 1.0,
            "成长派": 1.0,
            "质量派": 1.0,
            "市场派": 1.0,
            "责任派": 1.0,
            "情绪派": 1.0,
            "风控官": 1.0,
        }


# ---------- 辩论主流程 ----------

def debate_stock(stock_code: str) -> Dict[str, Any]:
    """对单只股票执行 7 Agent 两轮囚徒困境辩论

    Returns:
        {
            "stock_code": "600519",
            "stock_name": "贵州茅台",
            "buy_signal": True/False,   # buy票 > 50%
            "final_score": 加权平均分,
            "round1": {agent_name: {score, vote, reason}},
            "round2": {agent_name: {score, vote, cooperate, reason}},
            "vote_summary": {buy: n, hold: n, sell: n}
        }
    """
    analysts = _build_analysts()
    logger.info(f"开始 7 Agent 辩论: {stock_code}")

    # 获取数据
    stock_data = _fetch_stock_data(stock_code)
    stock_name = stock_data.get("name", stock_code)
    sentiment = _fetch_sentiment()

    # 构建数据摘要（Round 1 传给各Agent）
    d = stock_data
    data_summary_parts = []
    if d.get("market"):
        m = d["market"]
        data_summary_parts.append(f"市值: {m.get('total_market_cap', m.get('close_price', '?'))}, 收盘: {m.get('close_price', '?')}")
    if d.get("eval"):
        e = d["eval"]
        data_summary_parts.append(f"综合评估: {json.dumps(e, ensure_ascii=False)[:200]}")
    if d.get("dcf"):
        data_summary_parts.append(f"DCF估值: {json.dumps(d['dcf'], ensure_ascii=False)[:200]}")
    if d.get("roe"):
        data_summary_parts.append(f"ROE数据: {json.dumps(d['roe'], ensure_ascii=False)[:200]}")
    if d.get("roic"):
        data_summary_parts.append(f"ROIC数据: {json.dumps(d['roic'], ensure_ascii=False)[:200]}")
    if d.get("gpm"):
        data_summary_parts.append(f"毛利率: {json.dumps(d['gpm'], ensure_ascii=False)[:150]}")
    if d.get("npm"):
        data_summary_parts.append(f"净利率: {json.dumps(d['npm'], ensure_ascii=False)[:150]}")
    if d.get("div"):
        data_summary_parts.append(f"股息率: {json.dumps(d['div'], ensure_ascii=False)[:150]}")
    if d.get("esg_m"):
        data_summary_parts.append(f"妙盈ESG: {json.dumps(d['esg_m'], ensure_ascii=False)[:150]}")
    if d.get("esg_c"):
        data_summary_parts.append(f"华证ESG: {json.dumps(d['esg_c'], ensure_ascii=False)[:150]}")
    if d.get("esg_s"):
        data_summary_parts.append(f"商道融绿ESG: {json.dumps(d['esg_s'], ensure_ascii=False)[:150]}")
    data_summary_parts.append(f"市场舆情: {json.dumps(sentiment, ensure_ascii=False)[:300]}")
    if d.get("survey"):
        data_summary_parts.append(f"机构调研: {json.dumps(d['survey'], ensure_ascii=False)[:150]}")
    data_summary = "\n".join(data_summary_parts)

    # ═══════════ Round 1: 独立打分 ═══════════
    logger.info(f"  Round 1: 7 Agent 独立打分...")
    round1_results: Dict[str, Tuple[str, int, str]] = {}

    with ThreadPoolExecutor(max_workers=7) as pool:
        futures = {}
        for a in analysts:
            futures[pool.submit(_agent_round1, a, stock_name, stock_code, {}, data_summary)] = a.name

        for f in as_completed(futures):
            name = futures[f]
            vote, score, reason = f.result()
            round1_results[name] = (vote, score, reason)

    # ═══════════ Round 2: 囚徒困境 ═══════════
    logger.info(f"  Round 2: 囚徒困境博弈...")
    round2_results: Dict[str, Dict] = {}

    with ThreadPoolExecutor(max_workers=7) as pool:
        futures2 = {}
        for a in analysts:
            futures2[pool.submit(_agent_round2, a, stock_name, stock_code,
                                 round1_results, data_summary)] = a.name

        for f in as_completed(futures2):
            name = futures2[f]
            vote, score, reason, cooperate = f.result()
            round2_results[name] = {
                "score": score, "vote": vote, "cooperate": cooperate, "reason": reason
            }

    # ═══════════ 动态角色权重 ═══════════
    # 根据股票特征调整各分析师权重，避免对成长/价值股的系统性偏见
    role_weights = _compute_role_weights(stock_data)

    # ═══════════ 最终裁决 ═══════════
    # 合作者加成 × 角色权重
    weighted_votes = {"buy": 0, "hold": 0, "sell": 0}
    total_weight = 0
    total_score = 0
    weight_breakdown = {}  # 记录每个agent的最终权重供展示

    for name, r2 in round2_results.items():
        # 匹配角色权重
        rw = 1.0
        for role_prefix, w in role_weights.items():
            if name.startswith(role_prefix):
                rw = w
                break
        cooperate_bonus = 1.5 if r2["cooperate"] else 1.0
        final_w = round(rw * cooperate_bonus, 2)
        weight_breakdown[name] = {"role_weight": rw, "cooperate_bonus": cooperate_bonus, "final_weight": final_w}

        weighted_votes[r2["vote"]] += final_w
        total_weight += final_w
        total_score += r2["score"] * final_w

    final_score = round(total_score / total_weight, 1) if total_weight > 0 else 0
    buy_signal = weighted_votes["buy"] > (total_weight / 2)  # > 50%

    # 股票类型标签
    stock_type = _classify_stock_type(stock_data)

    # 构建结果
    r1_simple = {name: {"score": s, "vote": v, "reason": r}
                 for name, (v, s, r) in round1_results.items()}

    result = {
        "stock_code": stock_code,
        "stock_name": stock_name,
        "stock_type": stock_type,
        "buy_signal": buy_signal,
        "final_score": final_score,
        "round1": r1_simple,
        "round2": round2_results,
        "role_weights": {k: round(v, 1) for k, v in role_weights.items()},
        "weight_breakdown": weight_breakdown,
        "vote_summary": {
            "buy_weight": round(weighted_votes["buy"], 1),
            "hold_weight": round(weighted_votes["hold"], 1),
            "sell_weight": round(weighted_votes["sell"], 1),
            "total": round(total_weight, 1),
            "buy_ratio": round(weighted_votes["buy"] / total_weight * 100, 1) if total_weight > 0 else 0,
            "cooperators": [n for n, r2 in round2_results.items() if r2["cooperate"]],
            "defectors": [n for n, r2 in round2_results.items() if not r2["cooperate"]],
        },
        "sentiment": sentiment,
    }

    signal = "买入提醒" if buy_signal else "无买入信号"
    logger.info(f"  辩论完成: {stock_name}({stock_code}) [{stock_type}] → {signal}, 最终得分{final_score}, "
                f"buy{weighted_votes['buy']}, hold{weighted_votes['hold']}, sell{weighted_votes['sell']}, "
                f"角色权重: {role_weights}")

    return result


def debate_batch(stock_codes: List[str]) -> List[Dict[str, Any]]:
    """并行分析多只股票

    Returns:
        辩论结果列表，按 buy_signal 排序
    """
    results = []
    with ThreadPoolExecutor(max_workers=min(len(stock_codes), 3)) as pool:
        futures = {pool.submit(debate_stock, c): c for c in stock_codes[:10]}
        for f in as_completed(futures):
            results.append(f.result())

    # 排序：buy信号优先 → 得分高优先
    results.sort(key=lambda r: (r["buy_signal"], r["final_score"]), reverse=True)
    return results
