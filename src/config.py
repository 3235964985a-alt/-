"""
系统配置模块
"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# MCP Server 配置
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")

# -- 金融工具元数据（定义所有可用MCP工具的参数schema） --
MCP_TOOLS_META = [
    {
        "name": "stk_market_value",
        "description": "查询个股市值信息，包括收盘价、总市值、总股本等",
        "parameters": {
            "type": "object",
            "properties": {
                "security_code": {"type": "string", "description": "证券代码，如 600519"}
            },
            "required": ["security_code"]
        }
    },
    {
        "name": "stk_dcf",
        "description": "个股DCF估值诊断，返回估值分析和投资建议",
        "parameters": {
            "type": "object",
            "properties": {
                "security_code": {"type": "string", "description": "证券代码，如 600519"}
            },
            "required": ["security_code"]
        }
    },
    {
        "name": "stk_eval",
        "description": "个股综合评估，提供多维度分析建议",
        "parameters": {
            "type": "object",
            "properties": {
                "security_code": {"type": "string", "description": "证券代码，如 600519"}
            },
            "required": ["security_code"]
        }
    },
    {
        "name": "stk_survey",
        "description": "个股调研信息，获取机构调研详情",
        "parameters": {
            "type": "object",
            "properties": {
                "security_code": {"type": "string", "description": "证券代码，如 600519"}
            },
            "required": ["security_code"]
        }
    },
    {
        "name": "miotech_esg_rating",
        "description": "妙盈科技ESG评级查询",
        "parameters": {
            "type": "object",
            "properties": {
                "security_code": {"type": "string", "description": "证券代码，如 600519"}
            },
            "required": ["security_code"]
        }
    },
    {
        "name": "chindices_esg_rating",
        "description": "华证指数ESG评级查询",
        "parameters": {
            "type": "object",
            "properties": {
                "security_code": {"type": "string", "description": "证券代码，如 600519"}
            },
            "required": ["security_code"]
        }
    },
    {
        "name": "syntaogf_esg_rating",
        "description": "商道融绿ESG评级查询",
        "parameters": {
            "type": "object",
            "properties": {
                "security_code": {"type": "string", "description": "证券代码，如 600519"}
            },
            "required": ["security_code"]
        }
    },
]
_FILTER_PARAMS = {
    "type": "object",
    "properties": {
        "filter_value": {"type": "number", "description": "筛选阈值（百分比，如 15 表示 ROE≥15%）"},
        "filter_type": {
            "type": "integer",
            "description": "筛选类型：1=大于, 2=大于等于, 3=小于, 4=小于等于, 5=等于",
        },
    },
    "required": ["filter_value", "filter_type"],
}

_FILTER_TOOLS = [
    ("stk_eval_filter_by_roe_1y",    "按1年ROE筛选股票"),
    ("stk_eval_filter_by_roe_3y",    "按3年ROE筛选股票"),
    ("stk_eval_filter_by_roic_1y",   "按1年ROIC筛选股票"),
    ("stk_eval_filter_by_roic_3y",   "按3年ROIC筛选股票"),
    ("stk_eval_filter_by_gpm_1y",    "按1年毛利率筛选股票"),
    ("stk_eval_filter_by_gpm_3y",    "按3年毛利率筛选股票"),
    ("stk_eval_filter_by_npm_1y",    "按1年净利率筛选股票"),
    ("stk_eval_filter_by_npm_3y",    "按3年净利率筛选股票"),
    ("stk_eval_filter_by_div_rate",  "按股息率筛选股票"),
]

MCP_TOOLS_META = MCP_TOOLS_META + [
    {"name": name, "description": desc, "parameters": _FILTER_PARAMS}
    for name, desc in _FILTER_TOOLS
]
