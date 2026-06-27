"""
RAG（检索增强生成）模块

使用 ChromaDB 作为向量数据库，存储金融领域知识。
启动时自动初始化知识库。
"""

import logging
import os
from typing import List, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config import OPENAI_API_KEY, OPENAI_BASE_URL

logger = logging.getLogger(__name__)

# 向量数据库持久化路径
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")

# ---------- 金融知识库内容 ----------
FINANCE_KNOWLEDGE = [
    {
        "content": """
DCF（Discounted Cash Flow，现金流折现法）是一种绝对估值方法。其核心思想是：一家公司的内在价值等于其未来所有自由现金流的现值之和。
DCF估值步骤：
1. 预测公司未来5-10年的自由现金流
2. 选择合适的折现率（通常使用WACC，加权平均资本成本）
3. 计算终值（永续增长法或退出倍数法）
4. 将预测期现金流和终值折现到当前，求和得到企业价值
5. 企业价值减去净负债得到股权价值
6. 股权价值除以总股本得到每股内在价值
""",
        "metadata": {"topic": "估值方法", "key": "DCF"}
    },
    {
        "content": """
ROE（Return on Equity，净资产收益率）是衡量公司盈利能力的重要指标。计算公式：ROE = 净利润 / 平均净资产 × 100%。
ROE反映了股东权益的回报水平。巴菲特认为ROE是衡量公司质量的最重要指标之一。
一般来说，ROE持续高于15%的公司具有较强的护城河和竞争优势。
ROE可通过杜邦分析拆解为三部分：净利率 × 总资产周转率 × 权益乘数。
""",
        "metadata": {"topic": "财务指标", "key": "ROE"}
    },
    {
        "content": """
ROIC（Return on Invested Capital，投入资本回报率）衡量公司使用投入资本的效率。
计算公式：ROIC = NOPAT（税后净营业利润）/ Invested Capital（投入资本）。
ROIC > WACC（加权平均资本成本）说明公司在创造价值，反之在毁灭价值。
高ROIC且能持续保持是优秀企业的重要特征。
""",
        "metadata": {"topic": "财务指标", "key": "ROIC"}
    },
    {
        "content": """
毛利率（Gross Profit Margin）=（营业收入 - 营业成本）/ 营业收入 × 100%。
毛利率反映了公司产品或服务的定价能力和成本控制能力。
高毛利率通常意味着公司有较强的品牌溢价或技术壁垒。
不同行业的毛利率差异很大：消费品通常30%-60%，软件行业可达70%-90%，制造业可能10%-30%。
""",
        "metadata": {"topic": "财务指标", "key": "毛利率"}
    },
    {
        "content": """
净利率（Net Profit Margin）= 净利润 / 营业收入 × 100%。
净利率反映了公司将所有收入转化为净利润的能力，包含了所有成本和费用。
净利率高说明公司整体运营效率高，费用控制得当。
行业特性对净利率影响很大，应同行业比较。
""",
        "metadata": {"topic": "财务指标", "key": "净利率"}
    },
    {
        "content": """
股息率（Dividend Yield）= 每股股息 / 每股股价 × 100%。
股息率是衡量股票投资回报的指标之一。
高股息率通常出现在成熟型公司（如银行、公用事业），这些公司增长空间有限，更倾向将利润分配给股东。
投资高股息率股票需注意：股息是否可持续、公司是否有足够现金流支撑分红。
""",
        "metadata": {"topic": "财务指标", "key": "股息率"}
    },
    {
        "content": """
ESG是Environmental（环境）、Social（社会）和Governance（治理）的缩写，是衡量企业可持续发展的非财务指标框架。
- E（环境）：碳排放、能源使用、废物处理、生物多样性等
- S（社会）：员工福利、供应链管理、社区关系、产品责任等
- G（治理）：董事会结构、高管薪酬、股东权利、商业道德等

ESG评级帮助投资者识别非财务风险，越来越多的机构投资者将ESG纳入投资决策。
主要ESG评级机构：MSCI、Sustainalytics、妙盈科技、华证指数、商道融绿等。
""",
        "metadata": {"topic": "ESG", "key": "ESG概念"}
    },
    {
        "content": """
中国A股市场交易代码规则：
- 上海证券交易所（SH）：主板代码以600、601、603开头
- 深圳证券交易所（SZ）：主板代码以000、001开头；创业板以300开头
- 北京证券交易所（BJ）：代码以8开头（如83、87等）
- 科创板（上海）：代码以688开头
- 港股（HK）：代码为5位数字，如00700（腾讯控股）
""",
        "metadata": {"topic": "基础知识", "key": "股票代码"}
    },
    {
        "content": """
机构调研是投资者关系活动的重要组成部分。当基金公司、券商等机构投资者对上市公司进行实地调研或电话会议时，会上传调研纪要。
调研信息对投资者有重要参考价值：
- 反映机构关注度：机构调研数量多说明公司受关注程度高
- 获取公司经营信息：调研纪要中包含管理层对经营情况的说明
- 提前发现趋势：机构调研动向可能预示板块轮动
""",
        "metadata": {"topic": "投资方法", "key": "机构调研"}
    },
    {
        "content": """
PE（市盈率）= 股价 / 每股收益。反映市场愿意为每1元盈利支付的价格。
PB（市净率）= 股价 / 每股净资产。反映股价相对于每股净资产的溢价程度。
PS（市销率）= 总市值 / 营业收入。适用于高增长但尚未盈利的公司估值。

相对估值法（PE、PB、PS）与绝对估值法（DCF）配合使用效果更好。
""",
        "metadata": {"topic": "估值方法", "key": "相对估值"}
    },
    {
        "content": """
价值投资核心理念：
1. 买股票就是买公司——关注公司内在价值而非股价波动
2. 安全边际——以低于内在价值的价格买入
3. 能力圈——只投资自己理解的行业和公司
4. 市场先生——市场短期是投票机，长期是称重机

价值投资经典指标：低PE、低PB、高ROE、高股息率、稳定现金流。
""",
        "metadata": {"topic": "投资方法", "key": "价值投资"}
    },
]


# ---------- RAG 初始化 ----------
_vectorstore: Chroma = None
_rag_disabled: bool = False  # 一次性降级标记，避免重复尝试


def _get_embeddings():
    """获取嵌入模型（兼容 DeepSeek API）"""
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    kwargs = {"model": model}
    if OPENAI_API_KEY:
        kwargs["openai_api_key"] = OPENAI_API_KEY
    if OPENAI_BASE_URL:
        url = OPENAI_BASE_URL.rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        kwargs["openai_api_base"] = url
    return OpenAIEmbeddings(**kwargs)


def init_rag(force_rebuild: bool = False):
    """初始化RAG知识库，失败时返回None优雅降级"""
    global _vectorstore, _rag_disabled

    if _rag_disabled and not force_rebuild:
        return None

    if _vectorstore is not None and not force_rebuild:
        return _vectorstore

    try:
        embeddings = _get_embeddings()

        if os.path.exists(CHROMA_PERSIST_DIR) and not force_rebuild:
            try:
                _vectorstore = Chroma(
                    persist_directory=CHROMA_PERSIST_DIR,
                    embedding_function=embeddings,
                )
                logger.info(f"RAG知识库已加载，文档数: {_vectorstore._collection.count()}")
                return _vectorstore
            except Exception as e:
                logger.warning(f"加载已有知识库失败: {e}，将重建")

        docs = []
        for item in FINANCE_KNOWLEDGE:
            doc = Document(page_content=item["content"].strip(), metadata=item["metadata"])
            docs.append(doc)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
        split_docs = splitter.split_documents(docs)
        logger.info(f"RAG知识库: {len(docs)} 篇文档 → {len(split_docs)} 个文本块")

        _vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        logger.info("RAG知识库构建完成并已持久化")
        return _vectorstore

    except Exception as e:
        logger.warning(f"RAG初始化失败（可能Embedding API不可用）: {e}，将禁用RAG功能")
        _vectorstore = None
        _rag_disabled = True
        return None


def retrieve_knowledge(query: str, k: int = 3) -> List[Tuple[str, dict]]:
    """从知识库检索相关内容

    Returns:
        List[Tuple[str, dict]]: (内容, 元数据) 列表
    """
    vectorstore = init_rag()
    if vectorstore is None:
        return []

    try:
        docs = vectorstore.similarity_search(query, k=k)
        return [(doc.page_content, doc.metadata) for doc in docs]
    except Exception as e:
        logger.warning(f"RAG检索失败（Embedding可能不兼容）: {e}，已降级")
        init_rag.cache_clear() if hasattr(init_rag, "cache_clear") else None
        return []


def retrieve_knowledge_as_context(query: str, k: int = 3) -> str:
    """检索并格式化为上下文字符串"""
    results = retrieve_knowledge(query, k=k)
    if not results:
        return ""

    context_parts = []
    for i, (content, meta) in enumerate(results, 1):
        topic = meta.get("topic", "未知主题")
        context_parts.append(f"[参考{i}]（{topic}）: {content}")

    return "\n\n".join(context_parts)
