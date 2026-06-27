"""
8-Agent 辩论投票系统（兼容层）

逻辑已迁移到 src/analysts.py 和 src/data_layer.py。
此文件保留向后兼容的 re-export。
"""
from .analysts import (
    AnalystAgent,
    debate_stock,
    debate_batch,
)

__all__ = ["AnalystAgent", "debate_stock", "debate_batch"]
