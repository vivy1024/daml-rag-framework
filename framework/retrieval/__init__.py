# -*- coding: utf-8 -*-
"""
DAML-RAG框架检索模块 v2.0

提供三层检索架构的核心实现：
- 语义检索层（向量检索）
- 图谱检索层（关系推理）
- 约束验证层（专业规则）

支持多种检索策略：
- 向量检索（BGE、OpenAI等）
- 图检索（Neo4j、JanusGraph等）
- 混合检索（多源融合）
- 约束验证（安全检查、专业规则）

版本：v2.0.0
更新日期：2025-11-17
"""

# 核心检索实现
from .three_layer_retriever import (
    ThreeLayerRetriever,
    RetrievalLayer,
    RetrievalStrategy,
    LayerResult
)

# 向量检索引擎
from .vector_retriever import (
    VectorRetriever,
    VectorSearchConfig,
    VectorSearchResult
)

# 图检索引擎
from .graph_retriever import (
    GraphRetriever,
    GraphQueryConfig,
    GraphQueryResult
)

# 约束验证器
from .constraint_validator import (
    ConstraintValidator,
    ValidationRule,
    ValidationResult
)

# 重排序器
from .reranker import (
    Reranker,
    RerankingStrategy,
    RerankingResult
)

# 查询分析器
from .query_analyzer import (
    QueryAnalyzer,
    QueryComplexity,
    QueryIntent
)

# BGE-M3增强器
from .bge_enhancer import (
    BGEEnhancer,
    BGEMode,
    BGEResult
)

__all__ = [
    # 核心检索
    'ThreeLayerRetriever',
    'RetrievalLayer',
    'RetrievalStrategy',
    'LayerResult',

    # 向量检索
    'VectorRetriever',
    'VectorSearchConfig',
    'VectorSearchResult',

    # 图检索
    'GraphRetriever',
    'GraphQueryConfig',
    'GraphQueryResult',

    # 约束验证
    'ConstraintValidator',
    'ValidationRule',
    'ValidationResult',

    # 重排序
    'Reranker',
    'RerankingStrategy',
    'RerankingResult',

    # 查询分析
    'QueryAnalyzer',
    'QueryComplexity',
    'QueryIntent',

    # BGE增强
    'BGEEnhancer',
    'BGEMode',
    'BGEResult'
]

__version__ = "2.0.0"
__author__ = "DAML-RAG Team"