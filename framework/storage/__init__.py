# -*- coding: utf-8 -*-
"""
DAML-RAG框架存储模块 v2.0

提供统一的存储抽象层，支持：
- 向量存储（Qdrant等）
- 图存储（Neo4j等）
- 文档存储（Elasticsearch等）
- 缓存存储（Redis等）
- 会话存储

版本：v2.0.0
更新日期：2025-11-17
"""

# 抽象基类
from .abstract_storage import (
    StorageConfig,
    StorageMetrics,
    AbstractStorage,
    AbstractVectorStorage,
    AbstractGraphStorage,
    AbstractDocumentStorage
)

# 具体实现（后续添加）
# from .implementations import (
#     QdrantVectorStorage,
#     Neo4jGraphStorage,
#     ElasticsearchDocumentStorage,
#     RedisCacheStorage,
#     MemorySessionStorage
# )

__all__ = [
    # 抽象基类
    'StorageConfig',
    'StorageMetrics',
    'AbstractStorage',
    'AbstractVectorStorage',
    'AbstractGraphStorage',
    'AbstractDocumentStorage',

    # 具体实现（后续添加）
    # 'QdrantVectorStorage',
    # 'Neo4jGraphStorage',
    # 'ElasticsearchDocumentStorage',
    # 'RedisCacheStorage',
    # 'MemorySessionStorage'
]

__version__ = "2.0.0"
__author__ = "DAML-RAG Team"