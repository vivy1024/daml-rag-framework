#!/usr/bin/env python3
"""
DAML-RAG健身 框架 向量检索基类
定义向量检索的抽象接口和通用功能
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Protocol
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from ...daml_rag.models.base import Document, RetrievalResult

logger = logging.getLogger(__name__)


class VectorRetriever(Protocol):
    """向量检索器协议"""

    async def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到向量库"""
        ...

    async def search(
        self,
        query_vector: Union[List[float], np.ndarray],
        top_k: int = 10,
        score_threshold: float = 0.0,
        filter_condition: Optional[Dict[str, Any]] = None,
        include_payload: bool = True,
        include_vectors: bool = False
    ) -> RetrievalResult:
        """向量检索"""
        ...

    async def delete_documents(self, doc_ids: List[str]) -> bool:
        """删除文档"""
        ...

    async def get_document(self, doc_id: str) -> Optional[Document]:
        """获取单个文档"""
        ...


@dataclass
class VectorConfig:
    """向量检索配置基类"""
    vector_size: int = 768
    distance_metric: str = "cosine"
    batch_size: int = 100
    cache_enabled: bool = True
    cache_ttl: int = 300
    timeout: int = 30


@dataclass
class SearchResult:
    """搜索结果"""
    document_id: str
    score: float
    document: Document
    metadata: Dict[str, Any]
    distance: Optional[float] = None


class BaseVectorRetriever(ABC):
    """向量检索器抽象基类"""

    def __init__(self, config: VectorConfig):
        self.config = config
        self._initialized = False
        self._stats = {
            "total_documents": 0,
            "total_searches": 0,
            "total_additions": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

    @abstractmethod
    async def initialize(self) -> None:
        """初始化向量检索器"""
        pass

    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到向量库"""
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: Union[List[float], np.ndarray],
        top_k: int = 10,
        score_threshold: float = 0.0,
        filter_condition: Optional[Dict[str, Any]] = None,
        include_payload: bool = True,
        include_vectors: bool = False
    ) -> RetrievalResult:
        """向量检索"""
        pass

    @abstractmethod
    async def delete_documents(self, doc_ids: List[str]) -> bool:
        """删除文档"""
        pass

    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """获取单个文档"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass

    @abstractmethod
    async def close(self):
        """关闭连接"""
        pass

    async def batch_search(
        self,
        query_vectors: List[Union[List[float], np.ndarray]],
        top_k: int = 10,
        score_threshold: float = 0.0,
        filter_condition: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """批量搜索"""
        results = []

        for query_vector in query_vectors:
            result = await self.search(
                query_vector=query_vector,
                top_k=top_k,
                score_threshold=score_threshold,
                filter_condition=filter_condition
            )
            results.append(result)

        return results

    async def update_document(
        self,
        doc_id: str,
        document: Document,
        upsert: bool = True
    ) -> bool:
        """更新文档"""
        # 默认实现：先删除后添加
        if upsert:
            # 检查文档是否存在
            existing_doc = await self.get_document(doc_id)
            if existing_doc:
                await self.delete_documents([doc_id])

        return await self.add_documents([document]) != []

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "initialized": self._initialized,
            "config": {
                "vector_size": self.config.vector_size,
                "distance_metric": self.config.distance_metric,
                "batch_size": self.config.batch_size,
                "cache_enabled": self.config.cache_enabled
            }
        }

    def _normalize_vector(self, vector: Union[List[float], np.ndarray]) -> np.ndarray:
        """向量归一化"""
        vector = np.array(vector, dtype=np.float32)

        if self.config.distance_metric == "cosine":
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

        return vector

    def _calculate_cosine_similarity(
        self,
        vec1: Union[List[float], np.ndarray],
        vec2: Union[List[float], np.ndarray]
    ) -> float:
        """计算余弦相似度"""
        vec1 = np.array(vec1, dtype=np.float32)
        vec2 = np.array(vec2, dtype=np.float32)

        # 归一化
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def _calculate_euclidean_distance(
        self,
        vec1: Union[List[float], np.ndarray],
        vec2: Union[List[float], np.ndarray]
    ) -> float:
        """计算欧几里得距离"""
        vec1 = np.array(vec1, dtype=np.float32)
        vec2 = np.array(vec2, dtype=np.float32)
        return np.linalg.norm(vec1 - vec2)

    def _apply_post_filtering(
        self,
        results: List[SearchResult],
        filter_condition: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """应用后过滤"""
        if not filter_condition:
            return results

        filtered_results = []

        for result in results:
            if self._matches_filter(result, filter_condition):
                filtered_results.append(result)

        return filtered_results

    def _matches_filter(self, result: SearchResult, filter_condition: Dict[str, Any]) -> bool:
        """检查结果是否匹配过滤条件"""
        metadata = result.metadata or {}

        for field, condition in filter_condition.items():
            field_value = metadata.get(field)

            if field_value is None:
                return False

            if isinstance(condition, dict):
                # 处理复杂条件
                if "eq" in condition and field_value != condition["eq"]:
                    return False
                if "ne" in condition and field_value == condition["ne"]:
                    return False
                if "in" in condition and field_value not in condition["in"]:
                    return False
                if "nin" in condition and field_value in condition["nin"]:
                    return False
                if "gt" in condition and not (field_value > condition["gt"]):
                    return False
                if "gte" in condition and not (field_value >= condition["gte"]):
                    return False
                if "lt" in condition and not (field_value < condition["lt"]):
                    return False
                if "lte" in condition and not (field_value <= condition["lte"]):
                    return False
            else:
                # 简单等值匹配
                if field_value != condition:
                    return False

        return True

    async def validate_documents(self, documents: List[Document]) -> List[str]:
        """验证文档"""
        errors = []

        for i, doc in enumerate(documents):
            if not doc.id:
                errors.append(f"Document {i}: Missing ID")

            if not doc.content and not doc.vector:
                errors.append(f"Document {i}: Missing content and vector")

            if doc.vector is not None:
                vector_array = np.array(doc.vector)
                if vector_array.shape[0] != self.config.vector_size:
                    errors.append(
                        f"Document {i}: Vector dimension mismatch. "
                        f"Expected {self.config.vector_size}, got {vector_array.shape[0]}"
                    )

        return errors

    def _update_stats(self, operation: str, count: int = 1):
        """更新统计信息"""
        if operation == "add":
            self._stats["total_additions"] += count
            self._stats["total_documents"] += count
        elif operation == "search":
            self._stats["total_searches"] += count
        elif operation == "cache_hit":
            self._stats["cache_hits"] += count
        elif operation == "cache_miss":
            self._stats["cache_misses"] += count


class VectorRetrieverFactory:
    """向量检索器工厂"""

    _retrievers = {}

    @classmethod
    def register_retriever(cls, name: str, retriever_class):
        """注册向量检索器"""
        cls._retrievers[name] = retriever_class

    @classmethod
    def create_retriever(cls, name: str, config: VectorConfig) -> BaseVectorRetriever:
        """创建向量检索器"""
        if name not in cls._retrievers:
            raise ValueError(f"Unknown retriever: {name}. Available: {list(cls._retrievers.keys())}")

        return cls._retrievers[name](config)

    @classmethod
    def list_available_retrievers(cls) -> List[str]:
        """列出可用的检索器"""
        return list(cls._retrievers.keys())


# 便捷的向量检索器创建函数
def create_vector_retriever(
    backend: str,
    config: Optional[VectorConfig] = None,
    **kwargs
) -> BaseVectorRetriever:
    """创建向量检索器的便捷函数"""
    if config is None:
        config = VectorConfig(**kwargs)

    return VectorRetrieverFactory.create_retriever(backend, config)


# 预定义的配置模板
def get_qdrant_config(**overrides) -> VectorConfig:
    """获取Qdrant配置模板"""
    from .qdrant import QdrantConfig

    qdrant_config = QdrantConfig()
    base_config = VectorConfig(
        vector_size=qdrant_config.vector_size,
        distance_metric=qdrant_config.distance.value.lower()
    )

    # 应用覆盖
    for key, value in overrides.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)

    return base_config


def get_faiss_config(**overrides) -> VectorConfig:
    """获取FAISS配置模板"""
    from .faiss import FAISSConfig

    faiss_config = FAISSConfig()
    base_config = VectorConfig(
        vector_size=faiss_config.vector_size,
        distance_metric=faiss_config.metric_type
    )

    # 应用覆盖
    for key, value in overrides.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)

    return base_config