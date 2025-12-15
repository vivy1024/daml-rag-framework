#!/usr/bin/env python3
"""
 daml-rag-framework Qdrant向量检索实现
高性能向量数据库检索层
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import uuid
import json
import numpy as np
from datetime import datetime

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from qdrant_client.http.models import CollectionInfo
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from ..base import BaseRetriever, RetrievalResult, Document

logger = logging.getLogger(__name__)


@dataclass
class QdrantConfig:
    """Qdrant配置"""
    host: str = "localhost"
    port: int = 6333
    api_key: Optional[str] = None
    timeout: int = 30
    prefer_grpc: bool = False
    collection_name: str = "daml_rag_vectors"
    vector_size: int = 768
    distance: Distance = Distance.COSINE
    hnsw_config: Optional[Dict[str, Any]] = None
    on_disk: bool = False
    quantization_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.hnsw_config is None:
            self.hnsw_config = {
                "m": 16,
                "ef_construct": 100,
                "full_scan_threshold": 10000
            }


class QdrantVectorRetriever(BaseRetriever):
    """Qdrant向量检索器"""

    def __init__(self, config: QdrantConfig):
        super().__init__(config)
        self.config = config
        self.client: Optional[QdrantClient] = None
        self.collection_name = config.collection_name
        self._initialized = False

    async def initialize(self) -> None:
        """初始化Qdrant客户端"""
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant client not available. Install with: pip install qdrant-client"
            )

        try:
            # 创建Qdrant客户端
            self.client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                prefer_grpc=self.config.prefer_grpc
            )

            # 检查连接
            await self.client.http.client.health_check()

            # 创建集合（如果不存在）
            await self._ensure_collection_exists()

            self._initialized = True
            logger.info(f"Qdrant vector retriever initialized: {self.config.host}:{self.config.port}")

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

    async def _ensure_collection_exists(self) -> None:
        """确保集合存在"""
        try:
            # 检查集合是否存在
            collections = await self.client.http.collections_api.get_collections()
            collection_exists = any(
                col.name == self.collection_name
                for col in collections.collections
            )

            if not collection_exists:
                # 创建新集合
                await self.client.http.collections_api.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.vector_size,
                        distance=self.config.distance,
                        hnsw_config=self.config.hnsw_config,
                        on_disk=self.config.on_disk,
                        quantization_config=self.config.quantization_config
                    )
                )
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                # 验证集合配置
                collection_info = await self.client.http.collections_api.get_collection(
                    collection_name=self.collection_name
                )
                await self._validate_collection_config(collection_info)

        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise

    async def _validate_collection_config(self, collection_info: CollectionInfo) -> None:
        """验证集合配置"""
        vector_config = collection_info.config.params.vectors

        if vector_config.size != self.config.vector_size:
            logger.warning(
                f"Collection vector size {vector_config.size} differs from config {self.config.vector_size}"
            )

        if vector_config.distance != self.config.distance:
            logger.warning(
                f"Collection distance {vector_config.distance} differs from config {self.config.distance}"
            )

    async def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到向量库"""
        if not self._initialized:
            await self.initialize()

        if not documents:
            return []

        try:
            points = []
            doc_ids = []

            for doc in documents:
                # 确保有向量
                if doc.vector is None:
                    logger.warning(f"Document {doc.id} has no vector, skipping")
                    continue

                # 创建点
                point = PointStruct(
                    id=doc.id,
                    vector=doc.vector.tolist() if isinstance(doc.vector, np.ndarray) else doc.vector,
                    payload={
                        "content": doc.content,
                        "metadata": doc.metadata or {},
                        "title": doc.metadata.get("title", "") if doc.metadata else "",
                        "url": doc.metadata.get("url", "") if doc.metadata else "",
                        "created_at": datetime.now().isoformat()
                    }
                )
                points.append(point)
                doc_ids.append(doc.id)

            # 批量插入
            if points:
                await self.client.http.points_api.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Added {len(points)} documents to collection {self.collection_name}")

            return doc_ids

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

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
        if not self._initialized:
            await self.initialize()

        try:
            # 准备查询向量
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.tolist()

            # 构建过滤器
            qdrant_filter = self._build_filter(filter_condition) if filter_condition else None

            # 执行搜索
            search_result = await self.client.http.points_api.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=include_payload,
                with_vectors=include_vectors
            )

            # 转换结果
            documents = []
            scores = []
            metadatas = []

            for hit in search_result:
                if include_payload and hit.payload:
                    doc = Document(
                        id=str(hit.id),
                        content=hit.payload.get("content", ""),
                        metadata=hit.payload.get("metadata", {}),
                        vector=hit.vector if include_vectors else None
                    )
                    documents.append(doc)
                    scores.append(hit.score)
                    metadatas.append(hit.payload)
                else:
                    # 如果没有payload，创建基本文档
                    doc = Document(
                        id=str(hit.id),
                        content="",
                        metadata={},
                        vector=hit.vector if include_vectors else None
                    )
                    documents.append(doc)
                    scores.append(hit.score)
                    metadatas.append({})

            return RetrievalResult(
                query="",
                documents=documents,
                scores=scores,
                metadatas=metadatas,
                retrieval_method="qdrant_vector_search",
                total_found=len(search_result),
                search_time=0.0  # Qdrant不提供精确时间
            )

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            raise

    async def delete_documents(self, doc_ids: List[str]) -> bool:
        """删除文档"""
        if not self._initialized:
            await self.initialize()

        try:
            await self.client.http.points_api.delete(
                collection_name=self.collection_name,
                points_selector=doc_ids
            )
            logger.info(f"Deleted {len(doc_ids)} documents from collection {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

    async def update_document(self, doc_id: str, document: Document) -> bool:
        """更新文档"""
        if not self._initialized:
            await self.initialize()

        try:
            if document.vector is None:
                logger.error(f"Document {doc_id} has no vector for update")
                return False

            point = PointStruct(
                id=doc_id,
                vector=document.vector.tolist() if isinstance(document.vector, np.ndarray) else document.vector,
                payload={
                    "content": document.content,
                    "metadata": document.metadata or {},
                    "title": document.metadata.get("title", "") if document.metadata else "",
                    "url": document.metadata.get("url", "") if document.metadata else "",
                    "updated_at": datetime.now().isoformat()
                }
            )

            await self.client.http.points_api.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            logger.info(f"Updated document {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False

    async def get_document(self, doc_id: str) -> Optional[Document]:
        """获取单个文档"""
        if not self._initialized:
            await self.initialize()

        try:
            points = await self.client.http.points_api.get(
                collection_name=self.collection_name,
                ids=[doc_id],
                with_payload=True
            )

            if not points:
                return None

            point = points[0]
            payload = point.payload or {}

            return Document(
                id=str(point.id),
                content=payload.get("content", ""),
                metadata=payload.get("metadata", {}),
                vector=point.vector if point.vector else None
            )

        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None

    async def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        if not self._initialized:
            await self.initialize()

        try:
            collection_info = await self.client.http.collections_api.get_collection(
                collection_name=self.collection_name
            )

            return {
                "name": collection_info.name,
                "vectors_count": collection_info.points_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
                "hnsw_config": asdict(collection_info.config.params.vectors.hnsw_config) if collection_info.config.params.vectors.hnsw_config else None
            }

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    async def clear_collection(self) -> bool:
        """清空集合"""
        if not self._initialized:
            await self.initialize()

        try:
            await self.client.http.collections_api.delete_collection(
                collection_name=self.collection_name
            )

            # 重新创建集合
            await self._ensure_collection_exists()

            logger.info(f"Cleared and recreated collection {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def _build_filter(self, filter_condition: Dict[str, Any]) -> Dict[str, Any]:
        """构建Qdrant过滤器"""
        if not filter_condition:
            return {}

        # 这里实现Qdrant过滤器的构建逻辑
        # 根据filter_condition构建Qdrant的filter格式
        # 这是一个简化实现，实际可以根据需要扩展

        must_conditions = []
        should_conditions = []
        must_not_conditions = []

        for field, condition in filter_condition.items():
            if isinstance(condition, dict):
                if "eq" in condition:
                    must_conditions.append({
                        "field": field,
                        "match": {"value": condition["eq"]}
                    })
                elif "ne" in condition:
                    must_not_conditions.append({
                        "field": field,
                        "match": {"value": condition["ne"]}
                    })
                elif "in" in condition:
                    must_conditions.append({
                        "field": field,
                        "match": {"any": condition["in"]}
                    })
                elif "nin" in condition:
                    must_not_conditions.append({
                        "field": field,
                        "match": {"any": condition["nin"]}
                    })
                elif "gt" in condition:
                    must_conditions.append({
                        "field": field,
                        "range": {"gt": condition["gt"]}
                    })
                elif "gte" in condition:
                    must_conditions.append({
                        "field": field,
                        "range": {"gte": condition["gte"]}
                    })
                elif "lt" in condition:
                    must_conditions.append({
                        "field": field,
                        "range": {"lt": condition["lt"]}
                    })
                elif "lte" in condition:
                    must_conditions.append({
                        "field": field,
                        "range": {"lte": condition["lte"]}
                    })

        # 构建最终过滤器
        filter_dict = {}
        if must_conditions:
            filter_dict["must"] = must_conditions
        if should_conditions:
            filter_dict["should"] = should_conditions
        if must_not_conditions:
            filter_dict["must_not"] = must_not_conditions

        return filter_dict if filter_dict else {}

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if self.client:
                await self.client.http.client.health_check()
                return True
            return False
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    async def close(self):
        """关闭连接"""
        if self.client:
            await self.client.http.client.close()
            self.client = None
        logger.info("Qdrant client closed")


class QdrantVectorManager:
    """Qdrant向量管理器 - 提供高级管理功能"""

    def __init__(self, config: QdrantConfig):
        self.config = config
        self.retriever: Optional[QdrantVectorRetriever] = None

    async def initialize(self) -> None:
        """初始化管理器"""
        self.retriever = QdrantVectorRetriever(self.config)
        await self.retriever.initialize()

    async def batch_add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> List[str]:
        """批量添加文档"""
        if not self.retriever:
            await self.initialize()

        all_ids = []
        total_docs = len(documents)

        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            batch_ids = await self.retriever.add_documents(batch)
            all_ids.extend(batch_ids)

            logger.info(f"Batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1} completed")

        return all_ids

    async def create_index(self, field_name: str, index_type: str = "text") -> bool:
        """创建索引"""
        if not self.retriever:
            await self.initialize()

        try:
            # Qdrant在payload上创建索引的逻辑
            # 这是一个简化实现
            logger.info(f"Creating index on field: {field_name} (type: {index_type})")
            return True

        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.retriever:
            await self.initialize()

        collection_info = await self.retriever.get_collection_info()

        return {
            "collection_info": collection_info,
            "connection_info": {
                "host": self.config.host,
                "port": self.config.port,
                "collection_name": self.config.collection_name,
                "vector_size": self.config.vector_size,
                "distance": self.config.distance
            },
            "health": await self.retriever.health_check()
        }

    async def optimize_collection(self) -> bool:
        """优化集合"""
        if not self.retriever:
            await self.initialize()

        try:
            # Qdrant的优化操作
            await self.retriever.client.http.collections_api.update_collection(
                collection_name=self.retriever.collection_name,
                optimizer_config={
                    "deleted_threshold": 0.2,
                    "vacuum_min_vector_number": 1000,
                    "default_segment_number": 2
                }
            )

            logger.info("Collection optimization completed")
            return True

        except Exception as e:
            logger.error(f"Failed to optimize collection: {e}")
            return False

    async def close(self):
        """关闭管理器"""
        if self.retriever:
            await self.retriever.close()
            self.retriever = None