# -*- coding: utf-8 -*-
"""向量搜索引擎

提供语义搜索和向量相似度检索功能
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    SearchRequest
)
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[np.ndarray] = None


class VectorSearchEngine:
    """
    向量搜索引擎
    
    特性：
    - 基于Qdrant的高性能向量搜索
    - 支持多种embedding模型
    - 批量索引和检索
    - 过滤和混合检索
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "fitness_kg",
        vector_size: int = 1024,
        distance: str = "cosine",
        embedding_model: Optional[str] = "BAAI/bge-m3"
    ):
        """
        初始化向量搜索引擎
        
        Args:
            host: Qdrant服务器地址
            port: Qdrant端口
            collection_name: 集合名称
            vector_size: 向量维度
            distance: 距离度量 ("cosine", "euclidean", "dot")
            embedding_model: 嵌入模型名称
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # 连接Qdrant
        self.client = QdrantClient(host=host, port=port)
        
        # 距离度量映射
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT
        }
        self.distance = distance_map.get(distance, Distance.COSINE)
        
        # 初始化集合
        self._init_collection()
        
        # 嵌入模型
        self.embedding_model = embedding_model
        self.encoder = self._init_encoder(embedding_model)
        
        logger.info(
            f"✅ VectorSearchEngine已初始化 "
            f"(collection={collection_name}, dim={vector_size})"
        )
    
    def _init_collection(self):
        """初始化集合"""
        try:
            # 检查集合是否存在
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                # 创建新集合
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance
                    )
                )
                logger.info(f"✅ 创建Qdrant集合: {self.collection_name}")
            else:
                logger.info(f"✅ Qdrant集合已存在: {self.collection_name}")
        
        except Exception as e:
            logger.error(f"❌ 初始化Qdrant集合失败: {e}")
            raise
    
    def _init_encoder(self, model_name: Optional[str]):
        """
        初始化编码器
        
        Args:
            model_name: 模型名称
        
        Returns:
            编码器实例
        """
        if not model_name:
            logger.warning("⚠️ 未指定embedding模型，使用随机向量")
            return None
        
        try:
            # 尝试加载本地模型（如BGE）
            if "bge" in model_name.lower():
                from sentence_transformers import SentenceTransformer
                encoder = SentenceTransformer(model_name)
                logger.info(f"✅ 加载本地模型: {model_name}")
                return encoder
            
            # 尝试使用OpenAI
            elif "openai" in model_name.lower():
                import openai
                logger.info(f"✅ 使用OpenAI模型: {model_name}")
                return openai
            
            else:
                logger.warning(f"⚠️ 未知模型: {model_name}")
                return None
        
        except ImportError as e:
            logger.warning(f"⚠️ 无法加载模型 {model_name}: {e}")
            return None
    
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            text: 单个文本或文本列表
        
        Returns:
            向量或向量数组
        """
        if self.encoder is None:
            # 使用随机向量（仅用于测试）
            if isinstance(text, str):
                return np.random.rand(self.vector_size).astype(np.float32)
            else:
                return np.random.rand(len(text), self.vector_size).astype(np.float32)
        
        # 使用实际模型
        if hasattr(self.encoder, 'encode'):
            # SentenceTransformer
            vectors = self.encoder.encode(text, convert_to_numpy=True)
            return vectors.astype(np.float32)
        else:
            # OpenAI或其他
            logger.warning("⚠️ 编码器未正确配置")
            if isinstance(text, str):
                return np.random.rand(self.vector_size).astype(np.float32)
            else:
                return np.random.rand(len(text), self.vector_size).astype(np.float32)
    
    def add(
        self,
        id: str,
        vector: np.ndarray,
        payload: Dict[str, Any]
    ):
        """
        添加单个向量
        
        Args:
            id: 向量ID
            vector: 向量数据
            payload: 元数据
        """
        point = PointStruct(
            id=self._hash_id(id),
            vector=vector.tolist(),
            payload=payload
        )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
    
    def batch_add(
        self,
        ids: List[str],
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
        batch_size: int = 100
    ):
        """
        批量添加向量
        
        Args:
            ids: ID列表
            vectors: 向量数组
            payloads: 元数据列表
            batch_size: 批次大小
        """
        total = len(ids)
        
        for i in range(0, total, batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]
            
            points = [
                PointStruct(
                    id=self._hash_id(id_),
                    vector=vec.tolist(),
                    payload=payload
                )
                for id_, vec, payload in zip(batch_ids, batch_vectors, batch_payloads)
            ]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"  批次 {i//batch_size + 1}: 添加了 {len(points)} 个向量")
        
        logger.info(f"✅ 批量添加完成: 总共 {total} 个向量")
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        min_similarity: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        向量搜索

        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            filter: 过滤条件（向后兼容）
            min_similarity: 最小相似度阈值
            filters: 过滤条件（新版本）

        Returns:
            搜索结果列表
        """
        # 构建过滤器（支持新旧参数）
        filter_param = filters or filter
        qdrant_filter = None
        if filter_param:
            qdrant_filter = self._build_filter(filter_param)

        # 执行搜索（使用query_points方法）
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=qdrant_filter
        )

        # 转换结果并应用相似度过滤
        search_results = []
        # query_points返回ScoredPoint对象，需要从points属性获取
        if hasattr(results, 'points'):
            points = results.points
        else:
            points = results

        for r in points:
            # 处理不同的结果格式
            if hasattr(r, 'score'):
                score = r.score
                point_id = r.id
                payload = r.payload if hasattr(r, 'payload') else {}
            else:
                # 如果是tuple格式 (id, score, payload)
                if len(r) >= 3:
                    point_id, score, payload = r[0], r[1], r[2]
                elif len(r) == 2:
                    point_id, score = r
                    payload = {}
                else:
                    continue

            # 应用最小相似度阈值
            if min_similarity is not None and score < min_similarity:
                continue
            search_results.append(SearchResult(
                id=str(point_id),
                score=score,
                payload=payload
            ))

        return search_results
    
    def search_by_text(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        文本语义搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter: 过滤条件
        
        Returns:
            搜索结果列表
        """
        # 编码查询文本
        query_vector = self.encode(query)
        
        # 执行向量搜索
        return self.search(query_vector, top_k, filter)
    
    def batch_search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 10
    ) -> List[List[SearchResult]]:
        """
        批量搜索
        
        Args:
            query_vectors: 查询向量数组
            top_k: 每个查询返回的结果数量
        
        Returns:
            每个查询的搜索结果列表
        """
        results = []
        
        for query_vector in query_vectors:
            search_results = self.search(query_vector, top_k)
            results.append(search_results)
        
        return results
    
    def get_by_id(self, id: str) -> Optional[SearchResult]:
        """
        根据ID获取向量
        
        Args:
            id: 向量ID
        
        Returns:
            搜索结果
        """
        try:
            point = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[self._hash_id(id)]
            )
            
            if point:
                p = point[0]
                return SearchResult(
                    id=str(p.id),
                    score=1.0,
                    payload=p.payload,
                    vector=np.array(p.vector) if p.vector else None
                )
        except Exception as e:
            logger.warning(f"⚠️ 获取向量失败 (id={id}): {e}")
        
        return None
    
    def delete(self, id: str):
        """
        删除向量
        
        Args:
            id: 向量ID
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[self._hash_id(id)]
        )
    
    def count(self) -> int:
        """
        获取向量数量
        
        Returns:
            向量总数
        """
        info = self.client.get_collection(self.collection_name)
        return info.points_count
    
    def _hash_id(self, id: str) -> int:
        """
        将字符串ID哈希为整数
        
        Args:
            id: 字符串ID
        
        Returns:
            整数ID
        """
        return int(hashlib.md5(id.encode()).hexdigest()[:16], 16) % (10 ** 12)
    
    def _build_filter(self, filter_dict: Dict[str, Any]) -> Filter:
        """
        构建Qdrant过滤器
        
        Args:
            filter_dict: 过滤字典 {"field": "value"}
        
        Returns:
            Qdrant过滤器
        """
        conditions = [
            FieldCondition(
                key=key,
                match=MatchValue(value=value)
            )
            for key, value in filter_dict.items()
        ]
        
        return Filter(must=conditions)
    
    def clear(self):
        """清空集合"""
        try:
            self.client.delete_collection(self.collection_name)
            self._init_collection()
            logger.info(f"✅ 集合已清空: {self.collection_name}")
        except Exception as e:
            logger.error(f"❌ 清空集合失败: {e}")
    
    def close(self):
        """关闭连接"""
        if self.client:
            self.client.close()
            logger.info("🔒 Qdrant连接已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


# ==================== 辅助函数 ====================

def create_text_for_embedding(node: Dict[str, Any]) -> str:
    """
    为节点创建用于embedding的文本
    
    Args:
        node: 节点数据
    
    Returns:
        文本字符串
    """
    parts = []
    
    # 添加名称
    if "name_zh" in node:
        parts.append(node["name_zh"])
    elif "name" in node:
        parts.append(node["name"])
    
    # 添加描述
    if "description" in node:
        parts.append(node["description"])
    
    # 添加其他重要字段
    for key in ["category", "type", "difficulty", "equipment"]:
        if key in node and node[key]:
            parts.append(str(node[key]))
    
    return " ".join(parts)

