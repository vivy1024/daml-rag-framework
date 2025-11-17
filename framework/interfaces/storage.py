# -*- coding: utf-8 -*-
"""
DAML-RAG框架存储接口定义 v2.0

定义向量存储、图数据库、文档存储等存储相关接口。

版本：v2.0.0
更新日期：2025-11-17
设计原则：存储抽象、多模态支持、高性能访问
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from .base import IComponent, IConfigurable, IMonitorable


@dataclass
class Document:
    """文档对象"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    source: Optional[str] = None
    domain: Optional[str] = None


@dataclass
class VectorPoint:
    """向量点"""
    id: str
    vector: List[float]
    payload: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None


@dataclass
class GraphNode:
    """图谱节点"""
    id: str
    labels: List[str]
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None


@dataclass
class GraphRelationship:
    """图谱关系"""
    id: Optional[str]
    type: str
    source_node: str
    target_node: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None


class StorageType(Enum):
    """存储类型"""
    VECTOR = "vector"         # 向量存储
    GRAPH = "graph"           # 图数据库
    DOCUMENT = "document"     # 文档存储
    CACHE = "cache"           # 缓存存储
    SESSION = "session"       # 会话存储


class IndexType(Enum):
    """索引类型"""
    EXACT = "exact"           # 精确匹配
    APPROXIMATE = "approximate" # 近似匹配
    HYBRID = "hybrid"          # 混合索引


T = TypeVar('T')


class IStorage(IComponent, IConfigurable, IMonitorable, Generic[T]):
    """
    存储基础接口

    定义所有存储系统必须实现的基础功能。
    """

    @abstractmethod
    async def connect(self) -> bool:
        """
        连接到存储系统

        Returns:
            bool: 连接是否成功
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """断开存储连接"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        存储健康检查

        Returns:
            bool: 健康状态
        """
        pass

    @abstractmethod
    def get_storage_type(self) -> StorageType:
        """
        获取存储类型

        Returns:
            StorageType: 存储类型
        """
        pass

    @abstractmethod
    def get_connection_info(self) -> Dict[str, Any]:
        """
        获取连接信息

        Returns:
            Dict[str, Any]: 连接信息
        """
        pass

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """
        获取存储统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        pass


class IVectorStorage(IStorage[VectorPoint]):
    """
    向量存储接口

    提供向量嵌入的存储、检索和管理功能。
    """

    @abstractmethod
    async def add_vector(self, point: VectorPoint) -> bool:
        """
        添加向量点

        Args:
            point: 向量点

        Returns:
            bool: 添加是否成功
        """
        pass

    @abstractmethod
    async def add_vectors(self, points: List[VectorPoint]) -> bool:
        """
        批量添加向量点

        Args:
            points: 向量点列表

        Returns:
            bool: 添加是否成功
        """
        pass

    @abstractmethod
    async def search_vectors(self, query_vector: List[float], top_k: int = 10, score_threshold: float = 0.0, filters: Optional[Dict[str, Any]] = None) -> List[VectorPoint]:
        """
        向量相似度搜索

        Args:
            query_vector: 查询向量
            top_k: 返回数量
            score_threshold: 相似度阈值
            filters: 过滤条件

        Returns:
            List[VectorPoint]: 相似向量点列表
        """
        pass

    @abstractmethod
    async def delete_vector(self, point_id: str) -> bool:
        """
        删除向量点

        Args:
            point_id: 向量点ID

        Returns:
            bool: 删除是否成功
        """
        pass

    @abstractmethod
    async def update_vector(self, point_id: str, vector: Optional[List[float]] = None, payload: Optional[Dict[str, Any]] = None) -> bool:
        """
        更新向量点

        Args:
            point_id: 向量点ID
            vector: 新向量（可选）
            payload: 新负载（可选）

        Returns:
            bool: 更新是否成功
        """
        pass

    @abstractmethod
    async def get_vector(self, point_id: str) -> Optional[VectorPoint]:
        """
        获取向量点

        Args:
            point_id: 向量点ID

        Returns:
            Optional[VectorPoint]: 向量点
        """
        pass

    @abstractmethod
    async def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取集合统计信息

        Args:
            collection_name: 集合名称（可选）

        Returns:
            Dict[str, Any]: 统计信息
        """
        pass

    @abstractmethod
    def get_vector_dimension(self) -> int:
        """
        获取向量维度

        Returns:
            int: 向量维度
        """
        pass

    @abstractmethod
    async def create_index(self, field_name: str, index_type: IndexType) -> bool:
        """
        创建索引

        Args:
            field_name: 字段名称
            index_type: 索引类型

        Returns:
            bool: 创建是否成功
        """
        pass


class IGraphStorage(IStorage[Union[GraphNode, GraphRelationship]]):
    """
    图存储接口

    提供图谱数据的存储、查询和管理功能。
    """

    @abstractmethod
    async def add_node(self, node: GraphNode) -> bool:
        """
        添加节点

        Args:
            node: 图节点

        Returns:
            bool: 添加是否成功
        """
        pass

    @abstractmethod
    async def add_relationship(self, relationship: GraphRelationship) -> bool:
        """
        添加关系

        Args:
            relationship: 图关系

        Returns:
            bool: 添加是否成功
        """
        pass

    @abstractmethod
    async def find_nodes(self, labels: Optional[List[str]] = None, properties: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[GraphNode]:
        """
        查找节点

        Args:
            labels: 节点标签过滤
            properties: 属性过滤
            limit: 返回数量限制

        Returns:
            List[GraphNode]: 节点列表
        """
        pass

    @abstractmethod
    async def find_relationships(self, relationship_type: Optional[str] = None, source_node: Optional[str] = None, target_node: Optional[str] = None, limit: Optional[int] = None) -> List[GraphRelationship]:
        """
        查找关系

        Args:
            relationship_type: 关系类型过滤
            source_node: 源节点过滤
            target_node: 目标节点过滤
            limit: 返回数量限制

        Returns:
            List[GraphRelationship]: 关系列表
        """
        pass

    @abstractmethod
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        执行图查询

        Args:
            query: 查询语句
            parameters: 查询参数

        Returns:
            List[Dict[str, Any]]: 查询结果
        """
        pass

    @abstractmethod
    async def get_neighbors(self, node_id: str, relationship_types: Optional[List[str]] = None, direction: str = "both", limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取邻居节点

        Args:
            node_id: 节点ID
            relationship_types: 关系类型过滤
            direction: 方向（incoming, outgoing, both）
            limit: 返回数量限制

        Returns:
            List[Dict[str, Any]]: 邻居信息
        """
        pass

    @abstractmethod
    async def delete_node(self, node_id: str) -> bool:
        """
        删除节点

        Args:
            node_id: 节点ID

        Returns:
            bool: 删除是否成功
        """
        pass

    @abstractmethod
    async def delete_relationship(self, relationship_id: str) -> bool:
        """
        删除关系

        Args:
            relationship_id: 关系ID

        Returns:
            bool: 删除是否成功
        """
        pass

    @abstractmethod
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        获取图谱统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        pass

    @abstractmethod
    def get_supported_query_languages(self) -> List[str]:
        """
        获取支持的查询语言

        Returns:
            List[str]: 查询语言列表
        """
        pass


class IDocumentStorage(IStorage[Document]):
    """
    文档存储接口

    提供文档的存储、检索、版本管理等功能。
    """

    @abstractmethod
    async def add_document(self, document: Document) -> bool:
        """
        添加文档

        Args:
            document: 文档对象

        Returns:
            bool: 添加是否成功
        """
        pass

    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> bool:
        """
        批量添加文档

        Args:
            documents: 文档列表

        Returns:
            bool: 添加是否成功
        """
        pass

    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[Document]:
        """
        获取文档

        Args:
            document_id: 文档ID

        Returns:
            Optional[Document]: 文档对象
        """
        pass

    @abstractmethod
    async def search_documents(self, query: str, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Document]:
        """
        搜索文档

        Args:
            query: 搜索查询
            filters: 过滤条件
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            List[Document]: 文档列表
        """
        pass

    @abstractmethod
    async def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新文档

        Args:
            document_id: 文档ID
            updates: 更新内容

        Returns:
            bool: 更新是否成功
        """
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """
        删除文档

        Args:
            document_id: 文档ID

        Returns:
            bool: 删除是否成功
        """
        pass

    @abstractmethod
    async def get_document_versions(self, document_id: str) -> List[Document]:
        """
        获取文档版本历史

        Args:
            document_id: 文档ID

        Returns:
            List[Document]: 版本列表
        """
        pass

    @abstractmethod
    async def create_index(self, fields: List[str], index_type: IndexType) -> bool:
        """
        创建文档索引

        Args:
            fields: 索引字段列表
            index_type: 索引类型

        Returns:
            bool: 创建是否成功
        """
        pass

    @abstractmethod
    def get_document_statistics(self) -> Dict[str, Any]:
        """
        获取文档统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        pass


class ICacheStorage(IStorage):
    """
    缓存存储接口

    提供高性能的数据缓存功能。
    """

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值

        Args:
            key: 缓存键

        Returns:
            Optional[Any]: 缓存值
        """
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）

        Returns:
            bool: 设置是否成功
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        删除缓存

        Args:
            key: 缓存键

        Returns:
            bool: 删除是否成功
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        检查缓存是否存在

        Args:
            key: 缓存键

        Returns:
            bool: 是否存在
        """
        pass

    @abstractmethod
    async def clear(self, pattern: Optional[str] = None) -> int:
        """
        清空缓存

        Args:
            pattern: 键模式（可选）

        Returns:
            int: 清除的键数量
        """
        pass

    @abstractmethod
    async def get_multiple(self, keys: List[str]) -> Dict[str, Any]:
        """
        批量获取缓存

        Args:
            keys: 缓存键列表

        Returns:
            Dict[str, Any]: 键值对字典
        """
        pass

    @abstractmethod
    async def set_multiple(self, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        批量设置缓存

        Args:
            items: 键值对字典
            ttl: 过期时间（秒）

        Returns:
            bool: 设置是否成功
        """
        pass


class ISessionStorage(IStorage):
    """
    会话存储接口

    提供用户会话数据的存储和管理。
    """

    @abstractmethod
    async def create_session(self, session_id: str, user_id: str, initial_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        创建会话

        Args:
            session_id: 会话ID
            user_id: 用户ID
            initial_data: 初始数据

        Returns:
            bool: 创建是否成功
        """
        pass

    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话数据

        Args:
            session_id: 会话ID

        Returns:
            Optional[Dict[str, Any]]: 会话数据
        """
        pass

    @abstractmethod
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新会话数据

        Args:
            session_id: 会话ID
            updates: 更新数据

        Returns:
            bool: 更新是否成功
        """
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id: 会话ID

        Returns:
            bool: 删除是否成功
        """
        pass

    @abstractmethod
    async def get_user_sessions(self, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取用户会话列表

        Args:
            user_id: 用户ID
            limit: 返回数量限制

        Returns:
            List[Dict[str, Any]]: 会话列表
        """
        pass

    @abstractmethod
    async def cleanup_expired_sessions(self, ttl_hours: int = 24) -> int:
        """
        清理过期会话

        Args:
            ttl_hours: 生存时间（小时）

        Returns:
            int: 清理的会话数量
        """
        pass


# 便捷的存储基类
class BaseStorage(BaseComponent):
    """
    存储基础实现类
    """

    def __init__(self, name: str, storage_type: StorageType, version: str = "1.0.0"):
        super().__init__(name, version)
        self._storage_type = storage_type
        self._connected = False

    def get_storage_type(self) -> StorageType:
        return self._storage_type

    async def health_check(self) -> bool:
        return self._connected

    def get_connection_info(self) -> Dict[str, Any]:
        return {
            "storage_type": self._storage_type.value,
            "connected": self._connected,
            "version": self.version
        }

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "storage_type": self._storage_type.value,
            "connected": self._connected,
            **super().get_metrics()
        }


# 导出接口
__all__ = [
    # 核心数据结构
    'Document',
    'VectorPoint',
    'GraphNode',
    'GraphRelationship',
    'StorageType',
    'IndexType',

    # 存储接口
    'IStorage',
    'IVectorStorage',
    'IGraphStorage',
    'IDocumentStorage',
    'ICacheStorage',
    'ISessionStorage',

    # 基础实现
    'BaseStorage'
]