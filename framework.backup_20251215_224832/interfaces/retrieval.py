# -*- coding: utf-8 -*-
"""
DAML-RAG框架检索接口定义 v2.0

定义三层检索架构的标准接口，支持语义检索、图谱检索和混合检索。

版本：v2.0.0
更新日期：2025-11-17
设计原则：接口抽象、可插拔、性能优化
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from .base import IComponent, IConfigurable, IMonitorable


@dataclass
class QueryRequest:
    """查询请求"""
    query_text: str
    query_type: str = "hybrid"  # semantic, graph, hybrid
    domain: Optional[str] = None
    user_profile: Optional[Dict[str, Any]] = None
    top_k: int = 10
    filters: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


@dataclass
class RetrievalResult:
    """检索结果"""
    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_type: str = "unknown"
    confidence: float = 0.0
    relevance_score: float = 0.0


@dataclass
class RetrievalResponse:
    """检索响应"""
    query: QueryRequest
    results: List[RetrievalResult]
    total_found: int
    query_time_ms: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RetrievalMode(Enum):
    """检索模式"""
    SEMANTIC = "semantic"      # 纯语义检索
    GRAPH = "graph"           # 纯图谱检索
    HYBRID = "hybrid"         # 混合检索
    ADAPTIVE = "adaptive"     # 自适应检索


class QueryComplexity(Enum):
    """查询复杂度"""
    SIMPLE = "simple"         # 简单查询
    MEDIUM = "medium"         # 中等复杂度
    COMPLEX = "complex"       # 复杂查询
    EXPERT = "expert"         # 专家级查询


T = TypeVar('T')


class IRetriever(IComponent, IConfigurable, IMonitorable, Generic[T]):
    """
    检索器基础接口

    定义所有检索器必须实现的基础功能。
    """

    @abstractmethod
    async def search(self, request: QueryRequest) -> RetrievalResponse:
        """
        执行检索

        Args:
            request: 查询请求

        Returns:
            RetrievalResponse: 检索结果
        """
        pass

    @abstractmethod
    async def batch_search(self, requests: List[QueryRequest]) -> List[RetrievalResponse]:
        """
        批量检索

        Args:
            requests: 查询请求列表

        Returns:
            List[RetrievalResponse]: 检索结果列表
        """
        pass

    @abstractmethod
    def supported_modes(self) -> List[RetrievalMode]:
        """
        获取支持的检索模式

        Returns:
            List[RetrievalMode]: 支持的模式列表
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        获取检索器能力

        Returns:
            Dict[str, Any]: 能力描述
        """
        pass


class ISemanticRetriever(IRetriever):
    """
    语义检索接口

    基于向量嵌入的语义相似度检索。
    """

    @abstractmethod
    async def encode_query(self, query: str) -> List[float]:
        """
        编码查询文本为向量

        Args:
            query: 查询文本

        Returns:
            List[float]: 查询向量
        """
        pass

    @abstractmethod
    async def encode_documents(self, documents: List[str]) -> List[List[float]]:
        """
        批量编码文档为向量

        Args:
            documents: 文档列表

        Returns:
            List[List[float]]: 文档向量列表
        """
        pass

    @abstractmethod
    def get_embedding_model(self) -> str:
        """
        获取嵌入模型信息

        Returns:
            str: 模型名称或路径
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
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        添加文档到向量库

        Args:
            documents: 文档列表

        Returns:
            bool: 添加是否成功
        """
        pass

    @abstractmethod
    async def remove_documents(self, document_ids: List[str]) -> bool:
        """
        从向量库移除文档

        Args:
            document_ids: 文档ID列表

        Returns:
            bool: 移除是否成功
        """
        pass


class IGraphRetriever(IRetriever):
    """
    图谱检索接口

    基于知识图谱的关系推理检索。
    """

    @abstractmethod
    async def traverse_graph(self, start_node: str, relation_types: List[str], max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        图谱遍历

        Args:
            start_node: 起始节点
            relation_types: 关系类型列表
            max_depth: 最大遍历深度

        Returns:
            List[Dict[str, Any]]: 遍历结果
        """
        pass

    @abstractmethod
    async def find_neighbors(self, node_id: str, node_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        查找邻居节点

        Args:
            node_id: 节点ID
            node_types: 节点类型过滤

        Returns:
            List[Dict[str, Any]]: 邻居节点列表
        """
        pass

    @abstractmethod
    async def execute_cypher_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        执行Cypher查询

        Args:
            query: Cypher查询语句
            parameters: 查询参数

        Returns:
            List[Dict[str, Any]]: 查询结果
        """
        pass

    @abstractmethod
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        获取图谱统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        pass

    @abstractmethod
    async def add_relationship(self, from_node: str, to_node: str, relation_type: str, properties: Dict[str, Any]) -> bool:
        """
        添加关系

        Args:
            from_node: 源节点
            to_node: 目标节点
            relation_type: 关系类型
            properties: 关系属性

        Returns:
            bool: 添加是否成功
        """
        pass


class IConstraintValidator(IComponent):
    """
    约束验证器接口

    验证检索结果是否符合业务约束和安全规则。
    """

    @abstractmethod
    async def validate_query(self, request: QueryRequest) -> Tuple[bool, Optional[str]]:
        """
        验证查询请求

        Args:
            request: 查询请求

        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        pass

    @abstractmethod
    async def validate_results(self, request: QueryRequest, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        验证检索结果

        Args:
            request: 查询请求
            results: 原始检索结果

        Returns:
            List[RetrievalResult]: 过滤后的结果
        """
        pass

    @abstractmethod
    def get_constraint_rules(self) -> Dict[str, Any]:
        """
        获取约束规则

        Returns:
            Dict[str, Any]: 约束规则定义
        """
        pass

    @abstractmethod
    def add_constraint_rule(self, rule_name: str, rule_config: Dict[str, Any]) -> bool:
        """
        添加约束规则

        Args:
            rule_name: 规则名称
            rule_config: 规则配置

        Returns:
            bool: 添加是否成功
        """
        pass


class IThreeLayerRetriever(IRetriever):
    """
    三层检索器接口

    协调语义检索、图谱检索和约束验证三层检索架构。
    """

    @abstractmethod
    def get_semantic_retriever(self) -> Optional[ISemanticRetriever]:
        """
        获取语义检索器

        Returns:
            Optional[ISemanticRetriever]: 语义检索器实例
        """
        pass

    @abstractmethod
    def get_graph_retriever(self) -> Optional[IGraphRetriever]:
        """
        获取图谱检索器

        Returns:
            Optional[IGraphRetriever]: 图谱检索器实例
        """
        pass

    @abstractmethod
    def get_constraint_validator(self) -> Optional[IConstraintValidator]:
        """
        获取约束验证器

        Returns:
            Optional[IConstraintValidator]: 约束验证器实例
        """
        pass

    @abstractmethod
    async def execute_three_layer_search(self, request: QueryRequest) -> RetrievalResponse:
        """
        执行三层检索

        Args:
            request: 查询请求

        Returns:
            RetrievalResponse: 检索结果
        """
        pass

    @abstractmethod
    def set_retriever_weights(self, semantic_weight: float, graph_weight: float, constraint_weight: float) -> None:
        """
        设置检索器权重

        Args:
            semantic_weight: 语义检索权重
            graph_weight: 图谱检索权重
            constraint_weight: 约束验证权重
        """
        pass

    @abstractmethod
    def get_layer_performance_stats(self) -> Dict[str, Any]:
        """
        获取各层性能统计

        Returns:
            Dict[str, Any]: 性能统计信息
        """
        pass


class IReranker(IComponent):
    """
    结果重排序接口

    对检索结果进行智能重排序。
    """

    @abstractmethod
    async def rerank(self, query: str, results: List[RetrievalResult], top_k: int = 10) -> List[RetrievalResult]:
        """
        重排序检索结果

        Args:
            query: 原始查询
            results: 检索结果
            top_k: 返回数量

        Returns:
            List[RetrievalResult]: 重排序后的结果
        """
        pass

    @abstractmethod
    def get_reranking_model(self) -> str:
        """
        获取重排序模型信息

        Returns:
            str: 模型信息
        """
        pass

    @abstractmethod
    async def train_feedback(self, query: str, results: List[RetrievalResult], feedback: Dict[str, Any]) -> bool:
        """
        基于反馈训练重排序模型

        Args:
            query: 查询
            results: 结果
            feedback: 反馈信息

        Returns:
            bool: 训练是否成功
        """
        pass


# 便捷的检索器基类
class BaseRetriever(BaseComponent):
    """
    检索器基础实现类
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
        self._search_count = 0
        self._total_query_time = 0.0

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "search_count": self._search_count,
            "average_query_time": self._total_query_time / max(self._search_count, 1),
            "supported_modes": [mode.value for mode in self.supported_modes()]
        }

    def supported_modes(self) -> List[RetrievalMode]:
        return [RetrievalMode.SEMANTIC]

    async def batch_search(self, requests: List[QueryRequest]) -> List[RetrievalResponse]:
        """默认批量搜索实现（并发执行）"""
        tasks = [self.search(request) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)


# 导出接口
__all__ = [
    # 核心数据结构
    'QueryRequest',
    'RetrievalResult',
    'RetrievalResponse',
    'RetrievalMode',
    'QueryComplexity',

    # 检索器接口
    'IRetriever',
    'ISemanticRetriever',
    'IGraphRetriever',
    'IThreeLayerRetriever',
    'IConstraintValidator',
    'IReranker',

    # 基础实现
    'BaseRetriever'
]