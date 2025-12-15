# -*- coding: utf-8 -*-
"""
DAML-RAG框架存储抽象层 v2.0

提供存储系统的抽象基类和通用功能实现。

版本：v2.0.0
更新日期：2025-11-17
设计原则：接口驱动、多模态支持、高性能访问
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..interfaces.storage import (
    IStorage, IVectorStorage, IGraphStorage, IDocumentStorage,
    ICacheStorage, ISessionStorage,
    VectorPoint, Document, GraphNode, GraphRelationship,
    StorageType, IndexType
)
from ..interfaces.base import BaseComponent

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """存储配置"""
    connection_params: Dict[str, Any] = field(default_factory=dict)
    performance_params: Dict[str, Any] = field(default_factory=dict)
    security_params: Dict[str, Any] = field(default_factory=dict)
    retry_params: Dict[str, Any] = field(default_factory=dict)
    monitoring_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageMetrics:
    """存储指标"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_response_time: float = 0.0
    last_operation_time: Optional[float] = None
    storage_size: Optional[int] = None
    connection_pool_stats: Dict[str, Any] = field(default_factory=dict)


class AbstractStorage(BaseComponent):
    """
    存储抽象基类

    实现存储系统的通用功能。
    """

    def __init__(self, name: str, storage_type: StorageType, version: str = "2.0.0"):
        super().__init__(name, version)
        self._storage_type = storage_type
        self._config = StorageConfig()
        self._metrics = StorageMetrics()
        self._connected = False
        self._connection_pool = None

    @property
    def storage_type(self) -> StorageType:
        """获取存储类型"""
        return self._storage_type

    def set_config(self, config: Dict[str, Any]) -> None:
        """设置存储配置"""
        # 更新配置对象
        if 'connection_params' in config:
            self._config.connection_params.update(config['connection_params'])
        if 'performance_params' in config:
            self._config.performance_params.update(config['performance_params'])
        if 'security_params' in config:
            self._config.security_params.update(config['security_params'])
        if 'retry_params' in config:
            self._config.retry_params.update(config['retry_params'])
        if 'monitoring_params' in config:
            self._config.monitoring_params.update(config['monitoring_params'])

        logger.info(f"存储配置已更新: {self.name}")

    def get_config(self) -> Dict[str, Any]:
        """获取存储配置"""
        return {
            'connection_params': self._config.connection_params.copy(),
            'performance_params': self._config.performance_params.copy(),
            'security_params': self._config.security_params.copy(),
            'retry_params': self._config.retry_params.copy(),
            'monitoring_params': self._config.monitoring_params.copy()
        }

    async def connect(self) -> bool:
        """连接到存储系统"""
        if self._connected:
            return True

        try:
            logger.info(f"正在连接存储系统: {self.name}")
            success = await self._do_connect()
            if success:
                self._connected = True
                logger.info(f"✅ 存储系统连接成功: {self.name}")
            else:
                logger.error(f"❌ 存储系统连接失败: {self.name}")
            return success
        except Exception as e:
            logger.error(f"❌ 存储系统连接异常 {self.name}: {e}")
            return False

    async def disconnect(self) -> None:
        """断开存储连接"""
        if not self._connected:
            return

        try:
            await self._do_disconnect()
            self._connected = False
            logger.info(f"存储系统已断开连接: {self.name}")
        except Exception as e:
            logger.error(f"❌ 存储系统断开连接异常 {self.name}: {e}")

    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected

    async def health_check(self) -> bool:
        """健康检查"""
        if not self._connected:
            return False

        try:
            # 执行基础健康检查
            return await self._do_health_check()
        except Exception as e:
            logger.warning(f"健康检查失败 {self.name}: {e}")
            return False

    async def _do_connect(self) -> bool:
        """执行连接（子类实现）"""
        # 默认实现：假设连接成功
        return True

    async def _do_disconnect(self) -> None:
        """执行断开连接（子类实现）"""
        # 默认实现：什么都不做
        pass

    async def _do_health_check(self) -> bool:
        """执行健康检查（子类实现）"""
        # 默认实现：检查连接状态
        return self._connected

    async def execute_with_retry(
        self,
        operation: Callable,
        max_retries: int = None,
        retry_delay: float = None
    ) -> Any:
        """
        带重试的操作执行

        Args:
            operation: 要执行的操作
            max_retries: 最大重试次数
            retry_delay: 重试延迟

        Returns:
            Any: 操作结果
        """
        max_retries = max_retries or self._config.retry_params.get('max_retries', 3)
        retry_delay = retry_delay or self._config.retry_params.get('retry_delay', 1.0)

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                result = await operation()
                self._update_metrics(True, time.time() - start_time)
                return result
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(f"操作失败，{retry_delay}秒后重试 (尝试 {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"操作最终失败: {e}")

        self._update_metrics(False, 0)
        raise last_exception

    def _update_metrics(self, success: bool, response_time: float) -> None:
        """更新性能指标"""
        self._metrics.total_operations += 1
        if success:
            self._metrics.successful_operations += 1
        else:
            self._metrics.failed_operations += 1

        # 更新平均响应时间
        total_ops = self._metrics.total_operations
        current_avg = self._metrics.average_response_time
        self._metrics.average_response_time = (
            (current_avg * (total_ops - 1) + response_time) / total_ops
        )
        self._metrics.last_operation_time = response_time

    def get_metrics(self) -> Dict[str, Any]:
        """获取存储指标"""
        base_metrics = super().get_metrics()
        return {
            **base_metrics,
            'storage_type': self._storage_type.value,
            'connected': self._connected,
            'total_operations': self._metrics.total_operations,
            'successful_operations': self._metrics.successful_operations,
            'failed_operations': self._metrics.failed_operations,
            'success_rate': (
                self._metrics.successful_operations / max(self._metrics.total_operations, 1)
            ),
            'average_response_time': self._metrics.average_response_time,
            'last_operation_time': self._metrics.last_operation_time,
            'storage_size': self._metrics.storage_size,
            'connection_pool_stats': self._metrics.connection_pool_stats
        }

    async def get_statistics(self) -> Dict[str, Any]:
        """获取详细统计信息"""
        base_stats = await super().collect_statistics()
        storage_stats = {
            'storage_type': self._storage_type.value,
            'metrics': self.get_metrics(),
            'config': self.get_config(),
            'connection_info': self.get_connection_info()
        }
        return {**base_stats, **storage_stats}

    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""
        return {
            'connected': self._connected,
            'config': self.get_config()['connection_params'],
            'pool_size': len(self._connection_pool) if self._connection_pool else 0
        }

    async def batch_operation(
        self,
        operations: List[Callable],
        max_concurrency: Optional[int] = None,
        continue_on_error: bool = False
    ) -> List[Any]:
        """
        批量执行操作

        Args:
            operations: 操作列表
            max_concurrency: 最大并发数
            continue_on_error: 遇到错误时是否继续

        Returns:
            List[Any]: 操作结果列表
        """
        if not operations:
            return []

        max_concurrency = max_concurrency or min(
            len(operations),
            self._config.performance_params.get('max_concurrent_operations', 10)
        )

        semaphore = asyncio.Semaphore(max_concurrency)
        tasks = []

        async def bounded_operation(op):
            async with semaphore:
                try:
                    return await op()
                except Exception as e:
                    if not continue_on_error:
                        raise
                    logger.warning(f"批量操作中遇到错误（继续执行）: {e}")
                    return None

        # 创建并执行任务
        tasks = [bounded_operation(op) for op in operations]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"批量操作异常: {result}")
                if not continue_on_error:
                    raise result
                processed_results.append(None)
            else:
                processed_results.append(result)

        return processed_results


class AbstractVectorStorage(AbstractStorage, IVectorStorage):
    """向量存储抽象基类"""

    def __init__(self, name: str, version: str = "2.0.0"):
        super().__init__(name, StorageType.VECTOR, version)

    async def batch_add_vectors(self, points: List[VectorPoint], batch_size: int = 100) -> bool:
        """批量添加向量点"""
        if not points:
            return True

        success_count = 0
        total_count = len(points)

        for i in range(0, total_count, batch_size):
            batch = points[i:i + batch_size]
            if await self.add_vectors(batch):
                success_count += len(batch)
            else:
                logger.error(f"批量添加向量失败 (批次 {i//batch_size + 1})")

        logger.info(f"批量添加向量完成: {success_count}/{total_count} 成功")
        return success_count == total_count

    async def batch_search(
        self,
        query_vectors: List[List[float]],
        top_k: int = 10,
        score_threshold: float = 0.0
    ) -> List[List[VectorPoint]]:
        """批量向量搜索"""
        if not query_vectors:
            return []

        # 并发执行搜索
        tasks = [
            self.search_vectors(qv, top_k, score_threshold)
            for qv in query_vectors
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"批量搜索异常: {result}")
                processed_results.append([])
            else:
                processed_results.append(result)

        return processed_results


class AbstractGraphStorage(AbstractStorage, IGraphStorage):
    """图存储抽象基类"""

    def __init__(self, name: str, version: str = "2.0.0"):
        super().__init__(name, StorageType.GRAPH, version)

    async def batch_add_nodes(self, nodes: List[GraphNode]) -> bool:
        """批量添加节点"""
        if not nodes:
            return True

        success_count = 0
        for node in nodes:
            if await self.add_node(node):
                success_count += 1
            else:
                logger.error(f"添加节点失败: {node.id}")

        logger.info(f"批量添加节点完成: {success_count}/{len(nodes)} 成功")
        return success_count == len(nodes)

    async def batch_add_relationships(self, relationships: List[GraphRelationship]) -> bool:
        """批量添加关系"""
        if not relationships:
            return True

        success_count = 0
        for rel in relationships:
            if await self.add_relationship(rel):
                success_count += 1
            else:
                logger.error(f"添加关系失败: {rel.id}")

        logger.info(f"批量添加关系完成: {success_count}/{len(relationships)} 成功")
        return success_count == len(relationships)


class AbstractDocumentStorage(AbstractStorage, IDocumentStorage):
    """文档存储抽象基类"""

    def __init__(self, name: str, version: str = "2.0.0"):
        super().__init__(name, StorageType.DOCUMENT, version)

    async def batch_add_documents(self, documents: List[Document]) -> bool:
        """批量添加文档"""
        if not documents:
            return True

        success_count = 0
        for doc in documents:
            if await self.add_document(doc):
                success_count += 1
            else:
                logger.error(f"添加文档失败: {doc.id}")

        logger.info(f"批量添加文档完成: {success_count}/{len(documents)} 成功")
        return success_count == len(documents)


# 导出基类
__all__ = [
    'StorageConfig',
    'StorageMetrics',
    'AbstractStorage',
    'AbstractVectorStorage',
    'AbstractGraphStorage',
    'AbstractDocumentStorage'
]