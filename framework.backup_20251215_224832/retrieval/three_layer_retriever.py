# -*- coding: utf-8 -*-
"""
DAML-RAG三层检索引擎 v2.0

实现核心的三层检索架构：
1. 语义检索层 - 基于向量相似度的初步匹配
2. 图谱检索层 - 基于关系推理的精确过滤
3. 约束验证层 - 基于专业规则的最终验证

版本：v2.0.0
更新日期：2025-11-17
设计原则：渐进式精确、多源融合、质量保证
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..interfaces.retrieval import (
    IThreeLayerRetriever, IRetriever, ISemanticRetriever, IGraphRetriever,
    QueryRequest, RetrievalResult, RetrievalResponse, RetrievalMode,
    BaseRetriever
)
from ..interfaces.base import IConfigurable, IMonitorable

logger = logging.getLogger(__name__)


class RetrievalLayer(Enum):
    """检索层级"""
    SEMANTIC = "semantic"        # 语义检索层
    GRAPH = "graph"            # 图谱检索层
    CONSTRAINT = "constraint"   # 约束验证层


class RetrievalStrategy(Enum):
    """检索策略"""
    CONSERVATIVE = "conservative"    # 保守策略：高精度，低召回
    BALANCED = "balanced"           # 平衡策略：精度与召回平衡
    AGGRESSIVE = "aggressive"       # 激进策略：高召回，后过滤


@dataclass
class LayerResult:
    """层级检索结果"""
    layer: RetrievalLayer
    results: List[RetrievalResult]
    confidence: float
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalConfig:
    """检索配置"""
    # 检索策略
    strategy: RetrievalStrategy = RetrievalStrategy.BALANCED

    # 层级权重
    semantic_weight: float = 0.3
    graph_weight: float = 0.5
    constraint_weight: float = 0.2

    # 扩展因子
    semantic_expansion_factor: int = 3    # 语义层扩展倍数
    graph_pruning_ratio: float = 0.7     # 图层剪枝比例

    # 质量阈值
    min_confidence: float = 0.3
    min_similarity: float = 0.5

    # 性能参数
    max_parallel_layers: int = 2
    timeout_per_layer: float = 10.0

    # 约束配置
    enable_safety_check: bool = True
    enable_domain_rules: bool = True
    enable_evidence_validation: bool = True


class ThreeLayerRetriever(BaseRetriever, IThreeLayerRetriever):
    """
    三层检索引擎

    实现渐进式检索架构，通过多层级处理提高检索精度。
    """

    def __init__(self, name: str = "ThreeLayerRetriever", version: str = "2.0.0"):
        super().__init__(name, version)
        self._semantic_retriever: Optional[ISemanticRetriever] = None
        self._graph_retriever: Optional[IGraphRetriever] = None
        self._constraint_validator = None
        self._config = RetrievalConfig()
        self._metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'average_response_time': 0.0,
            'layer_stats': {
                'semantic': {'hits': 0, 'avg_time': 0.0},
                'graph': {'hits': 0, 'avg_time': 0.0},
                'constraint': {'hits': 0, 'avg_time': 0.0}
            }
        }

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化三层检索引擎"""
        try:
            if config:
                self._update_config(config)

            # 初始化子组件
            success = True
            if self._semantic_retriever:
                success &= await self._semantic_retriever.initialize(config.get('semantic', {}))
            if self._graph_retriever:
                success &= await self._graph_retriever.initialize(config.get('graph', {}))

            if success:
                logger.info(f"✅ 三层检索引擎初始化成功: {self.name}")
                self._state = self.ComponentState.READY
            else:
                logger.error(f"❌ 三层检索引擎初始化失败: {self.name}")
                self._state = self.ComponentState.ERROR

            return success

        except Exception as e:
            logger.error(f"❌ 三层检索引擎初始化异常 {self.name}: {e}")
            self._state = self.ComponentState.ERROR
            return False

    def set_semantic_retriever(self, retriever: ISemanticRetriever) -> None:
        """设置语义检索器"""
        self._semantic_retriever = retriever

    def set_graph_retriever(self, retriever: IGraphRetriever) -> None:
        """设置图谱检索器"""
        self._graph_retriever = retriever

    def set_constraint_validator(self, validator) -> None:
        """设置约束验证器"""
        self._constraint_validator = validator

    def _update_config(self, config: Dict[str, Any]) -> None:
        """更新配置"""
        # 更新检索配置
        if 'strategy' in config:
            self._config.strategy = RetrievalStrategy(config['strategy'])
        if 'semantic_weight' in config:
            self._config.semantic_weight = config['semantic_weight']
        if 'graph_weight' in config:
            self._config.graph_weight = config['graph_weight']
        if 'constraint_weight' in config:
            self._config.constraint_weight = config['constraint_weight']

        # 更新阈值
        if 'min_confidence' in config:
            self._config.min_confidence = config['min_confidence']
        if 'min_similarity' in config:
            self._config.min_similarity = config['min_similarity']

        logger.info(f"检索配置已更新: {self.name}")

    async def execute_three_layer_search(self, request: QueryRequest) -> RetrievalResponse:
        """
        执行三层检索

        Args:
            request: 查询请求

        Returns:
            RetrievalResponse: 检索响应
        """
        start_time = asyncio.get_event_loop().time()
        self._metrics['total_queries'] += 1

        try:
            # 1. 分析查询请求
            strategy = self._analyze_retrieval_strategy(request)

            # 2. 执行三层检索
            layer_results = await self._execute_layered_retrieval(request, strategy)

            # 3. 融合层级结果
            fused_results = await self._fuse_layer_results(layer_results, request)

            # 4. 应用最终排序和过滤
            final_results = self._apply_final_ranking(fused_results)

            # 5. 构建响应
            execution_time = asyncio.get_event_loop().time() - start_time
            response = RetrievalResponse(
                query_id=request.query_id,
                results=final_results,
                total_results=len(final_results),
                execution_time=execution_time,
                metadata={
                    'strategy': strategy.value,
                    'layer_results': len(layer_results),
                    'fusion_method': 'weighted_average',
                    'quality_score': self._calculate_quality_score(final_results)
                }
            )

            # 更新指标
            self._update_metrics(True, execution_time)
            self._state = self.ComponentState.READY

            logger.info(f"三层检索完成: {len(final_results)} 结果, {execution_time:.3f}s")
            return response

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(False, execution_time)
            self._state = self.ComponentState.ERROR
            logger.error(f"三层检索失败: {e}")

            # 返回错误响应
            return RetrievalResponse(
                query_id=request.query_id,
                results=[],
                total_results=0,
                execution_time=execution_time,
                error=str(e)
            )

    def _analyze_retrieval_strategy(self, request: QueryRequest) -> RetrievalStrategy:
        """分析检索策略"""
        # 基于查询复杂度和领域选择策略
        if request.complexity == QueryComplexity.SIMPLE:
            return RetrievalStrategy.CONSERVATIVE
        elif request.complexity == QueryComplexity.COMPLEX:
            return RetrievalStrategy.AGGRESSIVE
        else:
            return RetrievalStrategy.BALANCED

    async def _execute_layered_retrieval(
        self,
        request: QueryRequest,
        strategy: RetrievalStrategy
    ) -> List[LayerResult]:
        """执行层级检索"""
        layer_results = []

        # 第一层：语义检索
        if self._semantic_retriever:
            semantic_result = await self._execute_semantic_layer(request, strategy)
            layer_results.append(semantic_result)

        # 第二层：图谱检索（基于第一层结果）
        if self._graph_retriever and layer_results:
            graph_result = await self._execute_graph_layer(request, layer_results[0], strategy)
            layer_results.append(graph_result)

        # 第三层：约束验证（基于前两层结果）
        if self._constraint_validator and len(layer_results) >= 2:
            constraint_result = await self._execute_constraint_layer(request, layer_results, strategy)
            layer_results.append(constraint_result)

        return layer_results

    async def _execute_semantic_layer(
        self,
        request: QueryRequest,
        strategy: RetrievalStrategy
    ) -> LayerResult:
        """执行语义检索层"""
        start_time = asyncio.get_event_loop().time()

        try:
            # 根据策略调整参数
            top_k = request.top_k * self._config.semantic_expansion_factor

            # 构建语义检索请求
            semantic_request = QueryRequest(
                query_id=f"{request.query_id}_semantic",
                query_text=request.query_text,
                domain=request.domain,
                top_k=top_k,
                min_similarity=self._config.min_similarity,
                filters=request.filters,
                mode=RetrievalMode.SEMANTIC
            )

            # 执行语义检索
            response = await self._semantic_retriever.retrieve(semantic_request)

            # 提取结果
            results = response.results[:top_k] if response.results else []
            confidence = sum(r.score for r in results) / len(results) if results else 0.0
            execution_time = asyncio.get_event_loop().time() - start_time

            # 更新层级统计
            self._metrics['layer_stats']['semantic']['hits'] += len(results)
            self._metrics['layer_stats']['semantic']['avg_time'] = (
                (self._metrics['layer_stats']['semantic']['avg_time'] + execution_time) / 2
            )

            return LayerResult(
                layer=RetrievalLayer.SEMANTIC,
                results=results,
                confidence=confidence,
                execution_time=execution_time,
                metadata={
                    'original_top_k': request.top_k,
                    'expanded_top_k': top_k,
                    'strategy': strategy.value
                }
            )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"语义检索层失败: {e}")

            return LayerResult(
                layer=RetrievalLayer.SEMANTIC,
                results=[],
                confidence=0.0,
                execution_time=execution_time,
                metadata={'error': str(e)}
            )

    async def _execute_graph_layer(
        self,
        request: QueryRequest,
        semantic_result: LayerResult,
        strategy: RetrievalStrategy
    ) -> LayerResult:
        """执行图谱检索层"""
        start_time = asyncio.get_event_loop().time()

        try:
            # 使用语义检索结果作为输入
            candidate_ids = [r.document_id for r in semantic_result.results]

            # 构建图谱检索请求
            graph_request = QueryRequest(
                query_id=f"{request.query_id}_graph",
                query_text=request.query_text,
                domain=request.domain,
                top_k=request.top_k,
                filters={**request.filters, 'candidate_ids': candidate_ids},
                mode=RetrievalMode.GRAPH
            )

            # 执行图谱检索
            response = await self._graph_retriever.retrieve(graph_request)

            # 提取结果
            results = response.results[:request.top_k] if response.results else []
            confidence = sum(r.score for r in results) / len(results) if results else 0.0
            execution_time = asyncio.get_event_loop().time() - start_time

            # 更新层级统计
            self._metrics['layer_stats']['graph']['hits'] += len(results)
            self._metrics['layer_stats']['graph']['avg_time'] = (
                (self._metrics['layer_stats']['graph']['avg_time'] + execution_time) / 2
            )

            return LayerResult(
                layer=RetrievalLayer.GRAPH,
                results=results,
                confidence=confidence,
                execution_time=execution_time,
                metadata={
                    'candidate_count': len(candidate_ids),
                    'filtered_count': len(results),
                    'strategy': strategy.value
                }
            )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"图谱检索层失败: {e}")

            return LayerResult(
                layer=RetrievalLayer.GRAPH,
                results=[],
                confidence=0.0,
                execution_time=execution_time,
                metadata={'error': str(e)}
            )

    async def _execute_constraint_layer(
        self,
        request: QueryRequest,
        layer_results: List[LayerResult],
        strategy: RetrievalStrategy
    ) -> LayerResult:
        """执行约束验证层"""
        start_time = asyncio.get_event_loop().time()

        try:
            # 合并前两层结果
            all_results = []
            for result in layer_results:
                all_results.extend(result.results)

            # 去重
            unique_results = self._deduplicate_results(all_results)

            # 执行约束验证
            validated_results = []
            for result in unique_results:
                validation = await self._constraint_validator.validate(result, request)
                if validation.is_valid:
                    result.score *= validation.confidence  # 应用验证权重
                    validated_results.append(result)

            # 按分数排序
            validated_results.sort(key=lambda x: x.score, reverse=True)
            final_results = validated_results[:request.top_k]

            confidence = sum(r.score for r in final_results) / len(final_results) if final_results else 0.0
            execution_time = asyncio.get_event_loop().time() - start_time

            # 更新层级统计
            self._metrics['layer_stats']['constraint']['hits'] += len(final_results)
            self._metrics['layer_stats']['constraint']['avg_time'] = (
                (self._metrics['layer_stats']['constraint']['avg_time'] + execution_time) / 2
            )

            return LayerResult(
                layer=RetrievalLayer.CONSTRAINT,
                results=final_results,
                confidence=confidence,
                execution_time=execution_time,
                metadata={
                    'input_count': len(unique_results),
                    'validated_count': len(validated_results),
                    'final_count': len(final_results),
                    'strategy': strategy.value
                }
            )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"约束验证层失败: {e}")

            return LayerResult(
                layer=RetrievalLayer.CONSTRAINT,
                results=[],
                confidence=0.0,
                execution_time=execution_time,
                metadata={'error': str(e)}
            )

    async def _fuse_layer_results(
        self,
        layer_results: List[LayerResult],
        request: QueryRequest
    ) -> List[RetrievalResult]:
        """融合层级结果"""
        if not layer_results:
            return []

        # 使用权重融合算法
        weight_map = {
            RetrievalLayer.SEMANTIC: self._config.semantic_weight,
            RetrievalLayer.GRAPH: self._config.graph_weight,
            RetrievalLayer.CONSTRAINT: self._config.constraint_weight
        }

        # 收集所有结果并计算融合分数
        result_scores = {}
        for layer_result in layer_results:
            weight = weight_map.get(layer_result.layer, 0.0)
            for result in layer_result.results:
                if result.document_id not in result_scores:
                    result_scores[result.document_id] = {
                        'result': result,
                        'weighted_score': 0.0,
                        'layer_count': 0
                    }

                result_scores[result.document_id]['weighted_score'] += result.score * weight
                result_scores[result.document_id]['layer_count'] += 1

        # 排序并返回结果
        fused_results = []
        for doc_id, data in result_scores.items():
            result = data['result']
            result.score = data['weighted_score']  # 更新为融合分数
            fused_results.append(result)

        # 按分数排序并返回top_k
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results[:request.top_k]

    def _deduplicate_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """去重结果"""
        seen_ids = set()
        unique_results = []

        for result in results:
            if result.document_id not in seen_ids:
                seen_ids.add(result.document_id)
                unique_results.append(result)

        return unique_results

    def _apply_final_ranking(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """应用最终排序"""
        # 过滤低质量结果
        filtered_results = [
            r for r in results
            if r.score >= self._config.min_confidence
        ]

        # 确保至少返回一些结果
        if not filtered_results and results:
            filtered_results = results[:1]  # 返回最高分的结果

        return filtered_results

    def _calculate_quality_score(self, results: List[RetrievalResult]) -> float:
        """计算质量分数"""
        if not results:
            return 0.0

        # 基于分数分布计算质量分数
        scores = [r.score for r in results]
        avg_score = sum(scores) / len(scores)

        # 考虑结果数量的质量分数
        result_factor = min(len(results) / 10, 1.0)  # 假设10个结果为满分

        return (avg_score + result_factor) / 2

    def _update_metrics(self, success: bool, response_time: float) -> None:
        """更新指标"""
        if success:
            self._metrics['successful_queries'] += 1

        # 更新平均响应时间
        total_queries = self._metrics['total_queries']
        current_avg = self._metrics['average_response_time']
        self._metrics['average_response_time'] = (
            (current_avg * (total_queries - 1) + response_time) / total_queries
        )

    # IRetriever接口实现
    async def retrieve(self, request: QueryRequest) -> RetrievalResponse:
        """检索接口实现"""
        return await self.execute_three_layer_search(request)

    async def get_supported_modes(self) -> List[RetrievalMode]:
        """获取支持的检索模式"""
        return [
            RetrievalMode.SEMANTIC,
            RetrievalMode.GRAPH,
            RetrievalMode.HYBRID
        ]

    def get_metrics(self) -> Dict[str, Any]:
        """获取检索指标"""
        base_metrics = super().get_metrics()
        return {
            **base_metrics,
            'total_queries': self._metrics['total_queries'],
            'successful_queries': self._metrics['successful_queries'],
            'success_rate': (
                self._metrics['successful_queries'] / max(self._metrics['total_queries'], 1)
            ),
            'average_response_time': self._metrics['average_response_time'],
            'layer_statistics': self._metrics['layer_stats']
        }


# 导出
__all__ = [
    'ThreeLayerRetriever',
    'RetrievalLayer',
    'RetrievalStrategy',
    'LayerResult',
    'RetrievalConfig'
]