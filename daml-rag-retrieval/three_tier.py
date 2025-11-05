#!/usr/bin/env python3
"""
DAML-RAG Framework 三层检索系统
集成向量检索、知识图谱和规则过滤的完整检索系统
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time
import uuid

from .vector.base import BaseVectorRetriever, Document
from .knowledge.neo4j import Neo4jKnowledgeRetriever
from .rules.engine import RuleEngine, RuleContext

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """检索配置"""
    # 向量检索配置
    vector_top_k: int = 20
    vector_threshold: float = 0.6
    vector_weight: float = 0.3

    # 知识图谱配置
    graph_enabled: bool = True
    graph_top_k: int = 10
    graph_weight: float = 0.5

    # 规则过滤配置
    rules_enabled: bool = True
    rules_weight: float = 0.2

    # 缓存配置
    cache_enabled: bool = True
    cache_ttl: int = 300  # 5分钟

    # 超时配置
    total_timeout: float = 5.0  # 5秒超时
    vector_timeout: float = 1.0
    graph_timeout: float = 2.0
    rules_timeout: float = 0.5


@dataclass
class RetrievalRequest:
    """检索请求"""
    query: str
    query_vector: Optional[List[float]] = None
    context: Optional[Dict[str, Any]] = None
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 5
    include_metadata: bool = True
    include_explanations: bool = True


@dataclass
class RetrievalStage:
    """检索阶段结果"""
    stage_name: str
    results: List[Document]
    scores: List[float]
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    explanation: Optional[str] = None


@dataclass
class ThreeTierRetrievalResult:
    """三层检索结果"""
    query: str
    final_results: List[Document]
    final_scores: List[float]
    stages: List[RetrievalStage]
    total_execution_time: float
    token_estimate: int
    explanations: List[str]
    warnings: List[str]
    errors: List[str]
    statistics: Dict[str, Any] = field(default_factory=dict)


class ThreeTierRetriever:
    """三层检索系统"""

    def __init__(
        self,
        vector_retriever: BaseVectorRetriever,
        graph_retriever: Neo4jKnowledgeRetriever,
        rule_engine: RuleEngine,
        config: RetrievalConfig
    ):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.rule_engine = rule_engine
        self.config = config
        self._initialized = False

        # 缓存
        self._cache: Dict[str, Any] = {}

        # 统计信息
        self._stats = {
            "total_retrievals": 0,
            "stage_stats": {
                "vector": {"count": 0, "total_time": 0.0, "avg_time": 0.0},
                "graph": {"count": 0, "total_time": 0.0, "avg_time": 0.0},
                "rules": {"count": 0, "total_time": 0.0, "avg_time": 0.0}
            },
            "cache_stats": {
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0
            }
        }

    async def initialize(self) -> None:
        """初始化三层检索系统"""
        try:
            # 初始化各个组件
            await self.vector_retriever.initialize()
            if self.graph_retriever:
                await self.graph_retriever.initialize()
            # RuleEngine 不需要初始化

            self._initialized = True
            logger.info("Three-tier retrieval system initialized")

        except Exception as e:
            logger.error(f"Failed to initialize three-tier retriever: {e}")
            raise

    async def retrieve(self, request: RetrievalRequest) -> ThreeTierRetrievalResult:
        """执行三层检索"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        self._stats["total_retrievals"] += 1

        try:
            # 生成缓存键
            cache_key = self._generate_cache_key(request)

            # 检查缓存
            if self.config.cache_enabled:
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    self._stats["cache_stats"]["hits"] += 1
                    logger.debug("Cache hit for retrieval request")
                    return cached_result

            self._stats["cache_stats"]["misses"] += 1

            # 执行三层检索
            stages = []
            explanations = []
            warnings = []
            errors = []

            # 第一阶段：向量检索
            vector_stage = await self._vector_retrieval(request)
            stages.append(vector_stage)
            explanations.append(vector_stage.explanation or "向量检索完成")

            # 第二阶段：知识图谱检索
            graph_stage = await self._graph_retrieval(request, vector_stage.results)
            stages.append(graph_stage)
            explanations.append(graph_stage.explanation or "知识图谱检索完成")

            # 第三阶段：规则过滤
            rules_stage = await self._rules_retrieval(request, graph_stage.results)
            stages.append(rules_stage)
            explanations.append(rules_stage.explanation or "规则过滤完成")

            # 计算最终分数
            final_results, final_scores = self._calculate_final_scores(stages)

            # 生成解释
            retrieval_explanations = self._generate_explanations(stages, request)

            # 估算Token使用量
            token_estimate = self._estimate_tokens(final_results)

            total_time = time.time() - start_time

            # 更新统计信息
            self._update_stage_stats(stages)

            # 创建结果
            result = ThreeTierRetrievalResult(
                query=request.query,
                final_results=final_results,
                final_scores=final_scores,
                stages=stages,
                total_execution_time=total_time,
                token_estimate=token_estimate,
                explanations=retrieval_explanations,
                warnings=warnings,
                errors=errors,
                statistics=self._get_retrieval_statistics()
            )

            # 缓存结果
            if self.config.cache_enabled:
                self._store_in_cache(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Three-tier retrieval failed: {e}")
            # 创建错误结果
            return ThreeTierRetrievalResult(
                query=request.query,
                final_results=[],
                final_scores=[],
                stages=[],
                total_execution_time=time.time() - start_time,
                token_estimate=0,
                explanations=[f"检索失败: {str(e)}"],
                warnings=[],
                errors=[str(e)],
                statistics=self._get_retrieval_statistics()
            )

    async def _vector_retrieval(self, request: RetrievalRequest) -> RetrievalStage:
        """第一阶段：向量检索"""
        start_time = time.time()

        try:
            # 如果没有查询向量，需要生成
            query_vector = request.query_vector
            if query_vector is None:
                # 这里应该调用embedding模型生成向量
                # 暂时跳过向量检索
                logger.warning("No query vector provided, skipping vector retrieval")
                return RetrievalStage(
                    stage_name="vector_retrieval",
                    results=[],
                    scores=[],
                    execution_time=time.time() - start_time,
                    explanation="跳过向量检索：没有查询向量",
                    metadata={"skipped": True}
                )

            # 执行向量检索
            vector_result = await self.vector_retriever.search(
                query_vector=query_vector,
                top_k=self.config.vector_top_k,
                score_threshold=self.config.vector_threshold,
                filter_condition=request.filters
            )

            execution_time = time.time() - start_time

            return RetrievalStage(
                stage_name="vector_retrieval",
                results=vector_result.documents,
                scores=vector_result.scores,
                execution_time=execution_time,
                explanation=f"向量检索完成：找到 {len(vector_result.documents)} 个候选文档",
                metadata={
                    "vector_top_k": self.config.vector_top_k,
                    "vector_threshold": self.config.vector_threshold,
                    "total_found": vector_result.total_found
                }
            )

        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return RetrievalStage(
                stage_name="vector_retrieval",
                results=[],
                scores=[],
                execution_time=time.time() - start_time,
                explanation=f"向量检索失败: {str(e)}",
                metadata={"error": str(e)}
            )

    async def _graph_retrieval(self, request: RetrievalRequest, candidate_docs: List[Document]) -> RetrievalStage:
        """第二阶段：知识图谱检索"""
        start_time = time.time()

        if not self.config.graph_enabled or not self.graph_retriever:
            logger.info("Graph retrieval disabled, using vector results directly")
            return RetrievalStage(
                stage_name="graph_retrieval",
                results=candidate_docs,
                scores=[1.0] * len(candidate_docs),
                execution_time=0.0,
                explanation="跳过知识图谱检索：已禁用",
                metadata={"skipped": True}
            )

        if not candidate_docs:
            return RetrievalStage(
                stage_name="graph_retrieval",
                results=[],
                scores=[],
                execution_time=0.0,
                explanation="知识图谱检索：没有候选文档",
                metadata={"no_candidates": True}
            )

        try:
            # 构建图查询
            graph_results = []

            # 这里应该基于候选文档的ID和查询上下文构建Cypher查询
            # 简化实现：假设我们有一些预定义的查询模式
            for doc in candidate_docs[:self.config.graph_top_k]:
                doc_id = doc.id if hasattr(doc, 'id') else str(uuid.uuid4())

                # 示例图查询 - 根据领域不同会有不同的查询
                graph_docs = await self._execute_graph_query(doc_id, request)
                graph_results.extend(graph_docs)

            execution_time = time.time() - start_time

            return RetrievalStage(
                stage_name="graph_retrieval",
                results=graph_results,
                scores=[1.0] * len(graph_results),
                execution_time=execution_time,
                explanation=f"知识图谱检索完成：筛选出 {len(graph_results)} 个精确结果",
                metadata={
                    "graph_top_k": self.config.graph_top_k,
                    "candidates_count": len(candidate_docs),
                    "graph_results_count": len(graph_results)
                }
            )

        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            return RetrievalStage(
                stage_name="graph_retrieval",
                results=candidate_docs,  # 回退到向量检索结果
                scores=[1.0] * len(candidate_docs),
                execution_time=time.time() - start_time,
                explanation=f"知识图谱检索失败，使用向量结果: {str(e)}",
                metadata={"error": str(e), "fallback": True}
            )

    async def _rules_retrieval(self, request: RetrievalRequest, candidate_docs: List[Document]) -> RetrievalStage:
        """第三阶段：规则过滤"""
        start_time = time.time()

        if not self.config.rules_enabled:
            logger.info("Rules filtering disabled")
            return RetrievalStage(
                stage_name="rules_filtering",
                results=candidate_docs,
                scores=[1.0] * len(candidate_docs),
                execution_time=0.0,
                explanation="跳过规则过滤：已禁用",
                metadata={"skipped": True}
            )

        if not candidate_docs:
            return RetrievalStage(
                stage_name="rules_filtering",
                results=[],
                scores=[],
                execution_time=0.0,
                explanation="规则过滤：没有候选文档",
                metadata={"no_candidates": True}
            )

        try:
            # 创建规则上下文
            rule_context = RuleContext(
                user_profile=request.context.get("user_profile", {}),
                query_context=request.context or {},
                session_data=request.context.get("session_data", {}),
                system_config={}
            )

            # 转换文档为字典格式
            items = [doc.__dict__ if hasattr(doc, '__dict__') else {} for doc in candidate_docs]

            # 应用规则
            filter_result = await self.rule_engine.apply_rules(items, rule_context)

            # 转换回Document对象
            filtered_docs = []
            for i, item in enumerate(filter_result.filtered_items):
                if i < len(candidate_docs):
                    filtered_docs.append(candidate_docs[i])

            execution_time = time.time() - start_time

            explanation_parts = [
                f"规则过滤完成：{len(filtered_docs)}/{len(candidate_docs)} 文档通过规则检查",
                f"生成 {len(filter_result.warnings)} 个警告"
            ]
            if filter_result.errors:
                explanation_parts.append(f"遇到 {len(filter_result.errors)} 个错误")

            return RetrievalStage(
                stage_name="rules_filtering",
                results=filtered_docs,
                scores=[1.0] * len(filtered_docs),
                execution_time=execution_time,
                explanation="; ".join(explanation_parts),
                metadata={
                    "passed_count": len(filtered_docs),
                    "total_count": len(candidate_docs),
                    "warnings_count": len(filter_result.warnings),
                    "errors_count": len(filter_result.errors),
                    "modifications": filter_result.modifications
                }
            )

        except Exception as e:
            logger.error(f"Rules filtering failed: {e}")
            return RetrievalStage(
                stage_name="rules_filtering",
                results=candidate_docs,  # 回退到图检索结果
                scores=[1.0] * len(candidate_docs),
                execution_time=time.time() - start_time,
                explanation=f"规则过滤失败，使用原图结果: {str(e)}",
                metadata={"error": str(e), "fallback": True}
            )

    async def _execute_graph_query(self, doc_id: str, request: RetrievalRequest) -> List[Document]:
        """执行图查询"""
        try:
            # 这里应该根据领域和查询类型构建Cypher查询
            # 简化实现：返回空结果
            return []

        except Exception as e:
            logger.error(f"Graph query execution failed: {e}")
            return []

    def _calculate_final_scores(self, stages: List[RetrievalStage]) -> Tuple[List[Document], List[float]]:
        """计算最终分数"""
        if not stages:
            return [], []

        # 获取最后一个阶段的结果
        final_stage = stages[-1]
        documents = final_stage.results
        base_scores = final_stage.scores

        # 如果只有一个阶段，直接返回
        if len(stages) == 1:
            return documents, base_scores

        # 多阶段分数融合
        final_scores = []
        for i, doc in enumerate(documents):
            if i < len(base_scores):
                # 基于阶段权重计算最终分数
                score = 0.0
                total_weight = 0.0

                for stage in stages:
                    if stage.stage_name == "vector_retrieval":
                        weight = self.config.vector_weight
                    elif stage.stage_name == "graph_retrieval":
                        weight = self.config.graph_weight
                    elif stage.stage_name == "rules_filtering":
                        weight = self.config.rules_weight
                    else:
                        continue

                    # 如果该阶段有该文档的分数
                    stage_scores = stage.scores
                    if i < len(stage_scores):
                        score += stage_scores[i] * weight
                        total_weight += weight

                # 归一化分数
                if total_weight > 0:
                    final_scores.append(score / total_weight)
                else:
                    final_scores.append(base_scores[i])
            else:
                final_scores.append(0.0)

        return documents, final_scores

    def _generate_explanations(self, stages: List[RetrievalRequest], request: RetrievalRequest) -> List[str]:
        """生成检索解释"""
        explanations = []

        for stage in stages:
            if stage.explanation:
                explanations.append(f"【{stage.stage_name}】{stage.explanation}")

        # 添加权重说明
        if len(stages) > 1:
            weight_info = [
                f"向量检索权重: {self.config.vector_weight}",
                f"知识图谱权重: {self.config.graph_weight}",
                f"规则过滤权重: {self.config.rules_weight}"
            ]
            explanations.append("【权重配置】" + "; ".join(weight_info))

        return explanations

    def _estimate_tokens(self, documents: List[Document]) -> int:
        """估算Token使用量"""
        if not documents:
            return 0

        total_length = sum(len(doc.content) for doc in documents)
        # 估算：1个Token约等于4个字符
        return total_length // 4

    def _generate_cache_key(self, request: RetrievalRequest) -> str:
        """生成缓存键"""
        key_parts = [
            request.query,
            str(request.top_k),
            str(self.config.vector_top_k),
            str(self.config.vector_threshold),
            str(self.config.graph_enabled),
            str(self.config.rules_enabled)
        ]

        # 添加上下文信息
        if request.context:
            context_str = json.dumps(request.context, sort_keys=True)
            key_parts.append(context_str)

        import hashlib
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[ThreeTierRetrievalResult]:
        """从缓存获取结果"""
        if cache_key in self._cache:
            cached_time = self._cache[cache_key].get("timestamp", 0)
            current_time = time.time()

            if current_time - cached_time < self.config.cache_ttl:
                return self._cache[cache_key].get("result")
            else:
                del self._cache[cache_key]

        return None

    def _store_in_cache(self, cache_key: str, result: ThreeTierRetrievalResult):
        """存储结果到缓存"""
        self._cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }

        # 限制缓存大小
        if len(self._cache) > 1000:
            # 删除最旧的缓存项
            oldest_key = min(self.cache.keys(), key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]

    def _update_stage_stats(self, stages: List[RetrievalStage]):
        """更新阶段统计"""
        for stage in stages:
            stage_name = stage.stage_name.replace("_retrieval", "")
            if stage_name in self._stats["stage_stats"]:
                stats = self._stats["stage_stats"][stage_name]
                stats["count"] += 1
                stats["total_time"] += stage.execution_time
                stats["avg_time"] = stats["total_time"] / stats["count"]

    def _get_retrieval_statistics(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        return {
            "total_retrievals": self._stats["total_retrievals"],
            "stage_stats": self._stats["stage_stats"],
            "cache_stats": self._stats["cache_stats"],
            "config": {
                "vector_top_k": self.config.vector_top_k,
                "vector_threshold": self.config.vector_threshold,
                "vector_weight": self.config.vector_weight,
                "graph_enabled": self.config.graph_enabled,
                "graph_weight": self.config.graph_weight,
                "rules_enabled": self.config.rules_enabled,
                "rules_weight": self.config.rules_weight,
                "cache_enabled": self.config.cache_enabled,
                "cache_ttl": self.config.cache_ttl
            }
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        stats = self._get_retrieval_statistics()

        # 计算平均响应时间
        total_retrievals = stats["total_retrievals"]
        if total_retrievals > 0:
            avg_vector_time = stats["stage_stats"]["vector"]["avg_time"]
            avg_graph_time = stats["stage_stats"]["graph"]["avg_time"]
            avg_rules_time = stats["stage_stats"]["rules"]["avg_time"]
            total_avg_time = avg_vector_time + avg_graph_time + avg_rules_time
        else:
            total_avg_time = 0.0

        # 计算缓存命中率
        cache_hits = stats["cache_stats"]["hits"]
        cache_misses = stats["cache_stats"]["misses"]
        total_cache_requests = cache_hits + cache_misses
        cache_hit_rate = (cache_hits / total_cache_requests) if total_cache_requests > 0 else 0.0

        return {
            "average_response_time": total_avg_time,
            "cache_hit_rate": cache_hit_rate,
            "total_retrievals": total_retrievals,
            "vector_retrieval_avg_time": avg_vector_time,
            "graph_retrieval_avg_time": avg_graph_time,
            "rules_filtering_avg_time": avg_rules_time,
            "cache_hit_count": cache_hits,
            "cache_miss_count": cache_misses
        }

    async def close(self):
        """关闭连接"""
        if self.vector_retriever:
            await self.vector_retriever.close()
        if self.graph_retriever:
            await self.graph_retriever.close()
        # RuleEngine不需要关闭
        logger.info("Three-tier retriever closed")


# 便捷的创建函数
def create_three_tier_retriever(
    vector_retriever: BaseVectorRetriever,
    graph_retriever: Optional[Neo4jKnowledgeRetriever] = None,
    rule_engine: Optional[RuleEngine] = None,
    config: Optional[RetrievalConfig] = None
) -> ThreeTierRetriever:
    """创建三层检索器的便捷函数"""
    if config is None:
        config = RetrievalConfig()

    if rule_engine is None:
        from .rules.engine import create_default_rule_engine
        rule_engine = create_default_rule_engine()

    return ThreeTierRetriever(
        vector_retriever=vector_retriever,
        graph_retriever=graph_retriever,
        rule_engine=rule_engine,
        config=config
    )