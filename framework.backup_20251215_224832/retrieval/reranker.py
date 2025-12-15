# -*- coding: utf-8 -*-
"""
DAML-RAG重排序器 v2.0

实现多策略融合的结果重排序：
- 多维度权重融合
- 动态权重调整
- 多样性优化
- 时效性排序

版本：v2.0.0
更新日期：2025-11-17
设计原则：多策略融合、动态优化、多样性保证
"""

import asyncio
import logging
import math
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from ..interfaces.retrieval import RetrievalResult, QueryRequest, IReranker

logger = logging.getLogger(__name__)


class RerankingStrategy(Enum):
    """重排序策略"""
    WEIGHTED_FUSION = "weighted_fusion"      # 加权融合
    RECIPROCAL_RANK = "reciprocal_rank"      # 倒数排名融合
    DIVERSITY_PROMOTION = "diversity_promotion"  # 多样性提升
    TEMPORAL_RELEVANCE = "temporal_relevance"   # 时效性相关
    LEARNING_TO_RANK = "learning_to_rank"     # 学习排序
    HYBRID = "hybrid"                        # 混合策略


class ScoringDimension(Enum):
    """评分维度"""
    RELEVANCE = "relevance"           # 相关性
    QUALITY = "quality"              # 质量
    RECENCY = "recency"              # 时效性
    POPULARITY = "popularity"        # 流行度
    AUTHORITY = "authority"          # 权威性
    DIVERSITY = "diversity"          # 多样性
    PERSONALIZATION = "personalization"  # 个性化


@dataclass
class RerankingConfig:
    """重排序配置"""
    # 策略配置
    primary_strategy: RerankingStrategy = RerankingStrategy.WEIGHTED_FUSION
    secondary_strategy: Optional[RerankingStrategy] = None
    hybrid_weights: Dict[str, float] = field(default_factory=dict)

    # 维度权重
    dimension_weights: Dict[ScoringDimension, float] = field(default_factory=lambda: {
        ScoringDimension.RELEVANCE: 0.4,
        ScoringDimension.QUALITY: 0.3,
        ScoringDimension.RECENCY: 0.1,
        ScoringDimension.POPULARITY: 0.1,
        ScoringDimension.AUTHORITY: 0.1
    })

    # 多样性参数
    enable_diversity_promotion: bool = True
    diversity_threshold: float = 0.7     # 相似度阈值
    max_similar_items: int = 3           # 最大相似项目数

    # 时效性参数
    enable_recency_boost: bool = True
    recency_half_life_days: int = 30     # 半衰期天数
    recency_decay_factor: float = 0.1    # 衰减因子

    # 个性化参数
    enable_personalization: bool = False
    user_preference_weight: float = 0.2

    # 学习排序参数
    enable_learning_to_rank: bool = False
    ltr_model_path: Optional[str] = None

    # 性能参数
    max_results: int = 100              # 最大处理结果数
    parallel_processing: bool = True    # 并行处理


@dataclass
class RerankingResult:
    """重排序结果"""
    original_results: List[RetrievalResult]
    reranked_results: List[RetrievalResult]
    strategy_used: RerankingStrategy
    execution_time: float
    improvements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Reranker(IReranker):
    """
    重排序器

    对检索结果进行多策略融合和重新排序。
    """

    def __init__(self, name: str = "Reranker", version: str = "2.0.0"):
        super().__init__(name, version)
        self._config = RerankingConfig()
        self._user_preferences = {}
        self._item_similarity_cache = {}
        self._ltr_model = None

        # 性能指标
        self._metrics = {
            'total_rerankings': 0,
            'average_reranking_time': 0.0,
            'diversity_improvements': 0,
            'quality_improvements': 0,
            'strategy_usage': {strategy.value: 0 for strategy in RerankingStrategy}
        }

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化重排序器"""
        try:
            if config:
                self._update_config(config)

            # 初始化学习排序模型
            if self._config.enable_learning_to_rank:
                await self._initialize_ltr_model()

            logger.info(f"✅ 重排序器初始化成功: {self.name}")
            self._state = self.ComponentState.READY
            return True

        except Exception as e:
            logger.error(f"❌ 重排序器初始化失败 {self.name}: {e}")
            self._state = self.ComponentState.ERROR
            return False

    async def _initialize_ltr_model(self) -> None:
        """初始化学习排序模型"""
        # 这里应该加载预训练的LTR模型
        # 示例实现：
        # if self._config.ltr_model_path:
        #     self._ltr_model = joblib.load(self._config.ltr_model_path)
        pass

    def _update_config(self, config: Dict[str, Any]) -> None:
        """更新配置"""
        if 'primary_strategy' in config:
            self._config.primary_strategy = RerankingStrategy(config['primary_strategy'])
        if 'enable_diversity_promotion' in config:
            self._config.enable_diversity_promotion = config['enable_diversity_promotion']
        if 'diversity_threshold' in config:
            self._config.diversity_threshold = config['diversity_threshold']
        if 'dimension_weights' in config:
            for dim, weight in config['dimension_weights'].items():
                if dim in [d.value for d in ScoringDimension]:
                    self._config.dimension_weights[ScoringDimension(dim)] = weight

        logger.info(f"重排序配置已更新: {self.name}")

    async def rerank(
        self,
        results: List[RetrievalResult],
        request: QueryRequest,
        strategy: Optional[RerankingStrategy] = None
    ) -> RerankingResult:
        """执行重排序"""
        start_time = asyncio.get_event_loop().time()
        self._metrics['total_rerankings'] += 1

        try:
            if not results:
                return RerankingResult(
                    original_results=results,
                    reranked_results=results,
                    strategy_used=RerankingStrategy.WEIGHTED_FUSION,
                    execution_time=0.0
                )

            # 限制处理结果数量
            limited_results = results[:self._config.max_results]

            # 选择策略
            selected_strategy = strategy or self._config.primary_strategy

            # 执行重排序
            if selected_strategy == RerankingStrategy.WEIGHTED_FUSION:
                reranked = await self._weighted_fusion_reranking(limited_results, request)
            elif selected_strategy == RerankingStrategy.RECIPROCAL_RANK:
                reranked = await self._reciprocal_rank_fusion(limited_results, request)
            elif selected_strategy == RerankingStrategy.DIVERSITY_PROMOTION:
                reranked = await self._diversity_promotion_reranking(limited_results, request)
            elif selected_strategy == RerankingStrategy.TEMPORAL_RELEVANCE:
                reranked = await self._temporal_relevance_reranking(limited_results, request)
            elif selected_strategy == RerankingStrategy.LEARNING_TO_RANK:
                reranked = await self._learning_to_rank_reranking(limited_results, request)
            elif selected_strategy == RerankingStrategy.HYBRID:
                reranked = await self._hybrid_reranking(limited_results, request)
            else:
                reranked = limited_results  # 默认不重排

            # 计算改进指标
            improvements = self._calculate_improvements(limited_results, reranked)

            # 执行二次策略（如果配置）
            if self._config.secondary_strategy:
                reranked = await self._apply_secondary_strategy(reranked, request)

            execution_time = asyncio.get_event_loop().time() - start_time

            # 更新指标
            self._update_metrics(selected_strategy, execution_time, improvements)

            result = RerankingResult(
                original_results=results,
                reranked_results=reranked,
                strategy_used=selected_strategy,
                execution_time=execution_time,
                improvements=improvements,
                metadata={
                    'input_count': len(results),
                    'processed_count': len(limited_results),
                    'output_count': len(reranked),
                    'secondary_strategy': self._config.secondary_strategy.value if self._config.secondary_strategy else None
                }
            )

            logger.debug(f"重排序完成: {len(reranked)} 结果, {execution_time:.3f}s, 策略: {selected_strategy.value}")
            return result

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"重排序失败: {e}")

            return RerankingResult(
                original_results=results,
                reranked_results=results,
                strategy_used=RerankingStrategy.WEIGHTED_FUSION,
                execution_time=execution_time,
                metadata={'error': str(e)}
            )

    async def _weighted_fusion_reranking(
        self,
        results: List[RetrievalResult],
        request: QueryRequest
    ) -> List[RetrievalResult]:
        """加权融合重排序"""
        reranked = []

        for result in results:
            # 计算各维度分数
            scores = {}

            # 相关性分数（原始分数）
            scores[ScoringDimension.RELEVANCE] = result.score

            # 质量分数
            scores[ScoringDimension.QUALITY] = self._calculate_quality_score(result)

            # 时效性分数
            scores[ScoringDimension.RECENCY] = self._calculate_recency_score(result)

            # 流行度分数
            scores[ScoringDimension.POPULARITY] = self._calculate_popularity_score(result)

            # 权威性分数
            scores[ScoringDimension.AUTHORITY] = self._calculate_authority_score(result)

            # 多样性分数（需要全局信息，稍后计算）
            scores[ScoringDimension.DIVERSITY] = 0.0

            # 个性化分数
            if self._config.enable_personalization:
                scores[ScoringDimension.PERSONALIZATION] = self._calculate_personalization_score(result, request)
            else:
                scores[ScoringDimension.PERSONALIZATION] = 0.0

            # 计算加权总分
            total_score = 0.0
            for dimension, score in scores.items():
                weight = self._config.dimension_weights.get(dimension, 0.0)
                total_score += score * weight

            # 更新结果分数
            result.metadata['original_score'] = result.score
            result.metadata['dimension_scores'] = {d.value: s for d, s in scores.items()}
            result.score = total_score

            reranked.append(result)

        # 按新分数排序
        reranked.sort(key=lambda x: x.score, reverse=True)

        # 应用多样性提升
        if self._config.enable_diversity_promotion:
            reranked = await self._apply_diversity_boost(reranked)

        return reranked

    async def _reciprocal_rank_fusion(
        self,
        results: List[RetrievalResult],
        request: QueryRequest
    ) -> List[RetrievalResult]:
        """倒数排名融合"""
        # 这里简化实现，实际应该有多个来源的结果列表
        k = 60  # 常数，用于倒数排名融合

        for i, result in enumerate(results):
            # 计算倒数排名分数
            rr_score = 1.0 / (k + i + 1)

            # 与原始分数融合
            combined_score = 0.7 * result.score + 0.3 * rr_score
            result.score = combined_score
            result.metadata['rr_score'] = rr_score

        # 重新排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    async def _diversity_promotion_reranking(
        self,
        results: List[RetrievalResult],
        request: QueryRequest
    ) -> List[RetrievalResult]:
        """多样性提升重排序"""
        if not self._config.enable_diversity_promotion:
            return results

        return await self._apply_diversity_boost(results)

    async def _temporal_relevance_reranking(
        self,
        results: List[RetrievalResult],
        request: QueryRequest
    ) -> List[RetrievalResult]:
        """时效性相关重排序"""
        if not self._config.enable_recency_boost:
            return results

        for result in results:
            recency_score = self._calculate_recency_score(result)
            # 应用时效性提升
            boosted_score = result.score * (1.0 + self._config.recency_decay_factor * recency_score)
            result.score = boosted_score
            result.metadata['recency_boost'] = recency_score

        # 重新排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    async def _learning_to_rank_reranking(
        self,
        results: List[RetrievalResult],
        request: QueryRequest
    ) -> List[RetrievalResult]:
        """学习排序重排序"""
        if not self._config.enable_learning_to_rank or not self._ltr_model:
            return results

        # 这里应该使用LTR模型预测排序分数
        # 简化实现：使用随机分数作为示例
        for result in results:
            features = self._extract_ltr_features(result, request)
            ltr_score = self._ltr_model.predict([features])[0] if self._ltr_model else result.score
            result.metadata['ltr_score'] = ltr_score
            result.score = ltr_score

        # 重新排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    async def _hybrid_reranking(
        self,
        results: List[RetrievalResult],
        request: QueryRequest
    ) -> List[RetrievalResult]:
        """混合策略重排序"""
        # 执行多种策略并融合结果
        strategies = [
            RerankingStrategy.WEIGHTED_FUSION,
            RerankingStrategy.DIVERSITY_PROMOTION,
            RerankingStrategy.TEMPORAL_RELEVANCE
        ]

        strategy_results = {}
        for strategy in strategies:
            if strategy == RerankingStrategy.WEIGHTED_FUSION:
                strategy_results[strategy] = await self._weighted_fusion_reranking(results.copy(), request)
            elif strategy == RerankingStrategy.DIVERSITY_PROMOTION:
                strategy_results[strategy] = await self._diversity_promotion_reranking(results.copy(), request)
            elif strategy == RerankingStrategy.TEMPORAL_RELEVANCE:
                strategy_results[strategy] = await self._temporal_relevance_reranking(results.copy(), request)

        # 融合策略结果
        final_results = self._fuse_strategy_results(strategy_results, results)
        return final_results

    def _calculate_quality_score(self, result: RetrievalResult) -> float:
        """计算质量分数"""
        # 基于元数据计算质量分数
        quality_factors = {
            'evidence_level': result.metadata.get('evidence_level', 0.5),
            'safety_rating': 1.0 if result.metadata.get('safety_rating') == 'safe' else 0.5,
            'completeness': result.metadata.get('completeness', 0.5),
            'accuracy': result.metadata.get('accuracy', 0.5)
        }

        # 加权平均
        weights = [0.3, 0.3, 0.2, 0.2]
        score = sum(score * weight for score, weight in zip(quality_factors.values(), weights))
        return min(max(score, 0.0), 1.0)

    def _calculate_recency_score(self, result: RetrievalResult) -> float:
        """计算时效性分数"""
        if not self._config.enable_recency_boost:
            return 0.0

        # 获取文档时间戳
        timestamp = result.metadata.get('timestamp')
        if not timestamp:
            return 0.5  # 默认分数

        # 计算时间衰减
        import time
        current_time = time.time()
        age_days = (current_time - timestamp) / (24 * 3600)

        # 指数衰减
        decay_rate = math.log(2) / self._config.recency_half_life_days
        recency_score = math.exp(-decay_rate * age_days)

        return min(max(recency_score, 0.0), 1.0)

    def _calculate_popularity_score(self, result: RetrievalResult) -> float:
        """计算流行度分数"""
        # 基于访问次数、引用次数等计算
        views = result.metadata.get('views', 0)
        citations = result.metadata.get('citations', 0)
        likes = result.metadata.get('likes', 0)

        # 归一化处理
        max_views = 10000  # 假设的最大值
        max_citations = 1000
        max_likes = 1000

        normalized_views = min(views / max_views, 1.0)
        normalized_citations = min(citations / max_citations, 1.0)
        normalized_likes = min(likes / max_likes, 1.0)

        # 加权平均
        score = 0.4 * normalized_views + 0.4 * normalized_citations + 0.2 * normalized_likes
        return score

    def _calculate_authority_score(self, result: RetrievalResult) -> float:
        """计算权威性分数"""
        # 基于来源权威性、作者资质等计算
        source_authority = result.metadata.get('source_authority', 0.5)
        author_reputation = result.metadata.get('author_reputation', 0.5)
        peer_reviewed = 1.0 if result.metadata.get('peer_reviewed', False) else 0.0

        # 加权平均
        score = 0.4 * source_authority + 0.4 * author_reputation + 0.2 * peer_reviewed
        return score

    def _calculate_personalization_score(self, result: RetrievalResult, request: QueryRequest) -> float:
        """计算个性化分数"""
        user_id = request.metadata.get('user_id')
        if not user_id or user_id not in self._user_preferences:
            return 0.0

        user_prefs = self._user_preferences[user_id]
        score = 0.0

        # 基于用户历史偏好
        preferred_topics = user_prefs.get('preferred_topics', [])
        result_topics = result.metadata.get('topics', [])
        topic_overlap = len(set(preferred_topics) & set(result_topics))
        score += topic_overlap * 0.3

        # 基于用户交互历史
        interacted_sources = user_prefs.get('interacted_sources', [])
        if result.metadata.get('source') in interacted_sources:
            score += 0.4

        # 基于用户反馈
        positive_feedback = user_prefs.get('positive_feedback', set())
        if result.document_id in positive_feedback:
            score += 0.3

        return min(score, 1.0)

    async def _apply_diversity_boost(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """应用多样性提升"""
        if len(results) <= 1:
            return results

        diverse_results = [results[0]]  # 保留最高分的结果

        for candidate in results[1:]:
            # 检查与已选结果的相似度
            is_too_similar = False
            for selected in diverse_results:
                similarity = self._calculate_content_similarity(candidate, selected)
                if similarity > self._config.diversity_threshold:
                    is_too_similar = True
                    break

            if not is_too_similar or len(diverse_results) < self._config.max_similar_items:
                diverse_results.append(candidate)

        # 重新计算分数（应用多样性奖励）
        for i, result in enumerate(diverse_results):
            diversity_bonus = 1.0 - (i * 0.05)  # 排名越靠前，多样性奖励越高
            result.score *= diversity_bonus
            result.metadata['diversity_bonus'] = diversity_bonus

        # 最终排序
        diverse_results.sort(key=lambda x: x.score, reverse=True)
        return diverse_results

    def _calculate_content_similarity(self, result1: RetrievalResult, result2: RetrievalResult) -> float:
        """计算内容相似度"""
        # 简化的相似度计算（基于主题重叠）
        topics1 = set(result1.metadata.get('topics', []))
        topics2 = set(result2.metadata.get('topics', []))

        if not topics1 or not topics2:
            return 0.0

        intersection = topics1 & topics2
        union = topics1 | topics2

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _extract_ltr_features(self, result: RetrievalResult, request: QueryRequest) -> List[float]:
        """提取学习排序特征"""
        features = [
            result.score,  # 原始相关性分数
            self._calculate_quality_score(result),
            self._calculate_recency_score(result),
            self._calculate_popularity_score(result),
            self._calculate_authority_score(result),
            len(result.content),  # 内容长度
            len(result.metadata.get('topics', [])),  # 主题数量
        ]

        return features

    def _fuse_strategy_results(
        self,
        strategy_results: Dict[RerankingStrategy, List[RetrievalResult]],
        original_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """融合多个策略的结果"""
        # 使用Borda投票方法
        borda_scores = {}

        for strategy, results in strategy_results.items():
            weight = self._config.hybrid_weights.get(strategy.value, 1.0 / len(strategy_results))

            for i, result in enumerate(results):
                if result.document_id not in borda_scores:
                    borda_scores[result.document_id] = {
                        'result': result,
                        'score': 0.0,
                        'count': 0
                    }

                # Borda分数：N - rank
                borda_score = (len(results) - i) * weight
                borda_scores[result.document_id]['score'] += borda_score
                borda_scores[result.document_id]['count'] += 1

        # 按Borda分数排序
        fused_results = []
        for data in sorted(borda_scores.values(), key=lambda x: x['score'], reverse=True):
            result = data['result']
            result.score = data['score'] / data['count']  # 平均分数
            fused_results.append(result)

        return fused_results

    async def _apply_secondary_strategy(
        self,
        results: List[RetrievalResult],
        request: QueryRequest
    ) -> List[RetrievalResult]:
        """应用二次策略"""
        if not self._config.secondary_strategy:
            return results

        # 简化实现：对已排序的结果应用微调
        if self._config.secondary_strategy == RerankingStrategy.DIVERSITY_PROMOTION:
            return await self._apply_diversity_boost(results)
        elif self._config.secondary_strategy == RerankingStrategy.TEMPORAL_RELEVANCE:
            return await self._temporal_relevance_reranking(results, request)

        return results

    def _calculate_improvements(
        self,
        original: List[RetrievalResult],
        reranked: List[RetrievalResult]
    ) -> Dict[str, Any]:
        """计算改进指标"""
        # 计算多样性改进
        original_diversity = self._calculate_diversity(original)
        reranked_diversity = self._calculate_diversity(reranked)
        diversity_improvement = reranked_diversity - original_diversity

        # 计算质量改进
        original_quality = sum(r.score for r in original[:10]) / min(len(original), 10)
        reranked_quality = sum(r.score for r in reranked[:10]) / min(len(reranked), 10)
        quality_improvement = reranked_quality - original_quality

        return {
            'diversity_improvement': diversity_improvement,
            'quality_improvement': quality_improvement,
            'original_diversity': original_diversity,
            'reranked_diversity': reranked_diversity,
            'original_top10_quality': original_quality,
            'reranked_top10_quality': reranked_quality
        }

    def _calculate_diversity(self, results: List[RetrievalResult]) -> float:
        """计算结果多样性"""
        if len(results) <= 1:
            return 0.0

        total_similarity = 0.0
        comparisons = 0

        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                similarity = self._calculate_content_similarity(results[i], results[j])
                total_similarity += similarity
                comparisons += 1

        if comparisons == 0:
            return 0.0

        avg_similarity = total_similarity / comparisons
        diversity = 1.0 - avg_similarity
        return diversity

    def _update_metrics(
        self,
        strategy: RerankingStrategy,
        execution_time: float,
        improvements: Dict[str, Any]
    ) -> None:
        """更新性能指标"""
        # 更新策略使用统计
        self._metrics['strategy_usage'][strategy.value] += 1

        # 更新平均时间
        total_rerankings = self._metrics['total_rerankings']
        current_avg = self._metrics['average_reranking_time']
        self._metrics['average_reranking_time'] = (
            (current_avg * (total_rerankings - 1) + execution_time) / total_rerankings
        )

        # 更新改进统计
        if improvements.get('diversity_improvement', 0) > 0:
            self._metrics['diversity_improvements'] += 1
        if improvements.get('quality_improvement', 0) > 0:
            self._metrics['quality_improvements'] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """获取重排序指标"""
        base_metrics = super().get_metrics()
        return {
            **base_metrics,
            'total_rerankings': self._metrics['total_rerankings'],
            'average_reranking_time': self._metrics['average_reranking_time'],
            'diversity_improvements': self._metrics['diversity_improvements'],
            'quality_improvements': self._metrics['quality_improvements'],
            'strategy_usage': self._metrics['strategy_usage'],
            'enabled_strategies': [
                strategy.value for strategy in RerankingStrategy
                if self._metrics['strategy_usage'][strategy.value] > 0
            ]
        }


# 导出
__all__ = [
    'Reranker',
    'RerankingConfig',
    'RerankingResult',
    'RerankingStrategy',
    'ScoringDimension'
]