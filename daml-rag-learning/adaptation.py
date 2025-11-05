#!/usr/bin/env python3
"""
DAML-RAG Framework 自适应机制模块
实现基于经验和反馈的动态调整机制
"""

import asyncio
import json
import logging
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta

from memory import MemoryManager, Experience
from feedback import FeedbackProcessor, FeedbackAnalysis, FeedbackData
from model_provider import ModelProvider, ModelManager

logger = logging.getLogger(__name__)


class AdaptationStrategy(Enum):
    """适应策略枚举"""
    CONSERVATIVE = "conservative"  # 保守策略，小幅度调整
    MODERATE = "moderate"  # 适中策略，中等幅度调整
    AGGRESSIVE = "aggressive"  # 激进策略，大幅度调整


class AdaptationTarget(Enum):
    """适应目标枚举"""
    RETRIEVAL = "retrieval"  # 检索参数
    GENERATION = "generation"  # 生成参数
    ORCHESTRATION = "orchestration"  # 编排参数
    FILTERING = "filtering"  # 过滤参数


@dataclass
class AdaptationConfig:
    """适应配置"""
    strategy: AdaptationStrategy = AdaptationStrategy.MODERATE
    adaptation_interval: int = 100  # 每100次查询进行一次适应
    confidence_threshold: float = 0.7  # 置信度阈值
    improvement_threshold: float = 0.05  # 改进阈值
    max_adjustment_ratio: float = 0.3  # 最大调整比例
    enable_auto_adaptation: bool = True
    adaptation_history_size: int = 50


@dataclass
class AdaptationAction:
    """适应动作"""
    target: AdaptationTarget
    parameter: str
    old_value: Union[float, int, str, bool]
    new_value: Union[float, int, str, bool]
    adjustment_ratio: float
    confidence: float
    reason: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class AdaptationResult:
    """适应结果"""
    success: bool
    actions_taken: List[AdaptationAction]
    performance_change: float
    confidence_score: float
    adaptation_summary: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class AdaptiveLearner(ABC):
    """自适应学习器抽象基类"""

    @abstractmethod
    async def analyze_performance(self) -> Dict[str, float]:
        """分析当前性能"""
        pass

    @abstractmethod
    async def identify_adaptation_opportunities(
        self,
        performance_metrics: Dict[str, float]
    ) -> List[AdaptationAction]:
        """识别适应机会"""
        pass

    @abstractmethod
    async def apply_adaptations(self, actions: List[AdaptationAction]) -> AdaptationResult:
        """应用适应措施"""
        pass

    @abstractmethod
    async def evaluate_adaptation_impact(self, result: AdaptationResult) -> float:
        """评估适应影响"""
        pass


class ExperienceBasedLearner(AdaptiveLearner):
    """基于经验的自适应学习器"""

    def __init__(
        self,
        memory_manager: MemoryManager,
        feedback_processor: FeedbackProcessor,
        model_manager: ModelManager,
        config: AdaptationConfig
    ):
        self.memory_manager = memory_manager
        self.feedback_processor = feedback_processor
        self.model_manager = model_manager
        self.config = config

        # 性能指标历史
        self.performance_history: List[Dict[str, float]] = []
        self.adaptation_history: List[AdaptationResult] = []

        # 参数基准值
        self.parameter_baselines: Dict[str, Any] = {}

        # 统计信息
        self.stats = {
            "total_adaptations": 0,
            "successful_adaptations": 0,
            "failed_adaptations": 0,
            "total_performance_change": 0.0,
            "average_confidence": 0.0
        }

        self._initialized = False

    async def initialize(self) -> None:
        """初始化学习器"""
        # 记录参数基准值
        await self._record_parameter_baselines()

        # 初始化性能历史
        initial_performance = await self.analyze_performance()
        self.performance_history.append({
            "timestamp": datetime.now(),
            "metrics": initial_performance
        })

        self._initialized = True
        logger.info("Experience-based adaptive learner initialized")

    async def analyze_performance(self) -> Dict[str, float]:
        """分析当前性能"""
        metrics = {}

        # 从记忆管理器获取经验统计
        if hasattr(self.memory_manager, 'get_statistics'):
            memory_stats = await self.memory_manager.get_statistics()
            metrics.update({
                "experience_retrieval_success": memory_stats.get("retrieval_success_rate", 0.0),
                "experience_similarity_avg": memory_stats.get("average_similarity", 0.0),
                "experience_utilization": memory_stats.get("experience_count", 0) / max(memory_stats.get("max_experiences", 1), 1)
            })

        # 从反馈处理器获取反馈分析
        feedback_analysis = await self.feedback_processor.analyze_feedback()
        metrics.update({
            "user_satisfaction": feedback_analysis.average_rating,
            "feedback_positive_ratio": feedback_analysis.feedback_distribution.get("thumbs_up", 0) / max(feedback_analysis.total_feedbacks, 1),
            "feedback_confidence": feedback_analysis.confidence_score
        })

        # 从模型管理器获取性能统计
        if hasattr(self.model_manager, 'get_stats'):
            model_stats = self.model_manager.get_stats()
            metrics.update({
                "model_cost_efficiency": 1.0 - (model_stats.get("avg_cost_per_request", 0.0) / 0.1),  # 假设目标成本0.1
                "model_response_quality": 1.0 - model_stats.get("student_ratio", 0.0),  # 教师模型比例越高质量越好
                "cache_hit_ratio": model_stats.get("cache_hit_ratio", 0.0)
            })

        # 计算综合性能分数
        metrics["overall_performance"] = self._calculate_overall_performance(metrics)

        return metrics

    async def identify_adaptation_opportunities(
        self,
        performance_metrics: Dict[str, float]
    ) -> List[AdaptationAction]:
        """识别适应机会"""
        if not self._initialized:
            await self.initialize()

        actions = []

        # 检查是否需要适应
        if not await self._should_adapt(performance_metrics):
            return actions

        # 分析各个指标的适应机会
        actions.extend(await self._analyze_retrieval_adaptations(performance_metrics))
        actions.extend(await self._analyze_generation_adaptations(performance_metrics))
        actions.extend(await self._analyze_orchestration_adaptations(performance_metrics))
        actions.extend(await self._analyze_filtering_adaptations(performance_metrics))

        # 根据策略过滤和排序动作
        actions = self._filter_and_rank_actions(actions)

        return actions

    async def apply_adaptations(self, actions: List[AdaptationAction]) -> AdaptationResult:
        """应用适应措施"""
        if not actions:
            return AdaptationResult(
                success=True,
                actions_taken=[],
                performance_change=0.0,
                confidence_score=0.0,
                adaptation_summary="No adaptations needed"
            )

        start_time = time.time()
        successful_actions = []
        failed_actions = []

        logger.info(f"Applying {len(actions)} adaptations")

        for action in actions:
            try:
                success = await self._apply_single_adaptation(action)
                if success:
                    successful_actions.append(action)
                    logger.info(f"Successfully applied adaptation: {action.parameter} = {action.new_value}")
                else:
                    failed_actions.append(action)
                    logger.warning(f"Failed to apply adaptation: {action.parameter}")

            except Exception as e:
                logger.error(f"Error applying adaptation {action.parameter}: {e}")
                failed_actions.append(action)

        # 计算性能变化（预估值）
        performance_change = self._estimate_performance_change(successful_actions)
        confidence_score = np.mean([action.confidence for action in successful_actions]) if successful_actions else 0.0

        # 更新统计信息
        self.stats["total_adaptations"] += len(actions)
        self.stats["successful_adaptations"] += len(successful_actions)
        self.stats["failed_adaptations"] += len(failed_actions)
        self.stats["total_performance_change"] += performance_change
        self.stats["average_confidence"] = (
            (self.stats["average_confidence"] * (self.stats["total_adaptations"] - len(actions)) + confidence_score * len(actions))
            / self.stats["total_adaptations"]
        )

        # 记录适应结果
        result = AdaptationResult(
            success=len(successful_actions) > 0,
            actions_taken=successful_actions,
            performance_change=performance_change,
            confidence_score=confidence_score,
            adaptation_summary=self._generate_adaptation_summary(successful_actions, failed_actions),
            timestamp=datetime.now()
        )

        # 保存到历史记录
        self.adaptation_history.append(result)
        if len(self.adaptation_history) > self.config.adaptation_history_size:
            self.adaptation_history.pop(0)

        execution_time = time.time() - start_time
        logger.info(f"Adaptation completed in {execution_time:.2f}s: {result.adaptation_summary}")

        return result

    async def evaluate_adaptation_impact(self, result: AdaptationResult) -> float:
        """评估适应影响"""
        if not result.actions_taken:
            return 0.0

        # 等待一段时间让适应措施生效
        await asyncio.sleep(5)

        # 分析新的性能
        new_performance = await self.analyze_performance()

        # 计算性能变化
        if self.performance_history:
            previous_performance = self.performance_history[-1]["metrics"]["overall_performance"]
            performance_change = new_performance["overall_performance"] - previous_performance
        else:
            performance_change = 0.0

        # 记录新的性能数据
        self.performance_history.append({
            "timestamp": datetime.now(),
            "metrics": new_performance
        })

        # 限制历史记录大小
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)

        logger.info(f"Adaptation impact evaluated: {performance_change:.3f} change in overall performance")

        return performance_change

    async def _record_parameter_baselines(self):
        """记录参数基准值"""
        # 这里应该从实际的配置系统获取当前参数值
        # 暂时使用默认值
        self.parameter_baselines = {
            # 检索参数
            "retrieval_top_k": 5,
            "retrieval_similarity_threshold": 0.6,
            "retrieval_cache_ttl": 300,

            # 生成参数
            "generation_temperature": 0.7,
            "generation_max_tokens": 2000,
            "generation_model_complexity_threshold": 0.7,

            # 编排参数
            "orchestration_max_parallel_tasks": 10,
            "orchestration_timeout_seconds": 30,

            # 过滤参数
            "filter_quality_threshold": 0.5,
            "filter_enable_kg": True,
            "filter_enable_rules": True
        }

    async def _should_adapt(self, performance_metrics: Dict[str, float]) -> bool:
        """判断是否应该进行适应"""
        if not self.config.enable_auto_adaptation:
            return False

        # 检查置信度
        confidence = performance_metrics.get("feedback_confidence", 0.0)
        if confidence < self.config.confidence_threshold:
            return False

        # 检查性能是否低于阈值
        overall_performance = performance_metrics.get("overall_performance", 0.0)
        if overall_performance >= 0.8:  # 性能良好，不需要适应
            return False

        # 检查距离上次适应的时间
        if self.adaptation_history:
            last_adaptation = self.adaptation_history[-1].timestamp
            time_since_last = datetime.now() - last_adaptation
            if time_since_last < timedelta(minutes=30):  # 至少间隔30分钟
                return False

        return True

    async def _analyze_retrieval_adaptations(self, metrics: Dict[str, float]) -> List[AdaptationAction]:
        """分析检索适应机会"""
        actions = []

        # 经验检索成功率低
        if metrics.get("experience_retrieval_success", 0.0) < 0.6:
            similarity_threshold = self.parameter_baselines.get("retrieval_similarity_threshold", 0.6)
            new_threshold = max(0.3, similarity_threshold - 0.1)

            actions.append(AdaptationAction(
                target=AdaptationTarget.RETRIEVAL,
                parameter="retrieval_similarity_threshold",
                old_value=similarity_threshold,
                new_value=new_threshold,
                adjustment_ratio=abs(new_threshold - similarity_threshold) / similarity_threshold,
                confidence=0.8,
                reason="Experience retrieval success rate is low, lowering similarity threshold"
            ))

        # 经验利用率低
        if metrics.get("experience_utilization", 0.0) < 0.3:
            top_k = self.parameter_baselines.get("retrieval_top_k", 5)
            new_top_k = min(10, top_k + 2)

            actions.append(AdaptationAction(
                target=AdaptationTarget.RETRIEVAL,
                parameter="retrieval_top_k",
                old_value=top_k,
                new_value=new_top_k,
                adjustment_ratio=2 / top_k,
                confidence=0.7,
                reason="Experience utilization is low, increasing retrieval count"
            ))

        return actions

    async def _analyze_generation_adaptations(self, metrics: Dict[str, float]) -> List[AdaptationAction]:
        """分析生成适应机会"""
        actions = []

        # 用户满意度低
        if metrics.get("user_satisfaction", 0.0) < 0.6:
            temperature = self.parameter_baselines.get("generation_temperature", 0.7)
            # 降低创造性，提高准确性
            new_temperature = max(0.3, temperature - 0.1)

            actions.append(AdaptationAction(
                target=AdaptationTarget.GENERATION,
                parameter="generation_temperature",
                old_value=temperature,
                new_value=new_temperature,
                adjustment_ratio=abs(new_temperature - temperature) / temperature,
                confidence=0.8,
                reason="User satisfaction is low, reducing creativity for accuracy"
            ))

        # 成本效率低
        if metrics.get("model_cost_efficiency", 0.0) < 0.5:
            complexity_threshold = self.parameter_baselines.get("generation_model_complexity_threshold", 0.7)
            # 降低复杂度阈值，更多使用学生模型
            new_threshold = max(0.4, complexity_threshold - 0.1)

            actions.append(AdaptationAction(
                target=AdaptationTarget.GENERATION,
                parameter="generation_model_complexity_threshold",
                old_value=complexity_threshold,
                new_value=new_threshold,
                adjustment_ratio=abs(new_threshold - complexity_threshold) / complexity_threshold,
                confidence=0.7,
                reason="Cost efficiency is low, favoring student model more often"
            ))

        return actions

    async def _analyze_orchestration_adaptations(self, metrics: Dict[str, float]) -> List[AdaptationAction]:
        """分析编排适应机会"""
        actions = []

        # 整体性能低，可能需要更多并行处理
        if metrics.get("overall_performance", 0.0) < 0.5:
            max_parallel = self.parameter_baselines.get("orchestration_max_parallel_tasks", 10)
            new_max_parallel = min(20, max_parallel + 3)

            actions.append(AdaptationAction(
                target=AdaptationTarget.ORCHESTRATION,
                parameter="orchestration_max_parallel_tasks",
                old_value=max_parallel,
                new_value=new_max_parallel,
                adjustment_ratio=3 / max_parallel,
                confidence=0.6,
                reason="Overall performance is low, increasing parallel processing capacity"
            ))

        return actions

    async def _analyze_filtering_adaptations(self, metrics: Dict[str, float]) -> List[AdaptationAction]:
        """分析过滤适应机会"""
        actions = []

        # 经验相似度低，可能需要调整过滤策略
        if metrics.get("experience_similarity_avg", 0.0) < 0.5:
            quality_threshold = self.parameter_baselines.get("filter_quality_threshold", 0.5)
            new_threshold = max(0.3, quality_threshold - 0.1)

            actions.append(AdaptationAction(
                target=AdaptationTarget.FILTERING,
                parameter="filter_quality_threshold",
                old_value=quality_threshold,
                new_value=new_threshold,
                adjustment_ratio=abs(new_threshold - quality_threshold) / quality_threshold,
                confidence=0.6,
                reason="Experience similarity is low, relaxing quality filtering"
            ))

        return actions

    def _filter_and_rank_actions(self, actions: List[AdaptationAction]) -> List[AdaptationAction]:
        """根据策略过滤和排序动作"""
        if not actions:
            return actions

        # 根据策略调整调整幅度
        strategy_multiplier = {
            AdaptationStrategy.CONSERVATIVE: 0.5,
            AdaptationStrategy.MODERATE: 1.0,
            AdaptationStrategy.AGGRESSIVE: 1.5
        }

        multiplier = strategy_multiplier[self.config.strategy]

        # 调整动作的调整幅度
        for action in actions:
            if isinstance(action.old_value, (int, float)):
                adjustment = (action.new_value - action.old_value) * multiplier
                action.new_value = action.old_value + adjustment
                action.adjustment_ratio = abs(adjustment) / abs(action.old_value) if action.old_value != 0 else 0

        # 过滤超出最大调整比例的动作
        filtered_actions = [
            action for action in actions
            if action.adjustment_ratio <= self.config.max_adjustment_ratio
        ]

        # 按置信度和调整比例排序
        filtered_actions.sort(
            key=lambda a: (a.confidence, a.adjustment_ratio),
            reverse=True
        )

        # 限制动作数量
        max_actions = 3 if self.config.strategy == AdaptationStrategy.CONSERVATIVE else 5
        return filtered_actions[:max_actions]

    async def _apply_single_adaptation(self, action: AdaptationAction) -> bool:
        """应用单个适应措施"""
        # 这里应该实际修改系统参数
        # 暂时只是记录日志
        logger.info(f"Applying adaptation: {action.parameter} from {action.old_value} to {action.new_value}")

        # 模拟应用延迟
        await asyncio.sleep(0.1)

        return True

    def _estimate_performance_change(self, actions: List[AdaptationAction]) -> float:
        """估算性能变化"""
        if not actions:
            return 0.0

        # 简单的线性估算
        total_change = 0.0
        for action in actions:
            # 根据调整比例和置信度估算影响
            impact = action.adjustment_ratio * action.confidence
            total_change += impact

        return min(total_change / len(actions), 0.2)  # 限制最大变化

    def _generate_adaptation_summary(
        self,
        successful_actions: List[AdaptationAction],
        failed_actions: List[AdaptationAction]
    ) -> str:
        """生成适应总结"""
        if not successful_actions and not failed_actions:
            return "No adaptations were attempted"

        summary_parts = []

        if successful_actions:
            targets = [action.target.value for action in successful_actions]
            target_counts = {target: targets.count(target) for target in set(targets)}
            successful_summary = ", ".join([f"{count} {target}" for target, count in target_counts.items()])
            summary_parts.append(f"Applied {len(successful_actions)} successful adaptations ({successful_summary})")

        if failed_actions:
            summary_parts.append(f"Failed {len(failed_actions)} adaptations")

        return "; ".join(summary_parts)

    def _calculate_overall_performance(self, metrics: Dict[str, float]) -> float:
        """计算综合性能分数"""
        # 权重配置
        weights = {
            "experience_retrieval_success": 0.2,
            "user_satisfaction": 0.3,
            "model_cost_efficiency": 0.2,
            "model_response_quality": 0.15,
            "cache_hit_ratio": 0.1,
            "feedback_confidence": 0.05
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in metrics:
                weighted_sum += metrics[metric] * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def get_adaptation_stats(self) -> Dict[str, Any]:
        """获取适应统计信息"""
        if not self.adaptation_history:
            return {
                "total_adaptations": 0,
                "success_rate": 0.0,
                "average_performance_change": 0.0,
                "average_confidence": 0.0,
                "last_adaptation": None
            }

        successful_adaptations = sum(1 for result in self.adaptation_history if result.success)
        performance_changes = [result.performance_change for result in self.adaptation_history]
        confidences = [result.confidence_score for result in self.adaptation_history]

        return {
            "total_adaptations": len(self.adaptation_history),
            "successful_adaptations": successful_adaptations,
            "success_rate": successful_adaptations / len(self.adaptation_history),
            "average_performance_change": np.mean(performance_changes) if performance_changes else 0.0,
            "average_confidence": np.mean(confidences) if confidences else 0.0,
            "last_adaptation": self.adaptation_history[-1].timestamp.isoformat(),
            "recent_trend": self._calculate_recent_trend()
        }

    def _calculate_recent_trend(self) -> str:
        """计算最近的趋势"""
        if len(self.adaptation_history) < 3:
            return "insufficient_data"

        recent_results = self.adaptation_history[-3:]
        performance_changes = [result.performance_change for result in recent_results]

        avg_change = np.mean(performance_changes)
        if avg_change > 0.05:
            return "improving"
        elif avg_change < -0.05:
            return "declining"
        else:
            return "stable"