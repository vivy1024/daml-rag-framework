#!/usr/bin/env python3
"""
daml-rag-framework 反馈处理模块
实现用户反馈收集、分析和学习机制
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """反馈类型枚举"""
    THUMBS_UP = "thumbs_up"  # 点赞
    THUMBS_DOWN = "thumbs_down"  # 点踩
    CORRECTION = "correction"  # 纠正
    IMPROVEMENT = "improvement"  # 改进建议
    DETAILED_RATING = "detailed_rating"  # 详细评分


class FeedbackSource(Enum):
    """反馈来源枚举"""
    USER_EXPLICIT = "user_explicit"  # 用户明确反馈
    USER_IMPLICIT = "user_implicit"  # 用户隐式反馈
    SYSTEM_AUTO = "system_auto"  # 系统自动评估


@dataclass
class FeedbackData:
    """反馈数据"""
    query_id: str
    response_id: str
    feedback_type: FeedbackType
    feedback_source: FeedbackSource
    rating: Optional[float] = None  # 0-1评分
    comment: Optional[str] = None
    corrected_answer: Optional[str] = None
    improvement_suggestions: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class FeedbackAnalysis:
    """反馈分析结果"""
    total_feedbacks: int
    average_rating: float
    feedback_distribution: Dict[FeedbackType, int]
    common_issues: List[Tuple[str, int]]  # (问题描述, 出现次数)
    improvement_areas: List[str]
    performance_trend: str  # "improving", "declining", "stable"
    confidence_score: float


class FeedbackProcessor(ABC):
    """反馈处理器抽象基类"""

    @abstractmethod
    async def collect_feedback(self, feedback: FeedbackData) -> bool:
        """收集反馈"""
        pass

    @abstractmethod
    async def analyze_feedback(
        self,
        query_id: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> FeedbackAnalysis:
        """分析反馈"""
        pass

    @abstractmethod
    async def get_improvement_suggestions(self) -> List[str]:
        """获取改进建议"""
        pass

    @abstractmethod
    async def update_model_parameters(self) -> Dict[str, Any]:
        """更新模型参数"""
        pass


class SimpleFeedbackProcessor(FeedbackProcessor):
    """简单反馈处理器实现"""

    def __init__(self, storage_backend: Optional[Dict] = None):
        self.feedbacks: List[FeedbackData] = []
        self.storage_backend = storage_backend or {}
        self.analysis_cache: Dict[str, FeedbackAnalysis] = {}
        self._cache_ttl = 300  # 5分钟缓存

    async def collect_feedback(self, feedback: FeedbackData) -> bool:
        """收集反馈"""
        try:
            # 验证反馈数据
            if not self._validate_feedback(feedback):
                logger.warning(f"Invalid feedback data: {feedback}")
                return False

            # 存储反馈
            self.feedbacks.append(feedback)

            # 清除相关缓存
            self._clear_analysis_cache()

            # 异步存储到后端（如果配置了）
            if self.storage_backend:
                await self._persist_feedback(feedback)

            logger.info(f"Collected feedback: {feedback.feedback_type.value} for query {feedback.query_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to collect feedback: {e}")
            return False

    async def analyze_feedback(
        self,
        query_id: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> FeedbackAnalysis:
        """分析反馈"""
        # 检查缓存
        cache_key = self._generate_cache_key(query_id, time_range)
        cached_analysis = self._get_cached_analysis(cache_key)
        if cached_analysis:
            return cached_analysis

        # 过滤反馈数据
        filtered_feedbacks = self._filter_feedbacks(query_id, time_range)

        if not filtered_feedbacks:
            return FeedbackAnalysis(
                total_feedbacks=0,
                average_rating=0.0,
                feedback_distribution={},
                common_issues=[],
                improvement_areas=[],
                performance_trend="stable",
                confidence_score=0.0
            )

        # 计算统计指标
        total_feedbacks = len(filtered_feedbacks)
        ratings = [f.rating for f in filtered_feedbacks if f.rating is not None]
        average_rating = np.mean(ratings) if ratings else 0.0

        # 反馈分布
        feedback_distribution = self._calculate_feedback_distribution(filtered_feedbacks)

        # 常见问题分析
        common_issues = self._extract_common_issues(filtered_feedbacks)

        # 改进领域
        improvement_areas = self._identify_improvement_areas(filtered_feedbacks, common_issues)

        # 性能趋势
        performance_trend = self._analyze_performance_trend(filtered_feedbacks)

        # 置信度分数
        confidence_score = self._calculate_confidence_score(filtered_feedbacks)

        analysis = FeedbackAnalysis(
            total_feedbacks=total_feedbacks,
            average_rating=average_rating,
            feedback_distribution=feedback_distribution,
            common_issues=common_issues,
            improvement_areas=improvement_areas,
            performance_trend=performance_trend,
            confidence_score=confidence_score
        )

        # 缓存分析结果
        self._cache_analysis(cache_key, analysis)

        return analysis

    async def get_improvement_suggestions(self) -> List[str]:
        """获取改进建议"""
        # 获取最近的分析结果
        recent_analysis = await self.analyze_feedback()

        suggestions = []

        # 基于评分数值
        if recent_analysis.average_rating < 0.6:
            suggestions.append("整体评分偏低，需要改进回答质量和准确性")

        # 基于反馈分布
        negative_count = recent_analysis.feedback_distribution.get(FeedbackType.THUMBS_DOWN, 0)
        total_count = recent_analysis.total_feedbacks
        if total_count > 0 and negative_count / total_count > 0.3:
            suggestions.append("负面反馈比例较高，需要关注用户不满意的原因")

        # 基于常见问题
        for issue, count in recent_analysis.common_issues[:3]:  # 取前3个最常见问题
            if count > 2:  # 至少出现3次
                suggestions.append(f"频繁出现的问题：{issue}（出现{count}次）")

        # 基于改进领域
        if recent_analysis.improvement_areas:
            suggestions.extend([
                f"需要改进的领域：{area}"
                for area in recent_analysis.improvement_areas[:3]
            ])

        # 基于性能趋势
        if recent_analysis.performance_trend == "declining":
            suggestions.append("性能呈下降趋势，需要立即采取措施改进")
        elif recent_analysis.performance_trend == "stable":
            suggestions.append("性能稳定，可以尝试优化特定领域的表现")

        # 基于置信度
        if recent_analysis.confidence_score < 0.5:
            suggestions.append("反馈数据不足，需要收集更多用户反馈")

        return suggestions

    async def update_model_parameters(self) -> Dict[str, Any]:
        """更新模型参数"""
        analysis = await self.analyze_feedback()

        updates = {}

        # 基于评分调整模型温度
        if analysis.average_rating < 0.5:
            # 评分低，降低创造性，提高准确性
            updates["temperature_adjustment"] = -0.1
        elif analysis.average_rating > 0.8:
            # 评分高，可以适当增加创造性
            updates["temperature_adjustment"] = 0.05

        # 基于反馈分布调整
        total_negative = analysis.feedback_distribution.get(FeedbackType.THUMBS_DOWN, 0)
        total_positive = analysis.feedback_distribution.get(FeedbackType.THUMBS_UP, 0)
        total_feedbacks = analysis.total_feedbacks

        if total_feedbacks > 0:
            negative_ratio = total_negative / total_feedbacks
            if negative_ratio > 0.4:
                # 负面反馈多，增加验证步骤
                updates["enable_double_check"] = True
                updates["retrieval_top_k_increase"] = 2

        # 基于常见问题调整
        if "不准确" in str(analysis.common_issues):
            updates["increase_retrieval_precision"] = True
            updates["similarity_threshold_adjustment"] = 0.1

        if "不完整" in str(analysis.common_issues):
            updates["max_tokens_increase"] = 200
            updates["enable_follow_up_questions"] = True

        # 记录更新历史
        update_log = {
            "timestamp": datetime.now().isoformat(),
            "analysis_summary": asdict(analysis),
            "updates": updates
        }

        if not hasattr(self, 'update_history'):
            self.update_history = []
        self.update_history.append(update_log)

        logger.info(f"Model parameters updated: {updates}")
        return updates

    def _validate_feedback(self, feedback: FeedbackData) -> bool:
        """验证反馈数据"""
        if not feedback.query_id or not feedback.response_id:
            return False

        if not isinstance(feedback.feedback_type, FeedbackType):
            return False

        if not isinstance(feedback.feedback_source, FeedbackSource):
            return False

        if feedback.rating is not None and (feedback.rating < 0 or feedback.rating > 1):
            return False

        return True

    async def _persist_feedback(self, feedback: FeedbackData):
        """持久化反馈到后端存储"""
        # 这里可以实现到数据库、文件等的持久化
        feedback_dict = asdict(feedback)
        feedback_dict['timestamp'] = feedback.timestamp.isoformat()
        feedback_dict['feedback_type'] = feedback.feedback_type.value
        feedback_dict['feedback_source'] = feedback.feedback_source.value

        # 简单存储到内存字典（实际应用中应该是数据库）
        if 'feedbacks' not in self.storage_backend:
            self.storage_backend['feedbacks'] = []
        self.storage_backend['feedbacks'].append(feedback_dict)

    def _filter_feedbacks(
        self,
        query_id: Optional[str],
        time_range: Optional[Tuple[datetime, datetime]]
    ) -> List[FeedbackData]:
        """过滤反馈数据"""
        filtered = self.feedbacks

        if query_id:
            filtered = [f for f in filtered if f.query_id == query_id]

        if time_range:
            start_time, end_time = time_range
            filtered = [
                f for f in filtered
                if start_time <= f.timestamp <= end_time
            ]

        return filtered

    def _calculate_feedback_distribution(self, feedbacks: List[FeedbackData]) -> Dict[FeedbackType, int]:
        """计算反馈分布"""
        distribution = {feedback_type: 0 for feedback_type in FeedbackType}

        for feedback in feedbacks:
            distribution[feedback.feedback_type] += 1

        return distribution

    def _extract_common_issues(self, feedbacks: List[FeedbackData]) -> List[Tuple[str, int]]:
        """提取常见问题"""
        issue_counts = {}

        for feedback in feedbacks:
            # 从评论中提取问题关键词
            if feedback.comment:
                comment_lower = feedback.comment.lower()

                # 简单的关键词匹配
                issue_keywords = {
                    "不准确": ["不准确", "错误", "不对", "wrong", "inaccurate"],
                    "不完整": ["不完整", "缺少", "不够", "incomplete", "missing"],
                    "不清楚": ["不清楚", "模糊", "难懂", "unclear", "confusing"],
                    "太简单": ["太简单", "不够详细", "简单", "too simple", "basic"],
                    "太复杂": ["太复杂", "难懂", "复杂", "too complex", "complicated"]
                }

                for issue, keywords in issue_keywords.items():
                    if any(keyword in comment_lower for keyword in keywords):
                        issue_counts[issue] = issue_counts.get(issue, 0) + 1

            # 从改进建议中提取问题
            if feedback.improvement_suggestions:
                for suggestion in feedback.improvement_suggestions:
                    suggestion_lower = suggestion.lower()
                    if "准确" in suggestion_lower:
                        issue_counts["准确性"] = issue_counts.get("准确性", 0) + 1
                    elif "完整" in suggestion_lower:
                        issue_counts["完整性"] = issue_counts.get("完整性", 0) + 1
                    elif "清晰" in suggestion_lower:
                        issue_counts["清晰度"] = issue_counts.get("清晰度", 0) + 1

        # 按出现次数排序
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_issues[:5]  # 返回前5个最常见问题

    def _identify_improvement_areas(
        self,
        feedbacks: List[FeedbackData],
        common_issues: List[Tuple[str, int]]
    ) -> List[str]:
        """识别改进领域"""
        areas = []

        # 基于常见问题确定改进领域
        issue_to_area = {
            "不准确": "知识准确性",
            "不完整": "回答完整性",
            "不清楚": "表达清晰度",
            "太简单": "内容深度",
            "太复杂": "内容简化",
            "准确性": "知识准确性",
            "完整性": "回答完整性",
            "清晰度": "表达清晰度"
        }

        for issue, count in common_issues:
            if count >= 2:  # 至少出现2次
                area = issue_to_area.get(issue, issue)
                if area not in areas:
                    areas.append(area)

        # 基于评分确定改进领域
        low_rating_feedbacks = [f for f in feedbacks if f.rating and f.rating < 0.5]
        if low_rating_feedbacks:
            if len(low_rating_feedbacks) / len(feedbacks) > 0.3:
                areas.append("整体质量")

        return areas

    def _analyze_performance_trend(self, feedbacks: List[FeedbackData]) -> str:
        """分析性能趋势"""
        if len(feedbacks) < 5:
            return "stable"

        # 按时间排序
        sorted_feedbacks = sorted(feedbacks, key=lambda f: f.timestamp)

        # 分成两半比较
        mid_point = len(sorted_feedbacks) // 2
        first_half = sorted_feedbacks[:mid_point]
        second_half = sorted_feedbacks[mid_point:]

        # 计算平均评分
        first_avg = np.mean([f.rating for f in first_half if f.rating])
        second_avg = np.mean([f.rating for f in second_half if f.rating])

        # 计算正面反馈比例
        first_positive = sum(1 for f in first_half if f.feedback_type == FeedbackType.THUMBS_UP)
        second_positive = sum(1 for f in second_half if f.feedback_type == FeedbackType.THUMBS_UP)

        first_positive_ratio = first_positive / len(first_half) if first_half else 0
        second_positive_ratio = second_positive / len(second_half) if second_half else 0

        # 判断趋势
        if second_avg > first_avg + 0.1 or second_positive_ratio > first_positive_ratio + 0.1:
            return "improving"
        elif second_avg < first_avg - 0.1 or second_positive_ratio < first_positive_ratio - 0.1:
            return "declining"
        else:
            return "stable"

    def _calculate_confidence_score(self, feedbacks: List[FeedbackData]) -> float:
        """计算置信度分数"""
        if not feedbacks:
            return 0.0

        # 基于反馈数量
        count_score = min(len(feedbacks) / 10, 1.0)  # 10个反馈给满分

        # 基于反馈多样性
        feedback_types = set(f.feedback_type for f in feedbacks)
        diversity_score = min(len(feedback_types) / len(FeedbackType), 1.0)

        # 基于时间分布（最近7天的反馈权重更高）
        now = datetime.now()
        recent_feedbacks = [f for f in feedbacks if (now - f.timestamp).days <= 7]
        recency_score = min(len(recent_feedbacks) / len(feedbacks), 1.0) if feedbacks else 0

        # 综合计算
        confidence_score = (count_score * 0.4 + diversity_score * 0.3 + recency_score * 0.3)

        return round(confidence_score, 2)

    def _generate_cache_key(
        self,
        query_id: Optional[str],
        time_range: Optional[Tuple[datetime, datetime]]
    ) -> str:
        """生成缓存键"""
        import hashlib

        key_data = {
            "query_id": query_id,
            "time_range": time_range,
            "feedback_count": len(self.feedbacks)
        }

        if time_range:
            key_data["time_range"] = (time_range[0].isoformat(), time_range[1].isoformat())

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached_analysis(self, cache_key: str) -> Optional[FeedbackAnalysis]:
        """获取缓存的分析结果"""
        if cache_key in self.analysis_cache:
            cached_data, timestamp = self.analysis_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached_data
            else:
                del self.analysis_cache[cache_key]
        return None

    def _cache_analysis(self, cache_key: str, analysis: FeedbackAnalysis):
        """缓存分析结果"""
        self.analysis_cache[cache_key] = (analysis, time.time())

    def _clear_analysis_cache(self):
        """清除分析缓存"""
        self.analysis_cache.clear()

    def get_feedback_stats(self) -> Dict[str, Any]:
        """获取反馈统计信息"""
        if not self.feedbacks:
            return {
                "total_feedbacks": 0,
                "average_rating": 0.0,
                "feedback_types": {},
                "recent_activity": False
            }

        total_feedbacks = len(self.feedbacks)
        ratings = [f.rating for f in self.feedbacks if f.rating is not None]
        average_rating = np.mean(ratings) if ratings else 0.0

        # 反馈类型统计
        feedback_types = {}
        for feedback in self.feedbacks:
            type_name = feedback.feedback_type.value
            feedback_types[type_name] = feedback_types.get(type_name, 0) + 1

        # 最近活动（24小时内有反馈）
        now = datetime.now()
        recent_feedbacks = [
            f for f in self.feedbacks
            if (now - f.timestamp).total_seconds() < 24 * 3600
        ]
        recent_activity = len(recent_feedbacks) > 0

        return {
            "total_feedbacks": total_feedbacks,
            "average_rating": round(average_rating, 2),
            "feedback_types": feedback_types,
            "recent_activity": recent_activity,
            "cache_size": len(self.analysis_cache)
        }