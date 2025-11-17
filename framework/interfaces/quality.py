# -*- coding: utf-8 -*-
"""
DAML-RAG框架质量保证接口定义 v2.0

定义反幻觉验证、质量监控、内容安全等质量保证相关接口。

版本：v2.0.0
更新日期：2025-11-17
设计原则：多层验证、智能监控、安全可控
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from .base import IComponent, IConfigurable, IMonitorable


@dataclass
class QualityCheckResult:
    """质量检查结果"""
    check_name: str
    passed: bool
    score: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class QualityReport:
    """质量报告"""
    content_id: str
    content_type: str
    overall_score: float  # 0.0 - 1.0
    passed: bool
    checks: List[QualityCheckResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    version: str = "2.0.0"


class QualityDimension(Enum):
    """质量维度"""
    RELEVANCE = "relevance"       # 相关性
    ACCURACY = "accuracy"         # 准确性
    COMPLETENESS = "completeness" # 完整性
    SAFETY = "safety"             # 安全性
    COHERENCE = "coherence"       # 连贯性
    CONSISTENCY = "consistency"   # 一致性
    PROFESSIONALISM = "professionalism"  # 专业性


class ValidationLevel(Enum):
    """验证级别"""
    BASIC = "basic"       # 基础验证
    STANDARD = "standard" # 标准验证
    STRICT = "strict"     # 严格验证
    EXPERT = "expert"     # 专家验证


class IQualityChecker(IComponent, IConfigurable, IMonitorable):
    """
    质量检查器基础接口

    定义内容质量检查的基础功能。
    """

    @abstractmethod
    async def check_quality(self, content: str, context: Optional[Dict[str, Any]] = None) -> QualityReport:
        """
        执行质量检查

        Args:
            content: 待检查的内容
            context: 上下文信息

        Returns:
            QualityReport: 质量报告
        """
        pass

    @abstractmethod
    async def check_dimension(self, content: str, dimension: QualityDimension, context: Optional[Dict[str, Any]] = None) -> QualityCheckResult:
        """
        检查特定质量维度

        Args:
            content: 待检查的内容
            dimension: 质量维度
            context: 上下文信息

        Returns:
            QualityCheckResult: 检查结果
        """
        pass

    @abstractmethod
    def get_supported_dimensions(self) -> List[QualityDimension]:
        """
        获取支持的质量维度

        Returns:
            List[QualityDimension]: 支持的维度列表
        """
        pass

    @abstractmethod
    def set_validation_level(self, level: ValidationLevel) -> None:
        """
        设置验证级别

        Args:
            level: 验证级别
        """
        pass

    @abstractmethod
    def get_validation_level(self) -> ValidationLevel:
        """
        获取当前验证级别

        Returns:
            ValidationLevel: 当前验证级别
        """
        pass


class IAntiHallucinationChecker(IQualityChecker):
    """
    反幻觉检查器接口

    专门用于检测和防止AI生成内容中的幻觉。
    """

    @abstractmethod
    async def detect_hallucination(self, content: str, ground_truth: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> QualityCheckResult:
        """
        检测幻觉内容

        Args:
            content: 待检查的内容
            ground_truth: 真实参考（可选）
            context: 上下文信息

        Returns:
            QualityCheckResult: 检查结果
        """
        pass

    @abstractmethod
    async def verify_factual_accuracy(self, claims: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None) -> List[QualityCheckResult]:
        """
        验证事实准确性

        Args:
            claims: 事实声明列表
            context: 上下文信息

        Returns:
            List[QualityCheckResult]: 验证结果列表
        """
        pass

    @abstractmethod
    async def cross_reference_check(self, content: str, reference_sources: List[str]) -> QualityCheckResult:
        """
        交叉引用检查

        Args:
            content: 待检查的内容
            reference_sources: 参考源列表

        Returns:
            QualityCheckResult: 检查结果
        """
        pass

    @abstractmethod
    def get_hallucination_patterns(self) -> List[str]:
        """
        获取已知幻觉模式

        Returns:
            List[str]: 幻觉模式列表
        """
        pass


class ISafetyChecker(IQualityChecker):
    """
    安全检查器接口

    检查内容的安全性和合规性。
    """

    @abstractmethod
    async def check_safety(self, content: str, context: Optional[Dict[str, Any]] = None) -> QualityCheckResult:
        """
        检查内容安全性

        Args:
            content: 待检查的内容
            context: 上下文信息

        Returns:
            QualityCheckResult: 安全检查结果
        """
        pass

    @abstractmethod
    async def detect_harmful_content(self, content: str) -> QualityCheckResult:
        """
        检测有害内容

        Args:
            content: 待检查的内容

        Returns:
            QualityCheckResult: 检测结果
        """
        pass

    @abstractmethod
    async def validate_medical_safety(self, medical_content: str, user_profile: Optional[Dict[str, Any]] = None) -> QualityCheckResult:
        """
        验证医疗安全性

        Args:
            medical_content: 医疗相关内容
            user_profile: 用户档案

        Returns:
            QualityCheckResult: 安全验证结果
        """
        pass

    @abstractmethod
    def get_safety_rules(self) -> Dict[str, Any]:
        """
        获取安全规则

        Returns:
            Dict[str, Any]: 安全规则定义
        """
        pass


class IConsistencyChecker(IQualityChecker):
    """
    一致性检查器接口

    检查内容的一致性和逻辑连贯性。
    """

    @abstractmethod
    async def check_internal_consistency(self, content: str) -> QualityCheckResult:
        """
        检查内部一致性

        Args:
            content: 待检查的内容

        Returns:
            QualityCheckResult: 一致性检查结果
        """
        pass

    @abstractmethod
    async def check_context_consistency(self, content: str, context: Dict[str, Any]) -> QualityCheckResult:
        """
        检查上下文一致性

        Args:
            content: 待检查的内容
            context: 上下文信息

        Returns:
            QualityCheckResult: 一致性检查结果
        """
        pass

    @abstractmethod
    async def check_contradictions(self, statements: List[str]) -> List[QualityCheckResult]:
        """
        检查矛盾陈述

        Args:
            statements: 陈述列表

        Returns:
            List[QualityCheckResult]: 矛盾检查结果
        """
        pass

    @abstractmethod
    async def check_temporal_consistency(self, events: List[Dict[str, Any]]) -> QualityCheckResult:
        """
        检查时间一致性

        Args:
            events: 事件列表（包含时间信息）

        Returns:
            QualityCheckResult: 时间一致性检查结果
        """
        pass


class IProfessionalStandardsChecker(IQualityChecker):
    """
    专业标准检查器接口

    检查内容是否符合专业领域标准。
    """

    @abstractmethod
    async def check_professional_standards(self, content: str, domain: str, standards: Optional[List[str]] = None) -> QualityCheckResult:
        """
        检查专业标准合规性

        Args:
            content: 待检查的内容
            domain: 专业领域
            standards: 具体标准列表（可选）

        Returns:
            QualityCheckResult: 标准检查结果
        """
        pass

    @abstractmethod
    def get_domain_standards(self, domain: str) -> List[str]:
        """
        获取领域标准

        Args:
            domain: 专业领域

        Returns:
            List[str]: 标准列表
        """
        pass

    @abstractmethod
    async def validate_against_guidelines(self, content: str, guidelines: Dict[str, Any]) -> QualityCheckResult:
        """
        根据指导原则验证

        Args:
            content: 待检查的内容
            guidelines: 指导原则

        Returns:
            QualityCheckResult: 验证结果
        """
        pass


class IQualityMonitor(IComponent):
    """
    质量监控器接口

    监控和统计质量检查结果。
    """

    @abstractmethod
    async def record_quality_check(self, report: QualityReport) -> None:
        """
        记录质量检查结果

        Args:
            report: 质量报告
        """
        pass

    @abstractmethod
    def get_quality_statistics(self, time_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        获取质量统计信息

        Args:
            time_range: 时间范围（开始时间戳，结束时间戳）

        Returns:
            Dict[str, Any]: 统计信息
        """
        pass

    @abstractmethod
    def get_quality_trends(self, dimension: Optional[QualityDimension] = None) -> Dict[str, Any]:
        """
        获取质量趋势

        Args:
            dimension: 质量维度（可选）

        Returns:
            Dict[str, Any]: 趋势信息
        """
        pass

    @abstractmethod
    def get_quality_alerts(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        获取质量告警

        Args:
            threshold: 质量阈值

        Returns:
            List[Dict[str, Any]]: 告警列表
        """
        pass

    @abstractmethod
    def export_quality_report(self, format_type: str = "json", filters: Optional[Dict[str, Any]] = None) -> Union[str, bytes]:
        """
        导出质量报告

        Args:
            format_type: 导出格式（json, csv, excel）
            filters: 过滤条件

        Returns:
            Union[str, bytes]: 导出的报告
        """
        pass


class IFeedbackCollector(IComponent):
    """
    反馈收集器接口

    收集和处理用户反馈以改进质量检查。
    """

    @abstractmethod
    async def collect_feedback(self, content_id: str, feedback: Dict[str, Any]) -> bool:
        """
        收集用户反馈

        Args:
            content_id: 内容ID
            feedback: 反馈信息

        Returns:
            bool: 收集是否成功
        """
        pass

    @abstractmethod
    async def analyze_feedback_patterns(self) -> Dict[str, Any]:
        """
        分析反馈模式

        Returns:
            Dict[str, Any]: 模式分析结果
        """
        pass

    @abstractmethod
    def get_feedback_statistics(self, time_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        获取反馈统计

        Args:
            time_range: 时间范围

        Returns:
            Dict[str, Any]: 统计信息
        """
        pass

    @abstractmethod
    async def incorporate_feedback(self, feedback: Dict[str, Any]) -> bool:
        """
        将反馈纳入质量检查

        Args:
            feedback: 反馈信息

        Returns:
            bool: 纳入是否成功
        """
        pass


# 便捷的质量检查器基类
class BaseQualityChecker(BaseComponent):
    """
    质量检查器基础实现类
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
        self._validation_level = ValidationLevel.STANDARD
        self._check_count = 0
        self._total_score = 0.0

    def set_validation_level(self, level: ValidationLevel) -> None:
        self._validation_level = level

    def get_validation_level(self) -> ValidationLevel:
        return self._validation_level

    def get_supported_dimensions(self) -> List[QualityDimension]:
        return [QualityDimension.RELEVANCE, QualityDimension.SAFETY]

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "validation_level": self._validation_level.value,
            "check_count": self._check_count,
            "average_score": self._total_score / max(self._check_count, 1),
            "supported_dimensions": [d.value for d in self.get_supported_dimensions()],
            **super().get_metrics()
        }

    def _update_statistics(self, score: float) -> None:
        self._check_count += 1
        self._total_score += score


# 导出接口
__all__ = [
    # 核心数据结构
    'QualityCheckResult',
    'QualityReport',
    'QualityDimension',
    'ValidationLevel',

    # 质量检查接口
    'IQualityChecker',
    'IAntiHallucinationChecker',
    'ISafetyChecker',
    'IConsistencyChecker',
    'IProfessionalStandardsChecker',

    # 监控和反馈接口
    'IQualityMonitor',
    'IFeedbackCollector',

    # 基础实现
    'BaseQualityChecker'
]