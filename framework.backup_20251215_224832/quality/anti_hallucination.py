# -*- coding: utf-8 -*-
"""
DAML-RAG反幻觉验证系统 v2.0

实现多层次的内容真实性验证：
- 事实一致性检查
- 逻辑矛盾检测
- 证据充分性验证
- 安全性风险评估

核心特性：
- 多维度验证：事实性、一致性、充分性、安全性
- 智能证据检索：自动检索相关证据进行交叉验证
- 风险等级评估：基于不同维度的风险量化评估
- 实时监控：持续的验证过程和质量监控

版本：v2.0.0
更新日期：2025-11-17
设计原则：多维度验证、智能证据、风险评估
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import time
from collections import defaultdict

from ..interfaces import (
    IAntiHallucinationChecker, IQualityChecker,
    QualityCheckResult, ValidationLevel
)
from ..base import BaseComponent

logger = logging.getLogger(__name__)


class HallucinationType(Enum):
    """幻觉类型"""
    FACTUAL_INCONSISTENCY = "factual_inconsistency"    # 事实不一致
    LOGICAL_CONTRADICTION = "logical_contradiction"  # 逻辑矛盾
    UNSUPPORTED_CLAIM = "unsupported_claim"          # 无证据支持
    SAFETY_RISK = "safety_risk"                     # 安全风险
    CONTEXTUAL_MISMATCH = "contextual_mismatch"     # 上下文不匹配
    OVERGENERALIZATION = "overgeneralization"       # 过度概括


class RiskLevel(Enum):
    """风险等级"""
    LOW = 1          # 低风险
    MEDIUM = 2       # 中等风险
    HIGH = 3         # 高风险
    CRITICAL = 4     # 严重风险


@dataclass
class EvidenceItem:
    """证据项"""
    content: str
    source: str
    reliability_score: float
    relevance_score: float
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """验证结果"""
    is_hallucination: bool
    confidence_score: float
    detected_issues: List[HallucinationType]
    risk_level: RiskLevel
    evidence_support: List[EvidenceItem]
    safety_concerns: List[str]
    suggestions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FactCheck:
    """事实检查项"""
    claim: str
    is_supported: bool
    confidence: float
    supporting_evidence: List[EvidenceItem]
    contradicting_evidence: List[EvidenceItem]


class AntiHallucinationChecker(BaseComponent, IAntiHallucinationChecker):
    """
    反幻觉验证器

    实现多层次的内容真实性和安全性验证。
    """

    def __init__(self, name: str = "AntiHallucinationChecker", version: str = "2.0.0"):
        super().__init__(name, version)
        self._config = self._get_default_config()
        self._knowledge_base = None  # 知识库接口
        self._safety_checker = None  # 安全检查器

        # 验证指标
        self._validation_metrics = {
            'total_validations': 0,
            'hallucination_detected': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'average_validation_time': 0.0,
            'issue_distribution': {issue.value: 0 for issue in HallucinationType},
            'risk_level_distribution': {level.name: 0 for level in RiskLevel}
        }

        # 缓存
        self._validation_cache = {}
        self._evidence_cache = {}

        # 规则模式
        self._contradiction_patterns = self._init_contradiction_patterns()
        self._safety_patterns = self._init_safety_patterns()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'enable_fact_checking': True,
            'enable_consistency_checking': True,
            'enable_safety_checking': True,
            'enable_evidence_validation': True,
            'min_evidence_threshold': 0.3,
            'safety_threshold': 0.7,
            'fact_check_threshold': 0.6,
            'consistency_threshold': 0.5,
            'max_evidence_items': 10,
            'cache_ttl': 300,
            'enable_parallel_validation': True
        }

    def _init_contradiction_patterns(self) -> List[re.Pattern]:
        """初始化矛盾检测模式"""
        patterns = [
            # 自相矛盾模式
            re.compile(r'.*(?:但是|然而|不过).*((?:不|没|并非|无法).*?).*', re.IGNORECASE),
            re.compile(r'.*(?:虽然|尽管).*((?:不|没|并非|无法).*?).*', re.IGNORECASE),

            # 数字矛盾模式
            re.compile(r'(\d+).*?[-到至].*?(\d+)', re.IGNORECASE),
            re.compile(r'(?:增加|增长|提高).*?(\d+).*?(?:减少|降低|下降).*?(\d+)', re.IGNORECASE),

            # 时间矛盾模式
            re.compile(r'(\d{4}年|\d{1,2}月).*?之后.*?(\d{4}年|\d{1,2}月).*?之前', re.IGNORECASE),
        ]
        return patterns

    def _init_safety_patterns(self) -> List[re.Pattern]:
        """初始化安全风险模式"""
        patterns = [
            # 医疗风险
            re.compile(r'(?:无需|不用|可以).*(?:医生|医院|检查|治疗)', re.IGNORECASE),
            re.compile(r'(?:立即|马上|必须).*(?:停止|放弃).*?治疗', re.IGNORECASE),

            # 运动安全风险
            re.compile(r'(?:无视|忽略|不顾).*(?:疼痛|不适|警告)', re.IGNORECASE),
            re.compile(r'(?:最大|极限|过度).*(?:训练|运动|负荷)', re.IGNORECASE),

            # 绝对化风险
            re.compile(r'(?:绝对|完全|100%).*(?:安全|无害|无风险)', re.IGNORECASE),
            re.compile(r'(?:一定|必然|必定).*(?:成功|有效)', re.IGNORECASE),
        ]
        return patterns

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化反幻觉验证器"""
        try:
            if config:
                self._update_config(config)

            # 初始化组件
            await self._initialize_components()

            logger.info(f"✅ 反幻觉验证器初始化成功: {self.name}")
            self._state = self.ComponentState.READY
            return True

        except Exception as e:
            logger.error(f"❌ 反幻觉验证器初始化失败 {self.name}: {e}")
            self._state = self.ComponentState.ERROR
            return False

    async def _initialize_components(self) -> None:
        """初始化相关组件"""
        # 这里可以初始化知识库、安全检查器等组件
        pass

    def _update_config(self, config: Dict[str, Any]) -> None:
        """更新配置"""
        for key, value in config.items():
            if key in self._config:
                self._config[key] = value
        logger.info(f"反幻觉验证器配置已更新: {self.name}")

    async def check_hallucination(
        self,
        content: str,
        query_context: Optional[Dict[str, Any]] = None,
        ground_truth: Optional[str] = None,
        retrieved_evidence: Optional[List[Dict[str, Any]]] = None
    ) -> QualityCheckResult:
        """检测幻觉内容"""
        start_time = time.time()
        self._validation_metrics['total_validations'] += 1

        try:
            # 检查缓存
            cache_key = self._generate_cache_key(content, query_context)
            if cache_key in self._validation_cache:
                cached_result = self._validation_cache[cache_key]
                if time.time() - cached_result['timestamp'] < self._config['cache_ttl']:
                    return cached_result['result']

            # 执行验证
            validation_result = await self._perform_validation(
                content, query_context, ground_truth, retrieved_evidence
            )

            # 构建质量检查结果
            quality_result = QualityCheckResult(
                content=content,
                is_valid=not validation_result.is_hallucination,
                confidence=validation_result.confidence_score,
                validation_level=self._determine_validation_level(validation_result.risk_level),
                issues=[issue.value for issue in validation_result.detected_issues],
                suggestions=validation_result.suggestions,
                metadata={
                    'risk_level': validation_result.risk_level.name,
                    'evidence_count': len(validation_result.evidence_support),
                    'safety_concerns': validation_result.safety_concerns,
                    'validation_time': time.time() - start_time
                }
            )

            # 缓存结果
            self._cache_validation_result(cache_key, quality_result)

            # 更新指标
            execution_time = time.time() - start_time
            self._update_metrics(validation_result, execution_time)

            return quality_result

        except Exception as e:
            logger.error(f"幻觉检测失败: {e}")
            return QualityCheckResult(
                content=content,
                is_valid=False,
                confidence=0.0,
                validation_level=ValidationLevel.ERROR,
                issues=[f"validation_error: {str(e)}"],
                suggestions=["请检查输入内容格式"]
            )

    async def _perform_validation(
        self,
        content: str,
        query_context: Optional[Dict[str, Any]],
        ground_truth: Optional[str],
        retrieved_evidence: Optional[List[Dict[str, Any]]]
    ) -> ValidationResult:
        """执行验证检查"""
        detected_issues = []
        evidence_support = []
        safety_concerns = []
        suggestions = []

        # 1. 事实一致性检查
        if self._config['enable_fact_checking']:
            fact_check_result = await self._check_factual_consistency(
                content, retrieved_evidence
            )
            if not fact_check_result.is_supported:
                detected_issues.append(HallucinationType.FACTUAL_INCONSISTENCY)
                evidence_support.extend(fact_check_result.supporting_evidence)
                suggestions.append("请核实事实的准确性")

        # 2. 逻辑一致性检查
        if self._config['enable_consistency_checking']:
            consistency_result = await self._check_logical_consistency(content)
            if consistency_result:
                detected_issues.append(HallucinationType.LOGICAL_CONTRADICTION)
                suggestions.append("内容存在逻辑矛盾，请检查")

        # 3. 安全性检查
        if self._config['enable_safety_checking']:
            safety_result = await self._check_safety(content, query_context)
            if safety_result:
                detected_issues.append(HallucinationType.SAFETY_RISK)
                safety_concerns.extend(safety_result)
                suggestions.append("内容存在安全风险，需要谨慎处理")

        # 4. 证据充分性检查
        if self._config['enable_evidence_validation']:
            evidence_result = await self._check_evidence_adequacy(
                content, retrieved_evidence
            )
            if evidence_result < self._config['min_evidence_threshold']:
                detected_issues.append(HallucinationType.UNSUPPORTED_CLAIM)
                suggestions.append("缺乏充分的证据支持")

        # 5. 上下文一致性检查
        if query_context:
            context_result = await self._check_contextual_consistency(
                content, query_context
            )
            if not context_result:
                detected_issues.append(HallucinationType.CONTEXTUAL_MISMATCH)
                suggestions.append("内容与上下文不符")

        # 计算综合评分和风险等级
        confidence_score = self._calculate_confidence_score(
            detected_issues, evidence_support, safety_concerns
        )
        risk_level = self._assess_risk_level(detected_issues, safety_concerns)

        return ValidationResult(
            is_hallucination=len(detected_issues) > 0,
            confidence_score=confidence_score,
            detected_issues=detected_issues,
            risk_level=risk_level,
            evidence_support=evidence_support,
            safety_concerns=safety_concerns,
            suggestions=suggestions,
            metadata={
                'validation_timestamp': time.time(),
                'content_length': len(content),
                'query_context': query_context
            }
        )

    async def _check_factual_consistency(
        self,
        content: str,
        retrieved_evidence: Optional[List[Dict[str, Any]]]
    ) -> FactCheck:
        """检查事实一致性"""
        # 提取关键声明
        claims = self._extract_claims(content)

        if not claims:
            return FactCheck(
                claim=content,
                is_supported=True,
                confidence=1.0,
                supporting_evidence=[],
                contradicting_evidence=[]
            )

        # 收集证据
        evidence_items = []
        if retrieved_evidence:
            for evidence in retrieved_evidence:
                item = EvidenceItem(
                    content=evidence.get('content', ''),
                    source=evidence.get('source', 'unknown'),
                    reliability_score=evidence.get('reliability', 0.5),
                    relevance_score=self._calculate_relevance(content, evidence.get('content', '')),
                    metadata=evidence.get('metadata', {})
                )
                evidence_items.append(item)

        # 验证声明
        supported_claims = 0
        total_confidence = 0.0

        for claim in claims:
            claim_confidence = await self._verify_claim(claim, evidence_items)
            total_confidence += claim_confidence
            if claim_confidence >= self._config['fact_check_threshold']:
                supported_claims += 1

        overall_confidence = total_confidence / len(claims) if claims else 1.0
        is_supported = supported_claims / len(claims) >= 0.6 if claims else True

        return FactCheck(
            claim=content,
            is_supported=is_supported,
            confidence=overall_confidence,
            supporting_evidence=[e for e in evidence_items if e.relevance_score > 0.5],
            contradicting_evidence=[]
        )

    def _extract_claims(self, content: str) -> List[str]:
        """提取关键声明"""
        # 简化的声明提取，实际应该使用更复杂的NLP技术
        sentences = re.split(r'[.!?。！？]', content)
        claims = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # 过滤过短的句子
                # 识别包含事实性声明的句子
                if any(keyword in sentence for keyword in [
                    '是', '有', '能', '会', '需要', '应该', '必须', '不能',
                    '包含', '由', '因为', '由于', '导致', '影响', '产生'
                ]):
                    claims.append(sentence)

        return claims

    async def _verify_claim(self, claim: str, evidence_items: List[EvidenceItem]) -> float:
        """验证声明"""
        if not evidence_items:
            return 0.3  # 没有证据时的默认置信度

        max_relevance = 0.0
        total_reliability = 0.0

        for evidence in evidence_items:
            # 计算相关性
            relevance = self._calculate_relevance(claim, evidence.content)
            if relevance > max_relevance:
                max_relevance = relevance

            # 累积可靠性
            if relevance > 0.3:
                total_reliability += evidence.reliability_score * relevance

        # 综合评分
        confidence = (max_relevance + total_reliability / len(evidence_items)) / 2
        return min(confidence, 1.0)

    def _calculate_relevance(self, text1: str, text2: str) -> float:
        """计算文本相关性（简化实现）"""
        # 使用词汇重叠度计算相关性
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        jaccard_similarity = len(intersection) / len(union)
        return jaccard_similarity

    async def _check_logical_consistency(self, content: str) -> bool:
        """检查逻辑一致性"""
        for pattern in self._contradiction_patterns:
            if pattern.search(content):
                return True  # 发现矛盾
        return False

    async def _check_safety(
        self,
        content: str,
        query_context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """检查安全性"""
        safety_concerns = []

        for pattern in self._safety_patterns:
            matches = pattern.findall(content)
            if matches:
                safety_concerns.extend([f"安全风险: {match}" for match in matches])

        # 基于上下文的安全检查
        if query_context and query_context.get('domain') == 'fitness':
            fitness_safety_concerns = await self._check_fitness_safety(content)
            safety_concerns.extend(fitness_safety_concerns)

        return safety_concerns

    async def _check_fitness_safety(self, content: str) -> List[str]:
        """检查健身领域安全性"""
        concerns = []

        # 健身领域特定的安全模式
        fitness_patterns = [
            r'(?:无视|忽略).*?疼痛',
            r'(?:立即|马上).*?最大重量',
            r'(?:不需要|不用).*?热身',
            r'(每天|每次).*?必须.*?力竭',
            r'(任何|所有).*?人都适用'
        ]

        for pattern in fitness_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                concerns.append(f"健身安全风险: {pattern}")

        return concerns

    async def _check_evidence_adequacy(
        self,
        content: str,
        retrieved_evidence: Optional[List[Dict[str, Any]]]
    ) -> float:
        """检查证据充分性"""
        if not retrieved_evidence:
            return 0.0

        # 计算证据覆盖度
        content_words = set(content.lower().split())
        total_coverage = 0.0

        for evidence in retrieved_evidence[:self._config['max_evidence_items']]:
            evidence_content = evidence.get('content', '').lower()
            evidence_words = set(evidence_content.split())

            coverage = len(content_words & evidence_words) / len(content_words) if content_words else 0
            total_coverage += coverage

        return total_coverage / len(retrieved_evidence) if retrieved_evidence else 0.0

    async def _check_contextual_consistency(
        self,
        content: str,
        query_context: Dict[str, Any]
    ) -> bool:
        """检查上下文一致性"""
        # 检查领域一致性
        if 'domain' in query_context:
            domain = query_context['domain']
            if not self._is_domain_consistent(content, domain):
                return False

        # 检查用户档案一致性
        if 'user_profile' in query_context:
            user_profile = query_context['user_profile']
            if not self._is_user_profile_consistent(content, user_profile):
                return False

        return True

    def _is_domain_consistent(self, content: str, domain: str) -> bool:
        """检查领域一致性"""
        domain_keywords = {
            'fitness': ['运动', '训练', '健身', '锻炼', '体育', '健康'],
            'nutrition': ['营养', '饮食', '食物', '卡路里', '蛋白质', '维生素'],
            'medical': ['医疗', '治疗', '药物', '疾病', '症状', '诊断']
        }

        if domain in domain_keywords:
            keywords = domain_keywords[domain]
            return any(keyword in content for keyword in keywords)

        return True

    def _is_user_profile_consistent(self, content: str, user_profile: Dict[str, Any]) -> bool:
        """检查用户档案一致性"""
        # 简化实现，检查是否与用户的基本信息冲突
        if 'fitness_level' in user_profile:
            level = user_profile['fitness_level']
            # 检查内容是否适合用户的健身水平
            if level == 'beginner' and any(word in content for word in ['高级', '专业', '竞技']):
                return False

        return True

    def _calculate_confidence_score(
        self,
        detected_issues: List[HallucinationType],
        evidence_support: List[EvidenceItem],
        safety_concerns: List[str]
    ) -> float:
        """计算置信度分数"""
        # 基础分数
        base_score = 1.0

        # 问题扣分
        issue_penalty = {
            HallucinationType.FACTUAL_INCONSISTENCY: 0.4,
            HallucinationType.LOGICAL_CONTRADICTION: 0.3,
            HallucinationType.SAFETY_RISK: 0.5,
            HallucinationType.UNSUPPORTED_CLAIM: 0.2,
            HallucinationType.CONTEXTUAL_MISMATCH: 0.25,
            HallucinationType.OVERGENERALIZATION: 0.15
        }

        for issue in detected_issues:
            base_score -= issue_penalty.get(issue, 0.2)

        # 安全风险额外扣分
        if safety_concerns:
            base_score -= 0.3 * len(safety_concerns)

        # 证据支持加分
        if evidence_support:
            avg_evidence_score = sum(e.reliability_score * e.relevance_score for e in evidence_support) / len(evidence_support)
            base_score += avg_evidence_score * 0.2

        return max(0.0, min(1.0, base_score))

    def _assess_risk_level(
        self,
        detected_issues: List[HallucinationType],
        safety_concerns: List[str]
    ) -> RiskLevel:
        """评估风险等级"""
        # 安全风险直接提升到高风险
        if HallucinationType.SAFETY_RISK in detected_issues or safety_concerns:
            return RiskLevel.HIGH

        # 事实不一致为高风险
        if HallucinationType.FACTUAL_INCONSISTENCY in detected_issues:
            return RiskLevel.MEDIUM

        # 多个问题提升风险等级
        if len(detected_issues) >= 2:
            return RiskLevel.MEDIUM

        # 单个问题为低风险
        if detected_issues:
            return RiskLevel.LOW

        return RiskLevel.LOW

    def _determine_validation_level(self, risk_level: RiskLevel) -> ValidationLevel:
        """确定验证级别"""
        level_mapping = {
            RiskLevel.LOW: ValidationLevel.BASIC,
            RiskLevel.MEDIUM: ValidationLevel.STANDARD,
            RiskLevel.HIGH: ValidationLevel.THOROUGH,
            RiskLevel.CRITICAL: ValidationLevel.COMPREHENSIVE
        }
        return level_mapping.get(risk_level, ValidationLevel.STANDARD)

    def _generate_cache_key(self, content: str, context: Optional[Dict[str, Any]]) -> str:
        """生成缓存键"""
        import hashlib
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        context_hash = hashlib.md5(str(context).encode()).hexdigest()[:8] if context else "00000000"
        return f"{content_hash}_{context_hash}"

    def _cache_validation_result(self, cache_key: str, result: QualityCheckResult) -> None:
        """缓存验证结果"""
        self._validation_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }

        # 清理过期缓存
        self._cleanup_cache()

    def _cleanup_cache(self) -> None:
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []

        for key, cached_data in self._validation_cache.items():
            if current_time - cached_data['timestamp'] > self._config['cache_ttl']:
                expired_keys.append(key)

        for key in expired_keys:
            del self._validation_cache[key]

    def _update_metrics(self, result: ValidationResult, execution_time: float) -> None:
        """更新验证指标"""
        if result.is_hallucination:
            self._validation_metrics['hallucination_detected'] += 1

        # 更新问题分布
        for issue in result.detected_issues:
            self._validation_metrics['issue_distribution'][issue.value] += 1

        # 更新风险等级分布
        self._validation_metrics['risk_level_distribution'][result.risk_level.name] += 1

        # 更新平均验证时间
        total_validations = self._validation_metrics['total_validations']
        current_avg = self._validation_metrics['average_validation_time']
        self._validation_metrics['average_validation_time'] = (
            (current_avg * (total_validations - 1) + execution_time) / total_validations
        )

    def get_metrics(self) -> Dict[str, Any]:
        """获取验证器指标"""
        base_metrics = super().get_metrics()
        total_validations = self._validation_metrics['total_validations']

        return {
            **base_metrics,
            **self._validation_metrics,
            'hallucination_rate': (
                self._validation_metrics['hallucination_detected'] / max(total_validations, 1)
            ),
            'cache_size': len(self._validation_cache),
            'average_issues_per_validation': (
                sum(self._validation_metrics['issue_distribution'].values()) / max(total_validations, 1)
            )
        }


# 导出
__all__ = [
    'AntiHallucinationChecker',
    'HallucinationType',
    'RiskLevel',
    'ValidationResult',
    'EvidenceItem',
    'FactCheck'
]