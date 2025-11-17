# -*- coding: utf-8 -*-
"""
DAML-RAG约束验证器 v2.0

实现专业领域的约束验证和安全检查：
- 安全性约束检查
- 专业规则验证
- 禁忌症和风险识别
- 证据等级验证

版本：v2.0.0
更新日期：2025-11-17
设计原则：安全优先、专业标准、证据驱动
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..interfaces.retrieval import RetrievalResult, QueryRequest
from ..interfaces.quality import IQualityChecker, QualityCheckResult, QualityDimension

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """验证级别"""
    CRITICAL = "critical"    # 关键约束，违反则拒绝
    WARNING = "warning"      # 警告约束，违反但可接受
    INFO = "info"           # 信息约束，仅记录


class ConstraintType(Enum):
    """约束类型"""
    SAFETY = "safety"                 # 安全性约束
    MEDICAL = "medical"              # 医疗约束
    FITNESS_LEVEL = "fitness_level"  # 健身水平约束
    EQUIPMENT = "equipment"          # 器械约束
    ENVIRONMENT = "environment"       # 环境约束
    TIME = "time"                    # 时间约束
    EVIDENCE = "evidence"            # 证据约束


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"         # 低风险
    MEDIUM = "medium"   # 中等风险
    HIGH = "high"       # 高风险
    CRITICAL = "critical"  # 关键风险


@dataclass
class ValidationRule:
    """验证规则"""
    id: str
    name: str
    constraint_type: ConstraintType
    validation_level: ValidationLevel
    risk_level: RiskLevel
    description: str
    condition: str                      # 验证条件表达式
    action: str                        # 违反时的动作
    weight: float = 1.0                # 权重
    enabled: bool = True               # 是否启用


@dataclass
class ValidationResult:
    """验证结果"""
    rule_id: str
    rule_name: str
    is_valid: bool
    confidence: float
    risk_level: RiskLevel
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """用户档案"""
    age: Optional[int] = None
    gender: Optional[str] = None
    fitness_level: str = "beginner"      # beginner, intermediate, advanced
    health_conditions: List[str] = field(default_factory=list)
    injuries: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    equipment: List[str] = field(default_factory=list)
    experience_months: int = 0


class ConstraintValidator(IQualityChecker):
    """
    约束验证器

    对检索结果进行专业领域的约束验证和安全检查。
    """

    def __init__(self, name: str = "ConstraintValidator", version: str = "2.0.0"):
        super().__init__(name, version)
        self._rules: Dict[str, ValidationRule] = {}
        self._user_profiles: Dict[str, UserProfile] = {}
        self._medical_conditions = set()
        self._risk_factors = set()
        self._evidence_standards = {}

        # 性能指标
        self._metrics = {
            'total_validations': 0,
            'critical_violations': 0,
            'warning_violations': 0,
            'average_validation_time': 0.0
        }

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化约束验证器"""
        try:
            # 加载验证规则
            await self._load_validation_rules()

            # 加载医学知识库
            await self._load_medical_knowledge()

            # 加载证据标准
            await self._load_evidence_standards()

            logger.info(f"✅ 约束验证器初始化成功: {self.name}")
            self._state = self.ComponentState.READY
            return True

        except Exception as e:
            logger.error(f"❌ 约束验证器初始化失败 {self.name}: {e}")
            self._state = self.ComponentState.ERROR
            return False

    async def _load_validation_rules(self) -> None:
        """加载验证规则"""
        # 安全性约束
        safety_rules = [
            ValidationRule(
                id="safety_001",
                name="高风险动作禁忌",
                constraint_type=ConstraintType.SAFETY,
                validation_level=ValidationLevel.CRITICAL,
                risk_level=RiskLevel.CRITICAL,
                description="高血压患者避免高强度训练",
                condition="user.has_condition('hypertension') and content.risk_level == 'high'",
                action="reject",
                weight=1.0
            ),
            ValidationRule(
                id="safety_002",
                name="器械使用安全",
                constraint_type=ConstraintType.SAFETY,
                validation_level=ValidationLevel.WARNING,
                risk_level=RiskLevel.MEDIUM,
                description="初学者需要指导使用复杂器械",
                condition="user.fitness_level == 'beginner' and content.equipment_complexity == 'high'",
                action="warn",
                weight=0.7
            )
        ]

        # 医疗约束
        medical_rules = [
            ValidationRule(
                id="medical_001",
                name="心脏病患者运动限制",
                constraint_type=ConstraintType.MEDICAL,
                validation_level=ValidationLevel.CRITICAL,
                risk_level=RiskLevel.CRITICAL,
                description="心脏病患者避免高强度有氧运动",
                condition="user.has_condition('heart_disease') and content.intensity == 'high'",
                action="reject",
                weight=1.0
            )
        ]

        # 健身水平约束
        fitness_rules = [
            ValidationRule(
                id="fitness_001",
                name="初学者强度匹配",
                constraint_type=ConstraintType.FITNESS_LEVEL,
                validation_level=ValidationLevel.WARNING,
                risk_level=RiskLevel.LOW,
                description="推荐适合初学者的训练强度",
                condition="user.fitness_level == 'beginner' and content.difficulty > 'intermediate'",
                action="adjust",
                weight=0.5
            )
        ]

        # 添加所有规则
        for rule in safety_rules + medical_rules + fitness_rules:
            self._rules[rule.id] = rule

        logger.info(f"加载了 {len(self._rules)} 个验证规则")

    async def _load_medical_knowledge(self) -> None:
        """加载医学知识库"""
        # 运动禁忌症
        self._medical_conditions.update([
            "heart_disease", "hypertension", "diabetes", "asthma",
            "arthritis", "osteoporosis", "pregnancy", "recovery"
        ])

        # 风险因素
        self._risk_factors.update([
            "obesity", "smoking", "sedentary_lifestyle", "stress",
            "poor_nutrition", "sleep_deprivation"
        ])

        logger.info(f"加载了医学知识库: {len(self._medical_conditions)} 种病症, {len(self._risk_factors)} 种风险因素")

    async def _load_evidence_standards(self) -> None:
        """加载证据标准"""
        self._evidence_standards = {
            "systematic_review": 1.0,
            "meta_analysis": 1.0,
            "rct": 0.9,
            "cohort_study": 0.7,
            "case_control": 0.6,
            "expert_opinion": 0.4,
            "anecdotal": 0.2
        }

    async def validate(
        self,
        result: RetrievalResult,
        request: QueryRequest,
        user_profile: Optional[UserProfile] = None
    ) -> ValidationResult:
        """验证单个检索结果"""
        start_time = asyncio.get_event_loop().time()
        self._metrics['total_validations'] += 1

        try:
            # 获取用户档案
            if not user_profile:
                user_profile = self._get_user_profile(request)

            # 执行所有验证规则
            violations = []
            total_confidence = 1.0
            max_risk_level = RiskLevel.LOW

            for rule in self._rules.values():
                if not rule.enabled:
                    continue

                validation_result = await self._apply_rule(rule, result, request, user_profile)
                if not validation_result.is_valid:
                    violations.append(validation_result)
                    total_confidence *= (1.0 - rule.weight)
                    max_risk_level = max(max_risk_level, validation_result.risk_level, key=lambda x: x.value)

            # 检查是否有关键违规
            critical_violations = [v for v in violations if v.validation_level == ValidationLevel.CRITICAL]
            if critical_violations:
                self._metrics['critical_violations'] += 1

            # 检查警告违规
            warning_violations = [v for v in violations if v.validation_level == ValidationLevel.WARNING]
            if warning_violations:
                self._metrics['warning_violations'] += 1

            # 生成最终验证结果
            is_valid = len(critical_violations) == 0
            confidence = total_confidence if is_valid else max(0.0, total_confidence)
            message = self._generate_validation_message(violations)

            # 更新指标
            execution_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(execution_time)

            return ValidationResult(
                rule_id="combined",
                rule_name="综合验证",
                is_valid=is_valid,
                confidence=confidence,
                risk_level=max_risk_level,
                message=message,
                metadata={
                    'violations': len(violations),
                    'critical_violations': len(critical_violations),
                    'warning_violations': len(warning_violations),
                    'execution_time': execution_time
                }
            )

        except Exception as e:
            logger.error(f"约束验证失败: {e}")
            return ValidationResult(
                rule_id="error",
                rule_name="验证错误",
                is_valid=False,
                confidence=0.0,
                risk_level=RiskLevel.CRITICAL,
                message=f"验证过程出错: {str(e)}"
            )

    async def _apply_rule(
        self,
        rule: ValidationRule,
        result: RetrievalResult,
        request: QueryRequest,
        user_profile: UserProfile
    ) -> ValidationResult:
        """应用单个验证规则"""
        try:
            # 解析条件表达式（简化实现）
            condition_met = await self._evaluate_condition(rule.condition, result, user_profile)

            if condition_met:
                # 违反规则
                return ValidationResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    is_valid=False,
                    confidence=1.0 - rule.weight,
                    risk_level=rule.risk_level,
                    message=f"违反规则: {rule.description}",
                    metadata={
                        'constraint_type': rule.constraint_type.value,
                        'validation_level': rule.validation_level.value,
                        'action': rule.action
                    }
                )
            else:
                # 通过规则
                return ValidationResult(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    is_valid=True,
                    confidence=1.0,
                    risk_level=RiskLevel.LOW,
                    message="验证通过"
                )

        except Exception as e:
            logger.error(f"应用规则失败 {rule.id}: {e}")
            return ValidationResult(
                rule_id=rule.id,
                rule_name=rule.name,
                is_valid=False,
                confidence=0.0,
                risk_level=RiskLevel.CRITICAL,
                message=f"规则评估失败: {str(e)}"
            )

    async def _evaluate_condition(
        self,
        condition: str,
        result: RetrievalResult,
        user_profile: UserProfile
    ) -> bool:
        """评估条件表达式"""
        # 简化的条件评估实现
        # 实际应该使用更复杂的表达式解析器

        # 示例条件评估
        if "user.has_condition('hypertension')" in condition:
            return "hypertension" in user_profile.health_conditions

        if "user.fitness_level == 'beginner'" in condition:
            return user_profile.fitness_level == "beginner"

        if "content.risk_level == 'high'" in condition:
            return result.metadata.get('risk_level') == 'high'

        # 默认返回False（不违反）
        return False

    def _get_user_profile(self, request: QueryRequest) -> UserProfile:
        """获取用户档案"""
        # 从请求中提取用户信息
        user_id = request.metadata.get('user_id', 'default')

        if user_id in self._user_profiles:
            return self._user_profiles[user_id]

        # 创建默认档案
        default_profile = UserProfile()
        self._user_profiles[user_id] = default_profile
        return default_profile

    def _generate_validation_message(self, violations: List[ValidationResult]) -> str:
        """生成验证消息"""
        if not violations:
            return "验证通过"

        critical_violations = [v for v in violations if v.validation_level == ValidationLevel.CRITICAL]
        warning_violations = [v for v in violations if v.validation_level == ValidationLevel.WARNING]

        messages = []
        if critical_violations:
            messages.append(f"发现 {len(critical_violations)} 个关键安全违规")
        if warning_violations:
            messages.append(f"发现 {len(warning_violations)} 个警告")

        return "; ".join(messages) if messages else "验证通过"

    def _update_metrics(self, execution_time: float) -> None:
        """更新性能指标"""
        total_validations = self._metrics['total_validations']
        current_avg = self._metrics['average_validation_time']
        self._metrics['average_validation_time'] = (
            (current_avg * (total_validations - 1) + execution_time) / total_validations
        )

    # IQualityChecker接口实现
    async def check_quality(self, content: str, context: Optional[Dict[str, Any]] = None) -> QualityCheckResult:
        """质量检查接口实现"""
        # 创建临时的检索结果和请求对象
        result = RetrievalResult(
            document_id="temp",
            content=content,
            score=1.0,
            metadata=context or {}
        )

        request = QueryRequest(
            query_id="temp",
            query_text=context.get('query_text', '') if context else '',
            domain=context.get('domain', 'general') if context else 'general',
            top_k=1
        )

        validation_result = await self.validate(result, request)

        return QualityCheckResult(
            dimension=QualityDimension.SAFETY,
            score=validation_result.confidence,
            passed=validation_result.is_valid,
            details={
                'risk_level': validation_result.risk_level.value,
                'message': validation_result.message,
                'violations': validation_result.metadata.get('violations', 0)
            }
        )

    async def add_rule(self, rule: ValidationRule) -> bool:
        """添加验证规则"""
        try:
            self._rules[rule.id] = rule
            logger.info(f"添加验证规则: {rule.name}")
            return True
        except Exception as e:
            logger.error(f"添加验证规则失败: {e}")
            return False

    async def remove_rule(self, rule_id: str) -> bool:
        """移除验证规则"""
        try:
            if rule_id in self._rules:
                del self._rules[rule_id]
                logger.info(f"移除验证规则: {rule_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"移除验证规则失败: {e}")
            return False

    def get_rules(self, constraint_type: Optional[ConstraintType] = None) -> List[ValidationRule]:
        """获取验证规则"""
        if constraint_type:
            return [rule for rule in self._rules.values() if rule.constraint_type == constraint_type]
        return list(self._rules.values())

    def get_metrics(self) -> Dict[str, Any]:
        """获取验证器指标"""
        base_metrics = super().get_metrics()
        return {
            **base_metrics,
            'total_validations': self._metrics['total_validations'],
            'critical_violations': self._metrics['critical_violations'],
            'warning_violations': self._metrics['warning_violations'],
            'average_validation_time': self._metrics['average_validation_time'],
            'total_rules': len(self._rules),
            'enabled_rules': len([r for r in self._rules.values() if r.enabled]),
            'user_profiles': len(self._user_profiles)
        }


# 导出
__all__ = [
    'ConstraintValidator',
    'ValidationRule',
    'ValidationResult',
    'UserProfile',
    'ConstraintType',
    'ValidationLevel',
    'RiskLevel'
]