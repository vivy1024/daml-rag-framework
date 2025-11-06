#!/usr/bin/env python3
"""
玉珍健身 框架 规则过滤引擎
实现业务规则验证和个性化约束
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime, timedelta
import re
import json

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """规则类型"""
    SAFETY = "safety"  # 安全规则
    EQUIPMENT = "equipment"  # 器械规则
    CAPACITY = "capacity"  # 容量规则
    REHAB = "rehab"  # 康复规则
    PERSONALIZATION = "personalization"  # 个性化规则
    CONSTRAINT = "constraint"  # 约束规则
    QUALITY = "quality"  # 质量规则


class RuleOperator(Enum):
    """规则操作符"""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_EQUAL = "lte"
    IN = "in"
    NOT_IN = "nin"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX = "regex"
    BETWEEN = "between"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class RulePriority(Enum):
    """规则优先级"""
    CRITICAL = 1  # 关键规则，违反则直接排除
    HIGH = 2      # 高优先级规则
    MEDIUM = 3    # 中等优先级规则
    LOW = 4       # 低优先级规则
    INFO = 5      # 信息性规则，仅记录


@dataclass
class Rule:
    """规则定义"""
    id: str
    name: str
    type: RuleType
    priority: RulePriority
    description: str
    condition: Dict[str, Any]
    action: str  # "exclude", "warn", "modify"
    message_template: str
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = RuleType(self.type)
        if isinstance(self.priority, str):
            self.priority = RulePriority(self.priority)


@dataclass
class RuleContext:
    """规则上下文"""
    user_profile: Dict[str, Any]
    query_context: Dict[str, Any]
    session_data: Dict[str, Any]
    system_config: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RuleResult:
    """规则执行结果"""
    rule_id: str
    rule_name: str
    passed: bool
    action: str
    message: str
    confidence: float = 1.0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterResult:
    """过滤结果"""
    passed_all_rules: bool
    filtered_items: List[Dict[str, Any]]
    rule_results: List[RuleResult]
    warnings: List[str]
    errors: List[str]
    modifications: Dict[str, Any]
    execution_summary: Dict[str, Any]


class RuleValidator(ABC):
    """规则验证器抽象基类"""

    @abstractmethod
    async def validate(self, rule: Rule, context: RuleContext, item: Dict[str, Any]) -> RuleResult:
        """验证规则"""
        pass


class ComparisonValidator(RuleValidator):
    """比较验证器"""

    async def validate(self, rule: Rule, context: RuleContext, item: Dict[str, Any]) -> RuleResult:
        """执行比较验证"""
        start_time = asyncio.get_event_loop().time()

        try:
            condition = rule.condition
            field_path = condition.get("field")
            operator = RuleOperator(condition.get("operator"))
            expected_value = condition.get("value")

            # 获取字段值
            actual_value = self._get_field_value(item, field_path, context)

            # 执行比较
            passed = self._compare_values(actual_value, operator, expected_value)

            # 生成消息
            message = rule.message_template.format(
                field=field_path,
                operator=operator.value,
                expected=expected_value,
                actual=actual_value
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            return RuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=passed,
                action=rule.action,
                message=message,
                execution_time=execution_time,
                metadata={
                    "field": field_path,
                    "operator": operator.value,
                    "expected_value": expected_value,
                    "actual_value": actual_value
                }
            )

        except Exception as e:
            logger.error(f"Rule validation failed: {e}")
            return RuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=False,
                action="exclude",
                message=f"规则验证失败: {str(e)}",
                execution_time=asyncio.get_event_loop().time() - start_time,
                metadata={"error": str(e)}
            )

    def _get_field_value(self, item: Dict[str, Any], field_path: str, context: RuleContext) -> Any:
        """获取字段值"""
        # 支持嵌套字段路径，如 "user.profile.age"
        parts = field_path.split(".")
        value = item

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return None

        return value

    def _compare_values(self, actual: Any, operator: RuleOperator, expected: Any) -> bool:
        """比较值"""
        if operator == RuleOperator.EQUALS:
            return actual == expected
        elif operator == RuleOperator.NOT_EQUALS:
            return actual != expected
        elif operator == RuleOperator.GREATER_THAN:
            try:
                return float(actual) > float(expected)
            except (ValueError, TypeError):
                return False
        elif operator == RuleOperator.GREATER_EQUAL:
            try:
                return float(actual) >= float(expected)
            except (ValueError, TypeError):
                return False
        elif operator == RuleOperator.LESS_THAN:
            try:
                return float(actual) < float(expected)
            except (ValueError, TypeError):
                return False
        elif operator == RuleOperator.LESS_EQUAL:
            try:
                return float(actual) <= float(expected)
            except (ValueError, TypeError):
                return False
        elif operator == RuleOperator.IN:
            return actual in expected if isinstance(expected, (list, tuple, set)) else False
        elif operator == RuleOperator.NOT_IN:
            return actual not in expected if isinstance(expected, (list, tuple, set)) else False
        elif operator == RuleOperator.CONTAINS:
            if isinstance(actual, (list, tuple, set)):
                return str(expected) in [str(item) for item in actual]
            else:
                return str(expected) in str(actual)
        elif operator == RuleOperator.NOT_CONTAINS:
            if isinstance(actual, (list, tuple, set)):
                return str(expected) not in [str(item) for item in actual]
            else:
                return str(expected) not in str(actual)
        elif operator == RuleOperator.REGEX:
            try:
                pattern = re.compile(str(expected))
                return bool(pattern.search(str(actual)))
            except re.error:
                return False
        elif operator == RuleOperator.BETWEEN:
            if isinstance(expected, (list, tuple)) and len(expected) == 2:
                try:
                    actual_float = float(actual)
                    return expected[0] <= actual_float <= expected[1]
                except (ValueError, TypeError):
                    return False
            return False
        elif operator == RuleOperator.IS_NULL:
            return actual is None
        elif operator == RuleOperator.IS_NOT_NULL:
            return actual is not None
        else:
            return False


class FunctionValidator(RuleValidator):
    """函数验证器"""

    async def validate(self, rule: Rule, context: RuleContext, item: Dict[str, Any]) -> RuleResult:
        """执行函数验证"""
        start_time = asyncio.get_event_loop().time()

        try:
            function_name = rule.condition.get("function")
            parameters = rule.condition.get("parameters", {})

            # 执行函数
            if function_name == "age_restriction":
                passed = self._validate_age_restriction(item, context, parameters)
            elif function_name == "equipment_check":
                passed = self._validate_equipment(item, context, parameters)
            elif function_name == "volume_limit":
                passed = self._validate_volume_limit(item, context, parameters)
            elif function_name == "rehab_phase":
                passed = self._validate_rehab_phase(item, context, parameters)
            elif function_name == "custom_function":
                passed = await self._execute_custom_function(item, context, parameters)
            else:
                passed = False
                logger.warning(f"Unknown function: {function_name}")

            message = rule.message_template.format(
                function=function_name,
                parameters=parameters
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            return RuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=passed,
                action=rule.action,
                message=message,
                execution_time=execution_time,
                metadata={
                    "function": function_name,
                    "parameters": parameters
                }
            )

        except Exception as e:
            logger.error(f"Function validation failed: {e}")
            return RuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                passed=False,
                action="exclude",
                message=f"函数验证失败: {str(e)}",
                execution_time=asyncio.get_event_loop().time() - start_time,
                metadata={"error": str(e)}
            )

    def _validate_age_restriction(self, item: Dict[str, Any], context: RuleContext, params: Dict[str, Any]) -> bool:
        """年龄限制验证"""
        user_age = context.user_profile.get("age")
        if user_age is None:
            return True

        min_age = params.get("min_age", 0)
        max_age = params.get("max_age", 120)

        return min_age <= user_age <= max_age

    def _validate_equipment(self, item: Dict[str, Any], context: RuleContext, params: Dict[str, Any]) -> bool:
        """器械检查验证"""
        required_equipment = params.get("required_equipment", [])
        available_equipment = context.user_profile.get("available_equipment", [])

        if not required_equipment:
            return True

        # 检查是否有足够的器械
        for equipment in required_equipment:
            if equipment not in available_equipment:
                return False

        return True

    def _validate_volume_limit(self, item: Dict[str, Any], context: RuleContext, params: Dict[str, Any]) -> bool:
        """容量限制验证"""
        weekly_volume = context.session_data.get("weekly_volume", {})
        max_volume = params.get("max_volume", float('inf'))

        body_part = params.get("body_part")
        if not body_part:
            return True

        current_volume = weekly_volume.get(body_part, 0)
        return current_volume < max_volume

    def _validate_rehab_phase(self, item: Dict[str, Any], context: RuleContext, params: Dict[str, Any]) -> bool:
        """康复阶段验证"""
        rehab_info = context.user_profile.get("rehabilitation", {})
        if not rehab_info:
            return True

        current_phase = rehab_info.get("phase")
        allowed_phases = params.get("allowed_phases", [])

        if not allowed_phases:
            return True

        return current_phase in allowed_phases

    async def _execute_custom_function(self, item: Dict[str, Any], context: RuleContext, params: Dict[str, Any]) -> bool:
        """执行自定义函数"""
        function_code = params.get("code")
        if not function_code:
            return False

        try:
            # 在安全的环境中执行自定义函数
            # 这里应该有更严格的安全措施
            func_globals = {
                "item": item,
                "context": context,
                "params": params,
                "logger": logger
            }

            exec(function_code, func_globals)
            result = func_globals.get("result", False)
            return result

        except Exception as e:
            logger.error(f"Custom function execution failed: {e}")
            return False


class RuleEngine:
    """规则引擎"""

    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self.validators: Dict[str, RuleValidator] = {
            "comparison": ComparisonValidator(),
            "function": FunctionValidator()
        }
        self._stats = {
            "total_validations": 0,
            "rules_executed": 0,
            "execution_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }

    def add_rule(self, rule: Rule) -> None:
        """添加规则"""
        self.rules[rule.id] = rule
        logger.debug(f"Added rule: {rule.name} ({rule.id})")

    def remove_rule(self, rule_id: str) -> bool:
        """移除规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.debug(f"Removed rule: {rule_id}")
            return True
        return False

    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """获取规则"""
        return self.rules.get(rule_id)

    def list_rules(self, rule_type: Optional[RuleType] = None, enabled_only: bool = True) -> List[Rule]:
        """列出规则"""
        rules = list(self.rules.values())

        if rule_type:
            rules = [r for r in rules if r.type == rule_type]

        if enabled_only:
            rules = [r for r in rules if r.enabled]

        # 按优先级排序
        rules.sort(key=lambda r: r.priority.value)
        return rules

    async def apply_rules(
        self,
        items: List[Dict[str, Any]],
        context: RuleContext,
        rule_filters: Optional[List[str]] = None
    ) -> FilterResult:
        """应用规则"""
        start_time = asyncio.get_event_loop().time()

        # 获取要应用的规则
        applicable_rules = self._get_applicable_rules(rule_filters)

        all_results = []
        filtered_items = []
        warnings = []
        errors = []
        modifications = {}

        # 处理每个项目
        for item in items:
            item_results = []
            item_passed = True

            # 应用每个规则
            for rule in applicable_rules:
                try:
                    validator = self.validators.get(rule.condition.get("validator", "comparison"))
                    if not validator:
                        logger.warning(f"No validator found for rule: {rule.id}")
                        continue

                    result = await validator.validate(rule, context, item)
                    item_results.append(result)
                    all_results.append(result)

                    # 更新统计
                    self._stats["rules_executed"] += 1

                    # 处理规则结果
                    if not result.passed:
                        if rule.action == "exclude":
                            item_passed = False
                        elif rule.action == "warn":
                            warnings.append(f"[{rule.name}] {result.message}")
                        elif rule.action == "modify":
                            # 修改项目
                            modifications[item.get("id", "")] = result.metadata.get("modification", {})
                        elif rule.action == "error":
                            errors.append(f"[{rule.name}] {result.message}")
                            item_passed = False

                except Exception as e:
                    logger.error(f"Error applying rule {rule.id}: {e}")
                    error_result = RuleResult(
                        rule_id=rule.id,
                        rule_name=rule.name,
                        passed=False,
                        action="exclude",
                        message=f"规则执行错误: {str(e)}"
                    )
                    item_results.append(error_result)
                    all_results.append(error_result)
                    errors.append(f"[{rule.name}] 规则执行错误: {str(e)}")
                    item_passed = False

            # 如果通过了所有规则，添加到过滤结果
            if item_passed:
                filtered_items.append(item)

        execution_time = asyncio.get_event_loop().time() - start_time
        self._stats["total_validations"] += 1
        self._stats["execution_time"] += execution_time

        return FilterResult(
            passed_all_rules=len(warnings) == 0 and len(errors) == 0,
            filtered_items=filtered_items,
            rule_results=all_results,
            warnings=warnings,
            errors=errors,
            modifications=modifications,
            execution_summary={
                "total_items": len(items),
                "passed_items": len(filtered_items),
                "excluded_items": len(items) - len(filtered_items),
                "rules_applied": len(applicable_rules),
                "warnings_count": len(warnings),
                "errors_count": len(errors),
                "execution_time": execution_time
            }
        )

    def _get_applicable_rules(self, rule_filters: Optional[List[str]] = None) -> List[Rule]:
        """获取适用的规则"""
        rules = list(self.rules.values())

        # 过滤启用的规则
        rules = [r for r in rules if r.enabled]

        # 按ID过滤
        if rule_filters:
            rules = [r for r in rules if r.id in rule_filters]

        # 按优先级排序
        rules.sort(key=lambda r: r.priority.value)

        return rules

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
            "rules_by_type": {
                rule_type.value: len([r for r in self.rules.values() if r.type == rule_type])
                for rule_type in RuleType
            },
            "rules_by_priority": {
                priority.value: len([r for r in self.rules.values() if r.priority == priority])
                for priority in RulePriority
            }
        }

    def load_rules_from_config(self, config_file: str) -> int:
        """从配置文件加载规则"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            loaded_count = 0
            for rule_data in config.get("rules", []):
                rule = Rule(
                    id=rule_data["id"],
                    name=rule_data["name"],
                    type=RuleType(rule_data["type"]),
                    priority=RulePriority(rule_data["priority"]),
                    description=rule_data["description"],
                    condition=rule_data["condition"],
                    action=rule_data["action"],
                    message_template=rule_data["message_template"],
                    enabled=rule_data.get("enabled", True)
                )
                self.add_rule(rule)
                loaded_count += 1

            logger.info(f"Loaded {loaded_count} rules from {config_file}")
            return loaded_count

        except Exception as e:
            logger.error(f"Failed to load rules from {config_file}: {e}")
            return 0

    def save_rules_to_config(self, config_file: str) -> int:
        """保存规则到配置文件"""
        try:
            config = {
                "rules": []
            }

            for rule in self.rules.values():
                rule_data = {
                    "id": rule.id,
                    "name": rule.name,
                    "type": rule.type.value,
                    "priority": rule.priority.value,
                    "description": rule.description,
                    "condition": rule.condition,
                    "action": rule.action,
                    "message_template": rule.message_template,
                    "enabled": rule.enabled
                }
                config["rules"].append(rule_data)

            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(self.rules)} rules to {config_file}")
            return len(self.rules)

        except Exception as e:
            logger.error(f"Failed to save rules to {config_file}: {e}")
            return 0


# 预定义规则
PREDEFINED_RULES = {
    "age_safety": Rule(
        id="age_safety_001",
        name="年龄安全检查",
        type=RuleType.SAFETY,
        priority=RulePriority.CRITICAL,
        description="检查用户年龄是否适合执行某些动作",
        condition={
            "validator": "function",
            "function": "age_restriction",
            "parameters": {"min_age": 16, "max_age": 80}
        },
        action="exclude",
        message_template="年龄限制：用户年龄必须在{min_age}到{max_age}岁之间"
    ),

    "equipment_requirement": Rule(
        id="equipment_001",
        name="器械要求检查",
        type=RuleType.EQUIPMENT,
        priority=RulePriority.HIGH,
        description="检查用户是否拥有所需器械",
        condition={
            "validator": "function",
            "function": "equipment_check",
            "parameters": {}
        },
        action="warn",
        message_template="器械检查：可能缺少必要器械"
    ),

    "volume_limit": Rule(
        id="volume_001",
        name="训练容量限制",
        type=RuleType.CAPACITY,
        priority=RulePriority.MEDIUM,
        description="检查是否超过训练容量",
        condition={
            "validator": "function",
            "function": "volume_limit",
            "parameters": {}
        },
        action="warn",
        message_template="容量警告：可能接近最大可恢复量"
    ),

    "rehab_safety": Rule(
        id="rehab_001",
        name="康复安全检查",
        type=RuleType.REHAB,
        priority=RulePriority.CRITICAL,
        description="康复阶段安全检查",
        condition={
            "validator": "function",
            "function": "rehab_phase",
            "parameters": {}
        },
        action="exclude",
        message_template="康复安全：当前阶段不适合此动作"
    )
}


def create_default_rule_engine() -> RuleEngine:
    """创建默认规则引擎"""
    engine = RuleEngine()

    # 添加预定义规则
    for rule in PREDEFINED_RULES.values():
        engine.add_rule(rule)

    return engine