#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
健身领域知识图谱数据模型定义
基于1,603个完美健身练习数据集构建的专业健身知识图谱

实体类型:
- Exercise: 健身动作 (1,603个)
- Muscle: 肌肉群 (50+个)
- Equipment: 器械 (15+个)
- SafetyConstraint: 安全约束 (30+个)
- NutritionGuideline: 营养指导 (200+个)
- TrainingParameter: 训练参数 (100+个)
- DifficultyLevel: 难度等级 (4个)
- ForceType: 力作类型 (推/拉)
- MechanicType: 机械类型 (单/多关节)

关系类型:
- TARGETS_MUSCLE: 动作→肌肉
- USES_EQUIPMENT: 动作→器械
- HAS_DIFFICULTY: 动作→难度
- HAS_FORCE_TYPE: 动作→力作类型
- HAS_MECHANIC: 动作→机械类型
- SYNERGY_WITH: 肌肉↔肌肉
- ANTAGONIST_OF: 肌肉↔肌肉
- STABILIZES: 肌肉↔肌肉
- COMBINES_WITH: 动作↔动作
- ALTERNATIVE_TO: 动作↔动作
- PROGRESSION_TO: 动作→进阶动作
- SAFETY_CONSTRAINT: 动作→约束
- REQUIRES_NUTRITION: 肌肉→营养
- HAS_TRAINING_PARAM: 动作→参数
- SUITABLE_FOR: 器械→难度等级
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class EntityType(Enum):
    """实体类型枚举"""
    EXERCISE = "Exercise"
    MUSCLE = "Muscle"
    EQUIPMENT = "Equipment"
    SAFETY_CONSTRAINT = "SafetyConstraint"
    NUTRITION_GUIDELINE = "NutritionGuideline"
    TRAINING_PARAMETER = "TrainingParameter"
    DIFFICULTY_LEVEL = "DifficultyLevel"
    FORCE_TYPE = "ForceType"
    MECHANIC_TYPE = "MechanicType"


class RelationType(Enum):
    """关系类型枚举"""
    TARGETS_MUSCLE = "TARGETS_MUSCLE"
    USES_EQUIPMENT = "USES_EQUIPMENT"
    HAS_DIFFICULTY = "HAS_DIFFICULTY"
    HAS_FORCE_TYPE = "HAS_FORCE_TYPE"
    HAS_MECHANIC = "HAS_MECHANIC"
    SYNERGY_WITH = "SYNERGY_WITH"
    ANTAGONIST_OF = "ANTAGONIST_OF"
    STABILIZES = "STABILIZES"
    COMBINES_WITH = "COMBINES_WITH"
    ALTERNATIVE_TO = "ALTERNATIVE_TO"
    PROGRESSION_TO = "PROGRESSION_TO"
    SAFETY_CONSTRAINT = "SAFETY_CONSTRAINT"
    REQUIRES_NUTRITION = "REQUIRES_NUTRITION"
    HAS_TRAINING_PARAM = "HAS_TRAINING_PARAM"
    SUITABLE_FOR = "SUITABLE_FOR"


@dataclass
class NodeSchema:
    """节点模式定义"""
    label: EntityType
    properties: Dict[str, str]
    constraints: List[str]
    indexes: List[str]


@dataclass
class RelationshipSchema:
    """关系模式定义"""
    type: RelationType
    from_label: EntityType
    to_label: EntityType
    properties: Dict[str, str]


class FitnessGraphSchema:
    """健身知识图谱模式定义"""

    def __init__(self):
        self.nodes = self._define_nodes()
        self.relationships = self._define_relationships()
        self.constraints = self._define_constraints()
        self.indexes = self._define_indexes()

    def _define_nodes(self) -> Dict[EntityType, NodeSchema]:
        """定义所有节点模式"""
        return {
            EntityType.EXERCISE: NodeSchema(
                label=EntityType.EXERCISE,
                properties={
                    "id": "Integer - 唯一标识符",
                    "name_zh": "String - 中文名称",
                    "name_en": "String - 英文名称",
                    "description_zh": "String - 中文描述",
                    "description_en": "String - 英文描述",
                    "difficulty_level": "String - 难度等级",
                    "force_type": "String - 力作类型",
                    "mechanic_type": "String - 机械类型",
                    "primary_muscle_zh": "String - 主要目标肌群(中文)",
                    "primary_muscle_en": "String - 主要目标肌群(英文)",
                    "equipment_zh": "String - 使用器械(中文)",
                    "equipment_en": "String - 使用器械(英文)",
                    "grips": "List<String> - 握法",
                    "rep_range": "String - 重复次数范围",
                    "set_range": "String - 组数范围",
                    "rest_period": "String - 休息时间",
                    "safety_level": "String - 安全等级",
                    "muscle_grade": "String - 肌肉专业等级",
                    "calories_per_minute": "Float - 每分钟消耗卡路里",
                    "movement_pattern": "String - 动作模式",
                    "training_focus": "String - 训练重点"
                },
                constraints=[
                    "id: UNIQUE",
                    "name_zh: NOT NULL",
                    "name_en: NOT NULL"
                ],
                indexes=[
                    "id",
                    "name_zh",
                    "name_en",
                    "primary_muscle_zh",
                    "equipment_zh",
                    "difficulty_level",
                    "force_type",
                    "safety_level"
                ]
            ),

            EntityType.MUSCLE: NodeSchema(
                label=EntityType.MUSCLE,
                properties={
                    "id": "Integer - 唯一标识符",
                    "name_zh": "String - 中文名称",
                    "name_en": "String - 英文标准名称",
                    "location": "String - 身体位置",
                    "function_type": "String - 功能类型(屈/伸/内收/外展等)",
                    "muscle_group": "String - 肌肉群分类",
                    "anatomical_plane": "String - 解剖平面(矢状/冠状/水平)",
                    "movement_action": "String - 主要动作类型",
                    "stabilization_role": "String - 稳定化作用",
                    "fiber_type": "String - 肌纤维类型偏好",
                    "training_frequency": "String - 推荐训练频率",
                    "recovery_time": "Integer - 推荐恢复时间(小时)",
                    "synergy_groups": "List<String> - 协同肌群",
                    "antagonist_groups": "List<String> - 拮抗肌群"
                },
                constraints=[
                    "id: UNIQUE",
                    "name_zh: NOT NULL",
                    "name_en: NOT NULL"
                ],
                indexes=[
                    "id",
                    "name_zh",
                    "name_en",
                    "location",
                    "muscle_group",
                    "function_type"
                ]
            ),

            EntityType.EQUIPMENT: NodeSchema(
                label=EntityType.EQUIPMENT,
                properties={
                    "id": "Integer - 唯一标识符",
                    "name_zh": "String - 中文名称",
                    "name_en": "String - 英文名称",
                    "equipment_type": "String - 器械类型",
                    "difficulty_requirement": "String - 使用难度要求",
                    "space_requirement": "String - 空间需求",
                    "price_level": "String - 价格等级",
                    "availability": "String - 普及程度",
                    "maintenance_level": "String - 维护要求",
                    "safety_rating": "String - 安全等级",
                    "versatility_score": "Float - 多功能性评分",
                    "target_muscles": "List<String> - 适用肌群",
                    "exercise_count": "Integer - 支持动作数量"
                },
                constraints=[
                    "id: UNIQUE",
                    "name_zh: NOT NULL"
                ],
                indexes=[
                    "id",
                    "name_zh",
                    "equipment_type",
                    "difficulty_requirement",
                    "availability"
                ]
            ),

            EntityType.SAFETY_CONSTRAINT: NodeSchema(
                label=EntityType.SAFETY_CONSTRAINT,
                properties={
                    "id": "Integer - 唯一标识符",
                    "constraint_type": "String - 约束类型(年龄/损伤/健康等)",
                    "description_zh": "String - 约束描述",
                    "severity_level": "String - 严重程度",
                    "applicable_conditions": "List<String> - 适用条件",
                    "restricted_actions": "List<String> - 限制动作",
                    "alternative_suggestions": "List<String> - 替代建议",
                    "medical_consultation": "Boolean - 是否需要医疗咨询",
                    "professional_supervision": "Boolean - 是否需要专业监督"
                },
                constraints=[
                    "id: UNIQUE",
                    "constraint_type: NOT NULL"
                ],
                indexes=[
                    "id",
                    "constraint_type",
                    "severity_level"
                ]
            ),

            EntityType.NUTRITION_GUIDELINE: NodeSchema(
                label=EntityType.NUTRITION_GUIDELINE,
                properties={
                    "id": "Integer - 唯一标识符",
                    "target_muscle": "String - 目标肌群",
                    "key_nutrients": "List<String> - 关键营养素",
                    "recommended_foods": "List<String> - 推荐食物",
                    "chinese_foods": "List<String> - 中式食物推荐",
                    "timing_advice": "String - 时间建议",
                    "daily_requirements": "Dict - 日需求量",
                    "pre_workout": "String - 训练前建议",
                    "post_workout": "String - 训练后建议",
                    "hydration": "String - 补水建议",
                    "supplementation": "List<String> - 补剂建议"
                },
                constraints=[
                    "id: UNIQUE"
                ],
                indexes=[
                    "id",
                    "target_muscle",
                    "key_nutrients"
                ]
            ),

            EntityType.TRAINING_PARAMETER: NodeSchema(
                label=EntityType.TRAINING_PARAMETER,
                properties={
                    "id": "Integer - 唯一标识符",
                    "difficulty_level": "String - 适用难度",
                    "rep_range": "String - 重复次数范围",
                    "set_range": "String - 组数范围",
                    "rest_period": "String - 休息时间",
                    "intensity_percentage": "String - 强度百分比",
                    "frequency": "String - 训练频率",
                    "progression": "String - 进阶方式",
                    "technique_focus": "List<String> - 技术重点",
                    "volume_calculation": "String - 容量计算",
                    "recovery_guidelines": "String - 恢复指导"
                },
                constraints=[
                    "id: UNIQUE",
                    "difficulty_level: NOT NULL"
                ],
                indexes=[
                    "id",
                    "difficulty_level",
                    "intensity_percentage"
                ]
            ),

            EntityType.DIFFICULTY_LEVEL: NodeSchema(
                label=EntityType.DIFFICULTY_LEVEL,
                properties={
                    "id": "Integer - 唯一标识符",
                    "name_zh": "String - 中文等级名称",
                    "name_en": "String - 英文等级名称",
                    "level": "Integer - 难度级别(1-10)",
                    "prerequisites": "List<String> - 前置要求",
                    "expected_results": "List<String> - 预期效果",
                    "recommended_duration": "String - 建议持续时间",
                    "risk_assessment": "String - 风险评估"
                },
                constraints=[
                    "id: UNIQUE",
                    "name_zh: UNIQUE"
                ],
                indexes=[
                    "id",
                    "name_zh",
                    "level"
                ]
            ),

            EntityType.FORCE_TYPE: NodeSchema(
                label=EntityType.FORCE_TYPE,
                properties={
                    "id": "Integer - 唯一标识符",
                    "name_zh": "String - 中文类型名称",
                    "name_en": "String - 英文类型名称",
                    "movement_pattern": "String - 运动模式",
                    "primary_muscles": "List<String> - 主要肌群",
                    "joint_actions": "List<String> - 关节动作",
                    "training_focus": "String - 训练重点"
                },
                constraints=[
                    "id: UNIQUE",
                    "name_zh: UNIQUE"
                ],
                indexes=[
                    "id",
                    "name_zh",
                    "name_en"
                ]
            ),

            EntityType.MECHANIC_TYPE: NodeSchema(
                label=EntityType.MECHANIC_TYPE,
                properties={
                    "id": "Integer - 唯一标识符",
                    "name_zh": "String - 中文类型名称",
                    "name_en": "String - 英文类型名称",
                    "joint_involvement": "String - 关节参与度",
                    "muscle_isolation": "String - 肌肉孤立程度",
                    "coordination_requirement": "String - 协调性要求",
                    "skill_complexity": "String - 技能复杂度"
                },
                constraints=[
                    "id: UNIQUE",
                    "name_zh: UNIQUE"
                ],
                indexes=[
                    "id",
                    "name_zh",
                    "name_en"
                ]
            )
        }

    def _define_relationships(self) -> List[RelationshipSchema]:
        """定义所有关系模式"""
        return [
            # Exercise -> Muscle
            RelationshipSchema(
                type=RelationType.TARGETS_MUSCLE,
                from_label=EntityType.EXERCISE,
                to_label=EntityType.MUSCLE,
                properties={
                    "target_type": "String - 目标类型(主要/次要/稳定)",
                    "activation_level": "Float - 激活程度(0-1)",
                    "movement_role": "String - 运动角色(原动肌/协同肌/稳定肌)",
                    "contribution_percentage": "Float - 贡献百分比"
                }
            ),

            # Exercise -> Equipment
            RelationshipSchema(
                type=RelationType.USES_EQUIPMENT,
                from_label=EntityType.EXERCISE,
                to_label=EntityType.EQUIPMENT,
                properties={
                    "necessity": "String - 必要性(必需/可选/替代)",
                    "quantity": "Integer - 所需数量",
                    "weight_requirement": "String - 重量要求",
                    "setup_complexity": "String - 安装复杂度"
                }
            ),

            # Exercise -> DifficultyLevel
            RelationshipSchema(
                type=RelationType.HAS_DIFFICULTY,
                from_label=EntityType.EXERCISE,
                to_label=EntityType.DIFFICULTY_LEVEL,
                properties={
                    "rating": "Integer - 难度评分(1-10)",
                    "learning_curve": "String - 学习曲线",
                    "skill_prerequisites": "List<String> - 技能前置要求"
                }
            ),

            # Exercise -> ForceType
            RelationshipSchema(
                type=RelationType.HAS_FORCE_TYPE,
                from_label=EntityType.EXERCISE,
                to_label=EntityType.FORCE_TYPE,
                properties={
                    "dominance": "Float - 主导程度(0-1)",
                    "technique_emphasis": "String - 技术重点"
                }
            ),

            # Exercise -> MechanicType
            RelationshipSchema(
                type=RelationType.HAS_MECHANIC,
                from_label=EntityType.EXERCISE,
                to_label=EntityType.MECHANIC_TYPE,
                properties={
                    "complexity_score": "Float - 复杂度评分(0-1)",
                    "coordination_requirement": "String - 协调性要求"
                }
            ),

            # Muscle <-> Muscle (协同关系)
            RelationshipSchema(
                type=RelationType.SYNERGY_WITH,
                from_label=EntityType.MUSCLE,
                to_label=EntityType.MUSCLE,
                properties={
                    "synergy_strength": "Float - 协同强度(0-1)",
                    "functional_relationship": "String - 功能关系",
                    "training_combination": "String - 训练组合建议"
                }
            ),

            # Muscle <-> Muscle (拮抗关系)
            RelationshipSchema(
                type=RelationType.ANTAGONIST_OF,
                from_label=EntityType.MUSCLE,
                to_label=EntityType.MUSCLE,
                properties={
                    "antagonism_strength": "Float - 拮抗强度(0-1)",
                    "balance_importance": "String - 平衡重要性",
                    "training_ratio": "String - 训练比例建议"
                }
            ),

            # Muscle <-> Muscle (稳定关系)
            RelationshipSchema(
                type=RelationType.STABILIZES,
                from_label=EntityType.MUSCLE,
                to_label=EntityType.MUSCLE,
                properties={
                    "stabilization_role": "String - 稳定化作用",
                    "importance_level": "String - 重要程度"
                }
            ),

            # Exercise <-> Exercise (组合关系)
            RelationshipSchema(
                type=RelationType.COMBINES_WITH,
                from_label=EntityType.EXERCISE,
                to_label=EntityType.EXERCISE,
                properties={
                    "combination_type": "String - 组合类型(超级组/复合组/循环)",
                    "rest_between": "String - 组间休息",
                    "synergy_benefit": "String - 协同收益"
                }
            ),

            # Exercise <-> Exercise (替代关系)
            RelationshipSchema(
                type=RelationType.ALTERNATIVE_TO,
                from_label=EntityType.EXERCISE,
                to_label=EntityType.EXERCISE,
                properties={
                    "alternative_reason": "String - 替代原因",
                    "similarity_score": "Float - 相似度评分(0-1)",
                    "equipment_requirement": "String - 器械要求差异"
                }
            ),

            # Exercise -> Exercise (进阶关系)
            RelationshipSchema(
                type=RelationType.PROGRESSION_TO,
                from_label=EntityType.EXERCISE,
                to_label=EntityType.EXERCISE,
                properties={
                    "progression_type": "String - 进阶类型(难度/重量/技术)",
                    "readiness_criteria": "List<String> - 准备标准",
                    "progression_timeline": "String - 进阶时间线"
                }
            ),

            # Exercise -> SafetyConstraint
            RelationshipSchema(
                type=RelationType.SAFETY_CONSTRAINT,
                from_label=EntityType.EXERCISE,
                to_label=EntityType.SAFETY_CONSTRAINT,
                properties={
                    "constraint_severity": "String - 约束严重程度",
                    "violation_risk": "String - 违规风险",
                    "monitoring_required": "Boolean - 是否需要监控"
                }
            ),

            # Muscle -> NutritionGuideline
            RelationshipSchema(
                type=RelationType.REQUIRES_NUTRITION,
                from_label=EntityType.MUSCLE,
                to_label=EntityType.NUTRITION_GUIDELINE,
                properties={
                    "nutrition_priority": "String - 营养优先级",
                    "timing_importance": "String - 时间重要性"
                }
            ),

            # Exercise -> TrainingParameter
            RelationshipSchema(
                type=RelationType.HAS_TRAINING_PARAM,
                from_label=EntityType.EXERCISE,
                to_label=EntityType.TRAINING_PARAMETER,
                properties={
                    "parameter_applicability": "String - 参数适用性",
                    "customization_level": "String - 个性化程度"
                }
            ),

            # Equipment -> DifficultyLevel
            RelationshipSchema(
                type=RelationType.SUITABLE_FOR,
                from_label=EntityType.EQUIPMENT,
                to_label=EntityType.DIFFICULTY_LEVEL,
                properties={
                    "suitability_score": "Float - 适用性评分(0-1)",
                    "learning_curve": "String - 学习曲线"
                }
            )
        ]

    def _define_constraints(self) -> List[str]:
        """定义图约束"""
        return [
            # 节点唯一性约束
            "CREATE CONSTRAINT exercise_id_unique IF NOT EXISTS FOR (e:Exercise) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT muscle_id_unique IF NOT EXISTS FOR (m:Muscle) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT equipment_id_unique IF NOT EXISTS FOR (eq:Equipment) REQUIRE eq.id IS UNIQUE",
            "CREATE CONSTRAINT safety_constraint_id_unique IF NOT EXISTS FOR (s:SafetyConstraint) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT nutrition_id_unique IF NOT EXISTS FOR (n:NutritionGuideline) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT training_param_id_unique IF NOT EXISTS FOR (t:TrainingParameter) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT difficulty_id_unique IF NOT EXISTS FOR (d:DifficultyLevel) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT force_type_id_unique IF NOT EXISTS FOR (f:ForceType) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT mechanic_type_id_unique IF NOT EXISTS FOR (m:MechanicType) REQUIRE m.id IS UNIQUE",

            # 节点存在性约束
            "CREATE CONSTRAINT exercise_name_exists IF NOT EXISTS FOR (e:Exercise) REQUIRE e.name_zh IS NOT NULL",
            "CREATE CONSTRAINT muscle_name_exists IF NOT EXISTS FOR (m:Muscle) REQUIRE m.name_zh IS NOT NULL",
            "CREATE CONSTRAINT equipment_name_exists IF NOT EXISTS FOR (eq:Equipment) REQUIRE eq.name_zh IS NOT NULL"
        ]

    def _define_indexes(self) -> List[str]:
        """定义图索引"""
        return [
            # 节点属性索引
            "CREATE INDEX exercise_name_zh_index IF NOT EXISTS FOR (e:Exercise) ON (e.name_zh)",
            "CREATE INDEX exercise_name_en_index IF NOT EXISTS FOR (e:Exercise) ON (e.name_en)",
            "CREATE INDEX exercise_primary_muscle_index IF NOT EXISTS FOR (e:Exercise) ON (e.primary_muscle_zh)",
            "CREATE INDEX exercise_equipment_index IF NOT EXISTS FOR (e:Exercise) ON (e.equipment_zh)",
            "CREATE INDEX exercise_difficulty_index IF NOT EXISTS FOR (e:Exercise) ON (e.difficulty_level)",
            "CREATE INDEX exercise_safety_level_index IF NOT EXISTS FOR (e:Exercise) ON (e.safety_level)",
            "CREATE INDEX exercise_force_type_index IF NOT EXISTS FOR (e:Exercise) ON (e.force_type)",
            "CREATE INDEX exercise_mechanic_type_index IF NOT EXISTS FOR (e:Exercise) ON (e.mechanic_type)",

            "CREATE INDEX muscle_name_zh_index IF NOT EXISTS FOR (m:Muscle) ON (m.name_zh)",
            "CREATE INDEX muscle_name_en_index IF NOT EXISTS FOR (m:Muscle) ON (m.name_en)",
            "CREATE INDEX muscle_location_index IF NOT EXISTS FOR (m:Muscle) ON (m.location)",
            "CREATE INDEX muscle_group_index IF NOT EXISTS FOR (m:Muscle) ON (m.muscle_group)",
            "CREATE INDEX muscle_function_type_index IF NOT EXISTS FOR (m:Muscle) ON (m.function_type)",

            "CREATE INDEX equipment_name_zh_index IF NOT EXISTS FOR (eq:Equipment) ON (eq.name_zh)",
            "CREATE INDEX equipment_type_index IF NOT EXISTS FOR (eq:Equipment) ON (eq.equipment_type)",
            "CREATE INDEX equipment_availability_index IF NOT EXISTS FOR (eq:Equipment) ON (eq.availability)",

            # 关系索引
            "CREATE INDEX targets_muscle_index IF NOT EXISTS FOR ()-[r:TARGETS_MUSCLE]-() ON (r.target_type)",
            "CREATE INDEX uses_equipment_index IF NOT EXISTS FOR ()-[r:USES_EQUIPMENT]-() ON (r.necessity)",
            "CREATE INDEX has_difficulty_index IF NOT EXISTS FOR ()-[r:HAS_DIFFICULTY]-() ON (r.rating)",
            "CREATE INDEX synergy_strength_index IF NOT EXISTS FOR ()-[r:SYNERGY_WITH]-() ON (r.synergy_strength)",
            "CREATE INDEX antagonism_strength_index IF NOT EXISTS FOR ()-[r:ANTAGONIST_OF]-() ON (r.antagonism_strength)",
            "CREATE INDEX combination_type_index IF NOT EXISTS FOR ()-[r:COMBINES_WITH]-() ON (r.combination_type)",
            "CREATE INDEX similarity_score_index IF NOT EXISTS FOR ()-[r:ALTERNATIVE_TO]-() ON (r.similarity_score)",
            "CREATE INDEX progression_type_index IF NOT EXISTS FOR ()-[r:PROGRESSION_TO]-() ON (r.progression_type)"
        ]

    def get_schema_cypher(self) -> str:
        """获取完整的模式创建Cypher语句"""
        cypher_statements = []

        # 约束创建
        cypher_statements.append("// 创建约束")
        cypher_statements.extend(self.constraints)
        cypher_statements.append("")

        # 索引创建
        cypher_statements.append("// 创建索引")
        cypher_statements.extend(self.indexes)

        return "\n".join(cypher_statements)

    def get_node_schema_dict(self) -> Dict[str, Any]:
        """获取节点模式的字典表示"""
        return {
            node_type.value: {
                "properties": node.properties,
                "constraints": node.constraints,
                "indexes": node.indexes
            }
            for node_type, node in self.nodes.items()
        }

    def get_relationship_schema_dict(self) -> Dict[str, Any]:
        """获取关系模式的字典表示"""
        return {
            rel_type.value: {
                "from": rel.from_label.value,
                "to": rel.to_label.value,
                "properties": rel.properties
            }
            for rel_type, rel in enumerate(self.relationships)
        }