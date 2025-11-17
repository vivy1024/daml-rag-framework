# -*- coding: utf-8 -*-
"""
DAML-RAG查询分析器 v2.0

实现智能查询分析和复杂度评估：
- 查询意图识别
- 复杂度评估
- 实体和关系提取
- 查询扩展和改写

版本：v2.0.0
更新日期：2025-11-17
设计原则：智能分析、意图理解、自动优化
"""

import asyncio
import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

from ..interfaces.retrieval import QueryRequest, QueryComplexity

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """查询意图"""
    INFORMATION_SEEKING = "information_seeking"    # 信息查询
    COMPARISON = "comparison"                     # 比较查询
    RECOMMENDATION = "recommendation"            # 推荐查询
    TUTORIAL = "tutorial"                        # 教程查询
    SAFETY_CHECK = "safety_check"                # 安全检查
    PROBLEM_SOLVING = "problem_solving"         # 问题解决
    PLANNING = "planning"                        # 计划制定


class EntityType(Enum):
    """实体类型"""
    EXERCISE = "exercise"           # 运动动作
    MUSCLE = "muscle"              # 肌肉群
    EQUIPMENT = "equipment"        # 器械
    INJURY = "injury"              # 伤病
    GOAL = "goal"                  # 目标
    CONDITION = "condition"         # 状况
    TECHNIQUE = "technique"        # 技术动作


@dataclass
class QueryAnalysis:
    """查询分析结果"""
    original_query: str
    cleaned_query: str
    intent: QueryIntent
    complexity: QueryComplexity
    entities: List[Tuple[str, EntityType]]
    relations: List[Tuple[str, str, str]]  # (entity1, relation, entity2)
    keywords: List[str]
    sentiment: str                      # positive, negative, neutral
    confidence: float
    suggestions: List[str] = field(default_factory=list)


@dataclass
class QueryExpansion:
    """查询扩展"""
    synonyms: List[str]                 # 同义词
    related_terms: List[str]            # 相关词
    broader_terms: List[str]            # 上位词
    narrower_terms: List[str]           # 下位词
    expanded_queries: List[str]         # 扩展查询


class QueryAnalyzer:
    """
    查询分析器

    对用户查询进行深度分析和智能处理。
    """

    def __init__(self, name: str = "QueryAnalyzer", version: str = "2.0.0"):
        self.name = name
        self.version = version

        # 领域词典
        self._entity_dictionary = {}
        self._synonym_dictionary = {}
        self._relation_patterns = []
        self._intent_patterns = {}

        # 性能指标
        self._metrics = {
            'total_queries': 0,
            'average_analysis_time': 0.0,
            'intent_distribution': {},
            'complexity_distribution': {}
        }

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化查询分析器"""
        try:
            # 加载领域词典
            await self._load_domain_dictionaries()

            # 加载意图模式
            await self._load_intent_patterns()

            # 加载关系模式
            await self._load_relation_patterns()

            logger.info(f"✅ 查询分析器初始化成功: {self.name}")
            return True

        except Exception as e:
            logger.error(f"❌ 查询分析器初始化失败 {self.name}: {e}")
            return False

    async def _load_domain_dictionaries(self) -> None:
        """加载领域词典"""
        # 运动实体词典
        self._entity_dictionary[EntityType.EXERCISE] = {
            '深蹲', '卧推', '硬拉', '引体向上', '俯卧撑', '平板支撑', '卷腹',
            '划船', '推举', '飞鸟', '弯举', '臂屈伸', '弓步蹲', '臀桥',
            '登山跑', '开合跳', '波比跳', '土耳其起立', '风车', '壶铃摆动'
        }

        # 肌肉群词典
        self._entity_dictionary[EntityType.MUSCLE] = {
            '胸肌', '背肌', '腹肌', '腿部肌肉', '肩部', '手臂', '臀部',
            '大腿前侧', '大腿后侧', '小腿', '核心', '下背部', '上背部',
            '三角肌', '肱二头肌', '肱三头肌', '前臂', '斜方肌'
        }

        # 器械词典
        self._entity_dictionary[EntityType.EQUIPMENT] = {
            '哑铃', '杠铃', '壶铃', '弹力带', '健身房', '跑步机', '椭圆机',
            '健身球', '瑜伽垫', '引体向上杆', '双杠', '单杠', '划船机',
            '动感单车', '龙门架', '史密斯机', '腿举机', '推胸机'
        }

        # 目标词典
        self._entity_dictionary[EntityType.GOAL] = {
            '减脂', '增肌', '塑形', '耐力', '力量', '柔韧性', '平衡',
            '爆发力', '协调性', '核心稳定', '康复', '健康', '运动表现'
        }

        # 伤病词典
        self._entity_dictionary[EntityType.INJURY] = {
            '腰痛', '膝盖疼痛', '肩袖损伤', '网球肘', '腰间盘突出',
            '肌肉拉伤', '关节炎', '颈椎病', '跟腱炎', '应力性骨折'
        }

        # 同义词典
        self._synonym_dictionary = {
            '减脂': ['减肥', '瘦身', '燃脂', '减重'],
            '增肌': ['肌肉增长', '肌肉肥大', '力量训练', '肌肉塑形'],
            '胸肌': ['胸部', '胸大肌', '胸肌训练'],
            '深蹲': ['蹲腿', '蹲起', '深蹲起'],
            '卧推': ['卧推举', '胸推', '推胸'],
            '硬拉': ['硬拉举', '拉起']
        }

        logger.info(f"加载领域词典: {len(self._entity_dictionary)} 类实体")

    async def _load_intent_patterns(self) -> None:
        """加载意图模式"""
        self._intent_patterns = {
            QueryIntent.INFORMATION_SEEKING: [
                r'什么是', r'如何', r'怎么', r'怎样', r'介绍', r'解释', r'原理'
            ],
            QueryIntent.COMPARISON: [
                r'对比', r'比较', r'区别', r'差异', r'哪个好', r'优缺点'
            ],
            QueryIntent.RECOMMENDATION: [
                r'推荐', r'建议', r'哪个适合', r'选择', r'最好的'
            ],
            QueryIntent.TUTORIAL: [
                r'教程', r'步骤', r'方法', r'技巧', r'动作要领', r'正确做法'
            ],
            QueryIntent.SAFETY_CHECK: [
                r'安全', r'危险', r'禁忌', r'注意事项', r'风险', r'副作用'
            ],
            QueryIntent.PROBLEM_SOLVING: [
                r'解决', r'改善', r'纠正', r'调整', r'修复', r'处理'
            ],
            QueryIntent.PLANNING: [
                r'计划', r'安排', r'方案', r'制定', r'设计', r'课程'
            ]
        }

    async def _load_relation_patterns(self) -> None:
        """加载关系模式"""
        self._relation_patterns = [
            (r'(\w+)训练(\w+)', 'trains'),           # 训练关系
            (r'(\w+)针对(\w+)', 'targets'),          # 针对关系
            (r'(\w+)使用(\w+)', 'uses'),             # 使用关系
            (r'(\w+)适合(\w+)', 'suitable_for'),    # 适合关系
            (r'(\w+)有助于(\w+)', 'helps_with'),     # 有助于关系
            (r'(\w+)预防(\w+)', 'prevents'),         # 预防关系
            (r'(\w+)改善(\w+)', 'improves'),         # 改善关系
        ]

    async def analyze_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryAnalysis:
        """分析查询"""
        start_time = asyncio.get_event_loop().time()
        self._metrics['total_queries'] += 1

        try:
            # 1. 查询预处理
            cleaned_query = self._preprocess_query(query)

            # 2. 意图识别
            intent = self._identify_intent(cleaned_query)

            # 3. 复杂度评估
            complexity = self._assess_complexity(cleaned_query, intent)

            # 4. 实体提取
            entities = self._extract_entities(cleaned_query)

            # 5. 关系提取
            relations = self._extract_relations(cleaned_query, entities)

            # 6. 关键词提取
            keywords = self._extract_keywords(cleaned_query)

            # 7. 情感分析
            sentiment = self._analyze_sentiment(cleaned_query)

            # 8. 计算置信度
            confidence = self._calculate_confidence(intent, complexity, entities)

            # 9. 生成建议
            suggestions = self._generate_suggestions(cleaned_query, intent, entities)

            # 更新指标
            execution_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(intent, complexity, execution_time)

            analysis = QueryAnalysis(
                original_query=query,
                cleaned_query=cleaned_query,
                intent=intent,
                complexity=complexity,
                entities=entities,
                relations=relations,
                keywords=keywords,
                sentiment=sentiment,
                confidence=confidence,
                suggestions=suggestions
            )

            logger.debug(f"查询分析完成: {intent.value}, {complexity.value}, {len(entities)} 实体, {execution_time:.3f}s")
            return analysis

        except Exception as e:
            logger.error(f"查询分析失败: {e}")
            return QueryAnalysis(
                original_query=query,
                cleaned_query=query,
                intent=QueryIntent.INFORMATION_SEEKING,
                complexity=QueryComplexity.SIMPLE,
                entities=[],
                relations=[],
                keywords=[],
                sentiment='neutral',
                confidence=0.0,
                suggestions=[]
            )

    def _preprocess_query(self, query: str) -> str:
        """查询预处理"""
        # 去除多余空格
        query = re.sub(r'\s+', ' ', query.strip())

        # 去除特殊字符
        query = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', query)

        # 统一标点
        query = query.replace('？', '?').replace('！', '!')

        return query

    def _identify_intent(self, query: str) -> QueryIntent:
        """识别查询意图"""
        query_lower = query.lower()

        # 计算每个意图的匹配分数
        intent_scores = {}
        for intent, patterns in self._intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            intent_scores[intent] = score

        # 选择最高分意图
        if not intent_scores or max(intent_scores.values()) == 0:
            return QueryIntent.INFORMATION_SEEKING

        best_intent = max(intent_scores, key=intent_scores.get)
        return best_intent

    def _assess_complexity(self, query: str, intent: QueryIntent) -> QueryComplexity:
        """评估查询复杂度"""
        complexity_score = 0

        # 基于查询长度
        word_count = len(query.split())
        if word_count > 15:
            complexity_score += 2
        elif word_count > 8:
            complexity_score += 1

        # 基于实体数量
        entities = self._extract_entities(query)
        complexity_score += len(entities)

        # 基于关系数量
        relations = self._extract_relations(query, entities)
        complexity_score += len(relations) * 2

        # 基于意图类型
        intent_complexity = {
            QueryIntent.INFORMATION_SEEKING: 0,
            QueryIntent.COMPARISON: 2,
            QueryIntent.RECOMMENDATION: 2,
            QueryIntent.TUTORIAL: 1,
            QueryIntent.SAFETY_CHECK: 3,
            QueryIntent.PROBLEM_SOLVING: 3,
            QueryIntent.PLANNING: 4
        }
        complexity_score += intent_complexity.get(intent, 0)

        # 基于疑问词数量
        question_words = ['什么', '如何', '怎么', '为什么', '哪里', '哪个', 'when', 'where', 'what', 'how', 'why']
        question_count = sum(1 for word in question_words if word in query.lower())
        complexity_score += question_count

        # 转换为复杂度等级
        if complexity_score >= 6:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 3:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE

    def _extract_entities(self, query: str) -> List[Tuple[str, EntityType]]:
        """提取实体"""
        entities = []
        query_lower = query.lower()

        for entity_type, entity_set in self._entity_dictionary.items():
            for entity in entity_set:
                if entity in query_lower:
                    entities.append((entity, entity_type))

        # 去重并保持顺序
        seen = set()
        unique_entities = []
        for entity, entity_type in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append((entity, entity_type))

        return unique_entities

    def _extract_relations(
        self,
        query: str,
        entities: List[Tuple[str, EntityType]]
    ) -> List[Tuple[str, str, str]]:
        """提取关系"""
        relations = []
        entity_names = [entity for entity, _ in entities]

        for pattern, relation_type in self._relation_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    entity1 = groups[0].strip()
                    entity2 = groups[1].strip()

                    # 验证是否为已知实体
                    if entity1 in entity_names and entity2 in entity_names:
                        relations.append((entity1, relation_type, entity2))

        return relations

    def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取：去除停用词后的词汇
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}

        words = query.split()
        keywords = [word for word in words if word not in stopwords and len(word) > 1]

        return keywords

    def _analyze_sentiment(self, query: str) -> str:
        """情感分析"""
        positive_words = {'好', '棒', '优秀', '有效', '适合', '推荐', '喜欢', '满意', '成功', '改善'}
        negative_words = {'不好', '差', '无效', '疼痛', '受伤', '危险', '失败', '困难', '问题', '副作用'}

        positive_count = sum(1 for word in positive_words if word in query)
        negative_count = sum(1 for word in negative_words if word in query)

        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'

    def _calculate_confidence(
        self,
        intent: QueryIntent,
        complexity: QueryComplexity,
        entities: List[Tuple[str, EntityType]]
    ) -> float:
        """计算置信度"""
        confidence = 0.5  # 基础置信度

        # 意图置信度
        if intent != QueryIntent.INFORMATION_SEEKING:
            confidence += 0.1

        # 实体置信度
        confidence += min(len(entities) * 0.1, 0.3)

        # 复杂度置信度（复杂查询通常意味着更明确的需求）
        complexity_bonus = {
            QueryComplexity.SIMPLE: 0.0,
            QueryComplexity.MODERATE: 0.1,
            QueryComplexity.COMPLEX: 0.2
        }
        confidence += complexity_bonus[complexity]

        return min(confidence, 1.0)

    def _generate_suggestions(
        self,
        query: str,
        intent: QueryIntent,
        entities: List[Tuple[str, EntityType]]
    ) -> List[str]:
        """生成建议"""
        suggestions = []

        # 基于意图生成建议
        if intent == QueryIntent.INFORMATION_SEEKING:
            suggestions.append("可以尝试添加更多细节来获得更准确的信息")
        elif intent == QueryIntent.COMPARISON:
            suggestions.append("可以明确比较的维度，如效果、难度、安全性等")
        elif intent == QueryIntent.RECOMMENDATION:
            suggestions.append("可以提供更多个人情况，如健身水平、目标等")
        elif intent == QueryIntent.SAFETY_CHECK:
            suggestions.append("安全第一，如有疑问建议咨询专业教练或医生")

        # 基于实体生成建议
        if not entities:
            suggestions.append("尝试包含具体的运动名称、器械或目标")

        return suggestions

    async def expand_query(self, query: str, analysis: Optional[QueryAnalysis] = None) -> QueryExpansion:
        """查询扩展"""
        if not analysis:
            analysis = await self.analyze_query(query)

        expansion = QueryExpansion(
            synonyms=[],
            related_terms=[],
            broader_terms=[],
            narrower_terms=[],
            expanded_queries=[]
        )

        # 同义词扩展
        for word in analysis.keywords:
            if word in self._synonym_dictionary:
                expansion.synonyms.extend(self._synonym_dictionary[word])

        # 相关词扩展
        for entity, entity_type in analysis.entities:
            if entity_type == EntityType.EXERCISE:
                # 找到同类运动
                related_exercises = self._find_related_exercises(entity)
                expansion.related_terms.extend(related_exercises)

        # 生成扩展查询
        original_words = analysis.keywords
        synonym_sets = [expansion.synonyms]

        # 简单的扩展查询生成
        for synonyms in synonym_sets[:3]:  # 限制扩展数量
            for synonym in synonyms[:2]:   # 每个扩展最多2个词
                expanded_query = query.replace(synonyms[0], synonym, 1)
                if expanded_query != query:
                    expansion.expanded_queries.append(expanded_query)

        return expansion

    def _find_related_exercises(self, exercise: str) -> List[str]:
        """查找相关运动"""
        # 简化的相关运动查找
        exercise_groups = {
            '胸部': ['卧推', '俯卧撑', '飞鸟', '双杠臂屈伸'],
            '背部': ['引体向上', '划船', '硬拉', '下拉'],
            '腿部': ['深蹲', '腿举', '弓步蹲', '臀桥'],
            '肩部': ['推举', '侧平举', '前平举', '耸肩'],
            '手臂': ['弯举', '臂屈伸', '锤式弯举'],
            '核心': ['平板支撑', '卷腹', '俄罗斯转体', '登山跑']
        }

        for muscle, exercises in exercise_groups.items():
            if exercise in exercises:
                return [ex for ex in exercises if ex != exercise]

        return []

    def _update_metrics(
        self,
        intent: QueryIntent,
        complexity: QueryComplexity,
        execution_time: float
    ) -> None:
        """更新指标"""
        # 更新意图分布
        self._metrics['intent_distribution'][intent.value] = (
            self._metrics['intent_distribution'].get(intent.value, 0) + 1
        )

        # 更新复杂度分布
        self._metrics['complexity_distribution'][complexity.value] = (
            self._metrics['complexity_distribution'].get(complexity.value, 0) + 1
        )

        # 更新平均时间
        total_queries = self._metrics['total_queries']
        current_avg = self._metrics['average_analysis_time']
        self._metrics['average_analysis_time'] = (
            (current_avg * (total_queries - 1) + execution_time) / total_queries
        )

    def get_metrics(self) -> Dict[str, Any]:
        """获取分析器指标"""
        return {
            'total_queries': self._metrics['total_queries'],
            'average_analysis_time': self._metrics['average_analysis_time'],
            'intent_distribution': self._metrics['intent_distribution'],
            'complexity_distribution': self._metrics['complexity_distribution'],
            'dictionary_sizes': {
                entity_type: len(entities)
                for entity_type, entities in self._entity_dictionary.items()
            }
        }


# 导出
__all__ = [
    'QueryAnalyzer',
    'QueryAnalysis',
    'QueryExpansion',
    'QueryIntent',
    'EntityType'
]