"""
健身领域适配器实现
"""

import asyncio
from typing import Dict, Any, List, Optional
import json

from ..base import DomainAdapter
from ..models import Entity, Relation, IKnowledgeGraphRetriever
from .intent_matcher import FitnessIntentMatcher
from .knowledge import FitnessKnowledgeGraphBuilder
from .tools.registry import FitnessToolRegistry


class FitnessDomainAdapter(DomainAdapter):
    """健身领域适配器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("fitness", config)
        self.tool_registry = FitnessToolRegistry()
        self.intent_matcher = FitnessIntentMatcher()
        self.knowledge_graph_builder = FitnessKnowledgeGraphBuilder(config)
        self._initialized = False

        # MCP服务器配置
        self.mcp_servers = config.get('mcp_servers', [])
        self.mcp_connections = {}

    async def initialize(self) -> None:
        """初始化健身领域组件"""
        if self._initialized:
            return

        try:
            print("🏋️ 初始化健身领域适配器...")

            # 初始化知识图谱构建器
            await self.knowledge_graph_builder.initialize()
            print("✅ 知识图谱构建器初始化完成")

            # 注册MCP工具
            await self._register_mcp_tools()
            print("✅ MCP工具注册完成")

            # 初始化意图匹配器
            await self.intent_matcher.initialize()
            print("✅ 意图匹配器初始化完成")

            self._initialized = True
            print("🎉 健身领域适配器初始化完成")

        except Exception as e:
            raise RuntimeError(f"健身领域适配器初始化失败: {str(e)}")

    def get_entity_types(self) -> List[str]:
        """健身领域实体类型"""
        return [
            "Exercise",      # 动作
            "User",          # 用户
            "Equipment",     # 器械
            "Muscle",        # 肌群
            "Program",       # 训练计划
            "Nutrition",     # 营养
            "Injury"         # 损伤
        ]

    def get_relation_types(self) -> List[str]:
        """健身领域关系类型"""
        return [
            "TARGETS",       # 目标关系
            "REQUIRES",      # 需求关系
            "CONTAINS",      # 包含关系
            "PREVENTS",      # 预防关系
            "RECOMMENDS",   # 推荐关系
            "CONTRADICTS"   # 矛盾关系
        ]

    def get_tool_registry(self) -> Dict[str, 'IMCPTool']:
        """获取健身工具注册表"""
        return self.tool_registry.get_all_tools()

    def get_intent_patterns(self) -> List[str]:
        """健身领域意图模式"""
        return self.intent_matcher.get_patterns()

    async def build_knowledge_graph(self, data_source: str) -> IKnowledgeGraphRetriever:
        """构建健身领域知识图谱"""
        return await self.knowledge_graph_builder.build_graph(data_source)

    def get_domain_config(self) -> Dict[str, Any]:
        """获取领域特定配置"""
        return self.config.get('domain_specific', {})

    async def _register_mcp_tools(self) -> None:
        """注册MCP工具"""
        for server_config in self.mcp_servers:
            try:
                # 这里应该根据MCP协议连接到服务器
                # 暂时跳过实际连接
                print(f"  🔌 注册MCP服务器: {server_config.get('name', 'unknown')}")
            except Exception as e:
                print(f"  ⚠️  MCP服务器注册失败: {str(e)}")

    async def search_tools(self, query: str) -> List['IMCPTool']:
        """搜索工具"""
        return await self.tool_registry.search_tools(query)

    async def recommend_tools(self, intent: Dict[str, Any]) -> List[str]:
        """推荐工具"""
        return await self.intent_matcher.suggest_tools(intent)

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """提取实体"""
        return await self.intent_matcher.extract_entities(text)

    def get_entity_schema(self, entity_type: str) -> Dict[str, Any]:
        """获取实体模式"""
        schemas = {
            "Exercise": {
                "name": {"type": "string", "description": "动作名称"},
                "type": {"type": "string", "enum": ["力量", "有氧", "柔韧", "功能性"], "description": "动作类型"},
                "difficulty": {"type": "string", "enum": ["初级", "中级", "高级"], "description": "动作难度"},
                "equipment": {"type": "array", "items": {"type": "string"}, "description": "所需器械"},
                "target_muscles": {"type": "array", "items": {"type": "string"}, "description": "目标肌群"},
                "instructions": {"type": "string", "description": "动作要领"},
                "tips": {"type": "array", "items": {"type": "string"}, "description": "动作要点"}
            },
            "User": {
                "name": {"type": "string", "description": "用户名称"},
                "age": {"type": "integer", "minimum": 10, "maximum": 100, "description": "年龄"},
                "gender": {"type": "string", "enum": ["男", "女", "其他"], "description": "性别"},
                "weight": {"type": "number", "minimum": 30, "maximum": 300, "description": "体重(kg)"},
                "height": {"type": "number", "minimum": 100, "maximum": 250, "description": "身高(cm)"},
                "fitness_level": {"type": "string", "enum": ["初级", "中级", "高级"], "description": "健身水平"},
                "goals": {"type": "array", "items": {"type": "string"}, "description": "健身目标"},
                "injuries": {"type": "array", "items": {"type": "string"}, "description": "损伤史"},
                "preferences": {"type": "object", "description": "训练偏好"}
            },
            "Equipment": {
                "name": {"type": "string", "description": "器械名称"},
                "type": {"type": "string", "enum": ["力量器械", "有氧器械", "功能性器械", "自重"], "description": "器械类型"},
                "muscle_groups": {"type": "array", "items": {"type": "string"}, "description": "训练肌群"},
                "difficulty_level": {"type": "string", "enum": ["初级", "中级", "高级"], "description": "使用难度"},
                "availability": {"type": "string", "enum": ["家用", "商用", "户外"], "description": "可用场景"}
            }
        }
        return schemas.get(entity_type, {})

    def get_tool_categories(self) -> List[str]:
        """获取工具分类"""
        return [
            "Exercise",      # 动作工具 (4个)
            "Training",       # 训练工具 (4个)
            "Rehabilitation", # 康复工具 (4个)
            "Integrated",     # 综合工具 (3个)
            "Utility",        # 实用工具 (4个)
            "Nutrition"       # 营养工具 (4个)
        ]

    async def get_tool_examples(self, tool_name: str) -> List[Dict[str, Any]]:
        """获取工具使用示例"""
        examples = {
            "exercise_search": [
                {
                    "query": "我想练胸肌",
                    "expected_tools": ["exercise_search", "exercise_recommend"],
                    "explanation": "用户想要寻找胸部训练动作"
                },
                {
                    "query": "深蹲怎么做",
                    "expected_tools": ["exercise_details"],
                    "explanation": "用户询问具体动作要领"
                }
            ],
            "training_capacity": [
                {
                    "query": "我该怎么安排训练容量",
                    "expected_tools": ["training_capacity"],
                    "explanation": "用户询问训练容量规划"
                }
            ],
            "personalized_program": [
                {
                    "query": "帮我制定一个增肌计划",
                    "expected_tools": ["personalized_program", "exercise_recommend"],
                    "explanation": "用户要求制定个性化训练计划"
                }
            ]
        }
        return examples.get(tool_name, [])

    async def validate_user_input(self, input_text: str) -> Dict[str, Any]:
        """验证用户输入"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "normalized_text": input_text.strip(),
            "detected_entities": [],
            "intent_confidence": 0.0
        }

        # 基本清理
        normalized = input_text.strip()
        if not normalized:
            validation_result["is_valid"] = False
            validation_result["errors"].append("输入不能为空")
            return validation_result

        # 实体提取
        try:
            entities = await self.extract_entities(normalized)
            validation_result["detected_entities"] = entities
        except Exception as e:
            validation_result["warnings"].append(f"实体提取失败: {str(e)}")

        # 意图检测
        try:
            intent_result = await self.intent_matcher.match_intent(normalized)
            validation_result["intent_confidence"] = intent_result.get("confidence", 0.0)
        except Exception as e:
            validation_result["warnings"].append(f"意图检测失败: {str(e)}")

        return validation_result

    async def get_statistics(self) -> Dict[str, Any]:
        """获取领域统计信息"""
        try:
            stats = {
                "domain": "fitness",
                "version": "1.0.0",
                "initialized": self._initialized,
                "tools_count": len(self.tool_registry.get_all_tools()),
                "mcp_servers": len(self.mcp_servers),
                "active_connections": len(self.mcp_connections),
                "intent_patterns": len(self.get_intent_patterns()),
                "entity_types": len(self.get_entity_types()),
                "relation_types": len(self.get_relation_types())
            }

            # 知识图谱统计
            if hasattr(self.knowledge_graph_builder, 'get_statistics'):
                kg_stats = await self.knowledge_graph_builder.get_statistics()
                stats.update(kg_stats)

            return stats

        except Exception as e:
            return {"error": f"获取统计信息失败: {str(e)}"}

    async def health_check(self) -> Dict[str, Any]:
        """领域适配器健康检查"""
        health_status = {
            "adapter_healthy": self._initialized,
            "components": {},
            "overall_status": "healthy"
        }

        components_to_check = [
            (self.tool_registry, "tool_registry"),
            (self.intent_matcher, "intent_matcher"),
            (self.knowledge_graph_builder, "knowledge_graph_builder"),
        ]

        unhealthy_count = 0
        for component, name in components_to_check:
            try:
                if hasattr(component, 'health_check'):
                    is_healthy = await component.health_check()
                    health_status["components"][name] = "healthy" if is_healthy else "unhealthy"
                    if not is_healthy:
                        unhealthy_count += 1
                else:
                    health_status["components"][name] = "unknown"
            except Exception as e:
                health_status["components"][name] = f"error: {str(e)}"
                unhealthy_count += 1

        if unhealthy_count > 0:
            health_status["overall_status"] = "degraded" if unhealthy_count < len(components_to_check) else "unhealthy"

        return health_status

    def get_help_topics(self) -> List[Dict[str, Any]]:
        """获取帮助主题"""
        return [
            {
                "topic": "动作搜索",
                "description": "如何搜索和了解健身动作",
                "examples": [
                    "我想练胸肌",
                    "深蹲怎么做",
                    "推荐几个背部训练动作"
                ]
            },
            {
                "topic": "训练计划",
                "description": "如何制定个性化训练计划",
                "examples": [
                    "帮我制定一个增肌计划",
                    "我要减脂该怎么训练",
                    "力量训练计划推荐"
                ]
            },
            {
                "topic": "营养建议",
                "description": "获取营养和饮食指导",
                "examples": [
                    "增肌该吃什么",
                    "计算我的TDEE",
                    "减脂期营养建议"
                ]
            },
            {
                "topic": "损伤康复",
                "description": "运动损伤康复指导",
                "examples": [
                    "膝盖受伤后怎么练",
                    "腰部不适的训练替代方案",
                    "康复训练建议"
                ]
            }
        ]

    async def cleanup(self) -> None:
        """清理资源"""
        if not self._initialized:
            return

        try:
            print("🧹 清理健身领域适配器资源...")

            # 关闭MCP连接
            for connection_name, connection in self.mcp_connections.items():
                try:
                    if hasattr(connection, 'close'):
                        await connection.close()
                    print(f"  🔌 关闭连接: {connection_name}")
                except Exception as e:
                    print(f"  ⚠️  关闭连接失败 {connection_name}: {str(e)}")

            # 清理工具注册表
            if hasattr(self.tool_registry, 'cleanup'):
                await self.tool_registry.cleanup()

            # 清理意图匹配器
            if hasattr(self.intent_matcher, 'cleanup'):
                await self.intent_matcher.cleanup()

            # 清理知识图谱构建器
            if hasattr(self.knowledge_graph_builder, 'cleanup'):
                await self.knowledge_graph_builder.cleanup()

            self._initialized = False
            print("✅ 健身领域适配器资源清理完成")

        except Exception as e:
            print(f"❌ 健身领域适配器清理失败: {str(e)}")