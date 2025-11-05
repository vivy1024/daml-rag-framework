"""
领域适配器基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import asyncio


class DomainAdapter(ABC):
    """领域适配器基类"""

    def __init__(self, domain_name: str, config: Dict[str, Any]):
        self.domain_name = domain_name
        self.config = config
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """初始化适配器"""
        pass

    @abstractmethod
    def get_entity_types(self) -> List[str]:
        """获取领域实体类型"""
        pass

    @abstractmethod
    def get_relation_types(self) -> List[str]:
        """获取领域关系类型"""
        pass

    @abstractmethod
    def get_tool_registry(self) -> Dict[str, 'IMCPTool']:
        """获取工具注册表"""
        pass

    @abstractmethod
    def get_intent_patterns(self) -> List[str]:
        """获取意图模式"""
        pass

    @abstractmethod
    async def build_knowledge_graph(self, data_source: str) -> 'IKnowledgeGraphRetriever':
        """构建领域知识图谱"""
        pass

    def get_domain_config(self) -> Dict[str, Any]:
        """获取领域特定配置"""
        return self.config.get('domain_specific', {})

    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized

    def get_domain_name(self) -> str:
        """获取领域名称"""
        return self.domain_name

    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        return self.config

    async def validate_config(self) -> List[str]:
        """验证配置"""
        errors = []

        if not self.domain_name:
            errors.append("domain_name 不能为空")

        if not isinstance(self.config, dict):
            errors.append("config 必须是字典类型")

        return errors

    async def search_tools(self, query: str) -> List['IMCPTool']:
        """搜索工具（默认实现）"""
        tools = self.get_tool_registry()
        matching_tools = []

        query_lower = query.lower()
        for tool_name, tool in tools.items():
            # 简单的关键词匹配
            if (query_lower in tool_name.lower() or
                any(keyword in tool.get_description().lower() for keyword in query_lower.split())):
                matching_tools.append(tool)

        return matching_tools

    async def recommend_tools(self, intent: Dict[str, Any]) -> List[str]:
        """推荐工具（默认实现）"""
        # 默认推荐逻辑，子类可以重写
        intent_type = intent.get('intent', '')
        confidence = intent.get('confidence', 0.0)

        if confidence < 0.5:
            return []

        # 基于意图类型的简单推荐
        tool_recommendations = {
            'search': ['search_tool'],
            'create': ['create_tool', 'save_tool'],
            'update': ['update_tool', 'edit_tool'],
            'delete': ['delete_tool'],
            'analyze': ['analyze_tool', 'report_tool'],
        }

        return tool_recommendations.get(intent_type, [])

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """提取实体（默认实现）"""
        # 默认实现返回空列表，子类应该重写
        return []

    def get_entity_schema(self, entity_type: str) -> Dict[str, Any]:
        """获取实体模式（默认实现）"""
        return {}

    def get_tool_categories(self) -> List[str]:
        """获取工具分类（默认实现）"""
        return []

    async def get_tool_examples(self, tool_name: str) -> List[Dict[str, Any]]:
        """获取工具使用示例（默认实现）"""
        return []

    async def validate_user_input(self, input_text: str) -> Dict[str, Any]:
        """验证用户输入（默认实现）"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "normalized_text": input_text.strip(),
            "detected_entities": [],
            "intent_confidence": 0.0
        }

        # 基本验证
        normalized = input_text.strip()
        if not normalized:
            validation_result["is_valid"] = False
            validation_result["errors"].append("输入不能为空")

        return validation_result

    async def get_statistics(self) -> Dict[str, Any]:
        """获取领域统计信息（默认实现）"""
        return {
            "domain": self.domain_name,
            "initialized": self._initialized,
            "tools_count": len(self.get_tool_registry()),
            "entity_types": len(self.get_entity_types()),
            "relation_types": len(self.get_relation_types())
        }

    async def health_check(self) -> Dict[str, Any]:
        """领域适配器健康检查（默认实现）"""
        return {
            "adapter_healthy": self._initialized,
            "domain": self.domain_name,
            "components": {},
            "overall_status": "healthy" if self._initialized else "uninitialized"
        }

    def get_help_topics(self) -> List[Dict[str, Any]]:
        """获取帮助主题（默认实现）"""
        return []

    async def cleanup(self) -> None:
        """清理资源（默认实现）"""
        self._initialized = False

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(domain='{self.domain_name}', initialized={self._initialized})"

    def __repr__(self) -> str:
        return self.__str__()