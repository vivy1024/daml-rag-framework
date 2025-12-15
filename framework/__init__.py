# -*- coding: utf-8 -*-
"""
DAML-RAG Framework - 通用框架

设计原则：
- 领域无关：零领域依赖，通用框架
- 用户级：用户级向量库隔离
- 可扩展：可复用到教育、医疗、健身等领域
- 分层架构：框架层 / 应用层 / 接口层严格分离

框架层职责：
- 提供通用基础设施服务（存储、检索、编排）
- 定义抽象接口和通用组件
- 不包含任何领域特定的业务逻辑

作者：BUILD_BODY Team (框架层)
版本：v2.0.0
日期：2025-11-26
"""

__version__ = "2.0.0"

import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# 导出核心模块（简化版本 - 删除过度设计）
from .storage.metadata_database import MetadataDB
from .storage.user_memory import UserMemory
from .orchestration.mcp_orchestrator import MCPOrchestrator, Task, TaskStatus
from .core.simple_framework_initializer import (
    SimpleFrameworkInitializer,
    get_framework_initializer,
    initialize_framework,
    InitResult
)

# 导出通用接口
from .interfaces import (
    IDomainAdapter,
    IResultProcessor,
    AdapterRegistry,
    ProcessorRegistry,
    ProcessingStrategy
)

# 导出基础实现
from .adapters import BaseAdapter, BaseQueryAdapter, BaseWorkflowAdapter
from .processors import BaseResultProcessor, BaseSummarizationProcessor, BaseRecommendationProcessor

# KnowledgeGraphFull延迟导入（避免导入元学习模块）
def get_knowledge_graph():
    """延迟导入KnowledgeGraphFull"""
    from .retrieval.graph.kg_full import KnowledgeGraphFull
    return KnowledgeGraphFull


# GraphRAG查询工具实例（延迟导入）
_graphrag_tool = None

def get_graphrag_tool():
    """获取GraphRAG查询工具实例（单例模式）"""
    global _graphrag_tool
    if _graphrag_tool is None:
        try:
            from .retrieval.graphrag import GraphRAGQueryTool
            # 通过框架初始化器获取kg_full
            initializer = get_framework_initializer()
            if initializer and "kg_full" in initializer.components:
                kg_full = initializer.components["kg_full"]
                _graphrag_tool = GraphRAGQueryTool(kg_full)
                logger.info("✓ GraphRAG查询工具初始化成功（使用框架kg_full）")
            else:
                # 直接创建
                from .retrieval.graph.kg_full import KnowledgeGraphFull
                import os
                kg_full = KnowledgeGraphFull(
                    neo4j_uri=os.getenv('NEO4J_URI', 'bolt://neo4j:7687'),
                    neo4j_user=os.getenv('NEO4J_USER', 'neo4j'),
                    neo4j_password=os.getenv('NEO4J_PASSWORD', 'build_body_2024'),
                    qdrant_host=os.getenv('QDRANT_HOST', 'qdrant'),
                    qdrant_port=int(os.getenv('QDRANT_PORT', '6333'))
                )
                _graphrag_tool = GraphRAGQueryTool(kg_full)
                logger.info("✓ GraphRAG查询工具初始化成功（直接创建）")
        except Exception as e:
            logger.error(f"GraphRAG查询工具初始化失败: {e}")
            _graphrag_tool = None
    return _graphrag_tool


# 统一的三层检索查询接口（框架层提供通用能力）
async def query(
    query: str,
    domain: str = "general",
    user_id: str = None,
    context: Dict[str, Any] = None
) -> "FrameworkResponse":
    try:
        # 使用新的统一检索引擎 (v2.1.0)
        from .retrieval.unified_retrieval_interface import get_compatibility_adapter

        adapter = await get_compatibility_adapter()

        logger.info(f"🚀 开始统一三层检索: {domain}/{query[:50]}...")

        # 执行统一检索
        response = await adapter.query(
            query=query,
            domain=domain,
            user_id=user_id,
            context=context
        )

        # 更新元数据
        response.metadata.update({
            "domain": domain,
            "user_id": user_id,
            "retrieval_mode": "unified_framework",
            "framework_version": "v2.1.0",
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"✓ 统一三层检索完成: {len(response.results.get('final_recommendations', []))}个结果")

        return response

    except Exception as e:
        logger.error(f"统一三层检索失败: {e}")
        # 降级到原有的GraphRAG方式
        try:
            # 获取GraphRAG查询工具
            tool = get_graphrag_tool()

            if tool:
                # 构建GraphRAG查询输入
                query_input = {
                    "query_type": "three_layer",  # 使用真正的三层检索
                    "domain": domain,
                    "query_text": query,
                    "top_k": 10,
                    "user_profile": context.get("user_profile") if context else None,
                    "filters": context.get("filters") if context else {},
                    "return_reason": True
                }

                # 执行三层检索
                logger.info(f"🔄 降级到GraphRAG模式: {domain}/{query[:50]}...")
                result = await tool.query(query_input)

                # 提取结果
                results = result.get("results", [])
                three_layer_info = result.get("three_layer_result", {})

                # 构建FrameworkResponse
                response = FrameworkResponse(
                    query=query,
                    results={
                        "final_recommendations": results,
                        "three_layer_result": three_layer_info
                    },
                    metadata={
                        "domain": domain,
                        "user_id": user_id,
                        "retrieval_mode": "graphrag_fallback",
                        "layers_executed": three_layer_info.get("layers_executed", 0),
                        "pipeline": three_layer_info.get("pipeline", ""),
                        "timestamp": datetime.now().isoformat()
                    }
                )

                logger.info(f"✓ GraphRAG降级检索完成: {len(results)}个结果")
                return response

            else:
                # 进一步降级到简化版本
                logger.warning("GraphRAG查询工具不可用，使用降级版本")
                return await _fallback_query(query, domain, user_id, context)

        except Exception as fallback_e:
            logger.error(f"所有检索方式都失败: {fallback_e}")
            # 返回错误响应
            return FrameworkResponse(
                query=query,
                results=None,
                error=f"统一检索和降级都失败: {str(e)} | {str(fallback_e)}",
                metadata={
                    "domain": domain,
                    "user_id": user_id,
                    "error": str(e),
                    "fallback_error": str(fallback_e)
                }
            )


# 降级查询（简化版本，仅Layer 1）
async def _fallback_query(
    query: str,
    domain: str = "general",
    user_id: str = None,
    context: Dict[str, Any] = None
) -> "FrameworkResponse":
    """
    降级查询（仅Layer 1语义检索）

    框架层提供通用的语义检索能力，领域特定逻辑由应用层处理。
    """
    try:
        # 直接调用GraphRAG API
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://fitness_daml_rag:8001/api/graphrag/query",
                json={
                    "query_text": query,
                    "domain": domain,
                    "query_type": "semantic_search",
                    "top_k": 10
                },
                timeout=aiohttp.ClientTimeout(total=25)
            ) as response:
                if response.status == 200:
                    result_data = await response.json()
                    results = result_data.get("data", {})
                    logger.info(f"降级模式：GraphRAG API成功返回 {len(results.get('results', []))} 个结果")
                else:
                    logger.error(f"GraphRAG API返回错误状态: {response.status}")
                    results = {"results": [], "error": f"HTTP {response.status}"}

        # 构建响应（框架层不生成领域特定内容）
        response = FrameworkResponse(
            query=query,
            results=results,
            metadata={
                "domain": domain,
                "user_id": user_id,
                "retrieval_mode": "fallback_semantic_only",
                "note": "框架层提供原始数据，领域特定逻辑由应用层处理",
                "timestamp": datetime.now().isoformat()
            }
        )

        return response

    except Exception as e:
        logger.error(f"降级查询失败: {e}")
        return FrameworkResponse(
            query=query,
            results=None,
            error=str(e),
            metadata={"error": str(e), "mode": "fallback_failed"}
        )


class FrameworkResponse:
    """框架层响应对象

    兼容ThreeLayerRetrievalResponse格式，提供统一接口。
    应用层负责根据domain字段进行业务处理。
    """
    def __init__(
        self,
        query: str,
        results: Any = None,
        error: str = None,
        metadata: Dict[str, Any] = None,
        # 新增ThreeLayerRetrievalResponse兼容字段
        answer: str = None,
        sources: List[Dict[str, Any]] = None,
        confidence: float = None,
        retrieval_summary: Dict[str, Any] = None,
        anti_hallucination_result: Dict[str, Any] = None,
        standardization_result: Dict[str, Any] = None
    ):
        self.query = query
        self.results = results  # 原始检索结果（由应用层解释）
        self.error = error
        self.metadata = metadata or {}

        # ThreeLayerRetrievalResponse兼容字段
        self.answer = answer
        self.sources = sources or []  # 默认为空列表，避免AttributeError
        self.confidence = confidence or 0.0
        self.retrieval_summary = retrieval_summary
        self.anti_hallucination_result = anti_hallucination_result
        self.standardization_result = standardization_result

        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "query": self.query,
            "answer": self.answer or "",
            "sources": self.sources,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

        # 可选字段
        if self.retrieval_summary is not None:
            result["retrieval_summary"] = self.retrieval_summary
        if self.anti_hallucination_result is not None:
            result["anti_hallucination_result"] = self.anti_hallucination_result
        if self.standardization_result is not None:
            result["standardization_result"] = self.standardization_result

        return result


# 框架层通用工具函数

def get_adapter(domain: str):
    """获取领域适配器（应用层实现）"""
    return AdapterRegistry.get_adapter(domain)


def get_processor(domain: str):
    """获取结果处理器（应用层实现）"""
    return ProcessorRegistry.get_processor(domain)


def register_adapter(domain: str, adapter_class):
    """注册领域适配器（应用层使用）"""
    AdapterRegistry.register(domain, adapter_class)


def register_processor(domain: str, processor_class):
    """注册结果处理器（应用层使用）"""
    ProcessorRegistry.register(domain, processor_class)


__all__ = [
    # 核心组件（简化版本）
    "MetadataDB",
    "UserMemory",
    "MCPOrchestrator",
    "Task",
    "TaskStatus",
    "SimpleFrameworkInitializer",
    "get_framework_initializer",
    "initialize_framework",
    "InitResult",

    # 统一检索接口（简化版本）
    "query",
    "FrameworkResponse",

    # 工具函数
    "get_adapter",
    "get_processor",
    "register_adapter",
    "register_processor",
]


logger.info("✅ DAML-RAG Framework v2.1.0 加载完成 (统一检索架构 - 领域无关)")
