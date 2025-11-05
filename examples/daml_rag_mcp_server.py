#!/usr/bin/env python3
"""
玉珍健身框架 MCP服务器示例
基于三层检索系统的通用MCP工具集成

这个示例展示了如何创建一个精良的MCP服务器，
集成玉珍健身框架的三层检索系统，为各种应用提供智能检索能力。

作者：薛小川 (Xue Xiaochuan)
版本：v1.0.0
日期：2025-11-05
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# 导入玉珍健身框架核心组件
from daml_rag.core import DAMLRAGFramework
from daml_rag.config import DAMLRAGConfig
from daml_rag_retrieval.vector.qdrant import QdrantConfig
from daml_rag_retrieval.knowledge.neo4j import Neo4jConfig
from daml_rag_retrieval.three_tier import ThreeTierRetriever, RetrievalRequest
from daml_rag.models import Document

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 数据模型
# ============================================================================

@dataclass
class MCPTool:
    """MCP工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: callable


class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., description="用户查询")
    domain: Optional[str] = Field("general", description="领域")
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    retrieval_method: Optional[str] = Field("three_tier", description="检索方法")
    top_k: Optional[int] = Field(10, description="检索数量")
    filters: Optional[Dict[str, Any]] = Field(None, description="过滤条件")


class QueryResponse(BaseModel):
    """查询响应模型"""
    answer: str = Field(..., description="AI生成的回答")
    sources: List[Dict[str, Any]] = Field(..., description="检索到的来源")
    retrieval_metadata: Dict[str, Any] = Field(..., description="检索元数据")
    execution_time: float = Field(..., description="执行时间")
    model_used: str = Field(..., description="使用的模型")


class FeedbackRequest(BaseModel):
    """反馈请求模型"""
    session_id: str = Field(..., description="会话ID")
    query: str = Field(..., description="原查询")
    answer: str = Field(..., description="原回答")
    user_rating: int = Field(..., ge=1, le=5, description="用户评分(1-5)")
    user_feedback: Optional[str] = Field(None, description="用户反馈")
    improvement_suggestions: Optional[str] = Field(None, description="改进建议")


# ============================================================================
# 玉珍健身 MCP服务器核心类
# ============================================================================

class DAMLRAGMCPServer:
    """
    玉珍健身框架MCP服务器

    集成三层检索系统，提供智能检索和问答能力：
    1. 向量检索层 (Qdrant/FAISS)
    2. 知识图谱层 (Neo4j)
    3. 规则过滤层 (领域规则)
    """

    def __init__(self, config: DAMLRAGConfig):
        self.config = config
        self.framework: Optional[DAMLQRAGFramework] = None
        self.tools: Dict[str, MCPTool] = {}
        self.feedback_store: List[Dict[str, Any]] = []

    async def initialize(self) -> None:
        """初始化MCP服务器"""
        logger.info("🚀 正在初始化玉珍健身 MCP服务器...")

        try:
            # 初始化玉珍健身框架
            self.framework = DAMLRAGFramework(self.config)
            await self.framework.initialize()

            # 注册MCP工具
            await self._register_tools()

            logger.info("✅ 玉珍健身 MCP服务器初始化完成")

        except Exception as e:
            logger.error(f"❌ MCP服务器初始化失败: {str(e)}")
            raise

    async def _register_tools(self) -> None:
        """注册MCP工具"""

        # 工具1: 智能问答
        self.tools["intelligent_qa"] = MCPTool(
            name="intelligent_qa",
            description="基于三层检索系统的智能问答，提供高质量、基于权威数据的回答",
            parameters={
                "query": {"type": "string", "description": "用户问题"},
                "domain": {"type": "string", "description": "问题领域"},
                "user_id": {"type": "string", "description": "用户ID"},
                "top_k": {"type": "integer", "description": "检索数量"}
            },
            function=self._intelligent_qa
        )

        # 工具2: 文档检索
        self.tools["document_retrieval"] = MCPTool(
            name="document_retrieval",
            description="从知识库中检索相关文档，支持向量检索、图谱查询和规则过滤",
            parameters={
                "query": {"type": "string", "description": "检索查询"},
                "retrieval_method": {"type": "string", "description": "检索方法"},
                "filters": {"type": "object", "description": "过滤条件"},
                "top_k": {"type": "integer", "description": "检索数量"}
            },
            function=self._document_retrieval
        )

        # 工具3: 知识图谱查询
        self.tools["knowledge_graph_query"] = MCPTool(
            name="knowledge_graph_query",
            description="查询知识图谱中的实体关系，提供深度推理能力",
            parameters={
                "entities": {"type": "array", "description": "实体列表"},
                "relationship_types": {"type": "array", "description": "关系类型"},
                "max_depth": {"type": "integer", "description": "查询深度"}
            },
            function=self._knowledge_graph_query
        )

        # 工具4: 个性化推荐
        self.tools["personalized_recommendation"] = MCPTool(
            name="personalized_recommendation",
            description="基于用户历史和偏好的个性化内容推荐",
            parameters={
                "user_id": {"type": "string", "description": "用户ID"},
                "recommendation_type": {"type": "string", "description": "推荐类型"},
                "context": {"type": "object", "description": "上下文信息"}
            },
            function=self._personalized_recommendation
        )

        # 工具5: 质量评估
        self.tools["quality_assessment"] = MCPTool(
            name="quality_assessment",
            description="评估回答质量并提供改进建议",
            parameters={
                "query": {"type": "string", "description": "原始查询"},
                "answer": {"type": "string", "description": "生成的回答"},
                "sources": {"type": "array", "description": "来源文档"}
            },
            function=self._quality_assessment
        )

        logger.info(f"✅ 已注册 {len(self.tools)} 个MCP工具")

    # ========================================================================
    # MCP工具实现
    # ========================================================================

    async def _intelligent_qa(self, **kwargs) -> Dict[str, Any]:
        """智能问答工具"""
        query = kwargs.get("query")
        domain = kwargs.get("domain", "general")
        user_id = kwargs.get("user_id")
        top_k = kwargs.get("top_k", 10)

        if not query:
            raise HTTPException(status_code=400, detail="查询不能为空")

        try:
            # 使用玉珍健身框架处理查询
            result = await self.framework.process_query(
                query=query,
                context={
                    "domain": domain,
                    "user_id": user_id,
                    "tool_name": "intelligent_qa"
                },
                user_id=user_id
            )

            return {
                "answer": result.response,
                "sources": [
                    {
                        "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                        "metadata": doc.metadata,
                        "score": doc.metadata.get("score", 0.0)
                    }
                    for doc in result.sources
                ],
                "retrieval_metadata": result.metadata,
                "execution_time": result.execution_time,
                "model_used": result.model_used,
                "tool_used": "intelligent_qa"
            }

        except Exception as e:
            logger.error(f"智能问答失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"智能问答失败: {str(e)}")

    async def _document_retrieval(self, **kwargs) -> Dict[str, Any]:
        """文档检索工具"""
        query = kwargs.get("query")
        retrieval_method = kwargs.get("retrieval_method", "three_tier")
        filters = kwargs.get("filters", {})
        top_k = kwargs.get("top_k", 10)

        if not query:
            raise HTTPException(status_code=400, detail="查询不能为空")

        try:
            # 创建检索请求
            retrieval_request = RetrievalRequest(
                query=query,
                top_k=top_k,
                filters=filters,
                user_id=kwargs.get("user_id")
            )

            # 执行检索
            if retrieval_method == "three_tier" and self.framework.three_tier_retriever:
                result = await self.framework.three_tier_retriever.retrieve(retrieval_request)
                documents = result.final_results.documents if result.final_results else []
                metadata = {
                    "vector_count": len(result.vector_results.documents) if result.vector_results else 0,
                    "knowledge_count": len(result.knowledge_results.documents) if result.knowledge_results else 0,
                    "rules_count": len(result.final_results.documents) if result.final_results else 0,
                    "execution_time": result.total_execution_time
                }
            else:
                # 使用传统检索
                retrieval_result = await self.framework.retriever.retrieve(query, top_k=top_k)
                documents = retrieval_result.documents
                metadata = {"method": "traditional", "count": len(documents)}

            return {
                "documents": [
                    {
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "score": doc.metadata.get("score", 0.0)
                    }
                    for doc in documents
                ],
                "metadata": metadata,
                "tool_used": "document_retrieval"
            }

        except Exception as e:
            logger.error(f"文档检索失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"文档检索失败: {str(e)}")

    async def _knowledge_graph_query(self, **kwargs) -> Dict[str, Any]:
        """知识图谱查询工具"""
        entities = kwargs.get("entities", [])
        relationship_types = kwargs.get("relationship_types", [])
        max_depth = kwargs.get("max_depth", 2)

        if not entities:
            raise HTTPException(status_code=400, detail="实体列表不能为空")

        try:
            # 检查知识图谱检索器是否可用
            if not self.framework.knowledge_retriever:
                raise HTTPException(status_code=503, detail="知识图谱服务不可用")

            # 构建Cypher查询
            cypher_query = self._build_cypher_query(entities, relationship_types, max_depth)

            # 执行查询
            result = await self.framework.knowledge_retriever.execute_cypher(
                query=cypher_query,
                parameters={"entities": entities}
            )

            return {
                "nodes": result.get("nodes", []),
                "relationships": result.get("relationships", []),
                "cypher_query": cypher_query,
                "execution_time": result.get("execution_time", 0.0),
                "tool_used": "knowledge_graph_query"
            }

        except Exception as e:
            logger.error(f"知识图谱查询失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"知识图谱查询失败: {str(e)}")

    async def _personalized_recommendation(self, **kwargs) -> Dict[str, Any]:
        """个性化推荐工具"""
        user_id = kwargs.get("user_id")
        recommendation_type = kwargs.get("recommendation_type", "general")
        context = kwargs.get("context", {})

        if not user_id:
            raise HTTPException(status_code=400, detail="用户ID不能为空")

        try:
            # 基于用户历史生成推荐查询
            recommendation_query = self._generate_recommendation_query(
                user_id, recommendation_type, context
            )

            # 执行推荐检索
            result = await self.framework.process_query(
                query=recommendation_query,
                context={
                    "user_id": user_id,
                    "recommendation_type": recommendation_type,
                    "tool_name": "personalized_recommendation"
                }
            )

            return {
                "recommendations": [
                    {
                        "content": doc.content[:300] + "..." if len(doc.content) > 300 else doc.content,
                        "score": doc.metadata.get("score", 0.0),
                        "category": doc.metadata.get("category", "general")
                    }
                    for doc in result.sources[:5]
                ],
                "user_id": user_id,
                "recommendation_type": recommendation_type,
                "tool_used": "personalized_recommendation"
            }

        except Exception as e:
            logger.error(f"个性化推荐失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"个性化推荐失败: {str(e)}")

    async def _quality_assessment(self, **kwargs) -> Dict[str, Any]:
        """质量评估工具"""
        query = kwargs.get("query")
        answer = kwargs.get("answer")
        sources = kwargs.get("sources", [])

        if not all([query, answer]):
            raise HTTPException(status_code=400, detail="查询和回答不能为空")

        try:
            # 简单的质量评估逻辑
            quality_score = self._calculate_quality_score(query, answer, sources)

            # 生成改进建议
            suggestions = self._generate_improvement_suggestions(
                quality_score, query, answer, sources
            )

            return {
                "quality_score": quality_score,
                "assessment": self._get_quality_assessment(quality_score),
                "suggestions": suggestions,
                "metrics": {
                    "answer_length": len(answer),
                    "source_count": len(sources),
                    "relevance_score": quality_score * 0.8  # 简化的相关性评分
                },
                "tool_used": "quality_assessment"
            }

        except Exception as e:
            logger.error(f"质量评估失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"质量评估失败: {str(e)}")

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _build_cypher_query(self, entities: List[str], relationship_types: List[str], max_depth: int) -> str:
        """构建Cypher查询"""
        entity_filter = " OR ".join([f"e.name = '{entity}'" for entity in entities])
        rel_filter = " AND ".join([f"type(r) = '{rel_type}'" for rel_type in relationship_types]) if relationship_types else ""

        query = f"""
        MATCH (e {{name: '{entities[0]}'}})
        {'-' * max_depth}
        RETURN e, r, nodes, relationships
        LIMIT 50
        """

        return query

    def _generate_recommendation_query(self, user_id: str, recommendation_type: str, context: Dict) -> str:
        """生成推荐查询"""
        base_queries = {
            "fitness": f"为用户 {user_id} 推荐适合的健身计划",
            "nutrition": f"为用户 {user_id} 推荐营养建议",
            "exercise": f"为用户 {user_id} 推荐训练动作",
            "general": f"为用户 {user_id} 推荐通用建议"
        }

        base_query = base_queries.get(recommendation_type, base_queries["general"])

        if context:
            context_str = ", ".join([f"{k}: {v}" for k, v in context.items()])
            return f"{base_query}，考虑上下文：{context_str}"

        return base_query

    def _calculate_quality_score(self, query: str, answer: str, sources: List[Dict]) -> float:
        """计算质量评分"""
        # 简化的质量评分逻辑
        length_score = min(1.0, len(answer) / 200)  # 长度评分
        source_score = min(1.0, len(sources) / 3)   # 来源评分
        relevance_score = 0.8  # 简化的相关性评分

        return (length_score + source_score + relevance_score) / 3

    def _generate_improvement_suggestions(self, quality_score: float, query: str, answer: str, sources: List[Dict]) -> List[str]:
        """生成改进建议"""
        suggestions = []

        if quality_score < 0.6:
            suggestions.append("回答可以更详细一些")

        if len(sources) < 2:
            suggestions.append("建议引用更多的权威来源")

        if len(answer) < 100:
            suggestions.append("回答可以更加具体和详细")

        if quality_score > 0.8:
            suggestions.append("回答质量很好，继续保持")

        return suggestions

    def _get_quality_assessment(self, quality_score: float) -> str:
        """获取质量评估"""
        if quality_score >= 0.8:
            return "优秀"
        elif quality_score >= 0.6:
            return "良好"
        elif quality_score >= 0.4:
            return "一般"
        else:
            return "需要改进"

    async def submit_feedback(self, feedback: FeedbackRequest) -> Dict[str, Any]:
        """提交用户反馈"""
        try:
            feedback_data = {
                "timestamp": asyncio.get_event_loop().time(),
                "session_id": feedback.session_id,
                "query": feedback.query,
                "answer": feedback.answer,
                "user_rating": feedback.user_rating,
                "user_feedback": feedback.user_feedback,
                "improvement_suggestions": feedback.improvement_suggestions
            }

            self.feedback_store.append(feedback_data)

            logger.info(f"收到用户反馈: 评分={feedback.user_rating}, 会话={feedback.session_id}")

            # TODO: 将反馈存储到数据库或用于模型微调

            return {
                "status": "success",
                "message": "反馈已提交，感谢您的评价！",
                "feedback_id": len(self.feedback_store)
            }

        except Exception as e:
            logger.error(f"提交反馈失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"提交反馈失败: {str(e)}")

    async def get_statistics(self) -> Dict[str, Any]:
        """获取服务器统计信息"""
        if not self.framework:
            return {"status": "not_initialized"}

        # 获取框架统计
        framework_stats = await self.framework.get_detailed_framework_stats()

        # 获取MCP服务器统计
        mcp_stats = {
            "total_tools": len(self.tools),
            "available_tools": list(self.tools.keys()),
            "total_feedback": len(self.feedback_store),
            "average_rating": sum(f["user_rating"] for f in self.feedback_store) / len(self.feedback_store) if self.feedback_store else 0
        }

        return {
            "framework_stats": framework_stats,
            "mcp_stats": mcp_stats,
            "server_status": "running"
        }

    async def shutdown(self) -> None:
        """关闭MCP服务器"""
        if self.framework:
            await self.framework.shutdown()
        logger.info("玉珍健身 MCP服务器已关闭")


# ============================================================================
# FastAPI应用
# ============================================================================

# 全局服务器实例
mcp_server: Optional[DAMLQRAGMCPServer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global mcp_server

    # 启动时初始化
    config = create_sample_config()
    mcp_server = DAMLRAGMCPServer(config)
    await mcp_server.initialize()

    yield

    # 关闭时清理
    if mcp_server:
        await mcp_server.shutdown()


# 创建FastAPI应用
app = FastAPI(
    title="玉珍健身 MCP Server",
    description="基于三层检索系统的智能MCP服务器",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API路由
# ============================================================================

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "玉珍健身 MCP Server",
        "description": "基于三层检索系统的智能MCP服务器",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    if not mcp_server or not mcp_server.framework:
        raise HTTPException(status_code=503, detail="MCP服务器未初始化")

    # 检查框架健康状态
    framework_health = await mcp_server.framework.health_check()

    return {
        "status": "healthy",
        "framework_health": framework_health,
        "tools_available": list(mcp_server.tools.keys())
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """执行查询"""
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP服务器不可用")

    # 使用智能问答工具
    result = await mcp_server._intelligent_qa(
        query=request.query,
        domain=request.domain,
        user_id=request.user_id,
        top_k=request.top_k
    )

    return QueryResponse(**result)


@app.post("/tools/{tool_name}")
async def use_tool(tool_name: str, parameters: Dict[str, Any]):
    """使用特定MCP工具"""
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP服务器不可用")

    if tool_name not in mcp_server.tools:
        raise HTTPException(status_code=404, detail=f"工具 '{tool_name}' 不存在")

    tool = mcp_server.tools[tool_name]
    result = await tool.function(**parameters)

    return {
        "tool_name": tool_name,
        "result": result,
        "timestamp": asyncio.get_event_loop().time()
    }


@app.get("/tools")
async def list_tools():
    """列出所有可用工具"""
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP服务器不可用")

    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in mcp_server.tools.values()
        ]
    }


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """提交用户反馈"""
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP服务器不可用")

    return await mcp_server.submit_feedback(feedback)


@app.get("/statistics")
async def get_statistics():
    """获取统计信息"""
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP服务器不可用")

    return await mcp_server.get_statistics()


# ============================================================================
# 配置创建函数
# ============================================================================

def create_sample_config() -> DAMLRAGConfig:
    """创建示例配置"""
    return DAMLRAGConfig(
        domain="general",
        environment="development",
        debug=True,

        # 向量检索配置
        vector_config=QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="mcp_knowledge",
            vector_size=768,
            distance="Cosine"
        ),

        # 知识图谱配置
        knowledge_config=Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="鐜夌弽鍋ヨ韩_2024"
        ),

        # 三层检索权重
        vector_weight=0.4,
        knowledge_weight=0.4,
        rules_weight=0.2,

        # 其他配置
        cache_enabled=True,
        cache_ttl=300,
        top_k=10,
        score_threshold=0.0
    )


# ============================================================================
# 主函数
# ============================================================================

async def main():
    """主函数"""
    config = create_sample_config()

    # 创建并初始化MCP服务器
    server = DAMLRAGMCPServer(config)
    await server.initialize()

    # 测试工具
    tools = list(server.tools.keys())
    print(f"✅ 已注册 {len(tools)} 个MCP工具: {', '.join(tools)}")

    # 测试查询
    result = await server._intelligent_qa(
        query="什么是深蹲的正确动作要领？",
        domain="fitness",
        user_id="test_user",
        top_k=5
    )

    print(f"🎯 测试查询结果: {result['answer'][:100]}...")

    await server.shutdown()


if __name__ == "__main__":
    print("🚀 启动玉珍健身 MCP服务器...")
    print("📋 前置条件:")
    print("   - Qdrant服务运行在 localhost:6333")
    print("   - Neo4j服务运行在 localhost:7474")
    print("   - 已导入知识数据")
    print()

    # 运行主函数
    asyncio.run(main())

    # 或启动HTTP服务器
    # uvicorn.run(
    #     "daml_rag_mcp_server:app",
    #     host="0.0.0.0",
    #     port=8002,
    #     reload=True
    # )