#!/usr/bin/env python3
"""
DAML-RAG MCP客户端使用示例
展示如何与DAML-RAG MCP服务器进行交互

这个示例演示了：
1. MCP客户端的基本使用方法
2. 各种MCP工具的调用方式
3. 错误处理和重试机制
4. 异步批量处理

作者：BUILD_BODY Team
版本：v1.0.0
日期：2025-11-05
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

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
class MCPClientConfig:
    """MCP客户端配置"""
    base_url: str = "http://localhost:8002"
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class QueryRequest:
    """查询请求"""
    query: str
    domain: Optional[str] = "general"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    top_k: int = 10
    filters: Optional[Dict[str, Any]] = None


@dataclass
class FeedbackRequest:
    """反馈请求"""
    session_id: str
    query: str
    answer: str
    user_rating: int
    user_feedback: Optional[str] = None
    improvement_suggestions: Optional[str] = None


# ============================================================================
# MCP客户端类
# ============================================================================

class DAMLRAGMCPClient:
    """DAML-RAG MCP客户端"""

    def __init__(self, config: MCPClientConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.available_tools: List[str] = []

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

    async def connect(self) -> None:
        """连接到MCP服务器"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            headers={"Content-Type": "application/json"}
        )

        # 检查服务器健康状态
        health = await self.check_health()
        if not health.get("status") == "healthy":
            raise ConnectionError(f"MCP服务器不健康: {health}")

        # 获取可用工具列表
        tools_info = await self.list_tools()
        self.available_tools = [tool["name"] for tool in tools_info["tools"]]
        logger.info(f"✅ 已连接到MCP服务器，可用工具: {', '.join(self.available_tools)}")

    async def close(self) -> None:
        """关闭连接"""
        if self.session:
            await self.session.close()
            logger.info("🔌 MCP客户端连接已关闭")

    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """发起HTTP请求"""
        if not self.session:
            raise ConnectionError("MCP客户端未连接")

        url = f"{self.config.base_url}{endpoint}"

        for attempt in range(self.config.retry_attempts):
            try:
                if method.upper() == "GET":
                    async with self.session.get(url, params=data) as response:
                        return await self._handle_response(response)
                elif method.upper() == "POST":
                    async with self.session.post(url, json=data) as response:
                        return await self._handle_response(response)
                else:
                    raise ValueError(f"不支持的HTTP方法: {method}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == self.config.retry_attempts - 1:
                    logger.error(f"请求失败，已重试{self.config.retry_attempts}次: {str(e)}")
                    raise
                else:
                    logger.warning(f"请求失败，正在重试 ({attempt + 1}/{self.config.retry_attempts}): {str(e)}")
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))  # 指数退避

    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """处理HTTP响应"""
        if response.status == 200:
            return await response.json()
        else:
            error_text = await response.text()
            raise aiohttp.ClientResponseError(
                request_info=response.request_info,
                history=response.history,
                status=response.status,
                message=error_text
            )

    # ========================================================================
    # 基础API方法
    # ========================================================================

    async def check_health(self) -> Dict[str, Any]:
        """检查服务器健康状态"""
        return await self._make_request("GET", "/health")

    async def list_tools(self) -> Dict[str, Any]:
        """列出所有可用工具"""
        return await self._make_request("GET", "/tools")

    async def get_statistics(self) -> Dict[str, Any]:
        """获取服务器统计信息"""
        return await self._make_request("GET", "/statistics")

    # ========================================================================
    # 核心功能方法
    # ========================================================================

    async def query(self, request: QueryRequest) -> Dict[str, Any]:
        """执行智能问答查询"""
        data = {
            "query": request.query,
            "domain": request.domain,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "top_k": request.top_k,
            "filters": request.filters or {}
        }
        return await self._make_request("POST", "/query", data)

    async def use_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """使用特定MCP工具"""
        if tool_name not in self.available_tools:
            raise ValueError(f"工具 '{tool_name}' 不可用。可用工具: {', '.join(self.available_tools)}")

        return await self._make_request("POST", f"/tools/{tool_name}", parameters)

    async def submit_feedback(self, feedback: FeedbackRequest) -> Dict[str, Any]:
        """提交用户反馈"""
        data = {
            "session_id": feedback.session_id,
            "query": feedback.query,
            "answer": feedback.answer,
            "user_rating": feedback.user_rating,
            "user_feedback": feedback.user_feedback,
            "improvement_suggestions": feedback.improvement_suggestions
        }
        return await self._make_request("POST", "/feedback", data)

    # ========================================================================
    # 便捷方法
    # ========================================================================

    async def intelligent_qa(self, query: str, domain: str = "general", user_id: str = None) -> Dict[str, Any]:
        """智能问答便捷方法"""
        parameters = {
            "query": query,
            "domain": domain,
            "user_id": user_id
        }
        return await self.use_tool("intelligent_qa", parameters)

    async def document_retrieval(self, query: str, retrieval_method: str = "three_tier", top_k: int = 10) -> Dict[str, Any]:
        """文档检索便捷方法"""
        parameters = {
            "query": query,
            "retrieval_method": retrieval_method,
            "top_k": top_k
        }
        return await self.use_tool("document_retrieval", parameters)

    async def knowledge_graph_query(self, entities: List[str], relationship_types: List[str] = None, max_depth: int = 2) -> Dict[str, Any]:
        """知识图谱查询便捷方法"""
        parameters = {
            "entities": entities,
            "relationship_types": relationship_types or [],
            "max_depth": max_depth
        }
        return await self.use_tool("knowledge_graph_query", parameters)

    async def personalized_recommendation(self, user_id: str, recommendation_type: str = "general", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """个性化推荐便捷方法"""
        parameters = {
            "user_id": user_id,
            "recommendation_type": recommendation_type,
            "context": context or {}
        }
        return await self.use_tool("personalized_recommendation", parameters)

    async def quality_assessment(self, query: str, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """质量评估便捷方法"""
        parameters = {
            "query": query,
            "answer": answer,
            "sources": sources
        }
        return await self.use_tool("quality_assessment", parameters)


# ============================================================================
# 示例使用函数
# ============================================================================

async def example_basic_usage():
    """基础使用示例"""
    print("🔥 基础使用示例")
    print("=" * 50)

    config = MCPClientConfig(base_url="http://localhost:8002")

    async with DAMLRAGMCPClient(config) as client:
        # 1. 检查服务器状态
        health = await client.check_health()
        print(f"📊 服务器状态: {health['status']}")
        print(f"🛠️  可用工具: {', '.join(health['tools_available'])}")

        # 2. 执行智能问答
        query = "初学者如何制定健身计划？"
        result = await client.intelligent_qa(query, domain="fitness", user_id="demo_user")

        print(f"\n❓ 查询: {query}")
        print(f"🤖 回答: {result['answer'][:200]}...")
        print(f"📚 来源数量: {len(result['sources'])}")
        print(f"⏱️  执行时间: {result['execution_time']:.2f}秒")

        # 3. 文档检索
        doc_result = await client.document_retrieval(
            query="深蹲动作要领",
            retrieval_method="three_tier",
            top_k=5
        )

        print(f"\n📄 文档检索结果:")
        for i, doc in enumerate(doc_result['documents'][:3], 1):
            print(f"  {i}. {doc['content'][:100]}...")
            print(f"     评分: {doc['score']:.3f}")


async def example_knowledge_graph():
    """知识图谱查询示例"""
    print("\n🔍 知识图谱查询示例")
    print("=" * 50)

    config = MCPClientConfig(base_url="http://localhost:8002")

    async with DAMLRAGMCPClient(config) as client:
        # 查询实体关系
        entities = ["深蹲", "股四头肌"]
        result = await client.knowledge_graph_query(
            entities=entities,
            relationship_types=["锻炼", "相关"],
            max_depth=2
        )

        print(f"🔍 查询实体: {', '.join(entities)}")
        print(f"📊 节点数量: {len(result.get('nodes', []))}")
        print(f"🔗 关系数量: {len(result.get('relationships', []))}")
        print(f"⏱️  执行时间: {result['execution_time']:.2f}秒")


async def example_personalized_recommendation():
    """个性化推荐示例"""
    print("\n🎯 个性化推荐示例")
    print("=" * 50)

    config = MCPClientConfig(base_url="http://localhost:8002")

    async with DAMLRAGMCPClient(config) as client:
        # 为用户生成推荐
        user_id = "user_123"
        result = await client.personalized_recommendation(
            user_id=user_id,
            recommendation_type="fitness",
            context={"fitness_level": "beginner", "goals": ["增肌", "减脂"]}
        )

        print(f"👤 用户ID: {user_id}")
        print(f"🎯 推荐类型: {result['recommendation_type']}")
        print(f"📋 推荐内容:")

        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec['content'][:150]}...")
            print(f"     类别: {rec['category']}, 评分: {rec['score']:.3f}")


async def example_quality_assessment():
    """质量评估示例"""
    print("\n📈 质量评估示例")
    print("=" * 50)

    config = MCPClientConfig(base_url="http://localhost:8002")

    async with DAMLRAGMCPClient(config) as client:
        # 评估回答质量
        query = "什么是HIIT训练？"
        answer = "HIIT是高强度间歇训练，通过短时间高强度运动和休息交替进行，能够有效提高心肺功能和燃脂效率。"
        sources = [{"content": "HIIT训练原理研究", "score": 0.9}]

        result = await client.quality_assessment(query, answer, sources)

        print(f"❓ 查询: {query}")
        print(f"🤖 回答: {answer}")
        print(f"📊 质量评分: {result['quality_score']:.3f}")
        print(f"📝 评估结果: {result['assessment']}")
        print(f"💡 改进建议:")
        for suggestion in result['suggestions']:
            print(f"   • {suggestion}")


async def example_feedback_system():
    """反馈系统示例"""
    print("\n💬 反馈系统示例")
    print("=" * 50)

    config = MCPClientConfig(base_url="http://localhost:8002")

    async with DAMLRAGMCPClient(config) as client:
        # 提交用户反馈
        feedback = FeedbackRequest(
            session_id="session_123",
            query="如何提高深蹲重量？",
            answer="要提高深蹲重量，需要循序渐进地增加训练负荷，同时确保动作标准。建议每周增加5-10%的重量，并配合充分的休息和营养补充。",
            user_rating=5,
            user_feedback="回答很实用，提供了具体的建议",
            improvement_suggestions="可以增加一些具体的训练计划示例"
        )

        result = await client.submit_feedback(feedback)
        print(f"✅ 反馈提交状态: {result['status']}")
        print(f"📝 反馈ID: {result['feedback_id']}")
        print(f"💬 消息: {result['message']}")


async def example_batch_processing():
    """批量处理示例"""
    print("\n📦 批量处理示例")
    print("=" * 50)

    config = MCPClientConfig(base_url="http://localhost:8002")

    async with DAMLRAGMCPClient(config) as client:
        # 批量查询
        queries = [
            "什么是蛋白质补充剂？",
            "如何进行热身运动？",
            "有氧运动和无氧运动的区别？",
            "如何预防运动损伤？"
        ]

        print(f"🔄 开始批量处理 {len(queries)} 个查询...")
        start_time = time.time()

        # 并发执行查询
        tasks = [
            client.intelligent_qa(query, domain="fitness", user_id="batch_user")
            for query in queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        print(f"⏱️  批量处理完成，耗时: {end_time - start_time:.2f}秒")

        # 显示结果
        for i, (query, result) in enumerate(zip(queries, results), 1):
            if isinstance(result, Exception):
                print(f"❌ 查询 {i} 失败: {str(result)}")
            else:
                print(f"✅ 查询 {i}: {query}")
                print(f"   回答: {result['answer'][:100]}...")
                print(f"   耗时: {result['execution_time']:.2f}秒")


async def example_error_handling():
    """错误处理示例"""
    print("\n⚠️  错误处理示例")
    print("=" * 50)

    config = MCPClientConfig(
        base_url="http://localhost:8002",
        timeout=5,
        retry_attempts=2
    )

    async with DAMLRAGMCPClient(config) as client:
        # 1. 测试不存在的工具
        try:
            await client.use_tool("nonexistent_tool", {"query": "test"})
        except ValueError as e:
            print(f"✅ 成功捕获工具不存在错误: {str(e)}")

        # 2. 测试无效查询
        try:
            await client.query(QueryRequest(query=""))  # 空查询
        except Exception as e:
            print(f"✅ 成功捕获无效查询错误: {str(e)}")

        # 3. 测试网络错误处理
        try:
            # 连接到不存在的服务器
            bad_config = MCPClientConfig(base_url="http://localhost:9999")
            async with DAMLRAGMCPClient(bad_config) as bad_client:
                await bad_client.check_health()
        except ConnectionError as e:
            print(f"✅ 成功捕获连接错误: {str(e)}")


# ============================================================================
# 主函数
# ============================================================================

async def main():
    """主函数"""
    print("🚀 DAML-RAG MCP客户端示例")
    print("基于三层检索系统的智能MCP工具集成演示")
    print("=" * 60)

    try:
        # 运行各种示例
        await example_basic_usage()
        await example_knowledge_graph()
        await example_personalized_recommendation()
        await example_quality_assessment()
        await example_feedback_system()
        await example_batch_processing()
        await example_error_handling()

        print("\n✅ 所有示例运行完成！")
        print("\n💡 提示:")
        print("   - 确保DAML-RAG MCP服务器正在运行 (python daml_rag_mcp_server.py)")
        print("   - 确保Qdrant和Neo4j服务已启动")
        print("   - 查看服务器文档: http://localhost:8002/docs")

    except Exception as e:
        print(f"\n❌ 示例运行失败: {str(e)}")
        print("💡 请检查MCP服务器是否正在运行")


if __name__ == "__main__":
    print("🔧 启动DAML-RAG MCP客户端示例...")
    print("📋 前置条件:")
    print("   - DAML-RAG MCP服务器运行在 http://localhost:8002")
    print("   - Qdrant服务运行在 localhost:6333")
    print("   - Neo4j服务运行在 localhost:7474")
    print()

    asyncio.run(main())