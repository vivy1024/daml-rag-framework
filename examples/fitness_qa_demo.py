#!/usr/bin/env python3
"""
玉珍健身框架使用示例
演示如何使用三层检索系统进行智能问答
"""

import asyncio
import logging
from typing import Dict, Any, List

# 导入玉珍健身框架核心组件
from daml_rag.core import DAMLRAGFramework
from daml_rag.config import DAMLRAGConfig
from daml_rag.models import Document

# 导入三层检索系统组件
from daml_rag_retrieval.vector.qdrant import QdrantConfig
from daml_rag_retrieval.knowledge.neo4j import Neo4jConfig


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def create_sample_config() -> DAMLRAGConfig:
    """创建示例配置"""
    config = DAMLRAGConfig(
        domain="fitness",
        environment="development",
        debug=True,

        # 向量检索配置
        vector_config=QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="fitness_knowledge",
            vector_size=768,
            distance="Cosine",
            hnsw_config={
                "m": 16,
                "ef_construct": 200
            }
        ),

        # 知识图谱配置
        knowledge_config=Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="鐜夌弽鍋ヨ韩_2024",
            database="neo4j"
        ),

        # 三层检索权重配置
        vector_weight=0.4,
        knowledge_weight=0.4,
        rules_weight=0.2,

        # 缓存配置
        cache_enabled=True,
        cache_ttl=300,

        # 检索配置
        top_k=10,
        score_threshold=0.0,

        # 学习配置
        adaptive_threshold=0.7,
        max_experiences_per_query=5,

        # 健康检查配置
        health_check_interval=60
    )

    return config


async def sample_fitness_qa():
    """示例：健身问答"""
    print("🏋️ 玉珍健身框架健身问答示例")
    print("=" * 50)

    # 创建配置
    config = await create_sample_config()

    # 初始化框架
    framework = DAMLRAGFramework(config)

    try:
        print("🚀 正在初始化玉珍健身框架...")
        await framework.initialize()

        # 检查健康状态
        health = await framework.health_check()
        print(f"📊 框架健康状态: {health['overall_status']}")

        # 示例查询
        queries = [
            "初学者如何制定健身计划？",
            "深蹲的正确动作要领是什么？",
            "增肌和减脂的饮食区别",
            "运动后如何正确恢复？",
            "有哪些适合家庭健身的器械？"
        ]

        for i, query in enumerate(queries, 1):
            print(f"\n❓ 查询 {i}: {query}")
            print("-" * 40)

            try:
                # 处理查询
                result = await framework.process_query(
                    query=query,
                    context={"user_id": "demo_user", "session_id": "demo_session"},
                    user_id="demo_user"
                )

                # 显示结果
                print(f"🤖 回答: {result.response}")
                print(f"📈 模型: {result.model_used}")
                print(f"⏱️  耗时: {result.execution_time:.2f}秒")
                print(f"📚 来源: {len(result.sources)} 个")

                # 显示三层检索统计
                if hasattr(result.metadata, 'get') and result.metadata.get('retrieval_method') == 'three_tier':
                    retrieval_meta = result.metadata.get('retrieval_result', {}).get('retrieval_metadata', {})
                    if retrieval_meta:
                        print(f"🔍 三层检索统计:")
                        print(f"   - 向量检索: {retrieval_meta.get('vector_count', 0)} 条")
                        print(f"   - 知识图谱: {retrieval_meta.get('knowledge_count', 0)} 条")
                        print(f"   - 最终结果: {retrieval_meta.get('rules_count', 0)} 条")

            except Exception as e:
                print(f"❌ 查询失败: {str(e)}")

        # 显示框架统计
        print(f"\n📊 框架统计信息:")
        stats = await framework.get_detailed_framework_stats()

        query_stats = stats['query_stats']
        print(f"   - 总查询数: {query_stats['total_queries']}")
        print(f"   - 成功查询: {query_stats['successful_queries']}")
        print(f"   - 失败查询: {query_stats['failed_queries']}")
        print(f"   - 平均响应时间: {query_stats['average_response_time']:.2f}秒")

        # 三层检索统计
        three_tier = stats['three_tier_system']
        if three_tier['enabled'] and 'statistics' in three_tier:
            tier_stats = three_tier['statistics']
            print(f"   - 向量检索缓存命中: {tier_stats.get('vector_cache_hits', 0)}")
            print(f"   - 知识图谱缓存命中: {tier_stats.get('knowledge_cache_hits', 0)}")
            print(f"   - 总检索时间: {tier_stats.get('total_retrieval_time', 0):.2f}秒")

    except Exception as e:
        print(f"❌ 框架初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n🛑 正在关闭框架...")
        await framework.shutdown()
        print("✅ 框架已关闭")


async def demonstrate_three_tier_retrieval():
    """演示三层检索系统"""
    print("\n🔍 三层检索系统演示")
    print("=" * 50)

    from daml_rag_retrieval.three_tier import ThreeTierRetriever, RetrievalRequest
    from daml_rag_retrieval.vector.qdrant import QdrantVectorRetriever, QdrantConfig
    from daml_rag_retrieval.knowledge.neo4j import Neo4jKnowledgeRetriever, Neo4jConfig
    from daml_rag_retrieval.rules.engine import RuleEngine, RuleContext

    # 创建组件
    vector_config = QdrantConfig(
        host="localhost",
        port=6333,
        collection_name="fitness_knowledge"
    )

    knowledge_config = Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="鐜夌弽鍋ヨ韩_2024"
    )

    try:
        # 初始化组件
        vector_retriever = QdrantVectorRetriever(vector_config)
        knowledge_retriever = Neo4jKnowledgeRetriever(knowledge_config)
        rule_engine = RuleEngine()

        await vector_retriever.initialize()
        await knowledge_retriever.initialize()

        # 创建三层检索器
        three_tier = ThreeTierRetriever(
            vector_retriever=vector_retriever,
            knowledge_retriever=knowledge_retriever,
            rule_engine=rule_engine,
            weights={"vector": 0.4, "knowledge": 0.4, "rules": 0.2}
        )

        await three_tier.initialize()

        # 演示检索
        query = "健身初学者应该注意什么？"

        retrieval_request = RetrievalRequest(
            query=query,
            top_k=5,
            filters={"domain": "fitness", "difficulty": "beginner"},
            user_id="demo_user"
        )

        result = await three_tier.retrieve(retrieval_request)

        print(f"📝 查询: {query}")
        print(f"🔍 向量检索结果: {len(result.vector_results.documents) if result.vector_results else 0} 条")
        print(f"🧠 知识图谱结果: {len(result.knowledge_results.documents) if result.knowledge_results else 0} 条")
        print(f"⚖️  规则过滤结果: {len(result.final_results.documents) if result.final_results else 0} 条")
        print(f"⏱️  总耗时: {result.total_execution_time:.2f}秒")

        # 显示检索到的文档
        if result.final_results and result.final_results.documents:
            print(f"\n📚 检索到的文档:")
            for i, doc in enumerate(result.final_results.documents[:3], 1):
                print(f"{i}. {doc.content[:100]}...")

        await three_tier.close()

    except Exception as e:
        print(f"❌ 三层检索演示失败: {str(e)}")
        print("💡 请确保Qdrant和Neo4j服务正在运行")


async def main():
    """主函数"""
    setup_logging()

    print("🎯 玉珍健身框架使用示例")
    print("基于权威数据源的三层检索系统演示")
    print("=" * 60)

    try:
        # 演示健身问答
        await sample_fitness_qa()

        # 演示三层检索系统
        await demonstrate_three_tier_retrieval()

    except KeyboardInterrupt:
        print("\n👋 用户中断，退出演示")
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 启动玉珍健身框架演示...")
    print("💡 前置条件:")
    print("   - Qdrant服务运行在 localhost:6333")
    print("   - Neo4j服务运行在 localhost:7474")
    print("   - 数据库已导入健身知识数据")
    print()

    asyncio.run(main())