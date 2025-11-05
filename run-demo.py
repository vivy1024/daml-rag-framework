#!/usr/bin/env python3
"""
玉珍健身 框架 演示启动脚本
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# 添加框架路径
sys.path.insert(0, str(Path(__file__).parent))

from daml_rag import DAMLRAGFramework, DAMLRAGConfig
from daml_rag_adapters.fitness import FitnessDomainAdapter


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def create_demo_config():
    """创建演示配置"""
    config_data = {
        "domain": "fitness",
        "debug": True,
        "environment": "development",

        "retrieval": {
            "vector_model": "BAAI/bge-base-zh-v1.5",
            "top_k": 5,
            "similarity_threshold": 0.6,
            "cache_ttl": 300,
            "enable_kg": True,
            "enable_rules": True,
            "faiss_index_type": "flat"
        },

        "orchestration": {
            "max_parallel_tasks": 10,
            "timeout_seconds": 30,
            "retry_attempts": 3,
            "fail_fast": True
        },

        "learning": {
            "teacher_model": "deepseek",
            "student_model": "ollama-qwen2.5",
            "experience_threshold": 3.5,
            "adaptive_threshold": 0.7
        },

        "domain_config": {
            "knowledge_graph_path": "./data/knowledge_graph.db",
            "mcp_servers": [],
            "domain_specific": {}
        },

        "logging": {
            "log_level": "INFO",
            "log_to_file": False,
            "structured_logging": False
        },

        "max_concurrent_queries": 50,
        "query_timeout": 60,
        "health_check_interval": 30,
        "enable_performance_monitoring": True,
        "metrics_collection_enabled": True
    }

    config = DAMLRAGConfig.from_dict(config_data)
    return config


async def demo_basic_queries(framework):
    """演示基本查询"""
    print("\n" + "="*60)
    print("🎯 玉珍健身 框架 基本查询演示")
    print("="*60)

    demo_queries = [
        "我想制定一个增肌计划",
        "深蹲的正确动作要领是什么？",
        "膝盖有旧伤，如何安全训练腿部？",
        "增肌期间应该如何安排饮食？",
        "健身新手应该如何开始？"
    ]

    for i, query in enumerate(demo_queries, 1):
        print(f"\n📝 查询 {i}: {query}")
        print("-" * 40)

        try:
            result = await framework.process_query(query)

            print(f"🤖 响应: {result.response[:100]}...")
            if len(result.response) > 100:
                print(f"   ...({len(result.response)} 字符)")

            print(f"📊 统计:")
            print(f"   - 模型: {result.model_used}")
            print(f"   - 耗时: {result.execution_time:.2f}秒")
            print(f"   - 来源数: {len(result.sources)}")

            if result.metadata:
                print(f"   - 元数据: {list(result.metadata.keys())}")

        except Exception as e:
            print(f"❌ 错误: {str(e)}")

        print()


async def demo_framework_stats(framework):
    """演示框架统计信息"""
    print("\n" + "="*60)
    print("📊 框架统计信息")
    print("="*60)

    # 获取框架统计
    stats = framework.get_framework_stats()

    print(f"📋 框架状态:")
    print(f"   - 初始化: {'✅' if stats['initialized'] else '❌'}")
    print(f"   - 关闭: {'🔴' if stats['shutdown'] else '🟢'}")
    print(f"   - 活跃查询: {stats['active_queries']}")

    print(f"\n📈 查询统计:")
    query_stats = stats['query_stats']
    print(f"   - 总查询数: {query_stats['total_queries']}")
    print(f"   - 成功查询: {query_stats['successful_queries']}")
    print(f"   - 失败查询: {query_stats['failed_queries']}")
    print(f"   - 平均响应时间: {query_stats['average_response_time']:.3f}秒")
    print(f"   - 缓存命中: {query_stats['cache_hits']}")
    print(f"   - 缓存未命中: {query_stats['cache_misses']}")

    print(f"\n⚙️ 配置摘要:")
    config_summary = stats['config_summary']
    print(f"   - 领域: {config_summary['domain']}")
    print(f"   - 环境: {config_summary['environment']}")
    print(f"   - 调试模式: {'🔧' if config_summary['debug'] else '✅'}")

    # 健康检查
    health = await framework.health_check()
    print(f"\n🏥 健康检查:")
    print(f"   - 整体状态: {health['overall_status']}")

    for component, status in health['components'].items():
        status_icon = "✅" if status == "healthy" else "⚠️" if status == "unhealthy" else "❓"
        print(f"   - {component}: {status_icon} {status}")


async def demo_adapter_stats(adapter):
    """演示适配器统计"""
    print("\n" + "="*60)
    print("🔌 健身领域适配器统计")
    print("="*60)

    try:
        stats = await adapter.get_statistics()

        print(f"🏋️ 适配器信息:")
        print(f"   - 领域: {stats['domain']}")
        print(f"   - 版本: {stats.get('version', '1.0.0')}")
        print(f"   - 初始化: {'✅' if stats['initialized'] else '❌'}")

        print(f"\n🛠️ 工具统计:")
        print(f"   - 工具数量: {stats['tools_count']}")
        print(f"   - MCP服务器: {stats['mcp_servers']}")
        print(f"   - 活跃连接: {stats['active_connections']}")

        print(f"\n🧠 知识图谱:")
        print(f"   - 实体类型: {stats['entity_types']}")
        print(f"   - 关系类型: {stats['relation_types']}")
        print(f"   - 意图模式: {stats['intent_patterns']}")

        # 帮助主题
        help_topics = adapter.get_help_topics()
        if help_topics:
            print(f"\n💡 帮助主题:")
            for topic in help_topics:
                print(f"   - {topic['topic']}: {topic['description']}")

    except Exception as e:
        print(f"❌ 获取适配器统计失败: {str(e)}")


async def demo_config_validation(config):
    """演示配置验证"""
    print("\n" + "="*60)
    print("⚙️ 配置验证演示")
    print("="*60)

    # 验证配置
    errors = config.validate()

    if errors:
        print("❌ 配置验证失败:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ 配置验证通过")

    # 显示主要配置项
    print(f"\n📋 主要配置:")
    print(f"   - 检索配置: top_k={config.retrieval.top_k}, 相似度阈值={config.retrieval.similarity_threshold}")
    print(f"   - 编排配置: 最大并行任务={config.orchestration.max_parallel_tasks}, 超时={config.orchestration.timeout_seconds}s")
    print(f"   - 学习配置: 教师模型={config.learning.teacher_model}, 学生模型={config.learning.student_model}")
    print(f"   - 经验阈值: {config.learning.experience_threshold}, 自适应阈值={config.learning.adaptive_threshold}")


async def main():
    """主演示函数"""
    print("🚀 玉珍健身 框架 演示程序")
    print("="*60)
    print("正在初始化框架...")

    try:
        # 创建配置
        config = await create_demo_config()
        await demo_config_validation(config)

        # 创建框架
        framework = DAMLRAGFramework(config)

        # 创建适配器
        adapter = FitnessDomainAdapter(config.domain_config)

        # 初始化
        print("\n🔧 正在初始化组件...")
        await adapter.initialize()
        await framework.initialize()

        # 注册适配器
        framework.registry.register_component(FitnessDomainAdapter, adapter)

        print("✅ 框架初始化完成!")

        # 运行演示
        await demo_framework_stats(framework)
        await demo_adapter_stats(adapter)
        await demo_basic_queries(framework)

        # 最终统计
        print("\n" + "="*60)
        print("🎉 演示完成!")
        print("="*60)

        final_stats = framework.get_framework_stats()
        print(f"📊 最终统计:")
        print(f"   - 总查询数: {final_stats['query_stats']['total_queries']}")
        print(f"   - 成功率: {final_stats['query_stats']['successful_queries']}/{final_stats['query_stats']['total_queries']}")
        print(f"   - 平均响应时间: {final_stats['query_stats']['average_response_time']:.3f}秒")

        # 清理
        print("\n🧹 正在清理资源...")
        await framework.shutdown()
        await adapter.cleanup()
        print("✅ 清理完成!")

    except Exception as e:
        print(f"❌ 演示失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("玉珍健身 框架 v1.0.0 演示")
    print("基于 BUILD_BODY v2.0 的 玉珍健身 理论实现")
    print("="*60)

    asyncio.run(main())