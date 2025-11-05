#!/usr/bin/env python3
"""
玉珍健身框架配置示例
展示不同场景下的配置选项
"""

from typing import Dict, Any
from dataclasses import dataclass

from daml_rag.config import DAMLRAGConfig
from daml_rag_retrieval.vector.qdrant import QdrantConfig
from daml_rag_retrieval.vector.faiss import FAISSConfig
from daml_rag_retrieval.knowledge.neo4j import Neo4jConfig


def create_production_config() -> DAMLRAGConfig:
    """生产环境配置"""
    return DAMLRAGConfig(
        domain="fitness",
        environment="production",
        debug=False,

        # Qdrant生产配置
        vector_config=QdrantConfig(
            host="qdrant-service",
            port=6333,
            api_key="your_production_api_key",
            collection_name="fitness_production",
            vector_size=1024,  # 使用更大的向量维度
            distance="Cosine",
            batch_size=100,
            timeout=30,

            # 生产环境HNSW优化
            hnsw_config={
                "m": 32,              # 更高的连接度
                "ef_construct": 400,  # 更高的构建精度
                "full_scan_threshold": 20000
            }
        ),

        # Neo4j生产配置
        knowledge_config=Neo4jConfig(
            uri="bolt://neo4j-service:7687",
            username="neo4j",
            password="production_password",
            database="fitness_prod",
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
            connection_timeout=30
        ),

        # 三层检索权重（生产环境优化）
        vector_weight=0.35,
        knowledge_weight=0.45,
        rules_weight=0.20,

        # 缓存配置
        cache_enabled=True,
        cache_ttl=600,  # 10分钟缓存

        # 检索配置
        top_k=15,
        score_threshold=0.3,

        # 学习配置
        adaptive_threshold=0.8,
        max_experiences_per_query=10,
        learning_rate=0.01,

        # 性能配置
        health_check_interval=30,
        enable_performance_monitoring=True,
        max_concurrent_queries=100
    )


def create_development_config() -> DAMLRAGConfig:
    """开发环境配置"""
    return DAMLRAGConfig(
        domain="fitness",
        environment="development",
        debug=True,

        # 本地Qdrant配置
        vector_config=QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="fitness_dev",
            vector_size=768,
            distance="Cosine",
            batch_size=50,
            timeout=10,

            hnsw_config={
                "m": 16,
                "ef_construct": 100
            }
        ),

        # 本地Neo4j配置
        knowledge_config=Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="鐜夌弽鍋ヨ韩_2024",
            database="neo4j",
            max_connection_pool_size=10,
            connection_timeout=10
        ),

        # 三层检索权重（开发环境）
        vector_weight=0.4,
        knowledge_weight=0.4,
        rules_weight=0.2,

        # 缓存配置
        cache_enabled=True,
        cache_ttl=60,  # 1分钟缓存，便于开发调试

        # 检索配置
        top_k=10,
        score_threshold=0.0,

        # 学习配置
        adaptive_threshold=0.7,
        max_experiences_per_query=5,

        # 性能配置
        health_check_interval=60,
        enable_performance_monitoring=True
    )


def create_local_faiss_config() -> DAMLRAGConfig:
    """本地FAISS配置（无外部依赖）"""
    return DAMLRAGConfig(
        domain="fitness",
        environment="local",
        debug=True,

        # FAISS本地配置
        vector_config=FAISSConfig(
            index_type="hnsw",
            vector_size=768,
            metric_type="cosine",
            M=16,
            efConstruction=200,
            efSearch=50,
            index_path="./data/faiss_fitness.index",
            save_index=True,
            use_gpu=False
        ),

        # 轻量级Neo4j配置
        knowledge_config=Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="鐜夌弽鍋ヨ韩_2024",
            database="neo4j",
            max_connection_pool_size=5
        ),

        # 三层检索权重
        vector_weight=0.5,  # FAISS权重稍高
        knowledge_weight=0.3,
        rules_weight=0.2,

        # 缓存配置
        cache_enabled=True,
        cache_ttl=120,

        # 检索配置
        top_k=8,
        score_threshold=0.1,

        # 学习配置
        adaptive_threshold=0.6,
        max_experiences_per_query=3
    )


def create_high_performance_config() -> DAMLRAGConfig:
    """高性能配置（适用于高并发场景）"""
    return DAMLRAGConfig(
        domain="fitness",
        environment="performance",
        debug=False,

        # 高性能Qdrant配置
        vector_config=QdrantConfig(
            host="qdrant-cluster",
            port=6333,
            api_key="high_perf_api_key",
            collection_name="fitness_highperf",
            vector_size=1536,  # 使用高维向量
            distance="Cosine",
            batch_size=200,
            timeout=5,  # 更短的超时时间

            # 高性能HNSW配置
            hnsw_config={
                "m": 64,              # 高连接度
                "ef_construct": 800,  # 高构建精度
                "ef_search": 100,     # 高搜索精度
                "full_scan_threshold": 10000
            }
        ),

        # 高性能Neo4j配置
        knowledge_config=Neo4jConfig(
            uri="bolt://neo4j-cluster:7687",
            username="neo4j",
            password="high_perf_password",
            database="fitness_highperf",
            max_connection_lifetime=7200,
            max_connection_pool_size=100,
            connection_timeout=5,
            max_transaction_retry_time=30
        ),

        # 优化的权重配置
        vector_weight=0.3,
        knowledge_weight=0.5,
        rules_weight=0.2,

        # 缓存配置（高频访问优化）
        cache_enabled=True,
        cache_ttl=1800,  # 30分钟缓存

        # 检索配置
        top_k=20,
        score_threshold=0.4,

        # 学习配置（保守策略）
        adaptive_threshold=0.9,
        max_experiences_per_query=15,

        # 性能配置
        health_check_interval=15,
        enable_performance_monitoring=True,
        max_concurrent_queries=500
    )


def create_minimal_config() -> DAMLRAGConfig:
    """最小配置（适用于资源受限环境）"""
    return DAMLRAGConfig(
        domain="fitness",
        environment="minimal",
        debug=False,

        # 轻量级FAISS配置
        vector_config=FAISSConfig(
            index_type="flat",  # 使用最简单的索引
            vector_size=384,    # 较小的向量维度
            metric_type="cosine",
            save_index=False,   # 不保存索引
            use_gpu=False
        ),

        # 简化的Neo4j配置
        knowledge_config=Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="鐜夌弽鍋ヨ韩_2024",
            database="neo4j",
            max_connection_pool_size=2
        ),

        # 简化的权重
        vector_weight=0.6,
        knowledge_weight=0.2,
        rules_weight=0.2,

        # 禁用缓存以节省内存
        cache_enabled=False,

        # 基础检索配置
        top_k=5,
        score_threshold=0.0,

        # 最小学习配置
        adaptive_threshold=0.5,
        max_experiences_per_query=2,

        # 最小监控
        health_check_interval=300,
        enable_performance_monitoring=False
    )


@dataclass
class ConfigTemplate:
    """配置模板"""
    name: str
    description: str
    config_factory: callable
    requirements: Dict[str, Any]


def get_all_config_templates() -> list[ConfigTemplate]:
    """获取所有配置模板"""
    return [
        ConfigTemplate(
            name="production",
            description="生产环境配置 - 高可用、高性能、高可靠性",
            config_factory=create_production_config,
            requirements={
                "qdrant": "集群版Qdrant",
                "neo4j": "集群版Neo4j",
                "memory": ">= 8GB",
                "cpu": ">= 4核"
            }
        ),
        ConfigTemplate(
            name="development",
            description="开发环境配置 - 本地开发、调试友好",
            config_factory=create_development_config,
            requirements={
                "qdrant": "本地Qdrant实例",
                "neo4j": "本地Neo4j实例",
                "memory": ">= 4GB",
                "cpu": ">= 2核"
            }
        ),
        ConfigTemplate(
            name="local_faiss",
            description="本地FAISS配置 - 无外部向量数据库依赖",
            config_factory=create_local_faiss_config,
            requirements={
                "neo4j": "本地Neo4j实例",
                "memory": ">= 2GB",
                "cpu": ">= 2核",
                "storage": ">= 1GB"
            }
        ),
        ConfigTemplate(
            name="high_performance",
            description="高性能配置 - 适用于高并发、低延迟场景",
            config_factory=create_high_performance_config,
            requirements={
                "qdrant": "高性能Qdrant集群",
                "neo4j": "高性能Neo4j集群",
                "memory": ">= 16GB",
                "cpu": ">= 8核"
            }
        ),
        ConfigTemplate(
            name="minimal",
            description="最小配置 - 适用于资源受限环境",
            config_factory=create_minimal_config,
            requirements={
                "neo4j": "本地Neo4j实例",
                "memory": ">= 1GB",
                "cpu": ">= 1核",
                "storage": ">= 100MB"
            }
        )
    ]


def print_config_templates():
    """打印所有配置模板"""
    templates = get_all_config_templates()

    print("📋 玉珍健身框架配置模板")
    print("=" * 60)

    for template in templates:
        print(f"\n🏷️  {template.name}")
        print(f"📝 {template.description}")
        print(f"📋 系统要求:")

        for requirement, value in template.requirements.items():
            print(f"   • {requirement}: {value}")

    print(f"\n💡 使用示例:")
    print("   config = create_development_config()  # 创建开发环境配置")
    print("   framework = DAMLRAGFramework(config)  # 初始化框架")


if __name__ == "__main__":
    print_config_templates()