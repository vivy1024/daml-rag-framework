# -*- coding: utf-8 -*-
"""
简化的框架初始化器

初始化核心组件：
1. 存储层（MetadataDB, UserMemory）
2. GraphRAG（KnowledgeGraphFull）
3. MCP客户端池（ConfigurableMCPClient）
4. 质量监控（可选）

删除组件：
- 元学习引擎
- 工具性能追踪
- 复杂的验证逻辑

作者：BUILD_BODY Team
版本：v3.2.0
日期：2025-12-13
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class InitResult:
    """初始化结果"""
    success: bool
    components: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    duration_seconds: float = 0.0


class SimpleFrameworkInitializer:
    """
    简化的框架初始化器
    
    只初始化核心组件：
    1. 存储层（MetadataDB, UserMemory）
    2. GraphRAG（KnowledgeGraphFull）
    3. MCP编排（MCPOrchestrator）
    4. 模型调度（SimpleModelScheduler）
    5. 质量监控（SimpleQualityMonitor）
    
    删除内容：
    - 元学习引擎
    - 工具性能追踪
    - 复杂的验证逻辑
    - 详细的健康检查
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化框架初始化器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.components = {}
        
        logger.info("✅ 简化框架初始化器已创建")
    
    async def initialize(self) -> InitResult:
        """
        执行框架初始化
        
        Returns:
            InitResult: 初始化结果
        """
        start_time = datetime.now()
        logger.info("=" * 80)
        logger.info("🚀 开始DAML-RAG框架初始化 v3.0")
        logger.info("=" * 80)
        
        errors = {}
        
        try:
            # 1. 初始化存储层
            logger.info("\n📦 Step 1/5: 初始化存储层")
            try:
                from ..storage.metadata_database import MetadataDB
                from ..storage.user_memory import UserMemory
                
                self.components["metadata_db"] = MetadataDB(
                    db_path=self.config.get("metadata_db_path", "/tmp/metadata.db")
                )
                logger.info("  ✅ MetadataDB初始化成功")
                
                # 初始化Qdrant客户端
                from qdrant_client import QdrantClient
                qdrant_client = QdrantClient(
                    url=self.config.get("qdrant_url", "http://qdrant:6333")
                )
                
                self.components["user_memory"] = UserMemory(
                    qdrant_client=qdrant_client,
                    vector_size=1024  # 修复：与BGE-M3保持一致
                )
                logger.info("  ✅ UserMemory初始化成功")
                
            except Exception as e:
                logger.error(f"  ❌ 存储层初始化失败: {e}")
                errors["storage"] = str(e)
            
            # 2. 初始化GraphRAG
            logger.info("\n🔍 Step 2/5: 初始化GraphRAG")
            try:
                from ..retrieval.graph.kg_full import KnowledgeGraphFull
                
                self.components["kg_full"] = KnowledgeGraphFull(
                    neo4j_uri=self.config.get("neo4j_uri", "bolt://neo4j:7687"),
                    neo4j_user=self.config.get("neo4j_user", "neo4j"),
                    neo4j_password=self.config.get("neo4j_password", "build_body_2024"),
                    qdrant_host=self.config.get("qdrant_host", "qdrant"),
                    qdrant_port=self.config.get("qdrant_port", 6333),
                    embedding_model=self.config.get("embedding_model", "BAAI/bge-m3")
                )
                logger.info("  ✅ KnowledgeGraphFull初始化成功")
                
            except Exception as e:
                logger.error(f"  ❌ GraphRAG初始化失败: {e}")
                errors["graphrag"] = str(e)
            
            # 3. 初始化MCP客户端池
            logger.info("\n🔌 Step 3/5: 初始化MCP客户端池")
            try:
                from ..clients.mcp_client_v2 import create_configurable_mcp_client
                import os
                
                # 获取配置文件路径
                config_path = self.config.get(
                    "mcp_config_path",
                    os.getenv("MCP_CONFIG_PATH", "/app/config/mcp_registry.json")
                )
                
                # 创建MCP客户端
                mcp_client = create_configurable_mcp_client(
                    config_path=config_path,
                    auto_reload=False
                )
                
                # 连接MCP服务
                connected = await mcp_client.connect()
                
                if connected:
                    self.components["mcp_client"] = mcp_client
                    logger.info("  ✅ MCP客户端池初始化成功")
                    
                    # 显示可用服务器
                    servers = mcp_client.get_all_servers()
                    logger.info(f"  📋 可用MCP服务器: {list(servers.keys())}")
                else:
                    logger.warning("  ⚠️  MCP客户端连接失败，部分功能可能不可用")
                    errors["mcp_client"] = "连接失败"
                
            except Exception as e:
                logger.error(f"  ❌ MCP客户端池初始化失败: {e}")
                errors["mcp_client"] = str(e)
            
            # 4. 初始化质量监控（简化版）
            logger.info("\n📊 Step 4/5: 初始化质量监控")
            try:
                # 简单的质量监控（暂时跳过，后续实现）
                logger.info("  ⚠️  质量监控暂未实现（可选）")
                
            except Exception as e:
                logger.error(f"  ❌ 质量监控初始化失败: {e}")
                errors["quality_monitor"] = str(e)
            
            # 计算初始化时间
            duration = (datetime.now() - start_time).total_seconds()
            
            # 构建结果
            success = len(errors) == 0
            
            logger.info("\n" + "=" * 80)
            if success:
                logger.info(f"✅ DAML-RAG框架初始化完成，耗时: {duration:.2f}s")
                logger.info(f"📊 成功初始化组件: {len(self.components)}")
            else:
                logger.warning(f"⚠️  DAML-RAG框架部分初始化，耗时: {duration:.2f}s")
                logger.warning(f"📊 成功组件: {len(self.components)}, 失败组件: {len(errors)}")
                for component, error in errors.items():
                    logger.error(f"  ❌ {component}: {error}")
            logger.info("=" * 80)
            
            return InitResult(
                success=success,
                components=self.components,
                errors=errors,
                duration_seconds=duration
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ 框架初始化异常: {e}")
            return InitResult(
                success=False,
                components=self.components,
                errors={"framework": str(e)},
                duration_seconds=duration
            )
    
    async def shutdown(self):
        """清理资源"""
        logger.info("🔄 开始清理框架资源...")
        
        try:
            # 关闭MCP客户端
            if "mcp_client" in self.components:
                await self.components["mcp_client"].disconnect()
                logger.info("  ✅ MCP客户端已断开")
            
            # 关闭GraphRAG
            if "kg_full" in self.components:
                self.components["kg_full"].close()
                logger.info("  ✅ KnowledgeGraphFull已关闭")
            
            # 关闭MetadataDB
            if "metadata_db" in self.components:
                self.components["metadata_db"].close()
                logger.info("  ✅ MetadataDB已关闭")
            
            logger.info("✅ 框架资源清理完成")
            
        except Exception as e:
            logger.error(f"❌ 资源清理失败: {e}")
    
    def get_component(self, name: str) -> Optional[Any]:
        """
        获取组件
        
        Args:
            name: 组件名称
        
        Returns:
            组件实例或None
        """
        return self.components.get(name)


# 全局单例
_initializer_instance: Optional[SimpleFrameworkInitializer] = None


def get_framework_initializer(config: Optional[Dict[str, Any]] = None) -> SimpleFrameworkInitializer:
    """
    获取框架初始化器单例
    
    Args:
        config: 配置字典
    
    Returns:
        SimpleFrameworkInitializer实例
    """
    global _initializer_instance
    
    if _initializer_instance is None:
        _initializer_instance = SimpleFrameworkInitializer(config)
    
    return _initializer_instance


async def initialize_framework(config: Optional[Dict[str, Any]] = None) -> InitResult:
    """
    便捷函数：初始化框架
    
    Args:
        config: 配置字典
    
    Returns:
        InitResult: 初始化结果
    """
    initializer = get_framework_initializer(config)
    return await initializer.initialize()
