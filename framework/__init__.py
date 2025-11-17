# -*- coding: utf-8 -*-
"""
DAML-RAG框架 v2.0 - 主入口模块

DAML-RAG (Domain Adaptive Multi-source Learning RAG) 是一个面向垂直领域的
自适应多源学习型检索增强生成框架。

核心特性：
- 🏗️ 接口驱动设计：5层标准接口体系
- 🔧 组件注册系统：自动发现和依赖注入
- 📦 存储抽象层：多种存储后端统一接口
- 🎯 三层检索引擎：语义+图+约束验证
- ⚡ 质量保证体系：反幻觉和安全性检查
- 🚀 任务编排系统：工作流和工具管理

架构层次：
├── interfaces/     # 标准接口定义
├── registry/       # 组件注册和依赖注入
├── storage/        # 存储抽象层
├── retrieval/      # 检索引擎实现
├── orchestration/  # 任务编排系统
├── quality/        # 质量保证系统
└── domain/        # 领域特定实现

版本：v2.0.0
更新日期：2025-11-17
项目状态：🚧 开发中 - Phase 1 完成
"""

# 核心接口
from .interfaces import (
    # 基础接口
    IComponent,
    IConfigurable,
    IMonitorable,
    ILifecycleAware,
    IAsyncComponent,
    ComponentStatus,
    ComponentState,
    BaseComponent,

    # 检索接口
    QueryRequest,
    RetrievalResult,
    RetrievalResponse,
    RetrievalMode,
    QueryComplexity,
    IRetriever,
    ISemanticRetriever,
    IGraphRetriever,
    IConstraintValidator,
    IThreeLayerRetriever,
    IReranker,
    BaseRetriever,

    # 编排接口
    Task,
    TaskResult,
    Workflow,
    WorkflowResult,
    TaskStatus,
    WorkflowStatus,
    IOrchestrator,
    ITaskExecutor,
    IWorkflowEngine,
    ITool,
    IToolRegistry,
    IScheduler,
    BaseOrchestrator,

    # 质量接口
    QualityCheckResult,
    QualityReport,
    QualityDimension,
    ValidationLevel,
    IQualityChecker,
    IAntiHallucinationChecker,
    ISafetyChecker,
    IConsistencyChecker,
    IProfessionalStandardsChecker,
    IQualityMonitor,
    IFeedbackCollector,
    BaseQualityChecker,

    # 存储接口
    Document,
    VectorPoint,
    GraphNode,
    GraphRelationship,
    StorageType,
    IndexType,
    IStorage,
    IVectorStorage,
    IGraphStorage,
    IDocumentStorage,
    ICacheStorage,
    ISessionStorage,
    BaseStorage
)

# 组件注册系统
from .registry import (
    # 组件注册
    ComponentInfo,
    ComponentCategory,
    RegistryState,
    ComponentRegistry,
    get_global_registry,
    register_component,

    # 依赖注入
    DependencyDescriptor,
    InjectionScope,
    ServiceDescriptor,
    IContainer,
    IScope,
    DIContainer,
    Scope,
    inject,
    auto_register,
    get_container
)

# 存储抽象层
from .storage import (
    StorageConfig,
    StorageMetrics,
    AbstractStorage,
    AbstractVectorStorage,
    AbstractGraphStorage,
    AbstractDocumentStorage
)

# 框架信息
__version__ = "2.0.0"
__author__ = "DAML-RAG Team"
__description__ = "Domain Adaptive Multi-source Learning RAG Framework"
__status__ = "🚧 Phase 1 Complete - Architecture Ready"

# 导出所有公共组件
__all__ = [
    # 版本信息
    '__version__',
    '__author__',
    '__description__',
    '__status__',

    # 基础接口
    'IComponent',
    'IConfigurable',
    'IMonitorable',
    'ILifecycleAware',
    'IAsyncComponent',
    'ComponentStatus',
    'ComponentState',
    'BaseComponent',

    # 检索接口
    'QueryRequest',
    'RetrievalResult',
    'RetrievalResponse',
    'RetrievalMode',
    'QueryComplexity',
    'IRetriever',
    'ISemanticRetriever',
    'IGraphRetriever',
    'IConstraintValidator',
    'IThreeLayerRetriever',
    'IReranker',
    'BaseRetriever',

    # 编排接口
    'Task',
    'TaskResult',
    'Workflow',
    'WorkflowResult',
    'TaskStatus',
    'WorkflowStatus',
    'IOrchestrator',
    'ITaskExecutor',
    'IWorkflowEngine',
    'ITool',
    'IToolRegistry',
    'IScheduler',
    'BaseOrchestrator',

    # 质量接口
    'QualityCheckResult',
    'QualityReport',
    'QualityDimension',
    'ValidationLevel',
    'IQualityChecker',
    'IAntiHallucinationChecker',
    'ISafetyChecker',
    'IConsistencyChecker',
    'IProfessionalStandardsChecker',
    'IQualityMonitor',
    'IFeedbackCollector',
    'BaseQualityChecker',

    # 存储接口
    'Document',
    'VectorPoint',
    'GraphNode',
    'GraphRelationship',
    'StorageType',
    'IndexType',
    'IStorage',
    'IVectorStorage',
    'IGraphStorage',
    'IDocumentStorage',
    'ICacheStorage',
    'ISessionStorage',
    'BaseStorage',

    # 组件注册
    'ComponentInfo',
    'ComponentCategory',
    'RegistryState',
    'ComponentRegistry',
    'get_global_registry',
    'register_component',

    # 依赖注入
    'DependencyDescriptor',
    'InjectionScope',
    'ServiceDescriptor',
    'IContainer',
    'IScope',
    'DIContainer',
    'Scope',
    'inject',
    'auto_register',
    'get_container',

    # 存储抽象
    'StorageConfig',
    'StorageMetrics',
    'AbstractStorage',
    'AbstractVectorStorage',
    'AbstractGraphStorage',
    'AbstractDocumentStorage'
]


def get_framework_info() -> dict:
    """获取框架信息"""
    return {
        'name': 'DAML-RAG Framework',
        'version': __version__,
        'description': __description__,
        'status': __status__,
        'author': __author__,
        'architecture': {
            'layers': [
                'interfaces - 标准接口定义',
                'registry - 组件注册和依赖注入',
                'storage - 存储抽象层',
                'retrieval - 检索引擎实现',
                'orchestration - 任务编排系统',
                'quality - 质量保证系统',
                'domain - 领域特定实现'
            ],
            'principles': [
                '接口驱动设计',
                '组件化架构',
                '依赖注入',
                '异步优先',
                '类型安全',
                '可测试性'
            ]
        },
        'components': {
            'interfaces': 5,  # 5层接口体系
            'registry_systems': 2,  # 组件注册 + 依赖注入
            'storage_types': 5,  # 向量、图、文档、缓存、会话
            'retrieval_layers': 3,  # 语义、图、约束验证
            'quality_dimensions': 5  # 相关性、准确性、完整性、流畅性、安全性
        }
    }


async def initialize_framework(config: dict = None) -> bool:
    """
    初始化框架

    Args:
        config: 框架配置

    Returns:
        bool: 初始化是否成功
    """
    try:
        # 初始化全局注册器
        registry = get_global_registry()
        if config:
            registry.set_config(config.get('registry', {}))

        # 初始化依赖注入容器
        container = get_container()
        if config:
            container.set_config(config.get('di_container', {}))

        # 自动发现组件（如果配置了发现路径）
        discovery_paths = config.get('discovery_paths', []) if config else []
        if discovery_paths:
            for path in discovery_paths:
                registry.add_discovery_path(path)
            await registry.discover_components()

        print(f"🚀 DAML-RAG Framework v{__version__} 初始化成功")
        print(f"📋 框架状态: {__status__}")
        return True

    except Exception as e:
        print(f"❌ 框架初始化失败: {e}")
        return False


def create_todo_list():
    """创建开发任务列表"""
    return [
        "Phase 1: ✅ 基础架构搭建 - 创建标准化接口体系",
        "Phase 1: ✅ 设计组件注册和依赖注入系统",
        "Phase 1: ✅ 建立存储抽象层",
        "Phase 2: 🔄 从生产版本提取三层检索引擎",
        "Phase 2: ⏳ 迁移GraphRAG编排器",
        "Phase 2: ⏳ 集成反幻觉验证系统",
        "Phase 2: ⏳ 移除元学习引擎等废案组件",
        "Phase 3: ⏳ 实现具体存储后端（Qdrant、Neo4j等）",
        "Phase 3: ⏳ 开发工具注册和调度系统",
        "Phase 4: ⏳ 性能优化和监控集成",
        "Phase 5: ⏳ 文档完善和测试覆盖"
    ]