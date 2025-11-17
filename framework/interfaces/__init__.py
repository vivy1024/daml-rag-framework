# -*- coding: utf-8 -*-
"""
DAML-RAG框架接口包 v2.0

提供框架的核心接口定义，包括：
- 基础组件接口
- 检索系统接口
- 任务编排接口
- 质量保证接口
- 存储抽象接口

版本：v2.0.0
更新日期：2025-11-17
"""

# 基础接口
from .base import (
    IComponent,
    IConfigurable,
    IMonitorable,
    ILifecycleAware,
    IAsyncComponent,
    ComponentStatus,
    ComponentState,
    BaseComponent
)

# 检索接口
from .retrieval import (
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
    BaseRetriever
)

# 编排接口
from .orchestration import (
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
    BaseOrchestrator
)

# 质量接口
from .quality import (
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
    BaseQualityChecker
)

# 存储接口
from .storage import (
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

__all__ = [
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
    'BaseStorage'
]

__version__ = "2.0.0"
__author__ = "DAML-RAG Team"