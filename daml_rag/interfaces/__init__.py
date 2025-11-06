"""
框架接口定义
"""

from .retrieval import (
    IRetriever,
    IVectorRetriever,
    IKnowledgeGraphRetriever,
    IRuleFilter,
    ICacheManager,
)

from .orchestration import (
    ITask,
    IOrchestrator,
    IMCPTool,
    ITaskScheduler,
    ITaskExecutor,
)

from .learning import (
    IMemoryManager,
    IModelProvider,
    IAdaptiveLearner,
    IFeedbackProcessor,
)

from .quality import (
    IQualityMonitor,
    IAnomalyDetector,
    IReputationSystem,
)

from .base import (
    IComponent,
    IConfigurable,
    IMonitorable,
)

__all__ = [
    # Retrieval interfaces
    "IRetriever",
    "IVectorRetriever",
    "IKnowledgeGraphRetriever",
    "IRuleFilter",
    "ICacheManager",

    # Orchestration interfaces
    "ITask",
    "IOrchestrator",
    "IMCPTool",
    "ITaskScheduler",
    "ITaskExecutor",

    # Learning interfaces
    "IMemoryManager",
    "IModelProvider",
    "IAdaptiveLearner",
    "IFeedbackProcessor",

    # Quality interfaces
    "IQualityMonitor",
    "IAnomalyDetector",
    "IReputationSystem",

    # Base interfaces
    "IComponent",
    "IConfigurable",
    "IMonitorable",
]