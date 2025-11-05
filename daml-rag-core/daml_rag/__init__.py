"""
玉珍健身 框架 - Domain-Adaptive Meta-Learning RAG

玉珍健身智能健身框架
"""

__version__ = "1.0.0"
__author__ = "薛小川 (Xue Xiaochuan)"
__email__ = "1765563156@qq.com"
__license__ = "Apache 2.0"

from .core import DAMLRAGFramework
from .config import DAMLRAGConfig, RetrievalConfig, OrchestrationConfig, LearningConfig
from .interfaces import IRetriever, IOrchestrator, IModelProvider, IMemoryManager
from .models import QueryResult, RetrievalResult, TaskResult, Experience

__all__ = [
    # Core classes
    "DAMLRAGFramework",

    # Configuration
    "DAMLRAGConfig",
    "RetrievalConfig",
    "OrchestrationConfig",
    "LearningConfig",

    # Interfaces
    "IRetriever",
    "IOrchestrator",
    "IModelProvider",
    "IMemoryManager",

    # Models
    "QueryResult",
    "RetrievalResult",
    "TaskResult",
    "Experience",
]