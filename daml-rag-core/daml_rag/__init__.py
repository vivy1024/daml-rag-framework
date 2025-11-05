"""
DAML-RAG Framework - Domain-Adaptive Meta-Learning RAG

面向垂直领域的自适应多源学习型RAG框架
"""

__version__ = "1.0.0"
__author__ = "DAML-RAG Team"
__email__ = "team@daml-rag.org"
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