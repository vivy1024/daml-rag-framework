"""
DAML-RAG 框架 - Domain-Adaptive Meta-Learning RAG

领域自适应元学习RAG框架
"""

__version__ = "1.0.0"
__author__ = "薛小川 (Xue Xiaochuan)"
__email__ = "1765563156@qq.com"
__license__ = "Apache 2.0"

# 延迟导入以避免循环依赖
def __getattr__(name):
    """延迟导入模块"""
    if name == "DAMLRAGFramework":
        from .core import DAMLRAGFramework
        return DAMLRAGFramework
    elif name == "DAMLRAGConfig":
        from .config import DAMLRAGConfig
        return DAMLRAGConfig
    elif name == "RetrievalConfig":
        from .config import RetrievalConfig
        return RetrievalConfig
    elif name == "OrchestrationConfig":
        from .config import OrchestrationConfig
        return OrchestrationConfig
    elif name == "LearningConfig":
        from .config import LearningConfig
        return LearningConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    
    # Core classes (lazy loaded)
    "DAMLRAGFramework",
    "DAMLRAGConfig",
    "RetrievalConfig",
    "OrchestrationConfig",
    "LearningConfig",
]