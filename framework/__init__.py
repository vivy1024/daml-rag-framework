"""DAML-RAG Framework - Domain Adaptive Multi-source Learning RAG Framework.

A production-ready, enterprise-grade RAG framework with:
- Three-layer retrieval (Vector → Graph → Constraint)
- DAG-based orchestration
- Adaptive model selection
- Few-shot learning
- MCP tool integration
"""

__version__ = "2.0.0"
__author__ = "薛小川 (Xue Xiaochuan)"
__email__ = "1765563156@qq.com"

# Core exports
from framework.core.simple_framework_initializer import (
    SimpleFrameworkInitializer,
)

# Orchestration exports
from framework.orchestration.generic_dag_orchestrator import (
    GenericDAGOrchestrator,
)
from framework.orchestration.mcp_orchestrator import MCPOrchestrator
from framework.orchestration.tool_registry import ToolRegistry

# Model exports
from framework.models.adaptive_model_selector import (
    AdaptiveModelSelector,
)
from framework.models.query_complexity_classifier import (
    QueryComplexityClassifier,
)

# Retrieval exports
from framework.retrieval.enhanced_few_shot_retriever import (
    EnhancedFewShotRetriever,
)
from framework.retrieval.true_three_layer_engine import (
    TrueThreeLayerEngine,
)
from framework.retrieval.unified_retrieval_interface import (
    UnifiedRetrievalInterface,
)

# Client exports
from framework.clients.llm_client import LLMClient
from framework.clients.mcp_client_v2 import MCPClientV2
from framework.clients.neo4j_client import Neo4jClient

# Storage exports
from framework.storage.intelligent_cache_system import (
    IntelligentCacheSystem,
)
from framework.storage.intelligent_user_profile_cache import (
    IntelligentUserProfileCache,
)
from framework.storage.metadata_database import MetadataDatabase
from framework.storage.user_memory import UserMemory
from framework.storage.vector_store_abstract import VectorStoreAbstract

# Monitoring exports
from framework.monitoring.daml_workflow_monitor import DAMLWorkflowMonitor
from framework.monitoring.performance_monitor import PerformanceMonitor

__all__ = [
    # Version
    "__version__",
    # Core
    "SimpleFrameworkInitializer",
    # Orchestration
    "GenericDAGOrchestrator",
    "MCPOrchestrator",
    "ToolRegistry",
    # Models
    "AdaptiveModelSelector",
    "QueryComplexityClassifier",
    # Retrieval
    "EnhancedFewShotRetriever",
    "TrueThreeLayerEngine",
    "UnifiedRetrievalInterface",
    # Clients
    "LLMClient",
    "MCPClientV2",
    "Neo4jClient",
    # Storage
    "IntelligentCacheSystem",
    "IntelligentUserProfileCache",
    "MetadataDatabase",
    "UserMemory",
    "VectorStoreAbstract",
    # Monitoring
    "DAMLWorkflowMonitor",
    "PerformanceMonitor",
]
