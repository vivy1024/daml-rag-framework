"""
DAML-RAG 任务编排引擎模块
"""

from .orchestrator import TaskOrchestrator, FitnessOrchestrator
from .mcp_orchestrator import MCPOrchestrator, Task, TaskStatus

# 以下模块可能不存在，使用try-except处理
try:
    from .scheduler import TopologicalScheduler, PriorityScheduler
except ImportError:
    TopologicalScheduler = None
    PriorityScheduler = None

try:
    from .executor import TaskExecutor, ParallelExecutor
except ImportError:
    TaskExecutor = None
    ParallelExecutor = None

try:
    from .dag import TaskDAG, TaskBuilder
except ImportError:
    TaskDAG = None
    TaskBuilder = None

try:
    from .mcp_tools import MCPToolWrapper, MCPToolRegistry
except ImportError:
    MCPToolWrapper = None
    MCPToolRegistry = None

__all__ = [
    # Orchestrator
    "TaskOrchestrator",
    "FitnessOrchestrator",
    "MCPOrchestrator",  # v1.1.0新增
    "Task",
    "TaskStatus",

    # Scheduler
    "TopologicalScheduler",
    "PriorityScheduler",

    # Executor
    "TaskExecutor",
    "ParallelExecutor",

    # DAG
    "TaskDAG",
    "TaskBuilder",

    # MCP Tools
    "MCPToolWrapper",
    "MCPToolRegistry",
]