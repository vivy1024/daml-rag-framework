# -*- coding: utf-8 -*-
"""
DAML-RAG编排系统 v2.0

实现任务编排和工作流管理：
- 智能工具选择和组合
- 依赖关系解析和执行
- 意图识别和查询路由
- 任务调度和状态管理

版本：v2.0.0
更新日期：2025-11-17
"""

# 核心编排器
from .graphrag_orchestrator import (
    GraphRAGOrchestrator,
    IntentMatcher,
    TaskScheduler,
    DependencyResolver
)

# 工具注册系统
from .tool_registry import (
    ToolRegistry,
    Tool,
    ToolContext,
    ToolResult,
    ToolCategory
)

# 工作流引擎
from .workflow_engine import (
    WorkflowEngine,
    Workflow,
    Task,
    TaskStatus,
    WorkflowResult
)

# 查询分析器
from .query_analyzer import (
    QueryAnalyzer,
    QueryIntent,
    QueryComplexity,
    QueryAnalysisResult
)

__all__ = [
    # 核心编排
    'GraphRAGOrchestrator',
    'IntentMatcher',
    'TaskScheduler',
    'DependencyResolver',

    # 工具系统
    'ToolRegistry',
    'Tool',
    'ToolContext',
    'ToolResult',
    'ToolCategory',

    # 工作流
    'WorkflowEngine',
    'Workflow',
    'Task',
    'TaskStatus',
    'WorkflowResult',

    # 查询分析
    'QueryAnalyzer',
    'QueryIntent',
    'QueryComplexity',
    'QueryAnalysisResult'
]

__version__ = "2.0.0"
__author__ = "DAML-RAG Team"