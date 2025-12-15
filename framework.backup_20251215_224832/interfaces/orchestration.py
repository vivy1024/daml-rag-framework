# -*- coding: utf-8 -*-
"""
DAML-RAG框架编排接口定义 v2.0

定义任务编排、工作流管理和工具调用的标准接口。

版本：v2.0.0
更新日期：2025-11-17
设计原则：灵活编排、智能调度、工具可插拔
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from .base import IComponent, IConfigurable, IMonitorable


@dataclass
class Task:
    """任务定义"""
    id: str
    name: str
    task_type: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 0
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    """工作流定义"""
    id: str
    name: str
    description: Optional[str] = None
    tasks: List[Task] = field(default_factory=list)
    global_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """工作流执行结果"""
    workflow_id: str
    success: bool
    results: Dict[str, TaskResult] = field(default_factory=dict)
    execution_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class WorkflowStatus(Enum):
    """工作流状态"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class IOrchestrator(IComponent, IConfigurable, IMonitorable):
    """
    编排器基础接口

    负责任务和工作流的调度、执行和管理。
    """

    @abstractmethod
    async def execute_task(self, task: Task) -> TaskResult:
        """
        执行单个任务

        Args:
            task: 任务定义

        Returns:
            TaskResult: 任务执行结果
        """
        pass

    @abstractmethod
    async def execute_workflow(self, workflow: Workflow) -> WorkflowResult:
        """
        执行工作流

        Args:
            workflow: 工作流定义

        Returns:
            WorkflowResult: 工作流执行结果
        """
        pass

    @abstractmethod
    async def schedule_task(self, task: Task, delay: float = 0) -> str:
        """
        调度任务

        Args:
            task: 任务定义
            delay: 延迟执行时间（秒）

        Returns:
            str: 任务ID
        """
        pass

    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """
        取消任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 取消是否成功
        """
        pass

    @abstractmethod
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        获取任务状态

        Args:
            task_id: 任务ID

        Returns:
            Optional[TaskStatus]: 任务状态
        """
        pass

    @abstractmethod
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowStatus]:
        """
        获取工作流状态

        Args:
            workflow_id: 工作流ID

        Returns:
            Optional[WorkflowStatus]: 工作流状态
        """
        pass


class ITaskExecutor(IComponent):
    """
    任务执行器接口

    负责具体任务的执行逻辑。
    """

    @abstractmethod
    async def execute(self, task: Task) -> TaskResult:
        """
        执行任务

        Args:
            task: 任务定义

        Returns:
            TaskResult: 执行结果
        """
        pass

    @abstractmethod
    def supports_task_type(self, task_type: str) -> bool:
        """
        检查是否支持指定类型的任务

        Args:
            task_type: 任务类型

        Returns:
            bool: 是否支持
        """
        pass

    @abstractmethod
    def get_supported_task_types(self) -> List[str]:
        """
        获取支持的任务类型列表

        Returns:
            List[str]: 支持的任务类型
        """
        pass

    @abstractmethod
    async def validate_task(self, task: Task) -> Tuple[bool, Optional[str]]:
        """
        验证任务

        Args:
            task: 任务定义

        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        pass


class IWorkflowEngine(IComponent):
    """
    工作流引擎接口

    负责工作流的解析、优化和执行。
    """

    @abstractmethod
    async def parse_workflow(self, workflow_definition: Dict[str, Any]) -> Workflow:
        """
        解析工作流定义

        Args:
            workflow_definition: 工作流定义

        Returns:
            Workflow: 工作流对象
        """
        pass

    @abstractmethod
    async def optimize_workflow(self, workflow: Workflow) -> Workflow:
        """
        优化工作流

        Args:
            workflow: 原始工作流

        Returns:
            Workflow: 优化后的工作流
        """
        pass

    @abstractmethod
    async def validate_workflow(self, workflow: Workflow) -> Tuple[bool, Optional[str]]:
        """
        验证工作流

        Args:
            workflow: 工作流定义

        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        pass

    @abstractmethod
    def get_execution_plan(self, workflow: Workflow) -> List[List[str]]:
        """
        获取执行计划

        Args:
            workflow: 工作流定义

        Returns:
            List[List[str]]: 执行步骤列表（并行执行的步骤分组）
        """
        pass


class ITool(IComponent):
    """
    工具接口

    定义可被编排器调用的工具。
    """

    @abstractmethod
    def get_tool_name(self) -> str:
        """
        获取工具名称

        Returns:
            str: 工具名称
        """
        pass

    @abstractmethod
    def get_tool_description(self) -> str:
        """
        获取工具描述

        Returns:
            str: 工具描述
        """
        pass

    @abstractmethod
    def get_parameters_schema(self) -> Dict[str, Any]:
        """
        获取参数模式

        Returns:
            Dict[str, Any]: 参数定义
        """
        pass

    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行工具

        Args:
            parameters: 工具参数

        Returns:
            Dict[str, Any]: 执行结果
        """
        pass

    @abstractmethod
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        验证参数

        Args:
            parameters: 待验证的参数

        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        pass

    def get_tool_metadata(self) -> Dict[str, Any]:
        """
        获取工具元数据

        Returns:
            Dict[str, Any]: 工具元数据
        """
        return {
            "name": self.get_tool_name(),
            "description": self.get_tool_description(),
            "version": getattr(self, 'version', '1.0.0'),
            "category": getattr(self, 'category', 'general')
        }


class IToolRegistry(IComponent):
    """
    工具注册器接口

    管理工具的注册、发现和调用。
    """

    @abstractmethod
    def register_tool(self, tool: ITool) -> bool:
        """
        注册工具

        Args:
            tool: 工具实例

        Returns:
            bool: 注册是否成功
        """
        pass

    @abstractmethod
    def unregister_tool(self, tool_name: str) -> bool:
        """
        注销工具

        Args:
            tool_name: 工具名称

        Returns:
            bool: 注销是否成功
        """
        pass

    @abstractmethod
    def get_tool(self, tool_name: str) -> Optional[ITool]:
        """
        获取工具

        Args:
            tool_name: 工具名称

        Returns:
            Optional[ITool]: 工具实例
        """
        pass

    @abstractmethod
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """
        列出工具

        Args:
            category: 工具分类过滤

        Returns:
            List[str]: 工具名称列表
        """
        pass

    @abstractmethod
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用工具

        Args:
            tool_name: 工具名称
            parameters: 工具参数

        Returns:
            Dict[str, Any]: 执行结果
        """
        pass

    @abstractmethod
    def search_tools(self, query: str) -> List[str]:
        """
        搜索工具

        Args:
            query: 搜索查询

        Returns:
            List[str]: 匹配的工具名称列表
        """
        pass

    def get_tool_statistics(self) -> Dict[str, Any]:
        """
        获取工具统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "total_tools": len(self.list_tools()),
            "categories": {}
        }


class IScheduler(IComponent):
    """
    调度器接口

    负责任务的调度和时间管理。
    """

    @abstractmethod
    async def schedule_at(self, task: Task, execute_time: float) -> str:
        """
        定时调度

        Args:
            task: 任务
            execute_time: 执行时间戳

        Returns:
            str: 调度ID
        """
        pass

    @abstractmethod
    async def schedule_cron(self, task: Task, cron_expression: str) -> str:
        """
        Cron调度

        Args:
            task: 任务
            cron_expression: Cron表达式

        Returns:
            str: 调度ID
        """
        pass

    @abstractmethod
    async def cancel_schedule(self, schedule_id: str) -> bool:
        """
        取消调度

        Args:
            schedule_id: 调度ID

        Returns:
            bool: 取消是否成功
        """
        pass

    @abstractmethod
    def get_scheduled_tasks(self) -> List[Dict[str, Any]]:
        """
        获取已调度任务列表

        Returns:
            List[Dict[str, Any]]: 调度任务信息
        """
        pass


# 便捷的编排器基类
class BaseOrchestrator(BaseComponent):
    """
    编排器基础实现类
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
        self._task_executors: Dict[str, ITaskExecutor] = {}
        self._workflow_engine: Optional[IWorkflowEngine] = None
        self._scheduler: Optional[IScheduler] = None
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._completed_tasks: Dict[str, TaskResult] = {}

    def register_task_executor(self, task_type: str, executor: ITaskExecutor) -> bool:
        """注册任务执行器"""
        if executor.supports_task_type(task_type):
            self._task_executors[task_type] = executor
            return True
        return False

    def get_task_executor(self, task_type: str) -> Optional[ITaskExecutor]:
        """获取任务执行器"""
        return self._task_executors.get(task_type)

    def get_statistics(self) -> Dict[str, Any]:
        """获取编排统计"""
        return {
            "registered_executors": len(self._task_executors),
            "running_tasks": len(self._running_tasks),
            "completed_tasks": len(self._completed_tasks),
            **self.get_metrics()
        }


# 导出接口
__all__ = [
    # 核心数据结构
    'Task',
    'TaskResult',
    'Workflow',
    'WorkflowResult',
    'TaskStatus',
    'WorkflowStatus',

    # 编排接口
    'IOrchestrator',
    'ITaskExecutor',
    'IWorkflowEngine',
    'ITool',
    'IToolRegistry',
    'IScheduler',

    # 基础实现
    'BaseOrchestrator'
]