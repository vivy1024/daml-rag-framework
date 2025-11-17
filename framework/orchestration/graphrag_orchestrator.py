# -*- coding: utf-8 -*-
"""
DAML-RAG GraphRAG编排器 v2.0

实现智能查询编排和工具管理：
- 基于意图的工具选择
- 依赖关系解析和执行
- 任务调度和状态管理
- 上下文传递和优化

核心特性：
- 意图识别：自动识别用户查询意图
- 工具组合：智能选择最优工具组合
- 依赖管理：自动解析工具依赖关系
- 并行执行：支持任务并行执行
- 缓存优化：智能预加载和缓存

版本：v2.0.0
更新日期：2025-11-17
设计原则：智能编排、依赖管理、性能优化
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import re

from ..interfaces import (
    IOrchestrator, ITool, IToolRegistry, ITaskExecutor,
    Task, TaskResult, Workflow, WorkflowResult,
    IConfigurable, IMonitorable
)
from ..base import BaseComponent

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """查询意图"""
    INFORMATION_SEEKING = "information_seeking"    # 信息查询
    RECOMMENDATION = "recommendation"              # 推荐请求
    COMPARISON = "comparison"                       # 比较分析
    PLANNING = "planning"                          # 规划制定
    ASSESSMENT = "assessment"                      # 评估分析
    GUIDANCE = "guidance"                         # 指导建议
    PROBLEM_SOLVING = "problem_solving"           # 问题解决


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"           # 等待执行
    RUNNING = "running"           # 正在执行
    COMPLETED = "completed"       # 执行完成
    FAILED = "failed"             # 执行失败
    CANCELLED = "cancelled"       # 已取消
    SKIPPED = "skipped"           # 已跳过


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ToolDependency:
    """工具依赖关系"""
    tool_name: str
    depends_on: List[str]
    dependency_type: str  # "sequential", "parallel", "optional"


@dataclass
class ExecutionContext:
    """执行上下文"""
    user_id: str
    session_id: str
    query: str
    intent: QueryIntent
    parameters: Dict[str, Any] = field(default_factory=dict)
    preloaded_data: Dict[str, Any] = field(default_factory=dict)
    cache_enabled: bool = True


@dataclass
class TaskExecution:
    """任务执行实例"""
    task: Task
    status: TaskStatus
    result: Optional[TaskResult] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.MEDIUM


class IntentMatcher:
    """意图匹配器"""

    def __init__(self):
        self._intent_patterns = {
            QueryIntent.INFORMATION_SEEKING: [
                r'什么是|什么为|如何理解|解释',
                r'介绍一下|说明一下|告诉我',
                r'基本信息|资料|详情|概况'
            ],
            QueryIntent.RECOMMENDATION: [
                r'推荐|建议|适合.*选择',
                r'哪个好|最好|最佳选择',
                r'应该.*|如何选择'
            ],
            QueryIntent.COMPARISON: [
                r'对比|比较|区别|差异',
                r'优缺点|优势和劣势',
                r'.*和.*的区别|.*vs.*'
            ],
            QueryIntent.PLANNING: [
                r'计划|方案|安排',
                r'制定.*计划|设计.*方案',
                r'如何安排|怎样规划'
            ],
            QueryIntent.ASSESSMENT: [
                r'评估|评价|分析.*情况',
                r'水平|能力|状态.*如何',
                r'测试|检测|检查'
            ],
            QueryIntent.GUIDANCE: [
                r'指导|教程|方法',
                r'如何.*|怎么.*|怎样.*',
                r'步骤|流程|操作'
            ],
            QueryIntent.PROBLEM_SOLVING: [
                r'解决.*问题|处理.*困难',
                r'.*出现问题|.*失败',
                r'故障|错误|异常'
            ]
        }

    def match_intent(self, query: str) -> QueryIntent:
        """匹配查询意图"""
        query_lower = query.lower()

        # 计算每个意图的匹配分数
        intent_scores = {}
        for intent, patterns in self._intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            intent_scores[intent] = score

        # 返回得分最高的意图
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            if intent_scores[best_intent] > 0:
                return best_intent

        # 默认返回信息查询意图
        return QueryIntent.INFORMATION_SEEKING


class TaskScheduler:
    """任务调度器"""

    def __init__(self):
        self._running_tasks: Dict[str, TaskExecution] = {}
        self._task_queue: List[TaskExecution] = []
        self._max_concurrent_tasks = 5
        self._scheduler_metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_execution_time': 0.0
        }

    async def schedule_tasks(
        self,
        tasks: List[Task],
        dependencies: List[ToolDependency],
        context: ExecutionContext
    ) -> List[TaskExecution]:
        """调度任务执行"""
        # 创建任务执行实例
        executions = []
        for task in tasks:
            execution = TaskExecution(
                task=task,
                status=TaskStatus.PENDING,
                priority=self._determine_task_priority(task, context)
            )
            executions.append(execution)

        # 解析依赖关系
        dependency_map = self._build_dependency_map(dependencies)
        self._assign_dependencies(executions, dependency_map)

        # 按优先级排序
        self._task_queue.extend(executions)
        self._task_queue.sort(key=lambda x: (x.priority.value, x.task.name))

        # 执行任务调度
        await self._execute_schedule(context)

        return executions

    def _determine_task_priority(self, task: Task, context: ExecutionContext) -> TaskPriority:
        """确定任务优先级"""
        # 基于意图和任务类型确定优先级
        if task.name in ['get_user_profile', 'validate_input']:
            return TaskPriority.CRITICAL
        elif context.intent in [QueryIntent.RECOMMENDATION, QueryIntent.PROBLEM_SOLVING]:
            return TaskPriority.HIGH
        elif task.name in ['search_information', 'get_basic_data']:
            return TaskPriority.MEDIUM
        else:
            return TaskPriority.LOW

    def _build_dependency_map(self, dependencies: List[ToolDependency]) -> Dict[str, List[str]]:
        """构建依赖关系映射"""
        dependency_map = {}
        for dep in dependencies:
            dependency_map[dep.tool_name] = dep.depends_on
        return dependency_map

    def _assign_dependencies(
        self,
        executions: List[TaskExecution],
        dependency_map: Dict[str, List[str]]
    ) -> None:
        """分配任务依赖关系"""
        for execution in executions:
            tool_name = execution.task.name
            if tool_name in dependency_map:
                execution.dependencies = dependency_map[tool_name]

        # 设置反向依赖
        for execution in executions:
            for other_execution in executions:
                if execution.task.name in other_execution.dependencies:
                    execution.dependents.append(other_execution.task.name)

    async def _execute_schedule(self, context: ExecutionContext) -> None:
        """执行任务调度"""
        while self._task_queue or self._running_tasks:
            # 启动可执行的任务
            await self._start_ready_tasks(context)

            # 检查已完成的任务
            await self._check_completed_tasks()

            # 避免CPU占用过高
            await asyncio.sleep(0.1)

    async def _start_ready_tasks(self, context: ExecutionContext) -> None:
        """启动准备就绪的任务"""
        ready_tasks = []

        for execution in self._task_queue[:]:  # 复制列表以避免修改问题
            if self._can_start_task(execution):
                ready_tasks.append(execution)
                self._task_queue.remove(execution)

        # 限制并发任务数
        available_slots = self._max_concurrent_tasks - len(self._running_tasks)
        tasks_to_start = ready_tasks[:available_slots]

        # 启动任务
        for execution in tasks_to_start:
            await self._start_task(execution, context)

    def _can_start_task(self, execution: TaskExecution) -> bool:
        """检查任务是否可以开始"""
        if execution.status != TaskStatus.PENDING:
            return False

        # 检查依赖是否完成
        for dep_name in execution.dependencies:
            dep_execution = self._find_execution_by_name(dep_name)
            if not dep_execution or dep_execution.status != TaskStatus.COMPLETED:
                return False

        return True

    def _find_execution_by_name(self, name: str) -> Optional[TaskExecution]:
        """根据任务名称查找执行实例"""
        for execution in self._running_tasks.values():
            if execution.task.name == name:
                return execution
        return None

    async def _start_task(self, execution: TaskExecution, context: ExecutionContext) -> None:
        """启动任务执行"""
        execution.status = TaskStatus.RUNNING
        execution.start_time = time.time()
        self._running_tasks[execution.task.id] = execution

        # 异步执行任务
        asyncio.create_task(self._execute_task(execution, context))

    async def _execute_task(self, execution: TaskExecution, context: ExecutionContext) -> None:
        """执行具体任务"""
        try:
            # 获取依赖任务的结果
            dependency_results = self._get_dependency_results(execution)

            # 合并上下文和依赖结果
            task_context = {
                **context.parameters,
                **dependency_results,
                'preloaded_data': context.preloaded_data
            }

            # 执行任务
            if hasattr(execution.task, 'execute'):
                if asyncio.iscoroutinefunction(execution.task.execute):
                    result = await execution.task.execute(task_context)
                else:
                    result = execution.task.execute(task_context)
            else:
                result = TaskResult(
                    task_id=execution.task.id,
                    success=True,
                    data={},
                    message="Task completed successfully"
                )

            execution.result = result
            execution.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED

        except Exception as e:
            execution.result = TaskResult(
                task_id=execution.task.id,
                success=False,
                data={},
                message=str(e)
            )
            execution.status = TaskStatus.FAILED
            logger.error(f"任务执行失败 {execution.task.name}: {e}")

        finally:
            execution.end_time = time.time()
            self._update_metrics(execution)

            # 从运行任务中移除
            if execution.task.id in self._running_tasks:
                del self._running_tasks[execution.task.id]

    def _get_dependency_results(self, execution: TaskExecution) -> Dict[str, Any]:
        """获取依赖任务的结果"""
        dependency_results = {}
        for dep_name in execution.dependencies:
            dep_execution = self._find_execution_by_name(dep_name)
            if dep_execution and dep_execution.result:
                dependency_results[dep_name] = dep_execution.result.data

        return dependency_results

    async def _check_completed_tasks(self) -> None:
        """检查已完成的任务"""
        # 这里可以添加任务完成的回调处理
        pass

    def _update_metrics(self, execution: TaskExecution) -> None:
        """更新调度指标"""
        self._scheduler_metrics['total_tasks'] += 1

        if execution.status == TaskStatus.COMPLETED:
            self._scheduler_metrics['completed_tasks'] += 1
        else:
            self._scheduler_metrics['failed_tasks'] += 1

        # 更新平均执行时间
        if execution.start_time and execution.end_time:
            execution_time = execution.end_time - execution.start_time
            total_completed = self._scheduler_metrics['completed_tasks']
            current_avg = self._scheduler_metrics['average_execution_time']

            self._scheduler_metrics['average_execution_time'] = (
                (current_avg * (total_completed - 1) + execution_time) / total_completed
            )

    def get_metrics(self) -> Dict[str, Any]:
        """获取调度指标"""
        return {
            **self._scheduler_metrics,
            'running_tasks': len(self._running_tasks),
            'queued_tasks': len(self._task_queue),
            'max_concurrent_tasks': self._max_concurrent_tasks
        }


class DependencyResolver:
    """依赖关系解析器"""

    def __init__(self):
        self._dependency_graph: Dict[str, ToolDependency] = {}

    def register_dependency(self, dependency: ToolDependency) -> None:
        """注册工具依赖关系"""
        self._dependency_graph[dependency.tool_name] = dependency

    def resolve_execution_order(self, tools: List[str]) -> List[List[str]]:
        """解析工具执行顺序（返回层级列表）"""
        if not tools:
            return []

        # 构建执行层级
        levels = []
        remaining_tools = set(tools)
        processed_tools = set()

        while remaining_tools:
            current_level = []

            for tool in remaining_tools.copy():
                # 检查是否所有依赖都已处理
                dependencies = self._get_tool_dependencies(tool)
                if all(dep in processed_tools for dep in dependencies):
                    current_level.append(tool)
                    processed_tools.add(tool)
                    remaining_tools.remove(tool)

            if not current_level:
                # 检测循环依赖
                logger.warning(f"检测到可能的循环依赖: {remaining_tools}")
                current_level = list(remaining_tools)
                remaining_tools.clear()

            if current_level:
                levels.append(current_level)

        return levels

    def _get_tool_dependencies(self, tool_name: str) -> List[str]:
        """获取工具的依赖列表"""
        if tool_name in self._dependency_graph:
            return self._dependency_graph[tool_name].depends_on
        return []


class GraphRAGOrchestrator(BaseComponent, IOrchestrator):
    """
    GraphRAG编排器

    实现智能查询编排和工具协调执行。
    """

    def __init__(self, name: str = "GraphRAGOrchestrator", version: str = "2.0.0"):
        super().__init__(name, version)
        self._tool_registry: Optional[IToolRegistry] = None
        self._intent_matcher = IntentMatcher()
        self._task_scheduler = TaskScheduler()
        self._dependency_resolver = DependencyResolver()

        # 预定义的工具依赖关系
        self._register_default_dependencies()

        # 编排器指标
        self._orchestration_metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'average_orchestration_time': 0.0,
            'intent_distribution': {intent.value: 0 for intent in QueryIntent}
        }

    def _register_default_dependencies(self) -> None:
        """注册默认的工具依赖关系"""
        default_dependencies = [
            ToolDependency(
                tool_name="get_user_profile",
                depends_on=[],
                dependency_type="none"
            ),
            ToolDependency(
                tool_name="search_information",
                depends_on=[],
                dependency_type="none"
            ),
            ToolDependency(
                tool_name="recommend_exercises",
                depends_on=["get_user_profile", "search_information"],
                dependency_type="sequential"
            ),
            ToolDependency(
                tool_name="create_training_plan",
                depends_on=["get_user_profile", "recommend_exercises"],
                dependency_type="sequential"
            ),
            ToolDependency(
                tool_name="assess_fitness_level",
                depends_on=["get_user_profile"],
                dependency_type="sequential"
            ),
            ToolDependency(
                tool_name="compare_exercises",
                depends_on=["search_information"],
                dependency_type="sequential"
            )
        ]

        for dep in default_dependencies:
            self._dependency_resolver.register_dependency(dep)

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化编排器"""
        try:
            if config:
                self._update_config(config)

            # 验证工具注册器
            if not self._tool_registry:
                logger.error("工具注册器未设置")
                return False

            logger.info(f"✅ GraphRAG编排器初始化成功: {self.name}")
            self._state = self.ComponentState.READY
            return True

        except Exception as e:
            logger.error(f"❌ GraphRAG编排器初始化失败 {self.name}: {e}")
            self._state = self.ComponentState.ERROR
            return False

    def set_tool_registry(self, registry: IToolRegistry) -> None:
        """设置工具注册器"""
        self._tool_registry = registry

    def _update_config(self, config: Dict[str, Any]) -> None:
        """更新配置"""
        if 'max_concurrent_tasks' in config:
            self._task_scheduler._max_concurrent_tasks = config['max_concurrent_tasks']
        logger.info(f"编排器配置已更新: {self.name}")

    async def execute_task(self, task: Task, context: Optional[Dict[str, Any]] = None) -> TaskResult:
        """执行单个任务"""
        try:
            execution_context = ExecutionContext(
                user_id=context.get('user_id', '') if context else '',
                session_id=context.get('session_id', '') if context else '',
                query=task.parameters.get('query', ''),
                intent=QueryIntent.INFORMATION_SEEKING,
                parameters=context or {}
            )

            # 获取工具
            tool = self._tool_registry.get_tool(task.name)
            if not tool:
                return TaskResult(
                    task_id=task.id,
                    success=False,
                    data={},
                    message=f"工具未找到: {task.name}"
                )

            # 执行工具
            if asyncio.iscoroutinefunction(tool.execute):
                result_data = await tool.execute(task.parameters)
            else:
                result_data = tool.execute(task.parameters)

            return TaskResult(
                task_id=task.id,
                success=True,
                data=result_data,
                message="任务执行成功"
            )

        except Exception as e:
            logger.error(f"任务执行失败: {e}")
            return TaskResult(
                task_id=task.id,
                success=False,
                data={},
                message=str(e)
            )

    async def execute_workflow(self, workflow: Workflow, context: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """执行工作流"""
        start_time = time.time()
        self._orchestration_metrics['total_queries'] += 1

        try:
            # 创建执行上下文
            execution_context = ExecutionContext(
                user_id=context.get('user_id', '') if context else '',
                session_id=context.get('session_id', '') if context else '',
                query=workflow.parameters.get('query', ''),
                intent=self._intent_matcher.match_intent(workflow.parameters.get('query', '')),
                parameters=workflow.parameters,
                preloaded_data=context.get('preloaded_data', {}) if context else {}
            )

            # 更新意图分布统计
            self._orchestration_metrics['intent_distribution'][execution_context.intent.value] += 1

            # 选择相关工具
            selected_tools = await self._select_tools(execution_context)

            if not selected_tools:
                return WorkflowResult(
                    workflow_id=workflow.id,
                    success=False,
                    results=[],
                    message="未找到相关工具"
                )

            # 创建任务列表
            tasks = self._create_tasks_from_tools(selected_tools, execution_context)

            # 获取工具依赖关系
            dependencies = self._get_tool_dependencies(selected_tools)

            # 调度执行任务
            executions = await self._task_scheduler.schedule_tasks(
                tasks, dependencies, execution_context
            )

            # 收集结果
            results = []
            for execution in executions:
                if execution.result and execution.result.success:
                    results.append(execution.result)

            # 更新指标
            execution_time = time.time() - start_time
            self._update_metrics(True, execution_time)

            return WorkflowResult(
                workflow_id=workflow.id,
                success=len(results) > 0,
                results=results,
                execution_time=execution_time,
                metadata={
                    'intent': execution_context.intent.value,
                    'selected_tools': [tool.name for tool in selected_tools],
                    'total_tasks': len(tasks),
                    'successful_tasks': len(results)
                }
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            logger.error(f"工作流执行失败: {e}")

            return WorkflowResult(
                workflow_id=workflow.id,
                success=False,
                results=[],
                execution_time=execution_time,
                error=str(e)
            )

    async def _select_tools(self, context: ExecutionContext) -> List[ITool]:
        """选择相关工具"""
        if not self._tool_registry:
            return []

        # 基于意图选择工具
        intent_tool_mapping = {
            QueryIntent.INFORMATION_SEEKING: ['search_information', 'get_basic_data'],
            QueryIntent.RECOMMENDATION: ['recommend_exercises', 'get_user_profile'],
            QueryIntent.COMPARISON: ['compare_exercises', 'get_detailed_info'],
            QueryIntent.PLANNING: ['create_training_plan', 'assess_fitness_level'],
            QueryIntent.ASSESSMENT: ['assess_fitness_level', 'analyze_progress'],
            QueryIntent.GUIDANCE: ['get_instructions', 'demonstrate_technique'],
            QueryIntent.PROBLEM_SOLVING: ['diagnose_issue', 'provide_solution']
        }

        # 获取意图对应的工具名称
        tool_names = intent_tool_mapping.get(context.intent, ['search_information'])

        # 获取工具实例
        selected_tools = []
        for tool_name in tool_names:
            tool = self._tool_registry.get_tool(tool_name)
            if tool:
                selected_tools.append(tool)

        return selected_tools

    def _create_tasks_from_tools(self, tools: List[ITool], context: ExecutionContext) -> List[Task]:
        """从工具创建任务列表"""
        tasks = []
        for tool in tools:
            task = Task(
                id=f"task_{tool.name}_{int(time.time())}",
                name=tool.name,
                description=tool.description,
                parameters={
                    'query': context.query,
                    'user_id': context.user_id,
                    **context.parameters
                }
            )
            tasks.append(task)

        return tasks

    def _get_tool_dependencies(self, tools: List[ITool]) -> List[ToolDependency]:
        """获取工具依赖关系"""
        dependencies = []
        for tool in tools:
            if tool.name in self._dependency_resolver._dependency_graph:
                dependencies.append(self._dependency_resolver._dependency_graph[tool.name])

        return dependencies

    def _update_metrics(self, success: bool, execution_time: float) -> None:
        """更新编排指标"""
        if success:
            self._orchestration_metrics['successful_queries'] += 1
        else:
            self._orchestration_metrics['failed_queries'] += 1

        # 更新平均编排时间
        total_queries = self._orchestration_metrics['total_queries']
        current_avg = self._orchestration_metrics['average_orchestration_time']
        self._orchestration_metrics['average_orchestration_time'] = (
            (current_avg * (total_queries - 1) + execution_time) / total_queries
        )

    def get_metrics(self) -> Dict[str, Any]:
        """获取编排器指标"""
        base_metrics = super().get_metrics()
        scheduler_metrics = self._task_scheduler.get_metrics()

        return {
            **base_metrics,
            **self._orchestration_metrics,
            'success_rate': (
                self._orchestration_metrics['successful_queries'] /
                max(self._orchestration_metrics['total_queries'], 1)
            ),
            'scheduler_metrics': scheduler_metrics
        }


# 导出
__all__ = [
    'GraphRAGOrchestrator',
    'IntentMatcher',
    'TaskScheduler',
    'DependencyResolver',
    'QueryIntent',
    'TaskStatus',
    'TaskPriority',
    'ToolDependency',
    'ExecutionContext',
    'TaskExecution'
]