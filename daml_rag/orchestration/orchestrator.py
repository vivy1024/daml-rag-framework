"""
任务编排器实现
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
import logging
from datetime import datetime

from daml_rag.interfaces import IOrchestrator, ITask, ITaskScheduler, ITaskExecutor, IMCPTool
from daml_rag.models import Workflow, DAG, Task, TaskResult, TaskStatus, ValidationResult
from daml_rag.base import ConfigurableComponent


class TaskOrchestrator(IOrchestrator, ConfigurableComponent):
    """任务编排器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scheduler: Optional[ITaskScheduler] = None
        self.executor: Optional[ITaskExecutor] = None
        self.active_workflows: Dict[str, Workflow] = {}
        self.workflow_history: List[Workflow] = []
        self.max_concurrent_workflows = self.get_config_value('max_concurrent_workflows', 10)

    async def _do_initialize(self) -> None:
        """初始化编排器"""
        # 初始化调度器
        scheduler_config = self.get_config_value('scheduler', {})
        self.scheduler = self._create_scheduler(scheduler_config)

        # 初始化执行器
        executor_config = self.get_config_value('executor', {})
        self.executor = self._create_executor(executor_config)

    def _create_scheduler(self, config: Dict[str, Any]) -> ITaskScheduler:
        """创建任务调度器"""
        from .scheduler import TopologicalScheduler
        return TopologicalScheduler(config)

    def _create_executor(self, config: Dict[str, Any]) -> ITaskExecutor:
        """创建任务执行器"""
        from .executor import ParallelExecutor
        return ParallelExecutor(config)

    async def execute_workflow(self, workflow: Workflow,
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """执行工作流"""
        workflow_id = workflow.id
        start_time = datetime.now()

        try:
            # 检查并发限制
            if len(self.active_workflows) >= self.max_concurrent_workflows:
                raise RuntimeError(f"超过最大并发工作流限制: {self.max_concurrent_workflows}")

            # 注册工作流
            self.active_workflows[workflow_id] = workflow
            workflow.status = TaskStatus.RUNNING
            workflow.started_at = start_time

            self.logger.info(f"开始执行工作流: {workflow.name} (ID: {workflow_id})")

            # 验证工作流
            validation_result = await self.validate_workflow(workflow)
            if not validation_result.is_valid:
                workflow.status = TaskStatus.FAILED
                raise ValueError(f"工作流验证失败: {validation_result.errors}")

            # 执行任务
            execution_result = await self._execute_workflow_tasks(workflow, context)

            # 更新工作流状态
            workflow.status = TaskStatus.COMPLETED
            workflow.completed_at = datetime.now()
            workflow.results = execution_result

            # 清理
            self.active_workflows.pop(workflow_id, None)
            self.workflow_history.append(workflow)

            execution_time = (workflow.completed_at - start_time).total_seconds()
            self.logger.info(f"工作流执行完成: {workflow.name}, 耗时: {execution_time:.2f}s")

            return execution_result

        except Exception as e:
            workflow.status = TaskStatus.FAILED
            workflow.completed_at = datetime.now()
            self.active_workflows.pop(workflow_id, None)
            self.workflow_history.append(workflow)

            self.logger.error(f"工作流执行失败: {workflow.name}, 错误: {str(e)}")
            raise

    async def _execute_workflow_tasks(self, workflow: Workflow,
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """执行工作流任务"""
        results = {}
        completed_tasks = set()
        failed_tasks = set()

        # 获取任务执行计划
        execution_plan = await self.scheduler.schedule_tasks(
            list(workflow.dag.nodes.values()),
            context
        )

        # 按层级执行任务
        for level_tasks in execution_plan:
            if not level_tasks:
                continue

            # 过滤可执行的任务（依赖已完成）
            ready_tasks = [
                task for task in level_tasks
                if task.is_ready(completed_tasks) and task.id not in failed_tasks
            ]

            if not ready_tasks:
                continue

            self.logger.debug(f"执行任务层级: {len(ready_tasks)} 个任务")

            # 并行执行当前层级的任务
            level_results = await self.executor.execute_parallel(ready_tasks, context)

            # 处理结果
            for task_result in level_results:
                task_id = task_result.task_id
                results[task_id] = task_result

                if task_result.success:
                    completed_tasks.add(task_id)
                    self.logger.debug(f"任务完成: {task_id}")
                else:
                    failed_tasks.add(task_id)
                    self.logger.error(f"任务失败: {task_id}, 错误: {task_result.error}")

            # 如果有任务失败，根据配置决定是否继续
            if failed_tasks:
                fail_fast = self.get_config_value('fail_fast', True)
                if fail_fast:
                    raise RuntimeError(f"任务执行失败: {failed_tasks}")

        return results

    async def build_dag(self, tasks: List[ITask]) -> DAG:
        """构建任务DAG"""
        dag = DAG()

        # 添加所有任务
        for task in tasks:
            dag.add_task(task)

        # 检查循环依赖
        try:
            execution_order = dag.topological_sort()
            self.logger.debug(f"DAG构建成功，执行层级: {len(execution_order)}")
        except ValueError as e:
            raise ValueError(f"检测到循环依赖: {str(e)}")

        return dag

    async def schedule_tasks(self, dag: DAG,
                           context: Dict[str, Any]) -> List[List[ITask]]:
        """任务调度"""
        return await self.scheduler.schedule_tasks(list(dag.nodes.values()), context)

    async def validate_workflow(self, workflow: Workflow) -> ValidationResult:
        """验证工作流"""
        errors = []
        warnings = []

        # 检查DAG
        if not workflow.dag.nodes:
            errors.append("工作流没有任务")

        # 检查任务依赖
        for task_id, task in workflow.dag.nodes.items():
            # 检查依赖是否存在
            for dep_id in task.dependencies:
                if dep_id not in workflow.dag.nodes:
                    errors.append(f"任务 {task_id} 依赖的任务 {dep_id} 不存在")

        # 检查循环依赖
        try:
            workflow.dag.topological_sort()
        except ValueError as e:
            errors.append(f"检测到循环依赖: {str(e)}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    async def get_execution_status(self, workflow_id: str) -> Dict[str, Any]:
        """获取执行状态"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            return {
                'status': workflow.status.value,
                'started_at': workflow.started_at.isoformat() if workflow.started_at else None,
                'completed_at': workflow.completed_at.isoformat() if workflow.completed_at else None,
                'total_tasks': len(workflow.dag.nodes),
                'results': workflow.results
            }
        else:
            # 在历史记录中查找
            for workflow in self.workflow_history:
                if workflow.id == workflow_id:
                    return {
                        'status': workflow.status.value,
                        'started_at': workflow.started_at.isoformat() if workflow.started_at else None,
                        'completed_at': workflow.completed_at.isoformat() if workflow.completed_at else None,
                        'total_tasks': len(workflow.dag.nodes),
                        'results': workflow.results
                    }
            return {'error': '工作流不存在'}

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """取消工作流"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.status = TaskStatus.CANCELLED
            workflow.completed_at = datetime.now()

            self.active_workflows.pop(workflow_id, None)
            self.workflow_history.append(workflow)

            self.logger.info(f"工作流已取消: {workflow_id}")
            return True
        return False

    def get_active_workflows(self) -> List[str]:
        """获取活跃工作流"""
        return list(self.active_workflows.keys())

    def get_workflow_stats(self) -> Dict[str, Any]:
        """获取工作流统计信息"""
        total = len(self.workflow_history) + len(self.active_workflows)
        completed = sum(1 for w in self.workflow_history if w.status == TaskStatus.COMPLETED)
        failed = sum(1 for w in self.workflow_history if w.status == TaskStatus.FAILED)
        active = len(self.active_workflows)

        return {
            'total_workflows': total,
            'completed_workflows': completed,
            'failed_workflows': failed,
            'active_workflows': active,
            'success_rate': completed / total if total > 0 else 0.0,
            'max_concurrent_workflows': self.max_concurrent_workflows
        }


class FitnessOrchestrator(TaskOrchestrator):
    """健身领域专用编排器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.intent_patterns = self._load_intent_patterns()
        self.tool_dependencies = self._load_tool_dependencies()

    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """加载意图模式"""
        return {
            'exercise_search': ['exercise', '动作', '训练', '怎么练'],
            'program_design': ['计划', '方案', '制定', '设计'],
            'injury_rehab': ['损伤', '康复', '疼痛', '不舒服'],
            'nutrition': ['营养', '饮食', '吃', '减肥'],
            'assessment': ['评估', '水平', '测试', '能力']
        }

    def _load_tool_dependencies(self) -> Dict[str, List[str]]:
        """加载工具依赖关系"""
        return {
            'personalized_program': ['user_profile', 'exercise_search', 'training_capacity'],
            'injury_adjusted_program': ['injury_assessment', 'exercise_search', 'rehab_capacity_adjustment'],
            'nutrition_plan': ['tdee_calculator', 'user_profile'],
            'exercise_alternatives': ['exercise_search', 'user_profile'],
            'training_capacity': ['user_profile']
        }

    async def build_workflow_from_intent(self, query: str, context: Dict[str, Any]) -> Workflow:
        """根据意图构建工作流"""
        # 识别意图
        intent = await self._identify_intent(query)
        self.logger.info(f"识别到意图: {intent}")

        # 获取推荐工具
        tools = await self._recommend_tools(intent, context)
        self.logger.info(f"推荐工具: {tools}")

        # 构建任务
        tasks = await self._build_tasks_from_tools(tools, context)

        # 构建DAG
        dag = await self.build_dag(tasks)

        # 创建工作流
        workflow = Workflow(
            name=f"fitness_workflow_{int(datetime.now().timestamp())}",
            description=f"处理健身查询: {query[:50]}...",
            dag=dag,
            context=context
        )

        return workflow

    async def _identify_intent(self, query: str) -> str:
        """识别用户意图"""
        query_lower = query.lower()

        # 简单的关键词匹配
        for intent, keywords in self.intent_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent

        return 'general'  # 默认意图

    async def _recommend_tools(self, intent: str, context: Dict[str, Any]) -> List[str]:
        """推荐工具"""
        # 基于意图的工具推荐
        intent_tools = {
            'exercise_search': ['exercise_search', 'exercise_recommend'],
            'program_design': ['personalized_program', 'exercise_search', 'training_capacity'],
            'injury_rehab': ['injury_assessment', 'safe_alternatives', 'rehab_capacity_adjustment'],
            'nutrition': ['tdee_calculator', 'nutrition_suggestion'],
            'assessment': ['strength_assessment', 'training_capacity']
        }

        recommended = intent_tools.get(intent, ['exercise_search'])

        # 根据上下文调整推荐
        user_id = context.get('user_id')
        if user_id and user_id != 'anonymous':
            # 已登录用户可以获取更个性化的工具
            if intent == 'program_design':
                recommended = ['personalized_program_v2', 'exercise_search', 'training_capacity']

        return recommended

    async def _build_tasks_from_tools(self, tool_names: List[str],
                                    context: Dict[str, Any]) -> List[ITask]:
        """根据工具名称构建任务"""
        from .mcp_tools import MCPToolRegistry

        # 获取工具注册表
        tool_registry = MCPToolRegistry()
        await tool_registry.initialize()

        tasks = []

        for tool_name in tool_names:
            # 获取工具
            tool = await tool_registry.get_tool(tool_name)
            if not tool:
                self.logger.warning(f"工具不存在: {tool_name}")
                continue

            # 创建任务
            task = MCPToolTask(tool, context)
            tasks.append(task)

        # 设置任务依赖
        self._set_task_dependencies(tasks)

        return tasks

    def _set_task_dependencies(self, tasks: List[ITask]):
        """设置任务依赖"""
        task_map = {task.get_task_id(): task for task in tasks}

        for task in tasks:
            tool_name = task.get_task_id()
            dependencies = self.tool_dependencies.get(tool_name, [])

            for dep_tool_name in dependencies:
                if dep_tool_name in task_map:
                    task.add_dependency(task_map[dep_tool_name].get_task_id())


class MCPToolTask(ITask):
    """MCP工具任务"""

    def __init__(self, tool: IMCPTool, context: Dict[str, Any]):
        self.tool = tool
        self.context = context
        self.result: Optional[TaskResult] = None
        self.error: Optional[str] = None

    async def execute(self, context: Dict[str, Any]) -> TaskResult:
        """执行任务"""
        try:
            # 合并上下文
            merged_context = {**self.context, **context}

            # 验证参数
            validation_result = await self.validate(merged_context)
            if not validation_result.is_valid:
                return TaskResult(
                    task_id=self.get_task_id(),
                    success=False,
                    error=f"参数验证失败: {validation_result.errors}"
                )

            # 调用工具
            start_time = datetime.now()
            tool_result = await self.tool.call(merged_context)
            execution_time = (datetime.now() - start_time).total_seconds()

            # 转换结果
            success = tool_result.success if hasattr(tool_result, 'success') else True
            data = tool_result.data if hasattr(tool_result, 'data') else tool_result
            error = tool_result.error if hasattr(tool_result, 'error') else None

            self.result = TaskResult(
                task_id=self.get_task_id(),
                success=success,
                result=data,
                error=error,
                execution_time=execution_time
            )

            return self.result

        except Exception as e:
            self.error = str(e)
            return TaskResult(
                task_id=self.get_task_id(),
                success=False,
                error=str(e)
            )

    async def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """验证任务"""
        if hasattr(self.tool, 'validate_params'):
            return await self.tool.validate_params(context)
        return ValidationResult(is_valid=True)

    def get_dependencies(self) -> Set[str]:
        """获取任务依赖"""
        if hasattr(self.tool, 'get_dependencies'):
            return set(self.tool.get_dependencies())
        return set()

    def get_task_id(self) -> str:
        """获取任务ID"""
        return self.tool.get_tool_name()

    def get_task_name(self) -> str:
        """获取任务名称"""
        return self.tool.get_description()

    def estimate_execution_time(self) -> float:
        """估算执行时间"""
        return 5.0  # 默认5秒

    def can_retry(self) -> bool:
        """是否可以重试"""
        return True

    def get_max_retries(self) -> int:
        """获取最大重试次数"""
        return 3