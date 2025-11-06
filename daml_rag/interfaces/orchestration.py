"""
编排相关接口定义
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, AsyncIterator
import asyncio

from ..models import (
    Task,
    Workflow,
    DAG,
    TaskResult,
    ToolResult,
    ToolSchema,
    ValidationResult,
    TaskStatus,
)


class ITask(ABC):
    """任务抽象接口"""

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> TaskResult:
        """执行任务"""
        pass

    @abstractmethod
    async def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """验证任务"""
        pass

    @abstractmethod
    def get_dependencies(self) -> Set[str]:
        """获取任务依赖"""
        pass

    @abstractmethod
    def get_task_id(self) -> str:
        """获取任务ID"""
        pass

    @abstractmethod
    def get_task_name(self) -> str:
        """获取任务名称"""
        pass

    @abstractmethod
    def estimate_execution_time(self) -> float:
        """估算执行时间"""
        pass

    @abstractmethod
    def can_retry(self) -> bool:
        """是否可以重试"""
        pass

    @abstractmethod
    def get_max_retries(self) -> int:
        """获取最大重试次数"""
        pass


class IOrchestrator(ABC):
    """编排器抽象接口"""

    @abstractmethod
    async def execute_workflow(self, workflow: Workflow,
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """执行工作流"""
        pass

    @abstractmethod
    async def build_dag(self, tasks: List[ITask]) -> DAG:
        """构建任务DAG"""
        pass

    @abstractmethod
    async def schedule_tasks(self, dag: DAG,
                           context: Dict[str, Any]) -> List[List[ITask]]:
        """任务调度"""
        pass

    @abstractmethod
    async def validate_workflow(self, workflow: Workflow) -> ValidationResult:
        """验证工作流"""
        pass

    @abstractmethod
    async def get_execution_status(self, workflow_id: str) -> Dict[str, Any]:
        """获取执行状态"""
        pass

    @abstractmethod
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """取消工作流"""
        pass


class IMCPTool(ITask):
    """MCP工具接口"""

    @abstractmethod
    async def call(self, params: Dict[str, Any]) -> ToolResult:
        """调用工具"""
        pass

    @abstractmethod
    async def validate_params(self, params: Dict[str, Any]) -> ValidationResult:
        """参数验证"""
        pass

    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """获取工具模式"""
        pass

    @abstractmethod
    def get_tool_name(self) -> str:
        """获取工具名称"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """获取工具描述"""
        pass

    @abstractmethod
    def is_async(self) -> bool:
        """是否支持异步调用"""
        pass

    @abstractmethod
    def get_timeout(self) -> int:
        """获取超时时间"""
        pass


class ITaskScheduler(ABC):
    """任务调度器接口"""

    @abstractmethod
    async def schedule(self, tasks: List[ITask],
                      constraints: Optional[Dict[str, Any]] = None) -> List[List[ITask]]:
        """调度任务"""
        pass

    @abstractmethod
    async def optimize_schedule(self, tasks: List[ITask],
                              execution_history: Optional[Dict[str, Any]] = None) -> List[List[ITask]]:
        """优化调度"""
        pass

    @abstractmethod
    def get_scheduling_strategy(self) -> str:
        """获取调度策略"""
        pass

    @abstractmethod
    async def estimate_completion_time(self, tasks: List[ITask]) -> float:
        """估算完成时间"""
        pass


class ITaskExecutor(ABC):
    """任务执行器接口"""

    @abstractmethod
    async def execute_task(self, task: ITask,
                          context: Dict[str, Any]) -> TaskResult:
        """执行单个任务"""
        pass

    @abstractmethod
    async def execute_parallel(self, tasks: List[ITask],
                             context: Dict[str, Any]) -> List[TaskResult]:
        """并行执行任务"""
        pass

    @abstractmethod
    async def execute_with_retry(self, task: ITask,
                                context: Dict[str, Any],
                                max_retries: int = 3) -> TaskResult:
        """带重试的执行"""
        pass

    @abstractmethod
    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        pass


class IWorkflowBuilder(ABC):
    """工作流构建器接口"""

    @abstractmethod
    def add_task(self, task: ITask) -> 'IWorkflowBuilder':
        """添加任务"""
        pass

    @abstractmethod
    def add_dependency(self, task_id: str, depends_on: str) -> 'IWorkflowBuilder':
        """添加依赖"""
        pass

    @abstractmethod
    def set_context(self, context: Dict[str, Any]) -> 'IWorkflowBuilder':
        """设置上下文"""
        pass

    @abstractmethod
    def build(self) -> Workflow:
        """构建工作流"""
        pass

    @abstractmethod
    def reset(self) -> 'IWorkflowBuilder':
        """重置构建器"""
        pass


class ITaskRegistry(ABC):
    """任务注册表接口"""

    @abstractmethod
    async def register_tool(self, tool: IMCPTool) -> bool:
        """注册工具"""
        pass

    @abstractmethod
    async def unregister_tool(self, tool_name: str) -> bool:
        """注销工具"""
        pass

    @abstractmethod
    async def get_tool(self, tool_name: str) -> Optional[IMCPTool]:
        """获取工具"""
        pass

    @abstractmethod
    async def list_tools(self, category: Optional[str] = None) -> List[IMCPTool]:
        """列出工具"""
        pass

    @abstractmethod
    async def search_tools(self, query: str) -> List[IMCPTool]:
        """搜索工具"""
        pass

    @abstractmethod
    def get_tool_categories(self) -> List[str]:
        """获取工具分类"""
        pass


class IIntentMatcher(ABC):
    """意图匹配器接口"""

    @abstractmethod
    async def match_intent(self, query: str,
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """匹配意图"""
        pass

    @abstractmethod
    async def extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """提取实体"""
        pass

    @abstractmethod
    async def suggest_tools(self, intent: Dict[str, Any]) -> List[str]:
        """推荐工具"""
        pass

    @abstractmethod
    def add_pattern(self, intent_name: str, pattern: str) -> None:
        """添加模式"""
        pass

    @abstractmethod
    def get_confidence_threshold(self) -> float:
        """获取置信度阈值"""
        pass


class IResourceManager(ABC):
    """资源管理器接口"""

    @abstractmethod
    async def acquire_resource(self, resource_type: str,
                              amount: int = 1) -> bool:
        """获取资源"""
        pass

    @abstractmethod
    async def release_resource(self, resource_type: str,
                              amount: int = 1) -> bool:
        """释放资源"""
        pass

    @abstractmethod
    async def get_available_resources(self) -> Dict[str, int]:
        """获取可用资源"""
        pass

    @abstractmethod
    async def wait_for_resource(self, resource_type: str,
                               amount: int = 1,
                               timeout: Optional[float] = None) -> bool:
        """等待资源"""
        pass


class IExecutionMonitor(ABC):
    """执行监控器接口"""

    @abstractmethod
    async def start_monitoring(self, workflow_id: str) -> None:
        """开始监控"""
        pass

    @abstractmethod
    async def stop_monitoring(self, workflow_id: str) -> None:
        """停止监控"""
        pass

    @abstractmethod
    async def get_metrics(self, workflow_id: str) -> Dict[str, Any]:
        """获取指标"""
        pass

    @abstractmethod
    async def set_alert(self, workflow_id: str,
                       metric: str, threshold: float) -> None:
        """设置告警"""
        pass

    @abstractmethod
    async def get_active_workflows(self) -> List[str]:
        """获取活跃工作流"""
        pass