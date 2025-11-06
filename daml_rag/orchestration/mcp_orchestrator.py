"""
MCP编排器 - 基于Kahn拓扑排序的任务编排

从实际生产代码提取的通用MCP工具编排器
支持DAG任务分解、循环依赖检测、异步并行执行

作者：DAML-RAG Framework Team
版本：v1.0.0
日期：2025-11-06
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"       # 等待执行
    RUNNING = "running"       # 执行中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败
    SKIPPED = "skipped"       # 跳过（依赖失败）


@dataclass
class Task:
    """
    任务定义
    
    表示一个MCP工具调用任务，包含工具信息、参数和依赖关系
    
    Attributes:
        task_id: 任务唯一标识符
        mcp_server: MCP服务器名称
        tool_name: MCP工具名称
        params: 工具调用参数（字典）
        depends_on: 依赖的任务ID列表
        status: 任务执行状态
        result: 执行结果
        error: 错误信息（如果失败）
        start_time: 开始时间戳
        end_time: 结束时间戳
    """
    task_id: str
    mcp_server: str
    tool_name: str
    params: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class MCPOrchestrator:
    """
    MCP工具编排器（通用框架）
    
    核心算法：
    - Kahn拓扑排序：确定合法的任务执行顺序
    - DFS循环检测：检测并防止循环依赖
    - 异步并行执行：同层任务并行执行，最大化吞吐量
    - TTL缓存：避免短时间内重复调用相同工具
    
    工作流程：
    1. 构建DAG图（任务 + 依赖关系）
    2. 检测循环依赖（DFS算法）
    3. Kahn拓扑排序（分层执行计划）
    4. 异步并行执行（asyncio.gather）
    5. 结果聚合和缓存更新
    
    设计原则：
    - 领域无关：不依赖特定MCP工具或领域
    - 自动并行：自动识别可并行的任务
    - 容错机制：单个任务失败不影响其他任务
    - 性能优化：TTL缓存、并行限制
    """
    
    def __init__(
        self,
        metadata_db,  # 元数据数据库实例（用于缓存）
        cache_ttl: int = 300,  # 缓存TTL（秒）
        max_parallel: int = 5,   # 最大并行任务数
        mcp_client_pool = None,  # MCP客户端池实例（可选）
    ):
        """
        初始化编排器
        
        Args:
            metadata_db: 元数据数据库实例（用于缓存工具调用结果）
            cache_ttl: 缓存生存时间（秒，默认300秒）
            max_parallel: 最大并行任务数（默认5）
            mcp_client_pool: MCP客户端池实例（可选，如果不提供则使用mock模式）
        """
        self.metadata_db = metadata_db
        self.cache_ttl = cache_ttl
        self.max_parallel = max_parallel
        self.semaphore = asyncio.Semaphore(max_parallel)
        
        # MCP客户端池（真实的MCP调用）
        self.mcp_client_pool = mcp_client_pool
        
        logger.info(
            f"MCPOrchestrator initialized: cache_ttl={cache_ttl}s, "
            f"max_parallel={max_parallel}, "
            f"mcp_mode={'real' if self.mcp_client_pool else 'mock'}"
        )
    
    async def execute(
        self,
        tasks: List[Task],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行任务编排
        
        Args:
            tasks: 任务列表（包含依赖关系）
            user_id: 用户ID（可选，用于缓存命名空间）
        
        Returns:
            Dict[str, Any]: 任务结果字典 {task_id: result, ...}
        
        Raises:
            ValueError: 如果存在循环依赖
        """
        logger.info(f"Starting orchestration: {len(tasks)} tasks")
        
        # 1. 构建任务字典
        task_dict = {t.task_id: t for t in tasks}
        
        # 2. 检测循环依赖
        if self._has_cycle(task_dict):
            raise ValueError("Circular dependency detected in task graph")
        
        # 3. Kahn拓扑排序
        execution_order = self._topological_sort(task_dict)
        
        logger.debug(f"Execution order: {execution_order}")
        
        # 4. 异步执行任务
        results = {}
        
        for level_tasks in execution_order:
            # 并行执行同一层级的任务
            level_results = await asyncio.gather(
                *[
                    self._execute_task(
                        task_dict[task_id],
                        results,
                        user_id
                    )
                    for task_id in level_tasks
                ],
                return_exceptions=True
            )
            
            # 收集结果
            for task_id, result in zip(level_tasks, level_results):
                if isinstance(result, Exception):
                    task_dict[task_id].status = TaskStatus.FAILED
                    task_dict[task_id].error = str(result)
                    logger.error(f"Task {task_id} failed: {result}")
                else:
                    results[task_id] = result
        
        logger.info(
            f"Orchestration completed: "
            f"{sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)}/{len(tasks)} succeeded"
        )
        
        return results
    
    def _has_cycle(self, task_dict: Dict[str, Task]) -> bool:
        """
        检测循环依赖（DFS + 三色标记法）
        
        算法：深度优先搜索
        - WHITE（0）: 未访问
        - GRAY（1）: 访问中
        - BLACK（2）: 已完成
        
        如果访问到GRAY节点，说明存在环
        
        时间复杂度：O(V + E)，V=节点数，E=边数
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {task_id: WHITE for task_id in task_dict}
        
        def dfs(task_id: str) -> bool:
            if color[task_id] == GRAY:
                return True  # 找到环
            
            if color[task_id] == BLACK:
                return False  # 已访问过
            
            color[task_id] = GRAY
            
            for dep in task_dict[task_id].depends_on:
                if dep in task_dict and dfs(dep):
                    return True
            
            color[task_id] = BLACK
            return False
        
        # 检查所有节点
        for task_id in task_dict:
            if color[task_id] == WHITE:
                if dfs(task_id):
                    return True
        
        return False
    
    def _topological_sort(
        self,
        task_dict: Dict[str, Task]
    ) -> List[List[str]]:
        """
        Kahn拓扑排序（分层执行）
        
        返回分层的任务ID列表，同一层级的任务可以并行执行
        
        算法步骤：
        1. 计算每个任务的入度（依赖数量）
        2. 将入度为0的任务加入第一层
        3. 执行第一层任务后，更新依赖任务的入度
        4. 将新的入度为0的任务加入下一层
        5. 重复直到所有任务分配完毕
        
        时间复杂度：O(V + E)
        
        Returns:
            List[List[str]]: 分层的任务ID列表
                [[task1, task2], [task3], [task4, task5]]
                表示：第1层并行执行task1和task2，第2层执行task3，第3层并行执行task4和task5
        """
        # 计算入度
        in_degree = {task_id: 0 for task_id in task_dict}
        
        for task in task_dict.values():
            for dep in task.depends_on:
                if dep in in_degree:
                    in_degree[task.task_id] += 1
        
        # 分层执行
        levels = []
        remaining = set(task_dict.keys())
        
        while remaining:
            # 找到当前入度为0的任务
            current_level = [
                task_id for task_id in remaining
                if in_degree[task_id] == 0
            ]
            
            if not current_level:
                # 不应该发生（已检测环）
                break
            
            levels.append(current_level)
            
            # 更新入度
            for task_id in current_level:
                remaining.remove(task_id)
                
                # 减少后继任务的入度
                for other_id in remaining:
                    if task_id in task_dict[other_id].depends_on:
                        in_degree[other_id] -= 1
        
        return levels
    
    async def _execute_task(
        self,
        task: Task,
        results: Dict[str, Any],
        user_id: Optional[str]
    ) -> Any:
        """
        执行单个任务
        
        工作流程：
        1. 检查依赖是否都成功
        2. 检查缓存
        3. 执行MCP工具调用
        4. 更新缓存
        5. 返回结果
        """
        async with self.semaphore:
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()
            
            try:
                # 检查依赖
                for dep_id in task.depends_on:
                    if dep_id not in results:
                        raise RuntimeError(f"Dependency {dep_id} not completed")
                
                # 检查缓存
                cache_key = self._build_cache_key(
                    task.mcp_server,
                    task.tool_name,
                    task.params,
                    user_id
                )
                
                cached_result = self.metadata_db.get_cache(cache_key)
                
                if cached_result is not None:
                    logger.info(f"Cache hit for task {task.task_id}")
                    task.result = cached_result
                    task.status = TaskStatus.COMPLETED
                    task.end_time = time.time()
                    return cached_result
                
                # 执行MCP工具调用
                logger.info(
                    f"Executing task {task.task_id}: "
                    f"{task.mcp_server}.{task.tool_name}"
                )
                
                result = await self._call_mcp_tool(
                    task.mcp_server,
                    task.tool_name,
                    task.params
                )
                
                # 更新缓存
                self.metadata_db.set_cache(
                    cache_key,
                    result,
                    ttl=self.cache_ttl
                )
                
                # 更新任务状态
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.end_time = time.time()
                
                logger.info(
                    f"Task {task.task_id} completed in "
                    f"{task.end_time - task.start_time:.2f}s"
                )
                
                return result
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.end_time = time.time()
                
                logger.error(f"Task {task.task_id} failed: {e}")
                
                raise
    
    def _build_cache_key(
        self,
        mcp_server: str,
        tool_name: str,
        params: Dict[str, Any],
        user_id: Optional[str]
    ) -> str:
        """
        构建缓存键
        
        格式：mcp://{user_id}/{mcp_server}/{tool_name}?{params}
        """
        # 排序参数以保证一致性
        sorted_params = sorted(params.items())
        params_str = "&".join(f"{k}={v}" for k, v in sorted_params)
        
        user_prefix = f"{user_id}/" if user_id else ""
        
        return f"mcp://{user_prefix}{mcp_server}/{tool_name}?{params_str}"
    
    async def _call_mcp_tool(
        self,
        mcp_server: str,
        tool_name: str,
        params: Dict[str, Any]
    ) -> Any:
        """
        调用MCP工具
        
        根据是否提供MCP客户端池，自动切换真实调用或模拟调用
        """
        # 真实MCP调用
        if self.mcp_client_pool:
            try:
                result = await self.mcp_client_pool.call_tool(
                    server_name=mcp_server,
                    tool_name=tool_name,
                    arguments=params
                )
                
                return result
                
            except Exception as e:
                logger.error(
                    f"MCP tool call failed: server={mcp_server}, "
                    f"tool={tool_name}, error={e}"
                )
                raise RuntimeError(f"MCP tool '{tool_name}' failed: {e}")
        
        # Mock模式（用于测试）
        else:
            logger.warning(
                f"Using MOCK mode for MCP tool: server={mcp_server}, "
                f"tool={tool_name} (mcp_client_pool not provided)"
            )
            
            # 模拟异步I/O延迟
            import random
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            # 模拟返回结果
            return {
                "mcp_server": mcp_server,
                "tool_name": tool_name,
                "params": params,
                "result": f"Mock result from {tool_name}",
                "timestamp": time.time()
            }
    
    def get_execution_summary(self, tasks: List[Task]) -> Dict:
        """
        获取执行摘要
        
        Returns:
            Dict: 执行统计信息
                - total: 总任务数
                - completed: 完成任务数
                - failed: 失败任务数
                - avg_duration: 平均执行时间
                - total_duration: 总执行时间
                - parallel_efficiency: 并行效率（理论时长/实际时长）
        """
        total = len(tasks)
        completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
        skipped = sum(1 for t in tasks if t.status == TaskStatus.SKIPPED)
        
        # 计算执行时长
        durations = [
            t.end_time - t.start_time
            for t in tasks
            if t.start_time and t.end_time
        ]
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # 总时长（从最早开始到最晚结束）
        start_times = [t.start_time for t in tasks if t.start_time]
        end_times = [t.end_time for t in tasks if t.end_time]
        
        total_duration = (
            max(end_times) - min(start_times)
            if start_times and end_times
            else 0
        )
        
        # 并行效率 = 理论时长 / 实际时长
        theoretical_duration = sum(durations)
        parallel_efficiency = (
            theoretical_duration / total_duration
            if total_duration > 0
            else 0
        )
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "avg_duration": avg_duration,
            "total_duration": total_duration,
            "parallel_efficiency": parallel_efficiency
        }

