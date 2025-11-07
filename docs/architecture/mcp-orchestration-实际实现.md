# MCP编排器 - 实际实现

**版本**: v1.0.0  
**更新日期**: 2025-11-06  
**状态**: ✅ 生产代码提取

---

## 📋 文档说明

本文档记录从实际生产环境提取的MCP编排器代码，包含经过验证的Kahn拓扑排序算法实现。

**代码位置**：`daml-rag-orchestration/mcp_orchestrator.py`

---

## 🎯 核心实现

### 1. Task数据类

```python
@dataclass
class Task:
    """任务定义"""
    task_id: str               # 任务唯一标识
    mcp_server: str            # MCP服务器名称
    tool_name: str             # 工具名称
    params: Dict[str, Any]     # 工具参数
    depends_on: List[str]      # 依赖的任务ID列表
    status: TaskStatus         # 任务状态
    result: Optional[Any]      # 执行结果
    error: Optional[str]       # 错误信息
    start_time: Optional[float]  # 开始时间
    end_time: Optional[float]    # 结束时间
```

### 2. MCPOrchestrator编排器

#### 初始化

```python
orchestrator = MCPOrchestrator(
    metadata_db=metadata_db,        # 元数据数据库（用于缓存）
    cache_ttl=300,                 # 缓存TTL（秒）
    max_parallel=5,                # 最大并行数
    mcp_client_pool=mcp_client_pool  # MCP客户端池（可选）
)
```

#### 执行任务编排

```python
tasks = [
    Task("task1", "mcp1", "tool1", {}),
    Task("task2", "mcp2", "tool2", {}, depends_on=["task1"]),
    Task("task3", "mcp3", "tool3", {}, depends_on=["task1"]),
    Task("task4", "mcp4", "tool4", {}, depends_on=["task2", "task3"])
]

results = await orchestrator.execute(tasks, user_id="user123")
# 执行顺序：
# 第1层：task1
# 第2层：task2, task3（并行）
# 第3层：task4
```

---

## 🔧 核心算法

### 1. 循环依赖检测（DFS）

```python
def _has_cycle(self, task_dict: Dict[str, Task]) -> bool:
    """
    使用深度优先搜索 + 三色标记法检测循环依赖
    
    - WHITE（0）: 未访问
    - GRAY（1）: 访问中
    - BLACK（2）: 已完成
    
    如果访问到GRAY节点，说明存在环
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
```

**时间复杂度**：O(V + E)，V=节点数，E=边数

### 2. Kahn拓扑排序

```python
def _topological_sort(self, task_dict: Dict[str, Task]) -> List[List[str]]:
    """
    Kahn拓扑排序，返回分层的任务ID列表
    
    算法步骤：
    1. 计算每个任务的入度（依赖数量）
    2. 将入度为0的任务加入第一层
    3. 执行第一层任务后，更新依赖任务的入度
    4. 将新的入度为0的任务加入下一层
    5. 重复直到所有任务分配完毕
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
```

**返回示例**：
```python
[[task1], [task2, task3], [task4]]
# 表示：第1层执行task1，第2层并行执行task2和task3，第3层执行task4
```

**时间复杂度**：O(V + E)

### 3. 异步并行执行

```python
async def execute(self, tasks: List[Task], user_id: Optional[str] = None):
    # 1. 检测循环依赖
    if self._has_cycle(task_dict):
        raise ValueError("Circular dependency detected")
    
    # 2. 拓扑排序
    execution_order = self._topological_sort(task_dict)
    
    # 3. 异步并行执行
    results = {}
    
    for level_tasks in execution_order:
        # 并行执行同一层级的任务
        level_results = await asyncio.gather(
            *[self._execute_task(task_dict[task_id], results, user_id)
              for task_id in level_tasks],
            return_exceptions=True
        )
        
        # 收集结果
        for task_id, result in zip(level_tasks, level_results):
            if isinstance(result, Exception):
                task_dict[task_id].status = TaskStatus.FAILED
            else:
                results[task_id] = result
    
    return results
```

---

## ⚡ 性能优化

### 1. TTL缓存

避免短时间内重复调用相同工具：

```python
# 检查缓存
cache_key = f"mcp://{user_id}/{mcp_server}/{tool_name}?{params}"
cached_result = self.metadata_db.get_cache(cache_key)

if cached_result is not None:
    return cached_result  # 直接返回缓存结果

# 执行工具调用
result = await self._call_mcp_tool(...)

# 更新缓存
self.metadata_db.set_cache(cache_key, result, ttl=300)
```

### 2. 并发控制

使用信号量限制并发数：

```python
self.semaphore = asyncio.Semaphore(max_parallel)  # 默认5

async def _execute_task(self, task, results, user_id):
    async with self.semaphore:  # 限制并发
        # 执行任务
        ...
```

### 3. 执行统计

```python
summary = orchestrator.get_execution_summary(tasks)
# {
#     "total": 10,
#     "completed": 9,
#     "failed": 1,
#     "avg_duration": 0.25,
#     "total_duration": 1.5,
#     "parallel_efficiency": 0.75  # 并行效率（理论时长/实际时长）
# }
```

---

## 📖 使用示例

### 基础示例

```python
import asyncio
from daml_rag_orchestration import MCPOrchestrator, Task

async def main():
    # 1. 初始化编排器
    orchestrator = MCPOrchestrator(
        metadata_db=metadata_db,
        cache_ttl=300,
        max_parallel=5
    )
    
    # 2. 定义任务
    tasks = [
        Task("get_user", "user-service", "get_user_profile", {"user_id": "123"}),
        Task("get_items", "item-service", "search_items", {"query": "fitness"}),
        Task("generate", "ai-service", "generate_plan", {}, 
             depends_on=["get_user", "get_items"])
    ]
    
    # 3. 执行编排
    results = await orchestrator.execute(tasks, user_id="123")
    
    # 4. 获取结果
    print(results["generate"])

asyncio.run(main())
```

### Mock模式（测试用）

```python
# 不提供mcp_client_pool，自动使用mock模式
orchestrator = MCPOrchestrator(metadata_db=metadata_db)

results = await orchestrator.execute(tasks)
# 自动模拟MCP调用，返回mock数据
```

---

## 🔗 相关文档

- **框架总览**: [../theory/框架总览.md](../theory/框架总览.md)
- **多样性探索**: [框架多样性探索策略.md](./框架多样性探索策略.md)

---

**维护者**: DAML-RAG Framework Team  
**最后审查**: 2025-11-06

<div align="center">
<strong>✅ 生产验证 · 🚀 性能优化 · 📊 实际可用</strong>
</div>

