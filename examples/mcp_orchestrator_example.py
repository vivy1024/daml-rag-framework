"""
MCP编排器使用示例

演示如何使用MCPOrchestrator进行任务编排
"""

import asyncio
from daml_rag_orchestration import MCPOrchestrator, Task, TaskStatus


# Mock的元数据数据库（用于示例）
class MockMetadataDB:
    """Mock元数据数据库"""
    
    def __init__(self):
        self.cache = {}
    
    def get_cache(self, key: str):
        """获取缓存"""
        return self.cache.get(key)
    
    def set_cache(self, key: str, value, ttl: int = 300):
        """设置缓存"""
        self.cache[key] = value


async def example_basic():
    """基础示例：简单的任务编排"""
    print("=" * 60)
    print("示例1：基础任务编排")
    print("=" * 60)
    
    # 1. 初始化编排器（Mock模式）
    metadata_db = MockMetadataDB()
    orchestrator = MCPOrchestrator(
        metadata_db=metadata_db,
        cache_ttl=300,
        max_parallel=5
    )
    
    # 2. 定义任务
    tasks = [
        Task(
            task_id="task1",
            mcp_server="service1",
            tool_name="tool1",
            params={"param1": "value1"}
        ),
        Task(
            task_id="task2",
            mcp_server="service2",
            tool_name="tool2",
            params={"param2": "value2"},
            depends_on=["task1"]  # task2依赖task1
        ),
        Task(
            task_id="task3",
            mcp_server="service3",
            tool_name="tool3",
            params={"param3": "value3"},
            depends_on=["task1"]  # task3也依赖task1
        ),
    ]
    
    # 3. 执行编排
    print("\n开始执行任务编排...")
    results = await orchestrator.execute(tasks)
    
    # 4. 输出结果
    print("\n执行结果：")
    for task_id, result in results.items():
        print(f"  {task_id}: {result}")
    
    # 5. 执行摘要
    summary = orchestrator.get_execution_summary(tasks)
    print(f"\n执行摘要：")
    print(f"  总任务数: {summary['total']}")
    print(f"  完成: {summary['completed']}")
    print(f"  失败: {summary['failed']}")
    print(f"  平均耗时: {summary['avg_duration']:.3f}秒")
    print(f"  总耗时: {summary['total_duration']:.3f}秒")
    print(f"  并行效率: {summary['parallel_efficiency']:.2%}")
    print()


async def example_parallel():
    """示例2：并行执行"""
    print("=" * 60)
    print("示例2：并行任务执行")
    print("=" * 60)
    
    metadata_db = MockMetadataDB()
    orchestrator = MCPOrchestrator(
        metadata_db=metadata_db,
        max_parallel=10
    )
    
    # 定义多个无依赖的任务（将并行执行）
    tasks = [
        Task(f"task{i}", f"service{i}", f"tool{i}", {})
        for i in range(1, 6)
    ]
    
    print(f"\n创建了 {len(tasks)} 个无依赖任务（将并行执行）...")
    
    results = await orchestrator.execute(tasks)
    
    summary = orchestrator.get_execution_summary(tasks)
    print(f"\n并行执行摘要：")
    print(f"  总耗时: {summary['total_duration']:.3f}秒")
    print(f"  并行效率: {summary['parallel_efficiency']:.2%}")
    print(f"  （理论上5个任务应该几乎同时完成，并行效率接近5.0）")
    print()


async def example_complex_dag():
    """示例3：复杂DAG"""
    print("=" * 60)
    print("示例3：复杂任务依赖图（DAG）")
    print("=" * 60)
    
    metadata_db = MockMetadataDB()
    orchestrator = MCPOrchestrator(metadata_db=metadata_db)
    
    # 构建复杂的依赖图
    # task1 → task2 → task4
    #      → task3 → task5
    # task4, task5 → task6
    
    tasks = [
        Task("task1", "s1", "t1", {}),
        Task("task2", "s2", "t2", {}, depends_on=["task1"]),
        Task("task3", "s3", "t3", {}, depends_on=["task1"]),
        Task("task4", "s4", "t4", {}, depends_on=["task2"]),
        Task("task5", "s5", "t5", {}, depends_on=["task3"]),
        Task("task6", "s6", "t6", {}, depends_on=["task4", "task5"]),
    ]
    
    print("\nDAG结构：")
    print("  task1")
    print("   ├─→ task2 ─→ task4 ─┐")
    print("   └─→ task3 ─→ task5 ─┴→ task6")
    print("\n预期执行顺序：")
    print("  Level 1: task1")
    print("  Level 2: task2, task3 (并行)")
    print("  Level 3: task4, task5 (并行)")
    print("  Level 4: task6")
    
    results = await orchestrator.execute(tasks)
    
    print(f"\n✅ 成功执行 {len(results)} 个任务")
    print()


async def example_cycle_detection():
    """示例4：循环依赖检测"""
    print("=" * 60)
    print("示例4：循环依赖检测")
    print("=" * 60)
    
    metadata_db = MockMetadataDB()
    orchestrator = MCPOrchestrator(metadata_db=metadata_db)
    
    # 创建循环依赖：task1 → task2 → task3 → task1
    tasks = [
        Task("task1", "s1", "t1", {}, depends_on=["task3"]),  # 依赖task3
        Task("task2", "s2", "t2", {}, depends_on=["task1"]),
        Task("task3", "s3", "t3", {}, depends_on=["task2"]),  # 依赖task2，形成环
    ]
    
    print("\n尝试执行包含循环依赖的任务：")
    print("  task1 → task2 → task3 → task1 (循环！)")
    
    try:
        results = await orchestrator.execute(tasks)
        print("\n❌ 错误：应该检测到循环依赖！")
    except ValueError as e:
        print(f"\n✅ 成功检测到循环依赖：{e}")
    print()


async def example_cache():
    """示例5：缓存机制"""
    print("=" * 60)
    print("示例5：TTL缓存机制")
    print("=" * 60)
    
    metadata_db = MockMetadataDB()
    orchestrator = MCPOrchestrator(
        metadata_db=metadata_db,
        cache_ttl=60  # 60秒缓存
    )
    
    tasks = [
        Task("task1", "service1", "tool1", {"param": "value"})
    ]
    
    print("\n第一次执行（无缓存）：")
    results1 = await orchestrator.execute(tasks)
    print(f"  结果: {results1['task1']}")
    
    print("\n第二次执行（应该命中缓存）：")
    # 重置任务状态
    tasks[0].status = TaskStatus.PENDING
    tasks[0].start_time = None
    tasks[0].end_time = None
    
    results2 = await orchestrator.execute(tasks)
    print(f"  结果: {results2['task1']}")
    print(f"  （注意日志中的 'Cache hit' 消息）")
    print()


async def main():
    """运行所有示例"""
    await example_basic()
    await example_parallel()
    await example_complex_dag()
    await example_cycle_detection()
    await example_cache()
    
    print("=" * 60)
    print("所有示例执行完毕！")
    print("=" * 60)


if __name__ == "__main__":
    # 配置日志
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行示例
    asyncio.run(main())

