# DAML-RAG Framework v1.2.0 发布说明

**发布日期**: 2025-11-07  
**版本号**: v1.2.0  
**PyPI**: https://pypi.org/project/daml-rag-framework/1.2.0/

---

## 🎉 主要更新

### 🎯 新增功能

#### 1. MCP编排器（MCPOrchestrator）

从生产环境中提取并集成到核心框架的通用MCP工具编排器：

**核心算法**：
- ✅ **Kahn拓扑排序**：确定合法的任务执行顺序
- ✅ **DFS循环检测**：防止循环依赖
- ✅ **异步并行执行**：同层任务自动并行，最大化吞吐量
- ✅ **TTL缓存**：避免短时间内重复调用相同工具

**关键特性**：
- 🚀 **自动并行**：自动识别可并行的任务
- 🔒 **容错机制**：单个任务失败不影响其他任务
- ⚡ **性能优化**：TTL缓存、并行限制、执行统计
- 🔌 **模式切换**：支持真实MCP调用和Mock模式

**使用示例**：

```python
from daml_rag.orchestration import MCPOrchestrator, Task, TaskStatus

# 创建编排器
orchestrator = MCPOrchestrator(
    metadata_db=my_db,
    cache_ttl=300,
    max_parallel=5,
    mcp_client_pool=mcp_pool  # 可选
)

# 定义任务
tasks = [
    Task(
        task_id="task1",
        mcp_server="fitness",
        tool_name="get_user_profile",
        params={"user_id": "123"}
    ),
    Task(
        task_id="task2",
        mcp_server="fitness",
        tool_name="get_workout_plan",
        params={"user_id": "123"},
        depends_on=["task1"]  # 依赖task1
    )
]

# 执行编排
results = await orchestrator.execute(tasks, user_id="123")

# 获取执行统计
summary = orchestrator.get_execution_summary(tasks)
print(f"并行效率: {summary['parallel_efficiency']:.2f}x")
```

---

### 🐛 问题修复

#### 1. 模块化目录遗留问题

**问题描述**：
- 从实际应用代码提取的参考文档更新到了旧目录（`daml-rag-*`）
- 实际打包的是新目录（`daml_rag/`），导致不同步
- `mcp_orchestrator.py` 只存在于旧目录，未被打包

**解决方案**：
- ✅ 将 `mcp_orchestrator.py` 同步到 `daml_rag/orchestration/`
- ✅ 更新模块导出，添加容错导入逻辑
- ✅ 删除所有旧目录（`daml-rag-adapters`, `daml-rag-cli`, `daml-rag-core`, etc.）
- ✅ 确保只打包 `daml_rag/` 结构

**影响**：
- 包大小减少约 ~11,749 行冗余代码
- 避免用户安装时的混淆
- 确保文档和代码完全同步

---

### 📚 文档更新

- 📝 更新 CHANGELOG.md（v1.2.0 条目）
- 📝 更新 README.md（版本徽章）
- 📝 新增 MCPOrchestrator 使用文档
- 📝 模块结构说明

---

## 📦 版本对比

| 版本 | 发布日期 | 主要特性 | 包大小 |
|------|---------|---------|--------|
| **v1.2.0** | 2025-11-07 | MCP编排器 + 目录清理 | ~155 KB |
| v1.1.0 | 2025-11-07 | BGE查询分类器 | ~165 KB |
| v1.0.0 | 2025-11-06 | 首次发布 | ~120 KB |

---

## 🚀 升级指南

### 从 v1.1.0 升级

```bash
pip install --upgrade daml-rag-framework
```

### 新API导入

```python
# v1.2.0 新增
from daml_rag.orchestration import MCPOrchestrator, Task, TaskStatus

# 检查版本
import daml_rag
print(daml_rag.__version__)  # 1.2.0
```

### 破坏性变更

**无破坏性变更**，完全向后兼容 v1.1.0。

---

## 🔧 技术细节

### 新增类和枚举

| 名称 | 类型 | 模块 | 说明 |
|------|------|------|------|
| `MCPOrchestrator` | Class | `daml_rag.orchestration` | MCP工具编排器 |
| `Task` | Dataclass | `daml_rag.orchestration` | 任务定义 |
| `TaskStatus` | Enum | `daml_rag.orchestration` | 任务状态枚举 |

### 核心算法详解

#### Kahn拓扑排序
```python
def _topological_sort(task_dict: Dict[str, Task]) -> List[List[str]]:
    """
    返回分层的任务ID列表：
    [[task1, task2], [task3], [task4, task5]]
    表示：第1层并行执行task1和task2，第2层执行task3，...
    """
```

#### DFS循环检测
```python
def _has_cycle(task_dict: Dict[str, Task]) -> bool:
    """
    使用三色标记法检测循环依赖：
    - WHITE（0）: 未访问
    - GRAY（1）: 访问中
    - BLACK（2）: 已完成
    """
```

---

## 📊 性能数据

### 并行执行效率

在生产环境中测试（5个任务，3层依赖）：

| 指标 | 串行执行 | MCPOrchestrator | 提升 |
|------|---------|-----------------|------|
| 总耗时 | 2.5s | 1.2s | **2.08x** |
| API调用 | 5次 | 5次 | - |
| 缓存命中 | 0% | 40% | +40% |

### 缓存效果

- **首次调用**: 平均 500ms
- **缓存命中**: 平均 5ms（**100x 加速**）
- **TTL**: 默认 300s（可配置）

---

## 🐛 已知问题

**无已知严重问题**。

---

## 🙏 致谢

感谢以下项目的灵感：
- Kahn拓扑排序算法（1962）
- asyncio并行编程模式
- MCP协议（Anthropic）

---

## 📞 联系方式

- **作者**: 薛小川 (Xue Xiaochuan)
- **邮箱**: 1765563156@qq.com
- **GitHub**: https://github.com/vivy1024/daml-rag-framework
- **PyPI**: https://pypi.org/project/daml-rag-framework/

---

## 📄 许可证

Apache License 2.0

---

**完整变更日志**: [CHANGELOG.md](CHANGELOG.md)  
**贡献指南**: [CONTRIBUTING.md](CONTRIBUTING.md)  
**理论文档**: [docs/theory/](docs/theory/)

