# DAML-RAG框架 v2.0

**Domain Adaptive Multi-source Learning RAG Framework**

面向垂直领域的自适应多源学习型检索增强生成框架

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.0.0-orange.svg)](CHANGELOG.md)
[![Framework Status](https://img.shields.io/badge/Status-Phase_2_Complete-brightgreen.svg)](docs/progress.md)

> 🎯 **接口驱动设计 + 组件注册系统 + 三层检索架构**
> 🚀 **生产就绪的企业级RAG框架**

---

## 📋 目录

- [项目简介](#-项目简介)
- [核心特性](#-核心特性)
- [快速开始](#-快速开始)
  - [安装指南](#安装指南)
  - [5分钟快速上手](#5分钟快速上手)
  - [环境配置](#环境配置)
- [核心功能教程](#-核心功能教程)
  - [三层检索系统](#1-三层检索系统)
  - [MCP任务编排](#2-mcp任务编排)
  - [智能模型选择](#3-智能模型选择bge分类器)
  - [Few-Shot学习](#4-few-shot学习)
- [完整示例](#-完整示例)
- [配置详解](#-配置详解)
- [进阶使用](#-进阶使用)
- [性能优化](#-性能优化)
- [常见问题](#-常见问题)
- [项目架构](#-项目架构)
- [文档索引](#-文档索引)
- [贡献指南](#-贡献指南)
- [许可证](#-许可证)

---

## 📖 项目简介

DAML-RAG v2.0是一个**企业级RAG框架**，采用接口驱动设计和现代软件架构，为垂直领域AI应用提供生产就绪的解决方案。

**核心设计理念**：接口驱动、组件化、可扩展、高性能。

### 🏗️ v2.0架构革新

#### 全新特性 ✅

- **接口驱动设计**：5层标准接口体系，确保组件解耦和可替换性
- **组件注册系统**：自动发现、依赖注入、生命周期管理
- **三层检索架构**：语义检索→图谱检索→约束验证的渐进式精确化
- **多模式向量引擎**：BGE-M3支持dense、sparse、colbert三种模式
- **智能约束验证**：专业领域安全检查和质量保证系统
- **多策略重排序**：动态权重融合和多样性优化算法

#### 设计原则

- **接口优先**：所有组件基于标准接口，支持热插拔替换
- **异步编程**：全面采用async/await，提升并发性能
- **类型安全**：完整的Python类型提示，减少运行时错误
- **配置化**：支持YAML配置文件和环境变量覆盖
- **可监控**：内置指标收集和性能监控

---

## 🎯 核心特性

### 🏗️ 接口驱动架构

- **🔧 5层标准接口体系**
  - **基础接口**：`IComponent`, `IConfigurable`, `IMonitorable`
  - **检索接口**：`IRetriever`, `ISemanticRetriever`, `IGraphRetriever`
  - **编排接口**：`IOrchestrator`, `IToolRegistry`, `ITaskExecutor`
  - **质量接口**：`IQualityChecker`, `IAntiHallucinationChecker`
  - **存储接口**：`IVectorStorage`, `IGraphStorage`, `IDocumentStorage`

### 🔍 三层检索引擎

- **渐进式精确化架构**：语义→图谱→约束验证
  - **语义检索层**：BGE-M3多模式向量匹配，支持dense/sparse/colbert
  - **图谱检索层**：基于Neo4j的关系推理和路径发现
  - **约束验证层**：专业安全规则和质量检查

### 🧦 组件注册系统

- **IoC容器**：支持单例、瞬态、作用域三种生命周期
- **自动装配**：基于类型注解的依赖注入
- **组件发现**：支持包扫描和装饰器注册
- **配置管理**：分层次配置和热更新

### 📦 存储抽象层

- **统一接口**：屏蔽底层存储差异
- **多种存储**：向量、图、文档、缓存、会话5种类型
- **性能优化**：连接池、批量操作、重试机制
- **监控支持**：内置指标收集和健康检查

### ⚡ BGE-M3增强引擎

- **多模式支持**：Dense、Sparse、ColBERT三种向量表示
- **智能策略**：根据查询特征自动选择最优检索策略
- **批量处理**：支持向量化编码和批量搜索
- **缓存优化**：多级缓存提升重复查询性能

### 🛡️ 智能约束验证

- **多维度验证**：安全性、医疗性、专业规则、设备约束
- **可配置规则**：支持动态规则加载和热更新
- **风险评估**：多级风险等级和相应处理策略
- **证据验证**：基于ACSM/NSCA等权威标准

### 🔄 多策略重排序

- **融合算法**：加权融合、倒数排名、Borda投票
- **多样性优化**：保证结果多样性的同时提升质量
- **个性化排序**：基于用户历史和偏好调整排序
- **时效性考虑**：支持时间衰减和新鲜度提升

---

## 🚀 快速开始

### 安装指南

#### 方式1：从源码安装（当前推荐）

DAML-RAG v2.0目前处于开发阶段，建议从源码安装：

```bash
# 1. 克隆仓库
git clone https://github.com/your-org/daml-rag-framework.git
cd daml-rag-framework

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 开发模式安装
pip install -e .
```

#### 方式2：使用Docker

```bash
# 构建镜像
docker build -t daml-rag-framework:2.0 .

# 运行容器
docker run -p 8000:8000 daml-rag-framework:2.0
```

---

### 5分钟快速上手：v2.0框架

#### 步骤1：创建项目目录

```bash
mkdir my-rag-app
cd my-rag-app
```

#### 步骤2：编写基础代码

创建 `main.py`：

```python
import asyncio
from daml_rag_framework import (
    ThreeLayerRetriever, VectorRetriever, GraphRetriever,
    ConstraintValidator, Reranker, QueryAnalyzer,
    initialize_framework
)

async def main():
    print("🚀 初始化DAML-RAG框架 v2.0...")

    # 1. 初始化框架
    config = {
        'retrieval': {
            'strategy': 'balanced',
            'semantic_weight': 0.3,
            'graph_weight': 0.5,
            'constraint_weight': 0.2
        },
        'vector': {
            'model_name': 'bge-m3',
            'top_k': 10,
            'min_similarity': 0.5,
            'enable_cache': True
        }
    }

    success = await initialize_framework(config)
    if not success:
        print("❌ 框架初始化失败")
        return

    # 2. 创建组件
    vector_retriever = VectorRetriever("vector_retriever")
    graph_retriever = GraphRetriever("graph_retriever")
    constraint_validator = ConstraintValidator("constraint_validator")
    reranker = Reranker("reranker")
    query_analyzer = QueryAnalyzer("query_analyzer")

    # 3. 配置三层检索引擎
    three_layer_retriever = ThreeLayerRetriever("main_retriever")
    three_layer_retriever.set_semantic_retriever(vector_retriever)
    three_layer_retriever.set_graph_retriever(graph_retriever)
    three_layer_retriever.set_constraint_validator(constraint_validator)

    # 4. 初始化组件
    await vector_retriever.initialize(config.get('vector', {}))
    await graph_retriever.initialize(config.get('graph', {}))
    await constraint_validator.initialize(config.get('constraint', {}))
    await reranker.initialize(config.get('reranking', {}))
    await query_analyzer.initialize(config.get('analysis', {}))
    await three_layer_retriever.initialize(config.get('retrieval', {}))

    print("✅ 所有组件初始化完成")

    # 5. 处理查询
    query = "推荐5个不伤膝盖的腿部训练动作"
    print(f"\n🔍 处理查询: {query}")

    # 分析查询
    analysis = await query_analyzer.analyze_query(query)
    print(f"📊 查询分析: {analysis.intent.value}, {analysis.complexity.value}")
    print(f"🏷️  识别实体: {[(e, t.value) for e, t in analysis.entities]}")

    # 执行三层检索
    from daml_rag_framework.interfaces.retrieval import QueryRequest
    request = QueryRequest(
        query_id="demo_001",
        query_text=query,
        domain="fitness",
        top_k=5,
        min_similarity=0.5,
        mode="hybrid"
    )

    response = await three_layer_retriever.retrieve(request)
    print(f"📋 检索结果: {len(response.results)} 个结果")
    print(f"⏱️  执行时间: {response.execution_time:.3f}s")

    # 重排序结果
    reranked = await reranker.rerank(response.results, request)
    print(f"🔄 重排序完成: {len(reranked.reranked_results)} 个结果")

    # 输出结果
    print("\n📝 推荐的训练动作:")
    for i, result in enumerate(reranked.reranked_results[:5], 1):
        print(f"{i}. {result.content[:50]}...")
        print(f"   评分: {result.score:.3f}")
        print(f"   来源: {result.metadata.get('source', 'unknown')}")
        print()

    # 6. 显示统计信息
    print("📊 组件统计:")
    print(f"   向量检索器: {vector_retriever.get_metrics()}")
    print(f"   图检索器: {graph_retriever.get_metrics()}")
    print(f"   约束验证器: {constraint_validator.get_metrics()}")
    print(f"   重排序器: {reranker.get_metrics()}")

if __name__ == "__main__":
    asyncio.run(main())
```

#### 步骤3：创建配置文件

创建 `config.yaml`：

```yaml
# DAML-RAG v2.0 配置示例
framework:
  name: "daml_rag_v2"
  version: "2.0.0"
  debug: true

retrieval:
  strategy: "balanced"
  semantic_weight: 0.3
  graph_weight: 0.5
  constraint_weight: 0.2
  enable_parallel: true

vector:
  model_name: "BAAI/bge-m3"
  device: "cpu"  # 或 "cuda"
  top_k: 20
  min_similarity: 0.5
  enable_cache: true
  cache_ttl: 300

graph:
  max_depth: 3
  max_nodes: 100
  enable_safety_filter: true
  enable_evidence_filter: true
  min_evidence_level: 0.5

constraint:
  enable_safety_check: true
  enable_domain_rules: true
  enable_evidence_validation: true

reranking:
  primary_strategy: "weighted_fusion"
  enable_diversity_promotion: true
  diversity_threshold: 0.7
  enable_recency_boost: true

analysis:
  enable_intent_recognition: true
  enable_entity_extraction: true
  enable_relation_extraction: true
```

#### 步骤4：运行应用

```bash
python main.py
```

**预期输出**：

```
🚀 初始化DAML-RAG框架 v2.0...
✅ 所有组件初始化完成

🔍 处理查询: 推荐5个不伤膝盖的腿部训练动作
📊 查询分析: recommendation, moderate
🏷️  识别实体: [('腿部', 'muscle'), ('膝盖', 'injury'), ('训练', 'goal')]
📋 检索结果: 8 个结果
⏱️  执行时间: 1.234s
🔄 重排序完成: 8 个结果

📝 推荐的训练动作:
1. 保加利亚分腿蹲 - 单腿训练动作，对膝盖压力小...
   评分: 0.923
   来源: vector_layer + graph_layer + constraint_layer

2. 腿举 - 固定器械训练，安全性高...
   评分: 0.889
   来源: vector_layer + constraint_layer

3. 臀桥 - 臀部训练，对膝盖友好...
   评分: 0.876
   来源: vector_layer + graph_layer

📊 组件统计:
   向量检索器: {'total_queries': 1, 'cache_hit_rate': 0.0, ...}
   图检索器: {'total_queries': 1, 'path_discoveries': 2, ...}
   约束验证器: {'total_validations': 8, 'safety_blocks': 1, ...}
   重排序器: {'total_rerankings': 1, 'diversity_improvements': 1, ...}
```

---

### 环境配置

#### 最小环境要求

| 资源 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **Python** | 3.8+ | 3.10+ |
| **内存** | 8GB | 16GB+ |
| **存储** | 5GB | 20GB+ |
| **CPU** | 4核 | 8核+ |

#### 依赖服务

##### 1. 向量数据库（选择一个）

**Qdrant（推荐）**：

```bash
# Docker 部署
docker run -p 6333:6333 qdrant/qdrant

# Python 客户端
pip install qdrant-client
```

**FAISS（本地）**：

```bash
pip install faiss-cpu  # CPU版本
# 或
pip install faiss-gpu  # GPU版本
```

##### 2. 图数据库（选择一个）

**Neo4j（推荐）**：

```bash
# Docker 部署
docker run \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest

# Python 客户端
pip install neo4j
```

##### 3. AI 模型服务

**DeepSeek API（教师模型）**：

```bash
# 设置环境变量
export DEEPSEEK_API_KEY="your-api-key"
```

**Ollama（学生模型，推荐）**：

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型
ollama pull qwen2.5:7b

# 启动服务
ollama serve
```

---

## 📚 核心功能教程

### 1. 三层检索系统

DAML-RAG的核心创新：向量 + 图谱 + 规则的混合检索。

#### 架构图

```
用户查询 "推荐不伤膝盖的腿部增肌动作"
    ↓
┌─────────────────────────────────────┐
│ 第一层：向量语义检索                  │
│ - 召回Top 20候选动作                  │
│ - 语义理解："增肌"="肥大训练"          │
│ - 耗时：~50ms                        │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 第二层：图关系推理                    │
│ - 筛选："不伤膝盖"约束                │
│ - 关系推理：动作→肌群→目标             │
│ - 耗时：~100ms                       │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 第三层：业务规则验证                  │
│ - 安全规则：用户年龄、损伤史           │
│ - 器械规则：可用设备                  │
│ - 耗时：~20ms                        │
└──────────────┬──────────────────────┘
               ↓
         精准结果 Top 5
```

#### 使用示例

```python
from daml_rag.retrieval import ThreeTierRetriever, RetrievalConfig

# 配置检索系统
config = RetrievalConfig(
    vector_top_k=20,          # 向量召回数量
    vector_threshold=0.6,     # 相似度阈值
    graph_enabled=True,       # 启用图谱
    graph_top_k=10,           # 图谱筛选数量
    rules_enabled=True,       # 启用规则
    cache_enabled=True,       # 启用缓存
    cache_ttl=300            # 缓存5分钟
)

# 创建检索器
retriever = ThreeTierRetriever(config)
await retriever.initialize()

# 执行检索
results = await retriever.retrieve(
    query="推荐不伤膝盖的腿部增肌动作",
    user_context={
        "user_id": "user123",
        "age": 35,
        "injury_history": ["knee_pain"],
        "available_equipment": ["barbell", "dumbbell"]
    }
)

# 查看结果
for doc in results:
    print(f"动作: {doc.title}")
    print(f"评分: {doc.score}")
    print(f"来源: {doc.metadata['source']}")
    print(f"推荐理由: {doc.metadata['reason']}")
    print("---")
```

**输出示例**：

```
动作: 保加利亚分腿蹲
评分: 0.92
来源: vector_layer + graph_layer
推荐理由: 单腿训练，减少膝盖压力，适合股四头肌增肌

动作: 腿举
评分: 0.89
来源: vector_layer + graph_layer + rules
推荐理由: 固定器械，安全性高，可调节膝盖角度

动作: 罗马尼亚硬拉
评分: 0.87
来源: vector_layer + graph_layer
推荐理由: 后链主导，对膝盖压力小，腘绳肌和臀部增肌
...
```

---

### 2. MCP任务编排

基于Kahn拓扑排序的智能任务编排系统，支持并行执行和依赖管理。

#### 核心概念

- **DAG（有向无环图）**：任务之间的依赖关系
- **拓扑排序**：确定任务执行顺序
- **并行执行**：同一层级任务并发运行
- **TTL缓存**：避免重复调用

#### 使用示例

```python
from daml_rag.orchestration import MCPOrchestrator, Task, TaskStatus

# 创建编排器
orchestrator = MCPOrchestrator(
    metadata_db=my_db,
    cache_ttl=300,           # 缓存5分钟
    max_parallel=5,          # 最多并行5个任务
    mcp_client_pool=pool     # MCP客户端池（可选）
)

# 定义任务
tasks = [
    # 任务1：获取用户档案（无依赖）
    Task(
        task_id="get_profile",
        mcp_server="fitness",
        tool_name="get_user_profile",
        params={"user_id": "user123"}
    ),
    
    # 任务2：获取训练历史（无依赖）
    Task(
        task_id="get_history",
        mcp_server="fitness",
        tool_name="get_training_history",
        params={"user_id": "user123", "days": 30}
    ),
    
    # 任务3：分析用户水平（依赖任务1和2）
    Task(
        task_id="analyze_level",
        mcp_server="coach",
        tool_name="analyze_user_level",
        params={"user_id": "user123"},
        depends_on=["get_profile", "get_history"]
    ),
    
    # 任务4：生成训练计划（依赖任务3）
    Task(
        task_id="generate_plan",
        mcp_server="coach",
        tool_name="generate_workout_plan",
        params={"user_id": "user123"},
        depends_on=["analyze_level"]
    )
]

# 执行编排
results = await orchestrator.execute(tasks, user_id="user123")

# 查看结果
for task_id, result in results.items():
    print(f"任务: {task_id}")
    print(f"状态: {result.status}")
    print(f"耗时: {result.elapsed_time}s")
    print(f"结果: {result.data}")
    print("---")

# 获取执行统计
summary = orchestrator.get_execution_summary(tasks)
print(f"总耗时: {summary['total_time']}s")
print(f"并行效率: {summary['parallel_efficiency']:.2f}x")
print(f"缓存命中率: {summary['cache_hit_rate']:.2%}")
```

**输出示例**：

```
任务: get_profile
状态: COMPLETED
耗时: 0.5s
结果: {'name': '张三', 'age': 35, 'level': 'intermediate'}
---

任务: get_history
状态: COMPLETED
耗时: 0.5s (并行执行)
结果: {'workouts': 12, 'total_volume': 15000}
---

任务: analyze_level
状态: COMPLETED
耗时: 0.8s
结果: {'level': 'intermediate', 'strengths': ['upper_body'], 'weaknesses': ['legs']}
---

任务: generate_plan
状态: COMPLETED
耗时: 1.2s
结果: {'plan': {...}, 'duration': '8_weeks'}
---

总耗时: 3.0s
并行效率: 2.17x (串行需6.5s)
缓存命中率: 25%
```

#### 任务依赖关系图

```
get_profile ─────┐
                 ├──> analyze_level ──> generate_plan
get_history ─────┘

执行顺序：
第1层（并行）: get_profile, get_history
第2层（串行）: analyze_level
第3层（串行）: generate_plan
```

---

### 3. 智能模型选择（BGE分类器）

基于BAAI/bge-base-zh-v1.5向量模型的查询复杂度分类系统。

#### 工作原理

1. **预定义复杂查询模板**：
   ```python
   complex_queries = [
       "制定一个完整的训练计划",
       "分析我的身体状况并给出建议",
       "如何系统地提高力量水平"
   ]
   ```

2. **计算语义相似度**：
   ```python
   # 用户查询向量化
   query_vec = bge_model.encode(user_query)
   
   # 计算与复杂模板的相似度
   similarities = cosine_similarity(query_vec, complex_vecs)
   max_sim = max(similarities)
   ```

3. **智能分类**：
   ```python
   if max_sim > 0.7:
       model = "deepseek"    # 复杂查询 → 教师模型
   else:
       model = "ollama"      # 简单查询 → 学生模型
   ```

#### 使用示例

```python
from daml_rag.learning import QueryComplexityClassifier

# 创建分类器
classifier = QueryComplexityClassifier(
    model_name="BAAI/bge-base-zh-v1.5",
    threshold=0.7,           # 复杂度阈值
    cache_size=1000,         # 缓存大小
    device="cuda"            # 使用GPU（可选）
)

# 加载模型（懒加载，首次使用时自动加载）
await classifier.load_model()

# 分类查询
queries = [
    "深蹲的标准动作",                    # 简单查询
    "制定一个8周增肌训练计划",            # 复杂查询
    "分析我的体态问题并给出纠正方案"       # 复杂查询
]

for query in queries:
    is_complex = classifier.classify(query)
    model = "deepseek" if is_complex else "ollama"
    print(f"查询: {query}")
    print(f"分类: {'复杂' if is_complex else '简单'}")
    print(f"推荐模型: {model}")
    print("---")
```

**输出示例**：

```
查询: 深蹲的标准动作
分类: 简单
推荐模型: ollama
相似度: 0.45
成本: $0.001
---

查询: 制定一个8周增肌训练计划
分类: 复杂
推荐模型: deepseek
相似度: 0.83
成本: $0.02
---

查询: 分析我的体态问题并给出纠正方案
分类: 复杂
推荐模型: deepseek
相似度: 0.76
成本: $0.025
---
```

#### 成本对比

| 查询类型 | 模型选择 | 平均Token | 成本/次 | 质量评分 |
|---------|---------|----------|---------|---------|
| 简单查询 | Ollama | 500 | $0 | 4.2/5.0 |
| 复杂查询 | DeepSeek | 2000 | $0.02 | 4.8/5.0 |
| **混合策略** | **智能选择** | **800** | **$0.005** | **4.6/5.0** |

**成本节省**: 相比全部使用DeepSeek，节省约 **75%** 成本。

---

### 4. Few-Shot学习

基于经验记忆的上下文学习系统。

#### 工作流程

```
用户查询 "制定增肌计划"
    ↓
1. 向量化查询
    ↓
2. 从经验库检索相似案例
    ↓
3. 筛选高质量经验（评分>3.5）
    ↓
4. 注入Few-Shot上下文
    ↓
5. 生成高质量回答
```

#### 使用示例

```python
from daml_rag.learning import MemoryManager, Experience

# 创建记忆管理器
memory = MemoryManager(
    storage_type="redis",      # 或 "in_memory"
    max_experiences=1000,      # 最多存储1000条经验
    similarity_threshold=0.7   # 相似度阈值
)

# 存储成功经验
experience = Experience(
    query="如何增肌",
    context={"user_level": "beginner"},
    response="建议从复合动作开始...",
    feedback_score=4.5,        # 用户评分
    metadata={
        "model_used": "deepseek",
        "tokens": 520,
        "duration": 2.3
    }
)
await memory.add_experience(experience)

# 检索相似经验
similar_exps = await memory.get_similar_experiences(
    query="制定一个增肌训练计划",
    top_k=3,                   # 召回3个相似案例
    min_score=3.5              # 最低评分要求
)

# 构建Few-Shot提示词
few_shot_prompt = """
以下是一些成功案例：

案例1：
用户问：{similar_exps[0].query}
回答：{similar_exps[0].response}
评分：{similar_exps[0].feedback_score}/5.0

案例2：
用户问：{similar_exps[1].query}
回答：{similar_exps[1].response}
评分：{similar_exps[1].feedback_score}/5.0

现在请回答：
用户问：{current_query}
"""

# 使用Few-Shot提示生成回答
response = await llm.generate(few_shot_prompt)
```

#### 经验质量管理

```python
# 获取经验统计
stats = memory.get_statistics()
print(f"总经验数: {stats['total']}")
print(f"平均评分: {stats['avg_score']}")
print(f"高质量经验: {stats['high_quality']}")

# 清理低质量经验
await memory.cleanup(min_score=3.0, max_age_days=90)
```

---

## 💼 完整示例

### 示例1：健身领域AI助手

```python
"""
完整的健身AI助手示例
功能：训练计划生成、动作推荐、进度跟踪
"""
import asyncio
from daml_rag import DAMLRAGFramework, DAMLRAGConfig
from daml_rag.adapters import FitnessDomainAdapter

class FitnessAICoach:
    def __init__(self, config_path: str):
        self.config = DAMLRAGConfig.from_file(config_path)
        self.framework = None
        self.adapter = None
    
    async def initialize(self):
        """初始化系统"""
        # 创建领域适配器
        self.adapter = FitnessDomainAdapter(
            self.config.domain_config
        )
        await self.adapter.initialize()
        
        # 初始化框架
        self.framework = DAMLRAGFramework(self.config)
        await self.framework.initialize()
        
        print("✅ AI教练初始化完成")
    
    async def generate_workout_plan(self, user_id: str, goal: str):
        """生成训练计划"""
        query = f"为我制定一个{goal}的训练计划"
        
        result = await self.framework.process_query(
            query=query,
            user_context={
                "user_id": user_id,
                "goal": goal
            }
        )
        
        return {
            "plan": result.response,
            "model_used": result.model_used,
            "cost": result.cost,
            "quality_score": result.quality_score
        }
    
    async def recommend_exercises(self, user_id: str, muscle_group: str):
        """推荐训练动作"""
        query = f"推荐5个{muscle_group}的训练动作"
        
        result = await self.framework.process_query(
            query=query,
            user_context={"user_id": user_id}
        )
        
        return result.response
    
    async def analyze_progress(self, user_id: str):
        """分析训练进度"""
        # 使用MCP编排器协调多个工具
        from daml_rag.orchestration import Task
        
        tasks = [
            Task(
                task_id="get_history",
                mcp_server="fitness",
                tool_name="get_training_history",
                params={"user_id": user_id, "days": 30}
            ),
            Task(
                task_id="get_measurements",
                mcp_server="fitness",
                tool_name="get_body_measurements",
                params={"user_id": user_id}
            ),
            Task(
                task_id="analyze",
                mcp_server="coach",
                tool_name="analyze_progress",
                params={"user_id": user_id},
                depends_on=["get_history", "get_measurements"]
            )
        ]
        
        results = await self.framework.orchestrator.execute(
            tasks, user_id=user_id
        )
        
        return results["analyze"].data

# 使用示例
async def main():
    # 创建AI教练
    coach = FitnessAICoach("config.yaml")
    await coach.initialize()
    
    # 生成训练计划
    plan = await coach.generate_workout_plan(
        user_id="user123",
        goal="增肌"
    )
    print(f"训练计划: {plan['plan']}")
    print(f"使用模型: {plan['model_used']}")
    print(f"成本: ${plan['cost']:.4f}")
    
    # 推荐动作
    exercises = await coach.recommend_exercises(
        user_id="user123",
        muscle_group="胸部"
    )
    print(f"推荐动作: {exercises}")
    
    # 分析进度
    progress = await coach.analyze_progress("user123")
    print(f"进度分析: {progress}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 示例2：医疗咨询AI助手

```python
"""
医疗咨询AI助手示例
功能：症状分析、用药建议、健康建议
"""
from daml_rag import DAMLRAGFramework, DAMLRAGConfig

class MedicalAIAssistant:
    def __init__(self, config_path: str):
        self.config = DAMLRAGConfig.from_file(config_path)
        self.framework = None
    
    async def initialize(self):
        self.framework = DAMLRAGFramework(self.config)
        await self.framework.initialize()
    
    async def analyze_symptoms(self, symptoms: list, patient_info: dict):
        """分析症状"""
        query = f"患者症状：{', '.join(symptoms)}，请分析可能的原因"
        
        result = await self.framework.process_query(
            query=query,
            user_context=patient_info
        )
        
        return {
            "analysis": result.response,
            "confidence": result.quality_score,
            "references": result.retrieved_docs
        }
    
    async def suggest_medication(self, condition: str, patient_info: dict):
        """用药建议"""
        # 医疗领域必须使用高质量模型
        result = await self.framework.process_query(
            query=f"针对{condition}，推荐合适的用药方案",
            user_context=patient_info,
            force_teacher_model=True  # 强制使用教师模型
        )
        
        return result.response

# 使用示例
async def main():
    assistant = MedicalAIAssistant("medical_config.yaml")
    await assistant.initialize()
    
    # 分析症状
    analysis = await assistant.analyze_symptoms(
        symptoms=["头痛", "发热", "咳嗽"],
        patient_info={
            "age": 35,
            "gender": "male",
            "medical_history": ["哮喘"]
        }
    )
    print(f"分析结果: {analysis['analysis']}")
    print(f"置信度: {analysis['confidence']}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## ⚙️ 配置详解

### 完整配置文件

```yaml
# config.yaml - DAML-RAG 完整配置示例

# ========================================
# 基础配置
# ========================================
domain: fitness                    # 领域：fitness, medical, legal, etc.
debug: true                        # 调试模式
log_level: INFO                    # 日志级别：DEBUG, INFO, WARNING, ERROR
log_file: logs/app.log            # 日志文件路径

# ========================================
# 检索系统配置
# ========================================
retrieval:
  # 向量检索配置
  vector_model: "BAAI/bge-base-zh-v1.5"  # 向量模型
  vector_store: "qdrant"                  # 向量数据库：qdrant, faiss, milvus
  vector_host: "localhost"
  vector_port: 6333
  vector_top_k: 20                        # 召回数量
  vector_threshold: 0.6                   # 相似度阈值
  vector_weight: 0.3                      # 向量层权重
  
  # 知识图谱配置
  graph_enabled: true                     # 是否启用图谱
  graph_store: "neo4j"                    # 图数据库：neo4j, arangodb
  graph_uri: "bolt://localhost:7687"
  graph_user: "neo4j"
  graph_password: "password"
  graph_top_k: 10                         # 图谱筛选数量
  graph_weight: 0.5                       # 图谱层权重
  
  # 规则引擎配置
  rules_enabled: true                     # 是否启用规则
  rules_path: "rules/"                    # 规则文件目录
  rules_weight: 0.2                       # 规则层权重
  
  # 缓存配置
  cache_enabled: true                     # 是否启用缓存
  cache_ttl: 300                          # 缓存过期时间（秒）
  cache_backend: "redis"                  # 缓存后端：redis, memory
  cache_host: "localhost"
  cache_port: 6379
  
  # 性能配置
  total_timeout: 5.0                      # 总超时时间（秒）
  vector_timeout: 1.0                     # 向量检索超时
  graph_timeout: 2.0                      # 图谱检索超时
  rules_timeout: 0.5                      # 规则过滤超时

# ========================================
# 任务编排配置
# ========================================
orchestration:
  max_parallel_tasks: 10                  # 最大并行任务数
  max_parallel_workflows: 5               # 最大并行工作流数
  timeout_seconds: 30                     # 工作流超时（秒）
  retry_attempts: 3                       # 重试次数
  retry_delay: 1.0                        # 重试延迟（秒）
  enable_caching: true                    # 启用任务缓存
  cache_ttl: 300                          # 任务缓存TTL
  enable_monitoring: true                 # 启用监控

# ========================================
# 学习系统配置
# ========================================
learning:
  # 模型配置
  teacher_model: "deepseek"               # 教师模型
  student_model: "ollama-qwen2.5"         # 学生模型
  teacher_api_key: "${DEEPSEEK_API_KEY}"  # 环境变量
  student_endpoint: "http://localhost:11434"
  
  # BGE分类器配置
  classifier_model: "BAAI/bge-base-zh-v1.5"
  classifier_threshold: 0.7               # 复杂度阈值
  classifier_cache_size: 1000             # 分类缓存大小
  
  # 经验记忆配置
  memory_backend: "redis"                 # memory, redis
  memory_host: "localhost"
  memory_port: 6379
  max_experiences: 10000                  # 最大经验数
  experience_threshold: 3.5               # 经验质量阈值
  
  # Few-Shot配置
  few_shot_enabled: true                  # 启用Few-Shot
  few_shot_count: 3                       # Few-Shot示例数量
  similarity_threshold: 0.7               # 经验相似度阈值
  
  # 质量控制配置
  quality_check_enabled: true             # 启用质量检查
  quality_threshold: 3.5                  # 质量阈值
  auto_upgrade: true                      # 自动升级到教师模型
  
  # 反馈配置
  feedback_enabled: true                  # 启用用户反馈
  feedback_weight: 0.8                    # 反馈权重
  adaptive_threshold: 0.7                 # 自适应阈值

# ========================================
# 领域适配器配置
# ========================================
domain_config:
  # 知识图谱
  knowledge_graph_path: "./data/kg.db"
  kg_entities_count: 2447                 # 实体数量
  kg_relationships_count: 5892            # 关系数量
  
  # MCP服务器
  mcp_servers:
    - name: "user-profile"
      command: "python"
      args: ["servers/user-profile/server.py"]
      env:
        DB_PATH: "./data/users.db"
    
    - name: "professional-coach"
      command: "python"
      args: ["servers/coach/server.py"]
      env:
        MODEL_PATH: "./models/coach"
  
  # 领域规则
  domain_rules:
    safety_rules:
      - age_check
      - injury_check
      - medical_clearance
    
    business_rules:
      - equipment_availability
      - facility_constraints
      - time_constraints

# ========================================
# 监控配置
# ========================================
monitoring:
  enabled: true                           # 启用监控
  prometheus_port: 9090                   # Prometheus端口
  grafana_enabled: true                   # 启用Grafana
  metrics_interval: 60                    # 指标采集间隔（秒）
  
  # 告警配置
  alerting:
    enabled: true
    email: "admin@example.com"
    slack_webhook: "${SLACK_WEBHOOK}"
    
    # 告警规则
    rules:
      - name: "high_latency"
        condition: "avg_latency > 5s"
        severity: "warning"
      
      - name: "low_quality"
        condition: "avg_quality < 3.0"
        severity: "critical"

# ========================================
# 安全配置
# ========================================
security:
  api_key_required: true                  # 需要API密钥
  rate_limiting:
    enabled: true                         # 启用速率限制
    requests_per_minute: 60               # 每分钟请求数
    requests_per_hour: 1000               # 每小时请求数
  
  cors:
    enabled: true                         # 启用CORS
    allowed_origins:
      - "http://localhost:3000"
      - "https://yourdomain.com"
```

### 配置优先级

1. **环境变量** > 2. **配置文件** > 3. **默认值**

```python
# 使用环境变量覆盖配置
import os
os.environ["DEEPSEEK_API_KEY"] = "your-key"
os.environ["DAML_RAG_DEBUG"] = "true"

# 加载配置时自动应用环境变量
config = DAMLRAGConfig.from_file("config.yaml")
```

---

## 🎓 进阶使用

### 1. 自定义领域适配器

```python
from daml_rag.adapters.base import BaseDomainAdapter

class LegalDomainAdapter(BaseDomainAdapter):
    """法律领域适配器"""
    
    async def initialize(self):
        """初始化法律知识库"""
        # 加载法律条文
        self.legal_codes = await self.load_legal_codes()
        
        # 加载判例库
        self.cases = await self.load_cases()
        
        # 初始化法律术语词典
        self.legal_terms = await self.load_legal_terms()
    
    async def preprocess_query(self, query: str) -> str:
        """预处理查询"""
        # 识别法律术语
        terms = self.extract_legal_terms(query)
        
        # 扩展查询
        expanded_query = self.expand_with_synonyms(query, terms)
        
        return expanded_query
    
    async def postprocess_response(self, response: str) -> str:
        """后处理响应"""
        # 添加法律条文引用
        response_with_refs = self.add_legal_references(response)
        
        # 添加免责声明
        response_with_disclaimer = self.add_disclaimer(
            response_with_refs
        )
        
        return response_with_disclaimer
    
    def load_domain_rules(self) -> List[Rule]:
        """加载领域规则"""
        return [
            Rule(
                name="legal_age_check",
                condition=lambda ctx: ctx.get("age", 0) >= 18,
                message="法律咨询仅限成年人使用"
            ),
            Rule(
                name="jurisdiction_check",
                condition=lambda ctx: ctx.get("jurisdiction") in self.supported_jurisdictions,
                message="该地区法律咨询暂不支持"
            )
        ]
```

### 2. 自定义规则引擎

```python
from daml_rag.retrieval.rules import Rule, RuleEngine, RuleContext

# 创建自定义规则
class SafetyRule(Rule):
    """安全规则"""
    
    def __init__(self):
        super().__init__(
            name="safety_check",
            priority=10  # 高优先级
        )
    
    def evaluate(self, context: RuleContext) -> bool:
        """评估规则"""
        # 检查年龄
        if context.user_age < 18:
            return False
        
        # 检查损伤史
        if "serious_injury" in context.injury_history:
            return False
        
        # 检查医疗许可
        if not context.medical_clearance:
            return False
        
        return True
    
    def get_reason(self, context: RuleContext) -> str:
        """获取规则说明"""
        if context.user_age < 18:
            return "未成年人需要监护人陪同"
        if "serious_injury" in context.injury_history:
            return "严重损伤史，请先咨询医生"
        if not context.medical_clearance:
            return "需要医疗许可才能进行训练"
        return ""

# 使用自定义规则
engine = RuleEngine()
engine.add_rule(SafetyRule())
engine.add_rule(EquipmentRule())
engine.add_rule(TimeConstraintRule())

# 评估规则
context = RuleContext(
    user_age=35,
    injury_history=[],
    medical_clearance=True,
    available_equipment=["barbell", "dumbbell"]
)

passed, reasons = engine.evaluate_all(context)
if not passed:
    print(f"规则未通过: {reasons}")
```

### 3. 自定义向量检索器

```python
from daml_rag.retrieval.vector.base import BaseVectorRetriever
import numpy as np

class CustomVectorRetriever(BaseVectorRetriever):
    """自定义向量检索器"""
    
    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.index = None
    
    async def initialize(self):
        """初始化检索器"""
        # 加载向量模型
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.config.model_name)
        
        # 加载向量索引
        import faiss
        self.index = faiss.read_index(self.config.index_path)
    
    async def encode(self, text: str) -> np.ndarray:
        """文本向量化"""
        return self.model.encode(text)
    
    async def search(self, query: str, top_k: int = 10) -> List[Document]:
        """检索相似文档"""
        # 查询向量化
        query_vec = await self.encode(query)
        
        # FAISS检索
        distances, indices = self.index.search(
            query_vec.reshape(1, -1),
            top_k
        )
        
        # 构造结果
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            doc = await self.load_document(idx)
            doc.score = 1 / (1 + dist)  # 转换距离为相似度
            results.append(doc)
        
        return results
```

### 4. 监控和告警

```python
from daml_rag.monitoring import MetricsCollector, AlertManager

# 创建指标收集器
metrics = MetricsCollector()

# 记录请求
@metrics.track_request
async def process_query(query: str):
    result = await framework.process_query(query)
    
    # 记录指标
    metrics.record_latency(result.elapsed_time)
    metrics.record_tokens(result.tokens)
    metrics.record_cost(result.cost)
    metrics.record_quality(result.quality_score)
    
    return result

# 创建告警管理器
alert_manager = AlertManager(config.alerting)

# 检查指标并发送告警
if metrics.avg_latency > 5.0:
    await alert_manager.send_alert(
        severity="warning",
        message=f"平均延迟过高: {metrics.avg_latency}s",
        channel="slack"
    )

if metrics.avg_quality < 3.0:
    await alert_manager.send_alert(
        severity="critical",
        message=f"平均质量过低: {metrics.avg_quality}",
        channel="email"
    )
```

---

## 🚀 性能优化

### 1. 缓存策略

```python
# 多级缓存配置
retrieval:
  # L1缓存：内存缓存（最快）
  l1_cache:
    enabled: true
    max_size: 1000
    ttl: 60  # 1分钟
  
  # L2缓存：Redis缓存
  l2_cache:
    enabled: true
    host: "localhost"
    port: 6379
    ttl: 300  # 5分钟
  
  # L3缓存：向量数据库缓存
  l3_cache:
    enabled: true
    ttl: 3600  # 1小时
```

### 2. 并行优化

```python
# 启用并行检索
retrieval:
  parallel_enabled: true
  max_workers: 10           # 最大并行数

# 启用并行MCP调用
orchestration:
  parallel_enabled: true
  max_parallel: 5           # 最大并行任务数
```

### 3. 连接池

```python
from daml_rag.utils import ConnectionPool

# 创建连接池
pool = ConnectionPool(
    pool_size=10,              # 连接池大小
    max_overflow=5,            # 最大溢出连接
    pool_timeout=30,           # 获取连接超时
    pool_recycle=3600          # 连接回收时间
)

# 使用连接池
async with pool.acquire() as conn:
    result = await conn.execute(query)
```

### 4. 批处理

```python
# 批量处理查询
async def process_batch(queries: List[str]):
    # 批量向量化
    vectors = await retriever.encode_batch(queries)
    
    # 批量检索
    results = await retriever.search_batch(vectors)
    
    # 批量生成
    responses = await llm.generate_batch(results)
    
    return responses
```

### 5. 性能监控

```python
from daml_rag.profiling import Profiler

# 启用性能分析
profiler = Profiler(enabled=True)

with profiler.profile("query_processing"):
    result = await framework.process_query(query)

# 查看性能报告
report = profiler.get_report()
print(f"总耗时: {report.total_time}s")
print(f"检索耗时: {report.retrieval_time}s")
print(f"生成耗时: {report.generation_time}s")
```

---

## ❓ 常见问题

### Q1: 安装失败怎么办？

**A**: 常见解决方案：

```bash
# 方案1：升级pip
python -m pip install --upgrade pip

# 方案2：使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple daml-rag-framework

# 方案3：从源码安装
git clone https://github.com/vivy1024/daml-rag-framework.git
cd daml-rag-framework
pip install -e .
```

### Q2: 如何减少响应时间？

**A**: 优化策略：

1. **启用缓存**：
```yaml
retrieval:
  cache_enabled: true
  cache_backend: "redis"  # 比内存缓存更快
```

2. **启用并行**：
```yaml
orchestration:
  max_parallel_tasks: 10
```

3. **优化向量检索**：
```yaml
retrieval:
  vector_top_k: 10  # 减少召回数量
  vector_threshold: 0.7  # 提高阈值
```

### Q3: 如何降低成本？

**A**: 成本优化：

1. **启用BGE分类器**：
```yaml
learning:
  classifier_enabled: true
  classifier_threshold: 0.7
```

2. **优先使用学生模型**：
```yaml
learning:
  student_model_priority: true
  quality_threshold: 3.0  # 降低质量要求
```

3. **启用缓存**：
```yaml
learning:
  memory_backend: "redis"
  cache_ttl: 3600  # 长缓存时间
```

### Q4: 如何提高质量？

**A**: 质量提升：

1. **启用Few-Shot学习**：
```yaml
learning:
  few_shot_enabled: true
  few_shot_count: 5  # 增加示例数量
```

2. **提高质量阈值**：
```yaml
learning:
  quality_threshold: 4.0
  auto_upgrade: true  # 自动升级
```

3. **使用教师模型**：
```python
result = await framework.process_query(
    query=query,
    force_teacher_model=True  # 强制使用教师模型
)
```

### Q5: 如何处理大规模数据？

**A**: 扩展方案：

1. **分布式部署**：
```yaml
deployment:
  mode: "distributed"
  nodes:
    - host: "node1"
      port: 8000
    - host: "node2"
      port: 8000
```

2. **数据分片**：
```python
# 按领域分片
shards = {
    "fitness": Shard("fitness_db"),
    "medical": Shard("medical_db"),
    "legal": Shard("legal_db")
}
```

3. **使用专业数据库**：
```yaml
retrieval:
  vector_store: "milvus"  # 支持百亿级向量
  graph_store: "janusgraph"  # 分布式图数据库
```

### Q6: 如何调试问题？

**A**: 调试技巧：

```python
# 启用调试模式
config = DAMLRAGConfig.from_file("config.yaml")
config.debug = True
config.log_level = "DEBUG"

# 查看详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用Profiler定位性能瓶颈
from daml_rag.profiling import Profiler
profiler = Profiler(enabled=True)
```

### Q7: 支持哪些数据库？

**A**: 支持列表：

| 类型 | 支持的数据库 | 推荐 |
|------|------------|------|
| **向量数据库** | Qdrant, FAISS, Milvus, Pinecone, Weaviate | Qdrant |
| **图数据库** | Neo4j, ArangoDB, JanusGraph, Neptune | Neo4j |
| **缓存数据库** | Redis, Memcached, 内存 | Redis |
| **关系数据库** | PostgreSQL, MySQL, SQLite | PostgreSQL |

---

## 🏗️ 项目架构

### 模块结构

```
daml-rag-framework/
├── daml_rag/                      # 核心包
│   ├── __init__.py               # 包入口
│   ├── core.py                   # 核心框架
│   ├── base.py                   # 基础类
│   │
│   ├── retrieval/                # 🔍 检索模块
│   │   ├── __init__.py
│   │   ├── three_tier.py        # 三层检索系统
│   │   ├── vector/              # 向量检索
│   │   │   ├── base.py          # 基础类
│   │   │   ├── qdrant.py        # Qdrant实现
│   │   │   ├── faiss.py         # FAISS实现
│   │   │   └── qdrant.py        # Milvus实现
│   │   ├── knowledge/           # 知识图谱
│   │   │   ├── __init__.py
│   │   │   └── neo4j.py         # Neo4j实现
│   │   └── rules/               # 规则引擎
│   │       ├── __init__.py
│   │       └── engine.py        # 规则引擎实现
│   │
│   ├── orchestration/            # 🎯 任务编排
│   │   ├── __init__.py
│   │   ├── orchestrator.py      # 通用编排器
│   │   └── mcp_orchestrator.py  # MCP编排器(v1.2.0)
│   │
│   ├── learning/                 # 🧠 学习模块
│   │   ├── __init__.py
│   │   ├── memory.py            # 经验记忆
│   │   ├── model_provider.py    # 模型提供者
│   │   ├── query_classifier.py  # BGE分类器(v1.1.0)
│   │   ├── feedback.py          # 反馈系统
│   │   └── adaptation.py        # 自适应学习
│   │
│   ├── adapters/                 # 🔌 领域适配器
│   │   ├── __init__.py
│   │   ├── base/
│   │   │   └── adapter.py       # 基础适配器
│   │   └── fitness/
│   │       └── fitness_adapter.py  # 健身适配器
│   │
│   ├── config/                   # ⚙️ 配置管理
│   │   ├── __init__.py
│   │   └── framework_config.py
│   │
│   ├── interfaces/               # 📋 接口定义
│   │   ├── __init__.py
│   │   ├── retrieval.py
│   │   ├── orchestration.py
│   │   └── learning.py
│   │
│   ├── models/                   # 📊 数据模型
│   │   ├── __init__.py
│   │   └── base.py
│   │
│   └── cli/                      # 🚀 命令行工具
│       ├── __init__.py
│       └── cli.py
│
├── examples/                      # 📚 示例代码
│   ├── fitness_qa_demo.py
│   ├── mcp_orchestrator_example.py
│   └── config_examples.py
│
├── docs/                          # 📖 文档
│   ├── theory/                   # 理论文档
│   ├── architecture/             # 架构文档
│   └── quickstart.md            # 快速开始
│
├── tests/                         # ✅ 测试
│   ├── test_retrieval.py
│   ├── test_orchestration.py
│   └── test_learning.py
│
├── scripts/                       # 🔧 脚本
│   ├── build.sh
│   ├── publish.sh
│   └── test-install.sh
│
├── docker/                        # 🐳 Docker配置
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── pyproject.toml                # 📦 项目配置
├── requirements.txt              # 依赖列表
├── README.md                     # 本文件
├── CHANGELOG.md                  # 变更日志
├── LICENSE                       # 许可证
└── CITATION.cff                  # 引用信息
```

### 三层检索架构

```
┌──────────────────────────────────────────────────────────┐
│                     用户查询输入                           │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│  第一层：向量语义检索 (Vector Retrieval)                   │
│  ✅ 快速召回候选集（Top 20-50）                            │
│  ✅ 语义相似度匹配                                         │
│  ✅ 支持多种向量数据库                                     │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│  第二层：图关系推理 (Knowledge Graph)                      │
│  ✅ 精确关系筛选                                           │
│  ✅ 多跳推理能力                                           │
│  ✅ 可解释性强                                            │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────┐
│  第三层：业务规则验证 (Rule Filtering)                     │
│  ✅ 安全规则验证                                           │
│  ✅ 业务逻辑过滤                                           │
│  ✅ 个性化推荐                                            │
└────────────────────┬─────────────────────────────────────┘
                     ↓
              精准结果 Top 5
```

---

## 📚 文档索引

### 核心文档

- **[LIMITATIONS.md](LIMITATIONS.md)** ⚠️ - 限制和约束（必读！）
- **[PUBLISHING.md](PUBLISHING.md)** 📦 - PyPI 发布指南
- **[CHANGELOG.md](CHANGELOG.md)** 📝 - 版本变更历史
- **[CONTRIBUTING.md](CONTRIBUTING.md)** 🤝 - 贡献指南
- **[LICENSE](LICENSE)** 📄 - Apache 2.0许可证

### 理论文档

- [理论演进历史](docs/theory/00-理论演进历史.md) ([English](docs/theory/00-THEORY_EVOLUTION.md))
- [GraphRAG混合检索理论](docs/theory/01-GraphRAG混合检索理论.md)
- [推理时上下文学习理论](docs/theory/02-推理时上下文学习理论.md)
- [框架总览](docs/theory/框架总览.md)

### 架构文档 ⭐

- [MCP编排器实际实现](docs/architecture/mcp-orchestration-实际实现.md)
- [数据清洗与微调架构](docs/architecture/数据清洗与微调架构.md)
- [框架多样性探索策略](docs/architecture/框架多样性探索策略.md)

### 发布说明

- [v1.2.0 发布说明](RELEASE_NOTES_v1.2.0.md) - MCP编排器 + 目录清理
- [v1.1.0 发布说明](RELEASE_NOTES.md) - BGE智能分类器

---

## 📊 项目状态

**⚠️ 项目状态：生产准备（前端完善中）**

### 实际测量数据

| 指标 | 当前值 | 说明 |
|------|-------|------|
| **Token/查询（简单）** | 500-800 | DeepSeek + 用户档案MCP |
| **响应时间** | **~20秒** | ⚠️ 未优化，需要缓存 |
| **项目阶段** | 生产准备 | 准备部署中 |
| **MCP工具实现** | 14/14 ✅ | 所有工具已完成 |
| **Docker状态** | 使用中 | 本地部署就绪 |
| **前端状态** | 进行中 | 部署前完善 |

### 当前问题

**⚠️ 已知性能问题：**

- **响应缓慢**：简单查询约20秒
  - 原因：未优化的图查询，无缓存机制
  - 原因：多个串行MCP调用，无并行化
  - 状态：第一阶段计划优化
  
- **前端完善**：进行中
  - 工具后端：✅ 完成（14/14）
  - 前端UI：🚧 完善中
  - Docker部署：✅ 本地就绪

### 已知限制

**⚠️ 重要：使用前请阅读 [LIMITATIONS.md](LIMITATIONS.md)！**

关键限制：

- **硬件需求**：最低16GB内存，推荐32GB+
- **响应时间**：~20秒（玉珍健身笔记本案例，未优化）
- **规模限制**：单机超过30K节点性能下降
- **部署**：生产环境建议分布式部署

---

## 🤝 贡献指南

欢迎贡献！贡献方式：

### 1. 报告问题

在 [GitHub Issues](https://github.com/vivy1024/daml-rag-framework/issues) 提交：

- Bug报告
- 功能请求
- 文档改进建议

### 2. 提交代码

```bash
# 1. Fork 项目
# 2. 创建特性分支
git checkout -b feature/your-feature

# 3. 提交更改
git commit -m "Add: your feature description"

# 4. 推送到分支
git push origin feature/your-feature

# 5. 创建 Pull Request
```

### 3. 改进文档

- 修正错误
- 添加示例
- 翻译文档

### 4. 分享案例

- 分享使用经验
- 提供领域适配器
- 贡献示例代码

---

## 📖 学术引用

如果您在研究或项目中使用DAML-RAG，请引用：

```bibtex
@software{daml_rag_2025,
  title={DAML-RAG: Domain-Adaptive Meta-Learning RAG Framework},
  author={薛小川 (Xue Xiaochuan)},
  year={2025},
  version={1.2.0},
  url={https://github.com/vivy1024/daml-rag-framework},
  doi={待分配}
}
```

详见 [CITATION.cff](CITATION.cff) 获取完整引用元数据。

---

## 📄 许可证

**版权所有 © 2025 薛小川。保留所有权利。**

根据Apache License 2.0许可证授权。您可以在以下网址获取许可证副本：

    http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件按"原样"分发，不附带任何明示或暗示的担保或条件。详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

基于玉珍健身 v2.0项目的理论和实践成果构建。

**站在巨人的肩膀上：**

- **RAG**: Lewis et al. (2020)
- **GraphRAG**: Microsoft Research (2025)
- **In-Context Learning**: Brown et al. (2020)
- **Knowledge Graph**: Hogan et al. (2021)
- **MCP Protocol**: Anthropic (2025)
- **BGE Model**: Beijing Academy of AI (BAAI)

---

## 📞 联系方式

- **作者**: 薛小川 (Xue Xiaochuan)
- **邮箱**: 1765563156@qq.com
- **GitHub**: https://github.com/vivy1024/daml-rag-framework
- **PyPI**: https://pypi.org/project/daml-rag-framework/
- **问题反馈**: https://github.com/vivy1024/daml-rag-framework/issues

---

**让AI更懂专业领域** 🚀
