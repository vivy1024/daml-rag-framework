# DAML-RAG 框架

**领域自适应元学习RAG** - 面向垂直领域AI应用的生产就绪框架

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-1.0.0-orange.svg)](CHANGELOG.md)
[![Package Status](https://img.shields.io/badge/Package-Ready_to_Publish-brightgreen.svg)](BUILD_AND_PUBLISH.md)
[![Build](https://img.shields.io/badge/Build-Passing-success.svg)](scripts/build.sh)

**[English](README_EN.md)** | 简体中文

> 📦 **打包状态**: 项目已完成打包配置，可以发布到 PyPI！详见 [BUILD_AND_PUBLISH.md](BUILD_AND_PUBLISH.md)

> 🎓 **结合GraphRAG、上下文学习、多智能体协同，打造成本高效的垂直领域AI系统**  
> 🚀 **生产就绪框架，实现Token优化和成本控制的工程最佳实践**

---

## 📖 项目简介

DAML-RAG是一个生产就绪框架，整合了经过验证的技术——GraphRAG混合检索、上下文学习、教师-学生协同和基于MCP的多智能体编排——用于构建垂直领域AI应用，实现Token节省和成本优化。

**不是新理论，而是面向实践者的工程最佳实践框架。**

---

## 🔬 学术定位

### DAML-RAG 是什么 ✅

- **工程框架**：系统整合RAG [1]、GraphRAG [2]、ICL [3]、知识图谱 [4]
- **生产系统**：在玉珍健身领域经过验证
- **成本优化**：通过教师-学生协同降低成本
- **垂直领域聚焦**：专为知识密集型领域设计

### DAML-RAG 不是什么 ❌

- ❌ **不是新的ML/AI理论**：没有创新算法或学习范式
- ❌ **不声称通用优越性**：为特定用例设计
- ❌ **不是自动化领域适配**：需要领域专家构建知识图谱
- ❌ **不是推理时"元学习"**：正确术语是"上下文学习"（v2.0修正）

---

## 🎯 核心特性

- 🎯 **GraphRAG混合检索**：向量 + 图谱 + 规则三层架构
- 🧠 **上下文学习** ⭐(v2.0修正)：Few-Shot + 案例推理维持质量
- ⚡ **教师-学生模型**：DeepSeek（教师）+ Ollama（学生）降低成本
- 🔌 **MCP编排**：基于Model Context Protocol的标准化多智能体协同
- 🛡️ **质量保障**：自动质量监控和升级机制
- 📊 **生产就绪**：完整的监控、缓存、容错系统

---

## 🏗️ 核心架构：三层检索系统

DAML-RAG的核心创新在于三层混合检索架构，完美结合向量检索、知识图谱和业务规则：

```
┌─────────────────────────────────────────────────────────────┐
│                    用户查询输入                               │
│         "推荐不伤膝盖的腿部增肌动作"                          │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  第一层：向量语义检索 (Vector Retrieval)                      │
│                                                              │
│  📊 支持多种向量数据库:                                        │
│    • Qdrant (推荐) - 高性能向量数据库                          │
│    • FAISS - Facebook AI相似度搜索                            │
│    • Milvus - 开源向量数据库                                  │
│    • Pinecone/Weaviate - 云端向量服务                        │
│                                                              │
│  🔍 语义相似度匹配:                                          │
│    • 余弦相似度（Cosine Similarity）                          │
│    • HNSW索引优化（< 50ms响应时间）                           │
│    • 多语言embedding模型支持                                 │
│                                                              │
│  🎯 核心功能:                                               │
│    • 理解用户意图（"增肌" = "肥大训练"）                       │
│    • 模糊匹配（拼写错误、同义词识别）                          │
│    • 快速召回候选集（Top 20-50）                             │
│    • 多模态检索支持（文本、图像、音频）                        │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  第二层：图关系推理 (Knowledge Graph)                        │
│                                                              │
│  🕸️ 支持多种图数据库:                                         │
│    • Neo4j (推荐) - 专业图数据库                             │
│    • ArangoDB - 多模型数据库                                  │
│    • JanusGraph - 分布式图数据库                              │
│    • Amazon Neptune - 云端图服务                             │
│                                                              │
│  🔗 结构化关系推理:                                          │
│    • Cypher查询语言（Neo4j）                                 │
│    • Gremlin图遍历语言                                       │
│    • SPARQL语义查询                                          │
│    • 多跳推理能力（< 100ms）                                 │
│                                                              │
│  🎯 核心功能:                                               │
│    • 精确筛选（基于2,447+实体节点）                          │
│    • 约束验证（"不压迫膝盖"）                                │
│    • 可解释性（清晰的推理路径）                               │
│    • 多跳推理（"动作→肌群→目标→约束"）                        │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  第三层：业务规则验证 (Rule Filtering)                       │
│                                                              │
│  📋 领域专业规则引擎:                                          │
│    • 安全规则（年龄、损伤、康复阶段）                          │
│    • 器械规则（可用设备、场地限制）                            │
│    • 容量规则（MRV、超量恢复、训练频率）                       │
│    • 个性化规则（用户偏好、目标水平）                          │
│                                                              │
│  🛡️ 智能验证系统:                                           │
│    • 动态规则加载（< 20ms）                                  │
│    • 规则优先级管理                                          │
│    • 规则冲突检测和解决                                      │
│    • 规则效果评估和优化                                       │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  📊 最终结果：5个精准推荐 + 推荐理由 + 置信度评分               │
│  💡 Token优化：理论设计目标（未验证）                          │
│  ⚡ 总响应时间：玉珍实测 ~20秒（笔记本，未优化）          │
│  🎯 用户满意度：设计目标（未验证）                             │
└─────────────────────────────────────────────────────────────┘
```

### 🔧 技术栈选型

DAML-RAG支持灵活的技术栈选型，开发者可以根据偏好和需求选择：

```yaml
# 向量数据库选择
向量数据库:
  Qdrant:      ⭐⭐⭐⭐⭐ 推荐（高性能、易部署）
  FAISS:       ⭐⭐⭐⭐   （本地部署、速度快）
  Milvus:      ⭐⭐⭐⭐   （开源、分布式）
  Pinecone:    ⭐⭐⭐     （云端、托管服务）
  Weaviate:    ⭐⭐⭐     （语义搜索、GraphQL）

# 图数据库选择
图数据库:
  Neo4j:       ⭐⭐⭐⭐⭐ 推荐（专业图数据库）
  ArangoDB:    ⭐⭐⭐⭐   （多模型、灵活）
  JanusGraph:  ⭐⭐⭐     （分布式、大数据）
  Neptune:     ⭐⭐⭐     （AWS集成）

# AI模型选择
大模型:
  DeepSeek:    ⭐⭐⭐⭐⭐ 教师模型（高质量、中文优化）
  GPT-4:       ⭐⭐⭐⭐   （通用能力强）
  Claude:      ⭐⭐⭐⭐   （安全性高）
  Qwen:        ⭐⭐⭐⭐   （开源、中文）

小模型:
  Ollama:      ⭐⭐⭐⭐⭐ 学生模型（本地部署、成本优化）
  Llama:       ⭐⭐⭐⭐   （开源、性能好）
  Phi:         ⭐⭐⭐     （微软、小而精）
  Gemma:       ⭐⭐⭐     （Google、轻量级）
```

---

## 📦 模块结构

```
daml-rag-framework/
├── daml-rag-core/              # 🔧 核心框架
│   ├── interfaces/             # 抽象接口定义
│   ├── models/                 # 数据模型
│   ├── config/                 # 配置管理
│   └── utils/                  # 工具函数
├── daml-rag-retrieval/         # 🔍 三层检索引擎
│   ├── vector/                 # 向量检索层
│   ├── knowledge/              # 知识图谱层
│   ├── rules/                  # 规则过滤层
│   └── cache/                  # 缓存管理
├── daml-rag-orchestration/     # 🎯 任务编排引擎
├── daml-rag-learning/          # 🧠 推理时学习
├── daml-rag-adapters/          # 🔌 领域适配器
├── daml-rag-cli/               # 🚀 命令行工具
└── examples/                   # 📚 示例项目
```

---

## 🚀 快速开始

### 安装

#### 方式1：从 PyPI 安装（推荐）✅

```bash
pip install daml-rag-framework
```

**PyPI 页面**: https://pypi.org/project/daml-rag-framework/

#### 方式2：从源码安装（开发版）

```bash
# 克隆仓库
git clone https://github.com/vivy1024/daml-rag-framework.git
cd daml-rag-framework

# 安装依赖并安装框架（开发模式）
pip install -e .

# 或者构建并安装
python -m pip install --upgrade build
python -m build
pip install dist/*.whl
```

#### 方式3：从 GitHub 直接安装

```bash
pip install git+https://github.com/vivy1024/daml-rag-framework.git
```

### 验证安装

```bash
# 验证导入
python -c "from daml_rag import DAMLRAGFramework; print('✅ 安装成功')"

# 测试 CLI
daml-rag --help
```

### 创建新项目

```bash
# 创建健身领域AI应用
daml-rag init my-fitness-app --domain fitness

# 创建医疗领域AI应用
daml-rag init my-medical-app --domain healthcare

# 创建自定义领域AI应用
daml-rag init my-custom-app --template custom
```

### 基本使用

```python
import asyncio
from daml_rag import DAMLRAGFramework, DAMLRAGConfig
from daml_rag.adapters import FitnessDomainAdapter

async def main():
    # 加载配置
    config = DAMLRAGConfig.from_file("config.yaml")
    
    # 创建框架实例
    framework = DAMLRAGFramework(config)
    
    # 初始化领域适配器
    adapter = FitnessDomainAdapter(config.domain_config)
    await adapter.initialize()
    
    # 初始化框架
    await framework.initialize()
    
    # 处理用户查询
    result = await framework.process_query("我想制定一个增肌计划")
    print(result.response)

if __name__ == "__main__":
    asyncio.run(main())
```

### 配置文件示例

```yaml
# config.yaml
domain: fitness
debug: false

retrieval:
  vector_model: "BAAI/bge-base-zh-v1.5"
  top_k: 5
  similarity_threshold: 0.6
  cache_ttl: 300
  enable_kg: true
  enable_rules: true

orchestration:
  max_parallel_tasks: 10
  timeout_seconds: 30
  retry_attempts: 3
  enable_caching: true

learning:
  teacher_model: "deepseek"
  student_model: "ollama-qwen2.5"
  experience_threshold: 3.5
  feedback_weight: 0.8
  adaptive_threshold: 0.7

domain_config:
  knowledge_graph_path: "./data/knowledge_graph.db"
  mcp_servers:
    - name: "user-profile"
      command: "python"
      args: ["user-profile-stdio/server.py"]
    - name: "professional-coach"
      command: "python"
      args: ["professional-coach-stdio/server.py"]
```

---

## 📊 当前状态

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

- **生产部署**：准备中
  - 本地Docker：✅ 使用中
  - 生产部署：🚧 前端完成后
  - 性能优化：⏳ 已计划

### 已知限制

**⚠️ 重要：使用前请阅读 [LIMITATIONS.md](LIMITATIONS.md)！**

关键限制：

- **硬件需求**：最低16GB内存，推荐32GB+
- **响应时间**：~20秒（玉珍健身笔记本案例，未优化）
- **规模限制**：单机超过30K节点性能下降
- **部署**：生产环境建议分布式部署

详细分析见 [LIMITATIONS.md](LIMITATIONS.md)。

### 设计目标（未验证）

以下是**理论设计目标**，非验证指标：

- 🎯 通过GraphRAG混合检索实现Token效率
- 🎯 通过教师-学生协同优化成本
- 🎯 通过结构化知识提升质量
- 🎯 通过向量+图谱+规则实现快速检索

**状态**：实施进行中，基准测试待进行。

---

## 📚 文档

### 必读文档

- **[LIMITATIONS.md](LIMITATIONS.md)** ⚠️ - 限制和约束（必读！）
- **[PUBLISHING.md](PUBLISHING.md)** 📦 - PyPI 发布指南（开发者必读）

### 理论基础

- [理论演进历史](docs/theory/00-理论演进历史.md)
- [GraphRAG混合检索理论](docs/theory/01-GraphRAG混合检索理论.md)
- [推理时上下文学习理论](docs/theory/02-推理时上下文学习理论.md)
- [框架总览](docs/theory/框架总览.md)

### 案例研究

- [玉珍健身案例研究](examples/YUZHEN_FITNESS_CASE_STUDY.md)（即将推出）- 参考实现

### 指南

- [快速开始](docs/tutorials/quickstart.md)（即将推出）
- [架构设计](docs/architecture/)（即将推出）
- [API文档](docs/api/)（即将推出）
- [部署指南](docs/tutorials/deployment.md)（即将推出）

### 开发者指南

- [打包和发布流程](PUBLISHING.md) - 如何发布到 PyPI
- [贡献指南](CONTRIBUTING.md)（即将推出）
- [开发环境设置](docs/development/)（即将推出）

### 参考文献

- [完整参考文献](REFERENCES.md)
- [学术引用](CITATION.cff)

---

## 📖 学术引用

如果您在研究或项目中使用DAML-RAG，请引用：

```bibtex
@software{daml_rag_2024,
  title={DAML-RAG: Domain-Adaptive Meta-Learning RAG Framework},
  author={薛小川 (Xue Xiaochuan)},
  year={2025},
  version={1.0.0},
  url={https://github.com/vivy1024/daml-rag-framework}
}
```

详见 [CITATION.cff](CITATION.cff) 获取完整引用元数据。

**版权所有 © 2025 薛小川。保留所有权利。**

---

## 🤝 贡献

欢迎贡献！请查看：
- [CONTRIBUTING.md](CONTRIBUTING.md)（即将推出）
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)（即将推出）

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
- RAG：Lewis et al. (2020)
- GraphRAG：Microsoft Research (2025)
- 上下文学习：Brown et al. (2020)
- 知识图谱：Hogan et al. (2021)
- MCP：Anthropic (2025)

---

**让AI更懂专业领域** 🚀
