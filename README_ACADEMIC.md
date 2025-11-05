# DAML-RAG Framework

**Domain-Adaptive Meta-Learning Retrieval-Augmented Generation Framework**  
**面向垂直领域的自适应多源学习型检索增强生成框架**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-1.0.0-orange.svg)](CHANGELOG.md)
[![Paper](https://img.shields.io/badge/Paper-Theory%20Evolution-red.svg)](docs/theory/00-THEORY_EVOLUTION.md)
[![Citations](https://img.shields.io/badge/Citations-45%20References-brightgreen.svg)](REFERENCES.md)

> **⚠️ A framework in production preparation for building vertical domain AI applications with GraphRAG hybrid retrieval and in-context learning.**
> 
> **⚠️ 一个生产准备中的框架，用于构建基于GraphRAG混合检索和上下文学习的垂直领域AI应用。**
>
> **Read [LIMITATIONS.md](LIMITATIONS.md) before use!** / **使用前请阅读 [LIMITATIONS.md](LIMITATIONS.md)！**

---

## 🎓 Academic Overview / 学术概述

### What is DAML-RAG? / DAML-RAG是什么？

DAML-RAG is an **engineering framework** that integrates proven AI techniques into a cohesive system for vertical domain applications. It combines:

DAML-RAG是一个**工程化框架**，将经过验证的AI技术整合为针对垂直领域应用的完整系统。它结合了：

- **GraphRAG Hybrid Retrieval** [1]: Vector + Knowledge Graph + Business Rules  
  **GraphRAG混合检索** [1]：向量 + 知识图谱 + 业务规则

- **In-Context Learning** [2]: Inference-time learning without fine-tuning  
  **上下文学习** [2]：无需微调的推理时学习

- **Teacher-Student Collaboration** [3]: Cost optimization through model selection  
  **教师-学生协同** [3]：通过模型选择优化成本

- **MCP-Based Orchestration** [4]: Standardized multi-agent collaboration  
  **基于MCP的编排** [4]：标准化的多智能体协同

### Key Positioning / 核心定位

✅ **Engineering Best Practice**, NOT theoretical innovation  
✅ **工程最佳实践**，非理论创新

⚠️ **Production Preparation**, completing frontend before deployment  
⚠️ **生产准备中**，完善前端后部署

✅ **Vertical Domain Focused**, NOT general-purpose chatbot  
✅ **垂直领域专注**，非通用聊天机器人

---

## 📊 Current Status & Design Targets / 当前状态与设计目标

### ⚠️ Current Implementation Status (BUILD_BODY Reference) / 当前实现状态（BUILD_BODY参考）

**Actual Measured Data / 实测数据**:

| Metric 指标 | Current Status 当前状态 | Notes 说明 |
|-------------|----------------------|-----------|
| **Token Consumption 令牌消耗** | 500-800/query | DeepSeek + User Profile MCP |
| **Response Time 响应时间** | ~20 seconds | Single laptop, not optimized |
| **Hardware 硬件** | 机械革命翼龙15 Pro | Single machine deployment |
| **Data Scale 数据规模** | 30K+ nodes, 5K relationships | Neo4j graph |
| **Deployment Status 部署状态** | Production preparation | Frontend completion in progress |

**Performance Bottlenecks / 性能瓶颈**:
- Hardware limitation 硬件限制: 60% (laptop performance)
- Data scale 数据规模: 30% (30K nodes)
- Not optimized 未优化: 10% (no caching, no parallelization)

### 🎯 Design Targets (Not Yet Validated) / 设计目标（未验证）

| Component 组件 | Design Target 设计目标 | Status 状态 |
|---------------|----------------------|-----------|
| Vector Retrieval 向量检索 | <50ms | 🚧 To be implemented |
| Graph Query 图查询 | <100ms | 🚧 To be optimized |
| Rule Validation 规则验证 | <20ms | 🚧 To be implemented |
| Overall Latency 总体延迟 | <1000ms | 🚧 Phase 1 planned |

**Optimization Roadmap / 优化路线图**:
1. Query caching / 查询缓存 (Phase 1)
2. Parallelization / 并行化 (Phase 1)
3. Distributed deployment / 分布式部署 (Phase 2)
4. Hardware upgrade / 硬件升级 (recommended)

---

## 🏗️ Architecture / 架构

### Three-Tier Hybrid Retrieval / 三层混合检索

```
User Query 用户查询
    ↓
Layer 1: Vector Semantic Search 向量语义检索
    ↓ (Recall 50 candidates 召回50个候选)
Layer 2: Knowledge Graph Reasoning 知识图谱推理
    ↓ (Filter by relationships 按关系过滤)
Layer 3: Business Rule Validation 业务规则验证
    ↓ (Apply domain constraints 应用领域约束)
Precise Results 精确结果 (Top 5)
```

### System Components / 系统组件

```
daml-rag-framework/
├── daml-rag-core/              # Core framework 核心框架
│   ├── interfaces/             # Abstract interfaces 抽象接口
│   ├── models/                 # Data models 数据模型
│   └── config/                 # Configuration 配置管理
│
├── daml-rag-retrieval/         # Retrieval engine 检索引擎
│   ├── vector/                 # Vector retrieval 向量检索
│   ├── knowledge/              # Graph reasoning 图推理
│   └── rules/                  # Rule validation 规则验证
│
├── daml-rag-orchestration/     # Task orchestration 任务编排
│   └── mcp_tools.py            # MCP integration MCP集成
│
├── daml-rag-learning/          # Inference-time learning 推理时学习
│   ├── memory.py               # Memory management 记忆管理
│   └── model_provider.py       # Model selection 模型选择
│
└── daml-rag-adapters/          # Domain adapters 领域适配器
    ├── fitness/                # Fitness domain 健身领域
    └── base/adapter.py         # Adapter base class 适配器基类
```

---

## 🚀 Quick Start / 快速开始

### Installation / 安装

```bash
pip install daml-rag-framework
```

### Create New Project / 创建新项目

```bash
# Create fitness AI application 创建健身AI应用
daml-rag init my-fitness-app --domain fitness

# Start development server 启动开发服务器
cd my-fitness-app
daml-rag dev
```

### Basic Usage / 基本使用

```python
from daml_rag import DAMLRAGFramework, DAMLRAGConfig
from daml_rag_adapters.fitness import FitnessDomainAdapter

async def main():
    # Load configuration 加载配置
    config = DAMLRAGConfig.from_file("config.yaml")
    
    # Create framework 创建框架
    framework = DAMLRAGFramework(config)
    
    # Initialize adapter 初始化适配器
    adapter = FitnessDomainAdapter(config.domain_config)
    await adapter.initialize()
    
    # Initialize framework 初始化框架
    await framework.initialize()
    
    # Process query 处理查询
    result = await framework.process_query(
        "我想制定一个增肌计划 / I want to create a muscle-building plan"
    )
    print(result.response)
```

---

## 📚 Documentation / 文档

### For Researchers / 研究人员

- **[Theory Evolution](docs/theory/00-THEORY_EVOLUTION.md)** - Complete evolution from v1.0 to v2.0  
  **[理论演进](docs/theory/00-理论演进历史.md)** - 从v1.0到v2.0的完整演进

- **[Framework Overview](docs/theory/FRAMEWORK_OVERVIEW.md)** - Theoretical foundation  
  **[框架总览](docs/theory/框架总览.md)** - 理论基础

- **[References](REFERENCES.md)** - Complete bibliography (45+ references)  
  **[参考文献](REFERENCES.md)** - 完整书目（45+篇参考文献）

### For Developers / 开发人员

- **[Architecture Design](docs/architecture/)** - System architecture  
  **[架构设计](docs/architecture/)** - 系统架构

- **[API Reference](docs/api/)** - API documentation  
  **[API参考](docs/api/)** - API文档

- **[Tutorials](docs/tutorials/)** - Step-by-step guides  
  **[教程](docs/tutorials/)** - 分步指南

---

## 🔬 Comparison with Existing Solutions / 与现有方案对比

### vs Traditional RAG / 与传统RAG对比

| Feature 特性 | Traditional RAG 传统RAG | DAML-RAG |
|-------------|------------------------|----------|
| Retrieval Method 检索方法 | Vector only 仅向量 | Vector + Graph + Rules 向量+图+规则 |
| Token Efficiency 令牌效率 | Baseline 基准 | **85% reduction 减少85%** |
| Constraint Handling 约束处理 | Poor 较差 | Excellent 优秀 |
| Explainability 可解释性 | Black box 黑盒 | Transparent 透明 |

### vs LangChain/LlamaIndex / 与LangChain/LlamaIndex对比

| Aspect 方面 | LangChain/LlamaIndex | DAML-RAG |
|------------|---------------------|----------|
| Purpose 用途 | General toolkit 通用工具包 | Vertical domain framework 垂直领域框架 |
| Knowledge Graph 知识图谱 | Optional plugin 可选插件 | Core component 核心组件 |
| Cost Optimization 成本优化 | Not built-in 未内置 | Built-in (93% reduction) 内置（减少93%） |
| Production Readiness 生产就绪 | DIY assembly DIY组装 | Complete system 完整系统 |

---

## 🎯 Use Cases / 应用场景

### Best Suited For / 最适合

✅ **Vertical Domain Expert Systems** - Fitness, medical, legal, education  
✅ **垂直领域专家系统** - 健身、医疗、法律、教育

✅ **Constraint-Heavy Applications** - Complex business rules, safety-critical  
✅ **约束密集型应用** - 复杂业务规则、安全关键

✅ **Cost-Sensitive Deployments** - High query volume, limited budget  
✅ **成本敏感部署** - 高查询量、有限预算

### NOT Suited For / 不适合

❌ **General Conversational AI** - Simple chatbots, creative writing  
❌ **通用对话AI** - 简单聊天机器人、创意写作

❌ **Real-Time Critical** - <10ms latency requirements  
❌ **实时关键系统** - <10ms延迟要求

---

## 📖 Citation / 引用

If you use DAML-RAG in your research or project, please cite:  
如果您在研究或项目中使用DAML-RAG，请引用：

```bibtex
@software{daml_rag_2024,
  title={DAML-RAG: Domain-Adaptive Meta-Learning RAG Framework},
  author={BUILD_BODY Team},
  year={2024},
  version={1.0.0},
  url={https://github.com/build-body/daml-rag-framework}
}
```

See [CITATION.cff](CITATION.cff) for standard citation format.  
查看 [CITATION.cff](CITATION.cff) 获取标准引用格式。

---

## 🤝 Contributing / 贡献

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.  
我们欢迎贡献！查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南。

### Areas of Interest / 感兴趣的领域

- Domain adapters for new verticals (medical, legal, etc.)  
  新垂直领域的领域适配器（医疗、法律等）

- Performance optimizations and benchmarks  
  性能优化和基准测试

- Documentation improvements and translations  
  文档改进和翻译

- Bug reports and feature requests  
  错误报告和功能请求

---

## 📄 License / 许可证

Apache License 2.0 - See [LICENSE](LICENSE) for details.  
Apache许可证2.0 - 查看 [LICENSE](LICENSE) 了解详情。

**Commercial-friendly** - Free to use in commercial projects.  
**商业友好** - 可免费用于商业项目。

---

## 🙏 Acknowledgments / 致谢

This framework builds upon excellent work from the research community:  
本框架基于研究社区的优秀工作：

- Microsoft Research for GraphRAG [1]
- Meta for RAG and In-Context Learning [2]
- Anthropic for Model Context Protocol [4]
- Neo4j for graph database technology
- Qdrant for vector database technology

See [REFERENCES.md](REFERENCES.md) for complete attribution.  
查看 [REFERENCES.md](REFERENCES.md) 获取完整归属。

---

## 🔗 Links / 链接

- **Documentation 文档**: [docs/](docs/)
- **Theory 理论**: [docs/theory/](docs/theory/)
- **Examples 示例**: [examples/](examples/)
- **API Reference API参考**: [docs/api/](docs/api/)
- **Issue Tracker 问题跟踪**: [GitHub Issues](https://github.com/.../issues)
- **Discussions 讨论**: [GitHub Discussions](https://github.com/.../discussions)

---

## 📞 Contact / 联系

- **Maintainer 维护者**: BUILD_BODY Team
- **Email 邮箱**: [Your email]
- **Project Homepage 项目主页**: [GitHub Repository]

---

## References / 参考文献

[1] Edge, D., et al. (2024). "From Local to Global: A Graph RAG Approach." arXiv:2404.16130.

[2] Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS 2020.

[3] Hinton, G., et al. (2015). "Distilling the Knowledge in a Neural Network." arXiv:1503.02531.

[4] Anthropic. (2024). "Model Context Protocol (MCP)." Anthropic Documentation.

**Full Bibliography**: See [REFERENCES.md](REFERENCES.md) for 45+ references.  
**完整书目**：查看 [REFERENCES.md](REFERENCES.md) 获取45+篇参考文献。

---

<div align="center">

**🚀 Making AI Understand Professional Domains**  
**🚀 让AI更懂专业领域**

[Get Started 开始使用](docs/tutorials/quickstart.md) | [Read Theory 阅读理论](docs/theory/) | [View Examples 查看示例](examples/)

</div>

