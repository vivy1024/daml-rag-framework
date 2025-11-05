# DAML-RAG Framework

**Domain-Adaptive Meta-Learning RAG** - Production-Ready Framework for Vertical Domain AI Applications

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-1.0.0-orange.svg)](CHANGELOG.md)

English | **[简体中文](README.md)**

> 🎓 **Combining GraphRAG, In-Context Learning, Multi-Agent Orchestration for Cost-Effective Vertical Domain AI**  
> 🚀 **Production-ready framework achieving token optimization and cost control through engineering best practices**

---

## 📖 Overview

DAML-RAG is a production-ready framework that integrates proven techniques—GraphRAG hybrid retrieval, In-Context Learning, Teacher-Student collaboration, and MCP-based multi-agent orchestration—for building vertical domain AI applications with token savings and cost optimization.

**NOT a new theory, but an engineering best practice framework for practitioners.**

---

## 🔬 Academic Positioning

### What DAML-RAG IS ✅

- **Engineering Framework**: Systematic integration of RAG [1], GraphRAG [2], ICL [3], Knowledge Graphs [4]
- **Production System**: Validated in BUILD_BODY fitness domain
- **Cost Optimization**: Teacher-student collaboration reducing costs
- **Vertical Domain Focus**: Specialized for knowledge-intensive domains

### What DAML-RAG is NOT ❌

- ❌ **NOT a new ML/AI theory**: No novel algorithms or learning paradigms
- ❌ **NOT claiming universal superiority**: Designed for specific use cases
- ❌ **NOT automated domain adaptation**: Requires domain expertise for knowledge graph construction
- ❌ **NOT inference-time "meta-learning"**: Correctly termed "In-Context Learning" (v2.0 correction)

---

## 🎯 Key Features

- 🎯 **GraphRAG Hybrid Retrieval**: Vector + Graph + Rules three-tier architecture
- 🧠 **In-Context Learning** ⭐(v2.0 corrected): Few-Shot + Case-Based Reasoning for quality maintenance
- ⚡ **Teacher-Student Model**: DeepSeek (teacher) + Ollama (student) for cost reduction
- 🔌 **MCP Orchestration**: Standardized multi-agent collaboration via Model Context Protocol
- 🛡️ **Quality Assurance**: Automatic quality monitoring and escalation
- 📊 **Production-Ready**: Complete monitoring, caching, fault tolerance

---

## 🏗️ Core Architecture: Three-Tier Retrieval System

DAML-RAG's core innovation is the three-tier hybrid retrieval architecture, perfectly combining vector retrieval, knowledge graphs, and business rules:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query Input                          │
│      "Recommend leg muscle building exercises that           │
│       don't stress the knees"                                │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Vector Semantic Retrieval                          │
│                                                              │
│  📊 Multiple Vector DB Support:                              │
│    • Qdrant (Recommended) - High-performance vector DB       │
│    • FAISS - Facebook AI Similarity Search                   │
│    • Milvus - Open-source vector database                    │
│    • Pinecone/Weaviate - Cloud vector services               │
│                                                              │
│  🔍 Semantic Similarity Matching:                            │
│    • Cosine Similarity                                       │
│    • HNSW Index Optimization (< 50ms response)               │
│    • Multi-language embedding model support                  │
│                                                              │
│  🎯 Core Functions:                                          │
│    • Understand user intent ("bulking" = "hypertrophy")      │
│    • Fuzzy matching (typos, synonym recognition)             │
│    • Fast candidate recall (Top 20-50)                       │
│    • Multi-modal retrieval (text, image, audio)              │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Knowledge Graph Reasoning                          │
│                                                              │
│  🕸️ Multiple Graph DB Support:                               │
│    • Neo4j (Recommended) - Professional graph database       │
│    • ArangoDB - Multi-model database                         │
│    • JanusGraph - Distributed graph database                 │
│    • Amazon Neptune - Cloud graph service                    │
│                                                              │
│  🔗 Structured Relationship Reasoning:                       │
│    • Cypher Query Language (Neo4j)                           │
│    • Gremlin Graph Traversal Language                        │
│    • SPARQL Semantic Query                                   │
│    • Multi-hop reasoning capability (< 100ms)                │
│                                                              │
│  🎯 Core Functions:                                          │
│    • Precise filtering (based on 2,447+ entity nodes)        │
│    • Constraint validation ("no knee stress")                │
│    • Explainability (clear reasoning paths)                  │
│    • Multi-hop reasoning ("exercise→muscle→goal→constraint") │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Business Rule Filtering                           │
│                                                              │
│  📋 Domain Expert Rule Engine:                               │
│    • Safety rules (age, injury, recovery stage)              │
│    • Equipment rules (available devices, venue limits)       │
│    • Capacity rules (MRV, supercompensation, frequency)      │
│    • Personalization rules (user preferences, goal level)    │
│                                                              │
│  🛡️ Intelligent Validation System:                          │
│    • Dynamic rule loading (< 20ms)                           │
│    • Rule priority management                                │
│    • Rule conflict detection and resolution                  │
│    • Rule effectiveness evaluation and optimization          │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  📊 Final Results: 5 precise recommendations + reasoning +   │
│                    confidence scores                         │
│  💡 Token Optimization: Design target (not validated)        │
│  ⚡ Total Response Time: BUILD_BODY measured ~20s (laptop,   │
│                         not optimized)                       │
│  🎯 User Satisfaction: Design target (not validated)         │
└─────────────────────────────────────────────────────────────┘
```

### 🔧 Technology Stack Selection

DAML-RAG supports flexible technology stack selection, developers can choose based on preferences and needs:

```yaml
# Vector Database Options
Vector Databases:
  Qdrant:      ⭐⭐⭐⭐⭐ Recommended (high-performance, easy deployment)
  FAISS:       ⭐⭐⭐⭐   (local deployment, fast)
  Milvus:      ⭐⭐⭐⭐   (open-source, distributed)
  Pinecone:    ⭐⭐⭐     (cloud, managed service)
  Weaviate:    ⭐⭐⭐     (semantic search, GraphQL)

# Graph Database Options
Graph Databases:
  Neo4j:       ⭐⭐⭐⭐⭐ Recommended (professional graph DB)
  ArangoDB:    ⭐⭐⭐⭐   (multi-model, flexible)
  JanusGraph:  ⭐⭐⭐     (distributed, big data)
  Neptune:     ⭐⭐⭐     (AWS integration)

# AI Model Selection
Large Models:
  DeepSeek:    ⭐⭐⭐⭐⭐ Teacher model (high-quality, Chinese-optimized)
  GPT-4:       ⭐⭐⭐⭐   (strong general capability)
  Claude:      ⭐⭐⭐⭐   (high security)
  Qwen:        ⭐⭐⭐⭐   (open-source, Chinese)

Small Models:
  Ollama:      ⭐⭐⭐⭐⭐ Student model (local deployment, cost optimization)
  Llama:       ⭐⭐⭐⭐   (open-source, good performance)
  Phi:         ⭐⭐⭐     (Microsoft, small and precise)
  Gemma:       ⭐⭐⭐     (Google, lightweight)
```

---

## 📦 Module Structure

```
daml-rag-framework/
├── daml-rag-core/              # 🔧 Core Framework
│   ├── interfaces/             # Abstract interface definitions
│   ├── models/                 # Data models
│   ├── config/                 # Configuration management
│   └── utils/                  # Utility functions
├── daml-rag-retrieval/         # 🔍 Three-tier Retrieval Engine
│   ├── vector/                 # Vector retrieval layer
│   ├── knowledge/              # Knowledge graph layer
│   ├── rules/                  # Rule filtering layer
│   └── cache/                  # Cache management
├── daml-rag-orchestration/     # 🎯 Task Orchestration Engine
├── daml-rag-learning/          # 🧠 Inference-time Learning
├── daml-rag-adapters/          # 🔌 Domain Adapters
├── daml-rag-cli/               # 🚀 Command Line Tools
└── examples/                   # 📚 Example Projects
```

---

## 🚀 Quick Start

### Installation

```bash
pip install daml-rag-framework
```

### Create New Project

```bash
# Create fitness domain AI application
daml-rag init my-fitness-app --domain fitness

# Create healthcare domain AI application
daml-rag init my-medical-app --domain healthcare

# Create custom domain AI application
daml-rag init my-custom-app --template custom
```

### Basic Usage

```python
from daml_rag import DAMLRAGFramework, DAMLRAGConfig
from daml_rag_adapters.fitness import FitnessDomainAdapter

async def main():
    # Load configuration
    config = DAMLRAGConfig.from_file("config.yaml")
    
    # Create framework instance
    framework = DAMLRAGFramework(config)
    
    # Initialize domain adapter
    adapter = FitnessDomainAdapter(config.domain_config)
    await adapter.initialize()
    
    # Initialize framework
    await framework.initialize()
    
    # Process user query
    result = await framework.process_query("I want to create a muscle building plan")
    print(result.response)

if __name__ == "__main__":
    asyncio.run(main())
```

### Configuration Example

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

## 📊 Current Status

**⚠️ Project Status: Production Preparation (Frontend Completion)**

### Actual Measured Data

| Metric | Current | Notes |
|--------|---------|-------|
| **Token/Query (Simple)** | 500-800 | DeepSeek + User Profile MCP |
| **Response Time** | **~20s** | ⚠️ Not optimized, caching needed |
| **Project Stage** | Production Prep | Preparing for deployment |
| **MCP Tools Implemented** | 14/14 ✅ | All tools completed |
| **Docker Status** | In Use | Local deployment ready |
| **Frontend Status** | In Progress | Completing before deployment |

### Current Issues

**⚠️ Known Performance Issues:**

- **Slow Response**: ~20 seconds for simple queries
  - Cause: Unoptimized graph queries, no caching mechanism
  - Cause: Multiple sequential MCP calls, no parallelization
  - Status: Optimization planned for Phase 1
  
- **Frontend Completion**: In progress
  - Tools backend: ✅ Complete (14/14)
  - Frontend UI: 🚧 Completing
  - Docker deployment: ✅ Ready locally

- **Production Deployment**: Preparing
  - Local Docker: ✅ In use
  - Production deployment: 🚧 After frontend completion
  - Performance optimization: ⏳ Planned

### Known Limitations

**⚠️ IMPORTANT: Read [LIMITATIONS.md](LIMITATIONS.md) before use!**

Key limitations:

- **Hardware Requirements**: Minimum 16GB RAM, 32GB+ recommended
- **Response Time**: ~20 seconds (BUILD_BODY case on laptop, not optimized)
- **Scale Limits**: Performance degrades with >30K nodes on single machine
- **Deployment**: Distributed deployment recommended for production

See detailed analysis in [LIMITATIONS.md](LIMITATIONS.md).

### Design Targets (Not Yet Validated)

The following are **theoretical design goals**, not validated metrics:

- 🎯 Token efficiency through GraphRAG hybrid retrieval
- 🎯 Cost optimization via teacher-student collaboration  
- 🎯 Quality improvement through structured knowledge
- 🎯 Fast retrieval via vector + graph + rules

**Status**: Implementation in progress, benchmarks pending.

---

## 📚 Documentation

### Essential Reading

- **[LIMITATIONS.md](LIMITATIONS.md)** ⚠️ - Limitations and constraints (READ FIRST!)

### Theory

- [Theory Evolution](docs/theory/00-THEORY_EVOLUTION.md)
- [GraphRAG Hybrid Retrieval](docs/theory/01-GraphRAG-Hybrid-Retrieval.md)
- [In-Context Learning](docs/theory/02-In-Context-Learning.md)
- [Framework Overview](docs/theory/FRAMEWORK_OVERVIEW.md)

### Case Studies

- [BUILD_BODY Case Study](examples/BUILD_BODY_CASE_STUDY.md) (Coming soon) - Reference implementation

### Guides

- [Quick Start](docs/tutorials/quickstart.md) (Coming soon)
- [Architecture Design](docs/architecture/) (Coming soon)
- [API Reference](docs/api/) (Coming soon)
- [Deployment Guide](docs/tutorials/deployment.md) (Coming soon)

### References

- [Complete Bibliography](REFERENCES.md)
- [Citation](CITATION.cff)

---

## 📖 Citation

If you use DAML-RAG in your research or project, please cite:

```bibtex
@software{daml_rag_2024,
  title={DAML-RAG: Domain-Adaptive Meta-Learning RAG Framework},
  author={薛小川 (Xue Xiaochuan)},
  year={2025},
  version={1.0.0},
  url={https://github.com/vivy1024/daml-rag-framework}
}
```

See [CITATION.cff](CITATION.cff) for detailed citation metadata.

**Copyright © 2025 薛小川 (Xue Xiaochuan). All rights reserved.**

---

## 🤝 Contributing

Contributions are welcome! Please check:
- [CONTRIBUTING.md](CONTRIBUTING.md) (Coming soon)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) (Coming soon)

---

## 📄 License

**Copyright © 2025 薛小川 (Xue Xiaochuan). All rights reserved.**

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built on theoretical and practical achievements from the BUILD_BODY v2.0 project.

**Standing on the shoulders of giants:**
- RAG: Lewis et al. (2020)
- GraphRAG: Microsoft Research (2025)
- In-Context Learning: Brown et al. (2020)
- Knowledge Graphs: Hogan et al. (2021)
- MCP: Anthropic (2025)

---

**Making AI Understand Professional Domains** 🚀

