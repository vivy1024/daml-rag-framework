# DAML-RAG Framework 快速开始指南

**5分钟构建你的垂直领域AI应用**

## 🎯 为什么选择 DAML-RAG？

- **10倍开发效率**：垂直领域AI应用开发从月缩短到周
- **85%成本节省**：智能检索和模型调度优化
- **开箱即用**：预置健身、医疗、教育等领域适配器
- **生产就绪**：完整的监控、缓存、容错机制

## 🚀 快速安装

### 1. 安装框架

```bash
# 使用 pip 安装
pip install daml-rag-framework

# 或使用 poetry
poetry add daml-rag-framework

# 或使用 uv
uv add daml-rag-framework
```

### 2. 创建新项目

```bash
# 创建健身领域AI应用
daml-rag init my-fitness-app --domain fitness

# 创建医疗领域AI应用
daml-rag init my-medical-app --domain healthcare

# 创建教育领域AI应用
daml-rag init my-education-app --domain education

# 创建自定义领域应用
daml-rag init my-custom-app --template custom
```

### 3. 启动开发服务器

```bash
cd my-fitness-app
daml-rag dev
```

访问 http://localhost:8000 开始使用！

## 💻 基本使用

### 最简单的示例

```python
from daml_rag import DAMLRAGFramework, DAMLRAGConfig
from daml_rag_adapters.fitness import FitnessDomainAdapter
import asyncio

async def main():
    # 1. 加载配置
    config = DAMLRAGConfig.from_file("config.yaml")

    # 2. 创建框架实例
    framework = DAMLRAGFramework(config)

    # 3. 初始化领域适配器
    adapter = FitnessDomainAdapter(config.domain_config)
    await adapter.initialize()

    # 4. 初始化框架
    await framework.initialize()

    # 5. 处理用户查询
    result = await framework.process_query("我想制定一个增肌计划")
    print(result.response)

if __name__ == "__main__":
    asyncio.run(main())
```

### 配置文件示例

创建 `config.yaml`：

```yaml
# 基本配置
domain: fitness
debug: false
environment: development

# 检索配置
retrieval:
  vector_model: "BAAI/bge-base-zh-v1.5"
  top_k: 5
  similarity_threshold: 0.6
  cache_ttl: 300
  enable_kg: true
  enable_rules: true

# 编排配置
orchestration:
  max_parallel_tasks: 10
  timeout_seconds: 30
  retry_attempts: 3

# 学习配置
learning:
  teacher_model: "deepseek"
  student_model: "ollama-qwen2.5"
  experience_threshold: 3.5
  adaptive_threshold: 0.7

# 领域配置
domain_config:
  knowledge_graph_path: "./data/knowledge_graph.db"
  mcp_servers:
    - name: "user-profile"
      command: "python"
      args: ["mcp-servers/user-profile-stdio/server.py"]
    - name: "professional-coach"
      command: "python"
      args: ["mcp-servers/professional-coach-stdio/server.py"]
```

## 🏋️ 健身领域示例

### 创建健身教练助手

```python
from fastapi import FastAPI
from daml_rag import DAMLRAGFramework, DAMLRAGConfig
from daml_rag_adapters.fitness import FitnessDomainAdapter
import uvicorn

app = FastAPI()

# 全局框架实例
framework = None
adapter = None

@app.on_event("startup")
async def startup():
    global framework, adapter

    # 初始化框架
    config = DAMLRAGConfig.from_file("config.yaml")
    framework = DAMLRAGFramework(config)

    # 初始化健身适配器
    adapter = FitnessDomainAdapter(config.domain_config)
    await adapter.initialize()

    # 初始化框架
    await framework.initialize()

@app.post("/chat")
async def chat(message: str):
    """处理聊天消息"""
    result = await framework.process_query(message)
    return {
        "response": result.response,
        "sources": result.sources,
        "model_used": result.model_used,
        "execution_time": result.execution_time
    }

@app.get("/health")
async def health():
    """健康检查"""
    framework_health = await framework.health_check()
    adapter_health = await adapter.health_check()

    return {
        "status": "healthy",
        "framework": framework_health,
        "adapter": adapter_health
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 测试对话

```bash
# 测试API
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "我想制定一个增肌计划"}'

# 健康检查
curl "http://localhost:8000/health"
```

## 🏥 医疗领域示例

```python
from daml_rag_adapters.healthcare import HealthcareDomainAdapter

async def create_healthcare_assistant():
    config = DAMLRAGConfig.from_file("healthcare_config.yaml")
    framework = DAMLRAGFramework(config)

    adapter = HealthcareDomainAdapter(config.domain_config)
    await adapter.initialize()

    await framework.initialize()

    # 医疗咨询
    result = await framework.process_query("头痛的可能原因有哪些？")
    return result.response
```

## 🎓 教育领域示例

```python
from daml_rag_adapters.education import EducationDomainAdapter

async def create_education_assistant():
    config = DAMLRAGConfig.from_file("education_config.yaml")
    framework = DAMLRAGFramework(config)

    adapter = EducationDomainAdapter(config.domain_config)
    await adapter.initialize()

    await framework.initialize()

    # 教育咨询
    result = await framework.process_query("如何设计Python入门课程？")
    return result.response
```

## 🔧 自定义领域适配器

### 创建自定义适配器

```python
from daml_rag_adapters.base import DomainAdapter
from daml_rag.interfaces import IKnowledgeGraphRetriever

class MyCustomAdapter(DomainAdapter):
    def __init__(self, config):
        super().__init__("my-domain", config)

    async def initialize(self):
        # 初始化自定义组件
        pass

    def get_entity_types(self):
        return ["CustomEntity1", "CustomEntity2"]

    def get_relation_types(self):
        return ["RELATES_TO", "PART_OF"]

    def get_tool_registry(self):
        # 返回自定义工具
        return {}

    def get_intent_patterns(self):
        return ["我想了解.*", "帮我分析.*"]

    async def build_knowledge_graph(self, data_source):
        # 构建自定义知识图谱
        pass

# 使用自定义适配器
async def use_custom_adapter():
    config = DAMLRAGConfig.from_file("config.yaml")
    framework = DAMLRAGFramework(config)

    adapter = MyCustomAdapter(config.domain_config)
    await adapter.initialize()

    await framework.initialize()

    result = await framework.process_query("自定义查询")
    return result
```

## 🛠️ 常用命令

### CLI 工具使用

```bash
# 查看帮助
daml-rag --help

# 初始化项目
daml-rag init my-project --domain fitness

# 创建工具脚手架
daml-rag scaffold my-tool --category exercise

# 部署项目
daml-rag deploy --platform docker

# 健康检查
daml-rag health

# 查看配置
daml-rag config show

# 设置配置
daml-rag config set retrieval.top_k 10
```

### 开发命令

```bash
# 启动开发服务器
daml-rag dev

# 运行测试
daml-rag test

# 代码格式化
daml-rag format

# 类型检查
daml-rag lint
```

## 📊 监控和调试

### 性能监控

```python
# 获取框架统计信息
stats = framework.get_framework_stats()
print(f"总查询数: {stats['query_stats']['total_queries']}")
print(f"平均响应时间: {stats['query_stats']['average_response_time']:.2f}s")

# 健康检查
health = await framework.health_check()
print(f"框架状态: {health['overall_status']}")

# 领域适配器统计
adapter_stats = await adapter.get_statistics()
print(f"工具数量: {adapter_stats['tools_count']}")
```

### 日志配置

```yaml
logging:
  log_level: "INFO"
  log_to_file: true
  log_file_path: "./logs/daml-rag.log"
  structured_logging: true
  component_log_levels:
    retrieval: "DEBUG"
    orchestration: "INFO"
```

## 🚀 部署指南

### Docker 部署

```bash
# 构建 Docker 镜像
docker build -t my-fitness-app .

# 运行容器
docker run -p 8000:8000 my-fitness-app

# 使用 Docker Compose
docker-compose up -d
```

### 生产环境部署

```bash
# 构建生产版本
daml-rag build --env production

# 部署到云平台
daml-rag deploy --platform aws --region us-west-2

# 监控部署状态
daml-rag deploy status
```

## 🔍 故障排除

### 常见问题

**Q: 模型调用失败**
```bash
# 检查模型配置
daml-rag config show learning.teacher_model

# 测试模型连接
daml-rag test model --name deepseek
```

**Q: 检索结果为空**
```bash
# 检查向量索引
daml-rag health check --component retrieval

# 重建索引
daml-rag rebuild-index --data-path ./data
```

**Q: MCP 工具连接失败**
```bash
# 检查 MCP 服务器状态
daml-rag health check --component mcp

# 重启 MCP 服务器
daml-rag restart mcp --server professional-coach
```

### 调试模式

```bash
# 启用详细日志
daml-rag dev --verbose

# 启用调试模式
daml-rag dev --debug

# 查看组件状态
daml-rag status --detailed
```

## 📚 下一步

- 📖 [详细文档](./architecture.md)
- 🔌 [领域适配器开发](./adapters.md)
- 🛠️ [API参考](./api.md)
- 🚀 [部署指南](./deployment.md)
- 💡 [最佳实践](./best-practices.md)

## 🤝 获取帮助

- 📖 [官方文档](https://docs.daml-rag.org)
- 💬 [社区讨论](https://github.com/vivy1024/daml-rag-framework/discussions)
- 🐛 [问题反馈](https://github.com/vivy1024/daml-rag-framework/issues)
- 📧 [邮件支持](mailto:support@daml-rag.org)

---

**开始构建你的垂直领域AI应用吧！** 🚀