# 贡献指南

欢迎为 DAML-RAG Framework 贡献代码！我们感谢您的每一个贡献。

## 🤝 如何贡献

### 报告问题

1. 检查 [Issues](https://github.com/daml-rag/daml-rag-framework/issues) 确认问题未被报告
2. 创建新的 Issue，使用适当的模板
3. 提供详细的问题描述、复现步骤和期望行为

### 提交代码

1. Fork 项目仓库
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 创建 Pull Request

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/daml-rag/daml-rag-framework.git
cd daml-rag-framework

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码格式化
black .
isort .

# 类型检查
mypy daml_rag
```

## 📋 开发指南

### 代码规范

- 使用 Python 3.8+
- 遵循 PEP 8 规范
- 使用 Black 进行代码格式化
- 使用 isort 进行导入排序
- 使用 mypy 进行类型检查

### 提交信息规范

使用 [Conventional Commits](https://conventionalcommits.org/) 格式：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

类型包括：
- `feat`: 新功能
- `fix`: 修复
- `docs`: 文档
- `style`: 格式
- `refactor`: 重构
- `test`: 测试
- `chore`: 构建/工具

示例：
```
feat(retrieval): add hybrid search support

Implements hybrid search combining vector and keyword search
for better retrieval accuracy.

Closes #123
```

### 测试要求

- 新功能必须包含测试
- 测试覆盖率应 > 80%
- 使用 pytest 运行测试
- 遵循 AAA 模式（Arrange, Act, Assert）

```python
def test_retrieval_should_return_relevant_documents():
    # Arrange
    retriever = VectorRetriever()
    query = "健身训练"

    # Act
    result = await retriever.retrieve(query, top_k=5)

    # Assert
    assert len(result.documents) > 0
    assert all(doc.score > 0.5 for doc in result.documents)
```

### 文档要求

- 公共 API 必须有 docstring
- 使用 Google 风格的 docstring
- 提供类型注解
- 包含使用示例

```python
def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
    """检索相关文档。

    Args:
        query: 查询字符串
        top_k: 返回的最大文档数量

    Returns:
        检索结果，包含相关文档和评分

    Raises:
        ValueError: 当 query 为空时

    Example:
        >>> retriever = VectorRetriever()
        >>> result = retriever.retrieve("健身训练", top_k=3)
        >>> print(len(result.documents))
        3
    """
```

## 🔌 插件开发

### 创建新领域适配器

1. 继承 `DomainAdapter` 基类
2. 实现必需的抽象方法
3. 添加测试和文档

```python
from daml_rag_adapters.base import DomainAdapter

class MyDomainAdapter(DomainAdapter):
    def __init__(self, config):
        super().__init__("my-domain", config)

    async def initialize(self):
        # 初始化逻辑
        pass

    def get_entity_types(self):
        return ["Entity1", "Entity2"]

    # ... 其他必需方法
```

### 添加新工具

1. 实现 `IMCPTool` 接口
2. 注册到工具注册表
3. 添加工具模式和验证

```python
from daml_rag.interfaces import IMCPTool

class MyTool(IMCPTool):
    async def call(self, params):
        # 工具逻辑
        return ToolResult(success=True, data=result)

    def get_schema(self):
        return {
            "name": "my_tool",
            "description": "我的自定义工具",
            "parameters": {
                "param1": {"type": "string", "description": "参数1"}
            }
        }
```

## 📝 文档贡献

### 改进文档

- 修复错误和拼写
- 添加更多示例
- 改进解释和说明
- 翻译成其他语言

### 文档结构

```
docs/
├── quickstart.md          # 快速开始
├── architecture.md        # 架构设计
├── api.md                 # API 参考
├── adapters.md            # 领域适配器
├── deployment.md          # 部署指南
├── best-practices.md      # 最佳实践
├── troubleshooting.md     # 故障排除
└── examples/              # 示例代码
```

## 🧪 测试指南

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_retrieval.py

# 运行带覆盖率的测试
pytest --cov=daml_rag --cov-report=html

# 运行性能测试
pytest tests/performance/
```

### 编写测试

- 单元测试：测试单个组件
- 集成测试：测试组件交互
- 端到端测试：测试完整流程
- 性能测试：测试性能指标

### 测试数据

使用 `pytest.fixture` 创建测试数据：

```python
@pytest.fixture
def sample_config():
    return {
        "domain": "test",
        "retrieval": {"top_k": 5},
        "learning": {"teacher_model": "test"}
    }

@pytest.fixture
def mock_retriever():
    return MockVectorRetriever()
```

## 📋 代码审查

### 审查清单

- [ ] 代码符合规范
- [ ] 测试通过
- [ ] 文档完整
- [ ] 类型正确
- [ ] 性能可接受
- [ ] 安全性考虑

### 审查流程

1. 自动化检查通过
2. 至少一个维护者审查
3. 所有讨论解决
4. 合并到主分支

## 🚀 发布流程

### 版本管理

使用语义化版本控制：
- `MAJOR.MINOR.PATCH`
- 例如：`1.2.3`

### 发布步骤

1. 更新版本号
2. 更新 CHANGELOG
3. 创建 Git 标签
4. 构建发布包
5. 发布到 PyPI

## 🏆 社区

### 行为准则

- 尊重他人
- 友善包容
- 乐于助人
- 保持专业

### 沟通渠道

- [GitHub Discussions](https://github.com/daml-rag/daml-rag-framework/discussions)
- [Discord 社区](https://discord.gg/daml-rag)
- [邮件列表](mailto:dev@daml-rag.org)

## 🙏 致谢

感谢所有为 DAML-RAG Framework 做出贡献的开发者！

### 核心贡献者

- [@contributor1](https://github.com/contributor1) - 核心框架
- [@contributor2](https://github.com/contributor2) - 健身适配器
- [@contributor3](https://github.com/contributor3) - 文档

### 特别感谢

- BUILD_BODY v2.0 项目提供的理论和技术基础
- 所有测试用户提供的反馈和建议
- 开源社区的宝贵贡献

---

有任何问题，欢迎通过 GitHub Issues 或邮件联系我们：[team@daml-rag.org](mailto:team@daml-rag.org)