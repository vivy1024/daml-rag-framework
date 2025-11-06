# DAML-RAG Framework - 快速开始指南

**5分钟上手 DAML-RAG 框架**

---

## 🎯 目标

通过这个快速开始指南，你将：
1. ✅ 构建 DAML-RAG 框架
2. ✅ 本地测试安装
3. ✅ 了解如何发布到 PyPI

**预计时间**: 5-10 分钟

---

## 📋 前置要求

- Python 3.8 或更高版本
- pip 和 git 已安装
- （可选）PyPI 账号（用于发布）

---

## 🚀 快速开始（推荐）

### Windows 用户

```powershell
cd F:\build_body\daml-rag-framework
.\scripts\quick-start.ps1
```

### Linux/Mac 用户

```bash
cd /path/to/daml-rag-framework
chmod +x scripts/*.sh
./scripts/quick-start.sh
```

这个脚本会自动完成：
1. 构建发布包
2. 本地安装测试
3. 验证所有模块正常工作

---

## 📝 手动步骤（可选）

如果你想了解每一步的细节：

### 第1步：构建发布包

**Windows**:
```powershell
.\scripts\build.ps1
```

**Linux/Mac**:
```bash
./scripts/build.sh
```

构建完成后，会在 `dist/` 目录生成两个文件：
- `daml_rag_framework-1.0.0-py3-none-any.whl` (wheel 包)
- `daml_rag_framework-1.0.0.tar.gz` (源码包)

### 第2步：本地测试安装

**Windows**:
```powershell
.\scripts\test-install.ps1
```

**Linux/Mac**:
```bash
./scripts/test-install.sh
```

测试会验证：
- ✅ 包能正常安装
- ✅ 核心模块能正常导入
- ✅ CLI 工具能正常运行

### 第3步：使用框架

#### 开发模式安装（推荐）

```bash
pip install -e .
```

这样可以边修改代码边测试，不需要重新安装。

#### 验证安装

```bash
# 测试导入
python -c "from daml_rag import DAMLRAGFramework; print('✅ 安装成功')"

# 测试 CLI
daml-rag --help
```

#### 简单示例

创建一个 `test_daml_rag.py`:

```python
import asyncio
from daml_rag import DAMLRAGFramework, DAMLRAGConfig

async def main():
    print("🚀 DAML-RAG Framework 测试")
    
    # 创建默认配置
    config = DAMLRAGConfig(
        domain="fitness",
        debug=True
    )
    
    # 创建框架实例
    framework = DAMLRAGFramework(config)
    print("✅ 框架实例创建成功")

if __name__ == "__main__":
    asyncio.run(main())
```

运行：

```bash
python test_daml_rag.py
```

---

## 📦 发布到 PyPI（可选）

### 准备工作

1. 注册 PyPI 账号: https://pypi.org/account/register/
2. 创建 API Token
3. 配置 `.pypirc` 文件

详细步骤请查看 [PUBLISHING.md](PUBLISHING.md)。

### 发布到 TestPyPI（测试）

```powershell
# Windows
.\scripts\publish.ps1 test

# Linux/Mac
./scripts/publish.sh test
```

### 发布到 PyPI（生产）

```powershell
# Windows
.\scripts\publish.ps1 prod

# Linux/Mac
./scripts/publish.sh prod
```

---

## 🛠️ 目录结构说明

```
daml-rag-framework/
├── daml_rag/              # 主包（所有代码在这里）
│   ├── __init__.py       # 包入口
│   ├── core.py           # 核心框架
│   ├── retrieval/        # 检索引擎
│   ├── learning/         # 学习模块
│   ├── orchestration/    # 任务编排
│   ├── adapters/         # 领域适配器
│   └── cli/              # 命令行工具
├── scripts/              # 构建和发布脚本
│   ├── build.ps1/sh      # 构建脚本
│   ├── test-install.ps1/sh
│   ├── publish.ps1/sh    # 发布脚本
│   └── quick-start.ps1/sh
├── pyproject.toml        # 项目配置
├── MANIFEST.in           # 打包清单
├── setup.py              # 兼容性配置
├── README.md             # 项目说明
└── PUBLISHING.md         # 发布指南
```

---

## ❓ 常见问题

### Q1: 构建失败怎么办？

**A**: 检查 Python 版本和依赖：

```bash
python --version  # 确保 >= 3.8
pip install --upgrade build twine setuptools wheel
```

### Q2: 导入模块失败？

**A**: 确保使用开发模式安装：

```bash
pip uninstall daml-rag-framework
pip install -e .
```

### Q3: CLI 工具找不到？

**A**: 重新安装并检查环境变量：

```bash
pip install -e .
which daml-rag  # Linux/Mac
where daml-rag  # Windows
```

### Q4: 旧的目录结构还在？

**A**: 这是正常的，新的包结构在 `daml_rag/` 目录：

- ✅ 使用: `daml_rag/`（新结构）
- ❌ 忽略: `daml-rag-core/`, `daml-rag-retrieval/` 等（旧结构）

---

## 📚 下一步

1. **阅读文档**
   - [理论基础](docs/theory/)
   - [架构设计](docs/architecture/)
   - [API 参考](docs/api/)

2. **查看示例**
   - [健身应用示例](examples/fitness-coach/)
   - [配置示例](examples/config_examples.py)
   - [MCP 示例](examples/mcp_client_example.py)

3. **参与开发**
   - [贡献指南](CONTRIBUTING.md)
   - [发布流程](PUBLISHING.md)

---

## 🆘 获取帮助

- **文档**: [README.md](README.md)
- **问题**: [GitHub Issues](https://github.com/vivy1024/daml-rag-framework/issues)
- **邮箱**: 1765563156@qq.com

---

**祝你使用愉快！🎉**

**维护者**: 薛小川  
**最后更新**: 2025-11-06


