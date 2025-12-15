# GitHub Actions 工作流

本目录包含DAML-RAG框架的CI/CD配置。

## 工作流说明

### 1. CI (ci.yml)

**触发条件**：
- Push到main或develop分支
- Pull Request到main或develop分支

**功能**：
- 在Python 3.8-3.11上运行测试
- 代码覆盖率报告
- 代码质量检查（flake8, black, isort, mypy）

**状态徽章**：
```markdown
[![CI](https://github.com/vivy1024/daml-rag-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/vivy1024/daml-rag-framework/actions/workflows/ci.yml)
```

### 2. Publish (publish.yml)

**触发条件**：
- 创建GitHub Release
- 手动触发（workflow_dispatch）

**功能**：
- 构建Python包
- 发布到Test PyPI（手动触发）
- 发布到PyPI（Release触发）
- 上传Release资产

**所需Secrets**：
- `PYPI_API_TOKEN`: PyPI API令牌
- `TEST_PYPI_API_TOKEN`: Test PyPI API令牌

**使用方法**：
```bash
# 1. 创建Release
git tag v2.0.0
git push origin v2.0.0

# 2. 在GitHub上创建Release
# 工作流会自动发布到PyPI
```

### 3. Documentation (docs.yml)

**触发条件**：
- Push到main分支（docs/或framework/目录变更）
- 手动触发

**功能**：
- 构建Sphinx文档
- 部署到GitHub Pages

**访问地址**：
https://vivy1024.github.io/daml-rag-framework/

## 配置Secrets

在GitHub仓库设置中添加以下Secrets：

1. **PYPI_API_TOKEN**
   - 访问 https://pypi.org/manage/account/token/
   - 创建新的API令牌
   - 复制令牌并添加到GitHub Secrets

2. **TEST_PYPI_API_TOKEN**
   - 访问 https://test.pypi.org/manage/account/token/
   - 创建新的API令牌
   - 复制令牌并添加到GitHub Secrets

## 本地测试

### 运行测试
```bash
pytest tests/ -v --cov=framework
```

### 代码质量检查
```bash
# 格式检查
black --check framework/
isort --check-only framework/

# 代码检查
flake8 framework/
mypy framework/ --ignore-missing-imports
```

### 构建包
```bash
python -m build
twine check dist/*
```

### 测试发布到Test PyPI
```bash
twine upload --repository testpypi dist/*
```

## 工作流状态

| 工作流 | 状态 | 说明 |
|--------|------|------|
| CI | [![CI](https://github.com/vivy1024/daml-rag-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/vivy1024/daml-rag-framework/actions/workflows/ci.yml) | 持续集成 |
| Publish | [![Publish](https://github.com/vivy1024/daml-rag-framework/actions/workflows/publish.yml/badge.svg)](https://github.com/vivy1024/daml-rag-framework/actions/workflows/publish.yml) | PyPI发布 |
| Docs | [![Docs](https://github.com/vivy1024/daml-rag-framework/actions/workflows/docs.yml/badge.svg)](https://github.com/vivy1024/daml-rag-framework/actions/workflows/docs.yml) | 文档构建 |

---

**维护者**: 薛小川  
**最后更新**: 2025-12-15
