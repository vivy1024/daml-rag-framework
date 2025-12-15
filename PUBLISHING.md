# PyPI发布指南

**版本**: v2.0.0  
**更新日期**: 2025-12-15

---

## 📦 发布准备清单

### 1. 配置PyPI API Token

#### 获取PyPI API Token

1. 访问 https://pypi.org/manage/account/token/
2. 登录你的PyPI账号
3. 点击"Add API token"
4. 设置Token名称：`daml-rag-framework`
5. 设置Scope：`Project: daml-rag-framework`（如果项目已存在）或 `Entire account`
6. 复制生成的Token（格式：`pypi-...`）

#### 配置GitHub Secrets

1. 访问 https://github.com/vivy1024/daml-rag-framework/settings/secrets/actions
2. 点击"New repository secret"
3. 添加以下Secrets：

**PYPI_API_TOKEN**:
- Name: `PYPI_API_TOKEN`
- Value: 你的PyPI API Token（`pypi-...`）

**TEST_PYPI_API_TOKEN** (可选，用于测试):
- Name: `TEST_PYPI_API_TOKEN`
- Value: 你的Test PyPI API Token
- 获取地址: https://test.pypi.org/manage/account/token/

---

## 🚀 发布方式

### 方式1：通过GitHub Release自动发布（推荐）

1. **创建Release**
   ```bash
   # 确保代码已推送
   git push origin main
   git push origin v2.0.0
   ```

2. **在GitHub上创建Release**
   - 访问: https://github.com/vivy1024/daml-rag-framework/releases/new
   - Tag: `v2.0.0`
   - Release title: `v2.0.0 - DAML-RAG框架正式发布`
   - Description: 
     ```markdown
     # DAML-RAG v2.0.0 正式发布 🎉

     ## 主要特性
     - ✅ P0重构: DAG编排器框架层/应用层分离
     - ✅ P1重构: 模型选择器和Few-Shot检索器分离
     - ✅ 三层检索引擎完整实现
     - ✅ MCP任务编排系统
     - ✅ 智能模型选择(BGE分类器)
     - ✅ Few-Shot学习系统

     ## 数据库状态
     - Neo4j: 3,657节点, 45,885关系
     - Qdrant: 向量数据完整
     - 15个Python MCP工具已实现

     ## 安装
     ```bash
     pip install daml-rag-framework
     ```

     ## 文档
     - [快速开始](https://github.com/vivy1024/daml-rag-framework#快速开始)
     - [完整文档](https://github.com/vivy1024/daml-rag-framework/tree/main/docs)
     ```

3. **发布Release**
   - 点击"Publish release"
   - GitHub Actions会自动触发发布到PyPI

4. **验证发布**
   - 等待GitHub Actions完成（约2-5分钟）
   - 访问: https://pypi.org/project/daml-rag-framework/
   - 确认版本v2.0.0已发布

---

### 方式2：手动发布

#### 步骤1：构建包

```bash
cd daml-rag-framework

# 安装构建工具
pip install build twine

# 清理旧的构建
rm -rf dist/ build/ *.egg-info

# 构建包
python -m build
```

#### 步骤2：检查包

```bash
# 检查包的完整性
twine check dist/*
```

#### 步骤3：测试发布到Test PyPI（可选）

```bash
# 发布到Test PyPI
twine upload --repository testpypi dist/*

# 输入你的Test PyPI凭据
# Username: __token__
# Password: 你的Test PyPI API Token

# 测试安装
pip install --index-url https://test.pypi.org/simple/ daml-rag-framework
```

#### 步骤4：发布到PyPI

```bash
# 发布到PyPI
twine upload dist/*

# 输入你的PyPI凭据
# Username: __token__
# Password: 你的PyPI API Token
```

#### 步骤5：验证发布

```bash
# 等待几分钟后测试安装
pip install daml-rag-framework==2.0.0

# 验证版本
python -c "import daml_rag_framework; print(daml_rag_framework.__version__)"
```

---

## 🔧 使用.pypirc配置（可选）

创建 `~/.pypirc` 文件：

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-你的PyPI_API_Token

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-你的Test_PyPI_API_Token
```

**注意**: 确保 `.pypirc` 文件权限为 600:
```bash
chmod 600 ~/.pypirc
```

使用配置文件后，可以简化发布命令：
```bash
# 发布到Test PyPI
twine upload --repository testpypi dist/*

# 发布到PyPI
twine upload dist/*
```

---

## 📝 发布后检查清单

- [ ] PyPI页面显示正确: https://pypi.org/project/daml-rag-framework/
- [ ] 版本号正确: v2.0.0
- [ ] README显示正常
- [ ] 依赖列表正确
- [ ] 可以通过pip安装: `pip install daml-rag-framework`
- [ ] 导入测试通过: `import daml_rag_framework`
- [ ] GitHub Release已创建
- [ ] 文档链接正常

---

## 🐛 常见问题

### 问题1：上传失败 - 403 Forbidden

**原因**: API Token权限不足或已过期

**解决方案**:
1. 重新生成PyPI API Token
2. 确保Token的Scope包含该项目
3. 更新GitHub Secrets中的Token

### 问题2：包名已存在

**原因**: PyPI上已有同名包

**解决方案**:
1. 如果是你的包，使用正确的API Token
2. 如果不是你的包，需要更改包名

### 问题3：构建失败

**原因**: pyproject.toml配置错误

**解决方案**:
1. 检查pyproject.toml语法
2. 确保所有依赖都已安装
3. 运行 `python -m build --verbose` 查看详细错误

### 问题4：GitHub Actions发布失败

**原因**: Secrets未配置或配置错误

**解决方案**:
1. 检查GitHub Secrets是否已添加
2. 确认Secret名称为 `PYPI_API_TOKEN`
3. 确认Token格式正确（以`pypi-`开头）

---

## 📚 相关文档

- [PyPI官方文档](https://packaging.python.org/tutorials/packaging-projects/)
- [Twine文档](https://twine.readthedocs.io/)
- [GitHub Actions发布文档](https://docs.github.com/en/actions/publishing-packages/publishing-python-packages)

---

**维护者**: 薛小川  
**最后更新**: 2025-12-15
