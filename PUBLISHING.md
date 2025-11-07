# DAML-RAG Framework - 发布指南

**版本**: v1.0.0  
**更新日期**: 2025-11-06  

---

## 📋 发布前检查清单

在发布到 PyPI 之前，请确认以下事项：

- [ ] 所有测试通过
- [ ] 文档已更新（README.md, CHANGELOG.md）
- [ ] 版本号已更新（pyproject.toml）
- [ ] LICENSE 文件存在且正确
- [ ] 代码符合 PEP 8 标准
- [ ] 已在本地测试安装

---

## 🔧 发布步骤

### 第1步：准备 PyPI 账号

#### 1.1 注册账号

- **PyPI（生产环境）**: https://pypi.org/account/register/
- **TestPyPI（测试环境）**: https://test.pypi.org/account/register/

#### 1.2 创建 API Token

1. 登录 PyPI/TestPyPI
2. 进入 Account Settings → API tokens
3. 点击 "Add API token"
4. 名称: `daml-rag-framework`
5. Scope: "Entire account" 或指定项目
6. 复制生成的 token（只显示一次！）

#### 1.3 配置 `.pypirc`

在用户目录创建 `~/.pypirc` 文件：

**Linux/Mac**: `~/.pypirc`  
**Windows**: `C:\Users\<用户名>\.pypirc`

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...  # 你的 PyPI API Token

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgENdGVzdC5weXBp...  # 你的 TestPyPI API Token
```

**⚠️ 重要**: `.pypirc` 包含敏感信息，不要提交到版本控制！

---

### 第2步：构建发布包

#### Windows:

```powershell
cd F:\build_body\daml-rag-framework
.\scripts\build.ps1
```

#### Linux/Mac:

```bash
cd /path/to/daml-rag-framework
./scripts/build.sh
```

构建完成后，会在 `dist/` 目录生成：

```
dist/
├── daml_rag_framework-1.0.0-py3-none-any.whl
└── daml_rag_framework-1.0.0.tar.gz
```

---

### 第3步：本地测试安装

#### Windows:

```powershell
.\scripts\test-install.ps1
```

#### Linux/Mac:

```bash
./scripts/test-install.sh
```

测试项目：
- ✅ 包能正常安装
- ✅ 核心模块能正常导入
- ✅ CLI 工具能正常运行

---

### 第4步：发布到 TestPyPI（测试）

**强烈建议先发布到 TestPyPI 测试！**

#### Windows:

```powershell
.\scripts\publish.ps1 test
```

#### Linux/Mac:

```bash
./scripts/publish.sh test
```

#### 从 TestPyPI 安装测试：

```bash
pip install --index-url https://test.pypi.org/simple/ daml-rag-framework
```

---

### 第5步：发布到 PyPI（生产）

**⚠️ 警告：发布后无法撤回，请谨慎操作！**

#### 确认清单：

- [ ] TestPyPI 测试通过
- [ ] 版本号正确
- [ ] CHANGELOG.md 已更新
- [ ] 确认构建产物正确

#### Windows:

```powershell
.\scripts\publish.ps1 prod
```

#### Linux/Mac:

```bash
./scripts/publish.sh prod
```

发布成功后，用户可以通过以下命令安装：

```bash
pip install daml-rag-framework
```

---

## 📦 手动发布（备用方法）

如果脚本无法使用，可以手动执行：

### 1. 清理旧构建

```bash
rm -rf build/ dist/ *.egg-info
```

### 2. 构建包

```bash
python -m build
```

### 3. 检查包

```bash
twine check dist/*
```

### 4. 上传到 TestPyPI

```bash
twine upload --repository testpypi dist/*
```

### 5. 上传到 PyPI

```bash
twine upload dist/*
```

---

## 🔄 版本更新流程

### 1. 更新版本号

编辑 `pyproject.toml`:

```toml
[project]
version = "1.1.0"  # 修改这里
```

### 2. 更新 CHANGELOG.md

```markdown
## [1.1.0] - 2025-11-XX

### 新增
- 新功能描述

### 修复
- Bug 修复

### 变更
- 破坏性变更（如果有）
```

### 3. 更新代码中的版本号

编辑 `daml_rag/__init__.py`:

```python
__version__ = "1.1.0"
```

### 4. 提交版本更新

```bash
git add .
git commit -m "chore: bump version to 1.1.0"
git tag -a v1.1.0 -m "Release v1.1.0"
git push origin main --tags
```

### 5. 重新构建和发布

按照上述步骤重新构建和发布。

---

## 📊 版本号规范

遵循 **语义化版本 2.0.0** (SemVer):

```
主版本号.次版本号.修订号

1.2.3
│ │ │
│ │ └─── 修订号: Bug 修复（向后兼容）
│ └───── 次版本号: 新功能（向后兼容）
└─────── 主版本号: 破坏性变更（不向后兼容）
```

**示例**：
- `1.0.0` → `1.0.1` : Bug 修复
- `1.0.1` → `1.1.0` : 新增功能
- `1.1.0` → `2.0.0` : API 破坏性变更

---

## ⚠️ 常见问题

### 问题1: `twine: command not found`

**解决方法**:
```bash
pip install twine
```

### 问题2: 上传失败 "401 Unauthorized"

**原因**: API Token 配置错误

**解决方法**:
1. 检查 `.pypirc` 文件格式
2. 确认 Token 正确复制（包含 `pypi-` 前缀）
3. 确认 Token 未过期

### 问题3: "File already exists"

**原因**: 相同版本号的包已经发布

**解决方法**:
1. 更新版本号（PyPI 不允许覆盖已发布的版本）
2. 重新构建和上传

### 问题4: Windows 脚本执行权限问题

**解决方法**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 🔐 安全建议

1. **不要在代码中硬编码 Token**
2. **使用 `.pypirc` 存储凭证**
3. **将 `.pypirc` 加入 `.gitignore`**
4. **定期轮换 API Token**
5. **为不同项目使用不同的 Token**

---

## 📚 参考资源

- [PyPI 官方文档](https://packaging.python.org/)
- [Twine 使用指南](https://twine.readthedocs.io/)
- [语义化版本规范](https://semver.org/lang/zh-CN/)
- [Python 打包用户指南](https://packaging.python.org/tutorials/packaging-projects/)

---

## ✅ 发布成功！

发布成功后，你的项目将在以下位置可见：

- **PyPI 页面**: https://pypi.org/project/daml-rag-framework/
- **下载统计**: https://pypistats.org/packages/daml-rag-framework
- **用户可以安装**: `pip install daml-rag-framework`

---

**维护者**: 薛小川  
**邮箱**: 1765563156@qq.com  
**最后更新**: 2025-11-06



