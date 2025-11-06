# DAML-RAG Framework - 构建与发布完整指南

**版本**: v1.0.0  
**更新日期**: 2025-11-06  
**状态**: ✅ 打包就绪

---

## 🎉 打包状态

### ✅ 已完成

- [x] 项目结构重组（统一到 `daml_rag/` 包）
- [x] `pyproject.toml` 配置优化
- [x] `MANIFEST.in` 打包清单
- [x] `setup.py` 兼容性配置
- [x] `.gitignore` 更新
- [x] 构建脚本（Windows + Linux/Mac）
- [x] 测试脚本（本地安装验证）
- [x] 发布脚本（TestPyPI + PyPI）
- [x] 完整文档（PUBLISHING.md, QUICKSTART.md）
- [x] README.md 安装说明更新

### 🚀 当前状态

**项目已经可以打包发布到 PyPI！**

---

## 📦 新的目录结构

```
daml-rag-framework/
├── daml_rag/                    # ✅ 主包（统一结构）
│   ├── __init__.py
│   ├── core.py
│   ├── base.py
│   ├── config/
│   ├── interfaces/
│   ├── models/
│   ├── retrieval/               # 检索引擎
│   │   ├── vector/
│   │   ├── knowledge/
│   │   └── rules/
│   ├── orchestration/           # 任务编排
│   ├── learning/                # 学习模块
│   ├── adapters/                # 领域适配器
│   │   ├── base/
│   │   └── fitness/
│   └── cli/                     # 命令行工具
│       └── commands/
├── scripts/                     # ✅ 自动化脚本
│   ├── build.sh / .ps1
│   ├── test-install.sh / .ps1
│   ├── publish.sh / .ps1
│   └── quick-start.sh / .ps1
├── pyproject.toml               # ✅ 项目配置
├── MANIFEST.in                  # ✅ 打包清单
├── setup.py                     # ✅ 兼容性
├── .gitignore                   # ✅ Git 忽略
├── PUBLISHING.md                # ✅ 发布指南
├── QUICKSTART.md                # ✅ 快速开始
└── README.md                    # ✅ 更新完成
```

### 🗑️ 旧结构（已废弃，但保留）

以下目录是旧的结构，不会被打包：

```
daml-rag-core/          ❌ 旧结构（已迁移到 daml_rag/）
daml-rag-retrieval/     ❌ 旧结构
daml-rag-learning/      ❌ 旧结构
daml-rag-orchestration/ ❌ 旧结构
daml-rag-adapters/      ❌ 旧结构
daml-rag-cli/           ❌ 旧结构
```

**注意**: 这些目录在 `.gitignore` 中被标记为忽略。

---

## 🚀 快速使用指南

### 方式1：一键构建和测试

**Windows**:
```powershell
cd F:\build_body\daml-rag-framework
.\scripts\quick-start.ps1
```

**Linux/Mac**:
```bash
cd /path/to/daml-rag-framework
chmod +x scripts/*.sh
./scripts/quick-start.sh
```

### 方式2：分步执行

#### 步骤1: 构建

**Windows**:
```powershell
.\scripts\build.ps1
```

**Linux/Mac**:
```bash
./scripts/build.sh
```

#### 步骤2: 本地测试

**Windows**:
```powershell
.\scripts\test-install.ps1
```

**Linux/Mac**:
```bash
./scripts/test-install.sh
```

#### 步骤3: 发布到 TestPyPI

**Windows**:
```powershell
.\scripts\publish.ps1 test
```

**Linux/Mac**:
```bash
./scripts/publish.sh test
```

#### 步骤4: 发布到 PyPI

**Windows**:
```powershell
.\scripts\publish.ps1 prod
```

**Linux/Mac**:
```bash
./scripts/publish.sh prod
```

---

## 📝 关键文件说明

### 1. pyproject.toml

现代 Python 项目配置文件，包含：
- 项目元数据（名称、版本、作者等）
- 依赖声明
- 构建系统配置
- 工具配置（black, isort, pytest等）

### 2. MANIFEST.in

指定哪些非 Python 文件需要打包：
- 文档（README.md, LICENSE 等）
- 配置文件（*.yaml, *.yml, *.json）
- 理论文档（docs/theory/）

### 3. setup.py

兼容性配置文件，支持旧版 pip。
实际配置在 `pyproject.toml` 中。

### 4. .gitignore

新增了对旧结构的忽略：
```
daml-rag-core/
daml-rag-retrieval/
daml-rag-learning/
daml-rag-orchestration/
daml-rag-adapters/
daml-rag-cli/
```

---

## 🔍 验证打包结果

### 检查构建产物

```bash
ls -lh dist/
```

应该看到：
```
daml_rag_framework-1.0.0-py3-none-any.whl
daml_rag_framework-1.0.0.tar.gz
```

### 检查包内容

```bash
# 查看 wheel 包内容
unzip -l dist/daml_rag_framework-1.0.0-py3-none-any.whl

# 或使用 tar 查看源码包
tar -tzf dist/daml_rag_framework-1.0.0.tar.gz
```

应该只包含 `daml_rag/` 包，不包含旧的 `daml-rag-*` 目录。

---

## 📦 PyPI 发布前准备

### 1. 注册 PyPI 账号

- **生产环境**: https://pypi.org/account/register/
- **测试环境**: https://test.pypi.org/account/register/

### 2. 创建 API Token

1. 登录 PyPI
2. 进入 Account Settings → API tokens
3. 点击 "Add API token"
4. 名称: `daml-rag-framework`
5. Scope: "Entire account"
6. 保存生成的 Token

### 3. 配置 .pypirc

创建 `~/.pypirc` 文件（Windows: `C:\Users\<用户名>\.pypirc`）:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...  # 你的 Token

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgENdGVzdC5weXBp...  # 你的 TestPyPI Token
```

**⚠️ 重要**: 不要将 `.pypirc` 提交到 Git！

---

## ✅ 发布检查清单

### 发布前

- [ ] 所有代码已提交到 Git
- [ ] 版本号已更新（pyproject.toml）
- [ ] CHANGELOG.md 已更新
- [ ] 本地构建测试通过
- [ ] 本地安装测试通过
- [ ] 已在 TestPyPI 测试成功

### 发布后

- [ ] 在 PyPI 上验证页面
- [ ] 测试从 PyPI 安装
- [ ] 创建 Git Tag
- [ ] 发布 GitHub Release
- [ ] 更新文档链接

---

## 🎯 使用场景

### 场景1: 本地开发测试

```bash
# 开发模式安装（推荐）
pip install -e .

# 修改代码后无需重新安装
python test_script.py
```

### 场景2: 分享给其他开发者

```bash
# 构建包
./scripts/build.sh

# 分享 dist/ 目录中的文件
# 其他人可以这样安装:
pip install daml_rag_framework-1.0.0-py3-none-any.whl
```

### 场景3: 发布到公司内部 PyPI

```bash
# 上传到私有 PyPI 服务器
twine upload --repository-url https://your-pypi-server.com dist/*
```

### 场景4: 正式发布

```bash
# 发布到 PyPI
./scripts/publish.sh prod

# 用户可以直接安装
pip install daml-rag-framework
```

---

## 🔧 故障排除

### 问题1: 导入失败

**症状**: `ModuleNotFoundError: No module named 'daml_rag'`

**解决方法**:
```bash
# 重新安装
pip uninstall daml-rag-framework
pip install -e .
```

### 问题2: CLI 工具找不到

**症状**: `daml-rag: command not found`

**解决方法**:
```bash
# 确认安装路径
pip show daml-rag-framework

# 重新安装
pip install -e .
```

### 问题3: 构建失败

**症状**: `ModuleNotFoundError: No module named 'build'`

**解决方法**:
```bash
pip install --upgrade build twine setuptools wheel
```

### 问题4: 旧包冲突

**症状**: 导入错误或版本混乱

**解决方法**:
```bash
# 完全卸载
pip uninstall daml-rag-framework -y

# 清理缓存
pip cache purge

# 重新安装
pip install -e .
```

---

## 📚 更多资源

- **快速开始**: [QUICKSTART.md](QUICKSTART.md)
- **发布指南**: [PUBLISHING.md](PUBLISHING.md)
- **项目说明**: [README.md](README.md)
- **限制说明**: [LIMITATIONS.md](LIMITATIONS.md)

---

## 🎉 恭喜！

你的 DAML-RAG 框架现在已经：

✅ 结构规范化  
✅ 可以打包  
✅ 可以发布  
✅ 可以通过 pip 安装  

**准备好发布到 PyPI 了！** 🚀

---

**维护者**: 薛小川  
**邮箱**: 1765563156@qq.com  
**最后更新**: 2025-11-06

