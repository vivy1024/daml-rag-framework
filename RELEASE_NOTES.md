# DAML-RAG Framework v1.1.0 Release Notes

**发布日期**: 2025-11-07  
**版本号**: v1.1.0  
**作者**: 薛小川 (Xue Xiaochuan)

---

## 🎯 重大更新

### BGE查询复杂度分类器

本次更新的核心是集成了基于 **BAAI/bge-base-zh-v1.5** 向量模型的智能查询复杂度分类器，大幅优化了教师-学生模型的选择策略。

---

## ✨ 新功能

### 1. QueryComplexityClassifier (查询复杂度分类器)

**位置**: `daml_rag.learning.query_classifier`

**核心功能**:
- 🎯 基于语义相似度的智能分类
- 🧠 使用 BGE 中文向量模型（768维）
- ⚡ 余弦相似度计算 + 动态阈值
- 🔄 懒加载 + 向量缓存优化
- 🛡️ 硬编码关键词兜底策略

**使用示例**:

```python
from daml_rag.learning import QueryComplexityClassifier

# 初始化分类器
classifier = QueryComplexityClassifier(
    similarity_threshold=0.7,  # 高相似度阈值
    moderate_threshold=0.5,    # 低相似度阈值
    model_name="BAAI/bge-base-zh-v1.5"
)

# 分类查询
is_complex, similarity, reason = classifier.classify_complexity(
    "帮我设计一套增肌训练计划，我有腰椎间盘突出"
)

print(f"复杂度: {is_complex}")
print(f"相似度: {similarity:.2f}")
print(f"理由: {reason}")

# 输出:
# 复杂度: True
# 相似度: 0.85
# 理由: 与复杂查询示例高度相似（相似度=0.85）: '帮我设计一套增肌训练计划，我有...'
```

**自定义复杂查询库**:

```python
classifier = QueryComplexityClassifier(
    complex_query_examples=[
        "制定详细的康复训练方案",
        "设计周期化力量训练计划",
        "个性化营养和训练指导"
    ]
)

# 动态添加示例
classifier.add_complex_example("全面的健身指导方案")
```

---

## 📊 技术原理

### 余弦相似度分类

```
数学公式:
    similarity = (A · B) / (||A|| * ||B||)

分类规则:
    - similarity ≥ 0.7  → 复杂查询 → Teacher Model (DeepSeek)
    - similarity < 0.5  → 简单查询 → Student Model (Ollama)
    - 0.5 ≤ similarity < 0.7 → 中等复杂度 → Context-Dependent
```

### 三层降级策略

1. **一级**: BGE向量模型语义分类（最优）
2. **二级**: 硬编码关键词匹配（降级）
3. **三级**: 保守策略（默认使用教师模型）

---

## 🚀 性能优化

### 懒加载机制
- 首次调用时才加载 BGE 模型
- 避免不必要的内存占用

### 向量缓存
- 预计算复杂查询向量库
- 避免重复编码，提升响应速度

### 兜底策略
- 模型加载失败时自动降级
- 使用关键词匹配保证可用性

---

## 🔧 API 变更

### 新增模块

- `daml_rag.learning.query_classifier`
  - `QueryComplexityClassifier` (主类)

### 新增依赖

无额外依赖（使用已有的 `sentence-transformers`）

---

## 📦 升级指南

### 从 v1.0.0 升级到 v1.1.0

**1. 更新包**:
```bash
pip install --upgrade daml-rag-framework
```

**2. 导入新模块**:
```python
from daml_rag.learning import QueryComplexityClassifier
```

**3. 集成到现有系统**:
```python
from daml_rag import DAMLRAGFramework
from daml_rag.learning import QueryComplexityClassifier, ModelManager

# 创建分类器
classifier = QueryComplexityClassifier()

# 在模型选择时使用
def select_model(query: str) -> str:
    is_complex, sim, reason = classifier.classify_complexity(query)
    
    if is_complex:
        return "teacher"  # DeepSeek
    else:
        return "student"  # Ollama
```

**向后兼容性**: ✅ 完全兼容 v1.0.0

---

## 🐛 Bug 修复

本次版本无 bug 修复（纯功能新增）

---

## 📚 文档更新

- ✅ README.md - 新增 BGE 分类器说明
- ✅ CHANGELOG.md - 详细记录变更
- ✅ RELEASE_NOTES.md - 本发布说明

---

## 🔮 下一步计划 (v1.2.0)

- [ ] 更多领域适配器（医疗、金融、法律）
- [ ] 图形化配置界面
- [ ] 高级监控仪表板
- [ ] 分布式部署支持

---

## 🙏 致谢

感谢 BUILD_BODY 项目在实际应用中的验证和反馈，使得 BGE 分类器得以成功集成。

---

## 📞 联系方式

- **作者**: 薛小川 (Xue Xiaochuan)
- **邮箱**: 1765563156@qq.com
- **GitHub**: https://github.com/vivy1024/daml-rag-framework
- **Issues**: https://github.com/vivy1024/daml-rag-framework/issues

---

**Happy Coding! 🚀**
