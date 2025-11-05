# 变更日志

本文档记录了 DAML-RAG Framework 的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并且项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 新增
- DAML-RAG Framework 初始发布
- 三层检索架构（向量+知识图谱+规则）
- 推理时学习机制
- 双模型智能调度
- 领域适配器系统
- CLI 工具
- 完整的配置管理
- 性能监控和健康检查

## [1.0.0] - 2025-11-05

### 新增
- 🎯 **核心框架**：DAML-RAG Framework v1.0.0 正式发布
- 🔍 **三层检索架构**：
  - 向量检索层：支持中文向量化模型和语义搜索
  - 知识图谱层：基于 Neo4j 的实体关系推理
  - 规则过滤层：质量评分和异常检测
- 🧠 **推理时学习**：
  - 经验存储和检索
  - 反馈学习机制
  - 自适应策略调整
- ⚡ **智能模型调度**：
  - 教师模型（DeepSeek）+ 学生模型（Ollama）
  - 85% Token 节省，93% 成本降低
  - 基于复杂度的自动模型选择
- 🔌 **领域适配器**：
  - 健身领域适配器（23个专业工具）
  - 医疗领域适配器（诊断、治疗工具）
  - 教育领域适配器（课程设计工具）
  - 自定义领域适配器模板
- 🛠️ **CLI 工具**：
  - `daml-rag init` - 项目初始化
  - `daml-rag dev` - 开发服务器
  - `daml-rag deploy` - 部署工具
  - `daml-rag health` - 健康检查
- ⚙️ **配置管理**：
  - YAML/JSON 配置文件支持
  - 环境变量配置
  - 配置验证和合并
- 📊 **监控系统**：
  - 实时性能指标
  - 查询统计和监控
  - 组件健康检查
- 🛡️ **质量保证**：
  - 知识污染防护
  - 异常检测和信誉系统
  - 自动质量评估

### 技术规格
- **框架语言**：Python 3.8+
- **协议支持**：MCP (Model Context Protocol)
- **数据库**：Neo4j, Qdrant, Redis
- **模型支持**：OpenAI, DeepSeek, Ollama
- **部署方式**：Docker, Kubernetes, 云平台

### 性能指标
- **响应时间**：< 1秒（GraphRAG检索）
- **缓存命中率**：> 60%
- **用户满意度**：4.4/5
- **质量提升**：38%

### 文档和示例
- 📖 完整的 API 文档
- 🚀 快速开始指南
- 💡 健身教练助手示例
- 🔧 自定义适配器开发指南
- 📚 最佳实践文档

### 许可证
- Apache License 2.0 - 商业友好开源协议

---

## 版本说明

### 版本号格式
使用语义化版本控制：`MAJOR.MINOR.PATCH`

- **MAJOR**：不兼容的 API 修改
- **MINOR**：向下兼容的功能性新增
- **PATCH**：向下兼容的问题修正

### 发布周期
- **主版本**：每年 1-2 次
- **次版本**：每季度 1-2 次
- **修订版本**：根据需要随时发布

### 支持政策
- **当前版本**：完全支持，包括新功能和修复
- **前一个主版本**：仅安全修复和关键 bug 修复
- **更早版本**：不再支持

### 升级指南
查看每个版本的详细升级说明：
- [v1.0.0 升级指南](./docs/upgrade/v1.0.0.md)

---

## 贡献者

感谢以下贡献者让 DAML-RAG Framework 变得更好：

### 核心团队
- [@creator](https://github.com/creator) - 项目创建者和维护者
- [@maintainer1](https://github.com/maintainer1) - 核心框架开发
- [@maintainer2](https://github.com/maintainer2) - 领域适配器开发

### 社区贡献者
- [@contributor1](https://github.com/contributor1) - 文档改进
- [@contributor2](https://github.com/contributor2) - Bug 修复
- [@contributor3](https://github.com/contributor3) - 测试用例

完整的贡献者列表：[Contributors](https://github.com/daml-rag/daml-rag-framework/graphs/contributors)

---

## 路线图

### v1.1.0 (计划中)
- [ ] 更多领域适配器（金融、法律等）
- [ ] 图形化配置界面
- [ ] 高级监控仪表板
- [ ] 自动化测试框架

### v1.2.0 (计划中)
- [ ] 多语言支持
- [ ] 分布式部署支持
- [ ] 高级缓存策略
- [ ] 自定义插件系统

### v2.0.0 (远期规划)
- [ ] 图形化工作流编辑器
- [ ] 机器学习模型训练支持
- [ ] 企业级功能
- [ ] 云原生架构

---

## 反馈和支持

- 📧 [邮件支持](mailto:support@daml-rag.org)
- 💬 [GitHub Discussions](https://github.com/daml-rag/daml-rag-framework/discussions)
- 🐛 [问题反馈](https://github.com/daml-rag/daml-rag-framework/issues)
- 📖 [官方文档](https://docs.daml-rag.org)