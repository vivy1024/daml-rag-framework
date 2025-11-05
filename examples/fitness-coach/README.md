# 健身教练助手示例

基于玉珍健身框架的智能健身教练助手，展示了如何使用框架构建垂直领域AI应用。

## 🎯 项目特性

- **智能对话**: 基于23个专业健身工具的智能对话
- **个性化计划**: 根据用户画像制定个性化训练计划
- ** injury康复**: 运动损伤康复指导和替代方案
- **营养建议**: TDEE计算和膳食计划推荐
- **知识图谱**: 2,447个健身实体节点的知识推理

## 🏗️ 技术架构

```
前端 (FastAPI + HTML)      ←→  玉珍健身 框架  ←→  MCP服务器池
     ↓                           ↓                         ↓
  Web界面                    三层检索引擎              健身专业工具
  用户交互                    向量+图谱+规则            23个MCP工具
  实时对话                    推理时学习               知识图谱推理
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/玉珍健身/fitness-coach-example.git
cd fitness-coach-example

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置设置

```bash
# 复制配置文件
cp config.example.yaml config.yaml

# 编辑配置文件
vim config.yaml
```

配置文件示例：
```yaml
domain: fitness
debug: true

retrieval:
  vector_model: "BAAI/bge-base-zh-v1.5"
  top_k: 5
  similarity_threshold: 0.6
  enable_kg: true
  enable_rules: true

orchestration:
  max_parallel_tasks: 10
  timeout_seconds: 30

learning:
  teacher_model: "deepseek"
  student_model: "ollama-qwen2.5"
  experience_threshold: 3.5

domain_config:
  mcp_servers:
    - name: "user-profile"
      command: "python"
      args: ["mcp-servers/user-profile-stdio/server.py"]
    - name: "professional-coach"
      command: "python"
      args: ["mcp-servers/professional-coach-stdio/server.py"]
```

### 3. 启动服务

```bash
# 启动玉珍健身框架
python app.py

# 或使用CLI工具
玉珍健身 dev
```

访问 http://localhost:8000 开始使用。

## 💡 示例对话

### 训练计划制定
```
用户: 我想制定一个增肌计划，每周训练4次

助手: 我来为您制定一个个性化的增肌计划。基于您的需求，我建议：

**训练分化**：
- 周一：胸部 + 三头肌
- 周二：背部 + 二头肌
- 周三：休息
- 周四：腿部 + 肩部
- 周五：手臂 + 核心训练
- 周末：休息

**动作推荐**：
1. 杠铃卧推 4组x8-12次
2. 引体向上 4组x6-10次
3. 深蹲 4组x8-12次
4. 硬拉 3组x6-8次

**营养建议**：
- 蛋白质：体重(kg) x 2g
- 碳水化合物：体重(kg) x 4g
- 脂肪：体重(kg) x 1g

需要我为您提供详细的动作要领吗？
```

### 损伤康复咨询
```
用户: 我膝盖有旧伤，想训练腿部有什么替代动作？

助手: 考虑到您的膝盖旧伤，我推荐以下安全的腿部训练替代方案：

**低冲击有氧运动**：
- 游泳：对关节零冲击，全身运动
- 固定单车：可调节阻力，膝盖压力小
- 椭圆机：模拟跑步但减少冲击

**力量训练替代**：
- 臀桥：强化臀部和腘绳肌
- 坐姿腿屈伸：孤立训练股四头肌
- 哑铃罗马尼亚硬拉：强化腘绳肌
- 提踵：强化小腿肌群

**注意事项**：
1. 避免深蹲、跳跃等高冲击动作
2. 始终保持正确姿势
3. 如有疼痛立即停止
4. 建议咨询物理治疗师

需要我为您设计具体的训练计划吗？
```

## 🔧 核心组件

### 1. 应用入口 (app.py)
```python
from daml_rag import DAMLRAGFramework, DAMLRAGConfig
from daml_rag_adapters.fitness import FitnessDomainAdapter

# 初始化框架
config = DAMLRAGConfig.from_file("config.yaml")
framework = DAMLRAGFramework(config)

# 初始化健身领域适配器
adapter = FitnessDomainAdapter(config.domain_config)
await adapter.initialize()

# 处理用户查询
result = await framework.process_query("我想制定增肌计划")
```

### 2. Web界面 (web.py)
```python
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/chat")
async def chat(request: ChatRequest):
    result = await framework.process_query(request.message)
    return {"response": result.response}
```

### 3. 配置管理 (config.yaml)
完整的框架配置，包括检索、编排、学习等各模块参数。

## 📊 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 响应时间 | < 1秒 | GraphRAG检索 + 模型生成 |
| Token节省 | 85% | 相比直接调用大模型 |
| 成本降低 | 93% | 双模型机制优化 |
| 用户满意度 | 4.4/5 | 基于反馈评分 |
| 工具数量 | 23个 | 覆盖健身全场景 |

## 🛠️ 扩展开发

### 添加新工具
```python
from daml_rag.interfaces import IMCPTool

class CustomFitnessTool(IMCPTool):
    async def call(self, params):
        # 工具实现
        return ToolResult(success=True, data=result)

# 注册工具
tool_registry.register_tool("custom_tool", CustomFitnessTool())
```

### 自定义检索策略
```python
from daml_rag.interfaces import IVectorRetriever

class CustomRetriever(IVectorRetriever):
    async def retrieve(self, query, top_k=5):
        # 自定义检索逻辑
        return RetrievalResult(...)
```

## 📚 学习资源

- [玉珍健身框架文档](https://docs.玉珍健身.org)
- [健身领域适配器指南](https://docs.玉珍健身.org/adapters/fitness)
- [MCP协议规范](https://modelcontextprotocol.org)
- [API参考文档](https://docs.玉珍健身.org/api)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个示例项目！

## 📄 许可证

MIT License