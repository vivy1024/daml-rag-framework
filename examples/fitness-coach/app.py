"""
健身教练Web应用 - 玉珍健身 框架示例
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 导入玉珍健身框架
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from daml_rag import DAMLRAGFramework, DAMLRAGConfig
from daml_rag_adapters.fitness import FitnessDomainAdapter


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局框架实例
framework: Optional[DAMLRAGFramework] = None
adapter: Optional[FitnessDomainAdapter] = None


# 数据模型
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    sources: list
    model_used: str
    execution_time: float
    session_id: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    framework: dict
    adapter: dict
    timestamp: str


class SessionInfo(BaseModel):
    session_id: str
    user_id: str
    created_at: str
    message_count: int
    last_activity: str


# 会话管理
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, user_id: str = "anonymous") -> str:
        """创建新会话"""
        session_id = f"session_{int(datetime.now().timestamp())}_{hash(user_id) % 10000}"
        self.sessions[session_id] = {
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'message_count': 0,
            'last_activity': datetime.now().isoformat(),
            'context': {}
        }
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        return self.sessions.get(session_id)

    def update_session(self, session_id: str, **kwargs):
        """更新会话信息"""
        if session_id in self.sessions:
            self.sessions[session_id].update(kwargs)
            self.sessions[session_id]['last_activity'] = datetime.now().isoformat()
            self.sessions[session_id]['message_count'] += 1

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """清理旧会话"""
        current_time = datetime.now()
        to_remove = []

        for session_id, session_info in self.sessions.items():
            last_activity = datetime.fromisoformat(session_info['last_activity'])
            age_hours = (current_time - last_activity).total_seconds() / 3600
            if age_hours > max_age_hours:
                to_remove.append(session_id)

        for session_id in to_remove:
            del self.sessions[session_id]

        logger.info(f"清理了 {len(to_remove)} 个旧会话")


session_manager = SessionManager()


# FastAPI应用
app = FastAPI(
    title="健身教练助手",
    description="基于玉珍健身 框架的智能健身教练助手",
    version="1.0.0"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    global framework, adapter

    try:
        logger.info("正在初始化健身教练助手...")

        # 加载配置
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        if not os.path.exists(config_path):
            # 创建默认配置
            await create_default_config(config_path)

        config = DAMLRAGConfig.from_file(config_path)
        logger.info("配置加载完成")

        # 创建框架实例
        framework = DAMLRAGFramework(config)
        logger.info("框架实例创建完成")

        # 初始化健身领域适配器
        adapter = FitnessDomainAdapter(config.domain_config)
        await adapter.initialize()
        logger.info("健身领域适配器初始化完成")

        # 初始化框架
        await framework.initialize()
        logger.info("框架初始化完成")

        # 注册组件
        framework.registry.register_component(FitnessDomainAdapter, adapter)
        logger.info("组件注册完成")

        logger.info("✅ 健身教练助手启动成功!")

    except Exception as e:
        logger.error(f"❌ 应用启动失败: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    global framework, adapter

    try:
        logger.info("正在关闭健身教练助手...")

        if framework:
            await framework.shutdown()
            logger.info("框架已关闭")

        if adapter:
            await adapter.cleanup()
            logger.info("适配器已清理")

        logger.info("✅ 健身教练助手已关闭")

    except Exception as e:
        logger.error(f"关闭应用时出错: {str(e)}")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "欢迎使用健身教练助手",
        "version": "1.0.0",
        "framework": "玉珍健身 框架",
        "endpoints": {
            "chat": "/api/chat",
            "health": "/api/health",
            "sessions": "/api/sessions",
            "docs": "/docs"
        }
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理聊天请求"""
    global framework

    if not framework:
        raise HTTPException(status_code=503, detail="服务未就绪")

    try:
        # 获取或创建会话
        session_id = request.session_id
        if not session_id:
            session_id = session_manager.create_session(request.user_id)
        else:
            session_info = session_manager.get_session(session_id)
            if not session_info:
                session_id = session_manager.create_session(request.user_id)

        # 构建上下文
        session_info = session_manager.get_session(session_id)
        context = {
            'user_id': request.user_id,
            'session_id': session_id,
            'session_context': session_info.get('context', {})
        }

        # 处理查询
        result = await framework.process_query(request.message, context)

        # 更新会话
        session_manager.update_session(session_id, last_query=request.message)

        response = ChatResponse(
            response=result.response,
            sources=result.sources,
            model_used=result.model_used,
            execution_time=result.execution_time,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )

        return response

    except Exception as e:
        logger.error(f"聊天处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    global framework, adapter

    framework_health = {"status": "not_initialized"}
    adapter_health = {"status": "not_initialized"}

    if framework:
        framework_health = await framework.health_check()

    if adapter:
        adapter_health = await adapter.health_check()

    overall_status = "healthy"
    if framework_health.get("overall_status") != "healthy":
        overall_status = "degraded"
    if adapter_health.get("overall_status") != "healthy":
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        framework=framework_health,
        adapter=adapter_health,
        timestamp=datetime.now().isoformat()
    )


@app.get("/api/sessions")
async def list_sessions():
    """列出所有会话"""
    sessions = []
    for session_id, session_info in session_manager.sessions.items():
        sessions.append(SessionInfo(
            session_id=session_id,
            user_id=session_info['user_id'],
            created_at=session_info['created_at'],
            message_count=session_info['message_count'],
            last_activity=session_info['last_activity']
        ))

    return {"sessions": sessions, "total": len(sessions)}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    if session_id in session_manager.sessions:
        del session_manager.sessions[session_id]
        return {"message": "会话已删除"}
    else:
        raise HTTPException(status_code=404, detail="会话不存在")


@app.get("/api/stats")
async def get_stats():
    """获取统计信息"""
    global framework, adapter

    stats = {
        "timestamp": datetime.now().isoformat(),
        "sessions": {
            "total": len(session_manager.sessions),
            "active": len([s for s in session_manager.sessions.values()
                         if (datetime.now() - datetime.fromisoformat(s['last_activity'])).total_seconds() < 3600])
        }
    }

    if framework:
        framework_stats = framework.get_framework_stats()
        stats["framework"] = framework_stats

    if adapter:
        adapter_stats = await adapter.get_statistics()
        stats["adapter"] = adapter_stats

    return stats


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket聊天接口"""
    await websocket.accept()

    try:
        # 获取或创建会话
        session_info = session_manager.get_session(session_id)
        if not session_info:
            session_id = session_manager.create_session()
            session_info = session_manager.get_session(session_id)

        await websocket.send_json({
            "type": "session_created",
            "session_id": session_id,
            "message": "会话已创建，开始对话吧！"
        })

        while True:
            # 接收消息
            data = await websocket.receive_json()
            message = data.get("message", "")
            user_id = data.get("user_id", "anonymous")

            if not message:
                await websocket.send_json({
                    "type": "error",
                    "message": "消息不能为空"
                })
                continue

            # 处理消息
            try:
                global framework
                if framework:
                    context = {
                        'user_id': user_id,
                        'session_id': session_id,
                        'session_context': session_info.get('context', {})
                    }

                    result = await framework.process_query(message, context)
                    session_manager.update_session(session_id, last_query=message)

                    await websocket.send_json({
                        "type": "response",
                        "response": result.response,
                        "sources": result.sources,
                        "model_used": result.model_used,
                        "execution_time": result.execution_time,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "服务未就绪"
                    })

            except Exception as e:
                logger.error(f"WebSocket消息处理失败: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": "处理消息时出错"
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket断开连接: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket连接错误: {str(e)}")


async def create_default_config(config_path: str):
    """创建默认配置文件"""
    default_config = """
domain: fitness
debug: true
environment: development

retrieval:
  vector_model: "BAAI/bge-base-zh-v1.5"
  top_k: 5
  similarity_threshold: 0.6
  cache_ttl: 300
  enable_kg: true
  enable_rules: true
  faiss_index_type: "flat"

orchestration:
  max_parallel_tasks: 10
  timeout_seconds: 30
  retry_attempts: 3
  fail_fast: true

learning:
  teacher_model: "deepseek"
  student_model: "ollama-qwen2.5"
  experience_threshold: 3.5
  adaptive_threshold: 0.7

domain_config:
  knowledge_graph_path: "./data/knowledge_graph.db"
  mcp_servers: []
  domain_specific: {}

logging:
  log_level: "INFO"
  log_to_file: true
  log_file_path: "./logs/fitness-coach.log"
  structured_logging: false

max_concurrent_queries: 50
query_timeout: 60
health_check_interval: 30
enable_performance_monitoring: true
metrics_collection_enabled: true
"""

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(default_config.strip())

    logger.info(f"默认配置文件已创建: {config_path}")


def create_app():
    """创建FastAPI应用实例"""
    return app


if __name__ == "__main__":
    # 直接运行应用
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )