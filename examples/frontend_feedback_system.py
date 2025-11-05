#!/usr/bin/env python3
"""
前端反馈收集系统示例
基于DAML-RAG框架的用户反馈收集和分析系统

这个示例展示了：
1. 前端反馈收集的完整流程
2. 反馈数据的存储和分析
3. 用户满意度和回答质量评估
4. 实时反馈展示和统计

作者：BUILD_BODY Team
版本：v1.0.0
日期：2025-11-05
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import statistics

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 数据模型
# ============================================================================

@dataclass
class FeedbackData:
    """反馈数据模型"""
    id: str
    timestamp: datetime
    session_id: str
    user_id: str
    query: str
    answer: str
    user_rating: int  # 1-5星评分
    user_feedback: Optional[str] = None
    improvement_suggestions: Optional[str] = None
    response_time: float = 0.0
    sources_count: int = 0
    domain: str = "general"
    device_type: str = "web"
    browser_info: Optional[str] = None


class FeedbackRequest(BaseModel):
    """反馈请求模型"""
    session_id: str = Field(..., description="会话ID")
    user_id: str = Field(..., description="用户ID")
    query: str = Field(..., description="原查询")
    answer: str = Field(..., description="AI回答")
    user_rating: int = Field(..., ge=1, le=5, description="用户评分(1-5)")
    user_feedback: Optional[str] = Field(None, description="用户反馈")
    improvement_suggestions: Optional[str] = Field(None, description="改进建议")
    response_time: float = Field(0.0, description="回答响应时间")
    sources_count: int = Field(0, description="引用来源数量")


class FeedbackAnalytics(BaseModel):
    """反馈分析模型"""
    total_feedbacks: int
    average_rating: float
    rating_distribution: Dict[int, int]
    satisfaction_rate: float
    common_issues: List[str]
    improvement_areas: List[str]
    top_queries: List[Dict[str, Any]]
    daily_stats: List[Dict[str, Any]]


# ============================================================================
# 反馈收集系统核心类
# ============================================================================

class FeedbackCollectionSystem:
    """前端反馈收集系统"""

    def __init__(self):
        self.feedbacks: List[FeedbackData] = []
        self.websocket_connections: List[WebSocket] = []
        self.analytics_cache: Optional[FeedbackAnalytics] = None
        self.cache_timestamp: Optional[datetime] = None

    async def submit_feedback(self, request: FeedbackRequest) -> Dict[str, Any]:
        """提交用户反馈"""
        try:
            # 创建反馈数据
            feedback = FeedbackData(
                id=f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.feedbacks)}",
                timestamp=datetime.now(),
                session_id=request.session_id,
                user_id=request.user_id,
                query=request.query,
                answer=request.answer,
                user_rating=request.user_rating,
                user_feedback=request.user_feedback,
                improvement_suggestions=request.improvement_suggestions,
                response_time=request.response_time,
                sources_count=request.sources_count
            )

            # 存储反馈
            self.feedbacks.append(feedback)

            # 清除分析缓存
            self.analytics_cache = None
            self.cache_timestamp = None

            logger.info(f"收到反馈: 评分={request.user_rating}, 用户={request.user_id}")

            # 实时推送到WebSocket连接
            await self._broadcast_feedback(feedback)

            return {
                "status": "success",
                "message": "感谢您的反馈！",
                "feedback_id": feedback.id,
                "timestamp": feedback.timestamp.isoformat()
            }

        except Exception as e:
            logger.error(f"提交反馈失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"提交反馈失败: {str(e)}")

    async def get_analytics(self, force_refresh: bool = False) -> FeedbackAnalytics:
        """获取反馈分析"""
        # 检查缓存
        if (not force_refresh and
            self.analytics_cache and
            self.cache_timestamp and
            (datetime.now() - self.cache_timestamp).seconds < 60):
            return self.analytics_cache

        try:
            if not self.feedbacks:
                return FeedbackAnalytics(
                    total_feedbacks=0,
                    average_rating=0.0,
                    rating_distribution={},
                    satisfaction_rate=0.0,
                    common_issues=[],
                    improvement_areas=[],
                    top_queries=[],
                    daily_stats=[]
                )

            # 计算基本统计
            total_feedbacks = len(self.feedbacks)
            ratings = [f.user_rating for f in self.feedbacks]
            average_rating = statistics.mean(ratings)

            # 评分分布
            rating_distribution = {}
            for rating in range(1, 6):
                rating_distribution[rating] = ratings.count(rating)

            # 满意率 (4-5星为满意)
            satisfaction_rate = sum(1 for r in ratings if r >= 4) / len(ratings) * 100

            # 常见问题分析
            common_issues = self._analyze_common_issues()

            # 改进领域
            improvement_areas = self._analyze_improvement_areas()

            # 热门查询
            top_queries = self._get_top_queries()

            # 每日统计
            daily_stats = self._get_daily_stats()

            analytics = FeedbackAnalytics(
                total_feedbacks=total_feedbacks,
                average_rating=round(average_rating, 2),
                rating_distribution=rating_distribution,
                satisfaction_rate=round(satisfaction_rate, 2),
                common_issues=common_issues,
                improvement_areas=improvement_areas,
                top_queries=top_queries,
                daily_stats=daily_stats
            )

            # 更新缓存
            self.analytics_cache = analytics
            self.cache_timestamp = datetime.now()

            return analytics

        except Exception as e:
            logger.error(f"获取分析数据失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"获取分析数据失败: {str(e)}")

    def _analyze_common_issues(self) -> List[str]:
        """分析常见问题"""
        issues = []

        # 收集低评分反馈
        low_rating_feedbacks = [f for f in self.feedbacks if f.user_rating <= 2]

        if low_rating_feedbacks:
            # 提取关键词
            issue_keywords = {}
            for feedback in low_rating_feedbacks:
                if feedback.user_feedback:
                    words = feedback.user_feedback.lower().split()
                    for word in words:
                        if len(word) > 3:  # 过滤短词
                            issue_keywords[word] = issue_keywords.get(word, 0) + 1

            # 获取最常见的关键词
            top_issues = sorted(issue_keywords.items(), key=lambda x: x[1], reverse=True)[:5]
            issues = [issue for issue, count in top_issues]

        return issues

    def _analyze_improvement_areas(self) -> List[str]:
        """分析改进领域"""
        improvement_suggestions = []

        # 收集改进建议
        suggestions = [f.improvement_suggestions for f in self.feedbacks
                      if f.improvement_suggestions and f.user_rating <= 3]

        if suggestions:
            # 简单的关键词提取
            area_keywords = {}
            for suggestion in suggestions:
                words = suggestion.lower().split()
                for word in words:
                    if len(word) > 3:
                        area_keywords[word] = area_keywords.get(word, 0) + 1

            # 获取最常见的改进领域
            top_areas = sorted(area_keywords.items(), key=lambda x: x[1], reverse=True)[:5]
            improvement_suggestions = [area for area, count in top_areas]

        return improvement_suggestions

    def _get_top_queries(self) -> List[Dict[str, Any]]:
        """获取热门查询"""
        query_counts = {}
        query_ratings = {}

        for feedback in self.feedbacks:
            query = feedback.query
            query_counts[query] = query_counts.get(query, 0) + 1
            if query not in query_ratings:
                query_ratings[query] = []
            query_ratings[query].append(feedback.user_rating)

        # 按查询次数排序
        top_queries = []
        for query, count in sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            avg_rating = statistics.mean(query_ratings[query])
            top_queries.append({
                "query": query,
                "count": count,
                "average_rating": round(avg_rating, 2)
            })

        return top_queries

    def _get_daily_stats(self) -> List[Dict[str, Any]]:
        """获取每日统计"""
        daily_stats = {}

        for feedback in self.feedbacks:
            date = feedback.timestamp.date()
            if date not in daily_stats:
                daily_stats[date] = {
                    "date": date.isoformat(),
                    "count": 0,
                    "ratings": []
                }

            daily_stats[date]["count"] += 1
            daily_stats[date]["ratings"].append(feedback.user_rating)

        # 计算每日平均评分
        result = []
        for date, stats in sorted(daily_stats.items(), reverse=True)[:30]:  # 最近30天
            avg_rating = statistics.mean(stats["ratings"]) if stats["ratings"] else 0
            result.append({
                "date": stats["date"],
                "count": stats["count"],
                "average_rating": round(avg_rating, 2)
            })

        return result

    async def get_feedback_by_user(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取用户反馈历史"""
        user_feedbacks = [f for f in self.feedbacks if f.user_id == user_id]
        user_feedbacks.sort(key=lambda x: x.timestamp, reverse=True)

        return [
            {
                "id": f.id,
                "timestamp": f.timestamp.isoformat(),
                "query": f.query,
                "rating": f.user_rating,
                "feedback": f.user_feedback
            }
            for f in user_feedbacks[:limit]
        ]

    async def export_feedback_data(self, format: str = "json") -> Dict[str, Any]:
        """导出反馈数据"""
        try:
            if format.lower() == "json":
                return {
                    "export_time": datetime.now().isoformat(),
                    "total_feedbacks": len(self.feedbacks),
                    "feedbacks": [
                        {
                            "id": f.id,
                            "timestamp": f.timestamp.isoformat(),
                            "session_id": f.session_id,
                            "user_id": f.user_id,
                            "query": f.query,
                            "answer": f.answer,
                            "rating": f.user_rating,
                            "feedback": f.user_feedback,
                            "suggestions": f.improvement_suggestions,
                            "response_time": f.response_time,
                            "sources_count": f.sources_count
                        }
                        for f in self.feedbacks
                    ]
                }
            else:
                raise ValueError(f"不支持的导出格式: {format}")

        except Exception as e:
            logger.error(f"导出数据失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"导出数据失败: {str(e)}")

    # WebSocket相关方法
    async def connect_websocket(self, websocket: WebSocket):
        """建立WebSocket连接"""
        await websocket.accept()
        self.websocket_connections.append(websocket)
        logger.info(f"WebSocket连接已建立，当前连接数: {len(self.websocket_connections)}")

    def disconnect_websocket(self, websocket: WebSocket):
        """断开WebSocket连接"""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)
            logger.info(f"WebSocket连接已断开，当前连接数: {len(self.websocket_connections)}")

    async def _broadcast_feedback(self, feedback: FeedbackData):
        """广播新反馈到所有WebSocket连接"""
        if not self.websocket_connections:
            return

        message = {
            "type": "new_feedback",
            "data": {
                "id": feedback.id,
                "timestamp": feedback.timestamp.isoformat(),
                "rating": feedback.user_rating,
                "query": feedback.query[:100] + "..." if len(feedback.query) > 100 else feedback.query
            }
        }

        disconnected_connections = []
        for connection in self.websocket_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"发送WebSocket消息失败: {str(e)}")
                disconnected_connections.append(connection)

        # 清理断开的连接
        for connection in disconnected_connections:
            self.disconnect_websocket(connection)


# ============================================================================
# FastAPI应用
# ============================================================================

# 全局反馈系统实例
feedback_system = FeedbackCollectionSystem()

# 创建FastAPI应用
app = FastAPI(
    title="前端反馈收集系统",
    description="基于DAML-RAG框架的用户反馈收集和分析系统",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 前端HTML页面
# ============================================================================

FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DAML-RAG 反馈收集系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }

        .content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 30px;
        }

        .feedback-form {
            background: #f8fafc;
            padding: 25px;
            border-radius: 15px;
            border: 1px solid #e2e8f0;
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #374151;
        }

        .form-group input,
        .form-group textarea,
        .form-group select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e5e7eb;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }

        .form-group input:focus,
        .form-group textarea:focus,
        .form-group select:focus {
            outline: none;
            border-color: #4f46e5;
        }

        .rating-input {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .star {
            font-size: 24px;
            color: #d1d5db;
            cursor: pointer;
            transition: color 0.2s;
        }

        .star.active {
            color: #fbbf24;
        }

        .star:hover {
            color: #fbbf24;
        }

        .submit-btn {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            width: 100%;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
        }

        .submit-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .analytics {
            background: #f8fafc;
            padding: 25px;
            border-radius: 15px;
            border: 1px solid #e2e8f0;
        }

        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }

        .stat-card h3 {
            color: #6b7280;
            font-size: 0.9em;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #1f2937;
        }

        .success-message {
            background: #10b981;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: none;
        }

        .error-message {
            background: #ef4444;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: none;
        }

        .live-feedback {
            background: #fef3c7;
            border: 1px solid #f59e0b;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
        }

        .live-feedback h4 {
            color: #92400e;
            margin-bottom: 10px;
        }

        .feedback-item {
            background: white;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 8px;
            font-size: 0.9em;
        }

        @media (max-width: 768px) {
            .content {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 DAML-RAG 反馈收集系统</h1>
            <p>基于三层检索系统的智能问答反馈收集和分析</p>
        </div>

        <div class="content">
            <div class="feedback-section">
                <div class="feedback-form">
                    <h2>💬 提交反馈</h2>

                    <div class="success-message" id="successMessage">
                        ✅ 反馈提交成功！感谢您的评价！
                    </div>

                    <div class="error-message" id="errorMessage">
                        ❌ 提交失败，请稍后重试。
                    </div>

                    <form id="feedbackForm">
                        <div class="form-group">
                            <label for="userId">用户ID</label>
                            <input type="text" id="userId" name="userId" required placeholder="请输入用户ID">
                        </div>

                        <div class="form-group">
                            <label for="sessionId">会话ID</label>
                            <input type="text" id="sessionId" name="sessionId" required placeholder="请输入会话ID">
                        </div>

                        <div class="form-group">
                            <label for="query">查询内容</label>
                            <textarea id="query" name="query" rows="3" required placeholder="请输入您的查询内容"></textarea>
                        </div>

                        <div class="form-group">
                            <label for="answer">AI回答</label>
                            <textarea id="answer" name="answer" rows="4" required placeholder="请输入AI的回答内容"></textarea>
                        </div>

                        <div class="form-group">
                            <label>评分</label>
                            <div class="rating-input">
                                <span class="star" data-rating="1">⭐</span>
                                <span class="star" data-rating="2">⭐</span>
                                <span class="star" data-rating="3">⭐</span>
                                <span class="star" data-rating="4">⭐</span>
                                <span class="star" data-rating="5">⭐</span>
                                <span id="ratingText">请选择评分</span>
                            </div>
                            <input type="hidden" id="rating" name="rating" required>
                        </div>

                        <div class="form-group">
                            <label for="userFeedback">用户反馈</label>
                            <textarea id="userFeedback" name="userFeedback" rows="3" placeholder="请描述您的使用体验（可选）"></textarea>
                        </div>

                        <div class="form-group">
                            <label for="suggestions">改进建议</label>
                            <textarea id="suggestions" name="suggestions" rows="3" placeholder="请提供改进建议（可选）"></textarea>
                        </div>

                        <button type="submit" class="submit-btn">提交反馈</button>
                    </form>
                </div>

                <div class="live-feedback">
                    <h4>🔴 实时反馈</h4>
                    <div id="liveFeedbackList">
                        <p>等待实时反馈...</p>
                    </div>
                </div>
            </div>

            <div class="analytics-section">
                <div class="analytics">
                    <h2>📊 实时分析</h2>

                    <div class="stat-card">
                        <h3>总反馈数</h3>
                        <div class="stat-value" id="totalFeedbacks">0</div>
                    </div>

                    <div class="stat-card">
                        <h3>平均评分</h3>
                        <div class="stat-value" id="averageRating">0.0</div>
                    </div>

                    <div class="stat-card">
                        <h3>满意率</h3>
                        <div class="stat-value" id="satisfactionRate">0%</div>
                    </div>

                    <div class="stat-card">
                        <h3>今日反馈</h3>
                        <div class="stat-value" id="todayFeedbacks">0</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 评分交互
        const stars = document.querySelectorAll('.star');
        const ratingInput = document.getElementById('rating');
        const ratingText = document.getElementById('ratingText');
        let currentRating = 0;

        stars.forEach(star => {
            star.addEventListener('click', () => {
                currentRating = parseInt(star.dataset.rating);
                ratingInput.value = currentRating;
                updateStars();
            });

            star.addEventListener('mouseenter', () => {
                const hoverRating = parseInt(star.dataset.rating);
                highlightStars(hoverRating);
            });
        });

        document.querySelector('.rating-input').addEventListener('mouseleave', () => {
            updateStars();
        });

        function updateStars() {
            highlightStars(currentRating);
            if (currentRating > 0) {
                const ratingTexts = ['', '不满意', '一般', '满意', '很满意', '非常满意'];
                ratingText.textContent = ratingTexts[currentRating];
            } else {
                ratingText.textContent = '请选择评分';
            }
        }

        function highlightStars(rating) {
            stars.forEach((star, index) => {
                if (index < rating) {
                    star.classList.add('active');
                } else {
                    star.classList.remove('active');
                }
            });
        }

        // 表单提交
        const feedbackForm = document.getElementById('feedbackForm');
        const successMessage = document.getElementById('successMessage');
        const errorMessage = document.getElementById('errorMessage');

        feedbackForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            if (currentRating === 0) {
                alert('请选择评分');
                return;
            }

            const formData = new FormData(feedbackForm);
            const data = {
                session_id: formData.get('sessionId'),
                user_id: formData.get('userId'),
                query: formData.get('query'),
                answer: formData.get('answer'),
                user_rating: currentRating,
                user_feedback: formData.get('userFeedback') || null,
                improvement_suggestions: formData.get('suggestions') || null,
                response_time: 0.0,
                sources_count: 0
            };

            try {
                const response = await fetch('/feedback', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (response.ok) {
                    successMessage.style.display = 'block';
                    errorMessage.style.display = 'none';
                    feedbackForm.reset();
                    currentRating = 0;
                    updateStars();

                    setTimeout(() => {
                        successMessage.style.display = 'none';
                    }, 5000);

                    // 刷新分析数据
                    loadAnalytics();
                } else {
                    throw new Error(result.detail || '提交失败');
                }
            } catch (error) {
                console.error('提交失败:', error);
                errorMessage.textContent = `❌ ${error.message}`;
                errorMessage.style.display = 'block';

                setTimeout(() => {
                    errorMessage.style.display = 'none';
                }, 5000);
            }
        });

        // 加载分析数据
        async function loadAnalytics() {
            try {
                const response = await fetch('/analytics');
                const analytics = await response.json();

                document.getElementById('totalFeedbacks').textContent = analytics.total_feedbacks;
                document.getElementById('averageRating').textContent = analytics.average_rating;
                document.getElementById('satisfactionRate').textContent = analytics.satisfaction_rate + '%';

                // 计算今日反馈数
                const today = new Date().toISOString().split('T')[0];
                const todayFeedbacks = analytics.daily_stats.find(stat => stat.date === today);
                document.getElementById('todayFeedbacks').textContent = todayFeedbacks ? todayFeedbacks.count : 0;
            } catch (error) {
                console.error('加载分析数据失败:', error);
            }
        }

        // WebSocket连接
        const ws = new WebSocket(`ws://${window.location.host}/ws/feedback`);
        const liveFeedbackList = document.getElementById('liveFeedbackList');

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'new_feedback') {
                const feedbackItem = document.createElement('div');
                feedbackItem.className = 'feedback-item';
                feedbackItem.innerHTML = `
                    <strong>评分: ${'⭐'.repeat(data.data.rating)}</strong><br>
                    <em>${data.data.query}</em><br>
                    <small>${new Date(data.data.timestamp).toLocaleTimeString()}</small>
                `;

                liveFeedbackList.insertBefore(feedbackItem, liveFeedbackList.firstChild);

                // 保持最多10条记录
                while (liveFeedbackList.children.length > 10) {
                    liveFeedbackList.removeChild(liveFeedbackList.lastChild);
                }
            }
        };

        ws.onopen = () => {
            console.log('WebSocket连接已建立');
            liveFeedbackList.innerHTML = '<p>🟢 实时反馈连接已建立</p>';
        };

        ws.onerror = (error) => {
            console.error('WebSocket错误:', error);
            liveFeedbackList.innerHTML = '<p>🔴 实时反馈连接失败</p>';
        };

        // 页面加载时获取初始数据
        loadAnalytics();
    </script>
</body>
</html>
"""


# ============================================================================
# API路由
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """获取前端页面"""
    return FRONTEND_HTML


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """提交用户反馈"""
    return await feedback_system.submit_feedback(request)


@app.get("/analytics")
async def get_analytics():
    """获取反馈分析"""
    return await feedback_system.get_analytics()


@app.get("/analytics/user/{user_id}")
async def get_user_feedbacks(user_id: str, limit: int = 50):
    """获取用户反馈历史"""
    return await feedback_system.get_feedback_by_user(user_id, limit)


@app.get("/export/{format}")
async def export_feedbacks(format: str):
    """导出反馈数据"""
    return await feedback_system.export_feedback_data(format)


@app.websocket("/ws/feedback")
async def websocket_feedback(websocket: WebSocket):
    """WebSocket实时反馈推送"""
    await feedback_system.connect_websocket(websocket)
    try:
        while True:
            # 保持连接活跃
            await websocket.receive_text()
    except WebSocketDisconnect:
        feedback_system.disconnect_websocket(websocket)


# ============================================================================
# 主函数
# ============================================================================

async def main():
    """主函数"""
    print("🚀 启动前端反馈收集系统...")
    print("📋 功能特性:")
    print("   • 实时反馈收集和分析")
    print("   • WebSocket实时推送")
    print("   • 用户满意度统计")
    print("   • 前端可视化界面")
    print("   • 反馈数据导出")
    print()
    print("🌐 访问地址: http://localhost:8003")
    print("📊 分析API: http://localhost:8003/analytics")
    print()

    # 启动FastAPI服务器
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())