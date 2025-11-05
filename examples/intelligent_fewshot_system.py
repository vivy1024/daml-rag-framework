#!/usr/bin/env python3
"""
智能FewShot存储和筛选系统
基于用户评分和反馈的智能示例管理系统

这个示例展示了：
1. 基于用户评分的fewshot质量评估
2. 智能的fewshot筛选和排序
3. 异常分数的检测和处理
4. 人工筛选界面和流程
5. 自适应的fewshot优化

作者：BUILD_BODY Team
版本：v1.0.0
日期：2025-11-05
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import statistics
import hashlib

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 数据模型和枚举
# ============================================================================

class FewShotQuality(str, Enum):
    """FewShot质量等级"""
    EXCELLENT = "excellent"      # 优秀 (4.5-5.0分)
    GOOD = "good"               # 良好 (3.5-4.5分)
    FAIR = "fair"               # 一般 (2.5-3.5分)
    POOR = "poor"               # 较差 (1.5-2.5分)
    VERY_POOR = "very_poor"     # 很差 (1.0-1.5分)
    SUSPICIOUS = "suspicious"   # 可疑 (异常分数)


@dataclass
class FewShotExample:
    """FewShot示例数据"""
    id: str
    query: str
    answer: str
    context: Dict[str, Any]
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    # 反馈相关
    user_ratings: List[float]
    average_rating: float
    total_feedbacks: int
    quality_level: FewShotQuality

    # 系统评估
    relevance_score: float
    completeness_score: float
    clarity_score: float
    overall_quality_score: float

    # 时间信息
    created_at: datetime
    last_updated: datetime
    usage_count: int

    # 筛选状态
    is_suspicious: bool
    suspicion_reason: Optional[str]
    needs_manual_review: bool
    manual_review_status: str  # "pending", "approved", "rejected"


class FewShotRequest(BaseModel):
    """FewShot请求模型"""
    query: str = Field(..., description="查询内容")
    domain: str = Field("general", description="领域")
    max_examples: int = Field(5, description="最大示例数")
    quality_threshold: float = Field(3.0, description="质量阈值")
    exclude_suspicious: bool = Field(True, description="排除可疑示例")


class FeedbackRequest(BaseModel):
    """反馈请求模型"""
    example_id: str = Field(..., description="示例ID")
    user_id: str = Field(..., description="用户ID")
    rating: float = Field(..., ge=1.0, le=5.0, description="评分")
    feedback: Optional[str] = Field(None, description="反馈内容")
    context: Optional[Dict[str, Any]] = Field(None, description="上下文")


class ManualReviewRequest(BaseModel):
    """人工审核请求模型"""
    example_id: str = Field(..., description="示例ID")
    reviewer_id: str = Field(..., description="审核员ID")
    action: str = Field(..., description="操作: approve/reject/flag")
    notes: Optional[str] = Field(None, description="审核备注")


# ============================================================================
# 智能FewShot管理系统
# ============================================================================

class IntelligentFewShotSystem:
    """智能FewShot存储和筛选系统"""

    def __init__(self):
        self.examples: Dict[str, FewShotExample] = {}
        self.feedback_history: List[Dict[str, Any]] = []
        self.quality_thresholds = {
            "excellent": 4.5,
            "good": 3.5,
            "fair": 2.5,
            "poor": 1.5,
            "suspicious": 1.0
        }
        self.anomaly_detection_enabled = True
        self.auto_approval_threshold = 4.0

    async def add_example(self, query: str, answer: str, context: Dict[str, Any],
                         sources: List[Dict[str, Any]]) -> str:
        """添加新的FewShot示例"""
        try:
            # 生成唯一ID
            example_id = self._generate_example_id(query, answer)

            # 计算初始质量分数
            relevance_score = self._calculate_relevance_score(query, answer)
            completeness_score = self._calculate_completeness_score(answer, sources)
            clarity_score = self._calculate_clarity_score(answer)
            overall_quality_score = (relevance_score + completeness_score + clarity_score) / 3

            # 创建示例
            example = FewShotExample(
                id=example_id,
                query=query,
                answer=answer,
                context=context,
                sources=sources,
                metadata={
                    "domain": context.get("domain", "general"),
                    "model_used": context.get("model", "unknown"),
                    "retrieval_method": context.get("retrieval_method", "unknown"),
                    "response_time": context.get("response_time", 0.0)
                },
                user_ratings=[],
                average_rating=0.0,
                total_feedbacks=0,
                quality_level=FewShotQuality.FAIR,  # 初始状态
                relevance_score=relevance_score,
                completeness_score=completeness_score,
                clarity_score=clarity_score,
                overall_quality_score=overall_quality_score,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                usage_count=0,
                is_suspicious=False,
                suspicion_reason=None,
                needs_manual_review=False,
                manual_review_status="pending"
            )

            self.examples[example_id] = example
            logger.info(f"添加FewShot示例: {example_id}")

            return example_id

        except Exception as e:
            logger.error(f"添加FewShot示例失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"添加示例失败: {str(e)}")

    async def submit_feedback(self, request: FeedbackRequest) -> Dict[str, Any]:
        """提交用户反馈"""
        try:
            if request.example_id not in self.examples:
                raise HTTPException(status_code=404, detail="示例不存在")

            example = self.examples[request.example_id]

            # 检测异常评分
            is_anomalous = await self._detect_rating_anomaly(request.rating, example)

            if is_anomalous:
                example.is_suspicious = True
                example.suspicion_reason = f"异常评分: {request.rating}"
                example.needs_manual_review = True
                logger.warning(f"检测到异常评分: {request.example_id}, 评分: {request.rating}")

            # 更新反馈
            example.user_ratings.append(request.rating)
            example.average_rating = statistics.mean(example.user_ratings)
            example.total_feedbacks += 1
            example.last_updated = datetime.now()

            # 更新质量等级
            example.quality_level = self._determine_quality_level(example.average_rating)

            # 检查是否需要人工审核
            if example.average_rating < 2.0 or is_anomalous:
                example.needs_manual_review = True

            # 记录反馈历史
            feedback_record = {
                "example_id": request.example_id,
                "user_id": request.user_id,
                "rating": request.rating,
                "feedback": request.feedback,
                "context": request.context,
                "timestamp": datetime.now().isoformat(),
                "is_anomalous": is_anomalous
            }
            self.feedback_history.append(feedback_record)

            logger.info(f"收到反馈: {request.example_id}, 评分: {request.rating}")

            return {
                "status": "success",
                "message": "反馈已提交",
                "example_id": request.example_id,
                "average_rating": example.average_rating,
                "quality_level": example.quality_level.value,
                "is_anomalous": is_anomalous
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"提交反馈失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"提交反馈失败: {str(e)}")

    async def retrieve_examples(self, request: FewShotRequest) -> List[Dict[str, Any]]:
        """检索相关的FewShot示例"""
        try:
            # 计算查询向量（简化版本，实际应使用embedding）
            query_embedding = self._get_text_embedding(request.query)

            # 评分和筛选所有示例
            scored_examples = []
            for example in self.examples.values():
                # 跳过可疑示例（如果要求）
                if request.exclude_suspicious and example.is_suspicious:
                    continue

                # 质量筛选
                if example.average_rating < request.quality_threshold:
                    continue

                # 计算相关性分数
                example_embedding = self._get_text_embedding(example.query)
                similarity = self._calculate_cosine_similarity(query_embedding, example_embedding)

                # 综合评分
                final_score = (
                    similarity * 0.4 +  # 相关性
                    example.average_rating / 5.0 * 0.4 +  # 用户评分
                    example.overall_quality_score * 0.2  # 系统质量
                )

                scored_examples.append((example, final_score))

            # 排序并返回前N个
            scored_examples.sort(key=lambda x: x[1], reverse=True)
            top_examples = scored_examples[:request.max_examples]

            # 更新使用计数
            for example, _ in top_examples:
                example.usage_count += 1

            # 格式化返回
            result = []
            for example, score in top_examples:
                result.append({
                    "id": example.id,
                    "query": example.query,
                    "answer": example.answer,
                    "score": round(score, 3),
                    "average_rating": example.average_rating,
                    "quality_level": example.quality_level.value,
                    "usage_count": example.usage_count,
                    "is_suspicious": example.is_suspicious,
                    "domain": example.metadata.get("domain", "general")
                })

            return result

        except Exception as e:
            logger.error(f"检索FewShot示例失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")

    async def get_suspicious_examples(self) -> List[Dict[str, Any]]:
        """获取需要人工审核的可疑示例"""
        try:
            suspicious_examples = [
                example for example in self.examples.values()
                if example.is_suspicious or example.needs_manual_review
            ]

            result = []
            for example in suspicious_examples:
                result.append({
                    "id": example.id,
                    "query": example.query,
                    "answer": example.answer[:200] + "..." if len(example.answer) > 200 else example.answer,
                    "average_rating": example.average_rating,
                    "total_feedbacks": example.total_feedbacks,
                    "quality_level": example.quality_level.value,
                    "is_suspicious": example.is_suspicious,
                    "suspicion_reason": example.suspicion_reason,
                    "needs_manual_review": example.needs_manual_review,
                    "manual_review_status": example.manual_review_status,
                    "created_at": example.created_at.isoformat()
                })

            # 按可疑程度排序
            result.sort(key=lambda x: (
                not x["is_suspicious"],  # 可疑的优先
                x["average_rating"],  # 低评分优先
                x["total_feedbacks"]  # 反馈少的优先
            ))

            return result

        except Exception as e:
            logger.error(f"获取可疑示例失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"获取可疑示例失败: {str(e)}")

    async def submit_manual_review(self, request: ManualReviewRequest) -> Dict[str, Any]:
        """提交人工审核结果"""
        try:
            if request.example_id not in self.examples:
                raise HTTPException(status_code=404, detail="示例不存在")

            example = self.examples[request.example_id]

            # 更新审核状态
            example.manual_review_status = request.action
            example.needs_manual_review = False

            if request.action == "approve":
                example.is_suspicious = False
                example.suspicion_reason = None
                logger.info(f"示例 {request.example_id} 已通过人工审核")
            elif request.action == "reject":
                # 可以考虑删除或标记为拒绝
                logger.info(f"示例 {request.example_id} 已被人工拒绝")
            elif request.action == "flag":
                example.needs_manual_review = True
                example.is_suspicious = True
                example.suspicion_reason = request.notes or "人工标记"
                logger.info(f"示例 {request.example_id} 已被人工标记")

            return {
                "status": "success",
                "message": f"人工审核结果已提交: {request.action}",
                "example_id": request.example_id
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"提交人工审核失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"提交审核失败: {str(e)}")

    async def get_quality_statistics(self) -> Dict[str, Any]:
        """获取质量统计信息"""
        try:
            if not self.examples:
                return {
                    "total_examples": 0,
                    "quality_distribution": {},
                    "average_rating": 0.0,
                    "suspicious_count": 0,
                    "pending_reviews": 0
                }

            total_examples = len(self.examples)
            quality_distribution = {}
            total_rating = 0
            total_feedbacks = 0
            suspicious_count = 0
            pending_reviews = 0

            for example in self.examples.values():
                # 质量分布
                quality = example.quality_level.value
                quality_distribution[quality] = quality_distribution.get(quality, 0) + 1

                # 评分统计
                if example.total_feedbacks > 0:
                    total_rating += example.average_rating * example.total_feedbacks
                    total_feedbacks += example.total_feedbacks

                # 可疑和待审核
                if example.is_suspicious:
                    suspicious_count += 1
                if example.needs_manual_review:
                    pending_reviews += 1

            average_rating = total_rating / total_feedbacks if total_feedbacks > 0 else 0.0

            return {
                "total_examples": total_examples,
                "quality_distribution": quality_distribution,
                "average_rating": round(average_rating, 2),
                "suspicious_count": suspicious_count,
                "pending_reviews": pending_reviews,
                "total_feedbacks": total_feedbacks
            }

        except Exception as e:
            logger.error(f"获取质量统计失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")

    # ========================================================================
    # 私有方法
    # ========================================================================

    def _generate_example_id(self, query: str, answer: str) -> str:
        """生成示例ID"""
        content = f"{query}_{answer}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _calculate_relevance_score(self, query: str, answer: str) -> float:
        """计算相关性分数"""
        # 简化版本：基于关键词重叠
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        if not query_words:
            return 0.5

        overlap = len(query_words & answer_words)
        return min(1.0, overlap / len(query_words))

    def _calculate_completeness_score(self, answer: str, sources: List[Dict[str, Any]]) -> float:
        """计算完整性分数"""
        # 基于回答长度和来源数量
        length_score = min(1.0, len(answer) / 200)  # 200字符为满分
        source_score = min(1.0, len(sources) / 3)     # 3个来源为满分

        return (length_score + source_score) / 2

    def _calculate_clarity_score(self, answer: str) -> float:
        """计算清晰度分数"""
        # 简化版本：基于句子结构和长度
        sentences = answer.split('.')
        if len(sentences) < 2:
            return 0.5

        avg_sentence_length = sum(len(s.strip()) for s in sentences) / len(sentences)

        # 理想的句子长度在10-50字符之间
        if 10 <= avg_sentence_length <= 50:
            return 1.0
        elif avg_sentence_length < 10:
            return 0.6
        else:
            return max(0.3, 1.0 - (avg_sentence_length - 50) / 100)

    def _determine_quality_level(self, average_rating: float) -> FewShotQuality:
        """确定质量等级"""
        if average_rating >= self.quality_thresholds["excellent"]:
            return FewShotQuality.EXCELLENT
        elif average_rating >= self.quality_thresholds["good"]:
            return FewShotQuality.GOOD
        elif average_rating >= self.quality_thresholds["fair"]:
            return FewShotQuality.FAIR
        elif average_rating >= self.quality_thresholds["poor"]:
            return FewShotQuality.POOR
        elif average_rating >= self.quality_thresholds["suspicious"]:
            return FewShotQuality.VERY_POOR
        else:
            return FewShotQuality.SUSPICIOUS

    async def _detect_rating_anomaly(self, rating: float, example: FewShotExample) -> bool:
        """检测评分异常"""
        if not self.anomaly_detection_enabled:
            return False

        if example.total_feedbacks < 3:
            # 反馈太少，难以判断
            return False

        # 使用Z-score检测异常
        ratings = example.user_ratings
        mean_rating = statistics.mean(ratings)
        std_rating = statistics.stdev(ratings) if len(ratings) > 1 else 0

        if std_rating == 0:
            # 所有评分相同，检查是否与当前评分差异过大
            return abs(rating - mean_rating) > 2.0

        # 计算Z-score
        z_score = abs(rating - mean_rating) / std_rating

        # Z-score > 2.0 认为是异常
        return z_score > 2.0

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """获取文本嵌入（简化版本）"""
        # 实际实现应使用embedding模型
        # 这里使用简单的hash向量作为示例
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()

        # 转换为固定长度的向量
        embedding = np.array([int(hash_hex[i:i+2], 16) / 255.0 for i in range(0, min(32, len(hash_hex)), 2)])

        # 填充到固定长度
        if len(embedding) < 16:
            embedding = np.pad(embedding, (0, 16 - len(embedding)))

        return embedding

    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


# ============================================================================
# FastAPI应用
# ============================================================================

# 全局FewShot系统实例
fewshot_system = IntelligentFewShotSystem()

# 创建FastAPI应用
app = FastAPI(
    title="智能FewShot管理系统",
    description="基于用户评分和反馈的智能示例管理系统",
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
# 人工审核前端页面
# ============================================================================

REVIEW_FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能FewShot人工审核系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }

        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }

        .stat-label {
            color: #666;
            font-size: 0.9em;
        }

        .review-section {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .review-item {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }

        .review-item.suspicious {
            border-color: #ff6b6b;
            background: #fff5f5;
        }

        .review-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .quality-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }

        .quality-excellent { background: #d4edda; color: #155724; }
        .quality-good { background: #d1ecf1; color: #0c5460; }
        .quality-fair { background: #fff3cd; color: #856404; }
        .quality-poor { background: #f8d7da; color: #721c24; }
        .quality-suspicious { background: #f8d7da; color: #721c24; }

        .review-content {
            margin-bottom: 15px;
        }

        .review-query {
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
        }

        .review-answer {
            color: #666;
            line-height: 1.5;
            margin-bottom: 10px;
        }

        .review-meta {
            font-size: 0.8em;
            color: #999;
            margin-bottom: 15px;
        }

        .review-actions {
            display: flex;
            gap: 10px;
        }

        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9em;
            transition: background-color 0.2s;
        }

        .btn-approve { background: #28a745; color: white; }
        .btn-reject { background: #dc3545; color: white; }
        .btn-flag { background: #ffc107; color: #212529; }
        .btn-refresh { background: #17a2b8; color: white; }

        .btn:hover { opacity: 0.8; }

        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }

        .no-items {
            text-align: center;
            padding: 40px;
            color: #999;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 智能FewShot人工审核系统</h1>
        <p>审核可疑的FewShot示例，确保内容质量</p>
    </div>

    <div class="container">
        <div class="stats-grid" id="statsGrid">
            <!-- 统计卡片将在这里动态生成 -->
        </div>

        <div class="review-section">
            <h2>待审核示例</h2>
            <div id="reviewList" class="loading">
                加载中...
            </div>
        </div>
    </div>

    <script>
        let reviewData = [];

        // 加载统计数据
        async function loadStats() {
            try {
                const response = await fetch('/fewshot/stats');
                const stats = await response.json();

                const statsHtml = `
                    <div class="stat-card">
                        <div class="stat-value">${stats.total_examples}</div>
                        <div class="stat-label">总示例数</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${stats.average_rating}</div>
                        <div class="stat-label">平均评分</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${stats.suspicious_count}</div>
                        <div class="stat-label">可疑示例</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${stats.pending_reviews}</div>
                        <div class="stat-label">待审核</div>
                    </div>
                `;

                document.getElementById('statsGrid').innerHTML = statsHtml;
            } catch (error) {
                console.error('加载统计数据失败:', error);
            }
        }

        // 加载待审核示例
        async function loadSuspiciousExamples() {
            try {
                const response = await fetch('/fewshot/suspicious');
                const examples = await response.json();
                reviewData = examples;
                renderReviewList();
            } catch (error) {
                console.error('加载示例失败:', error);
                document.getElementById('reviewList').innerHTML =
                    '<div class="no-items">加载失败，请稍后重试</div>';
            }
        }

        // 渲染审核列表
        function renderReviewList() {
            const listContainer = document.getElementById('reviewList');

            if (reviewData.length === 0) {
                listContainer.innerHTML = '<div class="no-items">没有待审核的示例 🎉</div>';
                return;
            }

            const html = reviewData.map(item => `
                <div class="review-item ${item.is_suspicious ? 'suspicious' : ''}">
                    <div class="review-header">
                        <span class="quality-badge quality-${item.quality_level}">
                            ${item.quality_level}
                        </span>
                        <span>评分: ${item.average_rating.toFixed(1)} (${item.total_feedbacks}条反馈)</span>
                    </div>

                    <div class="review-content">
                        <div class="review-query">${item.query}</div>
                        <div class="review-answer">${item.answer}</div>
                        <div class="review-meta">
                            ${item.suspicion_reason ? `可疑原因: ${item.suspicion_reason} | ` : ''}
                            创建时间: ${new Date(item.created_at).toLocaleString()}
                        </div>
                    </div>

                    <div class="review-actions">
                        <button class="btn btn-approve" onclick="submitReview('${item.id}', 'approve')">
                            ✅ 通过
                        </button>
                        <button class="btn btn-reject" onclick="submitReview('${item.id}', 'reject')">
                            ❌ 拒绝
                        </button>
                        <button class="btn btn-flag" onclick="submitReview('${item.id}', 'flag')">
                            🚩 标记
                        </button>
                    </div>
                </div>
            `).join('');

            listContainer.innerHTML = html;
        }

        // 提交审核结果
        async function submitReview(exampleId, action) {
            try {
                const response = await fetch('/fewshot/manual-review', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        example_id: exampleId,
                        reviewer_id: 'admin', // 实际应用中应该从登录状态获取
                        action: action,
                        notes: `管理员审核: ${action}`
                    })
                });

                if (response.ok) {
                    // 从列表中移除已审核的项目
                    reviewData = reviewData.filter(item => item.id !== exampleId);
                    renderReviewList();
                    loadStats(); // 刷新统计

                    // 显示成功消息
                    alert(`审核成功: ${action}`);
                } else {
                    throw new Error('审核失败');
                }
            } catch (error) {
                console.error('提交审核失败:', error);
                alert('审核失败，请稍后重试');
            }
        }

        // 初始化页面
        function init() {
            loadStats();
            loadSuspiciousExamples();
        }

        // 页面加载完成后初始化
        document.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>
"""


# ============================================================================
# API路由
# ============================================================================

@app.get("/review", response_class=HTMLResponse)
async def get_review_frontend():
    """获取人工审核前端页面"""
    return REVIEW_FRONTEND_HTML


@app.post("/fewshot/add")
async def add_fewshot_example(query: str, answer: str, context: str, sources: str):
    """添加FewShot示例"""
    try:
        import json
        context_data = json.loads(context) if context else {}
        sources_data = json.loads(sources) if sources else []

        example_id = await fewshot_system.add_example(query, answer, context_data, sources_data)

        return {
            "status": "success",
            "example_id": example_id,
            "message": "FewShot示例已添加"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fewshot/feedback")
async def submit_fewshot_feedback(request: FeedbackRequest):
    """提交FewShot反馈"""
    return await fewshot_system.submit_feedback(request)


@app.post("/fewshot/retrieve")
async def retrieve_fewshot_examples(request: FewShotRequest):
    """检索FewShot示例"""
    return await fewshot_system.retrieve_examples(request)


@app.get("/fewshot/suspicious")
async def get_suspicious_examples():
    """获取可疑示例"""
    return await fewshot_system.get_suspicious_examples()


@app.post("/fewshot/manual-review")
async def submit_manual_review(request: ManualReviewRequest):
    """提交人工审核"""
    return await fewshot_system.submit_manual_review(request)


@app.get("/fewshot/stats")
async def get_fewshot_statistics():
    """获取FewShot统计信息"""
    return await fewshot_system.get_quality_statistics()


# 示例数据初始化
async def initialize_sample_data():
    """初始化示例数据"""
    sample_examples = [
        {
            "query": "什么是深蹲的正确动作要领？",
            "answer": "深蹲的正确动作要领包括：1. 双脚与肩同宽站立；2. 背部挺直，核心收紧；3. 臀部向后坐下，如同坐在椅子上；4. 大腿与地面平行或更低；5. 膝盖不要超过脚尖；6. 脚跟贴地，重心在脚中。记住动作要缓慢控制，避免借力。",
            "context": {"domain": "fitness", "model": "gpt-4"},
            "sources": [{"title": "深蹲动作指南", "score": 0.95}]
        },
        {
            "query": "如何制定初学者的健身计划？",
            "answer": "初学者健身计划制定要点：1. 明确目标（增肌/减脂/塑形）；2. 每周3-4次训练，每次45-60分钟；3. 复合动作为主，如深蹲、卧推、硬拉；4. 循序渐进，避免过度训练；5. 保证充足休息和营养；6. 坚持记录训练日志。建议寻求专业指导确保动作标准。",
            "context": {"domain": "fitness", "model": "gpt-4"},
            "sources": [{"title": "初学者健身指南", "score": 0.92}]
        },
        {
            "query": "HIIT训练的优缺点是什么？",
            "answer": "HIIT（高强度间歇训练）的优点：1. 训练时间短，效率高；2. 燃脂效果好，后燃效应强；3. 提高心肺功能和代谢率；4. 可在家进行，器械要求低。缺点：1. 强度大，不适合初学者；2. 恢复需求高，需要充分休息；3. 受伤风险相对较高；4. 需要良好的体能基础。",
            "context": {"domain": "fitness", "model": "gpt-3.5"},
            "sources": [{"title": "HIIT训练研究", "score": 0.88}]
        }
    ]

    for i, example in enumerate(sample_examples):
        example_id = await fewshot_system.add_example(
            example["query"],
            example["answer"],
            example["context"],
            example["sources"]
        )

        # 添加一些模拟反馈
        if i == 0:
            # 正常反馈
            await fewshot_system.submit_feedback(FeedbackRequest(
                example_id=example_id,
                user_id=f"user_{i}_1",
                rating=5.0,
                feedback="回答很详细，动作要领解释清楚"
            ))
            await fewshot_system.submit_feedback(FeedbackRequest(
                example_id=example_id,
                user_id=f"user_{i}_2",
                rating=4.5,
                feedback="很实用的指导"
            ))
        elif i == 1:
            # 正常反馈
            await fewshot_system.submit_feedback(FeedbackRequest(
                example_id=example_id,
                user_id=f"user_{i}_1",
                rating=4.0,
                feedback="建议很全面"
            ))
            # 异常反馈
            await fewshot_system.submit_feedback(FeedbackRequest(
                example_id=example_id,
                user_id=f"user_{i}_2",
                rating=1.0,
                feedback="完全不实用"  # 这个评分可能触发异常检测
            ))
        else:
            # 低质量反馈
            await fewshot_system.submit_feedback(FeedbackRequest(
                example_id=example_id,
                user_id=f"user_{i}_1",
                rating=2.5,
                feedback="回答不够详细"
            ))


# ============================================================================
# 主函数
# ============================================================================

async def main():
    """主函数"""
    print("🚀 启动智能FewShot管理系统...")
    print("📋 功能特性:")
    print("   • 基于用户评分的质量评估")
    print("   • 异常评分检测和处理")
    print("   • 智能fewshot筛选和排序")
    print("   • 人工审核界面和流程")
    print("   • 实时质量统计和分析")
    print()

    # 初始化示例数据
    await initialize_sample_data()
    print("✅ 示例数据已初始化")

    print("🌐 访问地址:")
    print("   • 人工审核界面: http://localhost:8004/review")
    print("   • API文档: http://localhost:8004/docs")
    print()

    # 启动FastAPI服务器
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8004,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())