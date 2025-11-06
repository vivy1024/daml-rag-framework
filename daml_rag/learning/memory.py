#!/usr/bin/env python3
"""
daml-rag-framework 记忆管理器模块
实现经验存储、检索和反馈学习机制
"""

import asyncio
import json
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import uuid
import numpy as np

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """反馈类型枚举"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class Feedback:
    """反馈数据"""
    id: str
    experience_id: str
    user_id: str
    feedback_type: FeedbackType
    rating: float
    comment: str = ""
    metadata: Dict[str, Any] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetrievalResult:
    """检索结果"""
    query: str
    documents: List[Dict[str, Any]]
    scores: List[float]
    retrieval_method: str
    execution_time: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Experience:
    """经验数据"""
    id: str
    query: str
    response: str
    context: Dict[str, Any]
    retrieval_result: Optional[RetrievalResult] = None
    model_used: str = ""
    feedback_list: List[Feedback] = None
    quality_score: float = 0.0
    usage_count: int = 0
    success_rate: float = 0.0
    similarity_score: float = 0.0
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.feedback_list is None:
            self.feedback_list = []
        if self.context is None:
            self.context = {}

    def add_feedback(self, feedback: Feedback):
        """添加反馈"""
        self.feedback_list.append(feedback)
        self.updated_at = datetime.now()

    def update_usage(self):
        """更新使用次数"""
        self.usage_count += 1
        self.updated_at = datetime.now()


class MemoryManager(ABC):
    """记忆管理器抽象基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_experiences = config.get('max_experiences', 10000)
        self.experience_ttl = config.get('experience_ttl_days', 90) * 24 * 3600
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.max_retrieval_results = config.get('max_retrieval_results', 5)
        self._initialized = False

    async def initialize(self) -> None:
        """初始化记忆管理器"""
        await self._do_initialize()
        self._initialized = True
        logger.info("Memory manager initialized")

    @abstractmethod
    async def _do_initialize(self) -> None:
        """初始化实现（子类实现）"""
        pass

    @abstractmethod
    async def store_experience(self, experience: Experience) -> bool:
        """存储经验"""
        pass

    @abstractmethod
    async def retrieve_similar_experiences(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Experience]:
        """检索相似经验"""
        pass

    @abstractmethod
    async def update_feedback(self, experience_id: str, feedback: Feedback) -> bool:
        """更新反馈"""
        pass

    @abstractmethod
    async def get_experience_by_id(self, experience_id: str) -> Optional[Experience]:
        """根据ID获取经验"""
        pass

    @abstractmethod
    async def delete_experience(self, experience_id: str) -> bool:
        """删除经验"""
        pass

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        pass

    def _calculate_similarity(self, query1: str, query2: str, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """计算相似度"""
        # 简单的文本相似度计算
        query1_lower = query1.lower()
        query2_lower = query2.lower()

        # 分词
        words1 = set(query1_lower.split())
        words2 = set(query2_lower.split())

        # Jaccard相似度
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        if union == 0:
            return 0.0

        jaccard_sim = intersection / union

        # 长度相似度
        len_sim = 1 - abs(len(query1) - len(query2)) / max(len(query1), len(query2))

        # 加权平均
        return 0.7 * jaccard_sim + 0.3 * len_sim

    def _apply_filters(self, experiences: List[Experience], filters: Dict[str, Any]) -> List[Experience]:
        """应用过滤条件"""
        filtered = []

        for experience in experiences:
            # 时间过滤
            if 'created_after' in filters:
                min_date = filters['created_after']
                if isinstance(min_date, str):
                    min_date = datetime.fromisoformat(min_date)
                if experience.created_at < min_date:
                    continue

            if 'created_before' in filters:
                max_date = filters['created_before']
                if isinstance(max_date, str):
                    max_date = datetime.fromisoformat(max_date)
                if experience.created_at > max_date:
                    continue

            # 质量分数过滤
            if 'min_quality_score' in filters:
                if experience.quality_score < filters['min_quality_score']:
                    continue

            # 成功率过滤
            if 'min_success_rate' in filters:
                if experience.success_rate < filters['min_success_rate']:
                    continue

            # 模型过滤
            if 'model_used' in filters:
                if experience.model_used != filters['model_used']:
                    continue

            # 使用次数过滤
            if 'min_usage_count' in filters:
                if experience.usage_count < filters['min_usage_count']:
                    continue

            filtered.append(experience)

        return filtered


class InMemoryManager(MemoryManager):
    """内存记忆管理器 - 用于开发测试"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._experiences: Dict[str, Experience] = {}
        self._query_index: Dict[str, List[str]] = {}  # 查询词到经验ID的映射

    async def _do_initialize(self) -> None:
        """初始化内存管理器"""
        logger.info("In-memory manager initialized")

    async def store_experience(self, experience: Experience) -> bool:
        """存储经验"""
        try:
            # 存储到内存
            self._experiences[experience.id] = experience

            # 更新查询索引
            query_words = experience.query.lower().split()
            for word in query_words:
                if word not in self._query_index:
                    self._query_index[word] = []
                if experience.id not in self._query_index[word]:
                    self._query_index[word].append(experience.id)

            # 限制存储数量
            if len(self._experiences) > self.max_experiences:
                await self._cleanup_by_oldest()

            logger.debug(f"Experience stored: {experience.id[:8]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to store experience: {e}")
            return False

    async def retrieve_similar_experiences(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Experience]:
        """检索相似经验"""
        try:
            query_words = query.lower().split()
            experience_scores = {}

            # 根据查询词匹配计算分数
            for word in query_words:
                if word in self._query_index:
                    for exp_id in self._query_index[word]:
                        if exp_id in self._experiences:
                            experience_scores[exp_id] = experience_scores.get(exp_id, 0) + 1

            # 计算相似度并排序
            similar_experiences = []
            for exp_id, score in experience_scores.items():
                experience = self._experiences[exp_id]
                similarity = self._calculate_similarity(query, experience.query, {}, experience.context)
                if similarity >= self.similarity_threshold:
                    experience.similarity_score = similarity
                    similar_experiences.append(experience)

            # 按相似度排序
            similar_experiences.sort(key=lambda x: x.similarity_score, reverse=True)

            # 应用过滤
            if filters:
                similar_experiences = self._apply_filters(similar_experiences, filters)

            # 限制结果数量
            results = similar_experiences[:top_k]

            # 更新使用次数
            for exp in results:
                exp.update_usage()

            logger.debug(f"Retrieved {len(results)} similar experiences")
            return results

        except Exception as e:
            logger.error(f"Failed to retrieve experiences: {e}")
            return []

    async def update_feedback(self, experience_id: str, feedback: Feedback) -> bool:
        """更新反馈"""
        try:
            if experience_id in self._experiences:
                experience = self._experiences[experience_id]

                # 添加反馈
                experience.add_feedback(feedback)

                # 重新计算质量指标
                self._update_quality_metrics(experience)

                logger.debug(f"Feedback updated: {experience_id[:8]}...")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to update feedback: {e}")
            return False

    async def get_experience_by_id(self, experience_id: str) -> Optional[Experience]:
        """根据ID获取经验"""
        return self._experiences.get(experience_id)

    async def delete_experience(self, experience_id: str) -> bool:
        """删除经验"""
        try:
            if experience_id in self._experiences:
                experience = self._experiences[experience_id]
                del self._experiences[experience_id]

                # 更新查询索引
                query_words = experience.query.lower().split()
                for word in query_words:
                    if word in self._query_index and experience_id in self._query_index[word]:
                        self._query_index[word].remove(experience_id)
                        if not self._query_index[word]:
                            del self._query_index[word]

                logger.debug(f"Experience deleted: {experience_id[:8]}...")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to delete experience: {e}")
            return False

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            total_experiences = len(self._experiences)

            # 计算平均质量分数
            quality_scores = [exp.quality_score for exp in self._experiences.values() if exp.quality_score > 0]
            avg_quality = np.mean(quality_scores) if quality_scores else 0.0

            # 计算平均成功率
            success_rates = [exp.success_rate for exp in self._experiences.values() if exp.success_rate > 0]
            avg_success_rate = np.mean(success_rates) if success_rates else 0.0

            # 计算平均相似度
            similarities = [exp.similarity_score for exp in self._experiences.values() if exp.similarity_score > 0]
            avg_similarity = np.mean(similarities) if similarities else 0.0

            # 统计使用次数
            total_usage = sum(exp.usage_count for exp in self._experiences.values())
            avg_usage = total_usage / total_experiences if total_experiences > 0 else 0.0

            return {
                'total_experiences': total_experiences,
                'max_experiences': self.max_experiences,
                'experience_ttl_days': self.experience_ttl / (24 * 3600),
                'similarity_threshold': self.similarity_threshold,
                'average_quality_score': round(avg_quality, 3),
                'average_success_rate': round(avg_success_rate, 3),
                'average_similarity_score': round(avg_similarity, 3),
                'total_usage_count': total_usage,
                'average_usage_count': round(avg_usage, 1),
                'query_index_size': len(self._query_index),
                'memory_usage_mb': round(self._estimate_memory_usage(), 2)
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {'error': str(e)}

    async def _cleanup_by_oldest(self) -> None:
        """按最旧时间清理"""
        if len(self._experiences) <= self.max_experiences:
            return

        # 找到最旧的经验
        oldest_experience = min(self._experiences.values(), key=lambda x: x.created_at)
        await self.delete_experience(oldest_experience.id)

    def _update_quality_metrics(self, experience: Experience) -> None:
        """更新质量指标"""
        if experience.feedback_list:
            # 计算成功率
            positive_count = sum(1 for f in experience.feedback_list if f.feedback_type == FeedbackType.POSITIVE)
            total_count = len(experience.feedback_list)
            experience.success_rate = positive_count / total_count if total_count > 0 else 0.0

            # 计算加权平均分数
            total_score = sum(f.rating for f in experience.feedback_list if f.rating > 0)
            rating_count = sum(1 for f in experience.feedback_list if f.rating > 0)
            experience.quality_score = total_score / rating_count if rating_count > 0 else experience.quality_score

    def _estimate_memory_usage(self) -> float:
        """估算内存使用量（MB）"""
        import sys

        total_size = 0
        for exp in self._experiences.values():
            total_size += sys.getsizeof(exp)
            total_size += sys.getsizeof(exp.feedback_list)
            total_size += sys.getsizeof(exp.context)

        return total_size / (1024 * 1024)  # 转换为MB


class RedisMemoryManager(MemoryManager):
    """Redis记忆管理器 - 用于生产环境"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.redis_client = None
        self.redis_key_prefix = config.get('redis_key_prefix', 'daml_rag:memory:')

    async def _do_initialize(self) -> None:
        """初始化Redis连接"""
        try:
            import redis.asyncio as redis
            redis_config = self.config.get('redis', {})

            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                password=redis_config.get('password'),
                decode_responses=True
            )

            # 测试连接
            await self.redis_client.ping()
            logger.info("Redis connection established")

        except ImportError:
            raise ImportError("Redis library required: pip install redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def store_experience(self, experience: Experience) -> bool:
        """存储经验到Redis"""
        try:
            key = f"{self.redis_key_prefix}experience:{experience.id}"
            data = self._serialize_experience(experience)

            # 存储主要数据
            await self.redis_client.set(key, json.dumps(data))
            await self.redis_client.expire(key, self.experience_ttl)

            # 更新查询索引
            await self._update_query_index(experience)

            logger.debug(f"Experience stored in Redis: {experience.id[:8]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to store experience in Redis: {e}")
            return False

    async def retrieve_similar_experiences(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Experience]:
        """从Redis检索相似经验"""
        try:
            # 简化实现：获取所有经验并计算相似度
            pattern = f"{self.redis_key_prefix}experience:*"
            keys = await self.redis_client.keys(pattern)

            experiences = []
            for key in keys[:100]:  # 限制检查数量
                data = await self.redis_client.get(key)
                if data:
                    experience = self._deserialize_experience(json.loads(data))
                    similarity = self._calculate_similarity(query, experience.query, {}, experience.context)
                    if similarity >= self.similarity_threshold:
                        experience.similarity_score = similarity
                        experiences.append(experience)

            # 按相似度排序
            experiences.sort(key=lambda x: x.similarity_score, reverse=True)

            # 应用过滤
            if filters:
                experiences = self._apply_filters(experiences, filters)

            results = experiences[:top_k]

            # 更新使用次数
            for exp in results:
                exp.update_usage()
                await self._update_experience_usage(exp)

            logger.debug(f"Retrieved {len(results)} experiences from Redis")
            return results

        except Exception as e:
            logger.error(f"Failed to retrieve experiences from Redis: {e}")
            return []

    async def update_feedback(self, experience_id: str, feedback: Feedback) -> bool:
        """更新Redis中的反馈"""
        try:
            key = f"{self.redis_key_prefix}experience:{experience_id}"
            data = await self.redis_client.get(key)
            if data:
                experience_data = json.loads(data)
                experience = self._deserialize_experience(experience_data)

                # 添加反馈
                experience.add_feedback(feedback)
                self._update_quality_metrics(experience)

                # 更新Redis
                updated_data = self._serialize_experience(experience)
                await self.redis_client.set(key, json.dumps(updated_data))
                await self.redis_client.expire(key, self.experience_ttl)

                logger.debug(f"Feedback updated in Redis: {experience_id[:8]}...")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to update feedback in Redis: {e}")
            return False

    async def get_experience_by_id(self, experience_id: str) -> Optional[Experience]:
        """从Redis获取经验"""
        try:
            key = f"{self.redis_key_prefix}experience:{experience_id}"
            data = await self.redis_client.get(key)
            if data:
                return self._deserialize_experience(json.loads(data))
            return None

        except Exception as e:
            logger.error(f"Failed to get experience from Redis: {e}")
            return None

    async def delete_experience(self, experience_id: str) -> bool:
        """从Redis删除经验"""
        try:
            key = f"{self.redis_key_prefix}experience:{experience_id}"
            result = await self.redis_client.delete(key)

            # 删除查询索引中的条目
            await self._remove_from_query_index(experience_id)

            logger.debug(f"Experience deleted from Redis: {experience_id[:8]}...")
            return result > 0

        except Exception as e:
            logger.error(f"Failed to delete experience from Redis: {e}")
            return False

    async def get_statistics(self) -> Dict[str, Any]:
        """获取Redis统计信息"""
        try:
            # 获取经验数量
            pattern = f"{self.redis_key_prefix}experience:*"
            keys = await self.redis_client.keys(pattern)
            total_experiences = len(keys)

            # 获取Redis信息
            info = await self.redis_client.info('memory')
            memory_usage = info.get('used_memory', 0) / (1024 * 1024)  # MB

            return {
                'total_experiences': total_experiences,
                'max_experiences': self.max_experiences,
                'experience_ttl_days': self.experience_ttl / (24 * 3600),
                'similarity_threshold': self.similarity_threshold,
                'redis_memory_usage_mb': round(memory_usage, 2),
                'redis_connected': True
            }

        except Exception as e:
            logger.error(f"Failed to get Redis statistics: {e}")
            return {'error': str(e), 'redis_connected': False}

    async def _update_query_index(self, experience: Experience) -> None:
        """更新查询索引"""
        query_words = experience.query.lower().split()
        for word in query_words:
            index_key = f"{self.redis_key_prefix}index:{word}"
            await self.redis_client.sadd(index_key, experience.id)
            await self.redis_client.expire(index_key, self.experience_ttl)

    async def _remove_from_query_index(self, experience_id: str) -> None:
        """从查询索引中删除"""
        # 获取所有索引键
        pattern = f"{self.redis_key_prefix}index:*"
        keys = await self.redis_client.keys(pattern)

        for key in keys:
            await self.redis_client.srem(key, experience_id)

    async def _update_experience_usage(self, experience: Experience) -> None:
        """更新经验使用次数"""
        key = f"{self.redis_key_prefix}experience:{experience.id}"
        data = await self.redis_client.get(key)
        if data:
            experience_data = json.loads(data)
            experience_data['usage_count'] = experience.usage_count
            experience_data['updated_at'] = experience.updated_at.isoformat()
            await self.redis_client.set(key, json.dumps(experience_data))

    def _serialize_experience(self, experience: Experience) -> Dict[str, Any]:
        """序列化经验"""
        return {
            'id': experience.id,
            'query': experience.query,
            'response': experience.response,
            'context': experience.context,
            'retrieval_result': experience.retrieval_result.__dict__ if experience.retrieval_result else None,
            'model_used': experience.model_used,
            'feedback_list': [self._serialize_feedback(f) for f in experience.feedback_list],
            'quality_score': experience.quality_score,
            'usage_count': experience.usage_count,
            'success_rate': experience.success_rate,
            'created_at': experience.created_at.isoformat(),
            'updated_at': experience.updated_at.isoformat(),
        }

    def _deserialize_experience(self, data: Dict[str, Any]) -> Experience:
        """反序列化经验"""
        experience = Experience(
            id=data['id'],
            query=data['query'],
            response=data['response'],
            context=data.get('context', {}),
            quality_score=data.get('quality_score', 0.0),
            usage_count=data.get('usage_count', 0),
            success_rate=data.get('success_rate', 0.0),
        )

        # 设置时间戳
        if 'created_at' in data:
            experience.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            experience.updated_at = datetime.fromisoformat(data['updated_at'])

        # 设置其他属性
        if 'model_used' in data:
            experience.model_used = data['model_used']
        if 'retrieval_result' in data and data['retrieval_result']:
            experience.retrieval_result = RetrievalResult(**data['retrieval_result'])

        # 添加反馈
        for feedback_data in data.get('feedback_list', []):
            feedback = self._deserialize_feedback(feedback_data)
            experience.feedback_list.append(feedback)

        return experience

    def _serialize_feedback(self, feedback: Feedback) -> Dict[str, Any]:
        """序列化反馈"""
        return {
            'id': feedback.id,
            'experience_id': feedback.experience_id,
            'user_id': feedback.user_id,
            'feedback_type': feedback.feedback_type.value,
            'rating': feedback.rating,
            'comment': feedback.comment,
            'metadata': feedback.metadata,
            'created_at': feedback.created_at.isoformat(),
        }

    def _deserialize_feedback(self, data: Dict[str, Any]) -> Feedback:
        """反序列化反馈"""
        feedback = Feedback(
            id=data['id'],
            experience_id=data['experience_id'],
            user_id=data['user_id'],
            feedback_type=FeedbackType(data['feedback_type']),
            rating=data.get('rating', 0.0),
            comment=data.get('comment', ''),
            metadata=data.get('metadata', {}),
        )

        if 'created_at' in data:
            feedback.created_at = datetime.fromisoformat(data['created_at'])

        return feedback

    def _update_quality_metrics(self, experience: Experience) -> None:
        """更新质量指标"""
        if experience.feedback_list:
            positive_count = sum(1 for f in experience.feedback_list if f.feedback_type == FeedbackType.POSITIVE)
            total_count = len(experience.feedback_list)
            experience.success_rate = positive_count / total_count if total_count > 0 else 0.0

            total_score = sum(f.rating for f in experience.feedback_list if f.rating > 0)
            rating_count = sum(1 for f in experience.feedback_list if f.rating > 0)
            experience.quality_score = total_score / rating_count if rating_count > 0 else experience.quality_score